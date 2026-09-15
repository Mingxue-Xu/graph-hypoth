"""Deterministic, target-scoped association scoring.

The Evidence Reviewer sub-path "graph target -> deterministic signal channels ->
`EvidenceSignalBundle` -> triage top-K". `AssociationScorer` is a STANDALONE
component rather than a reconfiguration of claim-scoped ``EvidenceEnricher``. It scores one
evidence record against one ``AssociationTarget``,
producing per-(evidence, target) bundles that carry the four model-free signal
channels separately plus the combined ranking score.

Everything here is deterministic and model-free EXCEPT the optional embedding
vectors (DETERMINISTIC* given the pinned model). All coefficients come from
``scoring_defaults``; every bundle records ``scoring_defaults_version`` so its association score
is reproducible.

Scope boundary: this layer stops before the semantic verifier. It MUST NOT
consume workflow-state fields (`user_priority`, current uncertainty/verdict,
graph status, or agent confidence) as scoring inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.retrieval import scoring_defaults as sd
from src.retrieval.similarity import (
    Embedder,
    _cosine,
    clip01 as _clip01,
    identifiers_of,
    jaccard,
    normalized_terms,
    overlap,
    references_of,
)
from src.state import RetrievedEvidence

# Closed relation-role enumeration. ``direct`` and ``adjacent`` are
# citation-distance internal variable names only, never relation-role values.
EVIDENCE_ROLES: frozenset[str] = frozenset(
    {"support", "contradiction", "qualification", "mechanism", "confounder"}
)


def _active_weighted(pairs: list[tuple[float, float]]) -> float:
    """Return a weighted mean normalized by the active weight sum.

    A zero-weight channel disables cleanly: it drops out of both numerator and
    denominator rather than breaking the divide.
    """
    active = [(w, s) for w, s in pairs if w != 0]
    total = sum(w for w, _ in active)
    if total == 0:
        return 0.0
    return sum(w * s for w, s in active) / total


def _inverse_rank_rescale(rank: int, n: int) -> float:
    """Rescale inverse rank so rank 1 maps to 1 and rank N maps to 0.

    rank 1 -> 1, rank N -> 0, and = 1 for N = 1. A non-positive rank is treated
    as the best rank (1).
    """
    if n <= 1:
        return 1.0
    r = max(int(rank), 1)
    return ((1.0 / r) - (1.0 / n)) / (1.0 - (1.0 / n))


def _identity_key(item: RetrievedEvidence) -> str:
    """Stable primary key used to label citation neighbors (doi/source_id/id)."""
    return str(item.metadata.get("doi") or item.source_id or item.evidence_id)


@dataclass(frozen=True)
class _BatchCitationIndex:
    """Per-batch inverted index for citation-overlap scoring.

    Built ONCE per batch so the leg looks up only the candidate records that CAN
    share a citation token, instead of re-deriving refs/ids and computing overlap
    against the whole batch per evidence (the old O(N^2) scan). ``overlap`` is >0
    ONLY when two records share a token across (refs∩refs), (refs∩ids) or (ids∩refs),
    so candidates outside this union contribute exactly 0 — the result is identical.
    """

    batch: list[RetrievedEvidence]
    refs: list[set[str]]              # references_of per batch position
    ids: list[set[str]]              # identifiers_of per batch position
    by_ref: dict[str, list[int]]     # ref token -> batch positions (ascending)
    by_id: dict[str, list[int]]      # id token -> batch positions (ascending)


def _build_citation_index(batch: list[RetrievedEvidence]) -> _BatchCitationIndex:
    refs = [references_of(ev) for ev in batch]
    ids = [identifiers_of(ev) for ev in batch]
    by_ref: dict[str, list[int]] = {}
    by_id: dict[str, list[int]] = {}
    for pos, (ref_set, id_set) in enumerate(zip(refs, ids)):
        for token in ref_set:
            by_ref.setdefault(token, []).append(pos)
        for token in id_set:
            by_id.setdefault(token, []).append(pos)
    return _BatchCitationIndex(batch=batch, refs=refs, ids=ids, by_ref=by_ref, by_id=by_id)


@dataclass(frozen=True)
class AssociationTarget:
    """Scoring-side projection of a graph target.

    `target_id` is an opaque, replay-stable handle minted upstream
    (`stable_hash(node_or_edge_id, evidence_role)`); the scorer never parses the
    role back out of it. `evidence_role` is REQUIRED with NO default and MUST be
    one of the closed relation-role enum.
    """

    target_id: str
    verification_task_id: str
    target_kind: str  # concept_node | causal_edge
    label: str
    evidence_role: str
    aliases: tuple[str, ...] = ()
    definition: str = ""
    target_type: str = ""
    scope_qualifiers: tuple[str, ...] = ()
    question: str = ""
    criteria: str = ""

    def __post_init__(self) -> None:
        if self.evidence_role not in EVIDENCE_ROLES:
            raise ValueError(
                f"evidence_role must be one of {sorted(EVIDENCE_ROLES)}; "
                f"got {self.evidence_role!r}"
            )


@dataclass(frozen=True)
class GraphEvidenceContext:
    """Same-target / adjacent-target evidence sets feeding the citation legs.

    The scorer computes direct and adjacent
    legs from these sets and degrades to the batch-local leg only when this
    context is `None` (cold start, before any graph history).
    """

    same_target_evidence: tuple[RetrievedEvidence, ...] = ()
    adjacent_target_evidence: tuple[RetrievedEvidence, ...] = ()


@dataclass(frozen=True)
class EvidenceSignalBundle:
    """Deterministic target-scoped scoring bundle for one (evidence, target).

    Tuple-typed collections make the frozen dataclass hashable for receipt replay. The
    combined score is named `association_score`; `scoring_defaults_version` is
    REQUIRED (no default). `embedding_fallback` records the loud lexical fallback
    and is retained for audit provenance.
    """

    target_id: str
    verification_task_id: str
    evidence_id: str
    evidence_role: str
    lexical_score: float
    embedding_score: float
    citation_score: float
    source_prior: float
    association_score: float
    matched_terms: tuple[str, ...]
    matched_anchor: str | None
    matched_quote_span: str | None
    citation_neighbors: tuple[str, ...]
    scoring_defaults_version: str
    embedding_fallback: bool = False


@dataclass(frozen=True)
class CandidateEvidence:
    """One candidate: its ledger record paired with its deterministic bundle."""

    evidence: RetrievedEvidence
    bundle: EvidenceSignalBundle


@dataclass(frozen=True)
class CandidateEvidenceSet:
    """Typed collection of bundle-bearing candidates.

    Not a persisted class — the hand-off store between deterministic scoring and
    the triage agent.
    """

    items: tuple[CandidateEvidence, ...] = ()


def recompute_association_score(
    bundle: EvidenceSignalBundle, defaults: Any = sd
) -> float:
    """Recompute ``association_score`` from stored components.

    Independent of the fallback flag: on fallback `embedding_score` equals
    `lexical_score`, so the semantic term collapses to the lexical score exactly.
    """
    semantic = _active_weighted(
        [
            (defaults.SEMANTIC_WEIGHTS["embedding"], bundle.embedding_score),
            (defaults.SEMANTIC_WEIGHTS["lexical"], bundle.lexical_score),
        ]
    )
    association = _active_weighted(
        [
            (defaults.ASSOCIATION_WEIGHTS["semantic"], semantic),
            (defaults.ASSOCIATION_WEIGHTS["citation"], bundle.citation_score),
            (defaults.ASSOCIATION_WEIGHTS["source_prior"], bundle.source_prior),
        ]
    )
    return round(association, 6)


class AssociationScorer:
    """Deterministic target-scoped scorer (lexical, embedding, citation, prior).

    Inject a deterministic stub `embedder` in tests; the default suite never
    loads the pinned `specter2_base` model. `defaults` defaults to the
    ``scoring_defaults`` module so every coefficient is centralized.
    """

    def __init__(self, *, embedder: Embedder | None = None, defaults: Any = sd) -> None:
        self._embedder = embedder
        self._d = defaults

    # -- public API ---------------------------------------------------------

    def score(
        self,
        target: AssociationTarget,
        evidence: RetrievedEvidence,
        *,
        graph_neighbors: GraphEvidenceContext | None = None,
    ) -> EvidenceSignalBundle:
        """Score one evidence record against one target (batch of one)."""
        priors = self._source_priors([evidence])
        return self._score_one(
            target,
            evidence,
            source_prior=priors[evidence.evidence_id],
            batch=[evidence],
            graph_neighbors=graph_neighbors,
        )

    def score_candidates(
        self,
        target: AssociationTarget,
        evidences: list[RetrievedEvidence],
        *,
        graph_neighbors: GraphEvidenceContext | None = None,
    ) -> CandidateEvidenceSet:
        """Score a batch; source priors + batch-citation leg use the whole batch."""
        priors = self._source_priors(evidences)
        # Build the citation index once per batch, not once per evidence item.
        batch_index = _build_citation_index(evidences)
        items = tuple(
            CandidateEvidence(
                evidence=ev,
                bundle=self._score_one(
                    target,
                    ev,
                    source_prior=priors[ev.evidence_id],
                    batch=evidences,
                    graph_neighbors=graph_neighbors,
                    batch_index=batch_index,
                ),
            )
            for ev in evidences
        )
        return CandidateEvidenceSet(items=items)

    def gate(self, bundle: EvidenceSignalBundle) -> bool:
        """Apply the relevance gate over lexical, embedding, and citation channels.

        Uses ONLY the lexical, embedding, and citation channels — never the
        source prior or quote quality. A weak sanity floor (relevance selection
        is rank-based via triage).
        """
        return (
            max(bundle.lexical_score, bundle.embedding_score, bundle.citation_score)
            >= self._d.THETA_ASSOC
        )

    def triage(
        self, candidate_set: CandidateEvidenceSet, *, k_top: int | None = None
    ) -> tuple[CandidateEvidence, ...]:
        """Return the top gated candidates by descending association score.

        Ties break on `evidence_id` for a stable, reproducible order.
        """
        k = self._d.K_TOP if k_top is None else k_top
        gated = [c for c in candidate_set.items if self.gate(c.bundle)]
        gated.sort(key=lambda c: (-c.bundle.association_score, c.bundle.evidence_id))
        return tuple(gated[:k])

    # -- channels -----------------------------------------------------------

    def _score_one(
        self,
        target: AssociationTarget,
        evidence: RetrievedEvidence,
        *,
        source_prior: float,
        batch: list[RetrievedEvidence],
        graph_neighbors: GraphEvidenceContext | None,
        batch_index: _BatchCitationIndex | None = None,
    ) -> EvidenceSignalBundle:
        lexical, matched_terms = self._lexical(target, evidence)
        embedding, matched_anchor, fallback = self._embedding(target, evidence, lexical)
        citation, neighbors = self._citation(evidence, batch, graph_neighbors, batch_index)

        semantic = _active_weighted(
            [
                (self._d.SEMANTIC_WEIGHTS["embedding"], embedding),
                (self._d.SEMANTIC_WEIGHTS["lexical"], lexical),
            ]
        )
        association = round(
            _active_weighted(
                [
                    (self._d.ASSOCIATION_WEIGHTS["semantic"], semantic),
                    (self._d.ASSOCIATION_WEIGHTS["citation"], citation),
                    (self._d.ASSOCIATION_WEIGHTS["source_prior"], source_prior),
                ]
            ),
            6,
        )
        return EvidenceSignalBundle(
            target_id=target.target_id,
            verification_task_id=target.verification_task_id,
            evidence_id=evidence.evidence_id,
            evidence_role=target.evidence_role,
            lexical_score=lexical,
            embedding_score=embedding,
            citation_score=citation,
            source_prior=source_prior,
            association_score=association,
            matched_terms=matched_terms,
            matched_anchor=matched_anchor,
            matched_quote_span=evidence.quote or None,
            citation_neighbors=neighbors,
            scoring_defaults_version=self._d.SCORING_DEFAULTS_VERSION,
            embedding_fallback=fallback,
        )

    def _lexical(
        self, target: AssociationTarget, evidence: RetrievedEvidence
    ) -> tuple[float, tuple[str, ...]]:
        """Compute a hybrid lexical score with a hard anchor gate."""
        label_terms = normalized_terms(target.label)
        alias_terms: set[str] = set()
        for alias in target.aliases:
            alias_terms |= normalized_terms(alias)
        defn_terms = normalized_terms(
            target.definition + " " + " ".join(target.scope_qualifiers)
        )
        syn_terms = self._synonym_terms(target)

        paper_terms = normalized_terms(
            (evidence.title or "") + " " + (evidence.quote or "")
        )

        anchor_set = label_terms | alias_terms | syn_terms
        anchor = 1.0 if (anchor_set & paper_terms) else 0.0
        if anchor == 0.0:
            return 0.0, ()

        w_label = self._d.ANCHOR_TERM_WEIGHTS["label"]
        w_alias = self._d.ANCHOR_TERM_WEIGHTS["alias"]
        weighted_total = 0.0
        weighted_hit = 0.0
        for term in label_terms | alias_terms:
            weight = w_label if term in label_terms else w_alias
            weighted_total += weight
            if term in paper_terms:
                weighted_hit += weight
        anchor_coverage = weighted_hit / weighted_total if weighted_total else 0.0

        target_terms = label_terms | alias_terms | defn_terms
        jac = jaccard(target_terms, paper_terms)

        syn_denom = label_terms | alias_terms
        syn_coverage = (
            len(syn_terms & paper_terms) / len(syn_denom) if syn_denom else 0.0
        )

        weights = self._d.LEXICAL_WEIGHTS
        inner = (
            weights["label"] * jac
            + weights["anchor"] * anchor_coverage
            + weights["syn"] * syn_coverage
        )
        lexical = round(anchor * _clip01(inner), 6)
        matched = tuple(sorted((target_terms | syn_terms) & paper_terms))
        return lexical, matched

    def _synonym_terms(self, target: AssociationTarget) -> set[str]:
        """Expand label and alias phrases through the scorer-owned lexicon."""
        phrases = {target.label.lower(), *(alias.lower() for alias in target.aliases)}
        out: set[str] = set()
        for phrase in phrases:
            for synonym in self._d.SYNONYM_LEXICON.get(phrase, ()):
                out |= normalized_terms(synonym)
        return out

    def _anchor_texts(self, target: AssociationTarget) -> list[tuple[str, str]]:
        """Return ``(anchor_label, full_anchor_text)`` for every label and alias."""
        tail = " ".join(
            part
            for part in (
                target.definition,
                target.target_type,
                " ".join(target.scope_qualifiers),
                target.evidence_role,
                target.question,
                target.criteria,
            )
            if part
        )
        labels = [target.label, *target.aliases]
        return [(label, (label + " " + tail).strip()) for label in labels]

    def _embedding(
        self, target: AssociationTarget, evidence: RetrievedEvidence, lexical: float
    ) -> tuple[float, str | None, bool]:
        """Return max clamped cosine over anchors, with an explicit lexical fallback."""
        anchors = self._anchor_texts(target)
        paper_anchor = (evidence.title or "") + " " + (evidence.quote or "")
        if self._embedder is None or not anchors:
            return lexical, target.label, True
        try:
            vectors = self._embedder([paper_anchor] + [text for _, text in anchors])
        except Exception:  # noqa: BLE001 - any embedder failure -> loud fallback.
            return lexical, target.label, True
        paper_vec, anchor_vecs = vectors[0], vectors[1:]
        best = 0.0
        best_anchor = anchors[0][0]
        for (anchor_label, _), vector in zip(anchors, anchor_vecs):
            cosine = _cosine(paper_vec, vector)
            if cosine > best:
                best = cosine
                best_anchor = anchor_label
        return round(best, 6), best_anchor, False

    def _citation(
        self,
        evidence: RetrievedEvidence,
        batch: list[RetrievedEvidence],
        graph_neighbors: GraphEvidenceContext | None,
        batch_index: _BatchCitationIndex | None = None,
    ) -> tuple[float, tuple[str, ...]]:
        """Return the maximum attenuated citation score across neighbor buckets."""
        own_refs = references_of(evidence)
        own_ids = identifiers_of(evidence)

        def leg(others: tuple[RetrievedEvidence, ...] | list[RetrievedEvidence]):
            best = 0.0
            neighbors: list[tuple[float, str]] = []
            for other in others:
                if other.evidence_id == evidence.evidence_id:
                    continue
                value = overlap(
                    own_refs, own_ids, references_of(other), identifiers_of(other)
                )
                if value > 0:
                    neighbors.append((value, _identity_key(other)))
                best = max(best, value)
            return best, neighbors

        # The batch leg (the only O(N^2) one) goes through the inverted index: only
        # records sharing a citation token can score > 0, and we visit them in batch
        # order so sorted ties break identically to the naive scan.
        index = batch_index if batch_index is not None else _build_citation_index(batch)
        candidate_positions: set[int] = set()
        for token in own_refs:
            candidate_positions.update(index.by_ref.get(token, ()))
            candidate_positions.update(index.by_id.get(token, ()))
        for token in own_ids:
            candidate_positions.update(index.by_ref.get(token, ()))
        batch_overlap = 0.0
        batch_n: list[tuple[float, str]] = []
        for pos in sorted(candidate_positions):
            other = index.batch[pos]
            if other.evidence_id == evidence.evidence_id:
                continue
            value = overlap(own_refs, own_ids, index.refs[pos], index.ids[pos])
            if value > 0:
                batch_n.append((value, _identity_key(other)))
            batch_overlap = max(batch_overlap, value)

        same = graph_neighbors.same_target_evidence if graph_neighbors else ()
        adjacent = graph_neighbors.adjacent_target_evidence if graph_neighbors else ()
        direct, direct_n = leg(same)
        neighbor, neighbor_n = leg(adjacent)

        citation = round(
            max(
                direct,
                self._d.ADJACENT_TARGET_ATTENUATION * neighbor,
                self._d.BATCH_CITATION_ATTENUATION * batch_overlap,
            ),
            6,
        )
        ranked = sorted(direct_n + neighbor_n + batch_n, key=lambda pair: pair[0], reverse=True)
        seen: set[str] = set()
        keys: list[str] = []
        for _, key in ranked:
            if key not in seen:
                seen.add(key)
                keys.append(key)
        return citation, tuple(keys[:5])

    def _source_priors(self, evidences: list[RetrievedEvidence]) -> dict[str, float]:
        """Return per-source min-max priors, falling back to inverse rank."""
        n = len(evidences)
        by_source: dict[str, list[RetrievedEvidence]] = {}
        for ev in evidences:
            by_source.setdefault(ev.source, []).append(ev)

        priors: dict[str, float] = {}
        for group in by_source.values():
            scores = [float(ev.score) for ev in group if ev.score is not None]
            low = min(scores) if scores else None
            high = max(scores) if scores else None
            for ev in group:
                if ev.score is not None and low is not None and high is not None:
                    if high == low:
                        priors[ev.evidence_id] = 1.0
                    else:
                        priors[ev.evidence_id] = (float(ev.score) - low) / (high - low)
                else:
                    priors[ev.evidence_id] = _inverse_rank_rescale(ev.rank, n)
        return priors
