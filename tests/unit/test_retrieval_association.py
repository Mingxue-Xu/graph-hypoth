"""Unit tests for the deterministic target-scoped association scorer.

Covers the Evidence Reviewer sub-path "graph target -> deterministic signal channels ->
EvidenceSignalBundle -> triage top-K". Tests use the deterministic stub embedder (the pinned
`specter2_base` never loads here) and assert floats within 1e-6.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from src.retrieval import scoring_defaults as sd
from src.retrieval.association import (
    AssociationScorer,
    AssociationTarget,
    CandidateEvidenceSet,
    EvidenceSignalBundle,
    GraphEvidenceContext,
    recompute_association_score,
)
from src.state import RetrievedEvidence


# --- builders ---------------------------------------------------------------

def _target(
    *,
    target_id="tgt_a",
    verification_task_id="task_a",
    label="statin therapy",
    evidence_role="support",
    aliases=(),
    definition="",
    target_kind="concept_node",
    scope_qualifiers=(),
) -> AssociationTarget:
    return AssociationTarget(
        target_id=target_id,
        verification_task_id=verification_task_id,
        target_kind=target_kind,
        label=label,
        evidence_role=evidence_role,
        aliases=tuple(aliases),
        definition=definition,
        scope_qualifiers=tuple(scope_qualifiers),
    )


def _evidence(
    evidence_id="ev_000001",
    *,
    title="statin therapy reduces mortality",
    quote="",
    source="openalex",
    source_id=None,
    score=None,
    rank=1,
    metadata=None,
) -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id=evidence_id,
        source=source,
        source_id=source_id,
        title=title,
        quote=quote,
        relevance="r",
        retrieved_by="builder",
        tool_call_id="tc",
        score=score,
        rank=rank,
        trust_tier="indexed_metadata",
        metadata=metadata or {},
    )


@pytest.fixture
def alpha_embedder():
    """Vectors [1,0] for any text containing 'alpha', else [0,1].

    A label-anchor and a paper-anchor that both mention 'alpha' get cosine 1.0;
    everything else is orthogonal (cosine 0.0). No model load, no network.
    """
    def embed(texts):
        return [[1.0, 0.0] if "alpha" in t.lower() else [0.0, 1.0] for t in texts]
    return embed


# --- Association target: closed evidence-role enum ------------------------------------

def test_target_rejects_evidence_role_outside_closed_enum() -> None:
    with pytest.raises(ValueError):
        _target(evidence_role="direct")  # citation-distance term, NOT a relation role
    for role in ("support", "contradiction", "qualification", "mechanism", "confounder"):
        assert _target(evidence_role=role).evidence_role == role


# --- Immutable, hashable evidence-signal bundles --------------------------------------

def test_bundle_is_frozen_hashable_with_tuple_collections() -> None:
    scorer = AssociationScorer(embedder=None)
    bundle = scorer.score(_target(), _evidence())
    assert isinstance(bundle, EvidenceSignalBundle)
    assert isinstance(bundle.matched_terms, tuple)
    assert isinstance(bundle.citation_neighbors, tuple)
    assert hash(bundle) is not None  # hashable for receipt replay
    with pytest.raises(FrozenInstanceError):
        bundle.association_score = 0.0  # immutable


def test_every_bundle_carries_required_version_stamp() -> None:
    scorer = AssociationScorer(embedder=None)
    bundle = scorer.score(_target(), _evidence())
    assert bundle.scoring_defaults_version == sd.SCORING_DEFAULTS_VERSION


# --- Lexical channel: hybrid score and anchor gate ------------------------------------

def test_lexical_hybrid_value_matches_hand_oracle() -> None:
    # label "statin therapy" -> {statin, therapy}; paper terms add {reduces, mortality}.
    # anchor=1; AnchorCoverage=1.0; Jaccard=2/4=0.5; SynonymCoverage=0.
    # inner = 0.5*0.5 + 0.3*1.0 + 0.2*0.0 = 0.55 ; L = 1*clip01(0.55) = 0.55
    scorer = AssociationScorer(embedder=None)
    bundle = scorer.score(_target(), _evidence())
    assert bundle.lexical_score == pytest.approx(0.55, abs=1e-6)
    assert set(bundle.matched_terms) == {"statin", "therapy"}


def test_anchor_gate_forces_lexical_zero_on_definition_only_overlap() -> None:
    # Paper overlaps ONLY the definition (no label/alias/synonym term) -> L == 0.
    target = _target(label="statin therapy", definition="lipid lowering cohort study")
    paper = _evidence(title="lipid lowering cohort study design", quote="")
    scorer = AssociationScorer(embedder=None)
    bundle = scorer.score(target, paper)
    assert bundle.lexical_score == 0.0


def test_synonym_channel_uses_scorer_owned_lexicon() -> None:
    # "mortality" is a SYNONYM_LEXICON key (-> death/deaths/...); the paper says
    # "deaths" but not "mortality", so only the synonym channel + anchor fire.
    target = _target(label="mortality")
    paper = _evidence(title="study of deaths in adults", quote="")
    scorer = AssociationScorer(embedder=None)
    bundle = scorer.score(target, paper)
    # anchor fires via the synonym term "deaths"; SynonymCoverage > 0 -> L > 0.
    assert bundle.lexical_score > 0.0
    assert "deaths" in bundle.matched_terms


# --- Embedding channel: clamped maximum over anchors ----------------------------------

def test_embedding_is_max_clamped_cosine_over_anchors(alpha_embedder) -> None:
    target = _target(label="alpha", aliases=("beta",), definition="")
    paper = _evidence(title="alpha", quote="")
    scorer = AssociationScorer(embedder=alpha_embedder)
    bundle = scorer.score(target, paper)
    assert bundle.embedding_score == pytest.approx(1.0, abs=1e-6)  # label anchor matches
    assert bundle.matched_anchor == "alpha"
    assert bundle.embedding_fallback is False


# --- Explicit lexical fallback when the embedder is unavailable -----------------------

def test_embedder_absent_records_loud_lexical_fallback() -> None:
    scorer = AssociationScorer(embedder=None)
    bundle = scorer.score(_target(), _evidence())
    assert bundle.association_score is not None              # scoring still completes
    assert bundle.embedding_score == bundle.lexical_score    # channel == lexical
    assert bundle.embedding_fallback is True                 # recorded, not silent


def test_embedder_raising_falls_back_loudly() -> None:
    def boom(texts):
        raise RuntimeError("model exploded")
    scorer = AssociationScorer(embedder=boom)
    bundle = scorer.score(_target(), _evidence())
    assert bundle.embedding_fallback is True
    assert bundle.embedding_score == bundle.lexical_score


# --- Citation channel: maximum over graph-context buckets ------------------------------

def test_citation_direct_target_leg_is_unattenuated() -> None:
    cand = _evidence("ev_000001", metadata={"references": ["d1", "d2"]})
    same = _evidence("ev_000002", metadata={"references": ["d1", "d2"]})
    ctx = GraphEvidenceContext(same_target_evidence=(same,))
    bundle = AssociationScorer(embedder=None).score(_target(), cand, graph_neighbors=ctx)
    assert bundle.citation_score == pytest.approx(1.0, abs=1e-6)
    assert "ev_000002" in bundle.citation_neighbors


def test_citation_adjacent_leg_is_attenuated() -> None:
    cand = _evidence("ev_000001", metadata={"references": ["d1", "d2"]})
    adj = _evidence("ev_000002", metadata={"references": ["d1", "d2"]})
    ctx = GraphEvidenceContext(adjacent_target_evidence=(adj,))
    bundle = AssociationScorer(embedder=None).score(_target(), cand, graph_neighbors=ctx)
    assert bundle.citation_score == pytest.approx(sd.ADJACENT_TARGET_ATTENUATION, abs=1e-6)


def test_citation_degrades_to_batch_leg_at_cold_start() -> None:
    cand = _evidence("ev_000001", metadata={"references": ["d1", "d2"]})
    other = _evidence("ev_000002", metadata={"references": ["d1", "d2"]})
    # No graph_neighbors -> only the batch-local leg, attenuated by 0.3.
    cset = AssociationScorer(embedder=None).score_candidates(_target(), [cand, other])
    by_id = {c.evidence.evidence_id: c.bundle for c in cset.items}
    assert by_id["ev_000001"].citation_score == pytest.approx(
        sd.BATCH_CITATION_ATTENUATION, abs=1e-6
    )


# --- Combined association score --------------------------------------------------------

def test_combined_association_score_hand_oracle_with_fallback() -> None:
    # L = 0.55 ; fallback -> E = L = 0.55 ; C = 0 ; lone source_prior = 1.0
    # semantic = 0.8*0.55 + 0.2*0.55 = 0.55
    # association = 0.55*0.55 + 0.35*0 + 0.10*1.0 = 0.4025
    scorer = AssociationScorer(embedder=None)
    bundle = scorer.score(_target(), _evidence())
    assert bundle.citation_score == 0.0
    assert bundle.source_prior == pytest.approx(1.0, abs=1e-6)
    assert bundle.association_score == pytest.approx(0.4025, abs=1e-6)


def test_association_score_recomputable_from_components(alpha_embedder) -> None:
    scorer = AssociationScorer(embedder=alpha_embedder)
    bundle = scorer.score(_target(label="alpha"), _evidence(title="alpha statin therapy"))
    recomputed = recompute_association_score(bundle)
    assert recomputed == pytest.approx(bundle.association_score, abs=1e-6)


# --- Determinism -----------------------------------------------------------------------

def test_identical_inputs_reproduce_all_scores(alpha_embedder) -> None:
    scorer = AssociationScorer(embedder=alpha_embedder)
    target, paper = _target(label="alpha"), _evidence(title="alpha therapy")
    a = scorer.score(target, paper)
    b = scorer.score(target, paper)
    for field_name in ("lexical_score", "embedding_score", "citation_score",
                       "source_prior", "association_score"):
        assert getattr(a, field_name) == pytest.approx(getattr(b, field_name), abs=1e-6)


# --- Target-specific bundles for the same paper ---------------------------------------

def test_same_paper_two_targets_yields_distinct_bundles() -> None:
    paper = _evidence(title="statin therapy reduces mortality")
    t_related = _target(target_id="tgt_a", evidence_role="support", label="statin therapy")
    t_unrelated = _target(
        target_id="tgt_b", evidence_role="mechanism", label="quantum chromodynamics"
    )
    scorer = AssociationScorer(embedder=None)
    b1 = scorer.score(t_related, paper)
    b2 = scorer.score(t_unrelated, paper)
    assert b1.evidence_id == b2.evidence_id                              # same paper
    assert (b1.target_id, b1.evidence_role) != (b2.target_id, b2.evidence_role)
    assert b1.association_score != b2.association_score                  # never collapsed


# --- Relevance gate: lexical, embedding, and citation channels only --------------------

def test_source_prior_cannot_open_the_gate() -> None:
    # Maximal source prior but zero association on all three channels -> gated out.
    target = _target(label="statin therapy")
    paper = _evidence(title="completely unrelated cooking recipe", score=9.9, rank=1)
    scorer = AssociationScorer(embedder=None)
    cset = scorer.score_candidates(target, [paper])
    bundle = cset.items[0].bundle
    assert max(bundle.lexical_score, bundle.embedding_score, bundle.citation_score) == 0.0
    assert bundle.source_prior == pytest.approx(1.0, abs=1e-6)  # high standing...
    assert scorer.gate(bundle) is False                         # ...still gated out
    assert scorer.triage(cset) == ()                            # never reaches triage


def test_gate_passes_when_one_channel_clears_floor() -> None:
    scorer = AssociationScorer(embedder=None)
    bundle = scorer.score(_target(), _evidence())  # lexical 0.55 >= theta_assoc
    assert bundle.lexical_score >= sd.THETA_ASSOC
    assert scorer.gate(bundle) is True


# --- Top-K triage over gated candidates by association score ---------------------------

def test_triage_forwards_top_k_gated_ranked_descending() -> None:
    target = _target(label="statin therapy")
    gated = [
        _evidence("ev_000001", title="statin therapy reduces mortality"),
        _evidence("ev_000002", title="statin therapy and stroke"),
        _evidence("ev_000003", title="statin therapy review"),
    ]
    ungated = _evidence("ev_000009", title="unrelated cooking recipe")
    scorer = AssociationScorer(embedder=None)
    cset = scorer.score_candidates(target, [*gated, ungated])
    forwarded = scorer.triage(cset, k_top=2)
    assert len(forwarded) == 2                                  # min(k_top, gated)
    scores = [c.bundle.association_score for c in forwarded]
    assert scores == sorted(scores, reverse=True)               # ranked desc
    assert all(scorer.gate(c.bundle) for c in forwarded)        # all passed the gate
    assert "ev_000009" not in {c.evidence.evidence_id for c in forwarded}


def test_triage_no_padding_when_fewer_gated_than_k() -> None:
    scorer = AssociationScorer(embedder=None)
    cset = scorer.score_candidates(_target(), [_evidence("ev_000001")])
    assert len(scorer.triage(cset, k_top=5)) == 1


def test_triage_empty_when_zero_gated() -> None:
    scorer = AssociationScorer(embedder=None)
    cset = scorer.score_candidates(
        _target(label="statin therapy"),
        [_evidence("ev_000009", title="unrelated cooking recipe")],
    )
    assert scorer.triage(cset) == ()


# --- Scorer rejects forbidden workflow-state inputs ----------------------------------

@pytest.mark.parametrize(
    "forbidden",
    [
        {"user_priority": 0.9},
        {"uncertainty": 0.3},
        {"verdict": "supported"},
        {"status": "verified"},
        {"confidence": 0.8},
    ],
)
def test_scorer_rejects_forbidden_inputs(forbidden) -> None:
    scorer = AssociationScorer(embedder=None)
    with pytest.raises(TypeError):
        scorer.score(_target(), _evidence(), **forbidden)


# --- Configurable scoring coefficients ------------------------------------------------

def test_changing_a_default_weight_changes_the_score() -> None:
    target, paper = _target(), _evidence()
    base = AssociationScorer(embedder=None).score(target, paper)
    alt_defaults = SimpleNamespace(**{k: getattr(sd, k) for k in dir(sd) if k.isupper()})
    alt_defaults.ASSOCIATION_WEIGHTS = {"semantic": 0.9, "citation": 0.05, "source_prior": 0.05}
    alt_defaults.SCORING_DEFAULTS_VERSION = "alt-test"
    bumped = AssociationScorer(embedder=None, defaults=alt_defaults).score(target, paper)
    assert bumped.association_score != base.association_score        # path reads the module
    assert bumped.scoring_defaults_version == "alt-test"            # stamp follows defaults


# --- _clip01 re-export from retrieval.similarity (consolidation regression) --

def test_clip01_still_clamps_to_unit_interval() -> None:
    import src.retrieval.association as assoc

    assert assoc._clip01(-0.2) == 0.0
    assert assoc._clip01(1.5) == 1.0
    assert assoc._clip01(0.3) == pytest.approx(0.3, abs=1e-6)


# --- Typed candidate-evidence collection ----------------------------------------------

def test_score_candidates_returns_typed_collection() -> None:
    scorer = AssociationScorer(embedder=None)
    cset = scorer.score_candidates(_target(), [_evidence("ev_000001"), _evidence("ev_000002")])
    assert isinstance(cset, CandidateEvidenceSet)
    assert len(cset.items) == 2
    for item in cset.items:
        assert isinstance(item.bundle, EvidenceSignalBundle)
        assert item.bundle.evidence_id == item.evidence.evidence_id


# --- Opt-in live lane -----------------------------------------------------------------

@pytest.mark.live
def test_real_specter2_embedder_drives_embedding_channel() -> None:
    """Real `specter2_base` model through the scorer; topical paper ranks higher."""
    from src.retrieval.coherence import load_specter2_embedder

    embedder = load_specter2_embedder(sd.EMBEDDING_MODEL)
    if embedder is None:
        pytest.skip("specter2 embedder / retrieval-coherence extra unavailable")

    scorer = AssociationScorer(embedder=embedder)
    target = _target(label="statin therapy", definition="HMG-CoA reductase inhibitor")
    related = _evidence("ev_000001", title="Statins reduce cardiovascular mortality")
    unrelated = _evidence("ev_000002", title="Photosynthesis in tropical ferns")

    b_related = scorer.score(target, related)
    b_unrelated = scorer.score(target, unrelated)

    assert b_related.embedding_fallback is False  # real model used, not lexical
    assert b_unrelated.embedding_fallback is False
    assert b_related.embedding_score > b_unrelated.embedding_score
    # The pinned model produces the same value when rescored.
    assert scorer.score(target, related).embedding_score == pytest.approx(
        b_related.embedding_score, abs=1e-6
    )


@pytest.mark.live
@pytest.mark.live_openalex
def test_scores_real_openalex_evidence_reproducibly() -> None:
    """Fetch real OpenAlex papers, score them, and confirm reproducible bundles."""
    from src.config import OpenAlexSourceConfig
    from src.retrieval.models import SearchPaperFilters
    from src.retrieval.sources import OpenAlexPaperSource

    source = OpenAlexPaperSource(
        OpenAlexSourceConfig(mailto="graph-hypoth-tests@graph-hypoth.ai", timeout_seconds=30.0)
    )
    result = source.search(
        "statin therapy cardiovascular mortality", limit=5, filters=SearchPaperFilters()
    )
    if result.status.status != "success" or not result.results:
        pytest.skip("live OpenAlex returned no results")

    evidences = [
        RetrievedEvidence(
            evidence_id=f"ev_{index + 1:06d}",
            source="openalex",
            source_id=item.source_id,
            title=item.title,
            quote=item.summary or "",
            relevance="r",
            retrieved_by="builder",
            tool_call_id="tc",
            score=item.score,
            rank=index + 1,
            trust_tier="indexed_metadata",
            external_ids=dict(item.external_ids),
            metadata=dict(item.metadata),
        )
        for index, item in enumerate(result.results)
    ]

    scorer = AssociationScorer(embedder=None)  # retrieval path; embedder lane is separate
    target = _target(label="statin therapy", evidence_role="support")
    first = scorer.score_candidates(target, evidences)
    second = scorer.score_candidates(target, evidences)

    assert len(first.items) == len(evidences)
    for a, b in zip(first.items, second.items):
        assert a.bundle.association_score == pytest.approx(b.bundle.association_score, abs=1e-6)
        assert a.bundle.scoring_defaults_version == sd.SCORING_DEFAULTS_VERSION
    # triage forwards at most K_top gated candidates, all gate-passing.
    forwarded = scorer.triage(first)
    assert len(forwarded) <= sd.K_TOP
    assert all(scorer.gate(c.bundle) for c in forwarded)


# --- Batched citation scoring: subquadratic and byte-identical ----------------------------

def _cite_ev(eid, *, refs=(), doi=None, source_id=None, rank=1):
    return _evidence(
        eid, source="openalex", source_id=source_id, rank=rank,
        metadata={"references": list(refs)},
    ).model_copy(update={"external_ids": ({"doi": doi} if doi else {})})


def test_citation_batch_leg_outputs_are_byte_identical_oracle() -> None:
    # Golden capture from the naive O(N^2) impl: the de-quadratic index path must
    # reproduce the citation_score AND the stable, batch-order tie-broken neighbor
    # tuples EXACTLY (coupling, co-citation, ties, and a disjoint record).
    batch = [
        _cite_ev("ev_000001", refs=["r1", "r2", "r3"]),
        _cite_ev("ev_000002", refs=["r1", "r2"]),       # coupling 2/3 with e1
        _cite_ev("ev_000003", refs=["r2", "r3"]),       # coupling 2/3 with e1 (tie)
        _cite_ev("ev_000004", refs=["r1"]),             # coupling 1/3 with e1
        _cite_ev("ev_000005", doi="r1"),                # e1.refs ∩ e5.ids -> co_citation 1.0
        _cite_ev("ev_000006", refs=["zz"], doi="qq"),   # disjoint -> 0
    ]
    scorer = AssociationScorer(embedder=None)
    bundles = {
        item.bundle.evidence_id: item.bundle
        for item in scorer.score_candidates(_target(label="alpha"), batch).items
    }
    expected = {
        "ev_000001": (0.3, ("ev_000005", "ev_000002", "ev_000003", "ev_000004")),
        "ev_000002": (0.3, ("ev_000005", "ev_000001", "ev_000004", "ev_000003")),
        "ev_000003": (0.2, ("ev_000001", "ev_000002")),
        "ev_000004": (0.3, ("ev_000005", "ev_000002", "ev_000001")),
        "ev_000005": (0.3, ("ev_000001", "ev_000002", "ev_000004")),
        "ev_000006": (0.0, ()),
    }
    for eid, (score, neighbors) in expected.items():
        assert bundles[eid].citation_score == pytest.approx(score, abs=1e-6)
        assert bundles[eid].citation_neighbors == neighbors


def test_citation_batch_leg_is_subquadratic(monkeypatch) -> None:
    # 20 disjoint reference-pairs: each evidence cites at most one other, so the
    # candidate set is O(1) per evidence. The naive impl calls overlap() ~N*(N-1)
    # times (1560); the index path must call it ~O(N).
    import src.retrieval.association as assoc

    calls = {"n": 0}
    real_overlap = assoc.overlap

    def counting_overlap(*args, **kwargs):
        calls["n"] += 1
        return real_overlap(*args, **kwargs)

    monkeypatch.setattr(assoc, "overlap", counting_overlap)

    batch = []
    for pair in range(20):
        token = f"ref-{pair}"
        batch.append(_cite_ev(f"ev_{2 * pair + 1:06d}", refs=[token]))
        batch.append(_cite_ev(f"ev_{2 * pair + 2:06d}", refs=[token]))

    scorer = AssociationScorer(embedder=None)
    scorer.score_candidates(_target(label="alpha"), batch)

    n = len(batch)  # 40
    assert calls["n"] < 4 * n  # << n*(n-1) = 1560
