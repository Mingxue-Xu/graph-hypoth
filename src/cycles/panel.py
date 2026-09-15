"""Listwise, grounded Critic Panel reviews.

Each judge reads the full candidate pool's structural content, the shared corpus, and cited
passages. It grades field novelty, saturation, mechanism steps, and terminology; ranks every
candidate; and returns a pool-level critique. Deterministic aggregation combines median grades
and rank fusion before the eligibility, hypothesis-score, and rank-score calculations.

The panel never sees the proposer's rationale, idea scaffold, or self-assessed signals. Model
backends are injected, and parsing reuses ``camel_adapter`` so the deterministic core remains
LLM-free.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol

from src.relation_labels import normalize_relation_label

if TYPE_CHECKING:
    from src.cycles.hypothesis import HypothesisCandidate

# Per-judge system prompt with judge and reference-field placeholders.
CRITIC_PANEL_SYSTEM_PROMPT = """<role>
You are SENIOR REVIEWER #{judge_id} (of an independent panel) for the field of {reference_field}. In one turn you review the whole pool of proposed causal hypotheses for one claim graph: grade every candidate, rank the pool, and write one pool-level critique. You did not propose these hypotheses; judge them skeptically.
</role>

<rules>
Grounding:
1. You see only each candidate's structural content — its new concept nodes, new typed edges, its stated mechanism chain, and its verbatim source quotes — plus the shared retrieved corpus and cited passages. You never see, and must not request, the proposer's rationale, idea_scaffold, or llm_signals. Judge only from the material provided.

Per candidate — novelty and saturation:
2. Grade its FIELD-RELATIVE NOVELTY (how new is this to a practitioner in this field, relative to the STATE OF THE FIELD — NOT relative to any one graph) and its LITERATURE SATURATION (how strongly the retrieved corpus + the field already assert this relationship as an established result).
3. A textbook / common-sense restatement is LOW novelty and HIGH saturation; a genuine surprising cross-concept or predictor->outcome link is HIGHER novelty and LOWER saturation.
4. If a candidate is only novel/interesting to a different field (not judgeable by a practitioner of this field), grade its field_novelty low for this field and set not_judgeable_by_field=true.
5. Each candidate's justification must cite specific corpus papers.

Per candidate — mechanism chain:
6. Grade each stated link of its mechanism chain; mechanism_steps must mirror the candidate's stated chain, one entry per step, in order: 'sound' = meaningful and consistent with the typed edges/graph and not contradicted by the evidence; 'vague' = concepts juxtaposed without a stated meaningful relation; 'false' = the stated link contradicts the typed edge structure or the evidence. Never invent concepts absent from the candidate's chain or edges.

Per candidate — terminology:
7. Grade each technical or paper-coined term its nodes, edges, or chain rely on, judging the candidate's usage against the term's verbatim source quote(s): 'consistent' = same meaning; 'stretched' = related but extends or shifts the source meaning without saying so; 'misused' = contradicts or ignores the source meaning. A term with no source quote is 'no_source' — never guess a meaning. Judge meaning, not phrasing.

Pool level:
8. Rank the pool: decide which candidates should be surfaced first for this graph, weighing novelty, plausibility, testability, expected verification yield, mechanism specificity, centrality to the current graph, and usefulness to the user's claim. You are ranking the pool only — do not invent, rewrite, or report any numeric sub-score beyond the output fields specified below; ranking must list every candidate_id exactly once, best first.
9. Write a pool-level critique (2-4 sentences) a revision agent can act on: which traits win and which lose across this pool, and the strongest unaddressed assumption the surviving hypotheses still rest on, so the reviser can probe that warrant next. Be specific and grounded in the retrieved evidence.
</rules>

<output_format>
Return STRICT JSON only (no markdown):
{"candidates": [{"candidate_id", "field_novelty": 0..1, "saturation": 0..1, "already_established": true|false, "not_judgeable_by_field": true|false,
  "mechanism_steps": [{"from","relation","to","verdict":"sound|vague|false","note"}],
  "term_verdicts": [{"term","verdict":"consistent|stretched|misused|no_source","note"}],
  "justification": "cite specific corpus papers"}, …one object per candidate, in input order…],
 "ranking": [candidate ids, best first, every id exactly once],
 "critique": "2-4 sentences"}
</output_format>"""


def _edge_view(edge: Any, label_of: dict[str, str]) -> str:
    src = " & ".join(label_of.get(s, s) for s in edge.source_node_ids)
    tgt = " & ".join(label_of.get(t, t) for t in edge.target_node_ids)
    mechanism = f" [{edge.mechanism}]" if edge.mechanism else ""
    return f"{src} --{edge.relation_type}--> {tgt}{mechanism}"


def _critic_panel_candidate_pool_block(
    candidates: Sequence[HypothesisCandidate],
    node_labels: dict[str, str] | None,
    titles: dict[str, str] | None = None,
) -> str:
    """Render only each candidate's structural content, with no rationale,
    idea_scaffold, or llm_signals (independence). Edge endpoints resolve to labels via the committed
    ``node_labels`` map plus every candidate's own new nodes; ``titles`` maps evidence_id -> source
    title for the ``[evidence_id | title]`` source-quote render (empty title until wired)."""
    label_of = dict(node_labels or {})
    title_of = dict(titles or {})
    for cand in candidates:
        for node in cand.new_nodes:
            label_of.setdefault(node.node_id, node.label)

    blocks: list[str] = []
    for cand in candidates:
        nodes = "; ".join(f"{n.label} ({n.type}): {n.definition}" for n in cand.new_nodes) or "(none)"
        edges = "; ".join(_edge_view(e, label_of) for e in cand.new_edges) or "(none)"
        chain = "; ".join(
            f"{s.get('from', '')} --{s.get('relation', '')}--> {s.get('to', '')}"
            for s in cand.mechanism_chain
        ) or "(none)"
        quotes = "; ".join(
            f'[{q.get("evidence_id")} | {title_of.get(str(q.get("evidence_id")), "")}] '
            f'"{q.get("quote_span", "")}"'
            for q in cand.source_quotes
        ) or "(none)"
        blocks.append(
            f"### Candidate {cand.candidate_id}\n"
            f"New concept node(s): {nodes}\n"
            f"New causal edge(s): {edges}\n"
            f"Mechanism chain: {chain}\n"
            f"Source quotes (verbatim spans): {quotes}"
        )
    return "\n\n".join(blocks)


def _critic_panel_user(
    candidates: Sequence[HypothesisCandidate],
    *,
    corpus_list: str,
    cited_passages: str,
    node_labels: dict[str, str] | None,
    titles: dict[str, str] | None = None,
) -> str:
    return (
        '<retrieved_corpus note="your saturation reference set">\n'
        f"{corpus_list}\n</retrieved_corpus>\n\n"
        '<cited_passages note="shared grounding — the passages the candidates\' quotes are drawn from">\n'
        f"{cited_passages}\n</cited_passages>\n\n"
        f"<candidate_pool>\n{_critic_panel_candidate_pool_block(candidates, node_labels, titles)}\n</candidate_pool>\n\n"
        "Grade EVERY candidate, rank the pool best-first, and write the pool critique. Return STRICT JSON."
    )


@dataclass(frozen=True)
class CandidateGrade:
    """One judge's grade for one candidate. ``field_novelty`` / ``saturation`` are the DECISIVE
    ranking signals aggregated into HypScore / RankScore; ``mechanism_steps`` (absorbs ) and
    ``term_verdicts`` (absorbs ) drive the Research Synthesist revise-turn audit flags."""

    candidate_id: str
    field_novelty: float = 0.0
    saturation: float = 0.0
    already_established: bool = False
    not_judgeable_by_field: bool = False
    mechanism_steps: tuple[dict[str, Any], ...] = ()
    term_verdicts: tuple[dict[str, Any], ...] = ()
    justification: str = ""


@dataclass(frozen=True)
class PanelReview:
    """One judge's whole-pool review: per-candidate grades + a best-first ranking + a pool critique."""

    grades: tuple[CandidateGrade, ...] = ()
    ranking: tuple[str, ...] = ()
    critique: str = ""


def _parse_panel_review(data: dict[str, Any]) -> PanelReview:
    grades: list[CandidateGrade] = []
    for raw in data.get("candidates") or []:
        if not isinstance(raw, dict):
            continue
        cid = str(raw.get("candidate_id", "")).strip()
        if not cid:
            continue
        grades.append(
            CandidateGrade(
                candidate_id=cid,
                field_novelty=float(raw.get("field_novelty", 0.0)),
                saturation=float(raw.get("saturation", 0.0)),
                already_established=bool(raw.get("already_established", False)),
                not_judgeable_by_field=bool(raw.get("not_judgeable_by_field", False)),
                mechanism_steps=tuple(
                    {
                        "from": str(s.get("from", "")),
                        "relation": normalize_relation_label(s.get("relation", "")),
                        "to": str(s.get("to", "")),
                        "verdict": str(s.get("verdict", "")),
                        "note": str(s.get("note", "")),
                    }
                    for s in (raw.get("mechanism_steps") or [])
                    if isinstance(s, dict)
                ),
                term_verdicts=tuple(
                    {
                        "term": str(t.get("term", "")),
                        "verdict": str(t.get("verdict", "")),
                        "note": str(t.get("note", "")),
                    }
                    for t in (raw.get("term_verdicts") or [])
                    if isinstance(t, dict)
                ),
                justification=str(raw.get("justification", "")),
            )
        )
    return PanelReview(
        grades=tuple(grades),
        ranking=tuple(str(cid) for cid in (data.get("ranking") or [])),
        critique=str(data.get("critique", "")),
    )


class CriticPanelJudge(Protocol):
    """One Critic Panel judge (live LLM path; fixtured in the default suite). A listwise, grounded,
    independent reviewer of the whole candidate pool."""

    def review(
        self,
        candidates: Sequence[HypothesisCandidate],
        *,
        corpus_list: str = "",
        cited_passages: str = "",
        node_labels: dict[str, str] | None = None,
        titles: dict[str, str] | None = None,
    ) -> PanelReview: ...


class LLMCriticPanelJudge:
    """A real ``CriticPanelJudge`` over any CAMEL model backend (``.run(messages)``).

    ``judge_id`` and ``reference_field`` are baked into the Critic Panel system prompt via
    ``str.replace`` — the prompt contains literal JSON braces, so ``str.format`` is unusable); the
    retrieved corpus + cited passages + candidate pool go in the user prompt. A malformed/empty
    response degrades to an empty review. Parsing reuses ``camel_adapter`` so the deterministic core
    stays LLM-free."""

    def __init__(
        self,
        model_backend: Any,
        *,
        judge_id: int = 1,
        reference_field: str = "",
        system_prompt: str | None = None,
    ) -> None:
        self._backend = model_backend
        base = system_prompt or CRITIC_PANEL_SYSTEM_PROMPT
        self._system_prompt = base.replace("{judge_id}", str(judge_id)).replace(
            "{reference_field}", reference_field or "the field"
        )

    def review(
        self,
        candidates: Sequence[HypothesisCandidate],
        *,
        corpus_list: str = "",
        cited_passages: str = "",
        node_labels: dict[str, str] | None = None,
        titles: dict[str, str] | None = None,
    ) -> PanelReview:
        from src.camel_adapter import backend_json

        user = _critic_panel_user(
            candidates,
            corpus_list=corpus_list,
            cited_passages=cited_passages,
            node_labels=node_labels,
            titles=titles,
        )
        return _parse_panel_review(backend_json(self._backend, self._system_prompt, user))


# Deterministic aggregation combines median grades and rank fusion. The panel supplies grades;
# eligibility, scoring, and ordering remain deterministic.

# Severity orderings break majority-vote TIES toward the more-serious verdict, so a split panel
# surfaces the concern for the reviser rather than hiding it (deterministic, total order via value).
_STEP_SEVERITY = {"false": 2, "vague": 1, "sound": 0}
_TERM_SEVERITY = {"misused": 3, "no_source": 2, "stretched": 1, "consistent": 0}


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _majority_bool(values: Sequence[bool]) -> bool:
    """Strict-majority vote — mirrors the shipped ``aggregate_judgments`` reducer."""
    return sum(1 for v in values if v) * 2 > len(values)


def _majority_verdict(values: Sequence[str], severity: dict[str, int]) -> str:
    counts = Counter(values)
    return max(counts, key=lambda v: (counts[v], severity.get(v, 0), v))


def _aggregate_steps(grades: Sequence[CandidateGrade]) -> tuple[dict[str, Any], ...]:
    """Per-step majority verdict over the judges (mechanism_steps mirror the candidate's stated
    chain, so step ``i`` aligns across judges); labels from the first judge, note = first non-empty."""
    max_len = max((len(g.mechanism_steps) for g in grades), default=0)
    out: list[dict[str, Any]] = []
    for i in range(max_len):
        steps_i = [g.mechanism_steps[i] for g in grades if i < len(g.mechanism_steps)]
        if not steps_i:
            continue
        first = steps_i[0]
        out.append(
            {
                "from": first.get("from", ""),
                "relation": first.get("relation", ""),
                "to": first.get("to", ""),
                "verdict": _majority_verdict([s.get("verdict", "") for s in steps_i], _STEP_SEVERITY),
                "note": next((s.get("note", "") for s in steps_i if s.get("note")), ""),
            }
        )
    return tuple(out)


def _aggregate_terms(grades: Sequence[CandidateGrade]) -> tuple[dict[str, Any], ...]:
    """Per-term majority verdict, aggregated by term name (judges pick which terms to grade)."""
    by_term: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for grade in grades:
        for term in grade.term_verdicts:
            name = term.get("term", "")
            if name not in by_term:
                by_term[name] = []
                order.append(name)
            by_term[name].append(term)
    return tuple(
        {
            "term": name,
            "verdict": _majority_verdict([t.get("verdict", "") for t in by_term[name]], _TERM_SEVERITY),
            "note": next((t.get("note", "") for t in by_term[name] if t.get("note")), ""),
        }
        for name in order
    )


def _borda_ranking(reviews: Sequence[PanelReview], candidate_ids: Sequence[str]) -> list[str]:
    """Borda rank fusion of the per-judge rankings: a candidate at position p of an n-long ranking
    scores (n-1-p); summed across judges; sorted by (-score, candidate_id) for a total order
    (candidates no judge ranked score 0 and sort last, deterministically)."""
    scores = {cid: 0 for cid in candidate_ids}
    for review in reviews:
        n = len(review.ranking)
        for pos, cid in enumerate(review.ranking):
            scores[cid] = scores.get(cid, 0) + (n - 1 - pos)
    return sorted(scores, key=lambda cid: (-scores[cid], cid))


def panel_disagreement(reviews: Sequence[PanelReview]) -> bool:
    """Detect disagreement that requires a third judge.

    This threshold-free deterministic check returns True when base judges disagree on the pool's
    TOP candidate (else an arbitrary Borda tie-break would decide what surfaces first), OR split on
    any candidate's decisive boolean flag (a 1-1 vote no strict majority resolves). A differing
    median grade is NOT a disagreement — ``_median`` already fuses it. A single/empty panel never
    disagrees with itself, so no escalation fires."""
    tops = [r.ranking[0] for r in reviews if r.ranking]
    if len(tops) >= 2 and len(set(tops)) > 1:
        return True
    flags_by_cid: dict[str, list[tuple[bool, bool]]] = {}
    for review in reviews:
        for grade in review.grades:
            flags_by_cid.setdefault(grade.candidate_id, []).append(
                (grade.already_established, grade.not_judgeable_by_field)
            )
    return any(
        len({v[0] for v in votes}) > 1 or len({v[1] for v in votes}) > 1
        for votes in flags_by_cid.values()
    )


@dataclass(frozen=True)
class AggregatedGrade:
    """The panel's deterministic consensus for one candidate: median field_novelty / saturation
    (the DECISIVE ranking signals), majority flags, and merged mechanism_step / term verdicts."""

    candidate_id: str
    field_novelty: float = 0.0
    saturation: float = 0.0
    already_established: bool = False
    not_judgeable_by_field: bool = False
    mechanism_steps: tuple[dict[str, Any], ...] = ()
    term_verdicts: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class AggregatedPanel:
    grades: tuple[AggregatedGrade, ...] = ()
    ranking: tuple[str, ...] = ()
    critique: str = ""


def aggregate_panel(reviews: Sequence[PanelReview]) -> AggregatedPanel:
    """Reduce a panel of per-judge reviews to one deterministic consensus: median grades +
    majority flags/verdicts per candidate, Borda-fused ranking, and the joined pool critique."""
    order: list[str] = []
    grades_by_cid: dict[str, list[CandidateGrade]] = {}
    for review in reviews:
        for grade in review.grades:
            if grade.candidate_id not in grades_by_cid:
                grades_by_cid[grade.candidate_id] = []
                order.append(grade.candidate_id)
            grades_by_cid[grade.candidate_id].append(grade)

    aggregated = tuple(
        AggregatedGrade(
            candidate_id=cid,
            field_novelty=_median([g.field_novelty for g in grades_by_cid[cid]]),
            saturation=_median([g.saturation for g in grades_by_cid[cid]]),
            already_established=_majority_bool([g.already_established for g in grades_by_cid[cid]]),
            not_judgeable_by_field=_majority_bool([g.not_judgeable_by_field for g in grades_by_cid[cid]]),
            mechanism_steps=_aggregate_steps(grades_by_cid[cid]),
            term_verdicts=_aggregate_terms(grades_by_cid[cid]),
        )
        for cid in order
    )
    return AggregatedPanel(
        grades=aggregated,
        ranking=tuple(_borda_ranking(reviews, order)),
        critique="\n".join(r.critique for r in reviews if r.critique),
    )


def apply_panel(
    candidates: Sequence[HypothesisCandidate], aggregated: AggregatedPanel
) -> list[HypothesisCandidate]:
    """Write the panel's DECISIVE grades onto the candidates for the deterministic core: field
    novelty -> ``novelty_graded`` (HypScore's ranking novelty) and ``saturation`` (RankScore
    demotion). Candidates the panel did not grade pass through unchanged (mirrors the shipped
    ``apply_novelty_judgment``)."""
    grade_by_cid = {g.candidate_id: g for g in aggregated.grades}
    out: list[HypothesisCandidate] = []
    for candidate in candidates:
        grade = grade_by_cid.get(candidate.candidate_id)
        if grade is None:
            out.append(candidate)
        else:
            out.append(
                replace(
                    candidate,
                    novelty_graded=grade.field_novelty,
                    saturation=grade.saturation,
                    mechanism_steps=grade.mechanism_steps,
                    term_audit=grade.term_verdicts,
                )
            )
    return out


def panel_result_for_revise(aggregated: AggregatedPanel) -> dict[str, Any]:
    """Project panel consensus into the Research Synthesist revision input.

    Include the pool ranking, critique, and one audit-flag block per candidate that has
    any vague/false mechanism step or any stretched/misused/no_source term (unflagged = no block)."""
    audit_flags: list[dict[str, Any]] = []
    for grade in aggregated.grades:
        bad_steps = [
            dict(s) for s in grade.mechanism_steps if s.get("verdict") in ("vague", "false")
        ]
        bad_terms = [
            {"term": t.get("term", ""), "verdict": t.get("verdict", ""), "note": t.get("note", "")}
            for t in grade.term_verdicts
            if t.get("verdict") in ("stretched", "misused", "no_source")
        ]
        if bad_steps or bad_terms:
            audit_flags.append(
                {
                    "candidate_id": grade.candidate_id,
                    "mechanism_steps": bad_steps,
                    "term_issues": bad_terms,
                }
            )
    return {
        "ranking": list(aggregated.ranking),
        "critique": aggregated.critique,
        "audit_flags": audit_flags,
    }
