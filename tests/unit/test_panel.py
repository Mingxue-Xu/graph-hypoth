"""Critic Panel — listwise, grounded, different-model review.

Hermetic: a fake backend returns fixtured STRICT-JSON; the deterministic parse + aggregation are
asserted. The panel is independent — it NEVER sees the proposer's rationale / idea_scaffold /
``llm_signals``. Real-LLM behavior stays behind the ``live`` marker.
"""

from __future__ import annotations

import pytest

from src.cycles.hypothesis import HypothesisCandidate
from src.delta import build_node


class _FakeBackend:
    def __init__(self, content):
        self._content = content
        self.seen = None

    def run(self, messages, tools=None):
        self.seen = messages
        return {"choices": [{"message": {"content": self._content}}]}


def _candidate(cid, *, label="rate-distortion floor"):
    return HypothesisCandidate(
        candidate_id=cid,
        new_nodes=(build_node(label=label, type="mechanism", definition="min distortion"),),
        mechanism_chain=(
            {"from": label, "relation": "bounds", "to": "factuality", "mechanism": "caps signal"},
        ),
        source_quotes=(
            {"evidence_id": "ev_001", "quote_span": label, "role_in_hypothesis": "the mechanism"},
        ),
        rationale="PERSUASION-SENTINEL: obviously groundbreaking",
        idea_scaffold={"lever": "SCAFFOLD-SENTINEL"},
        novelty=0.5,
        testability=0.6,
    )


_REVIEW_JSON = (
    '{"candidates": [{"candidate_id": "h1", "field_novelty": 0.7, "saturation": 0.25,'
    ' "already_established": false, "not_judgeable_by_field": false,'
    ' "mechanism_steps": [{"from": "rate-distortion floor", "relation": "bounds",'
    ' "to": "factuality", "verdict": "sound", "note": "consistent with edges"}],'
    ' "term_verdicts": [{"term": "rate-distortion floor", "verdict": "consistent",'
    ' "note": "matches source"}], "justification": "novel per Corpus P1"}],'
    ' "ranking": ["h1"], "critique": "strong mechanism; probe the calibration assumption"}'
)


def test_panel_judge_parses_listwise_review():
    from src.cycles.panel import LLMCriticPanelJudge

    judge = LLMCriticPanelJudge(_FakeBackend(_REVIEW_JSON), judge_id=1, reference_field="LLM compression")
    review = judge.review([_candidate("h1")], corpus_list="P1: MDL study", cited_passages="ev_001: body")
    assert [g.candidate_id for g in review.grades] == ["h1"]
    g = review.grades[0]
    assert g.field_novelty == pytest.approx(0.7, abs=1e-6)
    assert g.saturation == pytest.approx(0.25, abs=1e-6)
    assert g.already_established is False and g.not_judgeable_by_field is False
    assert g.mechanism_steps[0]["verdict"] == "sound"
    assert g.term_verdicts[0]["verdict"] == "consistent"
    assert list(review.ranking) == ["h1"]
    assert "calibration" in review.critique


@pytest.mark.parametrize(
    ("relation_json", "expected"),
    [
        ('"--bounds-->"', "bounds"),
        ('"→bounds→"', "bounds"),
        ('"dose-dependent"', "dose-dependent"),
        ("null", ""),
        ("0", ""),
        ("false", ""),
    ],
)
def test_panel_normalizes_relation_labels(relation_json, expected):
    from src.cycles.panel import LLMCriticPanelJudge

    content = _REVIEW_JSON.replace(
        '"relation": "bounds"', f'"relation": {relation_json}'
    )
    review = LLMCriticPanelJudge(
        _FakeBackend(content), judge_id=1, reference_field="X"
    ).review([_candidate("h1")], corpus_list="P1", cited_passages="ev_001: body")
    assert review.grades[0].mechanism_steps[0]["relation"] == expected


def test_panel_judge_is_independent_of_proposer_rationale():
    from src.cycles.panel import LLMCriticPanelJudge

    backend = _FakeBackend(_REVIEW_JSON)
    LLMCriticPanelJudge(backend, judge_id=1, reference_field="X").review(
        [_candidate("h1")], corpus_list="P1", cited_passages="ev_001: body"
    )
    joined = " ".join(str(m.get("content", "")) for m in backend.seen)
    # independence: the panel must NOT see the proposer's rationale / idea_scaffold / llm_signals
    assert "PERSUASION-SENTINEL" not in joined
    assert "SCAFFOLD-SENTINEL" not in joined
    # ...but the candidate's OWN structural content IS present.
    assert "rate-distortion floor" in joined and "h1" in joined


def test_aggregate_panel_medians_grades_and_borda_fuses_ranking():  # median grades and rank fusion
    from src.cycles.panel import CandidateGrade, PanelReview, aggregate_panel

    a = PanelReview(
        grades=(
            CandidateGrade("h1", field_novelty=0.8, saturation=0.2, already_established=False),
            CandidateGrade("h2", field_novelty=0.4, saturation=0.6, already_established=True),
        ),
        ranking=("h1", "h2"),
        critique="A: sharpen h2",
    )
    b = PanelReview(
        grades=(
            CandidateGrade("h1", field_novelty=0.6, saturation=0.3, already_established=False),
            CandidateGrade("h2", field_novelty=0.5, saturation=0.5, already_established=True),
        ),
        ranking=("h2", "h1"),
        critique="B: h1 strongest",
    )
    agg = aggregate_panel([a, b])
    by_id = {g.candidate_id: g for g in agg.grades}
    assert by_id["h1"].field_novelty == pytest.approx(0.7, abs=1e-6)  # median(0.8, 0.6)
    assert by_id["h1"].saturation == pytest.approx(0.25, abs=1e-6)  # median(0.2, 0.3)
    assert by_id["h2"].field_novelty == pytest.approx(0.45, abs=1e-6)
    assert by_id["h2"].already_established is True  # majority vote
    # Borda over [h1,h2] + [h2,h1] ties (1 pt each) -> deterministic candidate_id tie-break
    assert list(agg.ranking) == ["h1", "h2"]
    assert "sharpen h2" in agg.critique and "h1 strongest" in agg.critique


def test_panel_disagreement_true_on_top_of_pool_conflict():  # third judge only on disagreement
    from src.cycles.panel import CandidateGrade, PanelReview, panel_disagreement

    a = PanelReview(grades=(CandidateGrade("h1"), CandidateGrade("h2")), ranking=("h1", "h2"))
    b = PanelReview(grades=(CandidateGrade("h1"), CandidateGrade("h2")), ranking=("h2", "h1"))
    # the base judges disagree on which candidate surfaces first -> escalate to the 3rd judge
    assert panel_disagreement([a, b]) is True


def test_panel_disagreement_true_on_decisive_boolean_split():  # third judge breaks a tie
    from src.cycles.panel import CandidateGrade, PanelReview, panel_disagreement

    a = PanelReview(grades=(CandidateGrade("h1", already_established=True),), ranking=("h1",))
    b = PanelReview(grades=(CandidateGrade("h1", already_established=False),), ranking=("h1",))
    assert panel_disagreement([a, b]) is True  # 1-1 on already_established -> no majority with 2 judges


def test_panel_disagreement_false_when_base_judges_concur():  # no escalation on agreement
    from src.cycles.panel import CandidateGrade, PanelReview, panel_disagreement

    a = PanelReview(
        grades=(CandidateGrade("h1", already_established=False, field_novelty=0.8),
                CandidateGrade("h2", already_established=True)),
        ranking=("h1", "h2"),
    )
    b = PanelReview(
        grades=(CandidateGrade("h1", already_established=False, field_novelty=0.6),
                CandidateGrade("h2", already_established=True)),
        ranking=("h1", "h2"),
    )
    # same top candidate + same booleans (a differing median field_novelty is not a "defined
    # disagreement" — median already fuses it) -> no 3rd judge
    assert panel_disagreement([a, b]) is False
    # a single-judge (or empty) panel can never disagree with itself
    assert panel_disagreement([a]) is False


def test_apply_panel_sets_novelty_graded_and_saturation():  # feeds hypothesis and rank scores
    from src.cycles.panel import CandidateGrade, PanelReview, aggregate_panel, apply_panel

    agg = aggregate_panel([PanelReview(
        grades=(CandidateGrade("h1", field_novelty=0.7, saturation=0.25),), ranking=("h1",))])
    out = apply_panel([_candidate("h1")], agg)
    assert out[0].novelty_graded == pytest.approx(0.7, abs=1e-6)
    assert out[0].saturation == pytest.approx(0.25, abs=1e-6)
    # a candidate the panel did not grade passes through unchanged (novelty_graded stays None)
    out2 = apply_panel([_candidate("hX")], agg)
    assert out2[0].novelty_graded is None


def test_panel_result_for_revise_flags_vague_steps_and_term_issues():  # revision input
    from src.cycles.panel import CandidateGrade, PanelReview, aggregate_panel, panel_result_for_revise

    flagged = PanelReview(
        grades=(CandidateGrade(
            "h1", field_novelty=0.5, saturation=0.5,
            mechanism_steps=({"from": "a", "relation": "r", "to": "b", "verdict": "vague", "note": "no mechanism"},),
            term_verdicts=({"term": "floor", "verdict": "stretched", "note": "extends meaning"},),
        ),),
        ranking=("h1",), critique="probe the calibration warrant",
    )
    pr = panel_result_for_revise(aggregate_panel([flagged]))
    assert pr["ranking"] == ["h1"]
    assert "calibration" in pr["critique"]
    assert len(pr["audit_flags"]) == 1 and pr["audit_flags"][0]["candidate_id"] == "h1"
    assert pr["audit_flags"][0]["mechanism_steps"][0]["verdict"] == "vague"
    assert pr["audit_flags"][0]["term_issues"][0]["verdict"] == "stretched"
    # an all-sound / consistent candidate yields NO flag block (nothing to repair)
    clean = PanelReview(grades=(CandidateGrade(
        "h2",
        mechanism_steps=({"from": "a", "relation": "r", "to": "b", "verdict": "sound", "note": ""},),
        term_verdicts=({"term": "x", "verdict": "consistent", "note": ""},),
    ),), ranking=("h2",))
    assert panel_result_for_revise(aggregate_panel([clean]))["audit_flags"] == []
