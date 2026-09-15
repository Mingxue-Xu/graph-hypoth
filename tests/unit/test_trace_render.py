"""Tests for deterministic per-hypothesis trace rendering without an LLM.

Rendering starts from structured ``GraphRunResult.surfaced`` rows, so every number, label,
and score is copied programmatically. Plain-language prose is the only LLM-authored part
(one structured completion per hypothesis) and is additive: the structural page renders complete
without it.
"""

from __future__ import annotations

from src.run_path import GraphRunResult, SurfacedHypothesis


class _FakeBackend:
    def __init__(self, content):
        self._content = content

    def run(self, messages, tools=None):
        return {"choices": [{"message": {"content": self._content}}]}


def _row(cid="h5", rank=1):
    return SurfacedHypothesis(
        rank=rank, candidate_id=cid, field_novelty=0.70, saturation=0.25, hyp_score=0.56,
        rank_score=0.42, cross_concept=True, common_sense=False,
        rationale="LLC predicts capability collapse", new_node_labels=("LLC threshold",),
    )


def _result(rows):
    return GraphRunResult(
        graph_id="g", version=3, statuses={}, receipts=(), open_risks=(),
        graph_json={"nodes": {}, "edges": {}}, edge_table=(), audit_memo="", surfaced=tuple(rows),
    )


def test_render_hypothesis_html_is_faithful_to_the_surfaced_row():  # copied programmatically
    from src.trace_render import render_hypothesis_html

    html = render_hypothesis_html(_row())
    for needle in ("h5", "0.70", "0.25", "0.42", "LLC threshold", "LLC predicts capability collapse"):
        assert needle in html
    assert "<html" in html.lower()


def test_render_hypothesis_html_renders_complete_without_prose():  # prose additive, not load-bearing
    from src.trace_render import render_hypothesis_html

    html = render_hypothesis_html(_row(), elaboration=None)
    assert "h5" in html and "0.70" in html  # full structural page with no LLM prose


def test_render_hypothesis_html_injects_prose_when_present():
    from src.trace_render import render_hypothesis_html

    elab = {"plain_sentence": "A skill subnetwork collapses past a threshold.",
            "elaboration": "Mechanism. Why it matters. Unverified but testable.",
            "key_terms": [{"term": "LLC", "source": "Paper A"}]}
    html = render_hypothesis_html(_row(), elaboration=elab)
    assert "A skill subnetwork collapses past a threshold." in html
    assert "LLC" in html and "Paper A" in html


def test_render_hypothesis_html_renders_plain_language_reader_card_first():  # reader-card behavior
    from src.trace_render import render_hypothesis_html

    elab = {
        "headline": "LLC thresholds may reveal when model skills become fragile.",
        "what_might_be_happening": "The model may rely on a small set of internal signals.",
        "why_it_matters": "This could help researchers spot fragile skills before deployment.",
        "how_to_check": "Compare models above and below the threshold on the same tasks.",
        "what_would_change_our_mind": "The idea weakens if failures do not change at the threshold.",
        "status": "This is a testable idea, not a proven conclusion.",
        "key_terms": [{"term": "LLC", "plain_meaning": "a threshold-like model signal"}],
    }
    html = render_hypothesis_html(_row(), elaboration=elab)
    card_start = html.index("Possible explanation")
    scores_start = html.index("field novelty")
    assert card_start < scores_start
    for needle in (
        "What might be happening",
        "Why it matters",
        "How to check",
        "What would change our mind",
        "This is a testable idea, not a proven conclusion.",
    ):
        assert needle in html


def test_plain_language_reader_card_does_not_echo_score_labels():  # reader-card behavior
    from src.trace_render import render_hypothesis_html

    elab = {
        "headline": "LLC thresholds may reveal when model skills become fragile.",
        "what_might_be_happening": "The model may rely on a small set of internal signals.",
        "why_it_matters": "This could help researchers spot fragile skills before deployment.",
        "how_to_check": "Compare models above and below the threshold on the same tasks.",
        "what_would_change_our_mind": "The idea weakens if failures do not change at the threshold.",
        "status": "This is a testable idea, not a proven conclusion.",
        "key_terms": [],
    }
    html = render_hypothesis_html(_row(), elaboration=elab)
    card = html.split("<table>", 1)[0]
    for forbidden in ("HypScore", "RankScore", "field novelty", "saturation", "cross-concept"):
        assert forbidden not in card


def test_seed_claim_precedes_the_plain_language_reader_card():
    from src.trace_render import render_hypothesis_html

    elab = {
        "headline": "LLC thresholds may reveal when model skills become fragile.",
        "what_might_be_happening": "The model may rely on a small set of internal signals.",
        "why_it_matters": "This could help researchers spot fragile skills before deployment.",
        "how_to_check": "Compare models above and below the threshold on the same tasks.",
        "what_would_change_our_mind": "The idea weakens if failures do not change at the threshold.",
        "status": "This is a testable idea, not a proven conclusion.",
        "key_terms": [],
    }
    html = render_hypothesis_html(
        _row(), claim="A long technical seed claim with saturation language.", elaboration=elab
    )
    assert html.index("Claim to investigate:") < html.index("Possible explanation")


def test_render_index_html_lists_the_ranking():
    from src.trace_render import render_index_html

    html = render_index_html(_result([_row("h5", 1), _row("h6", 2)]))
    assert "h5" in html and "h6" in html


def test_write_trace_writes_index_and_per_hypothesis(tmp_path):
    from src.trace_render import write_trace

    paths = write_trace(_result([_row("h5", 1), _row("h6", 2)]), tmp_path)
    names = {p.name for p in paths}
    assert "index.html" in names and "h5.html" in names and "h6.html" in names
    assert (tmp_path / "h5.html").read_text(encoding="utf-8").strip()  # non-empty


def test_elaborator_parses_prose_from_backend():  # the one optional LLM call per hypothesis
    from src.trace_render import LLMHypothesisElaborator

    content = (
        '{"headline": "X may collapse.", "what_might_be_happening": "because Y.",'
        ' "why_it_matters": "it matters", "how_to_check": "measure X",'
        ' "what_would_change_our_mind": "no change",'
        ' "status": "This is a testable idea, not a proven conclusion.",'
        ' "key_terms": [{"term": "LLC", "plain_meaning": "a model signal"}]}'
    )
    elab = LLMHypothesisElaborator(_FakeBackend(content)).elaborate(_row())
    assert elab["headline"] == "X may collapse."
    assert elab["key_terms"][0]["term"] == "LLC"


def test_elaborator_degrades_to_empty_on_malformed():
    from src.trace_render import LLMHypothesisElaborator

    assert LLMHypothesisElaborator(_FakeBackend("garbage")).elaborate(_row()) == {}


def test_write_trace_injects_elaborations_end_to_end(tmp_path):
    from src.trace_render import write_trace

    elaborations = {"h5": {"plain_sentence": "Plain hero sentence.", "elaboration": "", "key_terms": []}}
    write_trace(_result([_row("h5", 1)]), tmp_path, elaborations=elaborations)
    assert "Plain hero sentence." in (tmp_path / "h5.html").read_text(encoding="utf-8")


# --- Inline universal-definition glosses -------------------------------------------------
def test_render_key_terms_includes_universal_definition():
    from src.trace_render import render_hypothesis_html

    elab = {"plain_sentence": "X collapses.", "elaboration": "because Y.", "key_terms": [
        {"term": "LLC", "universal_definition": "effective parameter count / loss-landscape degeneracy",
         "source": "Paper A"}]}
    html = render_hypothesis_html(_row(), elaboration=elab)
    assert "LLC" in html
    assert "effective parameter count / loss-landscape degeneracy" in html  # the universal gloss
    assert "Paper A" in html


def test_elaboration_prompt_requests_universal_gloss_and_definition():
    from src.trace_render import _ELABORATION_SYSTEM_PROMPT

    low = _ELABORATION_SYSTEM_PROMPT.lower()
    assert "headline" in _ELABORATION_SYSTEM_PROMPT
    assert "what_might_be_happening" in _ELABORATION_SYSTEM_PROMPT
    assert "how_to_check" in _ELABORATION_SYSTEM_PROMPT
    assert "what_would_change_our_mind" in _ELABORATION_SYSTEM_PROMPT
    assert "plain_meaning" in _ELABORATION_SYSTEM_PROMPT
    assert "researcher who requested it" in low
    assert "reader profile" in low
    assert "hypscore" in low and "rankscore" in low  # explicitly banned from prose


def test_elaborator_feeds_concept_definitions_to_the_backend():
    from src.trace_render import LLMHypothesisElaborator

    class _Capturing:
        def __init__(self):
            self.seen = None

        def run(self, messages, tools=None):
            self.seen = messages
            return {"choices": [{"message": {"content": '{"plain_sentence": "x"}'}}]}

    cap = _Capturing()
    LLMHypothesisElaborator(cap).elaborate(
        _row(), concept_definitions={"LLC threshold": "a loss-landscape degeneracy measure"}
    )
    joined = " ".join(str(m.get("content", "")) for m in cap.seen)
    assert "a loss-landscape degeneracy measure" in joined  # the source for the universal gloss


def test_elaborator_feeds_public_idea_scaffold_to_the_backend():
    from src.run_path import SurfacedHypothesis
    from src.trace_render import LLMHypothesisElaborator

    class _Capturing:
        def __init__(self):
            self.seen = None

        def run(self, messages, tools=None):
            self.seen = messages
            return {"choices": [{"message": {"content": '{"plain_sentence": "x"}'}}]}

    row = SurfacedHypothesis(
        rank=1, candidate_id="h5", field_novelty=0.70, saturation=0.25, hyp_score=0.56,
        rank_score=0.42, cross_concept=True, common_sense=False,
        rationale="LLC predicts capability collapse", new_node_labels=("LLC threshold",),
        idea_scaffold={
            "claim_anchor": "compression can change model capability",
            "incumbent_limit": "raw scores hide the mechanism",
            "lever": "LLC threshold",
            "fail_safe": "no threshold shift would weaken it",
        },
    )
    cap = _Capturing()
    LLMHypothesisElaborator(cap).elaborate(row)
    joined = " ".join(str(m.get("content", "")) for m in cap.seen)
    assert "Public idea scaffold" in joined
    assert "claim_anchor: compression can change model capability" in joined
    assert "fail_safe: no threshold shift would weaken it" in joined


def test_elaborator_feeds_spec_context_to_the_backend():  # Elaborator spec context-in
    from src.run_path import SurfacedHypothesis
    from src.trace_render import LLMHypothesisElaborator

    class _Capturing:
        def __init__(self):
            self.seen = None

        def run(self, messages, tools=None):
            self.seen = messages
            return {"choices": [{"message": {"content": '{"plain_sentence": "x"}'}}]}

    row = SurfacedHypothesis(
        rank=1, candidate_id="h9", field_novelty=0.70, saturation=0.25, hyp_score=0.56,
        rank_score=0.42, cross_concept=True, common_sense=False,
        rationale="LLC predicts capability collapse", new_node_labels=("LLC threshold",),
        new_node_details=(
            {"label": "LLC threshold", "type": "mechanism",
             "definition": "a sharp change in local learning coefficient"},
        ),
        mechanism_steps=(
            {"from": "compression", "relation": "raises", "to": "LLC threshold",
             "mechanism": "fewer bits constrain representations", "verdict": "sound", "note": ""},
        ),
        term_audit=(
            {"term": "LLC threshold", "verdict": "consistent",
             "source_quote": "local learning coefficients identify phase transitions"},
        ),
        experiment_plan={
            "hypothesis_under_test": "compression --raises--> LLC threshold",
            "design": "ablation",
            "controls_and_confounders": [{"factor": "model size", "from_graph": True, "handling": "hold fixed"}],
            "materials_or_data": [{"item": "benchmark suite", "evidence_id": "ev-1"}],
            "feasibility": {"resources": "one GPU", "time": "1 week", "main_risk": "noisy threshold"},
            "grounding": [{"evidence_id": "ev-1", "quote_span": "benchmark suite"}],
        },
    )
    cap = _Capturing()
    LLMHypothesisElaborator(cap).elaborate(
        row,
        reader_profile={
            "field": "tensor factorization for LLM compression",
            "familiar_terms": ["low-rank adapter", "Tucker factorization"],
            "analogy_domains": ["matrix compression"],
            "expertise": "compression geometry",
        },
    )
    user = str(cap.seen[-1]["content"])
    assert "Reader profile:" in user
    assert "tensor factorization for LLM compression" in user
    assert "low-rank adapter" in user
    assert "LLC threshold (mechanism): a sharp change in local learning coefficient" in user
    assert "Mechanism audit (verdicts preserved)" in user
    assert "compression --raises--> LLC threshold [sound]: fewer bits constrain representations" in user
    assert "Terminology audit" in user
    assert "local learning coefficients identify phase transitions" in user
    assert "hypothesis_under_test: compression --raises--> LLC threshold" in user
    assert "controls_and_confounders:" in user
    assert "model size" in user
    assert "materials_or_data:" in user
    assert "feasibility:" in user
    assert "grounding:" in user


# --- Clickable referenced sources on the index -------------------------------------------
def _sources():
    return [
        {"title": "Matrix Compression via Low Rank", "url": "https://doi.org/10.52/x", "source": "crossref"},
        {"title": "Optimal Brain Decomposition", "url": "https://arxiv.org/pdf/2604.00821", "source": "exa"},
    ]


def test_render_index_html_links_sources():
    from src.trace_render import render_index_html

    html = render_index_html(_result([_row("h5", 1)]), sources=_sources())
    assert "<h2>Sources</h2>" in html
    assert '<a href="https://doi.org/10.52/x">Matrix Compression via Low Rank</a>' in html
    assert '<a href="https://arxiv.org/pdf/2604.00821">Optimal Brain Decomposition</a>' in html
    assert "(crossref)" in html and "(exa)" in html


def test_render_index_html_omits_sources_section_when_none():
    from src.trace_render import render_index_html

    assert "<h2>Sources</h2>" not in render_index_html(_result([_row("h5", 1)]))


def test_render_index_html_source_without_url_is_plain_text():
    from src.trace_render import render_index_html

    html = render_index_html(
        _result([_row("h5", 1)]), sources=[{"title": "No link paper", "source": "crossref"}]
    )
    after = html.split("<h2>Sources</h2>", 1)[1]
    assert "No link paper" in after and "<a href" not in after  # shown, not linkified


def test_render_hypothesis_html_links_back_to_index():
    from src.trace_render import render_hypothesis_html

    assert 'href="index.html"' in render_hypothesis_html(_row())


def test_write_trace_passes_sources_into_index(tmp_path):
    from src.trace_render import write_trace

    write_trace(_result([_row("h5", 1)]), tmp_path, sources=_sources())
    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert '<a href="https://arxiv.org/pdf/2604.00821">Optimal Brain Decomposition</a>' in index


def test_render_hypothesis_html_links_sources():
    from src.trace_render import render_hypothesis_html

    html = render_hypothesis_html(_row(), sources=_sources())
    assert "<h2>Sources</h2>" in html
    assert '<a href="https://arxiv.org/pdf/2604.00821">Optimal Brain Decomposition</a>' in html


def test_write_trace_puts_sources_on_each_hypothesis_page(tmp_path):
    from src.trace_render import write_trace

    write_trace(_result([_row("h5", 1), _row("h6", 2)]), tmp_path, sources=_sources())
    for name in ("h5.html", "h6.html"):
        page = (tmp_path / name).read_text(encoding="utf-8")
        assert '<a href="https://arxiv.org/pdf/2604.00821">Optimal Brain Decomposition</a>' in page


def test_key_terms_source_links_when_it_matches_a_retrieved_paper():
    from src.trace_render import render_hypothesis_html

    elab = {"plain_sentence": "x", "elaboration": "y", "key_terms": [
        {"term": "Spectral gap", "universal_definition": "d", "source": "Optimal Brain Decomposition"}]}
    html = render_hypothesis_html(_row(), elaboration=elab, sources=[
        {"title": "Optimal Brain Decomposition for Accurate LLM", "url": "https://arxiv.org/pdf/x",
         "source": "exa"}])
    assert '<a href="https://arxiv.org/pdf/x">Optimal Brain Decomposition</a>' in html


def test_key_terms_self_proposed_source_is_not_a_broken_citation():
    from src.trace_render import render_hypothesis_html

    elab = {"plain_sentence": "x", "elaboration": "y", "key_terms": [
        {"term": "Curvature spike", "universal_definition": "d", "source": "this paper (proposed)"}]}
    html = render_hypothesis_html(_row(), elaboration=elab, sources=[])
    assert "this paper (proposed)" in html            # attribution still shown
    assert "[this paper (proposed)]" not in html       # but NOT as a bracketed citation/link
    term_li = html.split("Curvature spike", 1)[1].split("</li>", 1)[0]
    assert "<a" not in term_li                          # and it is not a hyperlink


# --- Lineage report panel ---------------------------------------------------------------
def test_render_hypothesis_html_includes_lineage_panel():
    from src.trace_render import render_hypothesis_html

    lineage = {"wasDerivedFrom": ["h1"], "strategy": "divergent"}
    html = render_hypothesis_html(_row(), lineage=lineage)
    assert "Lineage" in html
    assert "<details" in html                              # collapsible panel
    assert "h1" in html and "divergent" in html            # the Research Synthesist derivation DAG (parent + strategy)


def test_render_hypothesis_html_omits_lineage_when_absent():  # optional lineage
    from src.trace_render import render_hypothesis_html

    html = render_hypothesis_html(_row())  # no lineage leaves the page unchanged
    assert "Lineage" not in html and "<details" not in html


def test_write_trace_threads_lineage_from_surfaced_rows(tmp_path):  # end-to-end trace report
    from src.trace_render import write_trace

    row = SurfacedHypothesis(
        rank=1, candidate_id="v1", field_novelty=0.9, saturation=0.1, hyp_score=0.5, rank_score=0.5,
        cross_concept=False, common_sense=False, rationale="r", new_node_labels=("v1 mech",),
        lineage={"wasDerivedFrom": ["h1"], "strategy": "divergent"},
    )
    write_trace(_result([row]), tmp_path)
    page = (tmp_path / "v1.html").read_text(encoding="utf-8")
    assert "Lineage" in page  # the panel appears on the WRITTEN page
    assert "h1" in page and "divergent" in page


# --- Mechanism-chain and problem sections with an audit sidecar ------------------------
def _audited_row(**overrides):
    kwargs = dict(
        rank=1, candidate_id="h9", field_novelty=0.9, saturation=0.1, hyp_score=0.5, rank_score=0.5,
        cross_concept=False, common_sense=False, rationale="r", new_node_labels=("mech",),
        idea_scaffold={"problem": "practitioners cannot bound posterior error at low bitwidths"},
        mechanism_steps=(
            {"from": "message quantizer", "relation": "reduces", "to": "posterior error",
             "mechanism": "coarser codes drop bits", "verdict": "sound", "note": "", "origin": "prose"},
            {"from": "posterior error", "relation": "increases", "to": "convergence time",
             "mechanism": "", "verdict": "vague", "note": "no stated relation", "origin": "prose"},
        ),
        scaffold_gaps=(),
    )
    kwargs.update(overrides)
    return SurfacedHypothesis(**kwargs)


def test_render_mechanism_steps_section_and_omits_when_empty():  #  render + byte-identity
    from src.trace_render import render_hypothesis_html

    page = render_hypothesis_html(_audited_row())
    assert "How the pieces connect" in page
    assert "message quantizer --reduces--&gt; posterior error" in page
    assert "[vague]" in page and "no stated relation" in page   # residual warning badge + note
    assert "[sound]" not in page                                 # sound steps render clean
    # absent field (pre-feature row) -> the section is absent and the page is byte-identical
    # to the ordinary renderer output for the same row.
    plain = render_hypothesis_html(_row())
    assert "How the pieces connect" not in plain and "What problem this solves" not in plain


def test_elaboration_prompt_renders_committed_experiment_plan():  # committed-plan context
    from src.trace_render import _elaboration_user_prompt

    plan = {
        "design": "ablation", "comparison_baseline": "uncompressed model",
        "procedure": ["compress at each bitrate", "evaluate factuality"],
        "metrics": [{"metric": "F1", "predicted_direction": "down", "evidence_id": None}],
        "expected_outcome": "factuality falls as bitrate drops",
        "falsification": "factuality flat across bitrates",
    }
    prompt = _elaboration_user_prompt(_audited_row(experiment_plan=plan), None)
    assert "render how_to_check and next_steps FROM this plan" in prompt
    assert "procedure: compress at each bitrate; evaluate factuality" in prompt
    assert "metrics: F1" in prompt
    assert "falsification: factuality flat across bitrates" in prompt
    # a row with no committed plan -> no plan block (byte-identical to the pre-experiment prompt)
    assert "experiment plan" not in _elaboration_user_prompt(_audited_row(), None)


def test_elaboration_prompt_experiment_plan_field_order_is_unchanged():  # stable prompt order
    from src.trace_render import _elaboration_user_prompt

    # The Elaborator prompt block predates experiment_plan_render.py and keeps its own field order,
    # independent of the shared EXPERIMENT_PLAN_FIELDS display list, so this oracle is pinned
    # here rather than derived from that list (a derived oracle would be tautological).
    fields = (
        "hypothesis_under_test", "design", "operationalization",
        "intervention_or_manipulation", "comparison_baseline",
        "controls_and_confounders", "materials_or_data", "metrics",
        "procedure", "expected_outcome", "falsification", "feasibility",
        "grounding",
    )
    plan = {key: f"v_{key}" for key in fields}
    plan["procedure"] = ["v_procedure"]  # the block joins list steps, so wrap the sentinel value
    plan["design_rationale"] = "v_design_rationale"  # populated on real Experiment Designer plans; must stay out
    prompt = _elaboration_user_prompt(_audited_row(experiment_plan=plan), None)
    positions = [prompt.index(f"{key}: v_{key}") for key in fields]
    assert positions == sorted(positions)
    assert "design_rationale" not in prompt


def test_render_problem_section_only_when_finalization_ran():  # finalization-dependent rendering
    from src.trace_render import render_hypothesis_html

    page = render_hypothesis_html(_audited_row())
    assert "What problem this solves" in page
    assert "practitioners cannot bound posterior error" in page
    # residual missing-problem gap -> explicit warning, not silence.
    gap = render_hypothesis_html(_audited_row(idea_scaffold={}, scaffold_gaps=("problem:missing",)))
    assert "No explicit problem statement" in gap
    # finalization did NOT run (no steps, no gaps) -> no problem section even with a scaffold.
    off = render_hypothesis_html(_audited_row(mechanism_steps=(), scaffold_gaps=()))
    assert "What problem this solves" not in off


def test_residual_misuse_renders_per_term_warning():  #  render + byte-identity
    import dataclasses

    from src.trace_render import render_hypothesis_html

    term_rows = (
        {"term": "posterior error", "verdict": "misused",
         "usage": "posterior error as runtime overhead",
         "source_quote": "posterior error measures deviation from the true marginal",
         "source_title": "Optimal Brain Decomposition", "evidence_id": "ev-1",
         "note": "contradicts the source sense", "origin": "mined_concept"},
        {"term": "GBP", "verdict": "consistent", "usage": "GBP as inference",
         "source_quote": "GBP performs approximate inference", "source_title": "Paper B",
         "evidence_id": "ev-2", "note": "", "origin": "prose"},
    )
    sources = [{"title": "Optimal Brain Decomposition", "url": "https://arxiv.org/pdf/x",
                "source": "exa"}]
    page = render_hypothesis_html(_audited_row(term_audit=term_rows), sources=sources)
    assert "Term check" in page
    warn = page.split("Term check", 1)[1].split("<table>", 1)[0]
    assert "posterior error" in warn and "misused" in warn
    assert "measures deviation from the true marginal" in warn          # the source quote
    assert '<a href="https://arxiv.org/pdf/x">Optimal Brain Decomposition</a>' in warn  # linkified
    assert "posterior error as runtime overhead" in warn                # this hypothesis's usage
    assert "GBP performs approximate inference" not in warn             # consistent -> NO warning
    # consistent-only -> no Term check block; the page is byte-identical to empty term_audit.
    consistent_only = _audited_row(term_audit=(term_rows[1],))
    page_consistent = render_hypothesis_html(consistent_only, sources=sources)
    assert "Term check" not in page_consistent
    empty = dataclasses.replace(consistent_only, term_audit=())
    assert page_consistent == render_hypothesis_html(empty, sources=sources)


def test_write_trace_sidecar_carries_audits_key(tmp_path):  #  sidecar (additive)
    import json as _json

    from src.trace_render import write_trace

    write_trace(_result([_audited_row()]), tmp_path)
    sidecar = _json.loads((tmp_path / "plain_language.json").read_text(encoding="utf-8"))
    audits = sidecar["audits"]["h9"]
    assert audits["mechanism_steps"][1]["verdict"] == "vague"
    assert "scaffold_gaps" not in audits            # empty entries are omitted, not nulled
    # rows without audit fields -> NO audits key at all (old sidecar schema byte-identical).
    write_trace(_result([_row()]), tmp_path / "plain")
    plain_sidecar = _json.loads((tmp_path / "plain" / "plain_language.json").read_text(encoding="utf-8"))
    assert "audits" not in plain_sidecar


def test_write_trace_sidecar_carries_hypothesis_edge_ids_key(tmp_path):  # exact edge-ID binding
    import dataclasses
    import json as _json

    from src.trace_render import write_trace

    row = dataclasses.replace(_row(), hypothesis_edge_ids=("e-h1",))
    write_trace(_result([row]), tmp_path)
    sidecar = _json.loads((tmp_path / "plain_language.json").read_text(encoding="utf-8"))
    assert sidecar["hypothesis_edge_ids"]["h5"] == ["e-h1"]
    # rows without edge ids -> NO hypothesis_edge_ids key at all (old sidecar schema byte-identical).
    write_trace(_result([_row()]), tmp_path / "plain")
    plain_sidecar = _json.loads((tmp_path / "plain" / "plain_language.json").read_text(encoding="utf-8"))
    assert "hypothesis_edge_ids" not in plain_sidecar


def test_write_trace_sidecar_carries_experiment_attempts_key(tmp_path):
    import dataclasses
    import json as _json

    from src.trace_render import write_trace

    attempt = {
        "candidate_id": "h5",
        "primary_edge_id": "e-h1",
        "status": "not_committed",
        "reason": "designer response was malformed",
    }
    row = dataclasses.replace(_row(), experiment_attempt=attempt)
    write_trace(_result([row]), tmp_path)
    sidecar = _json.loads((tmp_path / "plain_language.json").read_text(encoding="utf-8"))

    assert sidecar["experiment_attempts"]["h5"] == attempt
    write_trace(_result([_row()]), tmp_path / "plain")
    plain_sidecar = _json.loads(
        (tmp_path / "plain" / "plain_language.json").read_text(encoding="utf-8")
    )
    assert "experiment_attempts" not in plain_sidecar


def test_write_trace_sidecar_carries_seed_kind_key(tmp_path):  # seed identity
    import json as _json

    from src.trace_render import write_trace

    write_trace(
        _result([_row()]), tmp_path, claim="Make GBP cheaper.", seed_kind="research_goal",
    )
    sidecar = _json.loads((tmp_path / "plain_language.json").read_text(encoding="utf-8"))
    assert sidecar["claim"] == "Make GBP cheaper."
    assert sidecar["seed_kind"] == "research_goal"
    # no seed_kind -> NO seed_kind key at all (old sidecar schema byte-identical).
    write_trace(_result([_row()]), tmp_path / "plain")
    plain_sidecar = _json.loads((tmp_path / "plain" / "plain_language.json").read_text(encoding="utf-8"))
    assert "seed_kind" not in plain_sidecar


# --- Audit-aware elaboration of problem and next-step card fields ----------------------
def test_elaboration_prompt_requests_next_steps_and_problem_solved():
    from src.trace_render import _ELABORATION_SYSTEM_PROMPT

    low = _ELABORATION_SYSTEM_PROMPT.lower()
    assert "problem_solved" in _ELABORATION_SYSTEM_PROMPT
    assert "next_steps" in _ELABORATION_SYSTEM_PROMPT
    # the doing-order rubric: setup -> data -> baseline/comparison -> measurement -> outcome.
    for word in ("setup", "baseline/comparison", "measurement", "expected outcome", "doing order"):
        assert word in low
    # The writer must follow the audited mechanism and terminology structure when provided.
    assert "panel-audited mechanism chain" in low
    assert "verified terminology" in low
    # the existing prohibition stays.
    assert "Do NOT invent numbers" in _ELABORATION_SYSTEM_PROMPT


def test_card_renders_next_steps_as_ordered_list():
    from src.trace_render import render_hypothesis_html

    steps = [
        "Set up the two message quantizers side by side.",
        "Collect the same factor-graph inputs for both.",
        "Run the baseline quantizer first.",
        "Measure posterior error for each run.",
        "Expect the coarse codes to show larger error.",
    ]
    elab = {"headline": "Coarser codes may raise posterior error.", "next_steps": steps}
    page = render_hypothesis_html(_row(), elaboration=elab)
    assert "What to do next" in page
    ol = page.split("What to do next", 1)[1].split("<ol>", 1)[1].split("</ol>", 1)[0]
    positions = [ol.index(step) for step in steps]
    assert positions == sorted(positions)           # rendered in doing order
    assert ol.count("<li>") == 5
    # defensive truncation: more than 6 items -> only the first 6 render.
    elab8 = {"headline": "H.", "next_steps": [f"Step {i}." for i in range(1, 9)]}
    page8 = render_hypothesis_html(_row(), elaboration=elab8)
    ol8 = page8.split("What to do next", 1)[1].split("<ol>", 1)[1].split("</ol>", 1)[0]
    assert ol8.count("<li>") == 6
    assert "Step 6." in ol8 and "Step 7." not in ol8


def test_card_renders_problem_solved_section():
    from src.trace_render import render_hypothesis_html

    elab = {"headline": "H may hold.",
            "problem_solved": "Practitioners cannot bound quantization error today."}
    page = render_hypothesis_html(_row(), elaboration=elab)
    assert ("<p><b>What problem this solves:</b> "
            "Practitioners cannot bound quantization error today.</p>") in page


def test_card_without_new_keys_renders_unchanged():
    # Legacy card (no problem_solved / next_steps) -> BYTE-identical section HTML to the pre-Research Synthesist
    # renderer (the pinned string was captured from the pre-change implementation).
    from src.trace_render import _plain_language_card_html

    legacy = {
        "headline": "LLC thresholds may reveal when model skills become fragile.",
        "what_might_be_happening": "The model may rely on a small set of internal signals.",
        "status": "This is a testable idea, not a proven conclusion.",
        "key_terms": [],
    }
    assert _plain_language_card_html(legacy) == (
        "<section><h2>Possible explanation</h2>"
        "<p><b>LLC thresholds may reveal when model skills become fragile.</b></p>"
        "<p><b>What might be happening:</b> The model may rely on a small set of internal signals.</p>"
        "<p><b>Status:</b> This is a testable idea, not a proven conclusion.</p>"
        "</section>"
    )


def test_elaborator_feeds_mechanism_steps_and_verified_terms():
    from src.trace_render import LLMHypothesisElaborator

    class _Capturing:
        def __init__(self):
            self.seen = None

        def run(self, messages, tools=None):
            self.seen = messages
            return {"choices": [{"message": {"content": '{"plain_sentence": "x"}'}}]}

    audited = _audited_row(
        mechanism_steps=(
            {
                "from": "message quantizer",
                "relation": "--reduces-->",
                "to": "posterior error",
                "mechanism": "coarser codes drop bits",
                "verdict": "sound",
                "note": "",
                "origin": "prose",
            },
        ),
        term_audit=(
            {"term": "posterior error", "verdict": "consistent",
             "usage": "posterior error as deviation",
             "source_quote": "posterior error measures deviation from the true marginal",
             "source_title": "Paper A", "evidence_id": "ev-1", "note": "", "origin": "mined_concept"},
        ),
    )
    cap = _Capturing()
    LLMHypothesisElaborator(cap).elaborate(audited)
    user = str(cap.seen[-1]["content"])
    assert "Mechanism audit (verdicts preserved)" in user
    assert "message quantizer --reduces--> posterior error [sound]: coarser codes drop bits" in user
    assert "Terminology audit" in user
    assert "posterior error — consistent" in user
    assert "posterior error measures deviation from the true marginal" in user
    # absent attributes (old runs, flags OFF) -> the prompt is BYTE-identical to today's
    # (the pinned string was captured from the pre-change implementation).
    cap2 = _Capturing()
    LLMHypothesisElaborator(cap2).elaborate(_row())
    assert cap2.seen[-1]["content"] == (
        "Hypothesis id: h5\n"
        "Proposed new concept(s): LLC threshold\n"
        "Proposer rationale: LLC predicts capability collapse\n"
        "\n"
        "Write the plain-language reader card as STRICT JSON. Do not mention the numeric scores."
    )


def test_write_trace_sidecar_round_trips_next_steps(tmp_path):
    import json as _json

    from src.trace_render import write_trace

    steps = ["Set up A.", "Collect B.", "Compare with C.", "Measure D."]
    elaborations = {"h5": {"headline": "H may hold.", "problem_solved": "P is unsolved.",
                           "next_steps": steps, "key_terms": []}}
    write_trace(_result([_row("h5", 1)]), tmp_path, elaborations=elaborations)
    sidecar = _json.loads((tmp_path / "plain_language.json").read_text(encoding="utf-8"))
    assert sidecar["hypotheses"]["h5"]["next_steps"] == steps
    assert sidecar["hypotheses"]["h5"]["problem_solved"] == "P is unsolved."
    page = (tmp_path / "h5.html").read_text(encoding="utf-8")
    assert "What to do next" in page and "Compare with C." in page  # rendered on the WRITTEN page
    assert "What problem this solves" in page


# --- Reader-translation render layer: links, familiar glosses, panel, and sidecar -------
def _translation_records():
    return [
        {"original": "posterior error", "familiar": "reconstruction error",
         "first_use_rendering": "reconstruction error (posterior error)",
         "faithfulness": "confirmed", "rationale": "same L2 deviation object"},
        {"original": "message damping", "familiar": "step-size smoothing",
         "first_use_rendering": "step-size smoothing (message damping)",
         "faithfulness": "approximate", "rationale": "loses the per-edge schedule"},
        {"original": "loopy propagation", "familiar": "cyclic sweep",
         "first_use_rendering": "", "faithfulness": "rejected", "rationale": "different object"},
    ]


def test_elaboration_prompt_requires_self_contained_definitions():
    from src.trace_render import _ELABORATION_SYSTEM_PROMPT

    assert "SELF-CONTAINED" in _ELABORATION_SYSTEM_PROMPT
    assert "no undefined jargon inside a definition" in _ELABORATION_SYSTEM_PROMPT
    assert "unpack that term inline in plain words" in _ELABORATION_SYSTEM_PROMPT
    assert "key_terms plain_meaning" in _ELABORATION_SYSTEM_PROMPT


def test_key_terms_without_link_or_gloss_render_byte_identical():  # byte-identity pin
    # The pinned string captures legacy behavior: entries without the new
    # ``link`` / ``link_kind`` / ``familiar_gloss`` fields must render bit-for-bit today's HTML.
    from src.trace_render import _key_terms_html

    terms = [{"term": "LLC", "plain_meaning": "a model signal", "source": "Paper A"}]
    assert _key_terms_html(terms) == (
        '<ul><li><b>LLC</b> — a model signal <span class="k">(Paper A)</span></li></ul>'
    )


def test_key_terms_render_link_and_familiar_gloss():
    from src.trace_render import _key_terms_html, render_hypothesis_html

    terms = [{
        "term": "factor graph", "plain_meaning": "a map of variables", "source": "Paper A",
        "link": "https://en.wikipedia.org/wiki/Factor_graph", "link_kind": "wikipedia",
        "familiar_gloss": "like a tensor network diagram",
    }]
    html = _key_terms_html(terms)
    assert ('<b><a href="https://en.wikipedia.org/wiki/Factor_graph" title="wikipedia">'
            "factor graph</a></b>") in html
    assert '<span class="k"> — in your terms: like a tensor network diagram</span>' in html
    # end-to-end through the page renderer
    page = render_hypothesis_html(_row(), elaboration={"headline": "H.", "key_terms": terms})
    assert 'title="wikipedia"' in page and "in your terms: like a tensor network diagram" in page


def test_translations_panel_marks_and_hidden_when_absent():
    from src.trace_render import _plain_language_card_html, _translations_html

    html = _translations_html(_translation_records(), "tensor factorization")
    assert "<details><summary>Translated for a tensor factorization reader</summary>" in html
    assert "[confirmed]" in html and "[≈ approximate]" in html
    assert "same L2 deviation object" in html and 'class="k"' in html   # rationale in class=k
    assert "kept original — proposed analogy failed the faithfulness check" in html
    assert "cyclic sweep" not in html      # the rejected analogy phrasing never ships
    # hidden when absent/empty; generic summary when the home field is unknown
    assert _translations_html([], "x") == ""
    assert _translations_html(None, "x") == ""
    assert "Translated for this reader" in _translations_html(_translation_records()[:1], "")
    # the card appends the panel AFTER key_terms; without translations the card is unchanged
    card = {"headline": "H may hold.", "key_terms": [{"term": "T", "plain_meaning": "d"}],
            "translations": _translation_records()[:1]}
    with_panel = _plain_language_card_html(card, home_field="tensor factorization")
    assert with_panel.index("</ul>") < with_panel.index("<details>")
    no_panel = {"headline": "H may hold.", "key_terms": [{"term": "T", "plain_meaning": "d"}]}
    assert "<details>" not in _plain_language_card_html(no_panel)


def test_write_trace_sidecar_reader_block_optional(tmp_path):
    import json as _json

    from src.trace_render import write_trace

    elabs = {"h5": {"headline": "H may hold.", "key_terms": [],
                    "translations": _translation_records()[:1]}}
    reader = {"home_field": "tensor factorization", "lexicon_source": "local paper arXiv-X"}
    write_trace(_result([_row("h5", 1)]), tmp_path, elaborations=elabs, reader_context=reader)
    sidecar = _json.loads((tmp_path / "plain_language.json").read_text(encoding="utf-8"))
    assert sidecar["reader"] == reader
    assert sidecar["hypotheses"]["h5"]["translations"] == _translation_records()[:1]
    page = (tmp_path / "h5.html").read_text(encoding="utf-8")
    assert "Translated for a tensor factorization reader" in page   # summary names the home field
    # With no reader context, the sidecar is byte-identical to the legacy payload
    # (the pinned string was captured from the pre-change implementation).
    write_trace(_result([_row("h5", 1)]), tmp_path / "plain",
                elaborations={"h5": {"headline": "H.", "key_terms": []}}, reader_context=None)
    raw = (tmp_path / "plain" / "plain_language.json").read_text(encoding="utf-8")
    assert raw == (
        '{\n  "claim": "",\n  "hypotheses": {\n    "h5": {\n      "headline": "H.",\n'
        '      "key_terms": []\n    }\n  }\n}'
    )
