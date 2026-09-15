"""Tests for the connected renderer's ranking, lineage, and audit panels."""

from __future__ import annotations

from pathlib import Path

from src import connected_render


def test_missing_reader_card_does_not_promote_technical_experiment_text():
    graph = {
        "version": 1,
        "nodes": {"n1": {"label": "calibration", "provenance": []}},
        "edges": {"e1": {"source_node_ids": ["n1"], "target_node_ids": []}},
        "experiment_plans": {"e1": {
            "hypothesis_under_test": "calibration --negative--> functional drift",
            "design": "ablation",
        }},
    }
    row = {"cid": "h1", "rank": 1, "cell": "calibration", "scores": {}}
    for plain in (None, {}, {"headline": ""}):
        page, _ = connected_render.render_page(
            graph, "run", row, ["n1"], [], {}, plain=plain,
            claim="Original research question", seed_kind="claim",
        )
        hero = page.split('<h2>Proposed hypothesis</h2>', 1)[1].split('</section>', 1)[0]
        assert "Plain-language summary unavailable" in hero
        assert "functional drift" not in hero
        assert "calibration --negative--&gt; functional drift" in page
        assert "Original research question" in page


def test_legacy_audit_relations_render_with_one_arrow_in_both_reports():
    from src.trace_render import _mechanism_steps_html

    steps = [{"from": "calibration", "relation": "--negative-->", "to": "drift"},
             {"from": "cost", "relation": "dose-dependent", "to": "accuracy"}]
    for render in (connected_render._mechanism_steps_section, _mechanism_steps_html):
        page = render(steps)
        assert "calibration --negative--&gt; drift" in page
        assert "cost --dose-dependent--&gt; accuracy" in page
        assert "----" not in page
    # Rendering old artifacts must not modify their stored audit records.
    assert steps[0]["relation"] == "--negative-->"


def test_scores_section_includes_why_surfaced_and_lineage_panels():
    why = {"deterministic_rank": 7, "tournament_rank": 1, "elo": 1248.0, "record": "3-0-0",
           "rationale": "more mechanism-specific than its rivals"}
    lineage = {"wasDerivedFrom": ["h1"], "strategy": "inspire", "round": 1, "grounding": "grounded"}
    html = connected_render.scores_section(
        {"field novelty": "0.70"}, {}, why_surfaced=why, lineage=lineage
    )
    assert "Why surfaced" in html and "Lineage" in html
    assert "<details" in html
    assert "1248" in html and "3-0-0" in html
    assert "h1" in html and "inspire" in html


def test_scores_section_omits_panels_when_no_tournament_or_lineage():  # optional panels absent
    html = connected_render.scores_section({"field novelty": "0.70"}, {})
    assert "Why surfaced" not in html and "Lineage" not in html


# --- Audit warnings from the plain-language sidecar ------------------------------------
def test_connected_view_renders_audit_warnings_from_sidecar(tmp_path, monkeypatch):
    import json

    run_dir = tmp_path / "run-x"
    (run_dir / "trace").mkdir(parents=True)
    graph = {
        "version": 3,
        "nodes": {"n-1": {"label": "codebook collapse", "type": "mechanism", "provenance": []}},
        "edges": {},
    }
    (run_dir / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    (run_dir / "trace" / "h1.html").write_text(
        "<table>"
        "<tr><td class=k>rank</td><td>1</td></tr>"
        "<tr><td class=k>candidate</td><td>h1</td></tr>"
        "<tr><td class=k>new concept(s)</td><td>codebook collapse</td></tr>"
        "</table>",
        encoding="utf-8",
    )
    sidecar = {
        "claim": "",
        "hypotheses": {"h1": {"headline": "Collapse may matter.",
                              "what_might_be_happening": "Codes may merge.", "key_terms": []}},
        "audits": {"h1": {
            "mechanism_steps": [{"from": "quantizer", "relation": "causes",
                                 "to": "codebook collapse", "mechanism": "",
                                 "verdict": "vague", "note": "no stated relation",
                                 "origin": "prose"}],
            "term_audit": [{"term": "codebook collapse", "verdict": "misused",
                            "usage": "collapse as a speed-up",
                            "source_quote": "collapse merges distinct codewords",
                            "source_title": "Paper A", "evidence_id": "ev-1",
                            "note": "contradicts source", "origin": "mined_concept"}],
        }},
    }
    (run_dir / "trace" / "plain_language.json").write_text(json.dumps(sidecar), encoding="utf-8")
    monkeypatch.chdir(tmp_path)  # keep the *.sqlite glob away from the repo cwd
    assert connected_render.render_connected_hypotheses(run_dir)
    page = (run_dir / "h1-connected.html").read_text(encoding="utf-8")
    assert "Term check" in page
    assert "collapse merges distinct codewords" in page and "misused" in page
    assert "collapse as a speed-up" in page and "Paper A" in page
    assert "How the pieces connect" in page and "no stated relation" in page


def test_connected_view_unchanged_without_audits_key():
    plain = {"headline": "Collapse may matter.", "what_might_be_happening": "Codes may merge.",
             "key_terms": []}
    # hypothesis_prose: audits omitted/None -> byte-identical output, no audit sections.
    assert (connected_render.hypothesis_prose(plain, audits=None)
            == connected_render.hypothesis_prose(plain))
    assert "Term check" not in connected_render.hypothesis_prose(plain)
    assert "How the pieces connect" not in connected_render.hypothesis_prose(plain)
    # render_page: the audits kwarg defaults off -> byte-identical page (dropped-candidates path).
    graph = {"version": 1, "nodes": {"n-1": {"label": "x", "provenance": []}}, "edges": {}}
    row = {"cid": "h1", "rank": 1, "cell": "x", "scores": {"rank": "1"}}
    base, _ = connected_render.render_page(
        graph, "run", row, ["n-1"], [], {}, claim="", plain=plain,
    )
    again, _ = connected_render.render_page(
        graph, "run", row, ["n-1"], [], {}, claim="", plain=plain, audits=None,
    )
    assert base == again


# --- Research Synthesist problem and next-step fields in the connected view ------------
def test_hypothesis_prose_renders_next_steps_and_problem():
    steps = ["Set up A.", "Collect B.", "Run the baseline C.", "Measure D.", "Expect E."]
    plain = {"headline": "H may hold.", "problem_solved": "P is unsolved.",
             "next_steps": steps, "key_terms": []}
    html = connected_render.hypothesis_prose(plain)
    assert "<p><b>What problem this solves:</b> P is unsolved.</p>" in html
    assert "What to do next" in html
    ol = html.split("What to do next", 1)[1].split("<ol>", 1)[1].split("</ol>", 1)[0]
    positions = [ol.index(step) for step in steps]
    assert positions == sorted(positions)           # rendered in doing order
    assert ol.count("<li>") == 5


def test_hypothesis_prose_card_without_problem_or_next_steps_is_unchanged():
    # A card without the optional problem and next-step fields retains the compact rendering.
    plain = {
        "headline": "H may hold.",
        "what_might_be_happening": "W.",
        "why_it_matters": "Y.",
        "how_to_check": "C.",
        "what_would_change_our_mind": "M.",
        "status": "S.",
        "key_terms": [{"term": "T", "plain_meaning": "d", "source": "Proposed concept"}],
    }
    assert connected_render.hypothesis_prose(plain) == (
        '<section class="plain"><h2>Possible explanation</h2>'
        "<p><b>H may hold.</b></p>"
        "<p><b>What might be happening:</b> W.</p>"
        "<p><b>Why it matters:</b> Y.</p>"
        "<p><b>How to check:</b> C.</p>"
        "<p><b>What would change our mind:</b> M.</p>"
        "<p><b>Status:</b> S.</p>"
        '<ul><li><b>T</b> — d <span class="sub">(Proposed concept)</span></li></ul>'
        "</section>"
    )


def test_connected_view_renders_committed_experiment_plan():
    graph = {
        "version": 1,
        "nodes": {
            "n-src": {"label": "walk-summable convergence", "type": "condition", "provenance": []},
            "n-focus": {"label": "safe fixed-point bitwidth", "type": "mechanism", "provenance": []},
        },
        "edges": {
            "e-plan": {
                "source_node_ids": ["n-src"],
                "target_node_ids": ["n-focus"],
                "relation_type": "sets threshold",
                "status": "unverified",
            }
        },
        "experiment_plans": {
            "e-plan": {
                "design": "simulation",
                "design_rationale": "MARKER-DESIGN-RATIONALE.",
                "hypothesis_under_test": "Exact-arithmetic margins predict safe fixed-point bits.",
                "intervention_or_manipulation": (
                    "Sweep fixed-point bitwidth while holding the graph fixed."
                ),
                "operationalization": "MARKER-OPERATIONALIZATION.",
                "materials_or_data": [{"item": "MARKER-MATERIAL-ITEM"}],
                "comparison_baseline": "Exact-arithmetic GBP on the same graph.",
                "metrics": [{"metric": "Posterior-mean RMSE", "predicted_direction": "below tolerance"}],
                "procedure": ["Generate graph seeds.", "Run the fixed-point sweep."],
                "expected_outcome": "Predicted thresholds preserve posterior-mean accuracy.",
                "falsification": "Certified bitwidths fail at the stated error tolerance.",
                "controls_and_confounders": [
                    {"factor": "MARKER-CONTROL-FACTOR", "handling": "hold fixed"}
                ],
                "feasibility": {"resources": "MARKER-FEASIBILITY-RESOURCES"},
                "grounding": [{"evidence_id": "ev-1", "quote_span": "MARKER-GROUNDING-QUOTE"}],
            }
        },
    }
    row = {"cid": "h1", "rank": 1, "cell": "safe fixed-point bitwidth", "scores": {"rank": "1"}}
    page, _ = connected_render.render_page(
        graph, "run", row, ["n-focus"], [], {}, claim="", plain=None,
    )

    assert "Experiment plan" in page
    # Every committed plan field is rendered from ``EXPERIMENT_PLAN_FIELDS``.
    # renders both its heading and its content, not just a hand-picked subset.
    for label, marker in [
        ("Design", "simulation"),
        ("Design rationale", "MARKER-DESIGN-RATIONALE."),
        ("Hypothesis under test", "Exact-arithmetic margins predict safe fixed-point bits."),
        ("Intervention / manipulation", "Sweep fixed-point bitwidth while holding the graph fixed."),
        ("Operationalization", "MARKER-OPERATIONALIZATION."),
        ("Materials / data", "MARKER-MATERIAL-ITEM"),
        ("Comparison baseline", "Exact-arithmetic GBP on the same graph."),
        ("Metrics", "Posterior-mean RMSE"),
        ("Procedure", "Run the fixed-point sweep."),
        ("Expected outcome", "Predicted thresholds preserve posterior-mean accuracy."),
        ("Falsification", "Certified bitwidths fail at the stated error tolerance."),
        ("Controls and confounders", "MARKER-CONTROL-FACTOR"),
        ("Feasibility", "MARKER-FEASIBILITY-RESOURCES"),
        ("Grounding", "MARKER-GROUNDING-QUOTE"),
    ]:
        assert f"<h4>{label}</h4>" in page, f"missing field heading: {label}"
        assert marker in page, f"missing field content: {label}"


def test_experiment_plan_section_traces_sources_and_summarizes_core_idea():
    graph = {
        "version": 1,
        "nodes": {
            "n-src": {"label": "walk-summable convergence", "type": "condition", "provenance": []},
            "n-focus": {"label": "safe fixed-point bitwidth", "type": "mechanism", "provenance": []},
        },
        "edges": {
            "e-plan": {
                "source_node_ids": ["n-src"],
                "target_node_ids": ["n-focus"],
                "relation_type": "sets threshold",
                "status": "unverified",
            }
        },
        "experiment_plans": {
            "e-plan": {
                "design": "controlled_observational",
                "design_rationale": "rho is intrinsic, so matched strata isolate the threshold rule.",
                "hypothesis_under_test": "Walk-sum margins predict the minimum safe fixed-point bitwidth.",
                "intervention_or_manipulation": "Sweep fixed-point bitwidth around the predicted threshold.",
                "comparison_baseline": "FP32 GBP on the same graph.",
                "operationalization": "Measure posterior RMSE and bits transmitted per message.",
                "expected_outcome": "Predicted thresholds preserve posterior-mean accuracy.",
                "falsification": "Certified bitwidths fail at the stated error tolerance.",
                "metrics": [
                    {"metric": "Posterior-mean RMSE", "predicted_direction": "below tolerance",
                     "evidence_id": "ev-1"},
                    {"metric": "message bits", "predicted_direction": "down"},
                ],
                "materials_or_data": [{"item": "Synthetic Gaussian graphs", "evidence_id": "ev-1"}],
                "grounding": [{"evidence_id": "ev-1", "quote_span": "GBP source quote"}],
            }
        },
    }
    pool = [{
        "evidence_id": "ev-1",
        "title": "Quantized GBP Benchmarks",
        "url": "https://example.test/qgbp",
        "source": "exa",
        "quote": "A benchmark reports posterior-mean RMSE for quantized GBP.",
    }]

    html = connected_render.experiment_plan_section(graph, ["n-focus"], pool=pool)

    assert "<h4>Core idea</h4>" in html
    assert "Compare similar cases while holding key factors fixed or stratified" in html
    assert "<h4>Traceback</h4>" in html
    assert "ev-1" in html
    assert "Quantized GBP Benchmarks" in html
    assert "https://example.test/qgbp" in html
    assert "posterior-mean RMSE for quantized GBP" in html
    assert "1 resolved unique source; 3 citation uses" in html
    assert html.count("Quantized GBP Benchmarks") == 1


def test_experiment_traceback_resolves_duplicate_ids_within_one_matching_legacy_batch():
    plan = {
        "grounding": [
            {"evidence_id": "ev_000010", "quote_span": "magnitude based pruning"},
            {"evidence_id": "ev_000012", "quote_span": "acceleration on real hardware"},
        ],
        "materials_or_data": [{"item": "ImageNet", "evidence_id": "ev_000010"}],
        "metrics": [{"metric": "latency", "evidence_id": "ev_000012"}],
    }
    batches = [
        {
            "batch_id": "initial-literature",
            "evidence": [
                {"evidence_id": "ev_000010", "title": "Wrong LLM Survey",
                 "url": "https://example.test/wrong-llm", "quote": "language model abstract"},
                {"evidence_id": "ev_000012", "title": "Wrong Diffusion Survey",
                 "url": "https://example.test/wrong-diffusion", "quote": "diffusion abstract"},
            ],
        },
        {
            "batch_id": "h5-methods",
            "evidence": [
                {"evidence_id": "ev_000010", "title": "Methods for Pruning Deep Neural Networks",
                 "url": "https://example.test/pruning", "source": "openalex",
                 "quote": "The survey compares methods that use magnitude based pruning."},
                {"evidence_id": "ev_000012", "title": "Sparsity in Deep Learning",
                 "url": "https://example.test/sparsity", "source": "openalex",
                 "quote": "We show techniques for achieving\\nacceleration on real hardware."},
            ],
        },
    ]

    rendered = connected_render._experiment_traceback_html(plan, None, batches=batches)

    assert "2 resolved unique sources; 4 citation uses" in rendered
    assert rendered.count("Methods for Pruning Deep Neural Networks") == 1
    assert rendered.count("Sparsity in Deep Learning") == 1
    assert "Wrong LLM Survey" not in rendered
    assert "Wrong Diffusion Survey" not in rendered


def test_experiment_traceback_does_not_guess_when_legacy_batch_match_is_ambiguous():
    plan = {
        "grounding": [{"evidence_id": "ev_000001", "quote_span": "shared exact quote"}],
        "metrics": [{"metric": "latency", "evidence_id": "ev_000001"}],
    }
    batches = [
        {"batch_id": "batch-a", "evidence": [
            {"evidence_id": "ev_000001", "title": "Paper A",
             "url": "https://example.test/a", "quote": "A shared exact quote appears here."}
        ]},
        {"batch_id": "batch-b", "evidence": [
            {"evidence_id": "ev_000001", "title": "Paper B",
             "url": "https://example.test/b", "quote": "The shared exact quote also appears here."}
        ]},
    ]

    rendered = connected_render._experiment_traceback_html(plan, None, batches=batches)

    assert "0 resolved unique sources; 2 citation uses" in rendered
    assert "ambiguous legacy retrieval batch" in rendered
    assert "Paper A" not in rendered and "Paper B" not in rendered
    assert "https://example.test/a" not in rendered
    assert "https://example.test/b" not in rendered


def test_experiment_traceback_prefers_explicit_retrieval_batch_id():
    plan = {
        "retrieval_batch_id": "batch-b",
        "grounding": [{"evidence_id": "ev_000001", "quote_span": "quote"}],
    }
    batches = [
        {"batch_id": "batch-a", "evidence": [
            {"evidence_id": "ev_000001", "title": "Paper A", "quote": "quote"}
        ]},
        {"batch_id": "batch-b", "evidence": [
            {"evidence_id": "ev_000001", "title": "Paper B", "quote": "quote"}
        ]},
    ]

    rendered = connected_render._experiment_traceback_html(plan, None, batches=batches)

    assert "Paper B" in rendered
    assert "Paper A" not in rendered


def test_experiment_plan_traceback_states_no_source_when_plan_cites_no_evidence():
    graph = {
        "version": 1,
        "nodes": {
            "n-src": {"label": "walk-summable convergence", "type": "condition", "provenance": []},
            "n-focus": {"label": "safe fixed-point bitwidth", "type": "mechanism", "provenance": []},
        },
        "edges": {
            "e-plan": {
                "source_node_ids": ["n-src"],
                "target_node_ids": ["n-focus"],
                "relation_type": "sets threshold",
                "status": "unverified",
            }
        },
        "experiment_plans": {
            "e-plan": {
                "design": "simulation",
                "hypothesis_under_test": "Exact-arithmetic margins predict safe fixed-point bits.",
            }
        },
    }

    html = connected_render.experiment_plan_section(graph, ["n-focus"], pool=None)

    assert (
        "No retrieved source is cited by this committed plan; it is therefore traceable to "
        "the Experiment Designer/Experiment Validator design step, but not to a specific retrieved methods passage."
    ) in html


def test_experiment_plan_labels_no_relevant_methods_as_ungrounded_draft():
    graph = {
        "version": 1,
        "nodes": {"n-focus": {"label": "mechanism", "type": "mechanism"}},
        "edges": {
            "e-plan": {
                "source_node_ids": ["n-focus"],
                "target_node_ids": ["n-focus"],
                "relation_type": "tests",
            }
        },
        "experiment_plans": {
            "e-plan": {
                "hypothesis_under_test": "mechanism --tests--> outcome",
                "design": "ablation",
                "grounding": [],
                "grounding_status": "no_relevant_methods",
            }
        },
    }

    rendered = connected_render.experiment_plan_section(
        graph, ["n-focus"], edge_ids=("e-plan",)
    )

    assert "Ungrounded draft" in rendered
    assert "no relevant, citable methods passage" in rendered
    assert "Designer found no relevant passage" in rendered


def test_experiment_attempt_reason_renders_when_no_plan_committed():
    rendered = connected_render.experiment_plan_section(
        {"nodes": {}, "edges": {}, "experiment_plans": {}},
        [],
        edge_ids=("e-missing",),
        attempt={
            "status": "not_committed",
            "primary_edge_id": "e-missing",
            "reason": "designer response was malformed",
        },
    )

    assert "No committed plan" in rendered
    assert "e-missing" in rendered
    assert "designer response was malformed" in rendered


def test_plan_evidence_refs_strips_placeholder_ids():  # placeholder filtering
    plan = {
        "grounding": [
            {"evidence_id": "none_retrieved", "quote_span": "no source"},
            {"evidence_id": "ev-1", "quote_span": "real source"},
        ],
    }
    assert connected_render._plan_evidence_refs(plan) == [
        ("ev-1", "grounding", "real source")
    ]


def test_plan_evidence_refs_finds_ids_beyond_the_three_known_fields():  # recursive discovery
    plan = {
        # controls_and_confounders carries no evidence_id in today's schema; proves the traceback
        # is driven by the shared recursive collector, not a scan hardcoded to grounding,
        # materials_or_data, and metrics.
        "controls_and_confounders": [
            {"factor": "model size", "from_graph": True, "handling": "hold fixed",
             "evidence_id": "ev-9"},
        ],
    }
    assert connected_render._plan_evidence_refs(plan) == [
        ("ev-9", "controls_and_confounders", "")
    ]


def test_experiment_core_idea_uses_full_plain_design_without_truncation():
    rationale = (
        "Girth cannot be randomly assigned to a fixed graph, but a matched panel can hold spectral "
        "radius, average degree, graph size, message schedule, damping, quantizer family, and "
        "convergence tolerance fixed while varying girth, ending with the required tail marker "
        "FULL-CORE-RATIONALE-TAIL."
    )

    html = connected_render._experiment_core_summary({
        "design": "controlled_observational",
        "design_rationale": rationale,
    })

    assert "\u2026" not in html
    assert "Compare similar cases while holding key factors fixed or stratified" in html
    assert "FULL-CORE-RATIONALE-TAIL" in html


def test_experiment_plain_design_keeps_full_rationale_without_truncation():
    rationale = (
        "Girth cannot be randomly assigned to a fixed graph, but a matched panel can hold spectral "
        "radius, average degree, graph size, message schedule, damping, quantizer family, and "
        "convergence tolerance fixed while varying girth, ending with the required tail marker "
        "FULL-DESIGN-RATIONALE-TAIL."
    )

    text = connected_render._plain_design_text({
        "design": "controlled_observational",
        "design_rationale": rationale,
    })

    assert "\u2026" not in text
    assert "FULL-DESIGN-RATIONALE-TAIL" in text


def test_experiment_plain_design_covers_all_schema_design_labels():
    expected_phrases = {
        "randomized_controlled": "randomized controlled test",
        "controlled_observational": "Compare similar cases",
        "ablation": "ablation",
        "benchmark_comparison": "benchmark comparison",
        "simulation": "simulation study",
    }

    for design, phrase in expected_phrases.items():
        text = connected_render._plain_design_text({"design": design})
        assert phrase in text


def test_connected_view_promotes_headline_to_hypothesis_card():
    graph = {"version": 1, "nodes": {"n-1": {"label": "safe bitwidth", "provenance": []}}, "edges": {}}
    row = {"cid": "h1", "rank": 1, "cell": "safe bitwidth", "scores": {"rank": "1"}}
    plain = {
        "headline": "Fixed-point GBP needs a safe bitwidth certificate.",
        "what_might_be_happening": "Bounded errors may be absorbed.",
        "key_terms": [],
    }

    page, _ = connected_render.render_page(
        graph, "run", row, ["n-1"], [], {}, claim="Seed research goal text.", plain=plain,
        seed_kind="research_goal",
    )

    assert '<section class="claim"><h2>Proposed hypothesis</h2>' in page
    assert "Fixed-point GBP needs a safe bitwidth certificate." in page
    assert "<h2>Claim to investigate</h2>" not in page          # the old generic banner label does not return
    assert '<h2>Research goal</h2>' in page      # the seed's own context card, labeled by kind
    assert page.count("Seed research goal text.") == 1
    hypothesis_pos = page.index("<h2>Proposed hypothesis</h2>")
    seed_pos = page.index("<h2>Research goal</h2>")
    possible_pos = page.index("<h2>Possible explanation</h2>")
    what_pos = page.index("<b>What might be happening:</b>")
    assert seed_pos < hypothesis_pos < possible_pos < what_pos


def test_connected_view_does_not_repeat_hypothesis_in_explanation_lead():
    graph = {"version": 1, "nodes": {"n-1": {"label": "safe bitwidth", "provenance": []}}, "edges": {}}
    row = {"cid": "h1", "rank": 1, "cell": "safe bitwidth", "scores": {"rank": "1"}}
    headline = "Fixed-point GBP needs a safe bitwidth certificate."
    plain = {
        "headline": headline,
        "what_might_be_happening": "Bounded errors may be absorbed.",
        "key_terms": [],
    }

    page, _ = connected_render.render_page(
        graph, "run", row, ["n-1"], [], {}, claim="Seed research goal text.", plain=plain,
        seed_kind="research_goal",
    )

    assert page.count(headline) == 1
    assert page.count("Seed research goal text.") == 1
    possible = page.split("<h2>Possible explanation</h2>", 1)[1]
    assert possible.startswith("<p><b>What might be happening:</b>")


# --- Key-term links, familiar glosses, and translations in the connected view ----------
def _translation_records():
    return [
        {"original": "posterior error", "familiar": "reconstruction error",
         "first_use_rendering": "reconstruction error (posterior error)",
         "faithfulness": "confirmed", "rationale": "same L2 deviation object"},
        {"original": "loopy propagation", "familiar": "cyclic sweep",
         "first_use_rendering": "", "faithfulness": "rejected", "rationale": "different object"},
    ]


def test_connected_key_terms_render_link_and_familiar_gloss():  # linked key-term metadata
    terms = [{
        "term": "factor graph", "plain_meaning": "a map of variables", "source": "Paper A",
        "link": "https://en.wikipedia.org/wiki/Factor_graph", "link_kind": "wikipedia",
        "familiar_gloss": "like a tensor network diagram",
    }]
    html = connected_render._key_terms_html(terms)
    assert ('<b><a href="https://en.wikipedia.org/wiki/Factor_graph" title="wikipedia">'
            "factor graph</a></b>") in html
    assert " — in your terms: like a tensor network diagram" in html
    # entries without the new fields stay byte-identical (also enforced by the Research Synthesist legacy pin).
    plain = [{"term": "T", "plain_meaning": "d", "source": "Proposed concept"}]
    assert connected_render._key_terms_html(plain) == (
        '<ul><li><b>T</b> — d <span class="sub">(Proposed concept)</span></li></ul>'
    )


def test_connected_translations_panel_marks():  # translation rendering
    html = connected_render._translations_html(
        _translation_records(), "tensor factorization"
    )
    assert "<details><summary>Translated for a tensor factorization reader</summary>" in html
    assert "[confirmed]" in html
    assert "kept original — proposed analogy failed the faithfulness check" in html
    assert "cyclic sweep" not in html      # the rejected analogy phrasing never ships
    assert connected_render._translations_html([], "x") == ""
    # hypothesis_prose appends the panel after key_terms, fed by the sidecar's reader block
    plain = {"headline": "H may hold.", "key_terms": [{"term": "T", "plain_meaning": "d"}],
             "translations": _translation_records()[:1]}
    prose = connected_render.hypothesis_prose(
        plain, None, reader={"home_field": "tensor factorization"}
    )
    assert "Translated for a tensor factorization reader" in prose
    assert prose.index("</ul>") < prose.index("<details>")


def test_connected_view_reads_reader_block_from_sidecar(tmp_path, monkeypatch):  # reader plumbing
    import json

    run_dir = tmp_path / "run-x"
    (run_dir / "trace").mkdir(parents=True)
    graph = {
        "version": 3,
        "nodes": {"n-1": {"label": "codebook collapse", "type": "mechanism", "provenance": []}},
        "edges": {},
    }
    (run_dir / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    (run_dir / "trace" / "h1.html").write_text(
        "<table>"
        "<tr><td class=k>rank</td><td>1</td></tr>"
        "<tr><td class=k>candidate</td><td>h1</td></tr>"
        "<tr><td class=k>new concept(s)</td><td>codebook collapse</td></tr>"
        "</table>",
        encoding="utf-8",
    )
    sidecar = {
        "claim": "",
        "hypotheses": {"h1": {
            "headline": "Collapse may matter.",
            "key_terms": [{"term": "codebook", "plain_meaning": "a code table",
                           "link": "https://en.wikipedia.org/wiki/Codebook",
                           "link_kind": "wikipedia",
                           "familiar_gloss": "a lookup table of centroids"}],
            "translations": [{"original": "codebook", "familiar": "lookup table",
                              "first_use_rendering": "lookup table (codebook)",
                              "faithfulness": "confirmed", "rationale": "same object"}],
        }},
        "reader": {"home_field": "tensor factorization", "lexicon_source": "local paper arXiv-X"},
    }
    (run_dir / "trace" / "plain_language.json").write_text(json.dumps(sidecar), encoding="utf-8")
    monkeypatch.chdir(tmp_path)  # keep the *.sqlite glob away from the repo cwd
    assert connected_render.render_connected_hypotheses(run_dir)
    page = (run_dir / "h1-connected.html").read_text(encoding="utf-8")
    assert "Translated for a tensor factorization reader" in page
    assert "[confirmed]" in page
    assert 'title="wikipedia"' in page
    assert "in your terms: a lookup table of centroids" in page


def test_connected_old_sidecar_renders_unchanged():  # backward compatibility
    plain = {"headline": "H may hold.", "what_might_be_happening": "W.",
             "key_terms": [{"term": "T", "plain_meaning": "d", "source": "Proposed concept"}]}
    base = connected_render.hypothesis_prose(plain)
    assert base == connected_render.hypothesis_prose(plain, None, reader=None)
    assert base == connected_render.hypothesis_prose(plain, None, reader={})
    assert "Translated for" not in base and "<details" not in base
    # render_page: the reader kwarg defaults off -> byte-identical page (dropped-candidates path)
    graph = {"version": 1, "nodes": {"n-1": {"label": "x", "provenance": []}}, "edges": {}}
    row = {"cid": "h1", "rank": 1, "cell": "x", "scores": {"rank": "1"}}
    a, _ = connected_render.render_page(
        graph, "run", row, ["n-1"], [], {}, claim="", plain=plain,
    )
    b, _ = connected_render.render_page(
        graph, "run", row, ["n-1"], [], {}, claim="", plain=plain, reader=None,
    )
    assert a == b


# --- Script-wrapper and package-function parity ------------------------------------------
# The ONE remaining importlib path-load in this file: proves the CLI wrapper stays byte-identical
# to the package function it delegates to (every other test imports connected_render directly).
def test_script_wrapper_and_package_function_render_byte_identical_pages(tmp_path, monkeypatch):
    import importlib.util
    import json

    spec = importlib.util.spec_from_file_location(
        "rhc_under_test", Path("scripts/render_hypothesis_connected.py")
    )
    rhc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rhc)

    run_dir = tmp_path / "run-golden"
    (run_dir / "trace").mkdir(parents=True)
    graph = {
        "version": 1,
        "nodes": {"n-1": {"label": "golden concept", "type": "mechanism", "provenance": []}},
        "edges": {},
    }
    (run_dir / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    (run_dir / "trace" / "h1.html").write_text(
        "<table>"
        "<tr><td class=k>rank</td><td>1</td></tr>"
        "<tr><td class=k>candidate</td><td>h1</td></tr>"
        "<tr><td class=k>new concept(s)</td><td>golden concept</td></tr>"
        "</table>",
        encoding="utf-8",
    )
    sidecar = {
        "claim": "Golden seed claim.",
        "hypotheses": {"h1": {"headline": "Golden may matter.",
                              "what_might_be_happening": "Concept may hold.", "key_terms": []}},
    }
    (run_dir / "trace" / "plain_language.json").write_text(json.dumps(sidecar), encoding="utf-8")
    monkeypatch.chdir(tmp_path)  # keep the *.sqlite glob away from the repo cwd

    out_a = tmp_path / "out-package"
    out_b = tmp_path / "out-script"
    package_pages = connected_render.render_connected_hypotheses(run_dir, out_dir=out_a)
    assert rhc.main(["--run-dir", str(run_dir), "--out-dir", str(out_b)]) == 0

    assert len(package_pages) == 1
    for page_path in package_pages:
        assert (out_b / page_path.name).read_bytes() == page_path.read_bytes()


# --- Exact hypothesis-edge-ID plan binding ----------------------------------------------
def _shared_focus_run_dir(tmp_path):
    import json

    run_dir = tmp_path / "run-shared"
    (run_dir / "trace").mkdir(parents=True)
    graph = {
        "version": 5,
        "nodes": {
            "n-existing": {"label": "quantization noise", "type": "condition", "provenance": []},
            "n-1": {"label": "codebook drift", "type": "mechanism", "provenance": []},
            "n-2": {"label": "downstream calibration loss", "type": "outcome", "provenance": []},
        },
        # e-h2 chains off h1's own new concept (n-1), so n-1 is a focus node shared by BOTH edges.
        "edges": {
            "e-h1": {"source_node_ids": ["n-existing"], "target_node_ids": ["n-1"],
                     "relation_type": "drives", "status": "unverified"},
            "e-h2": {"source_node_ids": ["n-1"], "target_node_ids": ["n-2"],
                     "relation_type": "amplifies", "status": "unverified"},
        },
        "experiment_plans": {
            "e-h1": {"design": "simulation", "hypothesis_under_test": "PLAN-ONE-MARKER"},
            "e-h2": {"design": "ablation", "hypothesis_under_test": "PLAN-TWO-MARKER"},
        },
    }
    (run_dir / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    (run_dir / "trace" / "h1.html").write_text(
        "<table>"
        "<tr><td class=k>rank</td><td>1</td></tr>"
        "<tr><td class=k>candidate</td><td>h1</td></tr>"
        "<tr><td class=k>new concept(s)</td><td>codebook drift</td></tr>"
        "</table>",
        encoding="utf-8",
    )
    (run_dir / "trace" / "h2.html").write_text(
        "<table>"
        "<tr><td class=k>rank</td><td>2</td></tr>"
        "<tr><td class=k>candidate</td><td>h2</td></tr>"
        "<tr><td class=k>new concept(s)</td><td>downstream calibration loss</td></tr>"
        "</table>",
        encoding="utf-8",
    )
    return run_dir


def test_connected_render_binds_plans_by_exact_edge_id_when_edges_share_a_focus_node(tmp_path, monkeypatch):
    import json

    run_dir = _shared_focus_run_dir(tmp_path)
    sidecar = {"claim": "", "hypotheses": {},
               "hypothesis_edge_ids": {"h1": ["e-h1"], "h2": ["e-h2"]}}
    (run_dir / "trace" / "plain_language.json").write_text(json.dumps(sidecar), encoding="utf-8")
    monkeypatch.chdir(tmp_path)  # keep the *.sqlite glob away from the repo cwd

    out_dir = tmp_path / "out"
    connected_render.render_connected_hypotheses(run_dir, out_dir=out_dir)
    page_h1 = (out_dir / "h1-connected.html").read_text(encoding="utf-8")
    page_h2 = (out_dir / "h2-connected.html").read_text(encoding="utf-8")

    # e-h2 structurally touches h1's focus node (n-1), so adjacency alone would mis-bind h2's plan
    # onto h1's page too; the sidecar's exact edge id keeps each page bound to its own plan only.
    assert "PLAN-ONE-MARKER" in page_h1 and "PLAN-TWO-MARKER" not in page_h1
    assert "PLAN-TWO-MARKER" in page_h2 and "PLAN-ONE-MARKER" not in page_h2


def test_connected_render_falls_back_to_adjacency_for_a_legacy_sidecar(tmp_path, monkeypatch):
    import json

    run_dir = _shared_focus_run_dir(tmp_path)
    # A legacy sidecar has no ``hypothesis_edge_ids`` key.
    sidecar = {"claim": "", "hypotheses": {}}
    (run_dir / "trace" / "plain_language.json").write_text(json.dumps(sidecar), encoding="utf-8")
    monkeypatch.chdir(tmp_path)  # keep the *.sqlite glob away from the repo cwd

    out_dir = tmp_path / "out"
    connected_render.render_connected_hypotheses(run_dir, out_dir=out_dir)
    page_h1 = (out_dir / "h1-connected.html").read_text(encoding="utf-8")

    # without edge ids, the renderer still finds h1's own plan via focus-node adjacency.
    assert "PLAN-ONE-MARKER" in page_h1


# --- Seed identity and the separate seed-context card -----------------------------------
def _seed_card_run_dir(tmp_path):
    import json

    run_dir = tmp_path / "run-seed"
    (run_dir / "trace").mkdir(parents=True)
    graph = {
        "version": 1,
        "nodes": {"n-1": {"label": "codebook collapse", "type": "mechanism", "provenance": []}},
        "edges": {},
    }
    (run_dir / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    (run_dir / "trace" / "h1.html").write_text(
        "<table>"
        "<tr><td class=k>rank</td><td>1</td></tr>"
        "<tr><td class=k>candidate</td><td>h1</td></tr>"
        "<tr><td class=k>new concept(s)</td><td>codebook collapse</td></tr>"
        "</table>",
        encoding="utf-8",
    )
    return run_dir


def test_connected_render_shows_seed_context_card_when_sidecar_provides_seed_kind(tmp_path, monkeypatch):
    import json

    run_dir = _seed_card_run_dir(tmp_path)
    sidecar = {
        "claim": "Does quantization noise cause codebook collapse?",
        "seed_kind": "claim",
        "hypotheses": {"h1": {"headline": "Collapse may follow quantization noise.", "key_terms": []}},
    }
    (run_dir / "trace" / "plain_language.json").write_text(json.dumps(sidecar), encoding="utf-8")
    monkeypatch.chdir(tmp_path)  # keep the *.sqlite glob away from the repo cwd

    out_dir = tmp_path / "out"
    connected_render.render_connected_hypotheses(run_dir, out_dir=out_dir)
    page = (out_dir / "h1-connected.html").read_text(encoding="utf-8")

    assert '<section class="claim"><h2>Proposed hypothesis</h2>' in page
    assert "Collapse may follow quantization noise." in page
    assert "<h2>Claim to investigate</h2>" in page
    assert page.count("Does quantization noise cause codebook collapse?") == 1
    assert (page.index("<h2>Claim to investigate</h2>") < page.index("<h2>Proposed hypothesis</h2>")
            < page.index("<h2>Possible explanation</h2>"))


def test_connected_render_legacy_sidecar_without_seed_kind_omits_seed_card(tmp_path, monkeypatch):
    import json

    run_dir = _seed_card_run_dir(tmp_path)
    # A legacy sidecar has no ``seed_kind`` key, even when a headline exists.
    sidecar = {
        "claim": "Does quantization noise cause codebook collapse?",
        "hypotheses": {"h1": {"headline": "Collapse may follow quantization noise.", "key_terms": []}},
    }
    (run_dir / "trace" / "plain_language.json").write_text(json.dumps(sidecar), encoding="utf-8")
    monkeypatch.chdir(tmp_path)  # keep the *.sqlite glob away from the repo cwd

    out_dir = tmp_path / "out"
    connected_render.render_connected_hypotheses(run_dir, out_dir=out_dir)
    page = (out_dir / "h1-connected.html").read_text(encoding="utf-8")

    assert '<section class="claim"><h2>Proposed hypothesis</h2>' in page
    assert "Does quantization noise cause codebook collapse?" not in page
    assert "<h2>Claim to investigate</h2>" not in page


# --- Default post-trace connected-render pipeline ---------------------------------------
def _post_trace_run_root(tmp_path):
    import json

    run_root = tmp_path / "run-post-trace"
    (run_root / "trace").mkdir(parents=True)
    graph = {
        "version": 1,
        "nodes": {"n-1": {"label": "trace concept", "type": "mechanism", "provenance": []}},
        "edges": {},
    }
    (run_root / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    (run_root / "trace" / "h1.html").write_text(
        "<table>"
        "<tr><td class=k>rank</td><td>1</td></tr>"
        "<tr><td class=k>candidate</td><td>h1</td></tr>"
        "<tr><td class=k>new concept(s)</td><td>trace concept</td></tr>"
        "</table>",
        encoding="utf-8",
    )
    return run_root


def test_render_connected_after_trace_renders_one_page_per_surfaced_hypothesis(tmp_path, monkeypatch):
    run_root = _post_trace_run_root(tmp_path)
    monkeypatch.chdir(tmp_path)  # keep the *.sqlite glob away from the repo cwd

    pages = connected_render.render_connected_after_trace(run_root)

    assert [p.name for p in pages] == ["h1-connected.html"]
    assert (run_root / "h1-connected.html").exists()


def test_current_sidecar_skips_candidate_without_committed_edge_even_when_focus_exists(
    tmp_path, monkeypatch, capsys
):
    """Exact edge bindings override a coincidentally committed focus node."""
    import json

    run_root = _post_trace_run_root(tmp_path)
    sidecar = {
        "claim": "",
        "hypotheses": {},
        # A non-empty current-schema mapping exists, but h1 has no committed edge binding.
        "hypothesis_edge_ids": {"h2": ["e-h2"]},
    }
    (run_root / "trace" / "plain_language.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    assert connected_render.render_connected_hypotheses(run_root) == []
    assert not (run_root / "h1-connected.html").exists()
    assert "h1 (rank 1): not confirmed" in capsys.readouterr().out


def test_render_connected_hypotheses_uses_explicit_logical_run_id_for_audit_lookup(
    tmp_path, monkeypatch
):
    run_root = _post_trace_run_root(tmp_path)
    events_db = tmp_path / "events.sqlite"
    captured = {}

    def fake_events_db_for_run(candidates, run_id):
        captured["candidates"] = candidates
        captured["lookup_run_id"] = run_id
        return str(events_db)

    def fake_load_audit_events(db_path, run_id=None):
        captured["load_db_path"] = db_path
        captured["load_run_id"] = run_id
        return []

    monkeypatch.setattr(connected_render, "events_db_for_run", fake_events_db_for_run)
    monkeypatch.setattr(connected_render, "load_audit_events", fake_load_audit_events)
    monkeypatch.chdir(tmp_path)

    pages = connected_render.render_connected_hypotheses(
        run_root,
        events_db=events_db,
        run_id="logical-run-id",
    )

    assert [p.name for p in pages] == ["h1-connected.html"]
    assert captured == {
        "candidates": [str(events_db)],
        "lookup_run_id": "logical-run-id",
        "load_db_path": str(events_db),
        "load_run_id": "logical-run-id",
    }


def test_render_connected_after_trace_forwards_exact_audit_identity(tmp_path, monkeypatch):
    run_root = _post_trace_run_root(tmp_path)
    events_db = tmp_path / "events.sqlite"
    captured = {}

    def fake_render(run_dir, **kwargs):
        captured["run_dir"] = run_dir
        captured.update(kwargs)
        return []

    monkeypatch.setattr(connected_render, "render_connected_hypotheses", fake_render)

    assert connected_render.render_connected_after_trace(
        run_root,
        run_id="logical-run-id",
        events_db=events_db,
    ) == []
    assert captured == {
        "run_dir": run_root,
        "run_id": "logical-run-id",
        "events_db": events_db,
    }


def test_render_connected_after_trace_flag_disables_rendering_but_keeps_layout(tmp_path, monkeypatch):
    run_root = _post_trace_run_root(tmp_path)
    monkeypatch.chdir(tmp_path)  # keep the *.sqlite glob away from the repo cwd

    pages = connected_render.render_connected_after_trace(run_root, enabled=False)

    assert pages == []
    assert not (run_root / "h1-connected.html").exists()
    assert (run_root / "graph.json").exists()  # layout untouched by the escape hatch
    assert (run_root / "trace" / "h1.html").exists()


def test_render_connected_after_trace_warns_and_skips_on_incomplete_layout(tmp_path, monkeypatch, capsys):
    import json

    missing_graph = tmp_path / "run-no-graph"
    (missing_graph / "trace").mkdir(parents=True)

    missing_trace = tmp_path / "run-no-trace"
    missing_trace.mkdir(parents=True)
    (missing_trace / "graph.json").write_text(
        json.dumps({"version": 1, "nodes": {}, "edges": {}}), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)  # keep the *.sqlite glob away from the repo cwd

    assert connected_render.render_connected_after_trace(missing_graph) == []
    assert connected_render.render_connected_after_trace(missing_trace) == []
    assert capsys.readouterr().out.count("connected-render: skipped") == 2


# --- Literature provenance, verified edges, and the proposer self-assessment table ---------
# The renderer's provenance path (`node_class` -> `node_sources` -> `sources_section`, the
# clickable diagram links, and the verified/confidence edge styling) had no test: every graph
# in the fixtures above declares `"provenance": []`, so `node_class` always returned "hypo".
_PAPER_QUOTE = "quantization error accumulates across message passing rounds"


def _provenance_graph():
    return {
        "version": 1,
        "nodes": {
            "n-focus": {  # the hypothesis's own new concept: no provenance -> "hypo"
                "label": "safe bitwidth",
                "provenance": [],
            },
            "n-mined": {  # mined from the literature -> "mined", clickable, listed under Sources
                "label": "quantization error",
                "provenance": [
                    {
                        "source": "literature_enrichment",
                        "matched_quote_span": _PAPER_QUOTE,
                        "paper_id": "p-1",
                    }
                ],
            },
            "n-claim": {  # extracted from the seed claim -> "claim"
                "label": "message passing",
                "provenance": [{"source": "claim"}],
            },
            "n-prio": {  # author-pinned -> "prio", regardless of provenance
                "label": "hardware budget",
                "user_priority": True,
                "provenance": [{"source": "claim"}],
            },
        },
        "edges": {
            "e-supported": {
                "source_node_ids": ["n-mined"],
                "target_node_ids": ["n-focus"],
                "relation_type": "constrains",
                "status": "supported",
                "confidence": 0.87,
            },
            "e-insufficient": {
                "source_node_ids": ["n-claim"],
                "target_node_ids": ["n-focus"],
                "relation_type": "influences",
                "status": "insufficient",
                "confidence": 0.42,
            },
            "e-unverified": {
                "source_node_ids": ["n-prio"],
                "target_node_ids": ["n-focus"],
                "relation_type": "bounds",
                "status": "unverified",
                "confidence": None,
            },
        },
    }


def _rendered_provenance_page():
    pool = [
        {
            "quote": f"We find that {_PAPER_QUOTE} in fixed-point solvers.",
            "title": "Fixed-Point Belief Propagation",
            "url": "https://example.test/fixed-point.pdf",
            "source": "arxiv",
        }
    ]
    row = {"cid": "h1", "rank": 1, "cell": "safe bitwidth",
           "scores": {"rank": "1", "field novelty": "0.70"}}
    signals = {"novelty": 0.8, "testability": 0.6, "duplication": 0.1,
               "mechanism_specificity": "n/a"}
    page, _ = connected_render.render_page(
        _provenance_graph(), "run", row, ["n-focus"], pool, signals,
        claim="Fixed-point GBP needs a bitwidth certificate.",
    )
    return page


def test_connected_page_lists_mined_concepts_under_their_source_paper():
    page = _rendered_provenance_page()

    assert "Sources &mdash; where the literature-mined concepts came from" in page
    assert '<a href="https://example.test/fixed-point.pdf">Fixed-Point Belief Propagation</a>' in page
    assert "quantization error" in page
    assert _PAPER_QUOTE in page                 # the verbatim matched span is shown
    assert ">(arxiv)<" in page.replace('<span class="tag">', ">")
    # a concept the pool cannot resolve must not be attributed to this paper
    assert "hardware budget</b>" not in page.split("srcs")[-1]


def test_connected_diagram_classes_nodes_by_provenance_and_links_mined_ones():
    page = _rendered_provenance_page()

    # every provenance branch of node_class fires: focus, mined, claim, and author-pinned
    assert "class n0 focus;" in page
    for klass in ("mined", "claim", "prio"):
        assert f" {klass};" in page, klass
    # the mined node is clickable through to its source paper
    assert 'href "https://example.test/fixed-point.pdf"' in page
    assert "Fixed-Point Belief Propagation" in page


def test_connected_diagram_styles_verified_edges_and_labels_scored_ones():
    page = _rendered_provenance_page()

    # an evidence-supported edge is drawn solid green; an unverified one dashed orange
    assert "stroke:#4F6A46,stroke-width:3px;" in page
    assert "stroke-dasharray:6 4;" in page
    # only an insufficient-evidence edge carries its confidence on the label
    assert "influences · 0.42" in page
    assert "constrains · 0.87" not in page


def test_connected_scores_table_renders_the_proposer_self_assessment():
    page = _rendered_provenance_page()

    assert "Proposer&#x27;s self-assessment" in page or "Proposer's self-assessment" in page
    assert "Novelty (proposer&#x27;s estimate)" in page or "Novelty (proposer's estimate)" in page
    assert "0.80" in page and "0.60" in page      # floats formatted to two places
    assert "n/a" in page                          # a non-numeric rating is escaped, not crashed
    assert "were not logged by this run" not in page   # the no-signals fallback must not appear
