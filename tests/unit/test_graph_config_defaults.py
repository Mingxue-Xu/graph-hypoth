"""Presence and invariant tests for graph-domain configuration defaults.

Retrieval-scoring constants live in ``retrieval/scoring_defaults.py``; graph scheduler,
loop, transaction, and extraction-cycle constants live in a separate graph-domain
defaults module. These tests require every weight, threshold, and cap to be versioned
and audit-surfaced rather than hardcoded in domain truth.

Source-of-truth values:
- scheduler selection and continuation: w_p=w_u=w_g=1/3, theta_sel=0.5,
  N_max=15, N_stall=3; P_default=0.5.
- scope-gap formula: w_M=0.5, w_U=0.2, w_B=0.3; a_f=1.0 for PICO-core facets and
  0.5 others; clarification theta=0.5.
- extraction-confidence formula: w_span=0.45, w_qual=0.35, w_assumption=0.40
  (penalty in the numerator AND denominator -> the set does NOT sum to 1).
- merge-safety formula after dropping ScopeCompat: the
  remaining four renormalized -> w_label=0.273, w_embed=0.318, w_def=0.182, w_type=0.227;
  merge theta=0.55.

These are calibration proposals (audit-surfaced, NOT doc truth), so behaviour is
pinned GIVEN the version stamp, never as a "correct" value; floats compared at +/-1e-6.
"""

import pytest

from src import graph_config_defaults as gd


def test_graph_defaults_version_present():  # non-empty version stamp
    assert isinstance(gd.GRAPH_DEFAULTS_VERSION, str) and gd.GRAPH_DEFAULTS_VERSION


def test_scheduler_selection_weights_equal_third_and_sum_one():
    assert set(gd.SEL_WEIGHTS) == {"p", "u", "g"}  # exactly three weights, no phantom w_c
    for w in gd.SEL_WEIGHTS.values():
        assert w == pytest.approx(1 / 3, abs=1e-6)
    assert sum(gd.SEL_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-6)


def test_loop_and_threshold_constants():
    assert gd.THETA_SEL == pytest.approx(0.5, abs=1e-6)
    assert gd.N_MAX == 15
    assert gd.N_STALL == 3
    assert gd.P_DEFAULT == pytest.approx(0.5, abs=1e-6)


def test_merge_weights_without_scope_compat_sum_one():
    assert set(gd.MERGE_WEIGHTS) == {"label", "embed", "def", "type"}  # scope compatibility excluded
    assert gd.MERGE_WEIGHTS["label"] == pytest.approx(0.273, abs=1e-6)
    assert gd.MERGE_WEIGHTS["embed"] == pytest.approx(0.318, abs=1e-6)
    assert gd.MERGE_WEIGHTS["def"] == pytest.approx(0.182, abs=1e-6)
    assert gd.MERGE_WEIGHTS["type"] == pytest.approx(0.227, abs=1e-6)
    assert sum(gd.MERGE_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-6)
    assert gd.THETA_MERGE == pytest.approx(0.55, abs=1e-6)


def test_scope_gap_weights_and_facets():  # scope-gap coefficients
    assert gd.SCOPE_GAP_WEIGHTS["missing"] == pytest.approx(0.5, abs=1e-6)
    assert gd.SCOPE_GAP_WEIGHTS["terms"] == pytest.approx(0.2, abs=1e-6)
    assert gd.SCOPE_GAP_WEIGHTS["blockers"] == pytest.approx(0.3, abs=1e-6)
    assert sum(gd.SCOPE_GAP_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-6)
    assert gd.THETA_SCOPEGAP == pytest.approx(0.5, abs=1e-6)
    # Closed facet set F (6) with PICO-core weighted 1.0 and the rest 0.5.
    facets = gd.SCOPE_GAP_FACET_WEIGHTS
    assert set(facets) == {
        "population", "exposure/intervention", "outcome", "comparator", "context", "time",
    }
    for core in ("population", "exposure/intervention", "outcome"):
        assert facets[core] == pytest.approx(1.0, abs=1e-6)
    for other in ("comparator", "context", "time"):
        assert facets[other] == pytest.approx(0.5, abs=1e-6)


def test_extract_conf_weights_do_not_sum_to_one():  # assumption penalty is in the denominator
    assert gd.EXTRACT_CONF_WEIGHTS["span"] == pytest.approx(0.45, abs=1e-6)
    assert gd.EXTRACT_CONF_WEIGHTS["qual"] == pytest.approx(0.35, abs=1e-6)
    assert gd.EXTRACT_CONF_WEIGHTS["assumption"] == pytest.approx(0.40, abs=1e-6)
    # The achievable max is Sum(w_pos)/(Sum(w_pos)+w_assumption+eps) < 1, so the raw
    # three-weight set deliberately sums above 1.0 -- guard against a wrong "normalise".
    assert sum(gd.EXTRACT_CONF_WEIGHTS.values()) > 1.0


def test_synthesist_gate_thresholds_are_calibrated():
    # theta_novelty 0.4, theta_testability 0.5, theta_scope 0.5, theta_dup 0.8.
    t = gd.HYP_GATE_THRESHOLDS
    assert set(t) == {"novelty", "testability", "scope", "dup"}
    assert t["novelty"] == pytest.approx(0.4, abs=1e-6)
    assert t["testability"] == pytest.approx(0.5, abs=1e-6)
    assert t["scope"] == pytest.approx(0.5, abs=1e-6)
    assert t["dup"] == pytest.approx(0.8, abs=1e-6)


def test_hypscore_weights_equal_sixth_and_sum_one():  # six equal ranking weights
    w = gd.HYPSCORE_WEIGHTS
    assert set(w) == {"nov", "plaus", "test", "voi", "imp", "mech"}
    for value in w.values():
        assert value == pytest.approx(1 / 6, abs=1e-6)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)


def test_expansion_feature_flag_defaults_on():  # enabled-by-default contract
    # The run path consults this flag before invoking the Synthesist and Critic Panel seams.
    assert gd.EXPANSION_ENABLED is True


def test_synthesist_fusion_weights_and_budget_present():
    # Each LLM/deterministic fusion pair sums to 1.0; novelty and scope lean LLM-first,
    # while duplication leans deterministic.
    fusion = gd.HYP_FUSION_WEIGHTS
    assert set(fusion) == {"novelty", "testability", "scope", "duplication"}
    for pair in fusion.values():
        assert set(pair) == {"llm", "det"}
        assert pair["llm"] + pair["det"] == pytest.approx(1.0, abs=1e-6)
    assert fusion["novelty"]["llm"] > fusion["novelty"]["det"]       # LLM-primary
    assert fusion["duplication"]["det"] > fusion["duplication"]["llm"]  # deterministic-leaning
    assert isinstance(gd.EXPANSION_BUDGET_B, int) and gd.EXPANSION_BUDGET_B >= 1


def test_passage_prompt_max_chars_present_and_audit_surfaced():  # redundancy-fix: passage-cap
    # Single shared per-passage LLM-prompt cap for run_path/finalization/reader_translation
    # (previously 4000 defined independently 3x under 3 local names).
    assert gd.PASSAGE_PROMPT_MAX_CHARS == 4000
    audit = gd.audit_dict()
    assert audit["passage_prompt_max_chars"] == gd.PASSAGE_PROMPT_MAX_CHARS


def test_epsilon_floor():  # numerical-stability floor
    assert gd.EPSILON == pytest.approx(1e-9, abs=1e-15)


def test_audit_dict_surfaces_every_constant():  # every constant is audit-visible
    audit = gd.audit_dict()
    assert audit["graph_defaults_version"] == gd.GRAPH_DEFAULTS_VERSION
    module_constants = {n for n in vars(gd) if n.isupper()}
    for name in module_constants:
        assert name.lower() in audit, f"{name} is not surfaced in audit_dict()"
