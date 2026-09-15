"""Centralized graph-domain defaults for the causal-claim-graph pipeline.

Retrieval and evidence-scoring constants live in ``retrieval/scoring_defaults.py``;
scheduler, extraction, synthesis, and experiment defaults live here. Values are
versioned calibration inputs and are exposed through :func:`audit_dict` for replay.
"""

from __future__ import annotations

# Bump when a value below changes so replay can identify the calibration set.
GRAPH_DEFAULTS_VERSION = "graph-defaults-2026-08-30"

# ---------------------------------------------------------------------------
# Scheduler selection uses priority, uncertainty, and evidence-gap weights. Cost is a
# hard budget guard rather than a fourth weighted term.
# ---------------------------------------------------------------------------
SEL_WEIGHTS = {"p": 1 / 3, "u": 1 / 3, "g": 1 / 3}   # w_p (priority), w_u (uncertainty), w_g (gap)

# ---------------------------------------------------------------------------
# Loop and budget constants.
# ---------------------------------------------------------------------------
THETA_SEL = 0.5      # target selection floor, subject to affordability
N_MAX = 15           # iteration safety cap
N_STALL = 3          # stop after this many consecutive no-progress steps

# ---------------------------------------------------------------------------
# User-priority projection default: midpoint of the [0,1] scale.
# ---------------------------------------------------------------------------
P_DEFAULT = 0.5

# ---------------------------------------------------------------------------
# Extraction scope-gap weights use a clipped weighted sum without denominator normalization.
# ---------------------------------------------------------------------------
SCOPE_GAP_WEIGHTS = {"missing": 0.5, "terms": 0.2, "blockers": 0.3}  # w_M, w_U, w_B
THETA_SCOPEGAP = 0.5   # clarification threshold
# Facet weights over the closed six-facet set.
SCOPE_GAP_FACET_WEIGHTS = {
    "population": 1.0,
    "exposure/intervention": 1.0,
    "outcome": 1.0,
    "comparator": 0.5,
    "context": 0.5,
    "time": 0.5,
}

# ---------------------------------------------------------------------------
# Extraction-confidence weights. The assumption weight is a penalty
# in the numerator AND the denominator, so the achievable max is
# Sum(w_pos)/(Sum(w_pos)+w_assumption+eps) < 1; the three values do NOT sum to 1.
# ---------------------------------------------------------------------------
EXTRACT_CONF_WEIGHTS = {"span": 0.45, "qual": 0.35, "assumption": 0.40}  # w_span, w_qual, w_assumption

# ---------------------------------------------------------------------------
# Extraction merge safety combines label, embedding, definition, and type signals.
# Type is a subtractive penalty but remains part of the normalized weight set.
# ---------------------------------------------------------------------------
MERGE_WEIGHTS = {"label": 0.273, "embed": 0.318, "def": 0.182, "type": 0.227}
THETA_MERGE = 0.55     # merge decision threshold

# The canonicalization band sends ambiguous merge scores to an LLM adjudicator. The
# asymmetric band is wider below the threshold to catch synonym false negatives and
# narrower above it to catch near-threshold false positives.
# Scores outside the band are handled deterministically.
CANONICALIZATION_BAND_LOW = 0.25    # adjudicate down to THETA_MERGE - 0.25 (= 0.30)
CANONICALIZATION_BAND_HIGH = 0.05   # adjudicate up to THETA_MERGE + 0.05 (= 0.60)

# ---------------------------------------------------------------------------
# Synthesist hypothesis gates, scoring, and per-signal late-fusion weights.
# ---------------------------------------------------------------------------
HYP_GATE_THRESHOLDS = {"novelty": 0.4, "testability": 0.5, "scope": 0.5, "dup": 0.8}
HYPSCORE_WEIGHTS = {
    "nov": 1 / 6, "plaus": 1 / 6, "test": 1 / 6, "voi": 1 / 6, "imp": 1 / 6, "mech": 1 / 6,
}
HYP_FUSION_WEIGHTS = {
    "novelty": {"llm": 0.7, "det": 0.3},
    "testability": {"llm": 0.5, "det": 0.5},
    "scope": {"llm": 0.7, "det": 0.3},
    "duplication": {"llm": 0.3, "det": 0.7},
}
EXPANSION_BUDGET_B = 5   # top candidates surfaced under the expansion budget
# Expansion runs whenever the Research Synthesist and Critic Panel seams are wired.
EXPANSION_ENABLED = True

# ---------------------------------------------------------------------------
# Field-relative novelty and literature-saturation ranking. Saturation is a soft
# multiplicative demotion; the panel median resists a single outlier judge.
# ---------------------------------------------------------------------------
SATURATION_PENALTY_ENABLED = True
THETA_SATURATION = 0.95
NOVELTY_PANEL_REDUCER = "median"

# Claimless-discovery emphasis weights. ``blended`` combines authored concept weight with
# RankScore; ``author_directed`` is lexicographic and does not use these weights.
EMPHASIS_BLEND_WEIGHTS = {"author": 0.5, "rank": 0.5}

# ---------------------------------------------------------------------------
# Literature concept mining commits through extraction deltas so provenance distinguishes
# mined concepts without a separate delta family. The admission cap bounds graph growth.
# ---------------------------------------------------------------------------
ENRICHMENT_MAX_CONCEPTS = 20
ENRICHMENT_MAX_PASSAGES = 20
# Per-passage prompt cap shared by the run path and reader translation.
PASSAGE_PROMPT_MAX_CHARS = 4000

# Experiment Designer methods/datasets/baselines retrieval cap.
EXPERIMENT_METHODS_QUOTES = 5

# Small numeric floor for saturating ratios and denominators.
EPSILON = 1e-9


def audit_dict() -> dict[str, object]:
    """Return a flat, audit-surfaceable view of every graph-domain default."""
    return {
        "graph_defaults_version": GRAPH_DEFAULTS_VERSION,
        "sel_weights": SEL_WEIGHTS,
        "theta_sel": THETA_SEL,
        "n_max": N_MAX,
        "n_stall": N_STALL,
        "p_default": P_DEFAULT,
        "scope_gap_weights": SCOPE_GAP_WEIGHTS,
        "theta_scopegap": THETA_SCOPEGAP,
        "scope_gap_facet_weights": SCOPE_GAP_FACET_WEIGHTS,
        "extract_conf_weights": EXTRACT_CONF_WEIGHTS,
        "merge_weights": MERGE_WEIGHTS,
        "theta_merge": THETA_MERGE,
        "canonicalization_band_low": CANONICALIZATION_BAND_LOW,
        "canonicalization_band_high": CANONICALIZATION_BAND_HIGH,
        "hyp_gate_thresholds": HYP_GATE_THRESHOLDS,
        "hypscore_weights": HYPSCORE_WEIGHTS,
        "hyp_fusion_weights": HYP_FUSION_WEIGHTS,
        "expansion_budget_b": EXPANSION_BUDGET_B,
        "expansion_enabled": EXPANSION_ENABLED,
        "saturation_penalty_enabled": SATURATION_PENALTY_ENABLED,
        "theta_saturation": THETA_SATURATION,
        "novelty_panel_reducer": NOVELTY_PANEL_REDUCER,
        "emphasis_blend_weights": EMPHASIS_BLEND_WEIGHTS,
        "enrichment_max_concepts": ENRICHMENT_MAX_CONCEPTS,
        "enrichment_max_passages": ENRICHMENT_MAX_PASSAGES,
        "passage_prompt_max_chars": PASSAGE_PROMPT_MAX_CHARS,
        "experiment_methods_quotes": EXPERIMENT_METHODS_QUOTES,
        "epsilon": EPSILON,
    }
