"""Shared experiment-plan presentation and normalization logic.

Renderers of committed experiment plans share one ordered field list, one design vocabulary, and
one placeholder-evidence rule. The Elaborator prompt keeps its own field order because it serves a
different output contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Canonical ordered display fields: ``(field key, plain-language label)``.
EXPERIMENT_PLAN_FIELDS: tuple[tuple[str, str], ...] = (
    ("design", "Design"),
    ("design_rationale", "Design rationale"),
    ("hypothesis_under_test", "Hypothesis under test"),
    ("intervention_or_manipulation", "Intervention / manipulation"),
    ("operationalization", "Operationalization"),
    ("materials_or_data", "Materials / data"),
    ("comparison_baseline", "Comparison baseline"),
    ("metrics", "Metrics"),
    ("procedure", "Procedure"),
    ("expected_outcome", "Expected outcome"),
    ("falsification", "Falsification"),
    ("controls_and_confounders", "Controls and confounders"),
    ("feasibility", "Feasibility"),
    ("grounding", "Grounding"),
)


# Design enum to plain-language description.
DESIGN_PLAIN_LANGUAGE = {
    "randomized_controlled": (
        "Use a randomized controlled test: assign cases to the intervention or baseline so the "
        "comparison isolates the causal effect."
    ),
    "controlled_observational": "Compare similar cases while holding key factors fixed or stratified.",
    "ablation": "Use an ablation: remove or vary one component at a time and measure how the result changes.",
    "benchmark_comparison": (
        "Use a benchmark comparison: run the proposed method and baselines on the same tasks and metrics."
    ),
    "simulation": (
        "Use a simulation study: create controlled problem instances where the causal factor can be "
        "swept directly."
    ),
}


def _full_text(text: Any) -> str:
    return " ".join(str(text or "").split())


def plain_design_text(plan: Mapping[str, Any]) -> str:
    """Plain-language sentence for ``plan``'s ``design`` (+ ``design_rationale`` when present); an
    unrecognized design value still renders (a generic "Use a <design> design." fallback), never
    raises."""
    design = str(plan.get("design", "") or "")
    base = DESIGN_PLAIN_LANGUAGE.get(design, f"Use a {design.replace('_', ' ')} design.")
    rationale = _full_text(plan.get("design_rationale"))
    return base + (f" In this plan: {rationale}" if rationale else "")


# --- (c) placeholder evidence-id normalization (moved from cycles/experiment.py) ----------
NO_EVIDENCE_PLACEHOLDERS = frozenset(
    {
        "n/a",
        "na",
        "none",
        "none_retrieved",
        "no_retrieved_evidence",
        "no_retrieved_passage",
        "not_applicable",
        "null",
    }
)


def real_evidence_id(evidence_id: str | None) -> str:
    """``evidence_id`` stripped of Experiment Designer's "no retrieved evidence" placeholders (``n/a``,
    ``none_retrieved``, ...); ``""`` when absent or a placeholder, the original id otherwise."""
    eid = str(evidence_id or "").strip()
    normalized = eid.lower().replace("-", "_").replace(" ", "_")
    return "" if not eid or normalized in NO_EVIDENCE_PLACEHOLDERS else eid


def optional_evidence_id(evidence_id: str | None) -> str | None:
    return real_evidence_id(evidence_id) or None


# --- (d) recursive evidence-reference collector --------------------------------------------
def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    dump = getattr(value, "model_dump", None)
    return dump() if callable(dump) else None


def _walk_evidence_references(value: Any, path: str, refs: list[tuple[str, str]]) -> None:
    mapping = _as_mapping(value)
    if mapping is not None:
        if "evidence_id" in mapping:
            eid = real_evidence_id(mapping["evidence_id"])
            if eid:
                refs.append((f"{path}.evidence_id" if path else "evidence_id", eid))
        for key, child in mapping.items():
            if key == "evidence_id":
                continue
            _walk_evidence_references(child, f"{path}.{key}" if path else key, refs)
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _walk_evidence_references(item, f"{path}[{index}]", refs)


def collect_evidence_references(plan: Any) -> list[tuple[str, str]]:
    """Walk ``plan`` (a dict or a pydantic model, e.g. ``ExperimentPlan``) and collect every
    ``evidence_id`` it carries at any nesting depth — ``grounding`` / ``materials_or_data`` /
    ``metrics`` today, plus whatever nested field a future schema cites evidence on. Returns
    ``(path_label, evidence_id)`` pairs such as ``grounding[0].evidence_id`` or
    ``materials_or_data[1].evidence_id``, one per occurrence, so an id cited from several places
    keeps every path that cited it. Placeholder ids (``NO_EVIDENCE_PLACEHOLDERS``) are dropped."""
    refs: list[tuple[str, str]] = []
    _walk_evidence_references(plan, "", refs)
    return refs
