"""Deterministic trace render for the standalone Research Synthesist run and prose elaborator.

Answers Q1 in code: the structural per-hypothesis HTML needs NO LLM — it is rendered by copying
every number / label / score programmatically from the structured ``GraphRunResult.surfaced`` rows,
so it cannot drift from the run. The ONLY LLM-authored content is the plain-language prose
(``LLMHypothesisElaborator`` — one structured completion per hypothesis, no agent / no tools), and
it is additive: the structural page renders complete without it.
"""

from __future__ import annotations

import html as _html
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.progress import report_progress
from src.relation_labels import normalize_relation_label
from src.report_context import SEED_LABELS, relationship_section


def _esc(value: Any) -> str:
    return _html.escape(str(value))


# --- optional LLM prose (the single completion per hypothesis) ---------------------------
_ELABORATION_SYSTEM_PROMPT = (
    "You write a SHORT plain-language reader card for one proposed research hypothesis. The reader "
    "is the researcher who requested it; when a Reader profile is provided, use that field, "
    "expertise, familiar terminology, and analogy domains to decide which terms can stay technical "
    "and which need explaining. They need the idea, the practical reason to care, and the test, "
    "before any technical details. Use plain simple English, active voice, short sentences, and one idea per "
    "sentence. The headline MUST be one hedged sentence of 20 words or fewer and use may, might, or "
    "could. Explain every cross-field, paper-coined, or unfamiliar technical term the first time it "
    "appears. When a Public idea scaffold is provided, use it to preserve the claim anchor, the "
    "named lever, the scoped guarantee/fail-safe, and the problem-method-experiment structure. "
    "When a panel-audited mechanism chain is provided, what_might_be_happening must follow that chain "
    "in order and explain each relation in words. Do not copy node IDs, edge notation, or arrows "
    "into the headline or explanatory prose. When a verified terminology list is provided, "
    "use each term exactly in its verified sense. next_steps is the concrete path a reader could "
    "follow to test the idea: each step one plain sentence, in doing order, no jargon; use only "
    "the named concepts and the experiment/method scaffold fields. "
    "Explain the proposal's relationship to the starting input in relationship_to_seed: "
    "does it propose a mechanism, a condition, a failure boundary, a method, or a predictor? "
    "Use one or two plain sentences grounded in the supplied starting input and proposal. "
    "Say what part of the starting input it addresses and do not imply that testing this "
    "proposal proves the whole starting claim. A proposal may challenge or qualify that claim. "
    "For a research goal, explain how it advances the goal; do not turn the goal into a fact. "
    "For a research question, explain what part it helps answer and leave the answer open; "
    "do not describe the question as a claim that is already true. "
    "If no starting input is supplied or the connection is unclear, say so without inventing one. "
    "Do NOT mention or restate technical score labels such as HypScore, RankScore, "
    "saturation, field novelty, cross-concept, or common-sense. Return STRICT JSON only (no "
    "markdown): "
    '{"headline": "<20 words or fewer; hedged; plain English>", '
    '"relationship_to_seed": "<how this proposal relates to the starting input; empty if none supplied>", '
    '"what_might_be_happening": "<1-2 short sentences explaining the possible mechanism>", '
    '"why_it_matters": "<1 short sentence on practical or research importance>", '
    '"how_to_check": "<1 short sentence naming the comparison or measurement>", '
    '"what_would_change_our_mind": "<1 short sentence naming a result that would weaken it>", '
    '"problem_solved": "<1 short sentence naming the specific problem this idea addresses>", '
    '"next_steps": ["<3-5 ordered concrete steps: setup, data, baseline/comparison, measurement, '
    'expected outcome>"], '
    '"status": "This is a testable idea, not a proven conclusion.", '
    '"key_terms": [{"term": "...", "plain_meaning": "<short plain definition>", "source": "..."}]}. '
    "Do NOT invent numbers or claims beyond the provided rationale/definitions. "
    "Each key_terms plain_meaning must be SELF-CONTAINED: no undefined jargon inside a "
    "definition; the reader must understand it without reading any other definition or the "
    "paper; if a definition needs another technical term, unpack that term inline in plain words."
)


def _plain_item(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        if value.get("term") and value.get("gloss"):
            return f"{value['term']} ({value['gloss']})"
        if value.get("term"):
            return str(value["term"])
        if value.get("title"):
            return str(value["title"])
    term = getattr(value, "term", None)
    gloss = getattr(value, "gloss", None)
    if term and gloss:
        return f"{term} ({gloss})"
    if term:
        return str(term)
    title = getattr(value, "title", None)
    if title:
        return str(title)
    return str(value)


def _profile_value(profile: Any, name: str) -> Any:
    if isinstance(profile, Mapping):
        return profile.get(name)
    return getattr(profile, name, None)


def _reader_profile_block(reader_profile: Any) -> str:
    if not reader_profile:
        return ""
    lexicon = _profile_value(reader_profile, "reader_lexicon")
    field = (
        _profile_value(reader_profile, "field")
        or _profile_value(lexicon, "home_field")
        or ""
    )
    expertise = _profile_value(reader_profile, "expertise") or ""
    familiar_terms = (
        _profile_value(reader_profile, "familiar_terms")
        or _profile_value(lexicon, "familiar_terms")
        or _profile_value(reader_profile, "concepts")
        or ()
    )
    analogy_domains = (
        _profile_value(reader_profile, "analogy_domains")
        or _profile_value(lexicon, "analogy_domains")
        or ()
    )
    methods = _profile_value(reader_profile, "methods") or ()
    interest = _profile_value(reader_profile, "interest") or ""
    lines: list[str] = []
    if field:
        lines.append(f"- field: {field}")
    if expertise:
        lines.append(f"- expertise: {expertise}")
    familiar = ", ".join(_plain_item(item) for item in familiar_terms if str(_plain_item(item)).strip())
    if familiar:
        lines.append(f"- familiar_terms: {familiar}")
    analogies = ", ".join(str(item) for item in analogy_domains if str(item).strip())
    if analogies:
        lines.append(f"- analogy_domains: {analogies}")
    method_text = ", ".join(str(item) for item in methods if str(item).strip())
    if method_text:
        lines.append(f"- methods: {method_text}")
    if interest:
        lines.append(f"- interest: {interest}")
    return "\nReader profile:\n" + "\n".join(lines) + "\n" if lines else ""


def _new_concepts_text(surfaced: Any) -> str:
    details = getattr(surfaced, "new_node_details", ()) or ()
    rows: list[str] = []
    for node in details:
        if not isinstance(node, Mapping):
            continue
        label = str(node.get("label", "") or "")
        if not label:
            continue
        kind = str(node.get("type", "") or "")
        definition = str(node.get("definition", "") or "")
        typed = f" ({kind})" if kind else ""
        defined = f": {definition}" if definition else ""
        rows.append(f"{label}{typed}{defined}")
    if rows:
        return "; ".join(rows)
    return ", ".join(getattr(surfaced, "new_node_labels", ()) or ())


def _elaboration_user_prompt(
    surfaced: Any,
    concept_definitions: Mapping[str, str] | None,
    reader_profile: Any = None,
) -> str:
    nodes = _new_concepts_text(surfaced)
    defs = ""
    if concept_definitions:
        lines = "\n".join(
            f"- {label}: {definition}"
            for label, definition in concept_definitions.items()
            if definition
        )
        if lines:
            defs = f"\nConcept definitions (the source for your universal glosses):\n{lines}\n"
    scaffold = getattr(surfaced, "idea_scaffold", {}) or {}
    scaffold_text = ""
    if scaffold:
        lines = "\n".join(f"- {key}: {value}" for key, value in scaffold.items() if value)
        if lines:
            scaffold_text = f"\nPublic idea scaffold:\n{lines}\n"
    # Critic Panel mechanism/term grades (getattr-tolerant like lineage): these fields are the Session-C
    # report-integration sink (empty until the panel grades are wired onto the row), so rows
    # without them contribute nothing and the prompt stays byte-identical.
    steps_text = ""
    step_lines: list[str] = []
    for step in getattr(surfaced, "mechanism_steps", ()) or ():
        if not isinstance(step, Mapping):
            continue
        line = f"- {step.get('from', '')} --{normalize_relation_label(step.get('relation', ''))}--> {step.get('to', '')}"
        verdict = str(step.get("verdict", "") or "")
        if verdict:
            line += f" [{verdict}]"
        mechanism = str(step.get("mechanism", "") or "")
        if mechanism:
            line += f": {mechanism}"
        step_lines.append(line)
    if step_lines:
        steps_text = "\nMechanism audit (verdicts preserved):\n" + "\n".join(step_lines) + "\n"
    terms_text = ""
    term_lines: list[str] = []
    for verdict in getattr(surfaced, "term_audit", ()) or ():
        if not isinstance(verdict, Mapping) or not verdict.get("term"):
            continue
        line = f"- {verdict.get('term', '')} — {verdict.get('verdict', '')}"
        quote = str(verdict.get("source_quote", "") or "")
        if quote:
            line += f' — "{quote}"'
        term_lines.append(line)
    if term_lines:
        terms_text = "\nTerminology audit (term — verdict — source quote):\n" + "\n".join(term_lines) + "\n"
    # : when a committed experiment_plan is attached (Experiment Designer/Experiment Validator stage ran), how_to_check and
    # next_steps render FROM the plan, not the one-line idea_scaffold. getattr-tolerant like the
    # grades above, so rows without a plan keep the prompt byte-identical (pre-experiment path).
    plan_text = _experiment_plan_block(getattr(surfaced, "experiment_plan", None))
    reader_profile_text = _reader_profile_block(reader_profile)
    seed = _profile_value(reader_profile, "claim") if reader_profile else None
    seed_kind = _profile_value(reader_profile, "seed_kind") if reader_profile else None
    seed_text = (
        f"Starting input ({SEED_LABELS.get(seed_kind or 'claim', 'Starting input')}): {seed}\n"
        if seed else ""
    )
    return (
        f"Hypothesis id: {surfaced.candidate_id}\n"
        f"Proposed new concept(s): {nodes}\n"
        f"Proposer rationale: {surfaced.rationale}\n"
        f"{seed_text}"
        f"{reader_profile_text}"
        f"{scaffold_text}"
        f"{steps_text}"
        f"{terms_text}"
        f"{plan_text}"
        f"{defs}\n"
        "Write the plain-language reader card as STRICT JSON. Do not mention the numeric scores."
    )


def _experiment_plan_block(plan: Any) -> str:
    """Render the committed experiment plan for the Elaborator card prompt: the design, baseline,
    procedure, metrics, and both outcomes — the source of how_to_check / next_steps. Empty (byte-
    identical) when no plan is attached."""
    if not isinstance(plan, Mapping) or not plan:
        return ""
    parts: list[str] = []
    for key in (
        "hypothesis_under_test",
        "design",
        "operationalization",
        "intervention_or_manipulation",
        "comparison_baseline",
        "controls_and_confounders",
        "materials_or_data",
        "metrics",
        "procedure",
        "expected_outcome",
        "falsification",
        "feasibility",
        "grounding",
    ):
        if plan.get(key):
            value = plan[key]
            if key == "procedure":
                value = "; ".join(str(step) for step in value if step)
            elif key == "metrics":
                metric_names = [
                    str(m.get("metric", "")) for m in value
                    if isinstance(m, Mapping) and m.get("metric")
                ]
                value = ", ".join(metric_names) if metric_names else value
            if isinstance(value, (Mapping, list, tuple)):
                value = json.dumps(value, ensure_ascii=False, sort_keys=True)
            parts.append(f"{key}: {value}")
    if not parts:
        return ""
    return (
        "\nCommitted experiment plan (render how_to_check and next_steps FROM this plan, not the "
        "idea scaffold):\n" + "\n".join(f"- {part}" for part in parts) + "\n"
    )


class LLMHypothesisElaborator:
    """One structured completion per hypothesis -> the plain-language prose (no agent, no tools, no
    run-time web search). The structural trace renders fully without it; this only adds the hero.
    ``concept_definitions`` (label -> universal definition, from the miner) is the source for the
    inline glosses so paper-coined terms are expanded faithfully, not from the model's guess."""

    def __init__(self, model_backend: Any, *, system_prompt: str | None = None) -> None:
        self._backend = model_backend
        self._system_prompt = system_prompt or _ELABORATION_SYSTEM_PROMPT

    def elaborate(
        self,
        surfaced: Any,
        *,
        concept_definitions: Mapping[str, str] | None = None,
        reader_profile: Any = None,
    ) -> dict[str, Any]:
        from src.camel_adapter import backend_json

        return backend_json(
            self._backend,
            self._system_prompt,
            _elaboration_user_prompt(surfaced, concept_definitions, reader_profile),
        )


def elaborate_all(
    elaborator: LLMHypothesisElaborator,
    surfaced: Sequence[Any],
    *,
    concept_definitions: Mapping[str, str] | None = None,
    reader_profile: Any = None,
) -> dict[str, dict]:
    """Elaborate every surfaced hypothesis (one completion each). Pure fan-out over the seam;
    ``concept_definitions`` (label -> universal definition) is shared across all of them."""
    elaborations = {}
    for index, row in enumerate(surfaced, 1):
        report_progress(
            "Writing reports", "elaborating hypothesis",
            current=index, total=len(surfaced),
        )
        elaborations[row.candidate_id] = elaborator.elaborate(
            row, concept_definitions=concept_definitions, reader_profile=reader_profile
        )
    return elaborations


# --- deterministic structural render (NO LLM) --------------------------------------------
_STYLE = (
    "<style>body{font-family:system-ui,sans-serif;margin:2rem;max-width:60rem}"
    "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.3rem .6rem;text-align:left}"
    ".k{color:#555}</style>"
)
_PAGE = '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{title}</title>{style}</head><body>{body}</body></html>'


def _page(title: str, body: str) -> str:
    return _PAGE.format(title=_esc(title), style=_STYLE, body=body)


def _source_url_index(sources: Any) -> list[tuple[str, str]]:
    """(lowercased title, url) pairs for linkifying citations that name a retrieved paper."""
    pairs: list[tuple[str, str]] = []
    for src in sources or []:
        if not isinstance(src, Mapping):
            continue
        title = str(src.get("title", "") or "").strip().lower()
        url = str(src.get("url", "") or "").strip()
        if title and url.startswith(("http://", "https://")):
            pairs.append((title, url))
    return pairs


def _cite_html(source: str, paper_urls: list[tuple[str, str]]) -> str:
    """Render a Key-term ``source`` as a citation: a real link when it names a retrieved paper,
    otherwise a plain note (self-coined ``this paper …`` / field-standard ``standard …`` have no
    external paper to link, so they must NOT look like a bracketed/broken citation)."""
    source = source.strip()
    if not source:
        return ""
    low = source.lower()
    url = next((u for title, u in paper_urls if title in low or low in title), "")
    if url:
        return f' <span class="k">[<a href="{_esc(url)}">{_esc(source)}</a>]</span>'
    note = _esc(source) if ("(" in source or ")" in source) else f"({_esc(source)})"
    return f' <span class="k">{note}</span>'


def _key_terms_html(key_terms: Any, sources: Any = ()) -> str:
    paper_urls = _source_url_index(sources)
    items = []
    for term in key_terms or []:
        if not isinstance(term, Mapping):
            continue
        label = _esc(term.get("term", ""))
        # : a verified deeper-dive ``link`` (resolved DRIVER-side by wiki_links) makes the
        # term itself clickable; ``link_kind`` rides as the tooltip. A ``familiar_gloss`` (the
        # reader-translation seam's faithful phrasing in the reader's own vocabulary) appends
        # after the plain definition. Entries without these fields render byte-identically.
        link = str(term.get("link", "") or "")
        if link:
            link_kind = _esc(term.get("link_kind", ""))
            head = f'<b><a href="{_esc(link)}" title="{link_kind}">{label}</a></b>'
        else:
            head = f"<b>{label}</b>"
        definition = _esc(
            term.get("plain_meaning", "")
            or term.get("universal_definition", "")
            or term.get("definition", "")
        )
        gloss = f" — {definition}" if definition else ""
        familiar = str(term.get("familiar_gloss", "") or "")
        fam = f'<span class="k"> — in your terms: {_esc(familiar)}</span>' if familiar else ""
        cite = _cite_html(str(term.get("source", "") or ""), paper_urls)
        items.append(f"<li>{head}{gloss}{fam}{cite}</li>")
    return f"<ul>{''.join(items)}</ul>" if items else ""


def _next_steps_html(next_steps: Any) -> str:
    """Render the ordered, concrete path a reader could follow to test
    the idea (flat plain strings from the elaboration card, in doing order). Defensively truncated
    at 6 items; empty/absent -> "" (legacy cards render byte-identically)."""
    items = [
        f"<li>{_esc(step)}</li>"
        for step in next_steps or []
        if isinstance(step, str) and step.strip()
    ][:6]
    if not items:
        return ""
    return f"<p><b>What to do next:</b></p><ol>{''.join(items)}</ol>"


def _translations_html(translations: Any, home_field: str = "") -> str:
    """ reader-translation panel: the audited ``{original -> familiar}`` record, marked
    ``[confirmed]`` / ``[≈ approximate]`` with the translator's rationale; a rejected pair
    renders as "kept original" (the failed analogy phrasing never ships). Empty/absent -> ""
    (cards without translations render byte-identically); an unknown home field falls back to
    the generic "this reader" summary."""
    items: list[str] = []
    for record in translations or []:
        if not isinstance(record, Mapping):
            continue
        original = str(record.get("original", "") or "")
        if not original:
            continue
        if str(record.get("faithfulness", "") or "") == "rejected":
            items.append(
                f"<li><b>{_esc(original)}</b> — kept original — proposed analogy failed "
                "the faithfulness check</li>"
            )
            continue
        mark = "[confirmed]" if record.get("faithfulness") == "confirmed" else "[≈ approximate]"
        rationale = str(record.get("rationale", "") or "")
        note = f' <span class="k">{_esc(rationale)}</span>' if rationale else ""
        items.append(
            f"<li><b>{_esc(original)}</b> &rarr; {_esc(record.get('familiar', ''))} "
            f'<span class="k">{mark}</span>{note}</li>'
        )
    if not items:
        return ""
    reader = f"a {_esc(home_field)} reader" if home_field else "this reader"
    return (
        f"<details><summary>Translated for {reader}</summary>"
        f"<ul>{''.join(items)}</ul></details>"
    )


def _plain_language_card_html(
    elaboration: Mapping[str, Any], sources: Any = (), home_field: str = "",
    *, include_headline: bool = True,
) -> str:
    """Render the current reader-card schema, with the old prose schema as a compatibility fallback."""
    headline = str(elaboration.get("headline", "") or elaboration.get("plain_sentence", "")).strip()
    if not headline:
        return ""
    sections = [
        ("What might be happening", elaboration.get("what_might_be_happening")),
        ("Why it matters", elaboration.get("why_it_matters")),
        ("How to check", elaboration.get("how_to_check")),
        ("What would change our mind", elaboration.get("what_would_change_our_mind")),
        ("Status", elaboration.get("status")),
        ("What problem this solves", elaboration.get("problem_solved")),
    ]
    rows = [f"<p><b>{_esc(label)}:</b> {_esc(value)}</p>" for label, value in sections if value]
    if not rows and elaboration.get("elaboration"):
        rows.append(f"<p>{_esc(elaboration.get('elaboration', ''))}</p>")
    lead = f"<p><b>{_esc(headline)}</b></p>" if include_headline else ""
    return (
        "<section>"
        "<h2>Possible explanation</h2>"
        f"{lead}"
        f"{''.join(rows)}"
        f"{_next_steps_html(elaboration.get('next_steps'))}"
        f"{_key_terms_html(elaboration.get('key_terms'), sources)}"
        f"{_translations_html(elaboration.get('translations'), home_field)}"
        "</section>"
    )


def _sources_html(sources: Any) -> str:
    """The retrieved references as a linked list (so the trace links the sources it drew on). Each
    source is a mapping with ``title`` + optional ``url`` (+ ``source`` name). Only http(s) urls are
    linkified; a source without a url renders as plain text. Deduped by url-or-title; empty when none."""
    items: list[str] = []
    seen: set[str] = set()
    for src in sources or []:
        if not isinstance(src, Mapping):
            continue
        title = str(src.get("title", "") or "").strip()
        if not title:
            continue
        url = str(src.get("url", "") or "").strip()
        key = url or title
        if key in seen:
            continue
        seen.add(key)
        name = str(src.get("source", "") or "").strip()
        label = (
            f'<a href="{_esc(url)}">{_esc(title)}</a>'
            if url.startswith(("http://", "https://"))
            else _esc(title)
        )
        tag = f' <span class="k">({_esc(name)})</span>' if name else ""
        items.append(f"<li>{label}{tag}</li>")
    return f"<h2>Sources</h2><ol>{''.join(items)}</ol>" if items else ""


def _lineage_html(lineage: Mapping[str, Any] | None) -> str:
    """Render a collapsible derivation-lineage panel with parents,
    strategy, round, grounding status. Empty for proposer candidates / when evolution did not run."""
    if not lineage:
        return ""
    parents = lineage.get("wasDerivedFrom") or lineage.get("derived_from") or []
    parts = [
        f"derived from {' & '.join(str(p) for p in parents)}" if parents else "proposer candidate (no parent)"
    ]
    if lineage.get("strategy"):
        parts.append(f"strategy: {lineage['strategy']}")
    if lineage.get("round") is not None:
        parts.append(f"round {lineage['round']}")
    if lineage.get("grounding"):
        parts.append(f"grounding: {lineage['grounding']}")
    if lineage.get("agent"):
        parts.append(f"by: {lineage['agent']}")
    lis = "".join(f"<li>{_esc(part)}</li>" for part in parts)
    return f"<details><summary>Lineage</summary><ul>{lis}</ul></details>"


def _claim_html(claim: str | None, seed_kind: str | None = None) -> str:
    """The run's seed claim (the researcher's, from the profile), shown as context."""
    label = SEED_LABELS.get(seed_kind or "claim", "Starting input")
    return f'<p class=k>{label}: {_esc(claim)}</p>' if claim else ""


def _mechanism_steps_html(steps: Any) -> str:
    """Render the verified concept-to-relation-to-concept chain.
    Verdict badges render ONLY for residual ``vague``/``false`` steps (the honest warnings);
    sound/unaudited steps render clean. Empty/absent -> "" (pre-feature pages byte-identical)."""
    items: list[str] = []
    for step in steps or []:
        if not isinstance(step, Mapping):
            continue
        arrow = f"{step.get('from', '')} --{normalize_relation_label(step.get('relation', ''))}--> {step.get('to', '')}"
        mech = step.get("mechanism", "")
        mech_html = f' <span class="k">{_esc(mech)}</span>' if mech else ""
        verdict = str(step.get("verdict", "") or "")
        badge = ""
        if verdict in ("vague", "false"):
            note = str(step.get("note", "") or "")
            badge = f' <b>&#9888; [{_esc(verdict)}]</b>'
            if note:
                badge += f' <span class="k">{_esc(note)}</span>'
        items.append(f"<li>{_esc(arrow)}{mech_html}{badge}</li>")
    if not items:
        return ""
    return f"<h2>How the pieces connect</h2><ol>{''.join(items)}</ol>"


def _term_warnings_html(term_audit: Any, sources: Any = ()) -> str:
    """Render residual terminology warnings.

    Only ``misused`` and ``stretched`` verdicts render
    as warnings (with the ORIGINAL source quote + linkified title, so the reader can check),
    plus an info line for ``no_source`` terms. ``consistent`` terms and empty/absent audits
    render nothing (pre-feature pages byte-identical)."""
    paper_urls = _source_url_index(sources)
    items: list[str] = []
    for verdict in term_audit or []:
        if not isinstance(verdict, Mapping):
            continue
        term = str(verdict.get("term", "") or "")
        grade = str(verdict.get("verdict", "") or "")
        if not term:
            continue
        if grade in ("misused", "stretched"):
            quote = str(verdict.get("source_quote", "") or "")
            title = str(verdict.get("source_title", "") or "")
            usage = str(verdict.get("usage", "") or "")
            note = str(verdict.get("note", "") or "")
            parts = [f"<b>&#9888; '{_esc(term)}'</b> <span class=k>[{_esc(grade)}]</span>"]
            if quote:
                cite = _cite_html(title, paper_urls) if title else ""
                parts.append(
                    f" &mdash; the source uses this as: &ldquo;{_esc(quote)}&rdquo;{cite}"
                )
            if usage:
                parts.append(f"; this hypothesis: &ldquo;{_esc(usage)}&rdquo;")
            if note:
                parts.append(f' <span class="k">{_esc(note)}</span>')
            items.append(f"<li>{''.join(parts)}</li>")
        elif grade == "no_source":
            items.append(
                f"<li><span class=k>'{_esc(term)}' &mdash; no original source occurrence "
                "found; its meaning here is unverified.</span></li>"
            )
    if not items:
        return ""
    return f"<h2>Term check</h2><ul>{''.join(items)}</ul>"


def _problem_html(row: Any) -> str:
    """"What problem this solves" from the scaffold's ``problem`` key — rendered ONLY when the row
    carries Critic Panel mechanism-step / scaffold-gap grades (the Session-C report sink; empty until wired),
    so rows without them stay byte-identical. A residual missing/thin problem renders
    as an explicit warning instead of silence."""
    steps = getattr(row, "mechanism_steps", ()) or ()
    gaps = tuple(getattr(row, "scaffold_gaps", ()) or ())
    if not steps and not gaps:
        return ""
    problem = str((getattr(row, "idea_scaffold", {}) or {}).get("problem", "") or "").strip()
    warn = (
        '<p class=k>&#9888; No explicit problem statement survived the audit.</p>'
        if any(g.startswith("problem:") for g in gaps)
        else ""
    )
    text = f"<p>{_esc(problem)}</p>" if problem else ""
    if not text and not warn:
        return ""
    return f"<h2>What problem this solves</h2>{text}{warn}"


def render_hypothesis_html(
    surfaced: Any,
    *,
    claim: str | None = None,
    elaboration: Mapping[str, Any] | None = None,
    sources: Sequence[Any] = (),
    lineage: Mapping[str, Any] | None = None,
    reader_context: Mapping[str, Any] | None = None,
    seed_kind: str | None = None,
) -> str:
    """One per-hypothesis page. Every value is copied from the surfaced row — faithful by
    construction. ``claim`` (the run's seed claim) is shown as context; ``elaboration`` (when
    present) adds the plain-language hero; absent, the page is still complete (scores + structure).
    ``lineage`` adds the collapsible Research Synthesist derivation panel. ``reader_context``
    supplies the ``home_field`` named in the card's translations-panel summary."""
    fields = (
        ("rank", surfaced.rank),
        ("candidate", surfaced.candidate_id),
        ("field novelty", f"{surfaced.field_novelty:.2f}"),
        ("saturation", f"{surfaced.saturation:.2f}"),
        ("HypScore", f"{surfaced.hyp_score:.3f}"),
        ("RankScore", f"{surfaced.rank_score:.3f}"),
        ("cross-concept", surfaced.cross_concept),
        ("common-sense", surfaced.common_sense),
        ("new concept(s)", ", ".join(surfaced.new_node_labels)),
    )
    rows = "".join(f"<tr><td class=k>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in fields)
    home_field = str((reader_context or {}).get("home_field", "") or "")
    hero = _plain_language_card_html(
        elaboration, sources, home_field, include_headline=False
    ) if elaboration else ""
    headline = str((elaboration or {}).get("headline") or (elaboration or {}).get("plain_sentence") or "")
    lead = f"<p><b>{_esc(headline)}</b></p>" if headline else ""
    rationale = f"<p class=k>{_esc(surfaced.rationale)}</p>" if surfaced.rationale else ""
    body = (
        '<p class=k><a href="index.html">&larr; all hypotheses &amp; sources</a></p>'
        f"{_claim_html(claim, seed_kind)}"
        f"<h1>Proposed hypothesis {_esc(surfaced.candidate_id)}</h1>"
        f"{lead}"
        f"{relationship_section(elaboration, seed_kind) if claim else ''}"
        f"{hero}"
        # Critic Panel-graded sections are empty unless grades are attached to the row.
        f"{_problem_html(surfaced)}"
        f"{_mechanism_steps_html(getattr(surfaced, 'mechanism_steps', ()))}"
        f"{_term_warnings_html(getattr(surfaced, 'term_audit', ()), sources)}"
        f"<table>{rows}</table>"
        f"{rationale}"
        f"{_lineage_html(lineage)}"
        f"{_sources_html(sources)}"
    )
    return _page(f"Hypothesis {surfaced.candidate_id}", body)


def render_index_html(
    result: Any, *, claim: str | None = None, sources: Sequence[Any] = (),
    seed_kind: str | None = None,
) -> str:
    """The ranking index: every surfaced hypothesis with its discriminating scores, linked out, plus
    a linked ``Sources`` list of the references the run retrieved (omitted when ``sources`` is empty).
    ``claim`` (the run's seed claim, from the profile) is shown under the title when supplied."""
    header = (
        "<tr><th>rank</th><th>id</th><th>field novelty</th><th>saturation</th>"
        "<th>HypScore</th><th>RankScore</th></tr>"
    )
    rows = "".join(
        f'<tr><td>{row.rank}</td>'
        f'<td><a href="{_esc(row.candidate_id)}.html">{_esc(row.candidate_id)}</a></td>'
        f"<td>{row.field_novelty:.2f}</td><td>{row.saturation:.2f}</td>"
        f"<td>{row.hyp_score:.3f}</td><td>{row.rank_score:.3f}</td></tr>"
        for row in result.surfaced
    )
    body = (
        f"<h1>Research Synthesist hypotheses — graph {_esc(result.graph_id)} "
        f"version {_esc(result.version)}</h1>"
        f"{_claim_html(claim, seed_kind)}"
        f"<p class=k>{len(result.surfaced)} surfaced</p>"
        f"<table>{header}{rows}</table>"
        f"{_sources_html(sources)}"
    )
    return _page("Research Synthesist hypotheses", body)


def write_trace(
    result: Any,
    out_dir: str | Path,
    *,
    claim: str | None = None,
    elaborations: Mapping[str, Mapping[str, Any]] | None = None,
    sources: Sequence[Any] = (),
    reader_context: Mapping[str, Any] | None = None,
    seed_kind: str | None = None,
) -> list[Path]:
    """Write ``index.html`` + one ``<candidate_id>.html`` per surfaced hypothesis. ``claim`` (the
    run's seed claim, from the profile) is shown on every page. ``elaborations`` (cid -> prose dict)
    is optional; without it the structural pages render complete. ``sources`` (the retrieved
    references) are linked on the index when supplied. ``reader_context``
    ``{home_field, lexicon_source}``, set by the driver when a reader lexicon steered the
    translation seam) names the reader on the pages and rides the sidecar under the additive
    top-level ``"reader"`` key; None leaves pages and the sidecar unchanged. ``seed_kind``
    (the profile's ``claim | research_goal | raw_message``) rides the sidecar
    under the additive top-level ``"seed_kind"`` key, next to ``"claim"``, so an offline renderer
    can label the seed by its own kind; None -> sidecar byte-identical to today's.

    Also persists ``plain_language.json`` (the claim + per-hypothesis prose) so a later renderer
    (e.g. the connected-graph views) reuses the same plain-language layer without re-calling an LLM."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    elaborations = elaborations or {}
    written = [out / "index.html"]
    written[0].write_text(
        render_index_html(result, claim=claim, sources=sources, seed_kind=seed_kind),
        encoding="utf-8",
    )
    for row in result.surfaced:
        page = out / f"{row.candidate_id}.html"
        page.write_text(
            render_hypothesis_html(
                row, claim=claim, elaboration=elaborations.get(row.candidate_id), sources=sources,
                lineage=getattr(row, "lineage", None),  # None for an original proposal.
                reader_context=reader_context,
                seed_kind=seed_kind,
            ),
            encoding="utf-8",
        )
        written.append(page)
    sidecar = out / "plain_language.json"
    payload: dict[str, Any] = {
        "claim": claim or "",
        "hypotheses": {cid: dict(e) for cid, e in elaborations.items()},
    }
    if seed_kind:
        payload["seed_kind"] = seed_kind
    # Audit records ride under an additive top-level key so
    # offline renderers (connected views, dropped candidates) can reuse them without the run.
    audits: dict[str, Any] = {}
    for row in result.surfaced:
        entry: dict[str, Any] = {}
        for key in ("mechanism_steps", "scaffold_gaps", "term_audit"):
            value = getattr(row, key, ()) or ()
            if value:
                entry[key] = list(value)
        if entry:
            audits[row.candidate_id] = entry
    if audits:
        payload["audits"] = audits
    # Each confirmed hypothesis's committed edge IDs ride under an additive top-level key so an
    # offline renderer can bind its experiment plan by exact edge ID
    # instead of focus-node adjacency; absent -> byte-identical (old sidecar schema, or no
    # confirmed edges).
    edge_ids: dict[str, Any] = {}
    for row in result.surfaced:
        ids = getattr(row, "hypothesis_edge_ids", ()) or ()
        if ids:
            edge_ids[row.candidate_id] = list(ids)
    if edge_ids:
        payload["hypothesis_edge_ids"] = edge_ids
    # Per-hypothesis experiment outcomes are report-side audit data.  They make
    # an omitted plan explainable without putting failed attempts into durable
    # graph state.  Older runs omit this additive block.
    experiment_attempts = {
        row.candidate_id: dict(row.experiment_attempt)
        for row in result.surfaced
        if getattr(row, "experiment_attempt", None)
    }
    if experiment_attempts:
        payload["experiment_attempts"] = experiment_attempts
    # : the reader block ({home_field, lexicon_source}) rides the sidecar under an ADDITIVE
    # top-level key so the offline renderers can name the reader; absent -> byte-identical.
    if reader_context:
        payload["reader"] = dict(reader_context)
    sidecar.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    written.append(sidecar)
    return written
