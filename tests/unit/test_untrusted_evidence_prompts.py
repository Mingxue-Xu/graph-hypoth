"""Untrusted-input language across the prompt seams that read retrieved material.

Every prompt here is handed text the repo did not author — retrieved passages, source
quotes, corpus abstracts, web pages, user claims. This module pins down which of those
seams currently tell the model to treat that text as data rather than as instructions.

Two kinds of assertion live here, and the difference matters:

* ``test_web_search_prompts_declare_pages_untrusted`` guards protection that EXISTS.
  Deleting the clause from those prompts breaks the build. Since the legacy debate
  prompts were removed, the two web-retrieval agents are the ONLY seams in the repo
  that carry such a clause.
* ``test_graph_pipeline_prompts_have_no_untrusted_clause`` records a KNOWN GAP: the
  graph-pipeline prompts carry no such clause today. It is a characterization test, not
  an endorsement. When the clause is added to one of them, that test fails by design —
  move the constant into ``_SEAMS_WITH_UNTRUSTED_CLAUSE`` and update the public
  README or guide text that describes the affected seam in the same change.
"""

from __future__ import annotations

import pytest

from src.cycles.experiment import (
    _EXPERIMENT_DESIGNER_SYSTEM_PROMPT,
    _EXPERIMENT_VALIDATOR_SYSTEM_PROMPT,
)
from src.cycles.extraction import _EXTRACTOR_SYSTEM_PROMPT
from src.cycles.panel import CRITIC_PANEL_SYSTEM_PROMPT
from src.cycles.synthesist import SYNTHESIST_SYSTEM_PROMPT
from src.cycles.verification import _EVIDENCE_REVIEWER_SYSTEM_PROMPT
from src.retrieval.claude_web import _RESEARCH_PROMPT as _CLAUDE_WEB_PROMPT
from src.retrieval.codex_web import _RESEARCH_PROMPT as _CODEX_WEB_PROMPT

# Phrasing that tells the model supplied text is material to evaluate, not commands.
_UNTRUSTED_MARKERS = ("untrusted material to evaluate", "untrusted data, never as instructions")

# Graph-pipeline system prompts. None carries an untrusted-input clause today.
_GRAPH_PIPELINE_PROMPTS = {
    "CRITIC_PANEL_SYSTEM_PROMPT": CRITIC_PANEL_SYSTEM_PROMPT,
    "SYNTHESIST_SYSTEM_PROMPT": SYNTHESIST_SYSTEM_PROMPT,
    "_EVIDENCE_REVIEWER_SYSTEM_PROMPT": _EVIDENCE_REVIEWER_SYSTEM_PROMPT,
    "_EXPERIMENT_DESIGNER_SYSTEM_PROMPT": _EXPERIMENT_DESIGNER_SYSTEM_PROMPT,
    "_EXPERIMENT_VALIDATOR_SYSTEM_PROMPT": _EXPERIMENT_VALIDATOR_SYSTEM_PROMPT,
    "_EXTRACTOR_SYSTEM_PROMPT": _EXTRACTOR_SYSTEM_PROMPT,
}


def _has_untrusted_clause(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _UNTRUSTED_MARKERS)


# --------------------------------------------------------------------------------------
# Protection that exists — these must not regress.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "prompt"),
    [("claude_web", _CLAUDE_WEB_PROMPT), ("codex_web", _CODEX_WEB_PROMPT)],
)
def test_web_search_prompts_declare_pages_untrusted(name: str, prompt: str) -> None:
    """Web pages are the least trusted input in the system; both agents must say so."""
    assert _has_untrusted_clause(prompt), name


# --------------------------------------------------------------------------------------
# Known gap — characterization only. See the module docstring before changing.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_GRAPH_PIPELINE_PROMPTS))
def test_graph_pipeline_prompts_have_no_untrusted_clause(name: str) -> None:
    """Records that the graph-pipeline prompts do NOT bound prompt injection today.

    Retrieved passages, source quotes and graph labels reach these prompts without any
    instruction to treat them as data. If you are adding the clause: good — delete this
    prompt's entry from ``_GRAPH_PIPELINE_PROMPTS``, give it a positive assertion above,
    and update the public documentation that describes the seam.
    """
    assert not _has_untrusted_clause(_GRAPH_PIPELINE_PROMPTS[name])


# --------------------------------------------------------------------------------------
# Grounding properties the graph-pipeline prompts DO carry.
# --------------------------------------------------------------------------------------


def test_critic_panel_prompt_isolates_judges_from_proposer_internals() -> None:
    """The panel judges structure and quotes only — never the proposer's own reasoning."""
    assert "must not request" in CRITIC_PANEL_SYSTEM_PROMPT
    for hidden in ("rationale", "idea_scaffold", "llm_signals"):
        assert hidden in CRITIC_PANEL_SYSTEM_PROMPT
    assert "Judge only from the material provided." in CRITIC_PANEL_SYSTEM_PROMPT


def test_synthesist_prompt_binds_concepts_to_verbatim_passage_spans() -> None:
    """Mined concepts must carry provenance back to the passage they were read from."""
    assert "must come from the passage markers" in SYNTHESIST_SYSTEM_PROMPT
    assert "matched_quote_span is the verbatim phrase" in SYNTHESIST_SYSTEM_PROMPT


@pytest.mark.parametrize("name", sorted(_GRAPH_PIPELINE_PROMPTS))
def test_graph_pipeline_prompts_are_non_empty(name: str) -> None:
    assert _GRAPH_PIPELINE_PROMPTS[name].strip()
