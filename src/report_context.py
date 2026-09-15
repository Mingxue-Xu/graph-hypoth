"""Reader-facing labels and the explanation connecting a proposal to its input."""

import html
from collections.abc import Mapping
from typing import Any

SEED_LABELS = {
    "claim": "Claim to investigate",
    "research_question": "Research question",
    "research_goal": "Research goal",
    "raw_message": "Starting input",
}


def relationship_section(card: Mapping[str, Any] | None, seed_kind: str | None) -> str:
    """Render a saved explanation; never infer scientific support from graph adjacency."""
    text = (card or {}).get("relationship_to_seed")
    if not isinstance(text, str) or not text.strip():
        return ""
    target = {"claim": "starting claim", "research_goal": "research goal"}.get(
        seed_kind, "starting input"
    )
    heading = (
        "How this hypothesis addresses the question" if seed_kind == "research_question"
        else f"Relationship to the {target}"
    )
    return (
        f'<section class="relationship"><h2>{heading}</h2>'
        f"<p>{html.escape(text.strip())}</p></section>"
    )
