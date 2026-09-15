from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DIAGRAM = _REPO_ROOT / "docs/architecture-design/overview/agent-role-relationship.mmd"
_README = _REPO_ROOT / "README.md"


def _readme_mermaid_blocks() -> list[str]:
    lines = _README.read_text(encoding="utf-8").split("\n")
    blocks, current = [], None
    for line in lines:
        if current is None and line == "```mermaid":
            current = []
        elif current is not None and line == "```":
            blocks.append("\n".join(current))
            current = None
        elif current is not None:
            current.append(line)
    return blocks


def test_readme_mermaid_block_matches_the_diagram_source() -> None:
    """The README embeds the diagram; the .mmd is the source of truth.

    GitHub renders the fenced block, not the .mmd file, so the two are a copy
    of each other and will silently diverge the first time someone edits one.
    """
    blocks = _readme_mermaid_blocks()
    assert len(blocks) == 1, f"expected exactly one mermaid block, found {len(blocks)}"
    assert blocks[0] == _DIAGRAM.read_text(encoding="utf-8").rstrip("\n"), (
        "README.md's mermaid block and "
        "docs/architecture-design/overview/agent-role-relationship.mmd have diverged. "
        "Edit the .mmd, then copy it into the README fence verbatim."
    )


def test_diagram_declares_its_type_before_any_comment() -> None:
    """Mermaid 11 fails to parse a diagram whose first line is a `%%` comment.

    A leading comment block raises `Parse error on line 1` and GitHub renders a
    syntax-error box instead of the diagram, so the declaration must come first.
    Bare `%%` lines are also parsed as a node and must carry text.
    """
    lines = _DIAGRAM.read_text(encoding="utf-8").split("\n")
    first = next(line for line in lines if line.strip())
    assert first.startswith("flowchart "), (
        f"the diagram must open with its type declaration, got {first!r}"
    )
    assert not [line for line in lines if line.strip() == "%%"], (
        "a bare `%%` line renders as an empty node; give every comment text"
    )


def test_diagram_describes_the_current_seven_role_pipeline() -> None:
    """`critic_controller` and the legacy CAMEL path are not part of the seven roles."""
    text = _DIAGRAM.read_text(encoding="utf-8")
    for role in (
        "Extraction Agent",
        "Evidence Reviewer",
        "Research Synthesist",
        "Critic Panel",
        "Experiment Designer",
        "Experiment Validator",
        "Narrative Elaborator",
    ):
        assert role in text, f"workflow role missing from the diagram: {role}"
    assert "builder" in text and "skeptical_verifier" in text
    assert "critic_controller" not in text, (
        "critic_controller is not one of the seven workflow roles"
    )
