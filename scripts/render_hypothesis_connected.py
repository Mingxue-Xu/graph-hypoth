"""CLI wrapper for the connected-hypothesis renderer.

Renderer logic (extraction criteria, visual theme, plan/evidence traceback, and page assembly)
lives in ``src.connected_render``. This script only parses CLI arguments and calls
``render_connected_hypotheses()``. See that module's docstring for the inputs, outputs, and
determinism guarantees.

Usage:
  python scripts/render_hypothesis_connected.py --run-dir out/run-example
  python scripts/render_hypothesis_connected.py --run-dir out/run-example --only h1
  python scripts/render_hypothesis_connected.py            # auto-pick newest out/*/ with a graph.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Bootstrap the repository root so the package is importable when this file is
# invoked directly instead of through an installed console entry point.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.connected_render import (  # noqa: E402
    _experiment_core_summary as _experiment_core_summary,
    _key_terms_html as _key_terms_html,
    _plain_design_text as _plain_design_text,
    _plan_evidence_refs as _plan_evidence_refs,
    _translations_html as _translations_html,
    experiment_plan_section as experiment_plan_section,
    hypothesis_prose as hypothesis_prose,
    pick_run_dir,
    render_connected_hypotheses,
    render_page as render_page,
    scores_section as scores_section,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Render per-hypothesis connected-component graph views."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="run directory holding graph.json + trace/ (default: newest out/*/)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="where to write the HTML (default: the run dir itself)",
    )
    parser.add_argument(
        "--events-db",
        type=Path,
        default=None,
        help="SQLiteLogStore events DB for source links (default: auto-select)",
    )
    parser.add_argument(
        "--only", default=None, help="render a single candidate id (e.g. h3)"
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="research profile YAML; its claim is shown on each page",
    )
    parser.add_argument(
        "--claim",
        default=None,
        help="claim text to show (overrides --profile and the sidecar)",
    )
    parser.add_argument(
        "--lens",
        default=None,
        help="research lens/anchor to show as the banner for a claimless "
        "run; labelled 'Research lens' and used only when no claim is set",
    )
    args = parser.parse_args(argv)

    render_connected_hypotheses(
        args.run_dir or pick_run_dir(),
        out_dir=args.out_dir,
        events_db=args.events_db,
        claim=args.claim,
        lens=args.lens,
        profile=args.profile,
        only=args.only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
