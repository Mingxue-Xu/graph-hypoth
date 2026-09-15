"""Run the complete pipeline with the example research profile and live backends.

Usage from the repository root, after configuring model IDs and credentials:
  python scripts/run_example.py --config runtime_artifacts/config-live.yaml
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main(argv: Sequence[str] | None = None) -> int:
    from src.progress import add_progress_arguments

    parser = argparse.ArgumentParser(description=__doc__)
    add_progress_arguments(parser)
    parser.add_argument(
        "--config", required=True, type=Path,
        help="system config with usable model IDs and retrieval settings",
    )
    parser.add_argument(
        "--profile", type=Path, default=REPO_ROOT / "examples/research_profile.yaml",
        help="research profile (default: examples/research_profile.yaml)",
    )
    parser.add_argument(
        "--run-dir", type=Path, default=REPO_ROOT / "runtime_artifacts/example",
        help="graph, trace pages, and events database (default: runtime_artifacts/example)",
    )
    parser.add_argument(
        "--elaborate", action=argparse.BooleanOptionalAction, default=True,
        help="write plain-language hypothesis prose using additional model calls (default: enabled)",
    )
    args = parser.parse_args(argv)
    for name in ("config", "profile"):
        if not getattr(args, name).is_file():
            parser.error(f"--{name} file does not exist: {getattr(args, name)}")

    # Use the production driver so every stage, including confirmation,
    # experiment design, and connected-page rendering, follows the normal CLI.
    from src.synthesist_run import main as synthesist_main

    run_dir = args.run_dir.resolve()
    command = [
        "--profile", str(args.profile.resolve()),
        "--config", str(args.config.resolve()),
        "--events-db", str(run_dir / "events.sqlite"),
        "--export-dir", str(run_dir),
        "--trace-dir", str(run_dir / "trace"),
    ]
    command.append("--elaborate" if args.elaborate else "--no-elaborate")
    if args.quiet:
        command.append("--quiet")
    if args.progress_jsonl is not None:
        command.extend(["--progress-jsonl", str(args.progress_jsonl.resolve())])
    result = synthesist_main(command)
    if result == 0:
        print(f"run artifacts: {run_dir}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
