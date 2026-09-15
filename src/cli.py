from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from src.camel_adapter import _preflight_model_credentials
from src.config import DEFAULT_CONFIG_PATH, load_config
from src.credentials import load_api_keys_file
from src.graph_state_runtime import GraphStateDeps, build_graph_state_deps
from src.log_store import SQLiteLogStore
from src.progress import ProgressReporter, add_progress_arguments, report_progress
from src.retrieval.preflight import preflight_required_retrieval
from src.run_path import GraphRunResult, run_graph_state_workflow
from src.runtime_trace import runtime_artifact_dir


@dataclass(frozen=True)
class OrchestrationArgs:
    claim: str
    config: Path = DEFAULT_CONFIG_PATH
    events_db: Path | None = None
    run_id: str | None = None
    thread_id: str | None = None
    api_keys_file: Path | None = None
    adapter: Literal["real", "fake"] = "real"


@dataclass(frozen=True)
class GraphStateRunResult:
    run_id: str
    thread_id: str
    result: GraphRunResult
    export_dir: Path | None


def _raise_or_exit(
    parser: argparse.ArgumentParser | None,
    exc: OSError | RuntimeError | ValueError,
) -> None:
    if parser is not None:
        parser.exit(1, f"error: {exc}\n")
    raise exc


def _preflight_retrieval_credentials(config) -> None:
    """CLI seam for the shared retrieval preflight."""

    preflight_required_retrieval(config)


def _default_events_db_path() -> Path:
    artifact_dir = runtime_artifact_dir()
    if artifact_dir is not None:
        return artifact_dir / "events.sqlite"
    return Path(".graph-hypoth-events.sqlite")


def _graph_state_export_dir(args: OrchestrationArgs, run_id: str) -> Path:
    base = runtime_artifact_dir()
    if base is None:
        base = args.events_db.parent if args.events_db else Path("runtime_artifacts")
    return base / f"graph-state-{run_id}"


def run_graph_state_orchestration(
    args: OrchestrationArgs,
    *,
    parser: argparse.ArgumentParser | None = None,
    deps: GraphStateDeps | None = None,
) -> GraphStateRunResult:
    """Run the claim-based graph-state workflow."""

    run_id = args.run_id or str(uuid4())
    thread_id = args.thread_id or run_id
    try:
        report_progress("Preparing run", "loading configuration and checking credentials")
        load_api_keys_file(args.api_keys_file)
        config = load_config(args.config)
    except (OSError, RuntimeError, ValueError) as exc:
        _raise_or_exit(parser, exc)

    # The only CLI switch for the deterministic, no-credential retrieval path.
    if args.adapter == "fake":
        config.retrieval.fake_mode = True

    if deps is None:
        try:
            _preflight_model_credentials(config)
            _preflight_retrieval_credentials(config)
            events_db = args.events_db or _default_events_db_path()
            log_store = SQLiteLogStore(events_db)
            log_store.setup()
            deps = build_graph_state_deps(
                config,
                claim=args.claim,
                log_store=log_store,
                run_id=run_id,
                thread_id=thread_id,
                log_llm_calls=True,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            _raise_or_exit(parser, exc)

    export_dir = _graph_state_export_dir(args, run_id)
    try:
        result = run_graph_state_workflow(
            args.claim,
            extractor=deps.extractor,
            embedder=deps.embedder,
            retrieve_for_target=deps.retrieve_for_target,
            evidence_reviewer=deps.evidence_reviewer,
            initial_open_risks=deps.retrieval_open_risks,
            open_risks_provider=deps.retrieval_open_risks_provider,
            export_dir=export_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        _raise_or_exit(parser, exc)
    return GraphStateRunResult(
        run_id=run_id,
        thread_id=thread_id,
        result=result,
        export_dir=export_dir,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run GraphHypoth's claim-based, multi-agent graph-state extraction "
            "and evidence review workflow."
        )
    )
    parser.add_argument("--claim", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, type=Path)
    parser.add_argument("--events-db", default=None, type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--thread-id")
    add_progress_arguments(parser)
    parser.add_argument(
        "--adapter",
        choices=("real", "fake"),
        default="real",
        help=(
            "Retrieval adapter. 'real' (default) uses the configured sources; "
            "'fake' forces retrieval.fake_mode, the deterministic "
            "no-credential retrieval path. Model provider keys are still required."
        ),
    )
    parser.add_argument(
        "--api-keys-file",
        default=None,
        type=Path,
        help="Optional local NAME=value file to load before credential preflight.",
    )
    args = parser.parse_args(argv)
    orchestration_args = OrchestrationArgs(
        claim=args.claim,
        config=args.config,
        events_db=args.events_db,
        run_id=args.run_id or str(uuid4()),
        thread_id=args.thread_id,
        api_keys_file=args.api_keys_file,
        adapter=args.adapter,
    )

    try:
        with ProgressReporter(
            orchestration_args.run_id,
            quiet=args.quiet, jsonl_path=args.progress_jsonl,
        ):
            result = run_graph_state_orchestration(orchestration_args, parser=parser)
    except OSError as exc:
        parser.exit(1, f"error: {exc}\n")
    _print_graph_state_result(result)


def _print_graph_state_result(result: GraphStateRunResult) -> None:
    graph_result = result.result
    verified = sum(1 for row in graph_result.edge_table if row.verdict)
    print(f"run_id: {result.run_id}")
    print(f"thread_id: {result.thread_id}")
    print(f"graph_version: {graph_result.version}")
    print(f"verified_edges: {verified}/{len(graph_result.edge_table)}")
    if result.export_dir is not None:
        print(f"export_dir: {result.export_dir}")
    print(graph_result.audit_memo)


if __name__ == "__main__":
    main()
