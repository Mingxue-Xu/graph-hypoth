from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.events import EventType
from src.log_store import SQLiteLogStore
from tests.e2e.conftest import require_live_model_key


pytestmark = pytest.mark.e2e


def _pythonpath_env() -> dict[str, str]:
    env = os.environ.copy()
    project_path = str(Path.cwd())
    env["PYTHONPATH"] = (
        project_path
        if not env.get("PYTHONPATH")
        else f"{project_path}{os.pathsep}{env['PYTHONPATH']}"
    )
    return env


def _write_fake_codex(executable: Path) -> None:
    executable.write_text(
        f"""#!{sys.executable}
import json
import pathlib
import re
import sys

args = sys.argv[1:]
output_path = pathlib.Path(args[args.index("--output-last-message") + 1])
_backend_prompt, request_json = sys.stdin.read().split(
    "\\n\\nChat-completion request JSON follows:\\n", maxsplit=1
)
request = json.loads(request_json)
system_prompt = request["messages"][0]["content"]
user_prompt = request["messages"][1]["content"]

if "extract a causal-claim graph" in system_prompt:
    completion = {{
        "nodes": [
            {{
                "label": "smoking",
                "type": "exposure/intervention",
                "definition": "tobacco smoking",
                "aliases": [],
            }},
            {{
                "label": "lung cancer",
                "type": "outcome",
                "definition": "lung cancer incidence",
                "aliases": [],
            }},
        ],
        "edges": [
            {{
                "source": "smoking",
                "target": "lung cancer",
                "direction": "causal",
                "relation_type": "increases",
                "mechanism": "carcinogen exposure",
            }}
        ],
        "assumptions": [],
    }}
elif "You are the evidence reviewer" in system_prompt:
    evidence_ids = re.findall(r"evidence_id: ([^\\n]+)", user_prompt)
    completion = {{
        "passages": [
            {{
                "evidence_id": evidence_id,
                "entailment": {{"supports": 1.0}},
                "contradiction_risk": {{}},
                "context_fit": 1.0,
                "construct_match": 1.0,
                "design_strength": 1.0,
                "measurement_validity": 1.0,
                "population_fit": 1.0,
                "confounder_adjustment": 1.0,
                "bias_risk": 0.0,
            }}
            for evidence_id in evidence_ids
        ],
        "open_risks": [],
        "qualifiers": [],
    }}
else:
    raise SystemExit("unexpected model role")

output_path.write_text(json.dumps(completion), encoding="utf-8")
print(json.dumps({{"type": "thread.started", "thread_id": "fake-thread"}}))
print(json.dumps({{
    "type": "turn.completed",
    "usage": {{"input_tokens": 7, "output_tokens": 3}},
}}))
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)


def test_cli_graph_state_path_is_hermetic_end_to_end(tmp_path: Path) -> None:
    fake_codex = tmp_path / "fake-codex"
    _write_fake_codex(fake_codex)
    shim_dir = tmp_path / "import-shims"
    shim_dir.mkdir()
    (shim_dir / "sentence_transformers.py").write_text(
        "raise ImportError('disabled for hermetic e2e')\n", encoding="utf-8"
    )

    model = {
        "provider": "codex-cli",
        "model_id": "fake-model",
        "api_key_env": None,
        "base_url": None,
        "reasoning_effort": "low",
        "timeout_seconds": 5,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "agents": {
                    "builder": {"temperature": 0.0, "model": model},
                    "skeptical_verifier": {"temperature": 0.0, "model": model},
                },
                "retrieval": {
                    "enabled": True,
                    "fake_mode": True,
                    "sources": [],
                    "optional_sources": [],
                    "final_top_k": 2,
                    "per_source_top_k": 2,
                    "artifacts": {"enabled": False},
                    "coherence": {
                        "enabled": False,
                        "embedding_enabled": False,
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    events_db = tmp_path / "events.sqlite"
    env = _pythonpath_env()
    env["PYTHONPATH"] = f"{shim_dir}{os.pathsep}{env['PYTHONPATH']}"
    env["GRAPH_HYPOTH_CODEX_BIN"] = str(fake_codex)
    env.pop("GRAPH_HYPOTH_RUNTIME_LOG_DIR", None)
    env.pop("GRAPH_HYPOTH_RUNTIME_RUN_LABEL", None)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli",
            "--run-id",
            "hermetic-cli",
            "--thread-id",
            "thread-hermetic-cli",
            "--claim",
            "Smoking increases lung cancer.",
            "--config",
            str(config_path),
            "--events-db",
            str(events_db),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    output = f"{completed.stdout}\n{completed.stderr}"
    assert completed.returncode == 0, output
    assert "run_id: hermetic-cli" in completed.stdout
    assert "thread_id: thread-hermetic-cli" in completed.stdout
    assert "graph_version:" in completed.stdout
    assert "verified_edges: 1/1" in completed.stdout
    assert "Traceback" not in output

    export_dir = tmp_path / "graph-state-hermetic-cli"
    graph = json.loads((export_dir / "graph.json").read_text(encoding="utf-8"))
    assert len(graph["nodes"]) == 2
    assert len(graph["edges"]) == 1
    assert (export_dir / "edge_table.csv").is_file()
    assert (export_dir / "audit_memo.md").is_file()

    audit_events = SQLiteLogStore(events_db).list_events("hermetic-cli")
    model_events = [
        event
        for event in audit_events
        if event.sender_role in {"builder", "evidence_reviewer"}
    ]
    assert {event.sender_role for event in model_events} == {
        "builder",
        "evidence_reviewer",
    }
    assert all(event.provider == "codex-cli" for event in model_events)
    assert all(event.sender_role != "critic_controller" for event in model_events)
    retrieval_events = [
        event for event in audit_events if event.event_type == EventType.TOOL_EVENT
    ]
    assert len(retrieval_events) == 1
    assert retrieval_events[0].sender_role == "retrieve_evidence"
    assert retrieval_events[0].structured_payload["tool_name"] == "search_papers"
    assert retrieval_events[0].structured_payload["query"] == (
        "Smoking increases lung cancer."
    )
    retrieval_payload = retrieval_events[0].structured_payload
    assert retrieval_events[0].tool_calls == [
        {
            "id": retrieval_payload["tool_call_id"],
            "name": "search_papers",
            "arguments": {
                "query": retrieval_payload["query"],
                "sources": retrieval_payload["sources"],
                "filters": retrieval_payload["filters"],
            },
        }
    ]


@pytest.mark.live
def test_live_cli_graph_state_path_exports_artifacts(tmp_path: Path) -> None:
    require_live_model_key()
    events_db = tmp_path / "graph-hypoth-live-graph-events.sqlite"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli",
            "--run-id",
            "live-cli-graph",
            "--thread-id",
            "thread-live-cli-graph",
            "--claim",
            "Regular physical activity reduces all-cause mortality in older adults.",
            "--events-db",
            str(events_db),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_pythonpath_env(),
    )

    output = f"{completed.stdout}\n{completed.stderr}"
    assert completed.returncode == 0, output
    assert "run_id: live-cli-graph" in completed.stdout
    assert "graph_version:" in completed.stdout
    assert "verified_edges:" in completed.stdout
    assert "Traceback" not in output

    export_dir = tmp_path / "graph-state-live-cli-graph"
    assert (export_dir / "graph.json").is_file()
    assert (export_dir / "edge_table.csv").is_file()
    assert (export_dir / "audit_memo.md").is_file()
