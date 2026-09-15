"""``scripts/render_runtime_logs.py`` must escape untrusted trace content.

A runtime trace carries prompts and provider responses (README documents this),
so the script renders text this project never authored into an HTML page a person
opens in a browser. The deleted ``src/artifact_view.py`` had the repo's only
regression test for that escaping; README now points at this script instead, and
it had none.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path("scripts/render_runtime_logs.py").resolve()

# Every field below is attacker-controllable: an actor name, a payload string,
# and an artifact path all originate outside this project.
INJECTION = "<script>alert('xss')</script>"


def _write_run(root: Path, name: str, *, label: str) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"label": label, "pid": 4242, "created_at": "2026-09-01T00:00:00Z"}),
        encoding="utf-8",
    )
    events = [
        {
            "timestamp": "2026-09-01T00:00:01Z",
            "event_type": "agent_with_api",
            "direction": "request",
            "actor": f"builder {INJECTION}",
            "payload": {"prompt": f"summarise {INJECTION}"},
        },
        {
            "timestamp": "2026-09-01T00:00:02Z",
            "event_type": "artifact",
            "direction": "write",
            "actor": "exporter",
            "artifact_path": f"/tmp/{INJECTION}.html",
            "payload": {"event_type": "state_update", "sender_role": INJECTION},
        },
    ]
    with (run_dir / "trace.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")
        handle.write("\n")            # blank line: the loader must skip it
        handle.write("{not json}\n")  # malformed line: skipped with a warning, not a crash
    return run_dir


@pytest.fixture
def rendered(tmp_path: Path) -> tuple[Path, list[Path]]:
    first = _write_run(tmp_path, "run-a", label=f"label {INJECTION}")
    second = _write_run(tmp_path, "run-b", label="plain label")
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )
    assert completed.returncode == 0, f"{completed.stdout}\n{completed.stderr}"
    return tmp_path, [first, second]


def test_render_runtime_logs_writes_a_report_per_run_and_one_index(rendered) -> None:
    root, run_dirs = rendered

    assert (root / "index.html").is_file()
    for run_dir in run_dirs:
        assert (run_dir / "report.html").is_file()


def test_render_runtime_logs_escapes_untrusted_trace_content(rendered) -> None:
    root, run_dirs = rendered
    pages = [root / "index.html"] + [d / "report.html" for d in run_dirs]

    for page in pages:
        text = page.read_text(encoding="utf-8")
        # the injected markup must never survive as a live tag...
        assert INJECTION not in text, page
        assert "<script>alert" not in text, page
    # ...but the escaped text is still shown, so escaping is not silent dropping
    report = (run_dirs[0] / "report.html").read_text(encoding="utf-8")
    assert "&lt;script&gt;" in report
