# The local web app

`graph-hypoth-web` wraps the complete pipeline (the `graph-hypoth-synthesist`
driver) in a local web app. It exists for the two moments that need a person:
following a run that takes tens of minutes, and choosing which surfaced
hypotheses to commit. Both happened only in a terminal before. The pipeline's
gates, receipts, scoring, and exports are unchanged; the app calls the same
`run_synthesist` function the CLI calls, on a worker thread, and replaces the
stdin confirmation prompt with a hook that waits for the browser.

## Start it

Install the `web` extra (part of `all`) into the environment described in the
README, then run the server from the repository root so relative profile and
config paths resolve the same way they do for the CLI:

```bash
python -m pip install -e ".[web]"
graph-hypoth-web
```

The page opens at `http://127.0.0.1:8765/`. Options: `--port`, `--host`
(loopback by default; do not expose it, there is no authentication),
`--runs-root` (default `runtime_artifacts/web`), `--no-browser`, and `--quiet`
to keep the pipeline's progress lines off the server's terminal. The credentials
and backend configuration are the ones settings 2 and 3 already need; the app
adds nothing of its own.

To try the screens without credentials, run the mock instead: `python
scripts/web_demo.py` serves the same page over a fake pipeline that plays the
eight stages in about a minute, pauses for confirmation, and writes an invented
run (every source URL points at example.org) under `runtime_artifacts/web-demo`.

The demo GIF in the README is recorded from a real run with
`python scripts/record_web_demo.py live --out docs/assets/web-app-demo.gif`
(or `browse` for an existing run); rerun it after a visible UI change.

## Screens

**Start a run.** Pick a research profile and a system config (any YAML under
`examples/`, `config/`, or `runtime_artifacts/` is offered), and choose whether
to write plain-language prose and connected pages. The config must name real
models: `config/claude-code.yaml` and `config/codex.yaml` do, for a saved CLI
login, and the form defaults to the one whose CLI is installed. The shipped
`config/evidence-evaluation.yaml` is a template with `<MODEL_ID>` placeholders;
the form flags it and the launch is refused before any worker starts. Tick *ask me in the browser*
to decide the confirmation step yourself even when the profile's `confirm`
policy is automatic; a profile whose policy is `interactive` always asks in the
browser. The app runs one pipeline at a time.

**Progress.** The stage chips, the current activity, and the event log come from
the same `ProgressReporter` the CLI prints on stderr; the app writes them to
`progress.jsonl` in the run directory and the page polls that file. Pipeline
stdout lines, such as the final summary, appear under *Pipeline output*.

**Confirmation.** When the Research Synthesist's ranking is ready the run pauses
and the page shows one card per candidate: rank and scores, the coined concepts
with definitions, the proposed edges, the mechanism chain, rationale, scaffold,
assumptions, and grounding quotes. Tick the ones to commit, or use *Commit all*
or *Commit none*. The selection is applied exactly as the CLI would apply a
typed id list: only confirmed candidates become new `unverified` edges and go on
to experiment design.

**Results.** Links to the trace index, each connected page, the audit memo, the
edge table, and `graph.json`; a table of the surfaced hypotheses with their
plain-language headlines, committed edges, and experiment outcome, read from the
trace sidecar; and the claim graph on a pan-and-zoom canvas. Nodes are colored
and shaped by provenance (extracted, mined, proposed, priority-authored); edges
carry the verification status as color, dash pattern, and glyph. Concepts the
run mined but never joined to an edge sit in a grid under the connected part.
Click a node for its definition and source papers; click an edge for its
status, fused verdict, confidence, evidence quotes, open risks, and the
committed experiment plan. Filters by status and a concept search sit above the
canvas; the edge table below lists every edge and selects it on click.

**Existing runs.** Any directory with a `graph.json`, such as a CLI run, can be
opened from the run list without re-running. Nothing is written into it.

## Files

A web run lives in `runtime_artifacts/web/<run-id>/` and holds the same
artifacts as a CLI run (`graph.json`, `edge_table.csv`, `audit_memo.md`,
`trace/`, connected pages, `events.sqlite`) plus `progress.jsonl` and
`web_run.json`, the record the run list is built from. A run that was in
progress when the server stopped is listed as interrupted; there is no resume.
Treat these directories as private until reviewed, as the README says of every
export.

## API

The page is a static file; everything it shows comes from these routes, which
are also documented live at `/api/docs`.

| Route | Purpose |
| --- | --- |
| `GET /api/status` | Defaults, YAML choices, the active run id |
| `GET /api/runs`, `POST /api/runs` | List runs; start one from `profile_path`, `config_path`, and options |
| `POST /api/runs/import` | Register an existing run directory |
| `GET /api/runs/{id}` | The run record: status, options, summary, confirmation state |
| `GET /api/runs/{id}/events?after=N` | Progress rows after a cursor |
| `GET`/`POST /api/runs/{id}/confirmation` | The pending candidates; the selection (`candidate_ids` or `choice`) |
| `GET /api/runs/{id}/graph` | The canvas view-model built from the run's artifacts |
| `GET /api/runs/{id}/artifacts` | Files and named pages of the run |
| `GET /runs/{id}/files/{path}` | A file inside the run directory (paths outside it are refused) |

## Limits and troubleshooting

- One run at a time, and no cancel button: stop the server to abort a run.
- The canvas needs the `dagre` and `svg-pan-zoom` scripts from jsDelivr, the
  same CDN the connected pages use for Mermaid. Offline, the edge table and the
  details still work; only the drawing is skipped.
- `error: the web app needs the web extra` means FastAPI or uvicorn is missing
  from the environment; install the extra shown above.
- A run that fails shows the error in its header and keeps the full traceback
  in `error.txt` inside the run directory.
