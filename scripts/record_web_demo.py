"""Record the web app as an animated GIF, from a real run or from an existing run directory.

Drives a headless Chrome over the DevTools protocol against a running ``graph-hypoth-web`` server
and captures unedited screenshots; nothing is drawn over them.

* ``live`` starts a run through the page's own form with "ask me in the browser" ticked, captures a
  frame on each progress event (and a sparse one while a long model call is in flight), answers the
  confirmation step by ticking the top candidates, and tours the results when the run completes.
  A complete run takes tens of minutes and spends model quota.
* ``browse`` opens a finished run (``--run-id``) or imports a directory (``--import-dir``) and
  tours its results.

The browser is always the recorder's own: Chrome is launched with ``--remote-debugging-port=0`` in a
fresh profile and the port is read back from ``DevToolsActivePort``, and every frame checks that the
tab is still on the server being recorded. Needs Google Chrome and the ``websockets`` and ``Pillow``
packages (both arrive with the ``all`` extra).

Usage from the repository root, with the server already running:
  python scripts/record_web_demo.py live --out docs/assets/web-app-demo.gif
  python scripts/record_web_demo.py browse --import-dir runtime_artifacts/example --out demo.gif
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

DEFAULT_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
VIEWPORT = (1400, 980)
CLICK = "dispatchEvent(new MouseEvent('click', {bubbles: true}))"
MAX_RUN_SECONDS = 3 * 3600
MIN_EVENT_GAP = 4.0  # seconds between progress frames
HEARTBEAT_GAP = 75.0  # seconds between frames while one long call is in flight
MAX_PROGRESS_FRAMES = 160


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def wait_for_server(base: str, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            get_json(base + "api/status")
            return True
        except (OSError, ValueError):
            time.sleep(0.5)
    return False


class Recorder:
    """One DevTools session on our own tab, plus the frames captured so far."""

    def __init__(self, ws, base: str, frames_dir: Path | None) -> None:
        self.ws = ws
        self.base = base
        self.frames_dir = frames_dir
        self.next_id = 0
        self.frames: list[tuple[object, int, str]] = []

    async def call(self, method: str, **params):
        self.next_id += 1
        msg_id = self.next_id
        await self.ws.send(json.dumps({"id": msg_id, "method": method, "params": params}))
        while True:
            message = json.loads(await self.ws.recv())
            if message.get("id") == msg_id:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                return message.get("result", {})

    async def js(self, expression: str):
        result = await self.call(
            "Runtime.evaluate", expression=expression, awaitPromise=True, returnByValue=True
        )
        if "exceptionDetails" in result:
            text = result["exceptionDetails"].get("text", "js error")
            raise RuntimeError(f"{text} in {expression[:80]}")
        return result.get("result", {}).get("value")

    async def try_js(self, expression: str):
        try:
            return await self.js(expression)
        except RuntimeError as exc:
            print(f"  (skipped: {exc})", file=sys.stderr)
            return None

    async def frame(self, hold_ms: int, label: str, settle: float = 0.4) -> None:
        from PIL import Image

        await asyncio.sleep(settle)
        href = await self.js("location.href")
        if not str(href).startswith(self.base):
            raise RuntimeError(f"the tab left the recorded server: {href}")
        shot = await self.call("Page.captureScreenshot", format="png")
        image = Image.open(io.BytesIO(base64.b64decode(shot["data"]))).convert("RGB")
        self.frames.append((image, hold_ms, f"{label} @ {href}"))
        if self.frames_dir is not None:
            image.save(self.frames_dir / f"frame-{len(self.frames) - 1:03d}.png")

    async def navigate(self, url: str, wait: float) -> None:
        await self.call("Page.navigate", url=url)
        await asyncio.sleep(wait)

    async def wait_for(self, selector: str, timeout: float = 30.0) -> None:
        """Block until the page has rendered ``selector``; the page builds itself after its fetches."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if await self.js(f"document.querySelector({selector!r}) !== null"):
                return
            await asyncio.sleep(0.25)
        raise RuntimeError(f"{selector} did not appear within {timeout:.0f}s")


async def confirm_step(rec: Recorder, top: int) -> None:
    await rec.js("window.scrollTo(0, 0)")
    await rec.frame(2400, "confirmation cards", settle=1.0)
    await rec.js("window.scrollBy(0, 460)")
    await rec.frame(2000, "confirmation cards, scrolled")
    boxes = await rec.js("document.querySelectorAll('.cand input[type=checkbox]').length")
    for index in range(min(top, int(boxes or 0))):
        await rec.js(f"document.querySelectorAll('.cand input[type=checkbox]')[{index}].click()")
        await rec.frame(900, f"candidate {index + 1} ticked")
    commit = (
        "[...document.querySelectorAll('.confirm-actions button')]"
        ".find(b => b.textContent.startsWith('Commit selected'))"
    )
    await rec.js(f"{commit}.scrollIntoView({{block: 'center'}})")
    await rec.frame(900, "ready to commit")
    await rec.js(f"{commit}.click()")
    await rec.frame(1800, "selection committed", settle=1.0)


async def results_tour(rec: Recorder, run_id: str) -> None:
    await rec.wait_for(".linklist")
    try:
        await rec.wait_for("svg.graph", timeout=20.0)
    except RuntimeError:
        print("  (no graph canvas; touring without it)", file=sys.stderr)
    shown = await rec.js("(document.querySelector('#view code') || {}).textContent")
    if shown != run_id:
        raise RuntimeError(f"the page shows run {shown!r}, expected {run_id!r}")
    await rec.js("window.scrollTo(0, 0)")
    await rec.frame(2400, "run header", settle=0.8)
    await rec.try_js(
        "document.querySelector('.linklist').scrollIntoView({block: 'start'}); window.scrollBy(0, -90)"
    )
    await rec.frame(2800, "result links and surfaced hypotheses")
    await rec.try_js(
        "document.querySelector('.canvas-wrap').scrollIntoView({block: 'start'}); window.scrollBy(0, -150)"
    )
    await rec.frame(2800, "claim graph", settle=0.8)

    def click(selector: str) -> str:
        return (
            f"(() => {{ const e = document.querySelector({selector!r}); "
            f"if (!e) return false; e.{CLICK}; return true; }})()"
        )

    scroll_details = "(() => {{ const d = document.querySelector('.details'); d.scrollTop = {top}; }})()"
    if await rec.try_js(click("svg.graph .edge.hypothesis .hit-area")):
        await rec.frame(2600, "hypothesis edge")
        await rec.try_js(scroll_details.format(top=420))
        await rec.frame(2400, "experiment plan")
    for status in ("supported", "insufficient", "unverified"):
        if await rec.try_js(click(f"svg.graph .edge.st-{status}:not(.hypothesis) .hit-area")):
            await rec.frame(2600, f"{status} edge with its evidence")
            await rec.try_js(scroll_details.format(top=380))
            await rec.frame(2400, "evidence, scrolled")
            break
    if await rec.try_js(click("svg.graph .node.klass-mined")):
        await rec.frame(2400, "mined concept and its source")
    await rec.try_js(
        "(() => { const t = [...document.querySelectorAll('table.list')].pop(); "
        "t.scrollIntoView({block: 'start'}); window.scrollBy(0, -120); })()"
    )
    await rec.frame(2400, "edge table")
    connected = await rec.try_js(
        "(() => { const a = document.querySelector('.linklist a[href*=\"-connected.html\"]'); "
        "return a ? a.getAttribute('href') : null; })()"
    )
    if connected:
        await rec.navigate(rec.base.rstrip("/") + connected, wait=3.5)  # Mermaid renders from a CDN
        await rec.frame(2800, "connected hypothesis page")
        await rec.js("window.scrollBy(0, 700)")
        await rec.frame(2400, "connected hypothesis page, scrolled")
        await rec.navigate(f"{rec.base}#/runs/{run_id}", wait=3.0)
    await rec.js("window.scrollTo(0, 0)")
    await rec.frame(2600, "back on the run page")


async def record_live(rec: Recorder, args: argparse.Namespace) -> int:
    await rec.navigate(rec.base, wait=0.5)
    await rec.wait_for("form.launch input[name=confirm_in_browser]")
    if args.profile:
        await rec.js(f"document.querySelector('input[name=profile_path]').value = {args.profile!r}")
    if args.config:
        await rec.js(
            "(() => { const i = document.querySelector('input[name=config_path]'); "
            f"i.value = {args.config!r}; i.dispatchEvent(new Event('input')); }})()"
        )
    await rec.frame(1800, "home: the launch form")
    await rec.js("document.querySelector('input[name=confirm_in_browser]').click()")
    await rec.frame(1200, "ask me in the browser ticked")
    await rec.js("document.querySelector('form.launch button[type=submit]').click()")
    await asyncio.sleep(1.5)
    run_id = await rec.js("decodeURIComponent((location.hash.match(/^#\\/runs\\/(.+)$/) || [])[1] || '')")
    if not run_id:
        message = await rec.try_js("(document.querySelector('form.launch .error') || {}).textContent")
        print(f"no run started: {message}", file=sys.stderr)
        await rec.frame(3000, "launch refused")
        return 1
    print(f"run {run_id} started", flush=True)
    await rec.frame(1500, "the run page")
    api = f"{rec.base}api/runs/{run_id}"
    cursor = 0
    progress_frames = 0
    last_capture = started = time.monotonic()
    confirmed = False
    status = "running"
    while time.monotonic() - started < MAX_RUN_SECONDS:
        await asyncio.sleep(2.0)
        status = get_json(api)["status"]
        page = get_json(f"{api}/events?after={cursor}")
        cursor = page["next"]
        fresh = [e for e in page["events"] if e.get("event") != "heartbeat"]
        beats = [e for e in page["events"] if e.get("event") == "heartbeat"]
        if status == "waiting_confirmation" and not confirmed:
            await confirm_step(rec, args.confirm_top)
            confirmed = True
            last_capture = time.monotonic()
            continue
        if status in ("completed", "failed", "interrupted"):
            break
        now = time.monotonic()
        gap = MIN_EVENT_GAP * (1 + progress_frames // 60)  # thin the frames out on very long runs
        due = (fresh and now - last_capture >= gap) or (beats and now - last_capture >= HEARTBEAT_GAP)
        if due and progress_frames < MAX_PROGRESS_FRAMES:
            await rec.js("window.scrollTo(0, 0)")
            await rec.frame(450 if not fresh else 650, "progress", settle=0.3)
            progress_frames += 1
            last_capture = time.monotonic()
    minutes = (time.monotonic() - started) / 60
    print(f"run {run_id} {status} after {minutes:.1f} min; {len(rec.frames)} frames", flush=True)
    if status != "completed":
        await rec.js("window.scrollTo(0, 0)")
        await rec.frame(3000, f"run {status}", settle=1.0)
        return 2
    await asyncio.sleep(4.0)  # artifacts, graph fetch, canvas layout
    await results_tour(rec, run_id)
    return 0


async def record_browse(rec: Recorder, args: argparse.Namespace) -> int:
    run_id = args.run_id
    if args.import_dir:
        await rec.navigate(rec.base, wait=0.5)
        await rec.wait_for(".inline-form input")
        await rec.frame(1800, "home")
        await rec.js("document.querySelector('.inline-form input').focus()")
        for character in args.import_dir:
            await rec.call("Input.insertText", text=character)
            await asyncio.sleep(0.02)
        await rec.frame(1200, "run directory typed")
        await rec.js("document.querySelector('.inline-form button').click()")
        await asyncio.sleep(4.0)
        run_id = await rec.js("decodeURIComponent((location.hash.match(/^#\\/runs\\/(.+)$/) || [])[1] || '')")
    else:
        await rec.navigate(f"{rec.base}#/runs/{run_id}", wait=4.0)
    if not run_id:
        print("no run to browse", file=sys.stderr)
        return 1
    await results_tour(rec, run_id)
    return 0


def write_gif(frames, out: Path, width: int) -> None:
    from PIL import Image

    images = []
    for image, _, _ in frames:
        resized = image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)
        images.append(
            resized.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        out, save_all=True, append_images=images[1:], duration=[hold for _, hold, _ in frames],
        loop=0, optimize=False,
    )
    seconds = sum(hold for _, hold, _ in frames) / 1000
    print(f"{len(frames)} frames, {seconds:.0f}s per loop -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


def launch_chrome(chrome_path: str) -> tuple[subprocess.Popen, Path, int]:
    """Start our own headless Chrome and return it with its profile directory and DevTools port.

    Port 0 lets Chrome choose a free port, which it publishes in ``DevToolsActivePort`` inside the
    fresh profile, so the recorder can never attach to some other headless browser on the machine.
    """
    profile_dir = Path(tempfile.mkdtemp(prefix="graph-hypoth-recorder-"))
    chrome = subprocess.Popen(
        [
            chrome_path, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
            "--no-default-browser-check", "--remote-debugging-port=0",
            f"--user-data-dir={profile_dir}", f"--window-size={VIEWPORT[0]},{VIEWPORT[1]}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(80):
        active = profile_dir / "DevToolsActivePort"
        if active.is_file() and active.read_text().strip():
            return chrome, profile_dir, int(active.read_text().splitlines()[0])
        time.sleep(0.25)
    chrome.terminate()
    shutil.rmtree(profile_dir, ignore_errors=True)
    raise RuntimeError("Chrome did not publish its DevTools port")


async def session(args: argparse.Namespace, base: str, port: int) -> int:
    import websockets

    pages = [t for t in get_json(f"http://127.0.0.1:{port}/json") if t.get("type") == "page"]
    if len(pages) != 1 or pages[0].get("url") != "about:blank":
        print(f"unexpected tabs in our own browser: {[t.get('url') for t in pages]}", file=sys.stderr)
        return 1
    print(f"recording {base} with our own Chrome on port {port}", flush=True)
    frames_dir = args.frames_dir
    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)
    rec = None
    try:
        async with websockets.connect(
            pages[0]["webSocketDebuggerUrl"], max_size=80_000_000, ping_timeout=None
        ) as ws:
            rec = Recorder(ws, base, frames_dir)
            await rec.call("Page.enable")
            await rec.call("Runtime.enable")
            await rec.call(
                "Emulation.setDeviceMetricsOverride", width=VIEWPORT[0], height=VIEWPORT[1],
                deviceScaleFactor=1, mobile=False,
            )
            runner = record_live if args.mode == "live" else record_browse
            return await runner(rec, args)
    finally:
        if rec is not None and rec.frames:  # keep what was captured even when the flow failed
            write_gif(rec.frames, args.out, args.width)
            if frames_dir is not None:
                manifest = [{"hold_ms": hold, "label": label} for _, hold, label in rec.frames]
                (frames_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["live", "browse"])
    parser.add_argument("--url", default="http://127.0.0.1:8765/", help="the running web app")
    parser.add_argument("--out", type=Path, required=True, help="the GIF to write")
    parser.add_argument("--frames-dir", type=Path, default=None, help="also keep every frame as a PNG")
    parser.add_argument("--width", type=int, default=1000, help="GIF width in pixels")
    parser.add_argument("--chrome", default=DEFAULT_CHROME, help="path to the Chrome executable")
    parser.add_argument("--profile", default=None, help="live: research profile to type into the form")
    parser.add_argument("--config", default=None, help="live: system config to type into the form")
    parser.add_argument("--confirm-top", type=int, default=2, help="live: candidates to tick when asked")
    parser.add_argument("--run-id", default=None, help="browse: a run the server already lists")
    parser.add_argument("--import-dir", default=None, help="browse: a run directory to import first")
    args = parser.parse_args()
    if args.mode == "browse" and not (args.run_id or args.import_dir):
        parser.error("browse needs --run-id or --import-dir")
    base = args.url if args.url.endswith("/") else args.url + "/"
    if not wait_for_server(base):
        print(f"no graph-hypoth-web server at {base}", file=sys.stderr)
        return 1
    chrome, profile_dir, port = launch_chrome(args.chrome)
    try:
        return asyncio.run(session(args, base, port))
    finally:
        chrome.terminate()
        shutil.rmtree(profile_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
