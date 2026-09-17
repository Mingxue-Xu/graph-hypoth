/* GraphHypoth web app: one hash-routed page over the FastAPI routes under /api.
 *
 * Screens: the run list with the launch form (#/), and one run (#/runs/<id>) with live progress,
 * the confirmation step when the pipeline is waiting, and the finished run's pages and claim graph.
 * Everything is built with DOM nodes, never HTML strings, so run content is always shown verbatim.
 */
(function () {
  "use strict";

  // ---------- helpers ----------
  const NS = "http://www.w3.org/2000/svg";

  function build(el, attrs, children) {
    if (attrs) {
      for (const [key, value] of Object.entries(attrs)) {
        if (value === null || value === undefined || value === false) continue;
        if (key === "class") el.setAttribute("class", value);
        else if (key === "dataset") Object.assign(el.dataset, value);
        else if (key.startsWith("on") && typeof value === "function") el.addEventListener(key.slice(2), value);
        else el.setAttribute(key, value === true ? "" : value);
      }
    }
    for (const child of children.flat(Infinity)) {
      if (child === null || child === undefined || child === false) continue;
      el.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return el;
  }
  const h = (tag, attrs, ...children) => build(document.createElement(tag), attrs, children);
  // Replace an element's children. Unlike Element.replaceChildren this drops null and flattens
  // arrays instead of stringifying them.
  const fill = (el, ...children) => { el.replaceChildren(); return build(el, null, children); };
  const s = (tag, attrs, ...children) => build(document.createElementNS(NS, tag), attrs, children);

  async function api(path, options) {
    const response = await fetch(path, options);
    let body = null;
    try { body = await response.json(); } catch (_) { body = null; }
    if (!response.ok) {
      let detail = body && body.detail;
      if (Array.isArray(detail)) detail = detail.map((d) => `${(d.loc || []).join(".")}: ${d.msg}`).join("; ");
      throw new Error(typeof detail === "string" && detail ? detail : `${response.status} ${response.statusText}`);
    }
    return body;
  }
  const getJSON = (path) => api(path);
  const postJSON = (path, payload) => api(path, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
  });

  const STATUS_LABEL = {
    queued: "queued", running: "running", waiting_confirmation: "waiting for your confirmation",
    completed: "completed", failed: "failed", interrupted: "interrupted", imported: "imported",
  };
  const EDGE_STATUS = ["supported", "contradicted", "qualified", "not_causal", "insufficient", "unverified"];
  const EDGE_GLYPH = { supported: "✓", contradicted: "✕", qualified: "≈", not_causal: "⊘", insufficient: "?", unverified: "◌" };
  const EDGE_LABEL = {
    supported: "supported", contradicted: "contradicted", qualified: "qualified", not_causal: "not causal",
    insufficient: "insufficient", unverified: "unverified",
  };
  const NODE_CLASS_LABEL = {
    claim: "extracted from the seed", mined: "mined from the literature",
    hypo: "proposed by the Synthesist", prio: "priority-authored",
  };

  const pill = (status) => h("span", { class: `pill pill-${status}` }, STATUS_LABEL[status] || status);
  const edgeStatus = (status) => h("span", { class: `st st-${status}` },
    h("span", { class: "glyph" }, EDGE_GLYPH[status] || "·"), EDGE_LABEL[status] || status);
  const fmtTime = (iso) => (iso ? new Date(iso).toLocaleString() : "");
  const fmtNum = (value, digits) => (typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits === undefined ? 2 : digits) : (value === null || value === undefined ? "–" : String(value)));
  const mmss = (seconds) => { const total = Math.max(0, Math.floor(seconds || 0)); return `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`; };
  const truncate = (text, n) => (text && text.length > n ? `${text.slice(0, n - 1)}…` : (text || ""));
  const isHttp = (url) => /^https?:\/\//i.test(url || "");
  const fileUrl = (runId, path) => `/runs/${encodeURIComponent(runId)}/files/${path.split("/").map(encodeURIComponent).join("/")}`;
  const runHash = (runId) => `#/runs/${encodeURIComponent(runId)}`;
  // Paths inside the server's working directory are shown relative to it, as the CLI takes them.
  const relPath = (p) => (state.cwd && p && p.startsWith(`${state.cwd}/`) ? p.slice(state.cwd.length + 1) : p);
  const prettyKey = (key) => String(key).replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
  // A proposed edge may point at a node that already exists in the committed graph; the confirm hook
  // only knows the labels of the candidate's own new nodes, so such endpoints arrive as their
  // content-addressed id. Show a short form with the full id as a tooltip.
  const endpoint = (name) => (/^[0-9a-f]{32,}$/i.test(name)
    ? h("span", { class: "mono muted", title: name }, `${name.slice(0, 8)}… (committed concept)`)
    : name);

  function renderValue(value) {
    if (value === null || value === undefined || value === "") return h("span", { class: "muted" }, "–");
    if (Array.isArray(value)) {
      if (!value.length) return h("span", { class: "muted" }, "none");
      return h("ul", null, value.map((item) => h("li", null, renderValue(item))));
    }
    if (typeof value === "object") {
      const entries = Object.entries(value).filter(([, v]) => v !== null && v !== "" && !(Array.isArray(v) && !v.length));
      return h("dl", { class: "kv" }, entries.map(([k, v]) => [h("dt", null, prettyKey(k)), h("dd", null, renderValue(v))]));
    }
    if (typeof value === "number") return String(Number.isInteger(value) ? value : value.toFixed(3));
    return String(value);
  }

  // ---------- state and routing ----------
  const state = { pollTimer: null, run: null, events: [], cursor: 0, graph: null, artifacts: null, canvas: null, cwd: null };
  const view = document.getElementById("view");
  const activePill = document.getElementById("active-pill");
  const footInfo = document.getElementById("foot-info");

  function stopPolling() { if (state.pollTimer) { clearTimeout(state.pollTimer); state.pollTimer = null; } }
  function route() {
    stopPolling();
    const match = (location.hash || "#/").match(/^#\/runs\/([^/]+)/);
    if (match) showRun(decodeURIComponent(match[1]));
    else showHome();
  }
  window.addEventListener("hashchange", route);

  async function refreshStatus() {
    try {
      const status = await getJSON("/api/status");
      state.cwd = status.cwd;
      footInfo.textContent = `Working directory ${status.cwd} · runs under ${status.runs_root} · Python ${status.python}`;
      if (status.active_run_id) {
        activePill.textContent = `● run ${status.active_run_id} is active`;
        activePill.href = runHash(status.active_run_id);
        activePill.hidden = false;
      } else {
        activePill.hidden = true;
      }
      return status;
    } catch (error) {
      footInfo.textContent = `Cannot reach the server: ${error.message}`;
      return null;
    }
  }

  // ---------- home ----------
  async function showHome() {
    fill(view, h("p", { class: "muted" }, "Loading…"));
    const status = await refreshStatus();
    let runs = [];
    try { runs = (await getJSON("/api/runs")).runs; } catch (error) { fill(view, h("p", { class: "error" }, error.message)); return; }
    const choices = h("datalist", { id: "yaml-choices" }, (status ? status.choices : []).map((c) => h("option", { value: c })));
    const message = h("div", { hidden: true });
    const busy = Boolean(status && status.active_run_id);

    const profile = h("input", { type: "text", name: "profile_path", list: "yaml-choices", value: (status && status.defaults.profile_path) || "examples/research_profile.yaml", required: true });
    const config = h("input", { type: "text", name: "config_path", list: "yaml-choices", value: (status && status.defaults.config_path) || "config/claude-code.yaml", required: true });
    const configHint = h("p", { class: "small", style: "margin:4px 0 0" });
    const updateConfigHint = () => {
      const choice = ((status && status.config_choices) || []).find((c) => c.path === config.value.trim());
      if (!choice) {
        configHint.className = "small muted";
        configHint.textContent = "Any system config YAML; its model ids are checked when the run starts.";
      } else if (!choice.ready) {
        configHint.className = "small warn";
        configHint.textContent = `Template: the model ids for ${choice.placeholders.join(", ")} are still placeholders, so a run would stop at preflight. Choose config/claude-code.yaml or config/codex.yaml, or a copy with real model ids.`;
      } else {
        configHint.className = "small muted";
        configHint.textContent = `Ready: ${choice.providers.join(", ")} · models ${choice.models.join(", ")}`;
      }
    };
    config.addEventListener("input", updateConfigHint);
    updateConfigHint();
    const elaborate = h("input", { type: "checkbox", name: "elaborate", checked: true });
    const connected = h("input", { type: "checkbox", name: "connected_render", checked: true });
    const confirmHere = h("input", { type: "checkbox", name: "confirm_in_browser" });
    const startButton = h("button", { type: "submit", class: "primary", disabled: busy }, "Start run");
    const form = h("form", { class: "launch", onsubmit: async (event) => {
      event.preventDefault();
      startButton.disabled = true;
      message.hidden = true;
      try {
        const record = await postJSON("/api/runs", {
          profile_path: profile.value.trim(), config_path: config.value.trim(),
          elaborate: elaborate.checked, connected_render: connected.checked, confirm_in_browser: confirmHere.checked,
        });
        location.hash = runHash(record.run_id);
      } catch (error) {
        message.className = "error"; message.textContent = error.message; message.hidden = false; startButton.disabled = false;
      }
    } },
      h("label", null, "Research profile (YAML)", profile),
      h("label", null, "System config (YAML)", config, configHint),
      h("div", { class: "checks" },
        h("label", null, elaborate, "Write plain-language hypothesis prose (extra model calls)"),
        h("label", null, connected, "Render the connected page per committed hypothesis"),
        h("label", null, confirmHere, "Ask me in the browser before committing hypotheses (overrides the profile's confirm policy)"),
      ),
      h("div", { class: "actions" }, startButton,
        busy ? h("span", { class: "muted" }, "A run is active; the app runs one pipeline at a time.") :
          h("span", { class: "muted small" }, "Paths resolve against the server's working directory. A live run takes tens of minutes.")),
      message,
    );

    const importInput = h("input", { type: "text", placeholder: "runtime_artifacts/example-api", required: true });
    const importMessage = h("div", { hidden: true });
    const importForm = h("form", { class: "inline-form", onsubmit: async (event) => {
      event.preventDefault();
      try {
        const record = await postJSON("/api/runs/import", { run_dir: importInput.value.trim() });
        location.hash = runHash(record.run_id);
      } catch (error) { importMessage.className = "error"; importMessage.textContent = error.message; importMessage.hidden = false; }
    } }, importInput, h("button", { type: "submit" }, "Open"));

    fill(view, 
      h("section", { class: "card" },
        h("h1", null, "Start a run"),
        h("p", { class: "muted" }, "Runs the complete pipeline: planned retrieval, extraction, evidence review, priority, the Research Synthesist and Critic Panel, confirmation, experiment design, and export."),
        form, choices),
      h("section", { class: "card" },
        h("h2", null, "Runs"),
        runs.length ? runsTable(runs) : h("p", { class: "muted" }, "No runs yet."),
        h("h3", null, "Open an existing run directory"),
        h("p", { class: "muted small" }, "Any directory with a graph.json (for example a CLI run) can be browsed here without re-running."),
        importForm, importMessage),
    );
    if (busy) state.pollTimer = setTimeout(() => { if ((location.hash || "#/") === "#/") showHome(); }, 4000);
  }

  function runsTable(runs) {
    return h("table", { class: "list" },
      h("thead", null, h("tr", null, h("th", null, "Started"), h("th", null, "Status"), h("th", null, "Research question / claim"), h("th", null, "Source"), h("th", null, "Run id"))),
      h("tbody", null, runs.map((run) => h("tr", { class: "click", onclick: () => { location.hash = runHash(run.run_id); } },
        h("td", null, fmtTime(run.created_at)),
        h("td", null, pill(run.status)),
        h("td", null, truncate(run.seed, 110) || h("span", { class: "muted" }, "–")),
        h("td", null, run.source),
        h("td", { class: "v" }, run.run_id)))));
  }

  // ---------- run page ----------
  let sections = null;

  async function showRun(runId) {
    fill(view, h("p", { class: "muted" }, "Loading…"));
    state.run = null; state.events = []; state.cursor = 0; state.graph = null; state.artifacts = null; state.canvas = null;
    await refreshStatus();
    try { state.run = await getJSON(`/api/runs/${encodeURIComponent(runId)}`); }
    catch (error) { fill(view, h("section", { class: "card" }, h("p", { class: "error" }, error.message), h("p", null, h("a", { href: "#/" }, "Back to runs")))); return; }
    sections = {
      head: h("section", { class: "card" }),
      progress: h("section", { class: "card" }),
      confirm: h("section", { class: "card", hidden: true }),
      results: h("section", { class: "card", hidden: true }),
    };
    fill(view, sections.head, sections.confirm, sections.progress, sections.results);
    renderHead();
    await pullEvents();
    renderProgress();
    await renderStatusSections();
    if (state.run.is_active) schedulePoll(runId);
  }

  function schedulePoll(runId) {
    stopPolling();
    state.pollTimer = setTimeout(async () => {
      if (!state.run || state.run.run_id !== runId) return;
      try {
        const previous = state.run.status;
        state.run = await getJSON(`/api/runs/${encodeURIComponent(runId)}`);
        await pullEvents();
        renderHead();
        renderProgress();
        if (state.run.status !== previous) await renderStatusSections();
        refreshStatus();
      } catch (error) {
        sections.head.append(h("p", { class: "error" }, error.message));
      }
      if (state.run && state.run.is_active) schedulePoll(runId);
    }, 1500);
  }

  async function pullEvents() {
    const page = await getJSON(`/api/runs/${encodeURIComponent(state.run.run_id)}/events?after=${state.cursor}`);
    state.events.push(...page.events);
    state.cursor = page.next;
  }

  function renderHead() {
    const run = state.run;
    const policy = run.confirm_policy && run.confirm_policy.mode ? Object.entries(run.confirm_policy).map(([k, v]) => `${k}=${v}`).join(", ") : "";
    fill(sections.head, 
      h("div", { class: "row" },
        h("div", { class: "grow" },
          h("h1", null, run.seed ? truncate(run.seed, 160) : `Run ${run.run_id}`),
          h("p", null, pill(run.status), " ", h("span", { class: "muted" }, run.seed_kind ? `${prettyKey(run.seed_kind)} · ` : "", "run ", h("code", null, run.run_id))),
          run.error ? h("p", { class: "error" }, run.error) : null,
        ),
        h("dl", { class: "kv small" },
          run.profile_path ? [h("dt", null, "Profile"), h("dd", null, h("code", null, relPath(run.profile_path)))] : null,
          run.config_path ? [h("dt", null, "Config"), h("dd", null, h("code", null, relPath(run.config_path)))] : null,
          h("dt", null, "Run directory"), h("dd", null, h("code", null, relPath(run.run_dir))),
          run.models && Object.keys(run.models).length ? [h("dt", null, "Models"), h("dd", null,
            [...new Set(Object.values(run.models).map((m) => `${m.provider} ${m.model_id}`))].join(" · "))] : null,
          policy ? [h("dt", null, "Confirm policy"), h("dd", null, policy, run.options && run.options.confirm_in_browser ? " (overridden: confirm in browser)" : "")] : null,
          run.started_at ? [h("dt", null, "Started"), h("dd", null, fmtTime(run.started_at))] : null,
          run.finished_at ? [h("dt", null, "Finished"), h("dd", null, fmtTime(run.finished_at))] : null,
        )),
      run.summary ? summaryLine(run.summary) : null,
    );
  }

  function summaryLine(summary) {
    return h("p", null,
      h("strong", null, "Result: "), `graph v${summary.version ?? "?"}, ${summary.surfaced} hypotheses surfaced, `,
      `${summary.committed_edges} edges with a committed status`, summary.open_risks ? `, ${summary.open_risks} open risks` : "", ".");
  }

  function renderProgress() {
    const run = state.run;
    const events = state.events;
    const meaningful = events.filter((e) => e.event !== "heartbeat");
    const last = events[events.length - 1];
    const stages = [];
    for (const e of meaningful) {
      if (!e.stage) continue;
      const known = stages.find((st) => st.name === e.stage);
      if (known) { known.status = e.status; known.event = e.event; }
      else stages.push({ name: e.stage, status: e.status, event: e.event });
    }
    stages.forEach((st, index) => { st.done = index < stages.length - 1 || run.status === "completed"; });
    const chips = stages.map((st) => h("span", {
      class: `chip ${st.status === "waiting" ? "waiting" : st.done && run.status !== "failed" ? "done" : "now"}`,
    }, st.name));
    const activity = last ? h("div", { class: "activity" },
      run.is_active ? h("span", { class: "spinner" }) : null,
      h("div", null,
        h("strong", null, last.event === "heartbeat" ? "Still waiting: " : "", last.stage || ""),
        last.detail ? ` — ${last.detail}` : "",
        last.current !== null && last.current !== undefined && last.total ? ` ${last.current}/${last.total}` : "",
        last.provider ? ` · ${last.provider}` : "",
        h("div", { class: "muted small" }, `${mmss(last.elapsed_seconds)} since start`,
          last.event === "heartbeat" ? ` · ${mmss(last.operation_elapsed_seconds)} in this step` : ""))) :
      h("p", { class: "muted" }, run.source === "imported" ? "Imported directory: no progress file." : "Waiting for the first progress event…");
    const log = h("ul", { class: "log" }, meaningful.slice(-400).map((e) => h("li", { class: e.status || "" },
      h("span", { class: "t" }, mmss(e.elapsed_seconds)),
      h("span", null, h("span", { class: "stage" }, e.stage || ""), e.detail ? ` — ${e.detail}` : "",
        e.current !== null && e.current !== undefined && e.total ? ` ${e.current}/${e.total}` : "",
        e.provider ? ` · ${e.provider}` : "", e.status && e.status !== "running" ? ` · ${e.status}` : ""))));
    fill(sections.progress, 
      h("h2", null, "Progress"),
      chips.length ? h("div", { class: "stagebar" }, chips) : null,
      activity,
      meaningful.length ? log : null,
      run.output && run.output.length ? [h("h3", null, "Pipeline output"), h("pre", { class: "output" }, run.output.join("\n"))] : null,
    );
    log.scrollTop = log.scrollHeight;
  }

  async function renderStatusSections() {
    const run = state.run;
    if (run.status === "waiting_confirmation" && run.confirmation && run.confirmation.candidates) {
      sections.confirm.hidden = false;
      renderConfirmation(run.confirmation.candidates);
    } else {
      sections.confirm.hidden = true;
      fill(sections.confirm);
    }
    if (run.is_active) { sections.results.hidden = true; return; }
    sections.results.hidden = false;
    await renderResults();
  }

  // ---------- confirmation ----------
  function renderConfirmation(candidates) {
    const boxes = new Map();
    const countLabel = h("span", { class: "muted" });
    const message = h("div", { hidden: true });
    const buttons = [];
    const updateCount = () => {
      const n = [...boxes.values()].filter((b) => b.checked).length;
      countLabel.textContent = `${n} of ${candidates.length} selected`;
      buttons[0].textContent = `Commit selected (${n})`;
      for (const [cid, box] of boxes) box.closest(".cand").classList.toggle("checked", box.checked);
    };
    const submit = async (payload) => {
      buttons.forEach((b) => { b.disabled = true; });
      message.hidden = true;
      try {
        const result = await postJSON(`/api/runs/${encodeURIComponent(state.run.run_id)}/confirmation`, payload);
        message.className = "warn"; message.textContent = result.selected.length ? `Committing ${result.selected.join(", ")}. The experiment stage follows.` : "No hypothesis committed; the run finishes with the evidence core only."; message.hidden = false;
      } catch (error) { message.className = "error"; message.textContent = error.message; message.hidden = false; buttons.forEach((b) => { b.disabled = false; }); }
    };
    buttons.push(
      h("button", { class: "primary", onclick: () => submit({ candidate_ids: [...boxes].filter(([, b]) => b.checked).map(([cid]) => cid) }) }, "Commit selected"),
      h("button", { onclick: () => submit({ choice: "all" }) }, "Commit all"),
      h("button", { onclick: () => submit({ choice: "none" }) }, "Commit none"),
    );
    const cards = candidates.map((c) => {
      const box = h("input", { type: "checkbox", onchange: updateCount });
      boxes.set(c.candidate_id, box);
      return candidateCard(c, box);
    });
    fill(sections.confirm, 
      h("h2", null, "Confirm which hypotheses to commit"),
      h("p", { class: "muted" }, "The Research Synthesist surfaced these candidates after the Critic Panel review and the deterministic gates. Only the ones you confirm are committed as new unverified edges and go on to experiment design; the pipeline is paused until you answer."),
      cards,
      h("div", { class: "confirm-actions" }, buttons, countLabel, message),
    );
    updateCount();
  }

  function candidateCard(c, box) {
    const scaffold = c.idea_scaffold || {};
    const scaffoldOrder = ["problem", "claim_anchor", "challenged_assumption", "incumbent_limit", "lever", "synthesis", "guarantee", "fail_safe", "why", "method", "experiment", "critique"];
    const scaffoldKeys = [...scaffoldOrder.filter((k) => scaffold[k]), ...Object.keys(scaffold).filter((k) => !scaffoldOrder.includes(k))];
    return h("article", { class: "cand" },
      h("header", null,
        h("span", { class: "rank" }, `#${c.rank}`),
        h("label", null, box, c.candidate_id),
        h("span", { class: "badges" },
          h("span", { class: `badge ${c.cross_concept ? "on" : ""}` }, c.cross_concept ? "cross-concept" : "within-concept"),
          c.common_sense ? h("span", { class: "badge" }, "flagged common sense") : null,
          c.lineage ? h("span", { class: "badge" }, `derived (${c.lineage.strategy || "revision"})`) : null),
      ),
      c.new_nodes && c.new_nodes.length ? h("p", null, h("strong", null, "New concepts: "), c.new_nodes.map((n, i) => [i ? "; " : "", h("strong", null, n.label), n.type ? h("span", { class: "muted" }, ` (${n.type})`) : "", n.definition ? ` — ${n.definition}` : ""])) : null,
      c.new_edges && c.new_edges.length ? h("ul", { class: "chain" }, c.new_edges.map((e) => h("li", null,
        e.sources.map((x, i) => [i ? " + " : "", endpoint(x)]), h("span", { class: "arrow" }, ` —${e.relation_type}→ `),
        e.targets.map((x, i) => [i ? " + " : "", endpoint(x)]),
        e.mechanism ? h("span", { class: "muted" }, ` · ${e.mechanism}`) : ""))) : null,
      c.rationale ? h("p", null, c.rationale) : null,
      h("table", { class: "scores" },
        h("tr", null, h("th", null, "field novelty"), h("th", null, "saturation"), h("th", null, "HypScore"), h("th", null, "RankScore")),
        h("tr", null, h("td", null, fmtNum(c.field_novelty)), h("td", null, fmtNum(c.saturation)), h("td", null, fmtNum(c.hyp_score, 3)), h("td", null, fmtNum(c.rank_score, 3)))),
      c.mechanism_chain && c.mechanism_chain.length ? [h("h3", null, "Mechanism chain"), h("ul", { class: "chain" }, c.mechanism_chain.map((step) => h("li", null,
        step.from, h("span", { class: "arrow" }, ` —${step.relation || ""}→ `), step.to, step.mechanism ? h("span", { class: "muted" }, ` · ${step.mechanism}`) : "")))] : null,
      scaffoldKeys.length ? [h("h3", null, "Idea scaffold"), h("dl", { class: "kv" }, scaffoldKeys.map((k) => [h("dt", null, prettyKey(k)), h("dd", null, scaffold[k])]))] : null,
      c.assumptions && c.assumptions.length ? [h("h3", null, "Assumptions"), h("ul", null, c.assumptions.map((a) => h("li", null, a)))] : null,
      c.source_quotes && c.source_quotes.length ? [h("h3", null, "Grounding quotes"), c.source_quotes.map((q) => h("p", { class: "quote" }, q.quote_span || q.quote || "", q.role_in_hypothesis ? h("span", { class: "muted" }, ` (${q.role_in_hypothesis}${q.evidence_id ? `, ${q.evidence_id}` : ""})`) : ""))] : null,
    );
  }

  // ---------- results ----------
  async function renderResults() {
    const run = state.run;
    const runId = run.run_id;
    fill(sections.results, h("h2", null, "Results"), h("p", { class: "muted" }, "Loading artifacts…"));
    try { state.artifacts = await getJSON(`/api/runs/${encodeURIComponent(runId)}/artifacts`); }
    catch (error) { fill(sections.results, h("h2", null, "Results"), h("p", { class: "error" }, error.message)); return; }
    const pages = state.artifacts.pages;
    const links = h("div", { class: "linklist" },
      pages.trace_index ? h("a", { href: fileUrl(runId, pages.trace_index), target: "_blank", rel: "noopener" }, "Surfaced hypotheses (trace index)") : null,
      pages.connected.map((p) => h("a", { href: fileUrl(runId, p), target: "_blank", rel: "noopener" }, `${p.replace(/-connected\.html$/, "")} connected page`)),
      pages.audit_memo ? h("a", { href: fileUrl(runId, pages.audit_memo), target: "_blank", rel: "noopener" }, "Audit memo") : null,
      pages.edge_table ? h("a", { href: fileUrl(runId, pages.edge_table), target: "_blank", rel: "noopener" }, "Edge table (CSV)") : null,
      pages.graph ? h("a", { href: fileUrl(runId, pages.graph), target: "_blank", rel: "noopener" }, "graph.json") : null,
      pages.error ? h("a", { href: fileUrl(runId, pages.error), target: "_blank", rel: "noopener" }, "Error traceback") : null,
    );
    const tableHost = h("div");
    const graphHost = h("div", null, h("p", { class: "muted" }, "Loading the claim graph…"));
    // The run's own summary is the fallback; the trace sidecar (present for every run that wrote
    // trace pages, imported ones included) supplies headlines and experiment outcomes.
    if (run.summary && run.summary.hypotheses && run.summary.hypotheses.length) fill(tableHost, hypothesesTable(run.summary.hypotheses, pages));
    fill(sections.results,
      h("h2", null, "Results"),
      links.childElementCount ? links : h("p", { class: "muted" }, "No artifacts were written."),
      tableHost,
      graphHost,
    );
    if (!pages.graph) { fill(graphHost, h("p", { class: "muted" }, "No graph.json in this run directory.")); return; }
    try { state.graph = await getJSON(`/api/runs/${encodeURIComponent(runId)}/graph`); }
    catch (error) { fill(graphHost, h("p", { class: "error" }, error.message)); return; }
    if (state.graph.surfaced && state.graph.surfaced.length) fill(tableHost, h("h3", null, "Surfaced hypotheses"), surfacedTable(state.graph.surfaced, pages));
    fill(graphHost, graphSection(state.graph));
  }

  function surfacedTable(rows, pages) {
    const runId = state.run.run_id;
    return h("table", { class: "list" },
      h("thead", null, h("tr", null, h("th", null, "Hypothesis"), h("th", null, "Plain-language headline"), h("th", null, "Committed"), h("th", null, "Experiment"), h("th", null, "Pages"))),
      h("tbody", null, rows.map((r) => {
        const connected = pages.connected.includes(`${r.candidate_id}-connected.html`);
        const n = r.hypothesis_edge_ids.length;
        const experiment = r.experiment
          ? `${r.experiment.status || "attempted"}${r.experiment.grounding_status ? `, ${String(r.experiment.grounding_status).replace(/_/g, " ")}` : ""}`
          : (r.experiment_plan ? "plan committed" : null);
        return h("tr", null,
          h("td", { class: "v" }, r.candidate_id),
          h("td", null, r.headline || h("span", { class: "muted" }, "–")),
          h("td", null, n ? `yes (${n} edge${n > 1 ? "s" : ""})` : h("span", { class: "muted" }, "not confirmed")),
          h("td", null, experiment || h("span", { class: "muted" }, "–")),
          h("td", null,
            pages.trace_index ? h("a", { href: fileUrl(runId, `trace/${r.candidate_id}.html`), target: "_blank", rel: "noopener" }, "trace") : null,
            connected ? [" · ", h("a", { href: fileUrl(runId, `${r.candidate_id}-connected.html`), target: "_blank", rel: "noopener" }, "connected")] : null));
      })));
  }

  function hypothesesTable(rows, pages) {
    const runId = state.run.run_id;
    return h("table", { class: "list", style: "margin-top:10px" },
      h("thead", null, h("tr", null, h("th", null, "Rank"), h("th", null, "Hypothesis"), h("th", null, "RankScore"), h("th", null, "Committed"), h("th", null, "Experiment plan"), h("th", null, "Pages"))),
      h("tbody", null, rows.map((r) => {
        const connected = pages.connected.includes(`${r.candidate_id}-connected.html`);
        const trace = `trace/${r.candidate_id}.html`;
        return h("tr", null,
          h("td", null, r.rank), h("td", { class: "v" }, r.candidate_id), h("td", null, fmtNum(r.rank_score, 3)),
          h("td", null, r.hypothesis_edge_ids && r.hypothesis_edge_ids.length ? `yes (${r.hypothesis_edge_ids.length} edge${r.hypothesis_edge_ids.length > 1 ? "s" : ""})` : h("span", { class: "muted" }, "no")),
          h("td", null, r.experiment_plan ? "yes" : h("span", { class: "muted" }, "no")),
          h("td", null,
            pages.trace_index ? h("a", { href: fileUrl(runId, trace), target: "_blank", rel: "noopener" }, "trace") : null,
            connected ? [" · ", h("a", { href: fileUrl(runId, `${r.candidate_id}-connected.html`), target: "_blank", rel: "noopener" }, "connected")] : null));
      })));
  }

  // ---------- graph canvas ----------
  function wrapLabel(text, maxChars, maxLines) {
    const words = String(text || "").split(/\s+/).filter(Boolean);
    const lines = [];
    let current = "";
    for (const word of words) {
      if (!current) current = word;
      else if ((current + " " + word).length <= maxChars) current += " " + word;
      else { lines.push(current); current = word; }
    }
    if (current) lines.push(current);
    if (lines.length > maxLines) { lines.length = maxLines; lines[maxLines - 1] = truncate(lines[maxLines - 1], maxChars); }
    return lines.length ? lines : [""];
  }

  function graphSection(graph) {
    const stats = graph.stats;
    const counts = stats.status_counts || {};
    const filters = { statuses: new Set(EDGE_STATUS), hypothesisOnly: false, query: "" };
    const details = h("aside", { class: "details" }, h("p", { class: "muted" }, "Click a node or an edge to inspect it: definitions, sources, evidence quotes, verdicts, and the experiment plan."));
    const canvasWrap = h("div", { class: "canvas-wrap" });
    const table = edgeTable(graph);
    const nodeById = new Map(graph.nodes.map((n) => [n.id, n]));
    const edgeById = new Map(graph.edges.map((e) => [e.id, e]));
    const canvas = { nodeEls: new Map(), edgeEls: new Map(), selected: null, incident: new Map() };
    state.canvas = canvas;
    for (const e of graph.edges) for (const id of [...e.sources, ...e.targets]) { if (!canvas.incident.has(id)) canvas.incident.set(id, []); canvas.incident.get(id).push(e.id); }

    const select = (kind, id) => {
      if (canvas.selected) {
        const prev = canvas.selected.kind === "node" ? canvas.nodeEls.get(canvas.selected.id) : canvas.edgeEls.get(canvas.selected.id);
        (prev || []).forEach((el) => el.classList.remove("selected"));
        table.rows.forEach((tr) => tr.classList.remove("selected"));
      }
      canvas.selected = { kind, id };
      const els = kind === "node" ? canvas.nodeEls.get(id) : canvas.edgeEls.get(id);
      (els || []).forEach((el) => el.classList.add("selected"));
      if (kind === "edge") { const tr = table.rowById.get(id); if (tr) tr.classList.add("selected"); }
      fill(details, kind === "node" ? nodeDetails(nodeById.get(id), graph, edgeById, select) : edgeDetails(edgeById.get(id), nodeById, select));
    };
    table.onSelect = (id) => select("edge", id);

    const applyFilters = () => {
      const query = filters.query.trim().toLowerCase();
      const visibleEdges = new Set();
      for (const e of graph.edges) {
        const show = filters.statuses.has(e.status) && (!filters.hypothesisOnly || e.is_hypothesis);
        (canvas.edgeEls.get(e.id) || []).forEach((el) => el.classList.toggle("hidden", !show));
        if (show) visibleEdges.add(e.id);
      }
      for (const n of graph.nodes) {
        const incident = canvas.incident.get(n.id) || [];
        const show = !incident.length || incident.some((id) => visibleEdges.has(id));
        const hit = query && n.label.toLowerCase().includes(query);
        (canvas.nodeEls.get(n.id) || []).forEach((el) => { el.classList.toggle("hidden", !show); el.classList.toggle("hit", Boolean(hit)); el.classList.toggle("dim", Boolean(query) && !hit); });
      }
      table.rows.forEach((tr) => { tr.hidden = !visibleEdges.has(tr.dataset.edgeId); });
    };

    const toolbar = h("div", { class: "toolbar" },
      EDGE_STATUS.map((status) => h("label", null,
        h("input", { type: "checkbox", checked: true, onchange: (ev) => { if (ev.target.checked) filters.statuses.add(status); else filters.statuses.delete(status); applyFilters(); } }),
        edgeStatus(status), h("span", { class: "muted" }, ` ${counts[status] || 0}`))),
      h("label", null, h("input", { type: "checkbox", onchange: (ev) => { filters.hypothesisOnly = ev.target.checked; applyFilters(); } }), `Hypothesis edges only (${stats.hypothesis_edges})`),
      h("input", { type: "search", placeholder: "Find a concept…", oninput: (ev) => { filters.query = ev.target.value; applyFilters(); } }),
    );
    const legend = h("div", { class: "legend" },
      ["claim", "mined", "hypo", "prio"].map((k) => h("span", null, h("i", { class: `sw ${k}` }), NODE_CLASS_LABEL[k])),
      h("span", null, h("i", { class: "ln", style: "border-color: var(--st-supported)" }), "verified edge"),
      h("span", null, h("i", { class: "ln dash", style: "border-color: var(--st-unverified)" }), "unverified / hypothesis edge"),
    );
    const summary = h("p", { class: "muted small" },
      `${stats.nodes} concepts, ${stats.edges} edges (${stats.hypothesis_edges} hypothesis edges), ${stats.evidence_links} evidence links, ${stats.experiment_plans} experiment plans` +
      (stats.pool_items ? `, ${stats.pool_items} retrieved passages in the pool.` : ". No retrieval pool found next to the graph, so evidence bodies show only their matched spans."));

    const section = h("div", null,
      h("h3", null, "Claim graph"), summary, toolbar, legend,
      h("div", { class: "split" }, canvasWrap, details),
      h("h3", null, "Edges"), table.el);
    // Draw after insertion so svg-pan-zoom can measure the element.
    requestAnimationFrame(() => { drawGraph(canvasWrap, graph, canvas, select); applyFilters(); });
    return section;
  }

  function drawGraph(host, graph, canvas, select) {
    if (typeof dagre === "undefined") {
      fill(host, h("p", { class: "warn", style: "margin:12px" }, "The graph layout library (dagre, loaded from jsDelivr) is unavailable, so the canvas cannot be drawn offline. The edge table below still lists every edge with its evidence."));
      return;
    }
    const g = new dagre.graphlib.Graph({ multigraph: true });
    g.setGraph({ rankdir: "LR", nodesep: 26, ranksep: 90, marginx: 24, marginy: 24 });
    g.setDefaultEdgeLabel(() => ({}));
    const sized = new Map();
    for (const n of graph.nodes) {
      const lines = wrapLabel(n.label, 22, 3);
      const width = Math.max(96, Math.min(230, 7.6 * Math.max(...lines.map((l) => l.length)) + 30));
      sized.set(n.id, { width, height: 16 * lines.length + 20, lines, node: n });
    }
    // Only concepts with a committed edge take part in the ranked layout. A run mines many concepts
    // that never gain an edge; those go into a compact grid below so they do not stretch the graph.
    const linked = new Set();
    for (const e of graph.edges) for (const id of [...e.sources, ...e.targets]) linked.add(id);
    const isolated = graph.nodes.filter((n) => !linked.has(n.id));
    for (const n of graph.nodes) if (linked.has(n.id)) g.setNode(n.id, sized.get(n.id));
    let serial = 0;
    for (const e of graph.edges) {
      for (const source of e.sources) for (const target of e.targets) {
        const label = `${EDGE_GLYPH[e.status] || ""} ${truncate(e.relation, 28)}`.trim();
        g.setEdge(source, target, { edge: e, label, width: 6.4 * label.length + 14, height: 16 }, `${e.id}#${serial++}`);
      }
    }
    dagre.layout(g);
    const layout = g.graph();
    const placed = [];
    for (const id of g.nodes()) {
      const d = g.node(id);
      placed.push({ ...d, n: d.node, x: d.x - d.width / 2, y: d.y - d.height / 2 });
    }
    const baseWidth = g.nodeCount() ? Math.max(200, layout.width || 200) : 0;
    const baseHeight = g.nodeCount() ? Math.max(120, layout.height || 120) : 0;
    let width = baseWidth;
    let height = baseHeight;
    let caption = null;
    if (isolated.length) {
      const cellW = Math.max(...isolated.map((n) => sized.get(n.id).width)) + 14;
      const cellH = Math.max(...isolated.map((n) => sized.get(n.id).height)) + 12;
      const cols = Math.max(1, Math.min(isolated.length, Math.floor(Math.max(baseWidth, 980) / cellW)));
      const captionY = baseHeight ? baseHeight + 22 : 18;
      const top = captionY + 14;
      isolated.forEach((n, i) => {
        const d = sized.get(n.id);
        placed.push({ ...d, n, x: 24 + (i % cols) * cellW, y: top + Math.floor(i / cols) * cellH });
      });
      width = Math.max(width, 24 + cols * cellW + 10);
      height = top + Math.ceil(isolated.length / cols) * cellH + 12;
      caption = s("text", { class: "caption", x: 24, y: captionY },
        `${isolated.length} concept${isolated.length > 1 ? "s" : ""} without a committed edge`);
    }
    const svg = s("svg", { class: "graph", viewBox: `0 0 ${Math.max(200, width)} ${Math.max(120, height)}`, preserveAspectRatio: "xMidYMid meet" });
    const defs = s("defs", null, EDGE_STATUS.map((status) => s("marker", { id: `arrow-${status}`, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" },
      s("path", { d: "M 0 0 L 10 5 L 0 10 z", style: `fill: var(--st-${status})` }))));
    const edgesLayer = s("g", { class: "edges" });
    const nodesLayer = s("g", { class: "nodes" });
    for (const key of g.edges()) {
      const data = g.edge(key);
      const e = data.edge;
      const d = data.points.map((p, i) => `${i ? "L" : "M"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
      const group = s("g", { class: `edge st-${e.status} ${e.is_hypothesis ? "hypothesis" : ""}`, onclick: () => select("edge", e.id) },
        s("path", { class: "line", d, "marker-end": `url(#arrow-${e.status})` }),
        s("path", { class: "hit-area", d }),
        data.x !== undefined ? s("g", { class: "lbl" },
          s("rect", { x: data.x - data.width / 2, y: data.y - data.height / 2, width: data.width, height: data.height, rx: 4 }),
          s("text", { x: data.x, y: data.y + 0.5 }, data.label)) : null,
        s("title", null, `${e.relation} · ${EDGE_LABEL[e.status] || e.status}${e.confidence !== null && e.confidence !== undefined ? ` · confidence ${fmtNum(e.confidence)}` : ""}`));
      edgesLayer.append(group);
      if (!canvas.edgeEls.has(e.id)) canvas.edgeEls.set(e.id, []);
      canvas.edgeEls.get(e.id).push(group);
    }
    if (caption) nodesLayer.append(caption);
    for (const p of placed) {
      const n = p.n;
      const rx = n.klass === "mined" ? 10 : n.klass === "hypo" ? 16 : 4;
      const group = s("g", { class: `node klass-${n.klass}`, transform: `translate(${p.x.toFixed(1)},${p.y.toFixed(1)})`, onclick: () => select("node", n.id) },
        s("rect", { width: p.width, height: p.height, rx }),
        s("text", { x: p.width / 2, y: 16, "text-anchor": "middle" }, p.lines.map((line, i) => s("tspan", { x: p.width / 2, dy: i ? 16 : 4 }, line))),
        s("title", null, `${n.label}${n.definition ? ` — ${n.definition}` : ""}`));
      nodesLayer.append(group);
      canvas.nodeEls.set(n.id, [group]);
    }
    svg.append(defs, edgesLayer, nodesLayer);
    fill(host, svg);
    if (typeof svgPanZoom !== "undefined") {
      try { svgPanZoom(svg, { zoomEnabled: true, controlIconsEnabled: true, fit: true, center: true, minZoom: 0.1, maxZoom: 8 }); }
      catch (_) { /* the static SVG remains usable */ }
    }
  }

  function edgeTable(graph) {
    const rows = [];
    const rowById = new Map();
    const labelOf = (id) => { const n = graph.nodes.find((x) => x.id === id); return n ? n.label : id; };
    const handle = { rows, rowById, onSelect: null };
    const tbody = h("tbody", null, graph.edges.map((e) => {
      const tr = h("tr", { class: "click", dataset: { edgeId: e.id }, onclick: () => { if (handle.onSelect) handle.onSelect(e.id); } },
        h("td", null, e.sources.map(labelOf).join(" + "), h("span", { class: "muted" }, " → "), e.targets.map(labelOf).join(" + ")),
        h("td", null, e.relation, e.is_hypothesis ? h("span", { class: "badge", style: "margin-left:6px" }, e.candidate_id || "hypothesis") : null),
        h("td", null, edgeStatus(e.status)),
        h("td", null, e.verdict && e.verdict !== e.status ? e.verdict : h("span", { class: "muted" }, e.verdict ? "same" : "–")),
        h("td", null, fmtNum(e.confidence)),
        h("td", null, e.evidence.length || h("span", { class: "muted" }, "0")),
        h("td", null, e.plan ? "yes" : h("span", { class: "muted" }, "–")),
        h("td", null, e.open_risks.length || h("span", { class: "muted" }, "0")));
      rows.push(tr); rowById.set(e.id, tr);
      return tr;
    }));
    handle.el = h("table", { class: "list" },
      h("thead", null, h("tr", null, h("th", null, "Source → target"), h("th", null, "Relation"), h("th", null, "Status"), h("th", null, "Verdict"), h("th", null, "Confidence"), h("th", null, "Evidence"), h("th", null, "Plan"), h("th", null, "Risks"))),
      graph.edges.length ? tbody : h("tbody", null, h("tr", null, h("td", { colspan: 8, class: "muted" }, "The graph has no edges."))));
    return handle;
  }

  function nodeDetails(n, graph, edgeById, select) {
    const incident = graph.edges.filter((e) => e.sources.includes(n.id) || e.targets.includes(n.id));
    const labelOf = (id) => { const x = graph.nodes.find((y) => y.id === id); return x ? x.label : id; };
    return [
      h("h2", null, n.label),
      h("p", { class: "muted small" }, NODE_CLASS_LABEL[n.klass] || n.klass, n.type ? ` · ${n.type}` : "", n.user_priority !== null && n.user_priority !== undefined ? ` · priority ${fmtNum(n.user_priority)}` : ""),
      n.definition ? h("p", null, n.definition) : null,
      n.aliases.length ? h("p", { class: "small" }, h("strong", null, "Aliases: "), n.aliases.join(", ")) : null,
      n.scope_qualifiers.length ? h("p", { class: "small" }, h("strong", null, "Scope: "), n.scope_qualifiers.join("; ")) : null,
      n.sources.length ? [h("h3", null, "Source papers"), n.sources.map((src) => h("div", { class: "ev" },
        isHttp(src.url) ? h("a", { href: src.url, target: "_blank", rel: "noopener" }, src.title || src.url) : (src.title || h("span", { class: "muted" }, "untitled")),
        src.source ? h("span", { class: "muted" }, ` · ${src.source}`) : "",
        src.quote ? h("p", { class: "quote" }, `“${src.quote}”`) : null))] : null,
      n.provenance.length ? h("p", { class: "muted small" }, "Provenance: ", n.provenance.map((p) => p.source || "?").filter((v, i, a) => a.indexOf(v) === i).join(", ")) : h("p", { class: "muted small" }, "No provenance record: this concept was coined by the Research Synthesist."),
      h("h3", null, `Edges (${incident.length})`),
      incident.length ? h("ul", null, incident.map((e) => h("li", null,
        h("a", { href: "#", onclick: (ev) => { ev.preventDefault(); select("edge", e.id); } },
          `${e.sources.map(labelOf).join(" + ")} —${e.relation}→ ${e.targets.map(labelOf).join(" + ")}`),
        " ", edgeStatus(e.status)))) : h("p", { class: "muted" }, "No edges touch this concept."),
    ];
  }

  function edgeDetails(e, nodeById, select) {
    const runId = state.run.run_id;
    const pages = (state.artifacts && state.artifacts.pages) || { connected: [] };
    const labelOf = (id) => (nodeById.get(id) ? nodeById.get(id).label : id);
    const nodeLink = (id) => h("a", { href: "#", onclick: (ev) => { ev.preventDefault(); select("node", id); } }, labelOf(id));
    const ev = e.evidence;
    return [
      h("h2", null, e.sources.map((id, i) => [i ? " + " : "", nodeLink(id)]), h("span", { class: "muted" }, ` —${e.relation}→ `), e.targets.map((id, i) => [i ? " + " : "", nodeLink(id)])),
      h("p", null, edgeStatus(e.status),
        e.verdict && e.verdict !== e.status ? h("span", { class: "muted" }, ` · fused verdict ${e.verdict} (commit withheld)`) : "",
        e.confidence !== null && e.confidence !== undefined ? h("span", { class: "muted" }, ` · confidence ${fmtNum(e.confidence)}`) : "",
        e.direction ? h("span", { class: "muted" }, ` · ${e.direction}`) : ""),
      e.is_hypothesis ? h("p", { class: "small" }, h("span", { class: "badge on" }, e.candidate_id ? `hypothesis ${e.candidate_id}` : "hypothesis edge"),
        e.candidate_id && pages.trace_index ? [" ", h("a", { href: fileUrl(runId, `trace/${e.candidate_id}.html`), target: "_blank", rel: "noopener" }, "trace page")] : null,
        e.candidate_id && pages.connected.includes(`${e.candidate_id}-connected.html`) ? [" · ", h("a", { href: fileUrl(runId, `${e.candidate_id}-connected.html`), target: "_blank", rel: "noopener" }, "connected page")] : null) : null,
      e.mechanism ? h("p", null, h("strong", null, "Mechanism: "), e.mechanism) : null,
      e.conditions.length ? h("p", { class: "small" }, h("strong", null, "Conditions: "), e.conditions.join("; ")) : null,
      e.confounders.length ? h("p", { class: "small" }, h("strong", null, "Confounders: "), e.confounders.join("; ")) : null,
      e.open_risks.length ? [h("h3", null, "Open risks"), h("ul", null, e.open_risks.map((r) => h("li", null, r)))] : null,
      h("h3", null, `Evidence (${ev.length})`),
      ev.length ? ev.map((item) => h("div", { class: "ev" },
        h("div", null, h("span", { class: "badge on" }, item.role || "evidence"), " ",
          item.trust_tier ? h("span", { class: "badge" }, `tier ${item.trust_tier}`) : null, " ",
          item.association_score !== null && item.association_score !== undefined ? h("span", { class: "muted small" }, `association ${fmtNum(item.association_score)}`) : null),
        item.resolved ? h("div", null, isHttp(item.url) ? h("a", { href: item.url, target: "_blank", rel: "noopener" }, item.title || item.url) : (item.title || h("span", { class: "muted" }, "untitled")),
          item.source ? h("span", { class: "muted" }, ` · ${item.source}`) : "", item.published_date ? h("span", { class: "muted" }, ` · ${item.published_date}`) : "",
          item.ambiguous ? h("span", { class: "warn small", style: "margin-left:6px;padding:0 6px" }, "id matched several passages") : null) :
          h("div", { class: "muted small" }, `passage ${item.evidence_id} is not in the retrieval pool next to this graph`),
        item.quote ? h("p", { class: "quote" }, `“${truncate(item.quote, 600)}”`) : null,
        item.matched_quote_span && (!item.quote || !item.quote.includes(item.matched_quote_span)) ? h("p", { class: "small" }, h("strong", null, "Matched span: "), item.matched_quote_span) : null,
      )) : h("p", { class: "muted" }, e.status === "unverified" ? "No evidence review was scheduled for this edge (hypothesis edges are grounded by the experiment stage instead)." : "No evidence links recorded."),
      e.plan ? [h("h3", null, "Experiment plan"), planView(e.plan)] : null,
    ];
  }

  function planView(plan) {
    const order = ["hypothesis_under_test", "operationalization", "design", "design_rationale", "intervention_or_manipulation", "comparison_baseline", "controls_and_confounders", "materials_or_data", "metrics", "procedure", "expected_outcome", "falsification", "feasibility", "grounding", "grounding_status", "retrieval_batch_id"];
    const keys = [...order.filter((k) => plan[k] !== undefined && plan[k] !== null && plan[k] !== ""), ...Object.keys(plan).filter((k) => !order.includes(k))];
    return h("dl", { class: "kv" }, keys.map((k) => [h("dt", null, prettyKey(k)), h("dd", null, renderValue(plan[k]))]));
  }

  route();
})();
