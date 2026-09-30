"use strict";

// ---- helpers
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = n => Number(n).toLocaleString("en-US");
const pad = n => String(n).padStart(2, "0");
const tc = s => {
  s = Math.max(0, Math.floor(s));
  const h = Math.floor(s / 3600), m = Math.floor(s / 60) % 60, x = s % 60;
  return h ? `${h}:${pad(m)}:${pad(x)}` : `${m}:${pad(x)}`;
};
const CHARS_PER_SEC = 15; // same as dubflow/script.py: what fits a second of dub
const MAX_TEMPO = 1.35; // same as dubflow/dub.py: fastest a line may be sped up to fit
const spoken = t => t.replace(/\[[^\]]*\]/g, "").replace(/\s+/g, " ").trim();
const ACTIVE = ["uploading", "queued", "running", "stopping"];
const PALETTE = ["#2F5DA8", "#C9962B", "#3A8F6B", "#B8546A", "#7A5BB5", "#2C8C9C", "#C06A2B", "#6B7F2E", "#A04B9B", "#587089"];
const LANES = 7; // speakers with their own lane on the reel; the rest share the last one

const STEPS = [
  ["extract", "Audio", "Pulling out the audio"],
  ["separate", "Voice split", "Separating voices from music"],
  ["transcribe", "Transcript", "Transcribing the English"],
  ["analyze", "Speakers", "Finding who speaks when"],
  ["script", "Translation", "Translating to Mongolian"],
  ["cast", "Casting", "Picking voices"],
  ["review", "Your review", "Ready for your review"],
  ["dub", "Voices", "Generating Mongolian voices"],
  ["mux", "Final video", "Building the final video"],
];
const STAGE_TEXT = Object.fromEntries(STEPS.map(([k, , t]) => [k, t]));

async function api(path, opts = {}) {
  const init = { method: opts.method || "GET", headers: {} };
  if (opts.json !== undefined) {
    init.body = JSON.stringify(opts.json);
    init.headers["Content-Type"] = "application/json";
  }
  const r = await fetch(path, init);
  if (r.status === 401) { location.href = "/login"; throw new Error("Sign in first."); }
  const data = (r.headers.get("content-type") || "").includes("json") ? await r.json() : null;
  if (!r.ok) {
    const d = data && data.detail;
    throw new Error(typeof d === "string" ? d : d ? JSON.stringify(d) : `${r.status} ${r.statusText}`);
  }
  return data;
}

let toastTimer;
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("on");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("on"), 3200);
}

function confirmBox(title, body, ok, spend = false) {
  const d = $("#confirm");
  $("#confirm-title").textContent = title;
  $("#confirm-body").textContent = body;
  const btn = $("#confirm-ok");
  btn.textContent = ok;
  btn.className = spend ? "spend" : "";
  d.returnValue = "";
  d.showModal();
  return new Promise(res => d.addEventListener("close", () => res(d.returnValue === "ok"), { once: true }));
}

function statusText(j) {
  switch (j.status) {
    case "uploading": return "Uploading";
    case "queued": return "Waiting to start";
    case "running": return STAGE_TEXT[j.stage] || (j.phase === "dub" ? "Dubbing" : "Starting");
    case "stopping": return "Stopping";
    case "review": return "Ready for review";
    case "done": return "Dubbed";
    case "failed": return "Failed";
    case "stopped": return "Stopped";
    default: return j.status;
  }
}

// ---- state
const S = { config: null, jobs: [], job: null, mode: null, script: null, dirty: new Map(), voices: {}, poll: null, stopAt: null };
const main = () => $("#main");

// ---- sidebar
function renderJobs() {
  const nav = $("#jobs");
  nav.innerHTML = S.jobs.length
    ? S.jobs.map(j => `<a href="#/job/${j.id}" class="job${S.job && S.job.id === j.id ? " on" : ""}" data-status="${esc(j.status)}">
        <span class="t">${esc(j.title)}</span><span class="s">${esc(statusText(j))}</span></a>`).join("")
    : `<p class="empty">Nothing dubbed yet.</p>`;
}
async function refreshJobs() {
  try {
    S.jobs = await api("/api/jobs");
    renderJobs();
  } catch { /* server restarting; next tick retries */ }
}

// ---- router
async function route() {
  clearTimeout(S.poll);
  await flush();
  const [, view, id] = location.hash.split("/");
  if (view === "job" && id) return showJob(id);
  if (view === "new" || !S.jobs.length) return showNew();
  location.hash = `#/job/${S.jobs[0].id}`;
}

// ---- new dub
const BLANK_FILM = {
  genre: "drama",
  premise: "Who the main characters are and what happens, in two or three sentences.",
  narrator: "the main character",
  names: { "Anna": "Анна" },
  aliases: { "Anna": "Anna, Mom (female lead)" },
  address: "Who says та and who says чи to whom.",
  notes: "",
};

function parseTime(v) {
  v = v.trim();
  if (!v) return null;
  const parts = v.split(":").map(Number);
  if (parts.some(isNaN)) throw new Error(`"${v}" isn't a time. Use seconds (90) or minutes:seconds (1:30).`);
  return parts.reduce((a, b) => a * 60 + b, 0);
}

function showNew() {
  S.job = null; S.mode = null; S.script = null;
  $("#dubbar")?.remove();
  renderJobs();
  const c = S.config, d = c.defaults;
  const films = c.films.map(f => `<option value="${esc(f)}">${esc(f)}</option>`).join("");
  const dist = c.distances.map(x => `<option ${x === d.distance ? "selected" : ""}>${x}</option>`).join("");
  main().innerHTML = `
  <section class="new-dub">
    <header>
      <h1>New dub</h1>
      <p class="muted">Upload an English video. Transcription and translation start right away. Mongolian voices are
        only generated after you've reviewed the script and confirmed the cost.</p>
    </header>
    ${c.missing_keys.length ? `<div class="notice warn"><strong>Missing in .env: ${c.missing_keys.map(esc).join(", ")}</strong>
      <p>Stages that need these keys will fail until you add them and restart the server.</p></div>` : ""}
    <form id="newform">
      <label class="drop" id="drop">
        <input type="file" name="video" accept="video/*,.mkv" required>
        <strong id="dropname">Drop a video here or choose a file</strong>
        <span class="hint" id="dropinfo">MP4, MOV, MKV or WebM</span>
      </label>
      <label class="field">Title
        <input name="title" placeholder="Taken from the file name if empty" autocomplete="off">
      </label>
      <div class="field">
        <label class="field">Film profile
          <small>Character names with their Mongolian spelling, who is who, and how people address each other.
            The translator follows it, so a good profile gives a much better script.</small>
          <select id="filmpick"><option value="">Start from a blank profile</option>${films}</select>
        </label>
        <textarea class="film-json" id="filmjson" spellcheck="false" aria-label="Film profile JSON">${esc(JSON.stringify(BLANK_FILM, null, 1))}</textarea>
      </div>
      <div class="pair">
        <label class="field">Clip start <small>Try a few minutes first. Empty = from the beginning.</small>
          <input name="start" placeholder="e.g. 2:00" inputmode="numeric"></label>
        <label class="field">Clip end <small>Empty = to the end of the video.</small>
          <input name="end" placeholder="e.g. 5:00" inputmode="numeric"></label>
      </div>
      <details class="adv">
        <summary>Advanced settings</summary>
        <div class="grid3">
          <label class="field">Translation model <input name="model" value="${esc(d.model)}"></label>
          <label class="field">Polish model <small>Stronger model = more natural lines</small>
            <input name="polish_model" value="${esc(d.polish_model || "")}" placeholder="same as translation"></label>
          <label class="field">Voice model <input name="tts_model" value="${esc(d.tts_model)}"></label>
          <label class="field">Stability <small>Lower = more expressive</small>
            <input name="stability" type="number" min="0" max="1" step="0.05" value="${d.stability}"></label>
          <label class="field">Mic distance <select name="distance">${dist}</select></label>
          <label class="field">Number of speakers <small>If you know it</small>
            <input name="num_speakers" type="number" min="1" max="40" placeholder="auto"></label>
          <label class="check"><input type="checkbox" name="no_separate"> Don't separate voices from music</label>
        </div>
      </details>
      <div class="submit">
        <button type="submit" id="go">Upload and start</button>
        <div class="progress" id="prog" hidden style="flex:1;min-width:160px"><i></i></div>
        <span class="hint" id="progtext"></span>
      </div>
    </form>
  </section>`;

  const form = $("#newform"), drop = $("#drop"), file = form.video;
  const showFile = () => {
    const f = file.files[0];
    drop.classList.toggle("has", !!f);
    $("#dropname").textContent = f ? f.name : "Drop a video here or choose a file";
    $("#dropinfo").textContent = f ? `${(f.size / 1e6).toFixed(0)} MB` : "MP4, MOV, MKV or WebM";
  };
  file.addEventListener("change", showFile);
  drop.addEventListener("dragover", e => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", e => {
    e.preventDefault();
    drop.classList.remove("over");
    if (e.dataTransfer.files.length) { file.files = e.dataTransfer.files; showFile(); }
  });
  $("#filmpick").addEventListener("change", async e => {
    const name = e.target.value;
    const film = name ? await api(`/api/films/${encodeURIComponent(name)}`) : BLANK_FILM;
    if (film.title && !form.title.value) form.title.value = film.title;
    const { title, ...rest } = film;
    $("#filmjson").value = JSON.stringify(rest, null, 1);
  });
  form.addEventListener("submit", async e => {
    e.preventDefault();
    let film, options;
    try {
      film = JSON.parse($("#filmjson").value || "{}");
      options = {
        start: parseTime(form.start.value), end: parseTime(form.end.value),
        model: form.model.value.trim(), polish_model: form.polish_model.value.trim(),
        tts_model: form.tts_model.value.trim(), stability: form.stability.value, distance: form.distance.value,
        num_speakers: form.num_speakers.value, no_separate: form.no_separate.checked,
      };
    } catch (err) {
      return toast(err instanceof SyntaxError ? `Film profile isn't valid JSON: ${err.message}` : err.message);
    }
    const fd = new FormData();
    fd.append("video", file.files[0]);
    fd.append("title", form.title.value);
    fd.append("film", JSON.stringify(film));
    fd.append("options", JSON.stringify(options));
    $("#go").disabled = true;
    $("#prog").hidden = false;
    try {
      const job = await upload(fd, p => {
        $("#prog i").style.width = `${p * 100}%`;
        $("#progtext").textContent = p < 1 ? `Uploading ${Math.round(p * 100)}%` : "Saving the video…";
      });
      await refreshJobs();
      location.hash = `#/job/${job.id}`;
    } catch (err) {
      toast(err.message);
      $("#go").disabled = false;
      $("#prog").hidden = true;
      $("#progtext").textContent = "";
    }
  });
}

function upload(fd, onProgress) {
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest();
    x.open("POST", "/api/jobs");
    x.upload.onprogress = e => e.lengthComputable && onProgress(e.loaded / e.total);
    x.onload = () => {
      if (x.status === 401) { location.href = "/login"; return; }
      let d = null;
      try { d = JSON.parse(x.responseText); } catch { /* not JSON */ }
      x.status < 300 ? resolve(d) : reject(new Error((d && d.detail) || `Upload failed (${x.status})`));
    };
    x.onerror = () => reject(new Error("Upload failed: the server didn't answer."));
    x.send(fd);
  });
}

// ---- job
async function showJob(id) {
  let job;
  try {
    job = await api(`/api/jobs/${id}`);
  } catch (e) {
    S.job = null; S.mode = null;
    main().innerHTML = `<div class="notice error"><strong>${esc(e.message)}</strong><p><a href="#/new">Start a new dub</a></p></div>`;
    return;
  }
  if (!S.job || S.job.id !== id) {
    S.mode = null; S.script = null; S.voices = {};
    main().innerHTML = `<section class="job-view"><div id="jhead"></div><ol class="steps" id="steps"></ol>
      <div id="jfilm"></div><div id="jbody"></div></section>`;
  }
  S.job = job;
  renderJob();
  if (ACTIVE.includes(job.status)) S.poll = setTimeout(() => showJob(id), 2000);
}

function jobMode(j) {
  if (ACTIVE.includes(j.status)) return "active";
  if (j.status === "failed" || j.status === "stopped") return j.has_script ? "problem-script" : "problem";
  return j.has_script ? "script" : "problem";
}

function renderJob() {
  const j = S.job;
  renderJobs();
  $("#jhead").innerHTML = `<header class="jhead">
    <div class="title"><h1>${esc(j.title)}</h1><p class="status s-${esc(j.status)}">${esc(statusText(j))}</p></div>
    <div class="actions">
      ${j.has_final ? `<a class="button" href="/api/jobs/${j.id}/video/final?download=true">Download Mongolian video</a>` : ""}
      ${["running", "queued"].includes(j.status) ? `<button class="ghost" data-act="stop">Stop</button>` : ""}
      ${["failed", "stopped"].includes(j.status) ? `<button data-act="resume">Resume</button>` : ""}
      ${!ACTIVE.includes(j.status) ? `<button class="danger" data-act="delete">Delete</button>` : ""}
    </div></header>`;
  renderSteps();

  const mode = jobMode(j);
  if (mode !== S.mode) {
    S.mode = mode;
    $("#jfilm").innerHTML = mode === "active" ? "" : filmPanel();
    const body = $("#jbody");
    if (mode === "active") body.innerHTML = runPanel();
    else if (mode === "problem") body.innerHTML = problemPanel() + logPanel();
    else {
      body.innerHTML = (mode === "problem-script" ? problemPanel() : "") + `<div id="editor"><p class="muted">Loading the script…</p></div>`;
      loadEditor();
    }
    $("#dubbar")?.remove();
  }
  if (mode === "active") {
    $("#runtitle").textContent = j.status === "queued" ? "Waiting for the job ahead to finish" : statusText(j);
    const log = $("#log"), atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 30;
    log.textContent = j.log.join("\n");
    if (atEnd) log.scrollTop = log.scrollHeight;
  }
  if (mode.includes("problem")) $("#problem").textContent = j.error || "The pipeline stopped. See the log below.";
  renderDubBar();
}

function renderSteps() {
  const j = S.job;
  const steps = STEPS.filter(([k]) => !(k === "separate" && j.options && j.options.no_separate));
  const keys = steps.map(([k]) => k);
  let cur, cls;
  if (j.status === "done") cur = keys.length;
  else if (j.status === "review") { cur = keys.indexOf("review"); cls = "you"; }
  else {
    cur = keys.indexOf(j.stage) >= 0 ? keys.indexOf(j.stage) : keys.indexOf(j.phase === "dub" ? "dub" : "extract");
    cls = ["failed", "stopped"].includes(j.status) ? "bad" : "now";
  }
  $("#steps").innerHTML = steps.map(([, label], i) =>
    `<li class="${i < cur ? "done" : i === cur ? `now ${cls}` : ""}">${esc(label)}</li>`).join("");
}

function runPanel() {
  return `<section class="panel run">
    <h2 id="runtitle"></h2>
    <p class="muted">You can close this page. The work carries on on the server, and a full film takes a while.</p>
    <pre class="log" id="log"></pre></section>`;
}
function problemPanel() {
  return `<div class="notice error"><strong id="problem"></strong>
    <p>Fix the cause (a missing API key in .env, for example), then press Resume. Finished stages aren't repeated.</p></div>`;
}
function logPanel() {
  return `<details class="film"><summary>Log</summary><pre class="log">${esc(S.job.log.join("\n"))}</pre></details>`;
}
function filmPanel() {
  return `<details class="film" id="filmbox"><summary>Film profile</summary>
    <p class="muted">Names, who is who and how characters address each other. The translator reads this. After changing it,
      retranslate to rewrite the script.</p>
    <textarea class="film-json" id="filmedit" spellcheck="false" aria-label="Film profile JSON"></textarea>
    <div class="film-actions"><button id="savefilm">Save profile</button>
      <button class="ghost" id="retranslate">Retranslate the script</button></div></details>`;
}

// ---- actions (delegated)
document.addEventListener("click", async e => {
  const act = e.target.closest("[data-act]")?.dataset.act;
  const j = S.job;
  if (!act || !j) return;
  try {
    if (act === "stop") { await api(`/api/jobs/${j.id}/stop`, { method: "POST" }); toast("Stopping"); }
    if (act === "resume") { await api(`/api/jobs/${j.id}/resume`, { method: "POST" }); toast("Resumed"); }
    if (act === "delete") {
      const ok = await confirmBox("Delete this dub?",
        "Removes the job with its uploaded video, script and mixed audio. Voice lines already generated stay cached, so dubbing the same lines again later costs nothing.",
        "Delete");
      if (!ok) return;
      await api(`/api/jobs/${j.id}`, { method: "DELETE" });
      S.job = null; S.mode = null;
      toast("Deleted");
      await refreshJobs();
      location.hash = S.jobs.length ? `#/job/${S.jobs[0].id}` : "#/new";
      return;
    }
    clearTimeout(S.poll);
    await refreshJobs();
    showJob(j.id);
  } catch (err) { toast(err.message); }
});

document.addEventListener("toggle", async e => {
  if (e.target.id !== "filmbox" || !e.target.open || $("#filmedit").value) return;
  const film = await api(`/api/jobs/${S.job.id}/film`);
  $("#filmedit").value = JSON.stringify(film, null, 1);
}, true);

document.addEventListener("click", async e => {
  if (e.target.id !== "savefilm" && e.target.id !== "retranslate") return;
  const j = S.job;
  try {
    const film = JSON.parse($("#filmedit").value || "{}");
    await api(`/api/jobs/${j.id}/film`, { method: "PUT", json: film });
    if (e.target.id === "savefilm") return toast("Profile saved");
    const ok = await confirmBox("Retranslate the whole script?",
      "Your current script, with your edits, is kept as a backup file in the job folder. Then the script is rewritten from the English with this profile. Voices aren't touched until you dub again.",
      "Retranslate");
    if (!ok) return;
    await flush();
    const r = await api(`/api/jobs/${j.id}/retranslate`, { method: "POST" });
    toast(r.backup ? `Backup saved as ${r.backup}` : "Retranslating");
    S.mode = null;
    showJob(j.id);
  } catch (err) {
    toast(err instanceof SyntaxError ? `Film profile isn't valid JSON: ${err.message}` : err.message);
  }
});

// ---- editor
const colorOf = name => (S.script && S.script.colors[name]) || PALETTE[PALETTE.length - 1];

async function loadEditor() {
  const j = S.job;
  try {
    const sc = await api(`/api/jobs/${j.id}/script`);
    sc.colors = {};
    sc.lanes = {};
    sc.speakers.forEach((s, i) => { sc.colors[s.name] = PALETTE[i % PALETTE.length]; sc.lanes[s.name] = Math.min(i, LANES - 1); });
    S.script = sc;
  } catch (e) {
    $("#editor").innerHTML = `<div class="notice error"><strong>${esc(e.message)}</strong></div>`;
    return;
  }
  renderEditor();
}

function renderEditor() {
  const j = S.job, sc = S.script;
  const lanes = Math.min(sc.speakers.length, LANES) || 1;
  const speakerOpts = sc.speakers.map(s => `<option value="${esc(s.name)}">${esc(s.name)} (${s.lines})</option>`).join("");
  $("#editor").innerHTML = `
    <div class="job-view">
      <section class="stage">
        <div class="player">
          ${j.has_final ? `<div class="switch" role="group" aria-label="Soundtrack">
            <button type="button" data-src="source" aria-pressed="true">English original</button>
            <button type="button" data-src="final" aria-pressed="false">Mongolian dub</button></div>` : ""}
          <video id="vid" src="/api/jobs/${j.id}/video/source" controls preload="metadata"></video>
        </div>
        <aside class="panel cast" id="cast"></aside>
      </section>
      <div class="reel" id="reel" style="--lanes:${lanes}" title="Every line, by speaker. Click to jump there.">
        ${sc.lines.map(l => `<i data-i="${l.i}" style="--c:${colorOf(l.speaker)};top:${sc.lanes[l.speaker] * 9}px"></i>`).join("")}
        <div class="head" id="head"></div>
      </div>
      <div class="tools">
        <input type="search" id="q" placeholder="Search the English or Mongolian" aria-label="Search lines">
        <select id="who" aria-label="Show lines of"><option value="">Everyone</option>${speakerOpts}</select>
        <label class="check"><input type="checkbox" id="follow" checked> Follow video</label>
        <span class="save-state" id="savestate">${sc.lines.length} lines</span>
      </div>
      <div class="rows" id="rows">${sc.lines.map(rowHtml).join("")}</div>
      <datalist id="spk">${sc.speakers.map(s => `<option value="${esc(s.name)}">`).join("")}</datalist>
    </div>`;
  renderCast();
  placeReel();
  wireEditor();
  renderDubBar();
}

// Like dub.py: a line may run on until the next one starts, and be sped up a little to fit.
function maxChars(l) {
  const next = S.script.lines[l.i + 1];
  const slot = Math.max(next ? next.start - l.start - 0.05 : l.end - l.start + 2, l.end - l.start);
  return Math.max(10, Math.floor(Math.min(slot, 12) * CHARS_PER_SEC));
}
function lenHtml(l) {
  const n = spoken(l.text).length, max = maxChars(l);
  return `<span class="len${n > max * MAX_TEMPO ? " over" : ""}" title="Spoken characters / what fits before the next line">${n} / ${max}</span>`;
}
function rowHtml(l) {
  return `<div class="row" data-i="${l.i}" style="--c:${colorOf(l.speaker)}">
    <button type="button" class="tc" title="Play this line (Ctrl+Enter while editing)">${tc(l.start)}</button>
    <input class="spk" list="spk" value="${esc(l.speaker)}" aria-label="Speaker">
    <p class="en">${esc(l.en)}</p>
    <div class="mnw"><textarea class="mn" rows="2" lang="mn" aria-label="Mongolian line">${esc(l.text)}</textarea>${lenHtml(l)}</div>
  </div>`;
}

function renderCast() {
  const sc = S.script, pool = sc.pool;
  const known = new Set(pool.map(v => v.voice_id));
  const rows = sc.speakers.map(s => {
    const pending = S.voices[s.name];
    const entry = pending || sc.voices[s.name];
    const vid = entry ? (typeof entry === "string" ? entry : entry.voice_id) : "";
    const pitch = entry && typeof entry === "object" ? entry.pitch || 0 : 0;
    const opts = (vid ? "" : `<option value="" selected>Picked when dubbing</option>`)
      + (vid && !known.has(vid) ? `<option value="${esc(vid)}" selected>${esc(vid)}</option>` : "")
      + pool.map(v => `<option value="${esc(v.voice_id)}" ${v.voice_id === vid ? "selected" : ""}>${esc(v.name)} (${esc(v.gender)}, ${esc(v.age)})</option>`).join("");
    return `<div class="voice" style="--c:${colorOf(s.name)}"><span class="dot"></span><span class="nm">${esc(s.name)}</span>
      <span class="n">${s.lines} lines</span>
      <div class="pick"><select data-voice="${esc(s.name)}" aria-label="Voice for ${esc(s.name)}">${opts}</select>
      <input type="number" min="-12" max="12" step="1" value="${pitch}" data-pitch="${esc(s.name)}" title="Pitch shift in semitones" aria-label="Pitch for ${esc(s.name)}"></div></div>`;
  }).join("");
  const unsaved = Object.keys(S.voices).length;
  $("#cast").innerHTML = `<header><h2>Voices</h2>${unsaved ? `<button id="savevoices">Save voices</button>` : ""}</header>
    ${pool.length ? "" : `<p class="muted">No voice pool yet. It's made from your ElevenLabs Mongolian voices on the first dub.</p>`}${rows}`;
}

function placeReel() {
  const v = $("#vid"), sc = S.script;
  const last = sc.lines.length ? sc.lines[sc.lines.length - 1].end : 1;
  const total = Math.max(v.duration || 0, last);
  sc.total = total;
  $$("#reel i").forEach(el => {
    const l = sc.lines[el.dataset.i];
    el.style.left = `${(l.start / total) * 100}%`;
    el.style.width = `max(2px, ${((l.end - l.start) / total) * 100}%)`;
  });
}

function lineAt(t) {
  const L = S.script.lines;
  let lo = 0, hi = L.length - 1, ans = -1;
  while (lo <= hi) {
    const m = (lo + hi) >> 1;
    if (L[m].start <= t) { ans = m; lo = m + 1; } else hi = m - 1;
  }
  return ans >= 0 && t <= L[ans].end + 0.3 ? ans : -1;
}

let nowRow = -1;
function markNow(i, scroll) {
  if (i === nowRow) return;
  $(`#rows .row[data-i="${nowRow}"]`)?.classList.remove("now");
  $(`#reel i[data-i="${nowRow}"]`)?.classList.remove("hit");
  nowRow = i;
  if (i < 0) return;
  const row = $(`#rows .row[data-i="${i}"]`);
  row?.classList.add("now");
  $(`#reel i[data-i="${i}"]`)?.classList.add("hit");
  if (scroll && row && !row.hidden) row.scrollIntoView({ block: "center", behavior: "smooth" });
}

function playLine(i) {
  const v = $("#vid"), l = S.script.lines[i];
  v.currentTime = l.start;
  S.stopAt = l.end + 0.2;
  markNow(i, false);
  v.play().catch(() => {});
}

function wireEditor() {
  const v = $("#vid"), rows = $("#rows");
  v.addEventListener("loadedmetadata", placeReel);
  v.addEventListener("timeupdate", () => {
    const t = v.currentTime;
    if (S.stopAt !== null && t >= S.stopAt) { v.pause(); S.stopAt = null; }
    $("#head").style.left = `${(t / S.script.total) * 100}%`;
    const editing = document.activeElement && rows.contains(document.activeElement);
    markNow(lineAt(t), $("#follow").checked && !v.paused && !editing);
  });
  v.addEventListener("seeking", () => { if (S.stopAt !== null && Math.abs(v.currentTime - S.stopAt) > 30) S.stopAt = null; });

  $("#reel").addEventListener("click", e => {
    const r = e.currentTarget.getBoundingClientRect();
    const t = ((e.clientX - r.left) / r.width) * S.script.total;
    v.currentTime = t;
    S.stopAt = null;
    const i = lineAt(t);
    markNow(i, false);
    if (i >= 0) $(`#rows .row[data-i="${i}"]`)?.scrollIntoView({ block: "center", behavior: "smooth" });
  });

  $$(".switch button").forEach(b => b.addEventListener("click", () => {
    const t = v.currentTime, playing = !v.paused;
    $$(".switch button").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
    v.src = `/api/jobs/${S.job.id}/video/${b.dataset.src}`;
    v.addEventListener("loadedmetadata", () => { v.currentTime = t; if (playing) v.play(); }, { once: true });
  }));

  rows.addEventListener("click", e => {
    const b = e.target.closest(".tc");
    if (b) playLine(+b.closest(".row").dataset.i);
  });
  rows.addEventListener("keydown", e => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); playLine(+e.target.closest(".row").dataset.i); }
  });
  rows.addEventListener("input", e => {
    const row = e.target.closest(".row"), i = +row.dataset.i, l = S.script.lines[i];
    const edit = S.dirty.get(i) || { i };
    if (e.target.classList.contains("mn")) {
      l.text = edit.text = e.target.value;
      row.querySelector(".len").outerHTML = lenHtml(l);
    } else if (e.target.classList.contains("spk")) {
      l.speaker = edit.speaker = e.target.value.trim();
    } else return;
    S.dirty.set(i, edit);
    scheduleSave();
  });
  rows.addEventListener("change", e => {
    if (!e.target.classList.contains("spk")) return;
    const row = e.target.closest(".row"), name = e.target.value.trim();
    if (!S.script.colors[name]) {
      S.script.colors[name] = PALETTE[Object.keys(S.script.colors).length % PALETTE.length];
      S.script.lanes[name] = LANES - 1;
    }
    row.style.setProperty("--c", colorOf(name));
    const bar = $(`#reel i[data-i="${row.dataset.i}"]`);
    bar.style.setProperty("--c", colorOf(name));
    bar.style.top = `${S.script.lanes[name] * 9}px`;
  });

  const filter = () => {
    const q = $("#q").value.trim().toLowerCase(), who = $("#who").value;
    let shown = 0;
    $$("#rows .row").forEach(row => {
      const l = S.script.lines[row.dataset.i];
      const ok = (!who || l.speaker === who) && (!q || `${l.en}\n${l.text}\n${l.speaker}`.toLowerCase().includes(q));
      row.hidden = !ok;
      shown += ok;
    });
    setSaveState(q || who ? `${shown} of ${S.script.lines.length} lines` : "");
  };
  $("#q").addEventListener("input", filter);
  $("#who").addEventListener("change", filter);

  $("#cast").addEventListener("change", e => {
    const name = e.target.dataset.voice || e.target.dataset.pitch;
    if (!name) return;
    const box = e.target.closest(".voice");
    const vid = box.querySelector("select").value, pitch = +box.querySelector("input").value || 0;
    if (!vid) return toast("Pick a voice first, then the pitch.");
    S.voices[name] = { voice_id: vid, pitch };
    renderCast();
  });
  $("#cast").addEventListener("click", async e => {
    if (e.target.id !== "savevoices") return;
    try {
      const r = await api(`/api/jobs/${S.job.id}/voices`, { method: "PUT", json: S.voices });
      S.script.voices = r.voices;
      S.voices = {};
      S.job.estimate = r.estimate;
      renderCast();
      renderDubBar();
      toast("Voices saved");
    } catch (err) { toast(err.message); }
  });
}

// ---- saving script edits (automatic)
let saveTimer;
function setSaveState(text, bad = false) {
  const el = $("#savestate");
  if (!el) return;
  el.textContent = text;
  el.classList.toggle("bad", bad);
}
function scheduleSave() {
  clearTimeout(saveTimer);
  setSaveState("Editing…");
  saveTimer = setTimeout(flush, 1200);
}
async function flush() {
  clearTimeout(saveTimer);
  if (!S.dirty.size || !S.job) return true;
  const edits = [...S.dirty.values()];
  S.dirty.clear();
  setSaveState("Saving…");
  try {
    const r = await api(`/api/jobs/${S.job.id}/script`, { method: "PUT", json: { edits } });
    S.job.estimate = r.estimate;
    renderDubBar();
    setSaveState("All changes saved");
    return true;
  } catch (e) {
    edits.forEach(ed => S.dirty.set(ed.i, { ...ed, ...(S.dirty.get(ed.i) || {}) }));
    setSaveState(`Not saved: ${e.message}`, true);
    return false;
  }
}
window.addEventListener("beforeunload", e => { if (S.dirty.size) { flush(); e.preventDefault(); } });

// ---- the one button that spends money
function renderDubBar() {
  let bar = $("#dubbar");
  const j = S.job;
  if (!j || !S.mode || !S.mode.includes("script") || !j.estimate) { bar?.remove(); return; }
  if (!bar) {
    bar = document.createElement("div");
    bar.id = "dubbar";
    bar.className = "dubbar";
    document.body.append(bar);
    bar.addEventListener("click", e => e.target.id === "dubgo" && startDub());
  }
  const e = j.estimate, again = j.status === "done";
  const text = e.lines
    ? `<strong>${fmt(e.lines)} ${e.lines === 1 ? "line needs" : "lines need"} a voice</strong>, about ${fmt(e.credits)} ElevenLabs credits. Lines already voiced are reused for free.`
    : `Every line already has a voice. ${again ? "Dubbing again only rebuilds the video." : "Dubbing costs no credits."}`;
  bar.innerHTML = `<p>${text}</p><button id="dubgo" class="${e.lines ? "spend" : ""}">${!e.lines ? "Rebuild the video" : again ? "Dub the changes" : "Dub the film"}</button>`;
}

async function startDub() {
  if (!(await flush())) return toast("Your edits couldn't be saved. Fix that first.");
  if (Object.keys(S.voices).length) return toast("Save or undo your voice changes first.");
  const j = await api(`/api/jobs/${S.job.id}`);
  const e = j.estimate;
  const ok = await confirmBox(
    e.lines ? `Spend about ${fmt(e.credits)} credits?` : "Rebuild the video?",
    e.lines ? `ElevenLabs will voice ${fmt(e.lines)} lines, then the dub is mixed into the video. Characters without a voice get one from your pool automatically.`
      : "No new voice lines are needed. The audio is remixed and the video rebuilt.",
    e.lines ? `Spend ${fmt(e.credits)} credits` : "Rebuild", !!e.lines);
  if (!ok) return;
  try {
    await api(`/api/jobs/${j.id}/dub`, { method: "POST", json: { credits: e.credits } });
    S.mode = null;
    await refreshJobs();
    showJob(j.id);
  } catch (err) { toast(err.message); }
}

// ---- boot
(async () => {
  try {
    S.config = await api("/api/config");
  } catch (e) {
    main().innerHTML = `<div class="notice error"><strong>Can't reach the server: ${esc(e.message)}</strong></div>`;
    return;
  }
  if (S.config.login) $(".side").insertAdjacentHTML("beforeend", `<a class="signout" href="/logout">Sign out</a>`);
  await refreshJobs();
  setInterval(refreshJobs, 4000);
  window.addEventListener("hashchange", route);
  route();
})();
