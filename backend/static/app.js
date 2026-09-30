const API = "";  // same origin as this page

const el = (id) => document.getElementById(id);

function setStatus(message, kind) {
  const status = el("caseStatus");
  status.textContent = message;
  status.className = "case-status" + (kind ? ` status-${kind}` : "");
}

function showError(message) {
  const banner = el("errorBanner");
  banner.textContent = message;
  banner.hidden = false;
}

function hideError() {
  el("errorBanner").hidden = true;
}

async function openCase() {
  hideError();
  const indexDir = el("indexDir").value.trim() || "data/index";
  setStatus("Reading the case file…");

  try {
    const resp = await fetch(`${API}/api/open-case`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ index_dir: indexDir }),
    });
    const data = await resp.json();

    if (!resp.ok) {
      setStatus(data.detail || "Could not open case file.", "error");
      return;
    }

    setStatus(
      `${data.n_chunks} passages indexed\n${data.n_figures} figures/tables indexed`,
      "ok"
    );
  } catch (err) {
    setStatus(`Could not reach the backend: ${err.message}`, "error");
  }
}

function confidenceRingSVG(confidence, size = 88) {
  const radius = size / 2 - 6;
  const circumference = 2 * Math.PI * radius;
  const filled = circumference * Math.max(0, Math.min(confidence, 1));
  const color = confidence >= 0.6 ? "#4C8577" : "#B5541F";
  const center = size / 2;
  const pct = Math.round(confidence * 100);

  return `
    <svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
      <circle cx="${center}" cy="${center}" r="${radius}" fill="none" stroke="#2A323A" stroke-width="6"/>
      <circle cx="${center}" cy="${center}" r="${radius}" fill="none"
              stroke="${color}" stroke-width="6"
              stroke-dasharray="${filled.toFixed(1)} ${circumference.toFixed(1)}"
              stroke-linecap="round"
              transform="rotate(-90 ${center} ${center})"/>
      <text x="${center}" y="${center + 5}" text-anchor="middle"
            font-family="IBM Plex Mono, monospace" font-size="15" fill="${color}">${pct}%</text>
    </svg>`;
}

function renderTrace(trace) {
  const container = el("trace");
  container.innerHTML = "";
  for (const step of trace) {
    const isRetry = step.content.includes("Confidence below threshold");
    const div = document.createElement("div");
    div.className = "trace-step" + (isRetry ? " retry" : "");
    div.innerHTML = `
      <div class="trace-label">${escapeHTML(step.label)}</div>
      <div class="trace-content">${escapeHTML(step.content)}</div>`;
    container.appendChild(div);
  }
}

function renderFigures(figures) {
  const container = el("figures");
  container.innerHTML = "";
  if (!figures.length) {
    container.innerHTML = '<div class="no-figures-note">No figures matched this question.</div>';
    return;
  }
  for (const fig of figures) {
    const div = document.createElement("div");
    div.className = "figure-card";
    div.innerHTML = `
      <img src="${fig.url}" alt="${escapeHTML(fig.label)}">
      <div class="figure-caption">${escapeHTML(fig.label)} — ${escapeHTML(fig.source)} (match ${fig.score.toFixed(2)})</div>`;
    container.appendChild(div);
  }
}

function escapeHTML(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

async function investigate() {
  hideError();
  const question = el("question").value.trim();
  if (!question) return;

  const strategy = el("strategy").value;
  el("investigateBtn").disabled = true;
  el("loading").hidden = false;
  el("results").hidden = true;

  try {
    const resp = await fetch(`${API}/api/investigate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, strategy }),
    });
    const data = await resp.json();

    if (!resp.ok) {
      showError(data.detail || "The investigation failed.");
      return;
    }

    renderTrace(data.trace);
    renderFigures(data.figures);

    el("confidenceRing").innerHTML = confidenceRingSVG(data.confidence);
    el("finalAnswer").textContent = data.final_answer;
    const rounds = data.iterations === 1 ? "round" : "rounds";
    el("verdictMeta").textContent =
      `${data.iterations} investigation ${rounds} · confidence ${Math.round(data.confidence * 100)}%`;

    el("results").hidden = false;
  } catch (err) {
    showError(`Could not reach the backend: ${err.message}`);
  } finally {
    el("investigateBtn").disabled = false;
    el("loading").hidden = true;
  }
}

el("openCaseBtn").addEventListener("click", openCase);
el("investigateBtn").addEventListener("click", investigate);
el("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter") investigate();
});

// Check backend status on load, in case a case file is already open server-side.
(async function checkInitialStatus() {
  try {
    const resp = await fetch(`${API}/api/status`);
    const data = await resp.json();
    if (data.loaded) {
      setStatus(`${data.n_chunks} passages indexed\n${data.n_figures} figures/tables indexed`, "ok");
    }
  } catch (_) {
    // backend not reachable yet; leave default status message
  }
})();
