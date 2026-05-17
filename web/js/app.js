// -----------------------------------------------------------------------------
// app.js — Dispersion-Assisted GS Phase Recovery viewer
//
// Loads /data/sample.json (same-origin only — CSP enforces this) and renders:
//   - 2D Plotly traces of GS error / phase / hybrid channels / IQ constellation
//   - a 3D VTK.js surface plot of |u(t)| over GS iterations
//
// No third-party state leaves the browser.  No remote fetches beyond the
// pinned, SRI-protected CDN scripts in index.html.
// -----------------------------------------------------------------------------

'use strict';

const DATA_URL = './data/sample.json';

// ---- tiny utility helpers ---------------------------------------------------

/** Safe JSON fetch with explicit content-type check.  Same-origin only. */
async function loadJson(url) {
  const res = await fetch(url, {
    cache: 'no-store',
    credentials: 'omit',
    redirect: 'error',
  });
  if (!res.ok) throw new Error(`HTTP ${res.status} loading ${url}`);
  const ct = res.headers.get('content-type') || '';
  if (!/json/i.test(ct)) {
    console.warn(`Unexpected content-type ${ct} from ${url}`);
  }
  return res.json();
}

/** Defensive shape check on the dataset so we fail loudly on malformed JSON. */
function validateDataset(d) {
  const need = (path, type) => {
    const parts = path.split('.');
    let cur = d;
    for (const p of parts) {
      if (cur == null || !(p in cur)) throw new Error(`Missing ${path}`);
      cur = cur[p];
    }
    if (type === 'array' && !Array.isArray(cur))
      throw new Error(`${path} must be an array`);
  };
  need('gs.t', 'array');
  need('gs.intensity_time', 'array');
  need('gs.phase_true', 'array');
  need('gs.phase_recovered', 'array');
  need('gs.errors', 'array');
  need('gs.snapshots', 'array');
  need('hybrid.channels_intensity', 'array');
  need('hybrid.I_balanced', 'array');
  need('hybrid.Q_balanced', 'array');
  return d;
}

// ---- 2D plots via Plotly ----------------------------------------------------

function plotError(gs) {
  Plotly.newPlot('errPlot', [{
    y: gs.errors, mode: 'lines', line: { width: 2 }
  }], {
    paper_bgcolor: '#161b22', plot_bgcolor: '#0d1117',
    font: { color: '#c9d1d9' },
    yaxis: { type: 'log', title: 'error' },
    xaxis: { title: 'iteration' },
    margin: { l: 60, r: 20, t: 10, b: 40 },
  }, { displaylogo: false, responsive: true });
}

function plotPhase(gs) {
  Plotly.newPlot('phasePlot', [
    { x: gs.t, y: gs.phase_true,      name: 'true',      mode: 'lines' },
    { x: gs.t, y: gs.phase_recovered, name: 'recovered', mode: 'lines',
      line: { dash: 'dash' } },
  ], {
    paper_bgcolor: '#161b22', plot_bgcolor: '#0d1117',
    font: { color: '#c9d1d9' },
    xaxis: { title: 't' }, yaxis: { title: 'phase (rad)' },
    margin: { l: 60, r: 20, t: 10, b: 40 },
  }, { displaylogo: false, responsive: true });
}

function plotHybridChannels(hy) {
  const traces = hy.channels_intensity.map((row, k) => ({
    y: row, mode: 'lines',
    name: `port ${k} (${k === 0 ? '0°' : k === 1 ? '180°' : k === 2 ? '90°' : '270°'})`,
  }));
  Plotly.newPlot('hybridChannels', traces, {
    paper_bgcolor: '#161b22', plot_bgcolor: '#0d1117',
    font: { color: '#c9d1d9' },
    xaxis: { title: 'sample' }, yaxis: { title: '|E|²' },
    margin: { l: 60, r: 20, t: 10, b: 40 },
  }, { displaylogo: false, responsive: true });
}

function plotConstellation(hy) {
  Plotly.newPlot('constellation', [{
    x: hy.I_balanced, y: hy.Q_balanced, mode: 'markers',
    marker: { size: 4 },
  }], {
    paper_bgcolor: '#161b22', plot_bgcolor: '#0d1117',
    font: { color: '#c9d1d9' },
    xaxis: { title: 'I', zerolinecolor: '#30363d' },
    yaxis: { title: 'Q', zerolinecolor: '#30363d', scaleanchor: 'x' },
    margin: { l: 60, r: 20, t: 10, b: 40 },
  }, { displaylogo: false, responsive: true });
}

// ---- Plotly 3D surface ------------------------------------------------------

function renderSurface3D(gs) {
  // 2D matrix of |u(t,iter)|: rows are GS snapshots, columns are time samples.
  // Plotly 'surface' renders it as a WebGL height field.
  const snaps = gs.snapshots;
  if (!snaps.length) return;
  const t     = gs.t;
  const iters = snaps.map(s => s[0]);
  const z     = snaps.map(s => s[1]);

  Plotly.newPlot('surface3d', [{
    type: 'surface',
    x: t, y: iters, z: z,
    colorscale: 'Viridis',
    contours: { z: { show: true, usecolormap: true,
                     project: { z: true }, width: 1 } },
    showscale: true,
  }], {
    paper_bgcolor: '#161b22',
    font: { color: '#c9d1d9' },
    margin: { l: 0, r: 0, t: 0, b: 0 },
    scene: {
      xaxis: { title: 't',             backgroundcolor: '#0d1117',
               gridcolor: '#30363d' },
      yaxis: { title: 'GS iteration',  backgroundcolor: '#0d1117',
               gridcolor: '#30363d' },
      zaxis: { title: '|u(t)|',        backgroundcolor: '#0d1117',
               gridcolor: '#30363d' },
      camera: { eye: { x: 1.8, y: -1.4, z: 1.2 } },
    },
  }, { displaylogo: false, responsive: true });
}

// ---- metadata pane ----------------------------------------------------------

function showMeta(d) {
  const m = {
    version:      d.metadata.version,
    description:  d.metadata.description,
    pulse_length: d.gs.t.length,
    iterations:   d.gs.errors.length,
    snapshots:    d.gs.snapshots.length,
    final_error:  d.gs.errors[d.gs.errors.length - 1],
    hybrid_IL_S:  d.hybrid.matrix_used.IL_S,
    hybrid_phi_SLO_rad: d.hybrid.matrix_used.phi_SLO,
  };
  document.getElementById('meta').textContent = JSON.stringify(m, null, 2);
}

// ---- main -------------------------------------------------------------------

(async () => {
  try {
    const data = validateDataset(await loadJson(DATA_URL));
    showMeta(data);
    plotError(data.gs);
    plotPhase(data.gs);
    plotHybridChannels(data.hybrid);
    plotConstellation(data.hybrid);
    if (typeof Plotly !== 'undefined') {
      renderSurface3D(data.gs);
    } else {
      document.getElementById('surface3d').textContent =
        'Plotly failed to load (SRI mismatch or offline).';
    }
  } catch (err) {
    console.error(err);
    document.body.insertAdjacentHTML(
      'afterbegin',
      `<div style="background:#f85149;color:white;padding:.5rem 1rem">
         Failed to load dataset: ${String(err).replace(/[<>&]/g, '')}
       </div>`
    );
  }
})();
