"use strict";

// ── Chart.js global defaults ──────────────────────────────────────────────
Chart.defaults.color = "#94a3b8";
Chart.defaults.borderColor = "#2e3350";
Chart.defaults.font.family = "'Segoe UI', system-ui, sans-serif";
Chart.defaults.font.size = 11;

const ACCENT   = "#4f8ef7";
const ACCENT2  = "#6ee7b7";
const ACCENT3  = "#f97316";
const SURFACE2 = "#22263a";

// Active chart instances — destroyed before re-render to avoid canvas reuse warning
const _charts = {};

function destroyChart(id) {
  if (_charts[id]) { _charts[id].destroy(); delete _charts[id]; }
}

// ── Slider live-update ────────────────────────────────────────────────────
const sliderMeta = [
  { id: "num_leads",              fmt: v => v },
  { id: "lead_to_mql_rate",       fmt: v => v + "%" },
  { id: "mql_to_sql_rate",        fmt: v => v + "%" },
  { id: "sql_to_opp_rate",        fmt: v => v + "%" },
  { id: "opp_win_rate",           fmt: v => v + "%" },
  { id: "days_lead_to_mql_mean",  fmt: v => v + " d" },
  { id: "days_mql_to_sql_mean",   fmt: v => v + " d" },
  { id: "days_sql_to_opp_mean",   fmt: v => v + " d" },
  { id: "days_opp_to_close_mean", fmt: v => v + " d" },
  { id: "days_lead_to_mql_std",   fmt: v => v + " d" },
  { id: "days_mql_to_sql_std",    fmt: v => v + " d" },
  { id: "days_sql_to_opp_std",    fmt: v => v + " d" },
  { id: "days_opp_to_close_std",  fmt: v => v + " d" },
];

sliderMeta.forEach(({ id, fmt }) => {
  const slider = document.getElementById(id);
  const label  = document.getElementById(id + "_val");
  if (!slider || !label) return;
  const sync = () => { label.textContent = fmt(slider.value); };
  slider.addEventListener("input", sync);
  sync();
});

// ── Defaults ──────────────────────────────────────────────────────────────
const DEFAULTS = {
  num_leads: 500,
  lead_to_mql_rate: 40, mql_to_sql_rate: 35,
  sql_to_opp_rate: 60,  opp_win_rate: 25,
  days_lead_to_mql_mean: 7,  days_mql_to_sql_mean: 14,
  days_sql_to_opp_mean: 10,  days_opp_to_close_mean: 45,
  days_lead_to_mql_std: 3,   days_mql_to_sql_std: 7,
  days_sql_to_opp_std: 5,    days_opp_to_close_std: 20,
};

function resetDefaults() {
  sliderMeta.forEach(({ id, fmt }) => {
    const slider = document.getElementById(id);
    const label  = document.getElementById(id + "_val");
    if (!slider) return;
    slider.value = DEFAULTS[id];
    if (label) label.textContent = fmt(DEFAULTS[id]);
  });
}

// ── Collect params ────────────────────────────────────────────────────────
function collectParams() {
  const p = {};
  sliderMeta.forEach(({ id }) => {
    const el = document.getElementById(id);
    if (el) p[id] = parseFloat(el.value);
  });
  return p;
}

// ── Main simulation call ──────────────────────────────────────────────────
async function runSimulation() {
  const btn = document.getElementById("run-btn");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Running…';

  try {
    const resp = await fetch("/simulate", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(collectParams()),
    });
    const json = await resp.json();
    if (!json.ok) throw new Error(json.error || "Unknown error");
    renderResults(json.data);
  } catch (err) {
    alert("Simulation error: " + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = "&#9654; Run Simulation";
  }
}

// ── Render all results ────────────────────────────────────────────────────
function renderResults(d) {
  document.getElementById("placeholder").style.display = "none";
  document.getElementById("results-content").style.display = "";

  renderKPIs(d);
  renderFunnel(d.funnel);
  renderHistogram("chart-crm-hist",   d.time_to_crm,   ACCENT,  "Time to CRM (days)");
  renderHistogram("chart-close-hist", d.time_to_close, ACCENT3, "Time to Close (days)");
  renderStatsTable("stats-crm",   d.time_to_crm.stats,   d.time_to_crm.count);
  renderStatsTable("stats-close", d.time_to_close.stats, d.time_to_close.count);
  renderCDF(d.time_to_crm.cdf);
  renderStageBreakdown(d.stage_durations);
  renderConvTable(d.conversion_rates, d.funnel);
}

// ── KPIs ──────────────────────────────────────────────────────────────────
function renderKPIs(d) {
  setText("#kpi-total-leads .kpi-value", d.funnel.Lead.toLocaleString());
  setText("#kpi-won .kpi-value", d.funnel.Won.toLocaleString());
  setText("#kpi-overall-rate .kpi-value", d.conversion_rates.Overall + "%");
  setText("#kpi-median-crm .kpi-value",
    d.time_to_crm.count ? d.time_to_crm.stats.median + " d" : "—");
  setText("#kpi-median-close .kpi-value",
    d.time_to_close.count ? d.time_to_close.stats.median + " d" : "—");
}

function setText(sel, val) {
  const el = document.querySelector(sel);
  if (el) el.textContent = val;
}

// ── Funnel horizontal bar chart ───────────────────────────────────────────
function renderFunnel(funnel) {
  destroyChart("funnel");
  const stages = Object.keys(funnel);
  const counts = Object.values(funnel);
  const maxVal = counts[0] || 1;
  const colors = ["#4f8ef7","#7c3aed","#06b6d4","#6ee7b7","#f97316"];

  _charts["funnel"] = new Chart(document.getElementById("chart-funnel"), {
    type: "bar",
    data: {
      labels: stages,
      datasets: [{
        data: counts,
        backgroundColor: colors,
        borderRadius: 4,
      }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          grid: { color: "#2e3350" },
          ticks: { color: "#94a3b8" },
          max: maxVal * 1.15,
        },
        y: { grid: { display: false }, ticks: { color: "#e2e8f0", font: { size: 12, weight: "600" } } },
      },
    },
  });
}

// ── Histogram ─────────────────────────────────────────────────────────────
function renderHistogram(canvasId, distData, color, xLabel) {
  destroyChart(canvasId);
  if (!distData.count) return;

  _charts[canvasId] = new Chart(document.getElementById(canvasId), {
    type: "bar",
    data: {
      labels: distData.histogram.labels,
      datasets: [{
        data: distData.histogram.values,
        backgroundColor: color + "99",
        borderColor: color,
        borderWidth: 1,
        borderRadius: 2,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          grid: { color: "#2e3350" },
          title: { display: true, text: xLabel, color: "#94a3b8" },
          ticks: { maxTicksLimit: 10, color: "#94a3b8" },
        },
        y: {
          grid: { color: "#2e3350" },
          title: { display: true, text: "Count", color: "#94a3b8" },
          ticks: { color: "#94a3b8" },
        },
      },
    },
  });
}

// ── Stats table ───────────────────────────────────────────────────────────
function renderStatsTable(tableId, stats, count) {
  const tbl = document.getElementById(tableId);
  if (!tbl) return;
  const cols = ["count","min","p10","p25","median","mean","p75","p90","max"];
  const vals = { count, ...stats };
  const labels = { count:"N", min:"Min", p10:"P10", p25:"P25",
                   median:"Median", mean:"Mean", p75:"P75", p90:"P90", max:"Max" };
  tbl.innerHTML = `
    <thead><tr>${cols.map(c => `<th>${labels[c]}</th>`).join("")}</tr></thead>
    <tbody><tr>${cols.map(c => {
      const v = vals[c];
      const highlight = (c === "median" || c === "mean") ? ' class="highlight"' : "";
      const display = c === "count" ? (v||0).toLocaleString() : (v ?? "—") + " d";
      return `<td${highlight}>${display}</td>`;
    }).join("")}</tr></tbody>
  `;
}

// ── CDF ───────────────────────────────────────────────────────────────────
function renderCDF(cdf) {
  destroyChart("cdf");
  if (!cdf.x.length) return;

  _charts["cdf"] = new Chart(document.getElementById("chart-cdf"), {
    type: "line",
    data: {
      labels: cdf.x.map(v => v.toFixed(1)),
      datasets: [{
        label: "Cumulative %",
        data: cdf.y,
        borderColor: ACCENT2,
        backgroundColor: ACCENT2 + "22",
        fill: true,
        tension: 0.35,
        pointRadius: 0,
        borderWidth: 2,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => `${ctx.raw.toFixed(1)}% of leads reach CRM within ${ctx.label} days`,
          },
        },
      },
      scales: {
        x: {
          grid: { color: "#2e3350" },
          title: { display: true, text: "Days", color: "#94a3b8" },
          ticks: { maxTicksLimit: 10, color: "#94a3b8" },
        },
        y: {
          grid: { color: "#2e3350" },
          title: { display: true, text: "Cumulative %", color: "#94a3b8" },
          min: 0, max: 100,
          ticks: { color: "#94a3b8", callback: v => v + "%" },
        },
      },
    },
  });
}

// ── Stage duration breakdown (floating bar = box-plot approximation) ───────
function renderStageBreakdown(stageDurations) {
  destroyChart("stages");
  const stages = Object.keys(stageDurations);
  const colors = [ACCENT, "#7c3aed", "#06b6d4", ACCENT3];

  // Floating bars: [q1, q3] with a median marker dataset
  const floatData  = stages.map(s => [stageDurations[s].q1, stageDurations[s].q3]);
  const medianData = stages.map(s => stageDurations[s].median);
  const minData    = stages.map(s => stageDurations[s].min);
  const maxData    = stages.map(s => stageDurations[s].max);

  _charts["stages"] = new Chart(document.getElementById("chart-stages"), {
    type: "bar",
    data: {
      labels: stages,
      datasets: [
        {
          label: "P25–P75 (IQR)",
          data: floatData,
          backgroundColor: colors.map(c => c + "66"),
          borderColor: colors,
          borderWidth: 2,
          borderRadius: 3,
        },
        {
          label: "Median",
          data: medianData,
          type: "scatter",
          pointStyle: "line",
          pointRadius: 12,
          pointBorderWidth: 3,
          borderColor: "#fff",
          backgroundColor: "#fff",
          showLine: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: true, labels: { color: "#94a3b8", boxWidth: 12 } },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const s = stages[ctx.dataIndex];
              const sd = stageDurations[s];
              if (ctx.dataset.label === "Median") return `Median: ${sd.median} d`;
              return `IQR: ${sd.q1}–${sd.q3} d  |  Range: ${sd.min}–${sd.max} d`;
            },
          },
        },
      },
      scales: {
        x: { grid: { display: false }, ticks: { color: "#e2e8f0" } },
        y: {
          grid: { color: "#2e3350" },
          title: { display: true, text: "Days", color: "#94a3b8" },
          ticks: { color: "#94a3b8" },
        },
      },
    },
  });
}

// ── Conversion rates table ────────────────────────────────────────────────
function renderConvTable(rates, funnel) {
  const tbl = document.getElementById("conv-table");
  if (!tbl) return;

  const rows = [
    { stage: "Lead → MQL",        key: "Lead→MQL",    from: funnel.Lead,        to: funnel.MQL },
    { stage: "MQL → SQL",         key: "MQL→SQL",     from: funnel.MQL,         to: funnel.SQL },
    { stage: "SQL → Opportunity", key: "SQL→Opp",     from: funnel.SQL,         to: funnel.Opportunity },
    { stage: "Opportunity → Won", key: "Opp→Won",     from: funnel.Opportunity, to: funnel.Won },
    { stage: "Overall (Lead → Won)", key: "Overall",  from: funnel.Lead,        to: funnel.Won },
  ];

  tbl.innerHTML = `
    <thead><tr>
      <th>Stage</th>
      <th>Input</th>
      <th>Output</th>
      <th>Rate</th>
      <th style="width:160px">Visualisation</th>
    </tr></thead>
    <tbody>
    ${rows.map(r => {
      const rate = rates[r.key] ?? 0;
      return `<tr>
        <td>${r.stage}</td>
        <td>${(r.from||0).toLocaleString()}</td>
        <td>${(r.to||0).toLocaleString()}</td>
        <td class="rate">${rate}%</td>
        <td><span class="conv-bar" style="width:${Math.min(rate,100)}%"></span></td>
      </tr>`;
    }).join("")}
    </tbody>
  `;
}
