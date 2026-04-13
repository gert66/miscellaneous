import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Connect Four",
    page_icon="🔴",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Strip all Streamlit chrome so only the component is visible
st.markdown(
    """
<style>
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"]  { display: none; }
[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}
section.main > div.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Entire game lives in a single self-contained HTML/JS/CSS component ─────────
# The component handles hover, animation, win detection, and score tracking
# entirely in the browser — no Streamlit round-trips needed.
GAME_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@700;800;900&display=swap');

/* ── Layout tokens (must match JS constants CS/GAP/BP/PREV_H) ── */
:root {
  --cs:   66px;   /* cell size          */
  --gap:  7px;    /* gap between cells  */
  --bp:   13px;   /* board padding      */
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html {
  background: linear-gradient(160deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}
body {
  background: transparent;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  font-family: 'Nunito', sans-serif;
  padding: 22px 16px 18px;
}

/* ── Title ── */
#title {
  font-size: 2.8rem;
  font-weight: 900;
  background: linear-gradient(90deg, #ff6b6b 0%, #ffd93d 50%, #ff6b6b 100%);
  background-size: 200% auto;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  animation: shimmer 3s linear infinite;
  margin-bottom: 2px;
  letter-spacing: -1px;
}
@keyframes shimmer { to { background-position: 200% center; } }

.subtitle {
  color: rgba(255,255,255,0.3);
  font-size: 0.68rem;
  letter-spacing: 3px;
  text-transform: uppercase;
  margin-bottom: 16px;
}

/* ── Score cards ── */
#scores {
  display: flex;
  gap: 14px;
  margin-bottom: 14px;
}
.sc {
  background: rgba(255,255,255,0.06);
  border-radius: 14px;
  padding: 8px 28px;
  text-align: center;
  border: 2px solid rgba(255,255,255,0.08);
  color: white;
  min-width: 110px;
  transition: border-color 0.35s ease, box-shadow 0.35s ease;
}
.sc .lbl {
  font-size: 0.7rem; font-weight: 800;
  letter-spacing: 1px; text-transform: uppercase;
  opacity: 0.5; margin-bottom: 2px;
}
.sc .num { font-size: 2rem; font-weight: 900; line-height: 1; }
#sc-red            { border-color: rgba(255,71,87,0.25); }
#sc-yellow         { border-color: rgba(255,217,61,0.25); }
#sc-red.active     { border-color: rgba(255,71,87,0.8);  box-shadow: 0 0 18px rgba(255,71,87,0.3); }
#sc-yellow.active  { border-color: rgba(255,217,61,0.8); box-shadow: 0 0 18px rgba(255,217,61,0.3); }

/* ── Turn indicator ── */
#turn {
  background: rgba(255,255,255,0.07);
  border: 2px solid rgba(255,255,255,0.1);
  border-radius: 16px;
  padding: 10px 32px;
  color: white;
  font-size: 1.1rem;
  font-weight: 800;
  margin-bottom: 16px;
  min-width: 210px;
  text-align: center;
  transition: border-color 0.3s ease, box-shadow 0.3s ease, color 0.3s ease;
}
#turn.t-red    { border-color: rgba(255,71,87,0.75);  box-shadow: 0 0 24px rgba(255,71,87,0.3); }
#turn.t-yellow { border-color: rgba(255,217,61,0.75); box-shadow: 0 0 24px rgba(255,217,61,0.3); }
#turn.t-win    {
  border-color: gold; box-shadow: 0 0 32px rgba(255,215,0,0.45);
  color: gold; animation: turnPulse 1.2s ease-in-out infinite;
}
#turn.t-draw { border-color: rgba(255,255,255,0.35); }
@keyframes turnPulse { 0%,100%{transform:scale(1)} 50%{transform:scale(1.03)} }

/* ── Game area ── */
#game-area { position: relative; user-select: none; }

/* ── Preview row ──
   Height = --cs + 14px so the bouncing ghost piece has room to travel
   without clipping into the board. JS uses PREV_H = 80 (must match). */
#prev-row {
  display: flex;
  gap: var(--gap);
  padding: 0 var(--bp);
  height: calc(var(--cs) + 14px);   /* = 80px */
  align-items: flex-end;
}

.prev-cell {
  width: var(--cs);
  height: var(--cs);
  border-radius: 50%;
  flex-shrink: 0;
  opacity: 0;
  transition: opacity 0.12s ease;
  pointer-events: none;
  position: relative;
}
.prev-cell.on {
  opacity: 0.82;
  animation: prevBounce 0.75s ease-in-out infinite alternate;
}
/* Small downward arrow below the ghost piece */
.prev-cell.on::after {
  content: '';
  position: absolute;
  bottom: -9px;
  left: 50%;
  transform: translateX(-50%);
  border-left:  7px solid transparent;
  border-right: 7px solid transparent;
  border-top:   8px solid rgba(255,255,255,0.35);
}
@keyframes prevBounce {
  from { transform: translateY(0); }
  to   { transform: translateY(10px); }
}
.prev-cell.c-red {
  background: radial-gradient(circle at 35% 35%, #ff6b6b, #c62828);
  box-shadow: 0 0 24px rgba(255,71,87,0.7),
              inset 0 -4px 8px rgba(255,255,255,0.2),
              inset 0  4px 8px rgba(0,0,0,0.25);
}
.prev-cell.c-yellow {
  background: radial-gradient(circle at 35% 35%, #fff176, #f9a825);
  box-shadow: 0 0 24px rgba(255,217,61,0.7),
              inset 0 -4px 8px rgba(255,255,255,0.28),
              inset 0  4px 8px rgba(0,0,0,0.18);
}

/* ── Board ── */
#board {
  background: linear-gradient(150deg, #1a6ed8 0%, #1155b0 55%, #0c3f8a 100%);
  border-radius: 20px;
  padding: var(--bp);
  display: flex;
  gap: var(--gap);
  position: relative;
  box-shadow:
    0 28px 72px rgba(0,0,0,0.65),
    0 0 0 3px rgba(255,255,255,0.07),
    inset 0 2px 6px rgba(255,255,255,0.14),
    inset 0 -2px 5px rgba(0,0,0,0.3);
}

/* ── Column ── */
.col {
  display: flex;
  flex-direction: column;
  gap: var(--gap);
  padding: 3px;
  border-radius: 11px;
  cursor: pointer;
  transition: background 0.11s ease;
  position: relative;
}
/* Hover highlight + inset glow ring */
.col:not(.full):hover { background: rgba(255,255,255,0.115); }
.col:not(.full):hover::before {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: 11px;
  box-shadow: inset 0 0 0 2px rgba(255,255,255,0.28);
  pointer-events: none;
}
.col.full { cursor: not-allowed; }

/* ── Cells ── */
.cell {
  width: var(--cs);
  height: var(--cs);
  border-radius: 50%;
  flex-shrink: 0;
  will-change: transform;  /* GPU compositing hint */
}
.cell.empty {
  background: radial-gradient(circle at 35% 35%, #1e2b8a, #0d1547);
  box-shadow: inset 0 5px 14px rgba(0,0,0,0.72),
              inset 0 -2px 4px rgba(255,255,255,0.04);
}
.cell.red {
  background: radial-gradient(circle at 35% 35%, #ff6b6b, #c62828);
  box-shadow: inset 0 4px 8px rgba(0,0,0,0.32),
              inset 0 -3px 8px rgba(255,255,255,0.24),
              0 0 20px rgba(255,71,87,0.42);
}
.cell.yellow {
  background: radial-gradient(circle at 35% 35%, #fff176, #f9a825);
  box-shadow: inset 0 4px 8px rgba(0,0,0,0.22),
              inset 0 -3px 8px rgba(255,255,255,0.34),
              0 0 20px rgba(255,217,61,0.42);
}

/* ── Drop animation ──
   Per-keyframe timing-functions simulate gravity + elastic bounce:
   0→78%  : heavy ease-in  (accelerating fall like gravity)
   78→88% : quick overshoot (piece compresses at landing)
   88→95% : spring rebound
   95→100%: gentle settle
   Duration is computed in JS and stored in --dur.
   Starting offset is computed in JS and stored in --from.            */
.cell.dropping {
  animation: drop var(--dur, 0.45s) linear both;
}
@keyframes drop {
  0%   {
    transform: translateY(var(--from, -350px));
    animation-timing-function: cubic-bezier(0.45, 0, 0.95, 1);
  }
  78%  {
    transform: translateY(0);
    animation-timing-function: cubic-bezier(0.12, 0, 0.08, 1);
  }
  88%  {
    transform: translateY(8px);
    animation-timing-function: cubic-bezier(0.38, 0, 0.18, 1);
  }
  95%  {
    transform: translateY(-4px);
    animation-timing-function: cubic-bezier(0.08, 0, 0.25, 1);
  }
  100% { transform: translateY(0); }
}

/* ── Winning cells ── */
.cell.win-cell {
  animation: winGlow 0.65s ease-in-out infinite alternate;
}
@keyframes winGlow {
  from { filter: brightness(1); transform: scale(1); }
  to   { filter: brightness(1.8) drop-shadow(0 0 14px white); transform: scale(1.1); }
}

/* ── Restart button ── */
#btn-new {
  margin-top: 20px;
  padding: 12px 52px;
  background: linear-gradient(135deg, #667eea, #764ba2);
  border: none;
  border-radius: 14px;
  color: white;
  font-family: 'Nunito', sans-serif;
  font-size: 1rem;
  font-weight: 800;
  cursor: pointer;
  letter-spacing: 0.5px;
  box-shadow: 0 6px 28px rgba(102,126,234,0.45);
  transition: transform 0.18s ease, box-shadow 0.18s ease;
}
#btn-new:hover  { transform: translateY(-3px); box-shadow: 0 10px 38px rgba(102,126,234,0.65); }
#btn-new:active { transform: translateY(0); }

.hint {
  margin-top: 10px;
  color: rgba(255,255,255,0.18);
  font-size: 0.7rem;
  letter-spacing: 1.5px;
  text-transform: uppercase;
}
</style>
</head>
<body>

<div id="title">Connect Four</div>
<div class="subtitle">Classic Strategy Game</div>

<div id="scores">
  <div class="sc active" id="sc-red">
    <div class="lbl">🔴 Red</div>
    <div class="num" id="n-red">0</div>
  </div>
  <div class="sc" id="sc-yellow">
    <div class="lbl">🟡 Yellow</div>
    <div class="num" id="n-yellow">0</div>
  </div>
</div>

<div id="turn" class="t-red">🔴 Red's Turn</div>

<div id="game-area">
  <div id="prev-row"></div>
  <div id="board"></div>
</div>

<button id="btn-new">↺  New Game</button>
<div class="hint">Hover a column to preview · Click to drop</div>

<script>
// ── Constants (pixel values must match CSS :root variables and PREV_H above) ──
const ROWS=6, COLS=7, EMPTY=0, RED=1, YEL=2;
const CS=66, GAP=7, BP=13;
const PREV_H = 80; // height of #prev-row (CS + 14 from CSS)

// ── State ──────────────────────────────────────────────────────────────────────
let board, cur, over, busy;
const scores = {[RED]: 0, [YEL]: 0};

// ── Initialise / reset game ────────────────────────────────────────────────────
function init() {
  board = Array.from({length: ROWS}, () => Array(COLS).fill(EMPTY));
  cur = RED;
  over = false;
  busy = false;
  buildGrid();
  setTurn();
  setScoreHighlight();
}

// ── Build DOM ──────────────────────────────────────────────────────────────────
function buildGrid() {
  // Preview ghost pieces (one per column)
  const pr = document.getElementById('prev-row');
  pr.innerHTML = '';
  for (let c = 0; c < COLS; c++) {
    const d = document.createElement('div');
    d.className = 'prev-cell';
    d.id = 'pv' + c;
    pr.appendChild(d);
  }

  // Board columns
  const bd = document.getElementById('board');
  bd.innerHTML = '';
  for (let c = 0; c < COLS; c++) {
    const col = document.createElement('div');
    col.className = 'col';
    col.dataset.c = c;
    for (let r = 0; r < ROWS; r++) {
      const cell = document.createElement('div');
      cell.className = 'cell empty';
      cell.id = `cl${r}_${c}`;
      col.appendChild(cell);
    }
    // Hover → show ghost piece above column
    col.addEventListener('mouseenter', () => onEnter(c));
    col.addEventListener('mouseleave', () => onLeave(c));
    // Click → drop piece
    col.addEventListener('click',      () => onDrop(c));
    bd.appendChild(col);
  }
}

// ── Hover events ──────────────────────────────────────────────────────────────
function onEnter(c) {
  if (over || busy || board[0][c] !== EMPTY) return;
  for (let i = 0; i < COLS; i++) {
    const pv = document.getElementById('pv' + i);
    pv.className = 'prev-cell' + (i === c ? ' on ' + (cur === RED ? 'c-red' : 'c-yellow') : '');
  }
}

function onLeave(c) {
  document.getElementById('pv' + c).className = 'prev-cell';
}

function clearPrev() {
  for (let i = 0; i < COLS; i++)
    document.getElementById('pv' + i).className = 'prev-cell';
}

// ── Drop a piece ───────────────────────────────────────────────────────────────
function lowestEmpty(c) {
  for (let r = ROWS - 1; r >= 0; r--)
    if (board[r][c] === EMPTY) return r;
  return -1;
}

function onDrop(c) {
  if (over || busy) return;
  const row = lowestEmpty(c);
  if (row < 0) return;

  board[row][c] = cur;
  busy = true;
  clearPrev();

  // ── Animate the falling piece ──
  // translateY(--from) moves the cell from the ghost-piece position
  // (centre of preview row) down to its natural DOM position (translateY 0).
  //
  // fromY = distance upward from cell centre to preview-row centre:
  //   cell centre relative to board top = BP + row*(CS+GAP) + CS/2
  //   preview-row centre above board top = PREV_H/2
  //   total = BP + row*(CS+GAP) + CS/2 + PREV_H/2
  const fromY = -(BP + row * (CS + GAP) + CS / 2 + PREV_H / 2);

  // Duration scales with √distance so deeper drops feel heavier (gravity)
  const maxFromY = BP + 5 * (CS + GAP) + CS / 2 + PREV_H / 2; // row 5
  const dur = (0.22 + 0.30 * Math.sqrt(Math.abs(fromY) / maxFromY)).toFixed(3);

  const cell = document.getElementById(`cl${row}_${c}`);
  cell.className = `cell ${cur === RED ? 'red' : 'yellow'}`;
  cell.style.setProperty('--from', `${fromY}px`);
  cell.style.setProperty('--dur',  `${dur}s`);
  cell.classList.add('dropping');

  cell.addEventListener('animationend', () => {
    cell.classList.remove('dropping');
    busy = false;
    afterDrop(row, c);
  }, {once: true});
}

// ── Post-drop: check win/draw, switch player ───────────────────────────────────
function afterDrop(row, c) {
  const wc = findWinCells(row, c, cur);
  if (wc) {
    over = true;
    scores[cur]++;
    updateScores();
    wc.forEach(([r, cc]) =>
      document.getElementById(`cl${r}_${cc}`).classList.add('win-cell')
    );
    setTurn('win');
    return;
  }
  if (board[0].every(v => v !== EMPTY)) {
    over = true;
    setTurn('draw');
    return;
  }
  cur = cur === RED ? YEL : RED;
  setTurn();
  setScoreHighlight();
  // Disable full columns
  if (board[0][c] !== EMPTY)
    document.querySelector(`.col[data-c="${c}"]`).classList.add('full');
}

// ── Win detection ──────────────────────────────────────────────────────────────
function findWinCells(row, col, p) {
  const dirs = [[0,1],[1,0],[1,1],[1,-1]];
  for (const [dr, dc] of dirs) {
    const cells = [[row, col]];
    for (const s of [1, -1]) {
      let [r, c] = [row + s*dr, col + s*dc];
      while (r >= 0 && r < ROWS && c >= 0 && c < COLS && board[r][c] === p) {
        cells.push([r, c]);
        r += s*dr; c += s*dc;
      }
    }
    if (cells.length >= 4) return cells;
  }
  return null;
}

// ── UI updates ─────────────────────────────────────────────────────────────────
function setTurn(state) {
  const el = document.getElementById('turn');
  if (state === 'win') {
    el.className = 't-win';
    el.textContent = `🏆 ${cur === RED ? '🔴 Red' : '🟡 Yellow'} Wins!`;
  } else if (state === 'draw') {
    el.className = 't-draw';
    el.textContent = "🤝 It's a Draw!";
  } else {
    el.className = cur === RED ? 't-red' : 't-yellow';
    el.textContent = cur === RED ? "🔴 Red's Turn" : "🟡 Yellow's Turn";
  }
}

function updateScores() {
  document.getElementById('n-red').textContent    = scores[RED];
  document.getElementById('n-yellow').textContent = scores[YEL];
}

function setScoreHighlight() {
  document.getElementById('sc-red').classList.toggle('active',    cur === RED);
  document.getElementById('sc-yellow').classList.toggle('active', cur === YEL);
}

// ── New Game ───────────────────────────────────────────────────────────────────
document.getElementById('btn-new').addEventListener('click', init);

// ── Start ──────────────────────────────────────────────────────────────────────
init();
</script>
</body>
</html>
"""

components.html(GAME_HTML, height=900, scrolling=False)
