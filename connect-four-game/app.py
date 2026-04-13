import streamlit as st
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
ROWS, COLS = 6, 7
EMPTY, RED, YELLOW = 0, 1, 2
PLAYER_COLORS = {RED: "🔴", YELLOW: "🟡"}
PLAYER_NAMES = {RED: "Red", YELLOW: "Yellow"}
WINNING_LENGTH = 4

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Connect Four",
    page_icon="🔴",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@700;800;900&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

[data-testid="stAppViewContainer"] > .main { padding-top: 1rem; }

h1 {
    font-family: 'Nunito', sans-serif !important;
    font-weight: 900 !important;
    font-size: 3rem !important;
    text-align: center;
    background: linear-gradient(90deg, #ff6b6b, #ffd93d, #ff6b6b);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0 !important;
    letter-spacing: -1px;
    animation: shimmer 3s linear infinite;
}
@keyframes shimmer { 0%{background-position:0%} 100%{background-position:200%} }

/* ── Turn indicator ── */
.turn-box {
    background: rgba(255,255,255,0.08);
    border: 2px solid rgba(255,255,255,0.15);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 12px 24px;
    text-align: center;
    font-family: 'Nunito', sans-serif;
    font-size: 1.25rem;
    font-weight: 800;
    color: white;
    margin: 8px auto 16px auto;
    max-width: 340px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.turn-red  { border-color: #ff4757 !important; box-shadow: 0 0 20px rgba(255,71,87,0.4) !important; }
.turn-yellow { border-color: #ffd93d !important; box-shadow: 0 0 20px rgba(255,217,61,0.4) !important; }

/* ── Win banner ── */
.win-banner {
    background: linear-gradient(135deg, rgba(255,215,0,0.2), rgba(255,107,107,0.2));
    border: 2px solid gold;
    border-radius: 20px;
    padding: 16px 32px;
    text-align: center;
    font-family: 'Nunito', sans-serif;
    font-size: 1.6rem;
    font-weight: 900;
    color: gold;
    margin: 8px auto 16px auto;
    max-width: 400px;
    box-shadow: 0 0 40px rgba(255,215,0,0.3);
    animation: pulse 1.5s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{transform:scale(1)} 50%{transform:scale(1.03)} }

.draw-banner {
    background: rgba(255,255,255,0.08);
    border: 2px solid rgba(255,255,255,0.4);
    border-radius: 20px;
    padding: 16px 32px;
    text-align: center;
    font-family: 'Nunito', sans-serif;
    font-size: 1.6rem;
    font-weight: 900;
    color: white;
    margin: 8px auto 16px auto;
    max-width: 400px;
}

/* ── Board frame ── */
.board-frame {
    background: linear-gradient(145deg, #1565C0, #0D47A1);
    border-radius: 20px;
    padding: 16px;
    box-shadow:
        0 20px 60px rgba(0,0,0,0.5),
        0 0 0 4px rgba(255,255,255,0.08),
        inset 0 2px 4px rgba(255,255,255,0.15);
    margin: 0 auto;
    max-width: 560px;
}

/* ── Column buttons (arrow row) ── */
.stButton > button {
    background: rgba(255,255,255,0.12) !important;
    border: 2px solid rgba(255,255,255,0.2) !important;
    border-radius: 12px !important;
    color: white !important;
    font-size: 1.3rem !important;
    font-weight: 800 !important;
    width: 100% !important;
    padding: 4px 0 !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
    box-shadow: none !important;
}
.stButton > button:hover {
    background: rgba(255,255,255,0.25) !important;
    border-color: rgba(255,255,255,0.5) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Cell SVG circles ── */
.board-row {
    display: flex;
    gap: 6px;
    margin-bottom: 6px;
    justify-content: center;
}
.cell {
    width: 66px;
    height: 66px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: inset 0 4px 8px rgba(0,0,0,0.4);
}
.cell-empty {
    background: radial-gradient(circle at 35% 35%, #1a237e, #0d1547);
    box-shadow: inset 0 4px 12px rgba(0,0,0,0.6), inset 0 -2px 4px rgba(255,255,255,0.05);
}
.cell-red {
    background: radial-gradient(circle at 35% 35%, #ff6b6b, #c62828);
    box-shadow:
        inset 0 4px 8px rgba(0,0,0,0.3),
        inset 0 -2px 4px rgba(255,255,255,0.3),
        0 0 16px rgba(255,71,87,0.5);
}
.cell-yellow {
    background: radial-gradient(circle at 35% 35%, #fff176, #f9a825);
    box-shadow:
        inset 0 4px 8px rgba(0,0,0,0.2),
        inset 0 -2px 4px rgba(255,255,255,0.4),
        0 0 16px rgba(255,217,61,0.5);
}
.cell-win {
    animation: winPulse 0.7s ease-in-out infinite alternate;
}
@keyframes winPulse {
    from { filter: brightness(1); }
    to   { filter: brightness(1.5) drop-shadow(0 0 10px white); }
}

/* ── Score cards ── */
.score-area {
    display: flex;
    gap: 16px;
    justify-content: center;
    margin: 12px auto 4px auto;
    max-width: 400px;
}
.score-card {
    flex: 1;
    background: rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 10px 16px;
    text-align: center;
    border: 2px solid rgba(255,255,255,0.1);
    font-family: 'Nunito', sans-serif;
}
.score-card .label { font-size: 0.8rem; color: rgba(255,255,255,0.6); font-weight: 700; letter-spacing: 1px; text-transform: uppercase; }
.score-card .number { font-size: 2rem; font-weight: 900; color: white; line-height: 1.1; }
.score-card-red   { border-color: rgba(255,71,87,0.5) !important; }
.score-card-yellow{ border-color: rgba(255,217,61,0.5) !important; }

/* ── Restart button ── */
.restart-btn .stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    border: none !important;
    border-radius: 14px !important;
    font-size: 1rem !important;
    font-weight: 800 !important;
    padding: 10px 0 !important;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 20px rgba(102,126,234,0.4) !important;
    transition: all 0.2s ease !important;
}
.restart-btn .stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 30px rgba(102,126,234,0.6) !important;
}

/* ── Subtitle ── */
.subtitle {
    text-align: center;
    color: rgba(255,255,255,0.45);
    font-size: 0.85rem;
    font-family: 'Nunito', sans-serif;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: -4px;
    margin-bottom: 4px;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state init ─────────────────────────────────────────────────────────
def init_state():
    st.session_state.board = np.zeros((ROWS, COLS), dtype=int)
    st.session_state.current_player = RED
    st.session_state.winner = None
    st.session_state.winning_cells = set()
    st.session_state.game_over = False
    st.session_state.draw = False

if "board" not in st.session_state:
    init_state()
    st.session_state.score = {RED: 0, YELLOW: 0}

# ── Game logic ─────────────────────────────────────────────────────────────────
def drop_piece(board, col, player):
    for row in range(ROWS - 1, -1, -1):
        if board[row][col] == EMPTY:
            board[row][col] = player
            return row
    return -1

def get_winning_cells(board, row, col, player):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dr, dc in directions:
        cells = [(row, col)]
        for sign in (1, -1):
            r, c = row + sign * dr, col + sign * dc
            while 0 <= r < ROWS and 0 <= c < COLS and board[r][c] == player:
                cells.append((r, c))
                r += sign * dr
                c += sign * dc
        if len(cells) >= WINNING_LENGTH:
            return set(cells)
    return set()

def check_winner(board, row, col, player):
    cells = get_winning_cells(board, row, col, player)
    return cells if cells else None

def is_draw(board):
    return all(board[0][c] != EMPTY for c in range(COLS))

def handle_column_click(col):
    if st.session_state.game_over:
        return
    board = st.session_state.board
    if board[0][col] != EMPTY:
        return
    player = st.session_state.current_player
    row = drop_piece(board, col, player)
    if row == -1:
        return
    winning_cells = check_winner(board, row, col, player)
    if winning_cells:
        st.session_state.winner = player
        st.session_state.winning_cells = winning_cells
        st.session_state.game_over = True
        st.session_state.score[player] += 1
    elif is_draw(board):
        st.session_state.game_over = True
        st.session_state.draw = True
    else:
        st.session_state.current_player = YELLOW if player == RED else RED

# ── Render helpers ─────────────────────────────────────────────────────────────
def cell_class(value, r, c):
    if value == RED:
        base = "cell cell-red"
    elif value == YELLOW:
        base = "cell cell-yellow"
    else:
        base = "cell cell-empty"
    if (r, c) in st.session_state.winning_cells:
        base += " cell-win"
    return base

def render_board():
    board = st.session_state.board
    html_rows = []
    for r in range(ROWS):
        cells_html = []
        for c in range(COLS):
            cls = cell_class(board[r][c], r, c)
            cells_html.append(f'<div class="{cls}"></div>')
        html_rows.append(f'<div class="board-row">{"".join(cells_html)}</div>')
    st.markdown(f'<div class="board-frame">{"".join(html_rows)}</div>', unsafe_allow_html=True)

# ── UI Layout ──────────────────────────────────────────────────────────────────
st.markdown("<h1>Connect Four</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Classic Strategy Game</p>', unsafe_allow_html=True)

# Score display
s = st.session_state.score
st.markdown(
    f"""
<div class="score-area">
  <div class="score-card score-card-red">
    <div class="label">🔴 Red</div>
    <div class="number">{s[RED]}</div>
  </div>
  <div class="score-card score-card-yellow">
    <div class="label">🟡 Yellow</div>
    <div class="number">{s[YELLOW]}</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Turn / result banner
if st.session_state.game_over:
    if st.session_state.draw:
        st.markdown('<div class="draw-banner">🤝 It\'s a Draw!</div>', unsafe_allow_html=True)
    else:
        w = st.session_state.winner
        emoji = PLAYER_COLORS[w]
        name = PLAYER_NAMES[w]
        st.markdown(f'<div class="win-banner">🏆 {emoji} {name} Wins!</div>', unsafe_allow_html=True)
else:
    p = st.session_state.current_player
    css_class = "turn-red" if p == RED else "turn-yellow"
    st.markdown(
        f'<div class="turn-box {css_class}">{PLAYER_COLORS[p]} {PLAYER_NAMES[p]}\'s Turn</div>',
        unsafe_allow_html=True,
    )

# Column drop buttons
if not st.session_state.game_over:
    cols_ui = st.columns(COLS)
    for c, col_ui in enumerate(cols_ui):
        with col_ui:
            board = st.session_state.board
            col_full = board[0][c] != EMPTY
            label = "▼" if not col_full else "✕"
            if st.button(label, key=f"col_{c}", disabled=col_full):
                handle_column_click(c)
                st.rerun()

# Board
render_board()

# Restart button
st.markdown("")
restart_col = st.columns([1, 2, 1])[1]
with restart_col:
    st.markdown('<div class="restart-btn">', unsafe_allow_html=True)
    if st.button("↺  New Game", key="restart", use_container_width=True):
        score_backup = st.session_state.score
        init_state()
        st.session_state.score = score_backup
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# Footer hint
st.markdown(
    '<p style="text-align:center;color:rgba(255,255,255,0.2);font-size:0.75rem;'
    'font-family:Nunito,sans-serif;margin-top:12px;">Click ▼ above a column to drop your piece</p>',
    unsafe_allow_html=True,
)
