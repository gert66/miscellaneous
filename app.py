import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Connect Game", page_icon="🔴", layout="centered")

# ── Constants ─────────────────────────────────────────────────────────────────
EMPTY = 0
P1 = 1   # red
P2 = 2   # yellow

TOKEN = {EMPTY: "⚫", P1: "🔴", P2: "🟡"}
NAME  = {P1: "Player 1 🔴", P2: "Player 2 🟡"}

# ── Helper functions ──────────────────────────────────────────────────────────

def create_board(rows, cols):
    return [[EMPTY] * cols for _ in range(rows)]


def is_valid_column(board, col):
    """True if the top cell of a column is still empty."""
    return board[0][col] == EMPTY


def drop_piece(board, col, player):
    """Drop a piece into the lowest empty row of the given column."""
    rows = len(board)
    for row in range(rows - 1, -1, -1):
        if board[row][col] == EMPTY:
            board[row][col] = player
            return row
    return -1  # should never happen if is_valid_column was checked first


def check_win(board, player, win_len):
    """Return True if *player* has *win_len* in a row anywhere on the board."""
    rows = len(board)
    cols = len(board[0])

    # Horizontal
    for r in range(rows):
        for c in range(cols - win_len + 1):
            if all(board[r][c + k] == player for k in range(win_len)):
                return True

    # Vertical
    for r in range(rows - win_len + 1):
        for c in range(cols):
            if all(board[r + k][c] == player for k in range(win_len)):
                return True

    # Diagonal ↘
    for r in range(rows - win_len + 1):
        for c in range(cols - win_len + 1):
            if all(board[r + k][c + k] == player for k in range(win_len)):
                return True

    # Diagonal ↙
    for r in range(rows - win_len + 1):
        for c in range(win_len - 1, cols):
            if all(board[r + k][c - k] == player for k in range(win_len)):
                return True

    return False


def check_draw(board):
    """True when no empty cells remain in the top row (board full)."""
    return all(board[0][c] != EMPTY for c in range(len(board[0])))


def reset_game():
    """Initialise (or reinitialise) all game state in session_state."""
    s = st.session_state
    s.board = create_board(s.rows, s.cols)
    s.current_player = P1
    s.winner = None   # None = game ongoing, 0 = draw, P1/P2 = winner
    s.game_over = False


def render_board(board):
    """Display the board as an HTML table with coloured circles."""
    rows = len(board)
    cols = len(board[0])

    # Build one big HTML string — much faster than nested st calls
    cell_size = 70

    html = f"""
    <style>
      .connect-table {{
        border-collapse: separate;
        border-spacing: 6px;
        background: #1a3a6b;
        padding: 10px;
        border-radius: 12px;
        margin: 0 auto;
      }}
      .connect-table td {{
        width: {cell_size}px;
        height: {cell_size}px;
        text-align: center;
        vertical-align: middle;
        font-size: {cell_size - 10}px;
        line-height: 1;
      }}
    </style>
    <table class="connect-table">
    """
    for r in range(rows):
        html += "<tr>"
        for c in range(cols):
            html += f"<td>{TOKEN[board[r][c]]}</td>"
        html += "</tr>"
    html += "</table>"

    st.markdown(html, unsafe_allow_html=True)


# ── Sidebar settings ──────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    rows_choice = st.number_input("Rows", min_value=4, max_value=10, value=6, step=1)
    cols_choice = st.number_input("Columns", min_value=4, max_value=12, value=7, step=1)
    win_choice  = st.selectbox("Win length", [4, 5], index=0)

    # Detect setting changes and reset automatically
    settings_changed = (
        st.session_state.get("rows") != rows_choice
        or st.session_state.get("cols") != cols_choice
        or st.session_state.get("win_len") != win_choice
    )
    if settings_changed:
        st.session_state.rows    = rows_choice
        st.session_state.cols    = cols_choice
        st.session_state.win_len = win_choice
        reset_game()

    st.divider()
    if st.button("🔄 New Game", use_container_width=True):
        reset_game()

# ── First-run initialisation ──────────────────────────────────────────────────

if "board" not in st.session_state:
    st.session_state.rows    = rows_choice
    st.session_state.cols    = cols_choice
    st.session_state.win_len = win_choice
    reset_game()

# Convenience aliases
s = st.session_state

# ── Title & status ────────────────────────────────────────────────────────────

st.title("🎯 Connect Game")

if s.game_over:
    if s.winner == 0:
        st.success("🤝 It's a draw! Click **New Game** to play again.")
    else:
        st.success(f"🎉 {NAME[s.winner]} wins! Click **New Game** to play again.")
else:
    st.info(f"**{NAME[s.current_player]}'s turn** — click a column button to drop a piece.")

# ── Column-drop buttons ───────────────────────────────────────────────────────

cols_count = s.cols
button_cols = st.columns(cols_count)

for c in range(cols_count):
    with button_cols[c]:
        # Disable button if column full or game over
        disabled = s.game_over or not is_valid_column(s.board, c)
        if st.button(f"↓", key=f"col_{c}", disabled=disabled, use_container_width=True):
            drop_piece(s.board, c, s.current_player)

            if check_win(s.board, s.current_player, s.win_len):
                s.winner    = s.current_player
                s.game_over = True
            elif check_draw(s.board):
                s.winner    = 0
                s.game_over = True
            else:
                # Switch player
                s.current_player = P2 if s.current_player == P1 else P1

            st.rerun()

# ── Board display ─────────────────────────────────────────────────────────────

render_board(s.board)

# ── Legend ────────────────────────────────────────────────────────────────────

st.markdown(
    "<br><center>🔴 Player 1 &nbsp;&nbsp;&nbsp; 🟡 Player 2 &nbsp;&nbsp;&nbsp; ⚫ Empty</center>",
    unsafe_allow_html=True,
)
