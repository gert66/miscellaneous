"""
Streamlit Cloud entry point — Company Enrichment App.

DEPLOYMENT INSTRUCTIONS
-----------------------
In Streamlit Cloud → your app → Settings → General:
  Main file path → streamlit_app.py

commercial_fit_scoring.py is a helper/scoring module only.
enrich_clients_claude.py contains the full application logic.
This file is the thin Streamlit launcher that Cloud executes.

If you cannot find "Main file path" in the UI you must delete the existing
deployment and redeploy from this branch, choosing streamlit_app.py as the
main file during the "Deploy an app" wizard.
"""

import os
import pathlib

# Tell enrich_clients_claude.py that set_page_config / title / caption
# are already handled here, so it must skip its own header calls.
os.environ.setdefault("_STREAMLIT_ENTRYPOINT", "1")

import streamlit as st

# ── Page config — must be the very first Streamlit call ───────────────────────
st.set_page_config(
    page_title="mYngle · Company Enrichment",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Compact centred layout ────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .block-container {
        max-width: 880px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

import base64

# ── Logo + header (inline flex, tight gap, lowercase title) ──────────────────
_logo_path = pathlib.Path(__file__).parent / "Mynglelogofinal.jpg"
if _logo_path.exists():
    _logo_b64 = base64.b64encode(_logo_path.read_bytes()).decode()
    _logo_src  = f"data:image/jpeg;base64,{_logo_b64}"
else:
    _logo_src  = ""

st.markdown(
    f"""
    <style>
    .brand-header {{
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 0.25rem;
    }}
    .brand-logo {{
        height: 52px;
        width: auto;
        display: block;
        flex-shrink: 0;
    }}
    .brand-title {{
        font-size: 2rem;
        font-weight: 700;
        line-height: 1;
        margin: 0;
        padding: 0;
        color: inherit;
    }}
    </style>
    <div class="brand-header">
        {'<img src="' + _logo_src + '" class="brand-logo" alt="mYngle">' if _logo_src else ''}
        <div class="brand-title">company enrichment</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "Upload a company file. "
    "The app will enrich and score the companies, "
    "then generate an Excel report."
)

# ── Delegate to the main application ─────────────────────────────────────────
import runpy

_MAIN = pathlib.Path(__file__).parent / "enrich_clients_claude.py"
runpy.run_path(str(_MAIN), run_name="__main__")
