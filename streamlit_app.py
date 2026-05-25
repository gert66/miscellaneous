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

import base64

# ── Logo + header (single markdown block, inline image, word-space gap) ───────
_logo_path = pathlib.Path(__file__).parent / "Mynglelogofinal.jpg"
if _logo_path.exists():
    _logo_b64 = base64.b64encode(_logo_path.read_bytes()).decode()
    _logo_src  = f"data:image/jpeg;base64,{_logo_b64}"
else:
    _logo_src  = ""

_img_tag = (
    f'<img src="{_logo_src}" class="brand-logo" alt="mYngle" />'
    if _logo_src else ""
)

st.markdown(
    f"""
    <style>
    .block-container {{
        max-width: 880px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }}
    /* Zero out Streamlit's wrapper margin on the brand block */
    div[data-testid="stMarkdownContainer"]:has(.brand-header) {{
        overflow: visible;
        margin-bottom: 0;
    }}
    .brand-header {{
        font-size: 34px;
        font-weight: 700;
        color: #0B1F3A;
        line-height: 1.3;
        margin: 0.4rem 0 0.15rem 0;
        padding-top: 6px;
        overflow: visible;
        white-space: nowrap;
    }}
    /* Logo as an inline element so the gap is exactly one word-space */
    .brand-logo {{
        height: 1.1em;
        width: auto;
        display: inline;
        vertical-align: middle;
        margin-right: 0.12em;
        overflow: visible;
    }}
    .brand-subtitle {{
        font-size: 0.875rem;
        color: #6b7280;
        margin: 0 0 0.5rem 0;
        padding: 0;
    }}
    </style>
    <p class="brand-header">{_img_tag}company enrichment</p>
    <p class="brand-subtitle">Upload a company file. The app will enrich and score the companies, then generate an Excel report.</p>
    """,
    unsafe_allow_html=True,
)

# ── Delegate to the main application ─────────────────────────────────────────
import runpy

_MAIN = pathlib.Path(__file__).parent / "enrich_clients_claude.py"
runpy.run_path(str(_MAIN), run_name="__main__")
