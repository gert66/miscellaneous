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
    page_title="mYngle · lead prioritizer",
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
    div[data-testid="stMarkdownContainer"]:has(.brand-header) {{
        overflow: visible;
        margin-bottom: 1.5rem;
    }}
    .brand-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-top: 12px;
        overflow: visible;
    }}
    .brand-title-block {{
        flex: 0 0 45%;
    }}
    .brand-title {{
        font-size: 42px;
        font-weight: 700;
        color: #0B1F3A;
        line-height: 1.1;
        white-space: nowrap;
        margin: 0;
        padding: 0;
    }}
    .brand-logo-block {{
        flex: 0 0 55%;
        display: flex;
        justify-content: flex-end;
        align-items: center;
        overflow: visible;
    }}
    .brand-logo {{
        width: 300px;
        height: auto;
        display: block;
        object-fit: contain;
        overflow: visible;
    }}
    </style>
    <div class="brand-header">
      <div class="brand-title-block"><span class="brand-title">lead prioritizer</span></div>
      <div class="brand-logo-block">{_img_tag}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Delegate to the main application ─────────────────────────────────────────
import runpy

_MAIN = pathlib.Path(__file__).parent / "enrich_clients_claude.py"
runpy.run_path(str(_MAIN), run_name="__main__")
