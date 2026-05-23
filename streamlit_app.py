"""
Streamlit Cloud entry point — Commercial Fit Scoring App.

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

# Tell enrich_clients_claude.py that st.set_page_config / st.title / st.caption
# are already handled here, so it must skip its own calls.
os.environ.setdefault("_STREAMLIT_ENTRYPOINT", "1")

import streamlit as st

# ── Page config — must be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="Commercial Fit Scoring — Mingle",
    page_icon="🏢",
    layout="wide",
)

st.title("Commercial Fit Scoring App")
st.caption("App loaded successfully.")

# ── Delegate to the main application ─────────────────────────────────────────
# runpy.run_path executes enrich_clients_claude.py in-process, setting __file__
# correctly so all relative path logic in that file continues to work.
import pathlib
import runpy

_MAIN = pathlib.Path(__file__).parent / "enrich_clients_claude.py"
runpy.run_path(str(_MAIN), run_name="__main__")
