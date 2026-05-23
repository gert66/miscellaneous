"""
Streamlit Cloud entry point.

Streamlit Cloud's "main file path" must be set to streamlit_app.py.
This file delegates execution to enrich_clients_claude.py, which contains
the full enrichment + commercial-fit-scoring application.

Do NOT put application logic here. Keep this file as a thin redirect only.
"""

import pathlib
import runpy

_APP = pathlib.Path(__file__).parent / "enrich_clients_claude.py"
runpy.run_path(str(_APP), run_name="__main__")
