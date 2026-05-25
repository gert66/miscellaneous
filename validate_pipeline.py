"""
Standalone pipeline validation runner.

Run with:
    python validate_pipeline.py

Stubs heavy dependencies (streamlit, anthropic, etc.) then imports
enrich_clients_claude and calls _validate_type1_type2_pipeline().
"""

import sys
import types
from pathlib import Path

# ── Stub heavy imports ────────────────────────────────────────────────────────

def _make_pkg(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__path__ = []
    return mod

st_stub = _make_pkg("streamlit")
for _attr in [
    "session_state", "secrets", "set_page_config", "title", "caption",
    "markdown", "image", "info", "success", "error", "warning", "divider",
    "sidebar", "button", "file_uploader", "selectbox", "checkbox",
    "number_input", "slider", "text_input", "columns", "progress",
    "subheader", "expander", "download_button", "radio", "dataframe",
    "code", "metric", "status", "rerun", "stop", "write", "toast",
]:
    setattr(st_stub, _attr, lambda *a, **kw: None)
st_stub.session_state = {}

class _FakeSecrets(dict):
    def get(self, k, d=None): return d

st_stub.secrets = _FakeSecrets()
_st_components = _make_pkg("streamlit.components")
_st_components_v1 = _make_pkg("streamlit.components.v1")
_st_components_v1.html = lambda *a, **kw: None
_st_components_v1.iframe = lambda *a, **kw: None
st_stub.components = _st_components
_st_components.v1 = _st_components_v1
sys.modules["streamlit"] = st_stub
sys.modules["streamlit.components"] = _st_components
sys.modules["streamlit.components.v1"] = _st_components_v1

for _mod in ["anthropic", "jina", "playwright", "playwright.sync_api",
             "requests", "httpx"]:
    sys.modules.setdefault(_mod, _make_pkg(_mod))

_bs4 = _make_pkg("bs4")
class _FakeSoup:
    def __init__(self, *a, **kw): pass
    def get_text(self, *a, **kw): return ""
    def find(self, *a, **kw): return None
    def find_all(self, *a, **kw): return []
_bs4.BeautifulSoup = _FakeSoup
sys.modules.setdefault("bs4", _bs4)

sys.path.insert(0, str(Path(__file__).parent))

# ── Run validation ────────────────────────────────────────────────────────────
import enrich_clients_claude as e
e._validate_type1_type2_pipeline()
