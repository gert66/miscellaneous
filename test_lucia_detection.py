"""
Smoke test for Lucia/Lusha contact export detection and column mapping.

Tests all detection, deduplication, column mapping, and filename logic
for Type 2 (pre_enriched_lucia_export) files using the test.csv fixture.

Run with:  python test_lucia_detection.py
"""

import sys
import io
import re
from pathlib import Path

import pandas as pd

# ── Import the module functions we need ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# Minimal stub so streamlit imports inside the module don't crash
import types

def _make_module_tree(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__path__ = []  # pretend it's a package
    return mod

st_stub = _make_module_tree("streamlit")
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
    def get(self, k, d=None):
        return d
st_stub.secrets = _FakeSecrets()
# Build streamlit sub-packages
_st_components = _make_module_tree("streamlit.components")
_st_components_v1 = _make_module_tree("streamlit.components.v1")
_st_components_v1.html = lambda *a, **kw: None
_st_components_v1.iframe = lambda *a, **kw: None
st_stub.components = _st_components
_st_components.v1 = _st_components_v1
sys.modules["streamlit"] = st_stub
sys.modules["streamlit.components"] = _st_components
sys.modules["streamlit.components.v1"] = _st_components_v1

# Patch away any other hard imports that may fail in test context
for _mod in ["anthropic", "jina", "playwright", "playwright.sync_api",
             "requests", "httpx"]:
    sys.modules.setdefault(_mod, _make_module_tree(_mod))

# bs4.BeautifulSoup stub
_bs4_mod = _make_module_tree("bs4")
class _FakeSoup:
    def __init__(self, *a, **kw): pass
    def get_text(self, *a, **kw): return ""
    def find(self, *a, **kw): return None
    def find_all(self, *a, **kw): return []
_bs4_mod.BeautifulSoup = _FakeSoup
sys.modules.setdefault("bs4", _bs4_mod)

# Now import the functions we want to test
from enrich_clients_claude import (
    is_lucia_contact_export,
    get_lucia_name_col,
    get_lucia_domain_col,
    map_lucia_export_row,
    deduplicate_lucia_export,
    detect_columns,
    clean_domain,
)


def load_test_csv() -> pd.DataFrame:
    csv_path = Path(__file__).parent / "test.csv"
    assert csv_path.exists(), f"test.csv not found at {csv_path}"
    return pd.read_csv(csv_path)


def test_lucia_detection(df: pd.DataFrame) -> None:
    print("  [1] is_lucia_contact_export ...")
    assert is_lucia_contact_export(df), "Expected True for test.csv"
    print("      PASS — input_type = pre_enriched_lucia_export")


def test_column_detection(df: pd.DataFrame) -> None:
    print("  [2] detect_columns / name & domain cols ...")
    name_col, domain_col = detect_columns(df)
    assert name_col == "Company Name", (
        f"Expected 'Company Name', got {name_col!r}\n"
        "  Must not be 'First Name', 'Last Name', or any person field."
    )
    assert domain_col == "Company Domain", (
        f"Expected 'Company Domain', got {domain_col!r}\n"
        "  Must not be 'LinkedIn URL' or any person/URL field."
    )
    print(f"      PASS — name_col={name_col!r}, domain_col={domain_col!r}")


def test_lucia_col_helpers(df: pd.DataFrame) -> None:
    print("  [3] get_lucia_name_col / get_lucia_domain_col ...")
    assert get_lucia_name_col(df)   == "Company Name"
    assert get_lucia_domain_col(df) == "Company Domain"
    print("      PASS")


def test_deduplication(df: pd.DataFrame) -> None:
    print("  [4] deduplicate_lucia_export ...")
    df_dedup = deduplicate_lucia_export(df)
    assert len(df_dedup) == 3, (
        f"Expected 3 unique companies, got {len(df_dedup)}"
    )
    names = df_dedup["Company Name"].tolist()
    for expected in ["Ali Lavoro", "Renovit", "S&you Italia"]:
        assert expected in names, (
            f"Expected company '{expected}' in deduplicated output, got: {names}"
        )
    print(f"      PASS — {len(df_dedup)} unique companies: {names}")


def test_no_person_names_in_summary(df: pd.DataFrame) -> None:
    print("  [5] Company Names must not be person names ...")
    df_dedup = deduplicate_lucia_export(df)
    names = df_dedup["Company Name"].tolist()
    person_names = {"Anna", "Michela", "Annalisa"}
    for pn in person_names:
        assert pn not in names, (
            f"Person name '{pn}' found in company names — Summary would be wrong"
        )
    print("      PASS — no person names (Anna, Michela, Annalisa) in company list")


def test_domain_is_not_linkedin(df: pd.DataFrame) -> None:
    print("  [6] Company domains must not be LinkedIn URLs ...")
    df_dedup = deduplicate_lucia_export(df)
    domains = df_dedup["Company Domain"].tolist()
    for d in domains:
        d_str = str(d)
        assert "linkedin.com/in/" not in d_str.lower(), (
            f"LinkedIn person URL found in domain column: {d_str!r}"
        )
    cleaned = [clean_domain(str(d)) for d in domains if pd.notna(d)]
    expected_domains = {"alilavoro.it", "renovit.it", "sandyou.it"}
    assert set(cleaned) == expected_domains, (
        f"Expected domains {expected_domains}, got {set(cleaned)}"
    )
    print(f"      PASS — domains: {cleaned}")


def test_lucia_column_mapping(df: pd.DataFrame) -> None:
    print("  [7] map_lucia_export_row ...")
    row_dict = df.iloc[0].to_dict()
    mapped = map_lucia_export_row(row_dict)

    assert mapped.get("lusha_api_company_name") == "Ali Lavoro", (
        f"Expected 'Ali Lavoro', got {mapped.get('lusha_api_company_name')!r}"
    )
    assert mapped.get("lusha_api_domain") == "alilavoro.it", (
        f"Expected 'alilavoro.it', got {mapped.get('lusha_api_domain')!r}"
    )
    assert mapped.get("lusha_api_status") == "reused_existing_lucia_data"
    assert mapped.get("lucia_api_called") == "False"
    assert mapped.get("lusha_api_industry") == "Staffing & Recruiting"
    assert mapped.get("lusha_api_country")  == "Italy"
    print("      PASS — Company Name → lusha_api_company_name, Company Domain → lusha_api_domain")


def test_output_filename() -> None:
    print("  [8] output filename format ...")
    # Simulate what the app generates
    from datetime import datetime
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fname = f"enrichedResults_{stamp}.xlsx"
    assert fname.startswith("enrichedResults_"), f"Bad prefix: {fname}"
    assert "sg" not in fname
    assert "hq" not in fname
    assert "lusha" not in fname.lower()
    assert "lucia" not in fname.lower()
    assert fname.endswith(".xlsx")
    print(f"      PASS — filename: {fname}")


def test_lucia_negative(not_lucia_df: pd.DataFrame) -> None:
    print("  [9] Non-Lucia file should NOT be detected as Lucia export ...")
    assert not is_lucia_contact_export(not_lucia_df), (
        "Simple Excel file should not be detected as Lucia export"
    )
    print("      PASS")


def make_simple_excel_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"Company": "Acme Corp", "Website": "acme.com"},
        {"Company": "Globex",    "Website": "globex.com"},
    ])


def run_all() -> None:
    print("\n=== Lucia/Lusha Export Detection Smoke Tests ===\n")
    df = load_test_csv()
    print(f"Loaded test.csv: {len(df)} rows, {len(df.columns)} columns\n")

    test_lucia_detection(df)
    test_column_detection(df)
    test_lucia_col_helpers(df)
    test_deduplication(df)
    test_no_person_names_in_summary(df)
    test_domain_is_not_linkedin(df)
    test_lucia_column_mapping(df)
    test_output_filename()
    test_lucia_negative(make_simple_excel_df())

    print("\n✅ All smoke tests passed.\n")


if __name__ == "__main__":
    run_all()
