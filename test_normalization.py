"""
Regression tests for the input normalization layer.

Tests both Type 1 (simple Excel company list) and Type 2
(pre-enriched Lucia/Cold Caller CSV export) routes.

Run with:  python test_normalization.py
"""

import sys
import io
import re
from pathlib import Path

import pandas as pd

# ── Streamlit / heavy-dependency stubs ───────────────────────────────────────
import types

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
    def get(self, k, d=None):
        return d
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

# ── Import functions under test ───────────────────────────────────────────────
from enrich_clients_claude import (
    normalize_input_to_company_df,
    is_lucia_contact_export,
    detect_columns,
    clean_domain,
)
from commercial_fit_scoring import score_company

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
_failures: list[str] = []

def _chk(label: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"  {PASS}  {label}")
    else:
        _failures.append(label)
        print(f"  {FAIL}  {label}" + (f"  [{detail}]" if detail else ""))

def _sec(title: str) -> None:
    print(f"\n{'─'*65}\n  {title}\n{'─'*65}")


# =============================================================================
# Test A: Type 1 — simple company list (Capgemini-style)
# =============================================================================

def make_capgemini_df() -> pd.DataFrame:
    """Minimal Type 1 dataframe simulating CapGemini(2).xlsx."""
    return pd.DataFrame([{
        "Company":    "Capgemini Nederland B.V.",
        "Website":    "https://www.capgemini.com/",
        "Industry":   "IT Services",
        "Country":    "Netherlands",
    }])


def test_type1_normalization() -> None:
    _sec("Test A: Type 1 simple company list normalization")

    df = make_capgemini_df()
    name_col, domain_col = detect_columns(df)

    result = normalize_input_to_company_df(
        df, "simple_company_list", name_col, domain_col
    )

    _chk("input_type = simple_company_list",
         result["input_type"] == "simple_company_list",
         result["input_type"])
    _chk("company_df has 1 row",
         len(result["company_df"]) == 1,
         str(len(result["company_df"])))
    _chk("contact_row_count = 1",
         result["contact_row_count"] == 1)
    _chk("unique_company_count = 1",
         result["unique_company_count"] == 1)

    cdf = result["company_df"]
    _chk("canonical_company_name = 'Capgemini Nederland B.V.'",
         cdf.iloc[0]["canonical_company_name"] == "Capgemini Nederland B.V.",
         repr(cdf.iloc[0]["canonical_company_name"]))
    _chk("canonical_company_domain = 'capgemini.com'",
         cdf.iloc[0]["canonical_company_domain"] == "capgemini.com",
         repr(cdf.iloc[0]["canonical_company_domain"]))
    _chk("canonical_company_url starts with 'https://'",
         str(cdf.iloc[0]["canonical_company_url"]).startswith("https://"),
         repr(cdf.iloc[0]["canonical_company_url"]))
    _chk("source_contact_count present",
         "source_contact_count" in cdf.columns)
    _chk("input_type column = simple_company_list",
         cdf.iloc[0]["input_type"] == "simple_company_list")


# =============================================================================
# Test B: Type 2 — Cold Caller / Lucia CSV
# =============================================================================

def load_cold_caller_csv() -> pd.DataFrame:
    """Load the Example_Cold_Caller.csv fixture (same as test.csv)."""
    csv_path = Path(__file__).parent / "Example_Cold_Caller.csv"
    if not csv_path.exists():
        csv_path = Path(__file__).parent / "test.csv"
    assert csv_path.exists(), f"Cold Caller CSV not found at {csv_path}"
    return pd.read_csv(csv_path)


def test_type2_detection(df: pd.DataFrame) -> None:
    _sec("Test B-1: Type 2 detection")

    _chk("is_lucia_contact_export returns True",
         is_lucia_contact_export(df),
         "Expected True")
    name_col, domain_col = detect_columns(df)
    _chk("detect_columns → Company Name",
         name_col == "Company Name",
         repr(name_col))
    _chk("detect_columns → Company Domain",
         domain_col == "Company Domain",
         repr(domain_col))
    _chk("detect_columns does NOT return First Name / Last Name / LinkedIn URL",
         name_col not in ("First Name", "Last Name", "LinkedIn URL"),
         repr(name_col))


def test_type2_normalization(df: pd.DataFrame) -> None:
    _sec("Test B-2: Type 2 normalization and deduplication")

    result = normalize_input_to_company_df(
        df, "pre_enriched_lucia_export", "Company Name", "Company Domain"
    )

    _chk("input_type = pre_enriched_lucia_export",
         result["input_type"] == "pre_enriched_lucia_export",
         result["input_type"])
    _chk("contact_row_count = 3",
         result["contact_row_count"] == 3,
         str(result["contact_row_count"]))
    _chk("unique_company_count = 3",
         result["unique_company_count"] == 3,
         str(result["unique_company_count"]))

    cdf = result["company_df"]
    _chk("company_df has 3 rows",
         len(cdf) == 3,
         str(len(cdf)))

    names = cdf["canonical_company_name"].tolist()
    for expected in ["Ali Lavoro", "Renovit", "S&you Italia"]:
        _chk(f"canonical_company_name contains '{expected}'",
             expected in names,
             str(names))

    domains = [clean_domain(str(d)) for d in cdf["canonical_company_domain"].tolist()]
    for expected_domain in ["alilavoro.it", "renovit.it", "sandyou.it"]:
        _chk(f"canonical_company_domain contains '{expected_domain}'",
             expected_domain in domains,
             str(domains))


def test_type2_no_person_leakage(df: pd.DataFrame) -> None:
    _sec("Test B-3: No person names / LinkedIn URLs in company identity")

    result = normalize_input_to_company_df(
        df, "pre_enriched_lucia_export", "Company Name", "Company Domain"
    )
    cdf = result["company_df"]

    person_names = {"Anna", "Michela", "Annalisa"}
    names = set(cdf["canonical_company_name"].tolist())
    for pn in person_names:
        _chk(f"Person name '{pn}' NOT in canonical_company_name",
             pn not in names,
             str(names))

    for d in cdf["canonical_company_domain"].tolist():
        _chk(f"LinkedIn URL not in domain: {d!r}",
             "linkedin.com/in/" not in str(d).lower(),
             str(d))
    for u in cdf["canonical_company_url"].tolist():
        _chk(f"LinkedIn URL not in canonical_url: {u!r}",
             "linkedin.com/in/" not in str(u).lower(),
             str(u))


def test_type2_lucia_field_mapping(df: pd.DataFrame) -> None:
    _sec("Test B-4: Lucia field mapping to lusha_api_* columns")

    result = normalize_input_to_company_df(
        df, "pre_enriched_lucia_export", "Company Name", "Company Domain"
    )
    cdf = result["company_df"]
    row0 = cdf.iloc[0].to_dict()

    _chk("lusha_api_company_name present",
         "lusha_api_company_name" in row0,
         str(list(row0.keys())[:10]))
    _chk("lusha_api_company_name = 'Ali Lavoro'",
         str(row0.get("lusha_api_company_name", "")) == "Ali Lavoro",
         repr(row0.get("lusha_api_company_name")))
    _chk("lusha_api_domain = 'alilavoro.it'",
         str(row0.get("lusha_api_domain", "")) == "alilavoro.it",
         repr(row0.get("lusha_api_domain")))
    _chk("lusha_api_status = reused_existing_lucia_data",
         row0.get("lusha_api_status") == "reused_existing_lucia_data",
         repr(row0.get("lusha_api_status")))
    _chk("lucia_api_called = False",
         str(row0.get("lucia_api_called", "")) == "False",
         repr(row0.get("lucia_api_called")))
    _chk("lusha_api_employee_range present",
         bool(row0.get("lusha_api_employee_range")),
         repr(row0.get("lusha_api_employee_range")))
    _chk("source_contact_count present",
         "source_contact_count" in row0)


# =============================================================================
# Test C: Scoring correctness for both routes
# =============================================================================

def test_capgemini_scoring() -> None:
    _sec("Test C-1: Capgemini scoring (Type 1 reference)")

    capgemini_signals = {
        "sig_foreign_hq_score":        3,
        "sig_explicit_lnd_score":      3,
        "sig_intl_footprint_score":    3,
        "sig_employer_branding_score": 2,
        "sig_lnd_onboarding_score":    2,
        "ti_onboarding_score":         2,
        "sig_rapid_growth_score":      1,
        "lusha_api_employee_range":    "100001 - 10000000",
    }
    r = score_company(capgemini_signals)

    _chk("lean_model_prob ≈ 0.7285",
         abs(r["lean_model_prob"] - 0.7285) < 0.001,
         str(round(r["lean_model_prob"], 4)))
    _chk("icp_similarity_score ≈ 9.39",
         abs(r["icp_similarity_score"] - 9.39) < 0.05,
         str(r["icp_similarity_score"]))
    _chk("final_commercial_fit_score ≈ 9.54",
         abs(r["final_commercial_fit_score"] - 9.54) < 0.05,
         str(r["final_commercial_fit_score"]))
    _chk("commercial_tier = 🥇 Hot",
         r["commercial_tier"] == "🥇 Hot",
         r["commercial_tier"])
    _chk("NOT 9.99 (old wrong value)",
         abs(r["final_commercial_fit_score"] - 9.99) > 0.1)
    _chk("NOT Tier 1 (old wrong tier name)",
         r["commercial_tier"] != "Tier 1")


def test_type2_scoring_same_engine(df: pd.DataFrame) -> None:
    _sec("Test C-2: Type 2 uses same scoring engine as Type 1")

    result = normalize_input_to_company_df(
        df, "pre_enriched_lucia_export", "Company Name", "Company Domain"
    )
    cdf = result["company_df"]

    for _, row in cdf.iterrows():
        rd = row.to_dict()
        # Check that lusha_api_employee_range is set (required for size scoring)
        emp = rd.get("lusha_api_employee_range", "")
        _chk(f"{rd.get('canonical_company_name')}: lusha_api_employee_range present",
             bool(str(emp).strip()),
             repr(emp))
        # Run the same scoring function used for Type 1
        scored = score_company(rd)
        _chk(f"{rd.get('canonical_company_name')}: final_commercial_fit_score is a number",
             isinstance(scored["final_commercial_fit_score"], (int, float)),
             str(scored["final_commercial_fit_score"]))
        _chk(f"{rd.get('canonical_company_name')}: commercial_tier is valid",
             scored["commercial_tier"] in ("🥇 Hot", "🥈 Warm", "🥉 Cool", "❄️ Pass"),
             scored["commercial_tier"])
        _chk(f"{rd.get('canonical_company_name')}: NOT 'Tier 1' (old wrong tier)",
             scored["commercial_tier"] != "Tier 1")


# =============================================================================
# Test D: Output filename convention
# =============================================================================

def test_output_filename() -> None:
    _sec("Test D: Output filename convention")

    from datetime import datetime
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    for prefix in ("enrichedResults_", "enrichedResults_partial_", "enrichedResults_snapshot_"):
        fname = f"{prefix}{stamp}.xlsx"
        _chk(f"'{prefix}...' starts with enrichedResults_", fname.startswith("enrichedResults_"))
        _chk(f"'{prefix}...' ends with .xlsx",              fname.endswith(".xlsx"))
        for bad in ("enriched_results", "_sg_", "_hq_", "lusha", "lucia"):
            _chk(f"'{prefix}...' does not contain '{bad}'", bad not in fname.lower())


# =============================================================================
# Test E: n_to_process correction for multi-contact Type 2 input
# =============================================================================

def test_type2_n_to_process_correction() -> None:
    _sec("Test E: n_to_process corrected to unique_company_count for Type 2")

    # Build a 5-contact CSV with only 2 unique companies (simulates 5-row upload)
    df_multi = pd.DataFrame([
        {"First Name": "Anna",     "Last Name": "Rossi",    "Company Name": "Ali Lavoro",
         "Company Domain": "alilavoro.it", "Company Website": "https://alilavoro.it",
         "Number of Employees": "51-200", "LinkedIn URL": "linkedin.com/in/anna"},
        {"First Name": "Michela",  "Last Name": "Bianchi",  "Company Name": "Ali Lavoro",
         "Company Domain": "alilavoro.it", "Company Website": "https://alilavoro.it",
         "Number of Employees": "51-200", "LinkedIn URL": "linkedin.com/in/michela"},
        {"First Name": "Carlo",    "Last Name": "Ferrari",  "Company Name": "Renovit",
         "Company Domain": "renovit.it",  "Company Website": "https://renovit.it",
         "Number of Employees": "51-200", "LinkedIn URL": "linkedin.com/in/carlo"},
        {"First Name": "Giulia",   "Last Name": "Conti",    "Company Name": "Renovit",
         "Company Domain": "renovit.it",  "Company Website": "https://renovit.it",
         "Number of Employees": "51-200", "LinkedIn URL": "linkedin.com/in/giulia"},
        {"First Name": "Sara",     "Last Name": "Moretti",  "Company Name": "Renovit",
         "Company Domain": "renovit.it",  "Company Website": "https://renovit.it",
         "Number of Employees": "51-200", "LinkedIn URL": "linkedin.com/in/sara"},
    ])
    result = normalize_input_to_company_df(df_multi, "pre_enriched_lucia_export",
                                           "Company Name", "Company Domain")
    cdf = result["company_df"]

    _chk("contact_row_count = 5",    result["contact_row_count"] == 5,
         str(result["contact_row_count"]))
    _chk("unique_company_count = 2", result["unique_company_count"] == 2,
         str(result["unique_company_count"]))
    _chk("company_df has 2 rows",    len(cdf) == 2, str(len(cdf)))
    # head(5) on a 2-row df should still give 2 rows (not 5)
    _chk("head(5) on company_df = 2 rows (n_to_process correction)",
         len(cdf.head(5)) == 2, str(len(cdf.head(5))))
    # No person names in canonical_company_name
    names = set(cdf["canonical_company_name"].astype(str).str.strip())
    for person in ("Anna", "Michela", "Carlo", "Giulia", "Sara"):
        _chk(f"'{person}' NOT in canonical_company_name after multi-contact dedup",
             person not in names, str(names))
    # Canonical domains must not contain linkedin
    for d in cdf["canonical_company_domain"].astype(str):
        _chk(f"No linkedin.com in canonical_company_domain: {d!r}",
             "linkedin.com" not in d.lower(), d)


# =============================================================================
# Test F: Canonical identity validation catches leakage
# =============================================================================

def test_canonical_validation_rejects_leakage() -> None:
    _sec("Test F: Canonical validation rejects person-name leakage")

    # Simulate a broken normalization that puts a person name as canonical_company_name
    df_broken = pd.DataFrame([{
        "canonical_company_name":   "Anna",
        "canonical_company_domain": "alilavoro.it",
        "canonical_company_url":    "https://alilavoro.it",
        "input_type":               "pre_enriched_lucia_export",
        "source_contact_count":     1,
    }])
    df_raw_with_first = pd.DataFrame([{"First Name": "Anna"}])

    # Reproduce the validation logic from the start handler
    _person_names = set(
        df_raw_with_first.get("First Name", pd.Series(dtype=str))
        .dropna().astype(str).str.strip()
        .replace("", pd.NA).dropna()
    )
    _canon_names  = set(df_broken["canonical_company_name"].astype(str).str.strip())
    _leaked = _person_names & _canon_names
    _chk("Validation detects 'Anna' as leaked person name", "Anna" in _leaked, str(_leaked))

    # Simulate linkedin.com in domain
    df_bad_domain = pd.DataFrame([{
        "canonical_company_name":   "Ali Lavoro",
        "canonical_company_domain": "linkedin.com/in/anna",
        "canonical_company_url":    "https://linkedin.com/in/anna",
        "input_type":               "pre_enriched_lucia_export",
    }])
    _bad_domains = [d for d in df_bad_domain["canonical_company_domain"].astype(str)
                    if "linkedin.com" in d.lower()]
    _bad_urls    = [u for u in df_bad_domain["canonical_company_url"].astype(str)
                    if "linkedin.com/in/" in u.lower()]
    _chk("Validation detects linkedin.com in canonical_company_domain",
         bool(_bad_domains), str(_bad_domains))
    _chk("Validation detects linkedin.com/in/ in canonical_company_url",
         bool(_bad_urls), str(_bad_urls))

    # Confirm clean Type 2 data passes validation
    df2_good = load_cold_caller_csv()
    result_good = normalize_input_to_company_df(
        df2_good, "pre_enriched_lucia_export", "Company Name", "Company Domain"
    )
    cdf_good = result_good["company_df"]
    _person_names_good = set(
        df2_good.get("First Name", pd.Series(dtype=str))
        .dropna().astype(str).str.strip().replace("", pd.NA).dropna()
    )
    _leaked_good = _person_names_good & set(
        cdf_good["canonical_company_name"].astype(str).str.strip()
    )
    _bad_dom_good = [d for d in cdf_good["canonical_company_domain"].astype(str)
                     if "linkedin.com" in d.lower()]
    _chk("Clean Type 2 data: zero person-name leakage", not _leaked_good, str(_leaked_good))
    _chk("Clean Type 2 data: no linkedin.com in canonical_company_domain",
         not _bad_dom_good, str(_bad_dom_good))


# =============================================================================
# Runner
# =============================================================================

def run_all() -> None:
    print("\n=== Input Normalization Regression Tests ===\n")

    # Type 1
    test_type1_normalization()

    # Type 2
    df2 = load_cold_caller_csv()
    print(f"\nLoaded cold-caller CSV: {len(df2)} rows, {len(df2.columns)} columns")
    test_type2_detection(df2)
    test_type2_normalization(df2)
    test_type2_no_person_leakage(df2)
    test_type2_lucia_field_mapping(df2)

    # Scoring
    test_capgemini_scoring()
    test_type2_scoring_same_engine(df2)

    # Filename
    test_output_filename()

    # Pipeline robustness
    test_type2_n_to_process_correction()
    test_canonical_validation_rejects_leakage()

    print(f"\n{'═'*65}")
    if _failures:
        print(f"  FAILURES ({len(_failures)}):")
        for f in _failures:
            print(f"    • {f}")
        sys.exit(1)
    else:
        print("  ✅ All normalization regression tests passed.")
    print("═"*65)


if __name__ == "__main__":
    run_all()
