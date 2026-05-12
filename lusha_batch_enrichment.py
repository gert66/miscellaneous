"""
Lusha Batch Company Enrichment
================================
Upload a file with company names (and optionally domains/websites),
enrich each row with Lusha firmographic data, and download the result.

Architecture
------------
- One row is processed per Streamlit rerun so the Stop button works at any point.
- All mutable run state lives in st.session_state (keys prefixed with _).
- Normal mode: clean, non-technical UI.
- Debug mode: toggled via sidebar checkbox; shows API details, raw JSON, cache tools.
"""

import io
import json
import re
import time
import unicodedata
import zipfile
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

LUSHA_ENDPOINT = "https://api.lusha.com/v2/company"
CACHE_DIR = Path("lusha_json_cache")

# All columns added by enrichment (company-level firmographics only — no personal data)
LUSHA_FIELDS = [
    # ── Core firmographics ────────────────────────────────────────────────────
    "lusha_company_name",
    "lusha_domain",
    "lusha_industry",
    "lusha_sub_industry",
    "lusha_employee_count",
    "lusha_employee_range",
    "lusha_revenue",
    "lusha_description",
    "lusha_linkedin_url",
    "lusha_specialties",
    "lusha_technologies",
    "lusha_founded_year",
    # ── HQ location (single values for easy filtering) ────────────────────────
    "lusha_country",
    "lusha_city",
    "lusha_state",
    "lusha_headquarters_country",
    "lusha_headquarters_city",
    # ── Full location footprint ───────────────────────────────────────────────
    "lusha_location_count",
    "lusha_location_countries",
    "lusha_location_cities",
    "lusha_location_continents",
    "lusha_multi_country_presence",
    "lusha_location_summary",
    "lusha_international_footprint_score",
    # ── Enrichment metadata ───────────────────────────────────────────────────
    "enrichment_status",
    "match_confidence",
    "needs_manual_review",
    "match_notes",
    "lusha_error_message",
]

# A row counts as having usable data only if at least one of these is non-empty
LUSHA_DATA_FIELDS = [
    "lusha_company_name",
    "lusha_domain",
    "lusha_industry",
    "lusha_employee_range",
    "lusha_country",
    "lusha_description",
]

# Keys stripped from API responses before caching — we never store personal data
_PERSONAL_KEYS = frozenset({
    "contacts", "people", "emails", "phones", "mobileNumbers",
    "directDials", "personalEmails", "businessEmails", "phoneNumbers",
    "persons", "personData", "contactData",
})

# Column-name hints for auto-detection
_COMPANY_HINTS = ["company", "account", "organisation", "organization", "name", "naam", "bedrijf"]
_DOMAIN_HINTS  = ["domain", "website", "url", "web", "site", "domein"]

# Human-readable labels shown in the results summary
_STATUS_LABELS = {
    "enriched_by_domain":       "Enriched via domain",
    "enriched_by_company_name": "Enriched via name",
    "cached":                   "From cache",
    "no_match":                 "No match",
    "no_data_returned":         "No data returned",
    "api_error":                "API error",
}


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def clean_domain(raw: str) -> str:
    """Strip protocol, www, path and trailing slash from a URL or bare domain."""
    if not raw or not isinstance(raw, str):
        return ""
    d = raw.strip().lower()
    d = re.sub(r"^https?://", "", d)
    d = re.sub(r"^www\.", "", d)
    d = d.split("/")[0].strip()
    if d in {"nan", "none", ""} or " " in d:
        return ""
    return d


def safe_filename(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = re.sub(r"[^\w\s\-.]", "", text)
    text = re.sub(r"\s+", "_", text).strip("_")
    return text[:120] or "unknown"


def str_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def detect_columns(df: pd.DataFrame) -> tuple:
    """Return (name_col, domain_col); domain_col may be None."""
    cols      = df.columns.tolist()
    col_lower = [str(c).lower() for c in cols]

    def best(hints):
        scores = [(max(str_similarity(cl, h) for h in hints), i)
                  for i, cl in enumerate(col_lower)]
        score, idx = max(scores)
        return cols[idx], score

    name_col,   ns = best(_COMPANY_HINTS)
    domain_col, ds = best(_DOMAIN_HINTS)
    return (
        name_col   if ns >= 0.45 else None,
        domain_col if ds >= 0.55 and domain_col != name_col else None,
    )


def strip_personal_keys(data) -> dict:
    """Recursively remove personal/contact keys so we never cache personal data."""
    if isinstance(data, dict):
        return {k: strip_personal_keys(v) for k, v in data.items()
                if k not in _PERSONAL_KEYS}
    if isinstance(data, list):
        return [strip_personal_keys(x) for x in data]
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Cache
# ─────────────────────────────────────────────────────────────────────────────

def load_cache(cache_key: str):
    path = CACHE_DIR / f"{safe_filename(cache_key)}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def save_cache(cache_key: str, data: dict) -> None:
    """Save response to disk after stripping personal keys."""
    CACHE_DIR.mkdir(exist_ok=True)
    clean = strip_personal_keys(data)
    (CACHE_DIR / f"{safe_filename(cache_key)}.json").write_text(
        json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def get_cache_count() -> int:
    return len(list(CACHE_DIR.glob("*.json"))) if CACHE_DIR.exists() else 0


def list_cache_files() -> list:
    return sorted(CACHE_DIR.glob("*.json")) if CACHE_DIR.exists() else []


# ─────────────────────────────────────────────────────────────────────────────
# Lusha API  ← all API calls are made here
# ─────────────────────────────────────────────────────────────────────────────

def _api_get(params: dict, api_key: str) -> tuple:
    """
    GET https://api.lusha.com/v2/company
    Returns (parsed_json, response_object).
    Authentication: api_key header only (no Content-Type on a GET request).
    Raises requests.HTTPError on 4xx/5xx.
    """
    resp = requests.get(
        LUSHA_ENDPOINT,
        headers={"api_key": api_key},
        params=params,
        timeout=12,
    )
    resp.raise_for_status()
    return resp.json(), resp


def lusha_by_domain(domain: str, api_key: str) -> tuple:
    return _api_get({"domain": domain}, api_key)


def lusha_by_name(name: str, api_key: str) -> tuple:
    # v2 API: parameter is "company", NOT "name"
    return _api_get({"company": name}, api_key)


# ─────────────────────────────────────────────────────────────────────────────
# Field extraction  (firmographic only — no personal data)
# ─────────────────────────────────────────────────────────────────────────────

def extract_company_fields(raw: dict) -> dict:
    """
    Parse a Lusha v2 /company response into flat enrichment fields.
    v2 wraps all company data inside raw["data"].
    Key v2 field names: mainIndustry, companySize, revenueRange,
    companyLocations (array), employees (string range), social (dict).
    """
    data = raw.get("data") or raw  # v2 wraps in "data"; fall back to root

    def join_list(v):
        if isinstance(v, list):
            return ", ".join(str(x) for x in v if x)
        return str(v) if v else ""

    # ── Location extraction ───────────────────────────────────────────────────
    # companyLocations is an array of dicts; first pick the HQ entry for single-
    # value columns, then derive footprint metrics from the full array.
    locations = [loc for loc in (data.get("companyLocations") or [])
                 if isinstance(loc, dict)]
    hq = next((loc for loc in locations if loc.get("isHeadquarters")), locations[0] if locations else {})

    country = hq.get("country") or hq.get("country_iso2") or data.get("country") or ""
    city    = hq.get("city")    or data.get("city")    or ""
    state   = hq.get("state")   or hq.get("state_code") or data.get("state") or ""

    # Footprint — derived from the full locations list
    loc_countries   = [str(loc.get("country") or loc.get("country_iso2") or "").strip()
                       for loc in locations]
    loc_countries   = [c for c in loc_countries if c]
    loc_cities      = [str(loc.get("city") or "").strip() for loc in locations]
    loc_cities      = [c for c in loc_cities if c]
    loc_continents  = [str(loc.get("continent") or "").strip() for loc in locations]
    loc_continents  = [c for c in loc_continents if c]

    unique_countries  = list(dict.fromkeys(loc_countries))   # ordered, deduplicated
    unique_cities     = list(dict.fromkeys(loc_cities))
    unique_continents = list(dict.fromkeys(loc_continents))

    loc_count       = len(locations)
    multi_country   = "TRUE" if len(unique_countries) > 1 else ("FALSE" if unique_countries else "")

    # Footprint score 0-5 based on country and continent spread
    n_ctry = len(unique_countries)
    n_cont = len(unique_continents)
    if n_ctry == 0:
        footprint_score = 0
    elif n_ctry == 1 and loc_count == 1:
        footprint_score = 1
    elif n_ctry == 1:
        footprint_score = 2
    elif n_ctry <= 3:
        footprint_score = 3
    elif n_ctry <= 6 or n_cont >= 2:
        footprint_score = 4
    else:
        footprint_score = 5

    # Human-readable summary  e.g. "HQ: Amsterdam, Netherlands · 5 offices, 3 countries"
    hq_part   = ", ".join(p for p in [city, country] if p) or "unknown"
    loc_parts = []
    if loc_count > 1:
        loc_parts.append(f"{loc_count} offices")
    if len(unique_countries) > 1:
        loc_parts.append(f"{len(unique_countries)} countries")
    if unique_continents:
        loc_parts.append(f"{', '.join(unique_continents)}")
    loc_summary = f"HQ: {hq_part}"
    if loc_parts:
        loc_summary += " · " + ", ".join(loc_parts)

    # Employee range — "employees" is a human-readable string like "201 - 500"
    employee_range = data.get("employees") or ""
    company_size   = data.get("companySize") or {}
    if isinstance(company_size, dict) and not employee_range:
        lo, hi = company_size.get("min", ""), company_size.get("max", "")
        if lo or hi:
            employee_range = f"{lo} - {hi}" if (lo and hi) else str(lo or hi)
    employee_count = company_size.get("employeesInLinkedin") or ""

    # Revenue — revenueRange is an array of strings like ["$10M", "$50M"];
    # format as "$10M – $50M" when two bounds are present, otherwise join as-is.
    raw_rev = data.get("revenueRange") or data.get("revenue") or ""
    if isinstance(raw_rev, list):
        parts = [str(x).strip() for x in raw_rev if x]
        revenue = f"{parts[0]} – {parts[1]}" if len(parts) >= 2 else (parts[0] if parts else "")
    else:
        revenue = str(raw_rev).strip() if raw_rev else ""

    # LinkedIn URL — v2 returns it inside the "social" dict, sometimes as a
    # plain string and sometimes as {"url": "https://..."}.  Unwrap both forms.
    def _extract_url(v) -> str:
        if isinstance(v, dict):
            return str(v.get("url") or v.get("href") or "").strip()
        return str(v).strip() if v else ""

    social = data.get("social") if isinstance(data.get("social"), dict) else {}
    linkedin = (
        _extract_url(social.get("linkedin"))
        or _extract_url(social.get("linkedinUrl"))
        or _extract_url(data.get("linkedinUrl"))
        or _extract_url(data.get("linkedin"))
        or ""
    )

    technologies = join_list(data.get("technologies") or data.get("techStack"))
    specialties  = join_list(data.get("specialties")  or data.get("specialities"))

    return {
        # Core firmographics
        "lusha_company_name":   data.get("name")        or data.get("companyName")  or "",
        "lusha_domain":         data.get("domain")       or data.get("fqdn")         or data.get("emailDomain") or "",
        "lusha_industry":       data.get("mainIndustry") or data.get("industry")     or "",
        "lusha_sub_industry":   data.get("subIndustry")  or "",
        "lusha_employee_count": str(employee_count) if employee_count else "",
        "lusha_employee_range": employee_range,
        "lusha_revenue":        revenue,
        "lusha_description":    data.get("description") or "",
        "lusha_linkedin_url":   linkedin,
        "lusha_specialties":    specialties,
        "lusha_technologies":   technologies,
        "lusha_founded_year":   str(data.get("founded") or data.get("foundedYear") or ""),
        # HQ location
        "lusha_country":              country,
        "lusha_city":                 city,
        "lusha_state":                state,
        "lusha_headquarters_country": country,
        "lusha_headquarters_city":    city,
        # Full location footprint
        "lusha_location_count":                str(loc_count) if loc_count else "",
        "lusha_location_countries":            ", ".join(unique_countries),
        "lusha_location_cities":               ", ".join(unique_cities),
        "lusha_location_continents":           ", ".join(unique_continents),
        "lusha_multi_country_presence":        multi_country,
        "lusha_location_summary":              loc_summary,
        "lusha_international_footprint_score": str(footprint_score) if loc_count else "",
    }


def _empty(status: str, confidence: str, error_msg: str = "") -> dict:
    fields = {k: "" for k in LUSHA_FIELDS}
    fields["enrichment_status"]   = status
    fields["match_confidence"]    = confidence
    fields["lusha_error_message"] = error_msg
    fields["needs_manual_review"] = ""
    fields["match_notes"]         = ""
    return fields


def has_data(fields: dict) -> bool:
    return any(fields.get(f, "") for f in LUSHA_DATA_FIELDS)


def flag_review(fields: dict, input_company_name: str) -> dict:
    """
    Populate needs_manual_review (True/False) and match_notes.
    Called after enrichment_status and match_confidence are set.
    Rules:
      - match_confidence == "low"
      - enrichment_status in (no_match, api_error, no_data_returned)
      - returned Lusha company name is materially different from the input name
        (similarity < 0.55, ignoring case/punctuation)
    """
    reasons: list[str] = []
    status     = fields.get("enrichment_status", "")
    confidence = fields.get("match_confidence", "")
    returned   = fields.get("lusha_company_name", "")

    if status in ("no_match", "api_error", "no_data_returned"):
        reasons.append(f"enrichment_status is {status}")

    if confidence == "low":
        reasons.append("match confidence is low")

    if returned and input_company_name:
        sim = str_similarity(input_company_name, returned)
        if sim < 0.55:
            reasons.append(
                f"returned name '{returned}' differs from input '{input_company_name}' "
                f"(similarity {sim:.0%})"
            )

    fields["needs_manual_review"] = "TRUE" if reasons else "FALSE"
    fields["match_notes"]         = "; ".join(reasons) if reasons else ""
    return fields


def _http_error_msg(e: requests.HTTPError) -> str:
    code = e.response.status_code if e.response is not None else "?"
    try:
        body = e.response.json()
        msg  = body.get("message") or body.get("error") or body.get("detail") or str(body)
    except Exception:
        msg  = e.response.text[:400] if e.response is not None else str(e)
    return f"HTTP {code}: {msg}"


def _error_json(e: requests.HTTPError) -> dict:
    try:
        return e.response.json()
    except Exception:
        return {"raw_error": e.response.text[:500] if e.response else str(e)}


def _keep_headers(headers) -> dict:
    return {k: v for k, v in dict(headers).items()
            if any(x in k.lower() for x in
                   ["ratelimit", "x-credits", "x-lusha", "retry-after"])}


# ─────────────────────────────────────────────────────────────────────────────
# Per-row enrichment  ← enrichment logic lives here
# ─────────────────────────────────────────────────────────────────────────────

def enrich_one_row(
    company_name: str,
    raw_domain: str,
    api_key: str,
    delay: float,
) -> tuple:
    """
    Returns (lusha_fields: dict, debug_record: dict).

    Strategy:
      1. If a domain is present → domain lookup (cache first).
         HTTP 404 → fall through to name lookup.
      2. Company-name lookup (cache first).
    Personal keys are stripped from the raw JSON before caching.
    The debug_record is always populated, regardless of debug mode.
    """
    domain       = clean_domain(raw_domain)
    company_name = str(company_name).strip() if company_name else ""

    dbg: dict = {
        "input_company_name":  company_name,
        "input_domain":        raw_domain,
        "cleaned_domain":      domain,
        "lookup_method":       None,
        "endpoint":            LUSHA_ENDPOINT,
        "params":              {},
        "http_status":         None,
        "response_headers":    {},
        "raw_json":            None,
        "enrichment_status":   "",
        "match_confidence":    "",
        "lusha_error_message": "",
    }

    def _done(fields: dict) -> tuple:
        flag_review(fields, company_name)        # sets needs_manual_review + match_notes
        dbg["enrichment_status"]   = fields.get("enrichment_status", "")
        dbg["match_confidence"]    = fields.get("match_confidence",  "")
        dbg["lusha_error_message"] = fields.get("lusha_error_message", "")
        return fields, dbg

    # ── 1. Domain path ────────────────────────────────────────────────────────
    if domain:
        dbg["lookup_method"] = "domain"
        dbg["params"]        = {"domain": domain}
        cache_key = f"domain_{domain}"
        cached = load_cache(cache_key)

        if cached is not None:
            dbg["http_status"] = "cached"
            dbg["raw_json"]    = cached
            fields = extract_company_fields(cached)
            if not has_data(fields):
                return _done(_empty("no_data_returned", "no_match",
                                    "Cache hit but all company fields are empty"))
            fields["enrichment_status"] = "cached"
            fields["match_confidence"]  = "high"
            return _done(fields)

        try:
            time.sleep(delay)
            raw_json, resp = lusha_by_domain(domain, api_key)
            dbg["http_status"]      = resp.status_code
            dbg["response_headers"] = _keep_headers(resp.headers)
            dbg["raw_json"]         = strip_personal_keys(raw_json)
            save_cache(cache_key, raw_json)
            fields = extract_company_fields(raw_json)
            if not has_data(fields):
                return _done(_empty(
                    "no_data_returned", "no_match",
                    f"HTTP 200 but no company fields in response. Preview: {str(raw_json)[:300]}",
                ))
            fields["enrichment_status"] = "enriched_by_domain"
            fields["match_confidence"]  = "high"
            return _done(fields)

        except requests.HTTPError as e:
            dbg["http_status"] = e.response.status_code if e.response else "?"
            dbg["raw_json"]    = _error_json(e)
            code = e.response.status_code if e.response is not None else 0
            if code == 404:
                pass  # fall through to name lookup
            else:
                return _done(_empty("api_error", "no_match", _http_error_msg(e)))
        except Exception as e:
            dbg["http_status"] = "error"
            return _done(_empty("api_error", "no_match", str(e)))

    # ── 2. Name path ──────────────────────────────────────────────────────────
    if not company_name:
        dbg["lookup_method"] = "none"
        return _done(_empty("no_match", "no_match", "No company name or domain provided"))

    dbg["lookup_method"] = "company_name"
    dbg["params"]        = {"company": company_name}
    cache_key = f"name_{company_name}"
    cached = load_cache(cache_key)

    if cached is not None:
        dbg["http_status"] = "cached"
        dbg["raw_json"]    = cached
        fields = extract_company_fields(cached)
        if not has_data(fields):
            return _done(_empty("no_data_returned", "no_match",
                                "Cache hit but all company fields are empty"))
        conf = "medium" if str_similarity(company_name,
                                          fields.get("lusha_company_name", "")) >= 0.6 else "low"
        fields["enrichment_status"] = "cached"
        fields["match_confidence"]  = conf
        return _done(fields)

    try:
        time.sleep(delay)
        raw_json, resp = lusha_by_name(company_name, api_key)
        dbg["http_status"]      = resp.status_code
        dbg["response_headers"] = _keep_headers(resp.headers)
        dbg["raw_json"]         = strip_personal_keys(raw_json)
        save_cache(cache_key, raw_json)
        fields = extract_company_fields(raw_json)
        if not has_data(fields):
            return _done(_empty(
                "no_data_returned", "no_match",
                f"HTTP 200 but no company fields in response. Preview: {str(raw_json)[:300]}",
            ))
        conf = "medium" if str_similarity(company_name,
                                          fields.get("lusha_company_name", "")) >= 0.6 else "low"
        fields["enrichment_status"] = "enriched_by_company_name"
        fields["match_confidence"]  = conf
        return _done(fields)

    except requests.HTTPError as e:
        dbg["http_status"] = e.response.status_code if e.response else "?"
        dbg["raw_json"]    = _error_json(e)
        code = e.response.status_code if e.response is not None else 0
        if code == 404:
            return _done(_empty("no_match", "no_match", "HTTP 404: Company not found"))
        return _done(_empty("api_error", "no_match", _http_error_msg(e)))
    except Exception as e:
        dbg["http_status"] = "error"
        return _done(_empty("api_error", "no_match", str(e)))


# ─────────────────────────────────────────────────────────────────────────────
# Download helpers  ← output files are created here
# ─────────────────────────────────────────────────────────────────────────────

def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Enriched")
    return buf.getvalue()


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def records_to_jsonl_bytes(records: list) -> bytes:
    lines = [json.dumps(r, ensure_ascii=False) for r in records if r is not None]
    return "\n".join(lines).encode("utf-8")


def make_log_df(debug_records: list) -> pd.DataFrame:
    """Build the processing log DataFrame from per-row debug records."""
    rows = []
    for d in debug_records:
        rows.append({
            "input_company_name":  d.get("input_company_name", ""),
            "input_domain":        d.get("input_domain", ""),
            "lookup_method":       d.get("lookup_method", ""),
            "http_status":         d.get("http_status", ""),
            "enrichment_status":   d.get("enrichment_status", ""),
            "match_confidence":    d.get("match_confidence", ""),
            "needs_manual_review": d.get("needs_manual_review", ""),
            "match_notes":         d.get("match_notes", ""),
            "lusha_error_message": d.get("lusha_error_message", ""),
        })
    return pd.DataFrame(rows)


def cache_to_zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in list_cache_files():
            zf.write(f, f.name)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Session-state helpers
# ─────────────────────────────────────────────────────────────────────────────

def ss(key, default=None):
    return st.session_state.get(key, default)


def ss_set(**kwargs):
    for k, v in kwargs.items():
        st.session_state[k] = v


def reset_processing():
    ss_set(
        processing=False, stop_requested=False,
        process_index=0, results=[], debug_records=[],
        enrichment_done=False, df_enriched=None,
    )


def build_and_finish(results: list, debug_records: list, df_work: pd.DataFrame) -> None:
    """Assemble the enriched DataFrame from per-row results and mark done."""
    df_out      = df_work.copy().reset_index(drop=True)
    enriched_df = pd.DataFrame(results)
    for col in LUSHA_FIELDS:
        df_out[col] = enriched_df[col].values if col in enriched_df.columns else ""
    ss_set(
        processing=False, stop_requested=False,
        enrichment_done=True, df_enriched=df_out,
        debug_records=debug_records,
    )
    st.rerun()


# =============================================================================
# UI  — normal mode starts here
# =============================================================================

st.set_page_config(
    page_title="Lusha Company Enrichment",
    page_icon="🏢",
    layout="wide",
)
st.title("🏢 Lusha Company Enrichment")
st.caption(
    "Upload a file with company names, enrich each row with Lusha company data, "
    "and download the results."
)

# =============================================================================
# SIDEBAR  — always visible: API key + debug toggle
#            debug-only items shown only when debug mode is ON
# =============================================================================

with st.sidebar:
    st.header("Settings")

    api_key = st.text_input(
        "Lusha API key",
        type="password",
        key="api_key_input",
        placeholder="Paste your API key here",
        help="Your Lusha API key. It is never stored or transmitted except to the Lusha API.",
    )

    st.divider()

    # ── Debug mode toggle ─────────────────────────────────────────────────────
    debug_mode = st.checkbox(
        "Enable debug mode",
        value=False,
        help=(
            "Shows API request details, raw JSON responses, "
            "cache management tools, and additional downloads."
        ),
    )

    # ── Debug-only sidebar controls ───────────────────────────────────────────
    # [DEBUG MODE — SIDEBAR] delay, cache count, clear cache, cache path
    if debug_mode:
        st.divider()
        st.subheader("Debug settings")
        delay_sec = st.slider(
            "Delay between API calls (sec)",
            min_value=0.0, max_value=3.0, value=0.5, step=0.1,
            help="Small pause between calls to reduce rate-limit risk.",
        )
        st.divider()
        st.metric("Cached companies", get_cache_count())
        st.caption(f"Cache folder: `{CACHE_DIR.resolve()}`")
        if st.button("Clear cache", use_container_width=True):
            if CACHE_DIR.exists():
                for f in CACHE_DIR.glob("*.json"):
                    f.unlink()
            st.success("Cache cleared.")
            st.rerun()
    else:
        delay_sec = 0.5  # default, hidden in normal mode

# =============================================================================
# STEP 1 — Upload file
# [FILE UPLOAD LOCATION] — st.file_uploader is here
# =============================================================================

st.subheader("Step 1 · Upload your file")
uploaded = st.file_uploader(
    "Drag and drop here, or click to browse  (.xlsx · .xls · .csv)",
    type=["xlsx", "xls", "csv"],
)

# Detect file change (new upload or removal) and reload from scratch
new_file_key = f"{uploaded.name}___{uploaded.size}" if uploaded else "__none__"
if new_file_key != ss("_file_key"):
    ss_set(_file_key=new_file_key, df_raw=None, file_name=None,
           file_type=None, file_error=None)
    reset_processing()
    if uploaded is not None:
        try:
            fname = uploaded.name
            if fname.lower().endswith(".csv"):
                df_loaded = pd.read_csv(uploaded)
                ftype = "CSV"
            else:
                df_loaded = pd.read_excel(uploaded)
                ftype = "Excel"
            ss_set(df_raw=df_loaded, file_name=fname, file_type=ftype)
        except Exception as exc:
            ss_set(file_error=str(exc))

df_raw: pd.DataFrame | None = ss("df_raw")
file_error: str | None      = ss("file_error")

if file_error:
    st.error(f"Could not read the file: {file_error}")
elif uploaded and df_raw is not None:
    st.success(
        f"**{ss('file_name')}** loaded — "
        f"{len(df_raw):,} rows, {len(df_raw.columns)} columns"
    )
    st.info(
        "💡 For large batches, this may take time and use one Lusha credit per company lookup. "
        "Start with a small test batch first."
    )

# =============================================================================
# STEP 2 — Preview input
# Only shown after a file is successfully loaded.
# =============================================================================

name_col     = None
domain_col   = None
n_to_process = 0

if df_raw is not None:

    st.divider()
    st.subheader("Step 2 · Preview")
    st.dataframe(df_raw.head(), use_container_width=True)
    st.caption(f"{len(df_raw):,} rows · {len(df_raw.columns)} columns")

    # =========================================================================
    # STEP 3 — Select columns
    # =========================================================================

    st.divider()
    st.subheader("Step 3 · Select columns")

    auto_name_col, auto_domain_col = detect_columns(df_raw)
    cols = df_raw.columns.tolist()

    sel_l, sel_r = st.columns(2)
    with sel_l:
        name_col = st.selectbox(
            "Company name column *",
            options=cols,
            index=cols.index(auto_name_col) if auto_name_col in cols else 0,
            help="Auto-detected — change if the wrong column is selected.",
        )
    with sel_r:
        _NO_DOMAIN = "(none — use company name only)"
        dom_opts   = [_NO_DOMAIN] + cols
        def_dom    = (
            dom_opts.index(auto_domain_col)
            if auto_domain_col and auto_domain_col in dom_opts else 0
        )
        dom_choice = st.selectbox(
            "Domain / website column (optional)",
            options=dom_opts,
            index=def_dom,
            help=(
                "If this column contains a domain or URL, it is used for lookup first. "
                "Falls back to company name if not found."
            ),
        )
    domain_col = dom_choice if dom_choice != _NO_DOMAIN else None

    # Auto-detection note
    note_parts = []
    if auto_name_col:
        note_parts.append(f"company name → **{auto_name_col}**")
    if auto_domain_col:
        note_parts.append(f"domain → **{auto_domain_col}**")
    if note_parts:
        st.caption("Auto-detected: " + ",  ".join(note_parts))
    else:
        st.caption("Could not auto-detect columns — please select them manually above.")

    # =========================================================================
    # STEP 4 — Processing scope
    # =========================================================================

    st.divider()
    st.subheader("Step 4 · Processing scope")

    limit_rows = st.checkbox(
        "Limit rows for testing",
        value=False,
        help="Process only a small batch first — useful for testing without spending credits.",
    )
    if limit_rows:
        row_limit = st.number_input(
            "Number of rows to process",
            min_value=1,
            max_value=len(df_raw),
            value=min(10, len(df_raw)),
            step=1,
        )
        n_to_process = int(row_limit)
        st.caption(f"Will process the first **{n_to_process}** of {len(df_raw):,} rows.")
    else:
        n_to_process = len(df_raw)
        st.info(f"All **{n_to_process:,}** rows will be processed.")

# =============================================================================
# STEP 5 — Start enrichment
# [START BUTTON LOCATION] — "Start enrichment" button is here
# =============================================================================

st.divider()
currently_processing = ss("processing", False)
enrichment_done      = ss("enrichment_done", False)

# Collect all reasons why enrichment cannot start
blocking: list = []
if uploaded is None:
    blocking.append("No file uploaded yet.")
if file_error:
    blocking.append(f"File could not be read: {file_error}")
if df_raw is not None and name_col is None:
    blocking.append("No company name column selected.")
if not api_key:
    blocking.append("No Lusha API key entered — please paste it in the sidebar.")
if df_raw is not None and n_to_process == 0:
    blocking.append("Zero rows selected for processing.")

if blocking and not currently_processing:
    for reason in blocking:
        st.warning(f"⚠️ {reason}")
elif not blocking and not currently_processing and not enrichment_done:
    st.info(f"Ready to enrich **{n_to_process:,}** rows. Click the button below to start.")

start_btn = st.button(
    "▶ Start enrichment",
    type="primary",
    use_container_width=True,
    disabled=(bool(blocking) or currently_processing),
    key="start_button",
)

if start_btn and not blocking and not currently_processing:
    df_work = df_raw.head(n_to_process).copy()
    ss_set(
        processing=True,
        stop_requested=False,
        process_index=0,
        results=[],
        debug_records=[],
        enrichment_done=False,
        df_enriched=None,
        _df_work=df_work,
        _name_col=name_col,
        _domain_col=domain_col,
        _n_to_process=n_to_process,
        _api_key=api_key,
        _delay=delay_sec,
    )
    st.rerun()

# =============================================================================
# PROCESSING LOOP — one row per Streamlit rerun
# The Stop button triggers a rerun that sets stop_requested=True,
# which causes build_and_finish() to be called on the next rerun.
# =============================================================================

if ss("processing", False):
    idx           = ss("process_index", 0)
    results       = ss("results", [])
    debug_records = ss("debug_records", [])
    df_work       = ss("_df_work")
    _name_col     = ss("_name_col")
    _domain_col   = ss("_domain_col")
    _n            = ss("_n_to_process", 0)
    _api_key      = ss("_api_key", "")
    _delay        = ss("_delay", 0.5)

    # Stop button
    if st.button("⏹ Stop after current row", key="stop_button"):
        ss_set(stop_requested=True)
        st.rerun()

    # Progress bar
    st.progress(idx / _n if _n else 1.0, text=f"Row {idx} of {_n}")

    # Live counters
    cnt_enriched = sum(1 for r in results if r.get("enrichment_status") in
                       ("enriched_by_domain", "enriched_by_company_name"))
    cnt_cached   = sum(1 for r in results if r.get("enrichment_status") == "cached")
    cnt_nomatch  = sum(1 for r in results if r.get("enrichment_status") in
                       ("no_match", "no_data_returned"))
    cnt_error    = sum(1 for r in results if r.get("enrichment_status") == "api_error")

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Enriched",   cnt_enriched)
    mc2.metric("Cached",     cnt_cached)
    mc3.metric("No match",   cnt_nomatch)
    mc4.metric("API errors", cnt_error)

    if ss("stop_requested", False) or idx >= _n:
        build_and_finish(results, debug_records, df_work)
    else:
        row          = df_work.iloc[idx]
        company_name = str(row.get(_name_col, "")).strip()
        raw_domain   = str(row.get(_domain_col, "")).strip() if _domain_col else ""
        st.caption(
            f"Processing row {idx + 1} of {_n}: **{company_name or '(empty)'}**"
        )
        fields, dbg = enrich_one_row(company_name, raw_domain, _api_key, _delay)
        results.append(fields)
        debug_records.append(dbg)
        ss_set(results=results, debug_records=debug_records, process_index=idx + 1)
        st.rerun()

# =============================================================================
# RESULTS — shown after processing completes (or stops early)
# =============================================================================

if ss("enrichment_done", False):
    df_enriched: pd.DataFrame = ss("df_enriched")
    debug_records_done: list  = ss("debug_records", [])
    processed = len(df_enriched)

    st.divider()
    if ss("stop_requested", False):
        st.warning(
            f"Enrichment stopped after **{processed}** rows. "
            "Partial results are available below."
        )
    else:
        st.success(f"✅ Enrichment complete — **{processed:,}** rows processed.")

    # ── Summary metrics ───────────────────────────────────────────────────────
    status_counts  = df_enriched["enrichment_status"].value_counts().to_dict()
    needs_review_n = int((df_enriched.get("needs_manual_review", "") == "TRUE").sum())
    all_metric_cols = list(status_counts.items())
    n_cols = min(len(all_metric_cols) + 1, 7)
    if all_metric_cols:
        rcols = st.columns(n_cols)
        for i, (s, c) in enumerate(all_metric_cols):
            rcols[i % n_cols].metric(_STATUS_LABELS.get(s, s), c)
        rcols[len(all_metric_cols) % n_cols].metric("⚑ Needs review", needs_review_n)

    # ── Results table (normal mode) ───────────────────────────────────────────
    # [NORMAL MODE — RESULTS TABLE]
    st.subheader("Results")
    orig_cols    = [c for c in df_enriched.columns if c not in LUSHA_FIELDS]
    summary_cols = orig_cols + [
        c for c in [
            "enrichment_status", "match_confidence",
            "needs_manual_review", "match_notes",
            "lusha_company_name", "lusha_domain", "lusha_industry",
            "lusha_country", "lusha_employee_range", "lusha_revenue",
            "lusha_error_message",
        ]
        if c in df_enriched.columns
    ]
    st.dataframe(df_enriched[summary_cols], use_container_width=True, height=400)

    with st.expander("Show all enriched columns"):
        st.dataframe(df_enriched, use_container_width=True)

    # =========================================================================
    # DOWNLOADS — normal mode: all 4 files always available
    # =========================================================================
    st.subheader("Download results")
    st.caption(
        "All four files are always available. "
        "The JSONL file contains sanitized company-level data suitable for ICP analysis."
    )

    raw_jsons = [d.get("raw_json") for d in debug_records_done]
    log_df    = make_log_df(debug_records_done)

    dl1, dl2 = st.columns(2)
    dl3, dl4 = st.columns(2)

    with dl1:
        st.download_button(
            "⬇ Enriched Excel (.xlsx)",
            data=df_to_excel_bytes(df_enriched),
            file_name="lusha_enriched.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            help="Original columns + all Lusha company fields + location footprint + review flags.",
        )
    with dl2:
        st.download_button(
            "⬇ Enriched CSV",
            data=df_to_csv_bytes(df_enriched),
            file_name="lusha_enriched.csv",
            mime="text/csv",
            use_container_width=True,
            help="Same data as the Excel file, in CSV format.",
        )
    with dl3:
        st.download_button(
            "⬇ Raw Lusha JSONL",
            data=records_to_jsonl_bytes(raw_jsons),
            file_name="lusha_raw_responses.jsonl",
            mime="application/jsonlines",
            use_container_width=True,
            help=(
                "One sanitized JSON object per company. "
                "Personal contact data is excluded. "
                "Use this file for ICP analysis or further processing."
            ),
        )
    with dl4:
        st.download_button(
            "⬇ Processing log CSV",
            data=df_to_csv_bytes(log_df),
            file_name="lusha_processing_log.csv",
            mime="text/csv",
            use_container_width=True,
            help=(
                "One row per processed company showing lookup method, "
                "HTTP status, enrichment outcome, and review flags."
            ),
        )

    # =========================================================================
    # DEBUG MODE — technical inspection tools, hidden when debug mode is OFF
    # =========================================================================

    if debug_mode and debug_records_done:

        st.divider()
        st.subheader("🐛 Debug — Request details")
        st.caption("One row per processed company. Shows exactly what was sent to Lusha and what came back.")
        debug_df = pd.DataFrame([
            {
                "row":               i + 1,
                "input_company":     d.get("input_company_name", ""),
                "input_domain":      d.get("input_domain", ""),
                "cleaned_domain":    d.get("cleaned_domain", ""),
                "lookup_method":     d.get("lookup_method", ""),
                "endpoint":          d.get("endpoint", ""),
                "params_sent":       str(d.get("params", {})),
                "http_status":       d.get("http_status", ""),
                "enrichment_status": d.get("enrichment_status", ""),
                "match_confidence":  d.get("match_confidence", ""),
                "error_message":     d.get("lusha_error_message", ""),
            }
            for i, d in enumerate(debug_records_done)
        ])
        st.dataframe(debug_df, use_container_width=True)

        # ── Raw JSON viewer ───────────────────────────────────────────────────
        st.subheader("🐛 Raw Lusha API responses")
        st.caption("Select a processed company to inspect the raw JSON response.")
        company_labels = [
            f"{i + 1}. {d.get('input_company_name') or '(empty)'} [{d.get('enrichment_status', '')}]"
            for i, d in enumerate(debug_records_done)
        ]
        sel_idx = st.selectbox(
            "Select a company:",
            options=range(len(company_labels)),
            format_func=lambda i: company_labels[i],
            key="raw_json_selector",
        )
        if sel_idx is not None:
            raw = debug_records_done[sel_idx].get("raw_json")
            if raw:
                st.json(raw)
            else:
                st.info("No JSON response available for this row (network error or unparseable response).")

        # ── Debug Excel with JSON preview + cache ZIP ─────────────────────────
        st.subheader("🐛 Additional debug downloads")
        debug_enriched = df_enriched.copy()
        debug_enriched["lusha_raw_json_preview"] = [
            (json.dumps(d.get("raw_json"), ensure_ascii=False)[:2000] if d.get("raw_json") else "")
            for d in debug_records_done
        ] + [""] * max(0, len(debug_enriched) - len(debug_records_done))

        dbg_dl1, dbg_dl2 = st.columns(2)
        with dbg_dl1:
            st.download_button(
                "⬇ Debug Excel (with JSON preview column)",
                data=df_to_excel_bytes(debug_enriched),
                file_name="lusha_enriched_debug.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                help="Enriched Excel plus a lusha_raw_json_preview column (first 2000 chars of JSON).",
            )
        with dbg_dl2:
            cc = get_cache_count()
            if cc > 0:
                st.download_button(
                    f"⬇ Cache as ZIP ({cc} files)",
                    data=cache_to_zip_bytes(),
                    file_name="lusha_cache.zip",
                    mime="application/zip",
                    use_container_width=True,
                    help="All cached JSON files bundled into a single ZIP archive.",
                )
            else:
                st.info("Cache is empty — nothing to download.")

        # ── Cache viewer ──────────────────────────────────────────────────────
        st.subheader("🐛 Cache viewer")
        st.caption(f"Cache folder: `{CACHE_DIR.resolve()}` — {get_cache_count()} file(s)")
        cache_files = list_cache_files()
        if cache_files:
            sel_cache = st.selectbox(
                "Select a cached file to inspect:",
                options=[f.stem for f in cache_files],
                key="cache_file_selector",
            )
            if sel_cache:
                cache_path = CACHE_DIR / f"{sel_cache}.json"
                try:
                    st.json(json.loads(cache_path.read_text(encoding="utf-8")))
                except Exception as exc:
                    st.error(f"Could not read cache file: {exc}")
        else:
            st.info("Cache is empty. Run an enrichment first.")

    # ── Restart ───────────────────────────────────────────────────────────────
    st.divider()
    if st.button("↺ Start a new enrichment", use_container_width=True, key="restart_btn"):
        reset_processing()
        st.rerun()
