"""
Claude + Jina AI Batch Company Enrichment
==========================================
Upload a file with company names (and URLs/websites),
fetch each company's webpage via Jina AI Reader, extract company data with
Claude AI (claude-haiku-4-5-20251001), and download the enriched results.

Architecture
------------
- One row is processed per Streamlit rerun so the Stop button works at any point.
- All mutable run state lives in st.session_state (keys prefixed with _).
- Normal mode: clean, non-technical UI.
- Debug mode: toggled via sidebar checkbox; shows request details, raw responses, cache tools.
"""

import io
import json
import re
import time
import unicodedata
import zipfile
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import quote

import anthropic
import pandas as pd
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

JINA_READER_URL = "https://r.jina.ai/"
JINA_SEARCH_URL = "https://s.jina.ai/"
CACHE_DIR       = Path("claude_json_cache")
MODEL_ID        = "claude-haiku-4-5-20251001"

# Pricing per million tokens (claude-haiku-4-5)
_COST_INPUT_PER_M  = 0.80
_COST_OUTPUT_PER_M = 4.00

_EXTRACT_PROMPT = (
    "Extract company information from this webpage. "
    "Return ONLY raw JSON (no markdown, no code fences) with exactly these fields: "
    "company_name, description, main_industry, sub_industry, employee_range, "
    "revenue_range, founded_year, company_type, country, city, linkedin_url, specialties. "
    "Use empty string for any field not found."
)

# All columns added by enrichment (matches original schema for download compatibility)
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
    # ── HQ location ───────────────────────────────────────────────────────────
    "lusha_country",
    "lusha_city",
    "lusha_state",
    "lusha_headquarters_country",
    "lusha_headquarters_city",
    # ── Full location footprint (kept for schema compatibility) ───────────────
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
    # ── Token usage (Claude-specific) ─────────────────────────────────────────
    "tokens_input",
    "tokens_output",
    "estimated_cost_usd",
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

_COMPANY_HINTS = ["company", "account", "organisation", "organization", "name", "naam", "bedrijf"]
_DOMAIN_HINTS  = ["domain", "website", "url", "web", "site", "domein"]

_STATUS_LABELS = {
    "enriched_by_url":   "Enriched via URL",
    "enriched_by_name":  "Enriched via name",
    "cached":            "From cache",
    "no_match":          "No match",
    "no_data_returned":  "No data returned",
    "api_error":         "API error",
    "jina_error":        "Page fetch error",
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


def normalize_url(raw: str) -> str:
    """Ensure a URL has an https:// scheme for Jina Reader."""
    if not raw or not isinstance(raw, str):
        return ""
    raw = raw.strip()
    if raw.lower() in {"nan", "none", ""}:
        return ""
    if " " in raw:
        return ""
    if raw.startswith(("http://", "https://")):
        return raw
    return f"https://{raw}"


def safe_filename(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = re.sub(r"[^\w\s\-.]", "", text)
    text = re.sub(r"\s+", "_", text).strip("_")
    return text[:120] or "unknown"


def str_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# Legal-entity suffix helpers used by flag_review()
_LEGAL_RE = re.compile(
    r"[\s,\-\.]*\b("
    r"b\.?v\.?|n\.?v\.?|s\.?a\.?s?\.?|s\.?p\.?a\.?|a\.?/\.?s\.?|a\.?s\.?"
    r"|g\.?m\.?b\.?h\.?|ag|ltd\.?|limited|inc\.?|corp\.?|llc|llp|plc"
    r"|oy|ab|s\.?r\.?l\.?|s\.?n\.?c\.?|kft|s\.?r\.?o\.?|o\.?[üu]\.?"
    r"|pte\.?|pty\.?|cv|vof|gg"
    r")\b\.?",
    re.IGNORECASE,
)

def _strip_legal(name: str) -> str:
    return _LEGAL_RE.sub(" ", name).strip(" .,/-")

def _legal_suffix(name: str) -> str:
    hits = _LEGAL_RE.findall(name)
    return re.sub(r"[^a-z0-9]", "", hits[-1].lower()) if hits else ""

def _domain_root(domain: str) -> str:
    d = re.sub(r"^www\.", "", domain.lower().strip())
    return d.split(".")[0] if d else ""


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


def calc_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * _COST_INPUT_PER_M + output_tokens * _COST_OUTPUT_PER_M) / 1_000_000


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
    CACHE_DIR.mkdir(exist_ok=True)
    (CACHE_DIR / f"{safe_filename(cache_key)}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def get_cache_count() -> int:
    return len(list(CACHE_DIR.glob("*.json"))) if CACHE_DIR.exists() else 0


def list_cache_files() -> list:
    return sorted(CACHE_DIR.glob("*.json")) if CACHE_DIR.exists() else []


# ─────────────────────────────────────────────────────────────────────────────
# Jina AI  ← webpage fetching
# ─────────────────────────────────────────────────────────────────────────────

def fetch_via_jina_reader(url: str) -> str:
    """Fetch a webpage as readable text via Jina AI Reader."""
    resp = requests.get(
        f"{JINA_READER_URL}{url}",
        headers={"Accept": "text/plain", "X-Return-Format": "text"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.text


def fetch_via_jina_search(query: str) -> str:
    """Search for a company by name via Jina AI Search."""
    resp = requests.get(
        f"{JINA_SEARCH_URL}{quote(query)}",
        headers={"Accept": "text/plain"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.text


# ─────────────────────────────────────────────────────────────────────────────
# Claude API  ← extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_with_claude(webpage_text: str, api_key: str) -> tuple:
    """
    Send webpage text to Claude and extract company fields.
    Returns (claude_fields_dict, input_tokens, output_tokens).
    """
    client    = anthropic.Anthropic(api_key=api_key)
    truncated = webpage_text[:50_000]  # avoid excessive token usage

    message = client.messages.create(
        model=MODEL_ID,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"{_EXTRACT_PROMPT}\n\n{truncated}",
        }],
    )

    input_tokens  = message.usage.input_tokens
    output_tokens = message.usage.output_tokens
    raw_text      = message.content[0].text.strip()

    # Strip possible markdown code fences Claude may include
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$", "", raw_text.strip())

    fields = json.loads(raw_text)  # raises ValueError / JSONDecodeError on bad output
    return fields, input_tokens, output_tokens


# ─────────────────────────────────────────────────────────────────────────────
# Field mapping  (Claude output → enrichment schema)
# ─────────────────────────────────────────────────────────────────────────────

def map_claude_fields(claude_data: dict, source_url: str) -> dict:
    """Map Claude's extracted JSON to the enrichment field schema."""
    domain  = clean_domain(source_url) if source_url else ""
    country = str(claude_data.get("country") or "").strip()
    city    = str(claude_data.get("city")    or "").strip()

    loc_summary = ("HQ: " + ", ".join(p for p in [city, country] if p)) if (city or country) else ""

    return {
        "lusha_company_name":    str(claude_data.get("company_name") or "").strip(),
        "lusha_domain":          domain,
        "lusha_industry":        str(claude_data.get("main_industry") or "").strip(),
        "lusha_sub_industry":    str(claude_data.get("sub_industry")  or "").strip(),
        "lusha_employee_count":  "",
        "lusha_employee_range":  str(claude_data.get("employee_range") or "").strip(),
        "lusha_revenue":         str(claude_data.get("revenue_range")  or "").strip(),
        "lusha_description":     str(claude_data.get("description")    or "").strip(),
        "lusha_linkedin_url":    str(claude_data.get("linkedin_url")   or "").strip(),
        "lusha_specialties":     str(claude_data.get("specialties")    or "").strip(),
        "lusha_technologies":    "",
        "lusha_founded_year":    str(claude_data.get("founded_year")   or "").strip(),
        # HQ location
        "lusha_country":              country,
        "lusha_city":                 city,
        "lusha_state":                "",
        "lusha_headquarters_country": country,
        "lusha_headquarters_city":    city,
        # Footprint — single-page extraction; populate what we know
        "lusha_location_count":                "",
        "lusha_location_countries":            country,
        "lusha_location_cities":               city,
        "lusha_location_continents":           "",
        "lusha_multi_country_presence":        "",
        "lusha_location_summary":              loc_summary,
        "lusha_international_footprint_score": "",
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
    Populate needs_manual_review and match_notes.
    Checks: bad status, low confidence, name similarity, legal-entity mismatch,
    domain-name mismatch (when not URL-based).
    """
    reasons: list[str] = []
    status     = fields.get("enrichment_status", "")
    confidence = fields.get("match_confidence", "")
    returned   = fields.get("lusha_company_name", "")
    ret_domain = fields.get("lusha_domain", "")
    inp        = (input_company_name or "").strip()

    if status in ("no_match", "api_error", "no_data_returned", "jina_error"):
        reasons.append(f"enrichment_status is {status}")

    if confidence == "low":
        reasons.append("match confidence is low")

    if returned and inp:
        inp_core = _strip_legal(inp).strip() or inp
        ret_core = _strip_legal(returned).strip() or returned
        sim = str_similarity(inp_core, ret_core)
        if sim < 0.70:
            reasons.append(
                f"Returned company name '{returned}' differs from input '{inp}' "
                f"(similarity {sim:.0%})"
            )

    if inp and returned:
        inp_sfx = _legal_suffix(inp)
        ret_sfx = _legal_suffix(returned)
        if inp_sfx and ret_sfx and inp_sfx != ret_sfx:
            reasons.append(
                f"Legal entity type mismatch: input has '{inp_sfx.upper()}' "
                f"but returned '{ret_sfx.upper()}' — may be a different legal entity or country"
            )

    if ret_domain and inp and status not in ("enriched_by_url",):
        root = _domain_root(ret_domain)
        if root:
            dom_sim = str_similarity(root, _strip_legal(inp).strip() or inp)
            if dom_sim < 0.35:
                reasons.append(
                    f"Returned domain '{ret_domain}' appears unrelated to input company name '{inp}'"
                )

    fields["needs_manual_review"] = "TRUE" if reasons else "FALSE"
    fields["match_notes"]         = "; ".join(reasons) if reasons else ""
    return fields


# ─────────────────────────────────────────────────────────────────────────────
# Per-row enrichment  ← enrichment logic lives here
# ─────────────────────────────────────────────────────────────────────────────

def enrich_one_row(
    company_name: str,
    raw_url: str,
    api_key: str,
    delay: float,
) -> tuple:
    """
    Returns (lusha_fields: dict, debug_record: dict).

    Strategy:
      1. If a URL/domain is present → fetch via Jina Reader (cache first).
         On fetch failure (404/403) → fall through to name-based search.
      2. Company-name search via Jina AI Search (cache first).
    """
    url          = normalize_url(raw_url) if raw_url else ""
    company_name = str(company_name).strip() if company_name else ""

    dbg: dict = {
        "input_company_name":  company_name,
        "input_url":           raw_url,
        "normalized_url":      url,
        "lookup_method":       None,
        "jina_url":            None,
        "http_status":         None,
        "claude_model":        MODEL_ID,
        "raw_json":            None,
        "tokens_input":        0,
        "tokens_output":       0,
        "estimated_cost":      0.0,
        "enrichment_status":   "",
        "match_confidence":    "",
        "lusha_error_message": "",
    }

    def _done(fields: dict) -> tuple:
        flag_review(fields, company_name)
        dbg["enrichment_status"]   = fields.get("enrichment_status", "")
        dbg["match_confidence"]    = fields.get("match_confidence",  "")
        dbg["lusha_error_message"] = fields.get("lusha_error_message", "")
        dbg["needs_manual_review"] = fields.get("needs_manual_review", "")
        dbg["match_notes"]         = fields.get("match_notes", "")
        return fields, dbg

    def _apply_tokens(fields: dict, in_tok: int, out_tok: int) -> dict:
        cost = calc_cost(in_tok, out_tok)
        fields["tokens_input"]       = str(in_tok)
        fields["tokens_output"]      = str(out_tok)
        fields["estimated_cost_usd"] = f"{cost:.6f}"
        dbg["tokens_input"]   = in_tok
        dbg["tokens_output"]  = out_tok
        dbg["estimated_cost"] = cost
        return fields

    # ── 1. URL path ───────────────────────────────────────────────────────────
    if url:
        dbg["lookup_method"] = "url"
        dbg["jina_url"]      = f"{JINA_READER_URL}{url}"
        cache_key = f"url_{url}"
        cached = load_cache(cache_key)

        if cached is not None:
            fields = map_claude_fields(cached.get("claude_data", {}), url)
            if has_data(fields):
                dbg["http_status"] = "cached"
                dbg["raw_json"]    = cached
                in_tok  = int(cached.get("tokens_input", 0) or 0)
                out_tok = int(cached.get("tokens_output", 0) or 0)
                fields = _apply_tokens(fields, in_tok, out_tok)
                fields["enrichment_status"] = "cached"
                fields["match_confidence"]  = "high"
                return _done(fields)
            (CACHE_DIR / f"{safe_filename(cache_key)}.json").unlink(missing_ok=True)

        try:
            time.sleep(delay)
            webpage_text = fetch_via_jina_reader(url)
            dbg["http_status"] = 200
            claude_data, in_tok, out_tok = extract_with_claude(webpage_text, api_key)
            cache_payload = {"claude_data": claude_data, "tokens_input": in_tok, "tokens_output": out_tok}
            dbg["raw_json"] = cache_payload
            save_cache(cache_key, cache_payload)
            fields = map_claude_fields(claude_data, url)
            fields = _apply_tokens(fields, in_tok, out_tok)
            if not has_data(fields):
                return _done(_empty("no_data_returned", "no_match",
                                    "Claude returned no usable company fields from page"))
            fields["enrichment_status"] = "enriched_by_url"
            fields["match_confidence"]  = "high"
            return _done(fields)

        except requests.HTTPError as e:
            dbg["http_status"] = e.response.status_code if e.response else "?"
            code = e.response.status_code if e.response is not None else 0
            if code in (403, 404, 429):
                pass  # fall through to name lookup
            else:
                return _done(_empty("jina_error", "no_match",
                                    f"Jina HTTP {code}: {e.response.text[:200] if e.response else str(e)}"))
        except (json.JSONDecodeError, ValueError) as e:
            return _done(_empty("api_error", "no_match", f"Claude response parse error: {e}"))
        except anthropic.APIError as e:
            return _done(_empty("api_error", "no_match", f"Claude API error: {e}"))
        except Exception as e:
            dbg["http_status"] = "error"
            return _done(_empty("api_error", "no_match", str(e)))

    # ── 2. Name path ──────────────────────────────────────────────────────────
    if not company_name:
        dbg["lookup_method"] = "none"
        return _done(_empty("no_match", "no_match", "No company name or URL provided"))

    dbg["lookup_method"] = "company_name"
    dbg["jina_url"]      = f"{JINA_SEARCH_URL}{quote(company_name)}"
    cache_key = f"name_{company_name}"
    cached = load_cache(cache_key)

    if cached is not None:
        fields = map_claude_fields(cached.get("claude_data", {}), "")
        if has_data(fields):
            dbg["http_status"] = "cached"
            dbg["raw_json"]    = cached
            in_tok  = int(cached.get("tokens_input", 0) or 0)
            out_tok = int(cached.get("tokens_output", 0) or 0)
            fields = _apply_tokens(fields, in_tok, out_tok)
            conf = "medium" if str_similarity(company_name,
                                              fields.get("lusha_company_name", "")) >= 0.6 else "low"
            fields["enrichment_status"] = "cached"
            fields["match_confidence"]  = conf
            return _done(fields)
        (CACHE_DIR / f"{safe_filename(cache_key)}.json").unlink(missing_ok=True)

    try:
        time.sleep(delay)
        webpage_text = fetch_via_jina_search(company_name)
        dbg["http_status"] = 200
        claude_data, in_tok, out_tok = extract_with_claude(webpage_text, api_key)
        cache_payload = {"claude_data": claude_data, "tokens_input": in_tok, "tokens_output": out_tok}
        dbg["raw_json"] = cache_payload
        save_cache(cache_key, cache_payload)
        fields = map_claude_fields(claude_data, "")
        fields = _apply_tokens(fields, in_tok, out_tok)
        if not has_data(fields):
            return _done(_empty("no_data_returned", "no_match",
                                "Claude returned no usable company fields from name search"))
        conf = "medium" if str_similarity(company_name,
                                          fields.get("lusha_company_name", "")) >= 0.6 else "low"
        fields["enrichment_status"] = "enriched_by_name"
        fields["match_confidence"]  = conf
        return _done(fields)

    except requests.HTTPError as e:
        dbg["http_status"] = e.response.status_code if e.response else "?"
        return _done(_empty("jina_error", "no_match",
                            f"Jina search HTTP {e.response.status_code if e.response else '?'}"))
    except (json.JSONDecodeError, ValueError) as e:
        return _done(_empty("api_error", "no_match", f"Claude response parse error: {e}"))
    except anthropic.APIError as e:
        return _done(_empty("api_error", "no_match", f"Claude API error: {e}"))
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
    rows = []
    for d in debug_records:
        rows.append({
            "input_company_name":  d.get("input_company_name", ""),
            "input_url":           d.get("input_url", ""),
            "lookup_method":       d.get("lookup_method", ""),
            "http_status":         d.get("http_status", ""),
            "enrichment_status":   d.get("enrichment_status", ""),
            "match_confidence":    d.get("match_confidence", ""),
            "needs_manual_review": d.get("needs_manual_review", ""),
            "match_notes":         d.get("match_notes", ""),
            "tokens_input":        d.get("tokens_input", ""),
            "tokens_output":       d.get("tokens_output", ""),
            "estimated_cost_usd":  d.get("estimated_cost", ""),
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
        total_tokens_input=0, total_tokens_output=0, total_cost_usd=0.0,
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
    page_title="Claude Company Enrichment",
    page_icon="🏢",
    layout="wide",
)
st.title("🏢 Claude Company Enrichment")
st.caption(
    "Upload a file with company names and URLs, extract firmographic data with Claude AI "
    "via Jina AI, and download the enriched results."
)

# =============================================================================
# SIDEBAR  — always visible: API key + debug toggle
# =============================================================================

with st.sidebar:
    st.header("Settings")

    # Anthropic API key: read from secrets first, sidebar input as fallback
    _secret_key = ""
    try:
        _secret_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        pass

    if _secret_key:
        api_key = _secret_key
        st.success("✓ API key loaded from secrets")
    else:
        api_key = st.text_input(
            "Anthropic API key",
            type="password",
            key="api_key_input",
            placeholder="sk-ant-...",
            help=(
                "Your Anthropic API key. "
                "Can also be set via st.secrets['ANTHROPIC_API_KEY']."
            ),
        )

    st.divider()

    debug_mode = st.checkbox(
        "Enable debug mode",
        value=False,
        help=(
            "Shows request details, raw Claude responses, "
            "cache management tools, and additional downloads."
        ),
    )

    if debug_mode:
        st.divider()
        st.subheader("Debug settings")
        delay_sec = st.slider(
            "Delay between API calls (sec)",
            min_value=0.0, max_value=3.0, value=1.0, step=0.1,
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
        delay_sec = 1.0

# =============================================================================
# STEP 1 — Upload file
# =============================================================================

st.subheader("Step 1 · Upload your file")
uploaded = st.file_uploader(
    "Drag and drop here, or click to browse  (.xlsx · .xls · .csv)",
    type=["xlsx", "xls", "csv"],
)

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
        "💡 Each row fetches a webpage via Jina AI and calls Claude AI (billed per token). "
        "Start with a small test batch first."
    )

# =============================================================================
# STEP 2 — Preview input
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
            "Website / URL column (optional)",
            options=dom_opts,
            index=def_dom,
            help=(
                "If this column contains a URL or domain, Jina AI Reader fetches the page. "
                "Falls back to Jina AI Search on the company name if fetch fails."
            ),
        )
    domain_col = dom_choice if dom_choice != _NO_DOMAIN else None

    note_parts = []
    if auto_name_col:
        note_parts.append(f"company name → **{auto_name_col}**")
    if auto_domain_col:
        note_parts.append(f"URL/domain → **{auto_domain_col}**")
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
        help="Process only a small batch first — useful for testing before spending credits.",
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
# =============================================================================

st.divider()
currently_processing = ss("processing", False)
enrichment_done      = ss("enrichment_done", False)

blocking: list = []
if uploaded is None:
    blocking.append("No file uploaded yet.")
if file_error:
    blocking.append(f"File could not be read: {file_error}")
if df_raw is not None and name_col is None:
    blocking.append("No company name column selected.")
if not api_key:
    blocking.append("No Anthropic API key — paste it in the sidebar or set st.secrets['ANTHROPIC_API_KEY'].")
if df_raw is not None and n_to_process == 0:
    blocking.append("Zero rows selected for processing.")

if blocking and not currently_processing:
    for reason in blocking:
        st.warning(f"⚠️ {reason}")
elif not blocking and not currently_processing and not enrichment_done:
    est_cost = n_to_process * 0.001  # rough estimate: ~$0.001/row at haiku rates
    st.info(
        f"Ready to enrich **{n_to_process:,}** rows. "
        f"Rough estimated cost: ~${est_cost:.2f} (varies by page length)."
    )

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
        total_tokens_input=0,
        total_tokens_output=0,
        total_cost_usd=0.0,
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
    _delay        = ss("_delay", 1.0)
    total_in      = ss("total_tokens_input", 0)
    total_out     = ss("total_tokens_output", 0)
    total_cost    = ss("total_cost_usd", 0.0)

    if st.button("⏹ Stop after current row", key="stop_button"):
        ss_set(stop_requested=True)
        st.rerun()

    st.progress(idx / _n if _n else 1.0, text=f"Row {idx} of {_n}")

    cnt_enriched = sum(1 for r in results if r.get("enrichment_status") in
                       ("enriched_by_url", "enriched_by_name"))
    cnt_cached   = sum(1 for r in results if r.get("enrichment_status") == "cached")
    cnt_nomatch  = sum(1 for r in results if r.get("enrichment_status") in
                       ("no_match", "no_data_returned"))
    cnt_error    = sum(1 for r in results if r.get("enrichment_status") in
                       ("api_error", "jina_error"))

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Enriched",   cnt_enriched)
    mc2.metric("Cached",     cnt_cached)
    mc3.metric("No match",   cnt_nomatch)
    mc4.metric("Errors",     cnt_error)
    mc5.metric("Est. cost",  f"${total_cost:.4f}")

    if ss("stop_requested", False) or idx >= _n:
        build_and_finish(results, debug_records, df_work)
    else:
        row          = df_work.iloc[idx]
        company_name = str(row.get(_name_col, "")).strip()
        raw_url      = str(row.get(_domain_col, "")).strip() if _domain_col else ""
        st.caption(
            f"Processing row {idx + 1} of {_n}: **{company_name or '(empty)'}**"
            + (f"  ·  {raw_url}" if raw_url else "")
        )
        fields, dbg = enrich_one_row(company_name, raw_url, _api_key, _delay)
        results.append(fields)
        debug_records.append(dbg)

        # Accumulate token usage
        try:
            total_in   += int(fields.get("tokens_input",  0) or 0)
            total_out  += int(fields.get("tokens_output", 0) or 0)
            total_cost += float(fields.get("estimated_cost_usd", 0) or 0)
        except (ValueError, TypeError):
            pass

        ss_set(
            results=results, debug_records=debug_records, process_index=idx + 1,
            total_tokens_input=total_in, total_tokens_output=total_out, total_cost_usd=total_cost,
        )
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

    # ── Token usage & cost summary ────────────────────────────────────────────
    total_in_done   = ss("total_tokens_input", 0)
    total_out_done  = ss("total_tokens_output", 0)
    total_cost_done = ss("total_cost_usd", 0.0)

    with st.expander("💰 Token usage & cost", expanded=True):
        tc1, tc2, tc3 = st.columns(3)
        tc1.metric("Input tokens",   f"{total_in_done:,}")
        tc2.metric("Output tokens",  f"{total_out_done:,}")
        tc3.metric("Estimated cost", f"${total_cost_done:.4f}")
        st.caption(
            f"Model: `{MODEL_ID}` · "
            f"Pricing: ${_COST_INPUT_PER_M}/M input, ${_COST_OUTPUT_PER_M}/M output tokens. "
            "Actual charges may vary — check your Anthropic dashboard."
        )

    # ── Results table ─────────────────────────────────────────────────────────
    st.subheader("Results")
    orig_cols    = [c for c in df_enriched.columns if c not in LUSHA_FIELDS]
    summary_cols = orig_cols + [
        c for c in [
            "enrichment_status", "match_confidence",
            "needs_manual_review", "match_notes",
            "lusha_company_name", "lusha_domain", "lusha_industry",
            "lusha_country", "lusha_employee_range", "lusha_revenue",
            "tokens_input", "tokens_output", "estimated_cost_usd",
            "lusha_error_message",
        ]
        if c in df_enriched.columns
    ]
    st.dataframe(df_enriched[summary_cols], use_container_width=True, height=400)

    with st.expander("Show all enriched columns"):
        st.dataframe(df_enriched, use_container_width=True)

    # =========================================================================
    # DOWNLOADS
    # =========================================================================
    st.subheader("Download results")
    st.caption(
        "All four files are always available. "
        "The JSONL file contains Claude's extracted data suitable for further analysis."
    )

    raw_jsons = [d.get("raw_json") for d in debug_records_done]
    log_df    = make_log_df(debug_records_done)

    dl1, dl2 = st.columns(2)
    dl3, dl4 = st.columns(2)

    with dl1:
        st.download_button(
            "⬇ Enriched Excel (.xlsx)",
            data=df_to_excel_bytes(df_enriched),
            file_name="claude_enriched.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            help="Original columns + all extracted company fields + review flags + token usage.",
        )
    with dl2:
        st.download_button(
            "⬇ Enriched CSV",
            data=df_to_csv_bytes(df_enriched),
            file_name="claude_enriched.csv",
            mime="text/csv",
            use_container_width=True,
            help="Same data as the Excel file, in CSV format.",
        )
    with dl3:
        st.download_button(
            "⬇ Raw Claude JSONL",
            data=records_to_jsonl_bytes(raw_jsons),
            file_name="claude_raw_responses.jsonl",
            mime="application/jsonlines",
            use_container_width=True,
            help="One JSON object per company with Claude's extracted fields and token usage.",
        )
    with dl4:
        st.download_button(
            "⬇ Processing log CSV",
            data=df_to_csv_bytes(log_df),
            file_name="claude_processing_log.csv",
            mime="text/csv",
            use_container_width=True,
            help=(
                "One row per processed company showing lookup method, "
                "HTTP status, enrichment outcome, token usage, and review flags."
            ),
        )

    # =========================================================================
    # DEBUG MODE — technical inspection tools
    # =========================================================================

    if debug_mode and debug_records_done:

        st.divider()
        st.subheader("🐛 Debug — Request details")
        st.caption("One row per processed company. Shows exactly what was sent and what came back.")
        debug_df = pd.DataFrame([
            {
                "row":               i + 1,
                "input_company":     d.get("input_company_name", ""),
                "input_url":         d.get("input_url", ""),
                "normalized_url":    d.get("normalized_url", ""),
                "lookup_method":     d.get("lookup_method", ""),
                "jina_url":          d.get("jina_url", ""),
                "http_status":       d.get("http_status", ""),
                "enrichment_status": d.get("enrichment_status", ""),
                "match_confidence":  d.get("match_confidence", ""),
                "tokens_input":      d.get("tokens_input", ""),
                "tokens_output":     d.get("tokens_output", ""),
                "estimated_cost":    d.get("estimated_cost", ""),
                "error_message":     d.get("lusha_error_message", ""),
            }
            for i, d in enumerate(debug_records_done)
        ])
        st.dataframe(debug_df, use_container_width=True)

        # ── Raw JSON viewer ───────────────────────────────────────────────────
        st.subheader("🐛 Raw Claude API responses")
        st.caption("Select a processed company to inspect the raw Claude response.")
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

        # ── Debug Excel + cache ZIP ───────────────────────────────────────────
        st.subheader("🐛 Additional debug downloads")
        debug_enriched = df_enriched.copy()
        debug_enriched["claude_raw_json_preview"] = [
            (json.dumps(d.get("raw_json"), ensure_ascii=False)[:2000] if d.get("raw_json") else "")
            for d in debug_records_done
        ] + [""] * max(0, len(debug_enriched) - len(debug_records_done))

        dbg_dl1, dbg_dl2 = st.columns(2)
        with dbg_dl1:
            st.download_button(
                "⬇ Debug Excel (with JSON preview column)",
                data=df_to_excel_bytes(debug_enriched),
                file_name="claude_enriched_debug.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                help="Enriched Excel plus a claude_raw_json_preview column (first 2000 chars of JSON).",
            )
        with dbg_dl2:
            cc = get_cache_count()
            if cc > 0:
                st.download_button(
                    f"⬇ Cache as ZIP ({cc} files)",
                    data=cache_to_zip_bytes(),
                    file_name="claude_cache.zip",
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
