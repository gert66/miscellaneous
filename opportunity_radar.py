"""
mYngle · Opportunity Radar
==========================
Researches buying-window signals and contact routes for company lists.
Accepts Lead Prioritizer exports (with Opportunity Input sheet) or simple
company lists.  Uses Serper Google Search + Claude Haiku for signal extraction.

Entry point:  streamlit run opportunity_radar.py
"""

import base64
import hashlib
import io
import json
import pathlib
import re
import time
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

try:
    import anthropic as _anthropic_mod
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="mYngle · Opportunity Radar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# HEADER  (logo + title — mirrors lead prioritizer layout exactly)
# =============================================================================

_logo_path = pathlib.Path(__file__).parent / "mingle_local_final_fixed.png"
_logo_src  = (
    f"data:image/png;base64,{base64.b64encode(_logo_path.read_bytes()).decode()}"
    if _logo_path.exists() else ""
)
_img_tag = (
    f'<img src="{_logo_src}" class="brand-logo" alt="mYngle" />'
    if _logo_src else ""
)

st.markdown(
    f"""
    <style>
    .block-container {{
        max-width: 880px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }}

    div[data-testid="stMarkdownContainer"]:has(.brand-header) {{
        overflow: visible !important;
        margin-bottom: 1.0rem;
    }}

    .brand-header {{
        display: grid;
        grid-template-columns: 43% 57%;
        align-items: center;
        min-height: 140px;
        padding-top: 10px;
        padding-bottom: 6px;
        overflow: visible !important;
    }}

    .brand-title-block {{
        display: flex;
        align-items: center;
        justify-content: flex-start;
        overflow: visible !important;
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
        display: flex;
        justify-content: flex-end;
        align-items: center;
        padding: 0;
        line-height: 0;
        overflow: visible !important;
    }}

    .brand-logo {{
        width: 430px;
        max-width: 100%;
        height: auto;
        display: block;
        object-fit: contain;
        object-position: center center;
        overflow: visible !important;
    }}
    </style>

    <div class="brand-header">
      <div class="brand-title-block">
        <span class="brand-title">Opportunity Radar</span>
      </div>
      <div class="brand-logo-block">
        {_img_tag}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# CONSTANTS
# =============================================================================

CLAUDE_MODEL    = "claude-haiku-4-5-20251001"
SERPER_URL      = "https://google.serper.dev/search"
RADAR_CACHE_DIR = pathlib.Path("radar_cache")

# 5 query groups — one Serper call each
QUERY_GROUPS = [
    (
        "Annual Report / Financial",
        '"{name}" annual report fiscal year results investor',
    ),
    (
        "Hiring / Careers",
        '"{name}" hiring careers jobs vacancies 2024 2025',
    ),
    (
        "L&D / HR / Talent",
        '"{name}" "learning and development" OR "talent development" OR training OR academy HR',
    ),
    (
        "Growth / Expansion",
        '"{name}" expansion "new office" OR international OR "new market" OR global',
    ),
    (
        "M&A / Funding",
        '"{name}" acquisition OR merger OR funding OR investment OR "private equity"',
    ),
]

ALLOWED_TRIGGER_TYPES = [
    "Hiring wave",
    "International expansion",
    "New office",
    "M&A / integration",
    "Funding / growth",
    "L&D hiring",
    "Sales / customer success expansion",
    "Employer branding",
    "Annual planning signal",
    "None",
    "Other",
]

ALLOWED_ROUTES = [
    "L&D / Talent Development",
    "HR / People",
    "International HR",
    "Sales Enablement",
    "Customer Success",
    "Operations",
    "Procurement",
    "Unknown",
]

ALLOWED_RECOMMENDATIONS = [
    "Call now",
    "Call this month",
    "Call before budget cycle",
    "Monitor",
    "Manual research needed",
    "Low priority",
    "Internal / exclude",
]

# JSON schema Claude must return
_CLAUDE_SCHEMA = """{
  "trigger_found": true/false,
  "trigger_type": "<one of the allowed trigger types>",
  "trigger_date": "<date or empty string>",
  "trigger_score": <0-3>,
  "trigger_evidence": "<1-3 sentence summary of evidence>",
  "annual_report_found": true/false,
  "annual_report_url": "<url or empty>",
  "annual_report_date": "<date or empty>",
  "fiscal_year_pattern": "<Calendar year | Non-calendar fiscal year | Unknown>",
  "likely_buying_window": "<e.g. Q1 2026 or empty>",
  "buying_window_score": <0-3>,
  "buying_window_confidence": "<High | Medium | Low | Unknown>",
  "buying_window_reason": "<brief explanation>",
  "hiring_signal_score": <0-3>,
  "international_hiring_signal": <0-3>,
  "lnd_hr_hiring_signal": <0-3>,
  "sales_cs_hiring_signal": <0-3>,
  "onboarding_pressure_signal": <0-3>,
  "preferred_buyer_route": "<one of the allowed routes>",
  "backup_buyer_route": "<one of the allowed routes or empty>",
  "suggested_title_searches": "<comma-separated titles to search for>",
  "suggested_opener": "<1-2 sentence caller opener referencing a specific signal>",
  "why_now": "<1-2 sentence reason this company is worth calling now>",
  "evidence_sources": "<comma-separated URLs that support the conclusions>",
  "evidence_quality": "<Strong | Medium | Weak | Insufficient>",
  "confidence_level": "<High | Medium | Low | Unknown>",
  "manual_review_needed": true/false,
  "manual_review_reason": "<reason or empty>"
}"""

_EMPTY_CLAUDE_RESULT: dict = {
    "trigger_found": False,
    "trigger_type": "None",
    "trigger_date": "",
    "trigger_score": 0,
    "trigger_evidence": "",
    "annual_report_found": False,
    "annual_report_url": "",
    "annual_report_date": "",
    "fiscal_year_pattern": "Unknown",
    "likely_buying_window": "",
    "buying_window_score": 0,
    "buying_window_confidence": "Unknown",
    "buying_window_reason": "",
    "hiring_signal_score": 0,
    "international_hiring_signal": 0,
    "lnd_hr_hiring_signal": 0,
    "sales_cs_hiring_signal": 0,
    "onboarding_pressure_signal": 0,
    "preferred_buyer_route": "Unknown",
    "backup_buyer_route": "",
    "suggested_title_searches": "",
    "suggested_opener": "",
    "why_now": "",
    "evidence_sources": "",
    "evidence_quality": "Insufficient",
    "confidence_level": "Unknown",
    "manual_review_needed": True,
    "manual_review_reason": "No search results available",
}

# =============================================================================
# SESSION STATE HELPERS
# =============================================================================

def ss(key: str, default=None):
    return st.session_state.get(key, default)


def ss_set(**kwargs):
    for k, v in kwargs.items():
        st.session_state[k] = v


def reset():
    ss_set(
        _or_file_key="__none__",
        _or_df_raw=None,
        _or_file_name=None,
        _or_file_error=None,
        _or_name_col=None,
        _or_domain_col=None,
        _or_country_col=None,
        _or_score_col=None,
        _or_tier_col=None,
        _or_icp_col=None,
        _or_n_companies=0,
        _or_input_type=None,   # "enriched_export" | "simple_company_list"
        _or_processing=False,
        _or_done=False,
        _or_process_index=0,
        _or_company_list=None,
        _or_results=None,
        _or_raw_sources=None,
        _or_excel_bytes=None,
        _or_stop=False,
    )


# =============================================================================
# COLUMN DETECTION
# =============================================================================

_NAME_CANDIDATES = [
    "company_name", "company name", "canonical_company_name",
    "company", "name", "organisation", "organization",
]
_DOMAIN_CANDIDATES = [
    "canonical_company_url", "canonical_company_domain",
    "company_domain", "company_url", "domain",
    "company website", "company domain",
    "website", "url", "company url", "homepage",
]
_COUNTRY_CANDIDATES = [
    "country", "company_country", "company country",
    "hq country", "hq_country", "company_hq_country",
]
_SCORE_CANDIDATES = [
    "final_commercial_fit_score", "commercial_fit_score", "score",
]
_TIER_CANDIDATES = [
    "commercial_tier", "tier",
]
_ICP_CANDIDATES = [
    "icp_evidence", "icp evidence", "why_relevant", "why_is_this_company_relevant",
    "buying_signals", "icp_buying_signals", "purchasing_signals", "icp_signals",
]

# Columns whose presence signals an enriched export
_ENRICHED_SIGNAL_COLS = {
    "final_commercial_fit_score", "commercial_fit_score", "commercial_tier",
    "icp_evidence", "icp_buying_signals", "icp_lead_score", "icp_why_relevant",
    "icp_likely_training_interest", "icp_potential_buyer_function",
    "sig_intl_footprint_score", "sig_rapid_growth_score", "enrichment_status",
    "top_positive_signals", "top_score_drivers",
}


def _detect_col(df: pd.DataFrame, candidates: list) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _normalize_domain(raw: str) -> str:
    """Strip protocol, www, trailing path from a URL to get a bare domain."""
    if not raw:
        return raw
    raw = raw.strip()
    raw = re.sub(r"^https?://", "", raw, flags=re.IGNORECASE)
    raw = raw.split("/")[0].split("?")[0]
    if raw.lower().startswith("www."):
        raw = raw[4:]
    return raw.strip()


_INTERNAL_NAMES   = {"myngle"}
_INTERNAL_DOMAINS = {"myngle.com"}


def _is_internal(name: str, domain: str) -> bool:
    return (
        name.lower().strip() in _INTERNAL_NAMES
        or any(d in domain.lower() for d in _INTERNAL_DOMAINS)
    )


def _count_companies(df: pd.DataFrame, name_col: str | None) -> int:
    if name_col and name_col in df.columns:
        return int(
            df[name_col]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .nunique()
        )
    return len(df)


# =============================================================================
# FILE LOADING WITH SHEET PRIORITY
# =============================================================================

def _load_df_from_upload(uploaded_file) -> tuple:
    """
    Load the best DataFrame from an uploaded file.

    Sheet priority for Excel files:
      1. 'Opportunity Input' — preferred (Lead Prioritizer export)
      2. First sheet (usually 'Lead Scores')
      3. 'Enriched' — fallback if first sheet has no usable columns
      ('Company Profiles' is never read — it is formatted for humans)

    Returns (df, sheet_used) where sheet_used is a string label.
    CSV files always return (df, "csv").
    """
    fname = uploaded_file.name
    if fname.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file), "csv"

    xf = pd.ExcelFile(uploaded_file)
    sheet_names = xf.sheet_names

    # Priority 1 — dedicated Opportunity Input sheet
    if "Opportunity Input" in sheet_names:
        return xf.parse("Opportunity Input"), "Opportunity Input"

    # Priority 2 — first sheet (skip Company Profiles if it is somehow first)
    first = next(
        (s for s in sheet_names if s != "Company Profiles"),
        sheet_names[0] if sheet_names else None,
    )
    if first:
        df_first = xf.parse(first)
        # Check if it has at least a name or domain column
        lower_cols = {c.lower() for c in df_first.columns}
        has_identity = any(
            c in lower_cols
            for c in ("company_name", "company name", "company", "name",
                      "domain", "company domain", "company website", "website")
        )
        if has_identity:
            return df_first, first

    # Priority 3 — 'Enriched' hidden sheet
    if "Enriched" in sheet_names:
        return xf.parse("Enriched"), "Enriched"

    # Last resort — first sheet regardless
    return xf.parse(sheet_names[0]), sheet_names[0]


def _detect_input_type(df: pd.DataFrame) -> str:
    """
    Classify the uploaded file as 'enriched_export' or 'simple_company_list'.

    An enriched export contains at least one of the enrichment signal columns.
    """
    col_set = {c.lower() for c in df.columns}
    for sig_col in _ENRICHED_SIGNAL_COLS:
        if sig_col.lower() in col_set:
            return "enriched_export"
    return "simple_company_list"


# =============================================================================
# CACHE
# =============================================================================

def _cache_key(name: str, domain: str, input_type: str = "") -> str:
    # input_type is part of the key so enriched/simple runs never share a cache file
    raw = f"{name.lower().strip()}|{domain.lower().strip()}|{input_type}"
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_load(name: str, domain: str, input_type: str = "") -> dict | None:
    RADAR_CACHE_DIR.mkdir(exist_ok=True)
    p = RADAR_CACHE_DIR / f"{_cache_key(name, domain, input_type)}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _cache_save(name: str, domain: str, input_type: str, data: dict) -> None:
    RADAR_CACHE_DIR.mkdir(exist_ok=True)
    p = RADAR_CACHE_DIR / f"{_cache_key(name, domain, input_type)}.json"
    try:
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


# =============================================================================
# SERPER
# =============================================================================

def _serper_search(query: str, api_key: str, n: int = 5) -> list:
    try:
        resp = requests.post(
            SERPER_URL,
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": n},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("organic", [])[:n]:
            results.append({
                "title":   item.get("title", ""),
                "url":     item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "date":    item.get("date", ""),
            })
        return results
    except Exception:
        return []


def _run_searches(name: str, domain: str, api_key: str) -> dict:
    """Run all query groups; return {group_label: [results]}."""
    grouped: dict = {}
    for group_label, template in QUERY_GROUPS:
        query = template.format(name=name)
        results = _serper_search(query, api_key, n=5)
        grouped[group_label] = results
        time.sleep(0.3)  # stay within Serper rate limits
    return grouped


def _format_results_for_prompt(grouped: dict) -> str:
    parts = []
    for group_label, results in grouped.items():
        parts.append(f"=== {group_label.upper()} ===")
        if not results:
            parts.append("(no results)\n")
            continue
        for i, r in enumerate(results, 1):
            line = f"[{i}] {r['title']}"
            if r.get("date"):
                line += f"  ({r['date']})"
            line += f"\n    {r['url']}"
            if r.get("snippet"):
                line += f"\n    {r['snippet']}"
            parts.append(line)
        parts.append("")
    return "\n".join(parts)


def _collect_raw_sources(company_name: str, grouped: dict, input_type: str = "") -> list:
    rows = []
    for group_label, results in grouped.items():
        for r in results:
            rows.append({
                "company_name": company_name,
                "input_type":   input_type,
                "query_group":  group_label,
                "title":        r.get("title", ""),
                "url":          r.get("url", ""),
                "snippet":      r.get("snippet", ""),
                "date":         r.get("date", ""),
                "source_type":  "Organic search",
            })
    return rows


# =============================================================================
# CLAUDE EXTRACTION
# =============================================================================

_PROMPT_TEMPLATE = """\
You are a B2B sales intelligence analyst. Analyze the following web search results \
for {name} and extract structured buying-window signals.

TODAY'S DATE: {today}

COMPANY PROFILE:
- Name: {name}
- Domain: {domain}
- Country: {country}
- Commercial Fit Score: {fit_score}
- Commercial Tier: {tier}
- ICP Evidence: {icp_evidence}

WEB SEARCH RESULTS:
{search_text}

INSTRUCTIONS:
- trigger_score: 0=no signal, 1=weak, 2=moderate, 3=strong
- buying_window_score: 0=unclear/none, 1=possible, 2=likely, 3=imminent
- All *_signal scores: 0=none, 1=weak, 2=moderate, 3=strong
- preferred_buyer_route must be one of: {routes}
- trigger_type must be one of: {trigger_types}
- suggested_opener: a specific 1-2 sentence cold-call opener referencing an actual signal found
- why_now: 1-2 sentences on why this company is worth calling right now
- IMPORTANT: likely_buying_window must be a FUTURE date relative to today ({today}).
  If the best evidence points to a window that has already passed, project forward to the
  next likely planning cycle (e.g. next fiscal Q1, next budget season) and set
  buying_window_confidence to "Low". Do not leave buying_window empty if you can estimate
  a future window from fiscal year or annual report patterns.
- If evidence is absent for a field, use false / empty string / 0 / "Unknown" as appropriate
- Return ONLY the JSON object below — no markdown, no explanation

{schema}"""


def _extract_json_from_text(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {}


def _call_claude(
    name: str,
    domain: str,
    country: str,
    fit_score: str,
    tier: str,
    icp_evidence: str,
    grouped_results: dict,
    client,
) -> dict:
    search_text = _format_results_for_prompt(grouped_results)
    today_str = datetime.now().strftime("%Y-%m-%d")
    prompt = _PROMPT_TEMPLATE.format(
        today=today_str,
        name=name,
        domain=domain or "(unknown)",
        country=country or "(unknown)",
        fit_score=fit_score or "(not available)",
        tier=tier or "(not available)",
        icp_evidence=(icp_evidence or "(not available)")[:600],
        search_text=search_text,
        routes=", ".join(ALLOWED_ROUTES),
        trigger_types=", ".join(ALLOWED_TRIGGER_TYPES),
        schema=_CLAUDE_SCHEMA,
    )

    try:
        msg = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = msg.content[0].text if msg.content else ""
        result   = _extract_json_from_text(raw_text)
        if not result:
            result = dict(_EMPTY_CLAUDE_RESULT)
            result["manual_review_reason"] = "Claude returned unparseable response"
        return result
    except Exception as exc:
        result = dict(_EMPTY_CLAUDE_RESULT)
        result["manual_review_reason"] = f"Claude error: {exc}"
        return result


# =============================================================================
# SCORING
# =============================================================================

def _fit_bucket(fit_score_raw, tier_raw) -> int:
    """Convert commercial fit score / tier to 0-3 bucket."""
    tier_str = str(tier_raw or "").lower()
    if "hot" in tier_str:
        return 3
    if "warm" in tier_str:
        return 2
    if "cool" in tier_str:
        return 1
    if "pass" in tier_str:
        return 0

    try:
        score = float(str(fit_score_raw).strip())
        if score >= 8.0:
            return 3
        if score >= 6.0:
            return 2
        if score >= 4.0:
            return 1
        return 0
    except (ValueError, TypeError):
        pass

    return 1  # default: something present but no score


def _contact_route_score(preferred_route: str) -> int:
    route = str(preferred_route or "").lower()
    if any(k in route for k in ("l&d", "talent", "sales enablement", "international hr")):
        return 3
    if any(k in route for k in ("hr", "people", "customer success")):
        return 2
    if any(k in route for k in ("operations",)):
        return 1
    return 0  # Unknown or Procurement


def _opportunity_score(
    fit: int, trigger: int, window: int, route: int, input_type: str
) -> float:
    if input_type == "enriched_export":
        # 35% fit · 30% trigger · 25% window · 10% route
        return round(
            (fit / 3 * 10 * 0.35)
            + (trigger / 3 * 10 * 0.30)
            + (window / 3 * 10 * 0.25)
            + (route / 3 * 10 * 0.10),
            1,
        )
    else:
        # timing-only mode: 45% trigger · 35% window · 20% route
        return round(
            (trigger / 3 * 10 * 0.45)
            + (window / 3 * 10 * 0.35)
            + (route / 3 * 10 * 0.20),
            1,
        )


def _call_recommendation(
    fit: int,
    trigger: int,
    window: int,
    opp: float,
    manual: bool,
    input_type: str,
) -> str:
    if input_type == "enriched_export":
        # ICP fit is known — use full decision matrix
        if fit == 0:
            return "Low priority"
        if trigger >= 3 and fit >= 2:
            return "Call now"
        if fit >= 2 and (trigger >= 2 or window >= 2):
            return "Call this month"
        if window >= 2 and fit >= 2:
            return "Call before budget cycle"
        if trigger >= 2 and fit >= 1:
            return "Call this month"
        if manual:
            return "Manual research needed"
        if trigger == 0 and window == 0:
            return "Monitor"
        return "Monitor"
    else:
        # ICP fit unknown — conservative; weak evidence always routes to manual review
        if trigger == 0 and window == 0:
            return "Monitor"
        # Only the very strongest signal overrides a manual/weak-evidence flag
        if trigger >= 3 and window >= 2:
            return "Call this month"
        if manual:
            return "Manual research needed"
        if trigger >= 2 and window >= 1:
            return "Call this month"
        if trigger >= 2 or window >= 2:
            return "Call before budget cycle"
        if window >= 1 and trigger >= 1:
            return "Call before budget cycle"
        return "Manual research needed"


_QUARTER_RE = re.compile(
    r"Q([1-4])[- /](\d{4})|(\d{4})[- /]Q([1-4])", re.IGNORECASE
)
_YEAR_ONLY_RE = re.compile(r"\b(20\d{2})\b")

_QUARTER_STARTS = {1: (1, 1), 2: (4, 1), 3: (7, 1), 4: (10, 1)}


def _window_to_date(window_str: str):
    """Return the start date implied by a buying-window string, or None."""
    if not window_str:
        return None
    m = _QUARTER_RE.search(window_str)
    if m:
        q = int(m.group(1) or m.group(4))
        y = int(m.group(2) or m.group(3))
        month, day = _QUARTER_STARTS[q]
        try:
            return datetime(y, month, day)
        except ValueError:
            return None
    m = _YEAR_ONLY_RE.search(window_str)
    if m:
        try:
            return datetime(int(m.group(1)), 1, 1)
        except ValueError:
            return None
    return None


def _adjust_past_buying_window(claude_result: dict) -> dict:
    """
    If likely_buying_window refers to a date that has already passed, lower the
    buying_window_score to 0 and confidence to 'Low' so the scoring formulas
    do not recommend immediate action based on stale timing.
    Annual report evidence is preserved — only the recommended window is adjusted.
    """
    window = claude_result.get("likely_buying_window", "")
    if not window:
        return claude_result
    window_date = _window_to_date(window)
    if window_date is None:
        return claude_result
    today = datetime.now()
    if window_date >= today:
        return claude_result  # window is in the future — nothing to do

    result = dict(claude_result)
    result["buying_window_score"] = 0
    result["buying_window_confidence"] = "Low"
    old_reason = result.get("buying_window_reason", "")
    result["buying_window_reason"] = (
        f"[Window expired — {window} is in the past. "
        f"Next likely planning cycle estimated.] {old_reason}"
    ).strip()
    # Project forward by one year as a placeholder
    try:
        future_year = window_date.year + 1
        future_q = (window_date.month - 1) // 3 + 1
        result["likely_buying_window"] = f"Q{future_q} {future_year} (projected)"
    except Exception:
        result["likely_buying_window"] = "Next planning cycle (estimated)"
    return result


def _compute_scores(
    claude_result: dict,
    fit_score_raw,
    tier_raw,
    input_type: str = "enriched_export",
) -> tuple:
    """
    Returns (adjusted_claude_result, scores_dict).
    adjusted_claude_result has any expired buying window projected forward.
    """
    adj = _adjust_past_buying_window(dict(claude_result))

    # For simple lists, never use commercial fit in scoring
    fit     = _fit_bucket(fit_score_raw, tier_raw) if input_type == "enriched_export" else 1
    trigger = int(adj.get("trigger_score", 0) or 0)
    window  = int(adj.get("buying_window_score", 0) or 0)
    route   = _contact_route_score(adj.get("preferred_buyer_route", ""))
    opp     = _opportunity_score(fit, trigger, window, route, input_type)
    manual  = bool(adj.get("manual_review_needed", False))

    # For simple lists with weak evidence, cap recommendation conservatively
    if input_type == "simple_company_list":
        eq = str(adj.get("evidence_quality", "")).lower()
        cl = str(adj.get("confidence_level", "")).lower()
        if eq in ("weak", "insufficient") or cl in ("low", "unknown"):
            manual = True  # force manual review path in _call_recommendation

    rec = _call_recommendation(fit, trigger, window, opp, manual, input_type)

    scores = {
        "trigger_score":        trigger,
        "buying_window_score":  window,
        "contact_route_score":  route,
        "opportunity_score":    opp,
        "call_recommendation":  rec,
    }
    return adj, scores


# =============================================================================
# COMPANY LIST BUILDER
# =============================================================================

def _build_company_list(
    df: pd.DataFrame,
    name_col: str | None,
    domain_col: str | None,
    country_col: str | None,
    score_col: str | None,
    tier_col: str | None,
    icp_col: str | None,
    input_type: str = "enriched_export",
) -> list:
    """Deduplicate by company name and return list of dicts."""
    # If name_col wasn't detected, fall back to the first string-like column
    if name_col is None:
        for col in df.columns:
            if df[col].dtype == object:
                name_col = col
                break

    def _val(row, col):
        if col and col in row.index:
            v = row[col]
            return "" if pd.isna(v) else str(v).strip()
        return ""

    seen: set = set()
    companies = []
    for i, row in df.iterrows():
        name   = _val(row, name_col)
        domain = _normalize_domain(_val(row, domain_col))

        # Fallback: use domain as display key when name is absent
        key = name or domain
        if not key:
            continue  # skip rows with no identity at all

        # Deduplicate by normalised name (prefer name over domain)
        dedup_key = name.lower() if name else domain.lower()
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        # Exclude internal / self entries
        if _is_internal(name, domain):
            companies.append({
                "company_name":             name,
                "domain":                   domain,
                "country":                  _val(row, country_col),
                "fit_score":                "",
                "tier":                     "",
                "icp_evidence":             "",
                "input_type":               input_type,
                "commercial_fit_available": False,
                "internal":                 True,
            })
            continue

        fit_score = _val(row, score_col)
        tier      = _val(row, tier_col)
        fit_avail = bool(fit_score or tier) and input_type == "enriched_export"
        companies.append({
            "company_name":             name,
            "domain":                   domain,
            "country":                  _val(row, country_col),
            "fit_score":                fit_score,
            "tier":                     tier,
            "icp_evidence":             _val(row, icp_col),
            "input_type":               input_type,
            "commercial_fit_available": fit_avail,
            "internal":                 False,
        })
    return companies


# =============================================================================
# EXCEL BUILDER
# =============================================================================

def _build_excel_bytes(results: list, raw_sources: list) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:

        # ── Sheet 1: Opportunity Radar (main summary) ─────────────────────────
        radar_cols = [
            "company_name", "domain", "country",
            "input_type", "commercial_fit_available",
            "commercial_fit_score", "commercial_tier",
            "trigger_score", "buying_window_score",
            "contact_route_score", "opportunity_score",
            "call_recommendation", "why_now", "likely_buying_window",
            "preferred_buyer_route", "backup_buyer_route",
            "suggested_title_searches", "suggested_opener",
            "confidence_level", "evidence_quality", "manual_review_needed",
        ]
        radar_rows = []
        for r in results:
            c = r.get("claude", {})
            s = r.get("scores", {})
            radar_rows.append({
                "company_name":             r.get("company_name", ""),
                "domain":                   r.get("domain", ""),
                "country":                  r.get("country", ""),
                "input_type":               r.get("input_type", ""),
                "commercial_fit_available": r.get("commercial_fit_available", False),
                "commercial_fit_score":     r.get("fit_score", ""),
                "commercial_tier":          r.get("tier", ""),
                "trigger_score":            s.get("trigger_score", 0),
                "buying_window_score":      s.get("buying_window_score", 0),
                "contact_route_score":      s.get("contact_route_score", 0),
                "opportunity_score":        s.get("opportunity_score", 0),
                "call_recommendation":      s.get("call_recommendation", ""),
                "why_now":                  c.get("why_now", ""),
                "likely_buying_window":     c.get("likely_buying_window", ""),
                "preferred_buyer_route":    c.get("preferred_buyer_route", ""),
                "backup_buyer_route":       c.get("backup_buyer_route", ""),
                "suggested_title_searches": c.get("suggested_title_searches", ""),
                "suggested_opener":         c.get("suggested_opener", ""),
                "confidence_level":         c.get("confidence_level", ""),
                "evidence_quality":         c.get("evidence_quality", ""),
                "manual_review_needed":     c.get("manual_review_needed", False),
            })
        pd.DataFrame(radar_rows, columns=radar_cols).to_excel(
            writer, index=False, sheet_name="Opportunity Radar"
        )

        # ── Sheet 2: Trigger Evidence ─────────────────────────────────────────
        trig_rows = []
        for r in results:
            c = r.get("claude", {})
            trig_rows.append({
                "company_name":   r.get("company_name", ""),
                "trigger_found":  c.get("trigger_found", False),
                "trigger_type":   c.get("trigger_type", ""),
                "trigger_date":   c.get("trigger_date", ""),
                "trigger_score":  c.get("trigger_score", 0),
                "trigger_evidence": c.get("trigger_evidence", ""),
                "source_urls":    c.get("evidence_sources", ""),
            })
        pd.DataFrame(trig_rows).to_excel(
            writer, index=False, sheet_name="Trigger Evidence"
        )

        # ── Sheet 3: Buying Window ────────────────────────────────────────────
        bw_rows = []
        for r in results:
            c = r.get("claude", {})
            bw_rows.append({
                "company_name":          r.get("company_name", ""),
                "annual_report_found":   c.get("annual_report_found", False),
                "annual_report_url":     c.get("annual_report_url", ""),
                "annual_report_date":    c.get("annual_report_date", ""),
                "fiscal_year_pattern":   c.get("fiscal_year_pattern", ""),
                "likely_buying_window":  c.get("likely_buying_window", ""),
                "buying_window_score":   c.get("buying_window_score", 0),
                "buying_window_confidence": c.get("buying_window_confidence", ""),
                "buying_window_reason":  c.get("buying_window_reason", ""),
            })
        pd.DataFrame(bw_rows).to_excel(
            writer, index=False, sheet_name="Buying Window"
        )

        # ── Sheet 4: Contact Route ────────────────────────────────────────────
        cr_rows = []
        for r in results:
            c = r.get("claude", {})
            cr_rows.append({
                "company_name":            r.get("company_name", ""),
                "preferred_buyer_route":   c.get("preferred_buyer_route", ""),
                "backup_buyer_route":      c.get("backup_buyer_route", ""),
                "suggested_title_searches": c.get("suggested_title_searches", ""),
                "suggested_opener":        c.get("suggested_opener", ""),
            })
        pd.DataFrame(cr_rows).to_excel(
            writer, index=False, sheet_name="Contact Route"
        )

        # ── Sheet 5: Caller Brief ─────────────────────────────────────────────
        brief_rows = []
        for r in results:
            c = r.get("claude", {})
            s = r.get("scores", {})
            brief_rows.append({
                "company_name":      r.get("company_name", ""),
                "call_recommendation": s.get("call_recommendation", ""),
                "why_now":           c.get("why_now", ""),
                "opener":            c.get("suggested_opener", ""),
                "buyer_route":       c.get("preferred_buyer_route", ""),
                "title_searches":    c.get("suggested_title_searches", ""),
                "evidence_summary":  c.get("trigger_evidence", ""),
            })
        pd.DataFrame(brief_rows).to_excel(
            writer, index=False, sheet_name="Caller Brief"
        )

        # ── Sheet 6: Raw Sources ──────────────────────────────────────────────
        raw_cols = [
            "company_name", "input_type", "query_group", "title", "url",
            "snippet", "date", "source_type",
        ]
        pd.DataFrame(raw_sources or [], columns=raw_cols).to_excel(
            writer, index=False, sheet_name="Raw Sources"
        )

    return buf.getvalue()


# =============================================================================
# API KEY LOADING
# =============================================================================

_anthropic_key = ""
_serper_key    = ""
try:
    _anthropic_key = (st.secrets.get("ANTHROPIC_API_KEY", "") or "").strip()
    _serper_key    = (st.secrets.get("SERPER_API_KEY",    "") or "").strip()
except Exception:
    pass

_keys_ok = bool(_anthropic_key and _serper_key and _ANTHROPIC_AVAILABLE)

# =============================================================================
# STEP 1 — UPLOAD
# =============================================================================

st.divider()
st.subheader("Step 1 · Upload your file")
st.caption(
    "Upload a Lead Prioritizer export, an Opportunity Input sheet, "
    "or a simple company list with company name and website."
)

uploaded = st.file_uploader(
    "Drag and drop here, or click to browse  (.xlsx · .xls · .csv)",
    type=["xlsx", "xls", "csv"],
    label_visibility="collapsed",
)

new_key = f"{uploaded.name}___{uploaded.size}" if uploaded else "__none__"
if new_key != ss("_or_file_key", "__none__"):
    ss_set(
        _or_file_key      = new_key,
        _or_df_raw        = None,
        _or_file_name     = None,
        _or_file_error    = None,
        _or_name_col      = None,
        _or_domain_col    = None,
        _or_country_col   = None,
        _or_score_col     = None,
        _or_tier_col      = None,
        _or_icp_col       = None,
        _or_n_companies   = 0,
        _or_input_type    = None,
        _or_processing    = False,
        _or_done          = False,
        _or_process_index = 0,
        _or_company_list  = None,
        _or_results       = None,
        _or_raw_sources   = None,
        _or_excel_bytes   = None,
        _or_stop          = False,
    )
    if uploaded is not None:
        try:
            fname             = uploaded.name
            df_loaded, sheet  = _load_df_from_upload(uploaded)
            input_type        = _detect_input_type(df_loaded)
            name_col          = _detect_col(df_loaded, _NAME_CANDIDATES)
            domain_col        = _detect_col(df_loaded, _DOMAIN_CANDIDATES)
            country_col       = _detect_col(df_loaded, _COUNTRY_CANDIDATES)
            score_col         = _detect_col(df_loaded, _SCORE_CANDIDATES)
            tier_col          = _detect_col(df_loaded, _TIER_CANDIDATES)
            icp_col           = _detect_col(df_loaded, _ICP_CANDIDATES)
            n                 = _count_companies(df_loaded, name_col)
            ss_set(
                _or_df_raw      = df_loaded,
                _or_file_name   = fname,
                _or_name_col    = name_col,
                _or_domain_col  = domain_col,
                _or_country_col = country_col,
                _or_score_col   = score_col,
                _or_tier_col    = tier_col,
                _or_icp_col     = icp_col,
                _or_n_companies = n,
                _or_input_type  = input_type,
            )
        except Exception as exc:
            ss_set(_or_file_error=str(exc))

if ss("_or_file_error"):
    st.error(f"Could not read the file: {ss('_or_file_error')}")
elif ss("_or_df_raw") is not None and not ss("_or_processing", False) and not ss("_or_done", False):
    n          = ss("_or_n_companies", 0)
    itype      = ss("_or_input_type", "")
    itype_label = (
        "enriched export detected" if itype == "enriched_export"
        else "simple company list detected"
    )
    st.success(
        f"✓ **{ss('_or_file_name')}** loaded · "
        f"{n:,} {'company' if n == 1 else 'companies'} ready · "
        f"{itype_label}"
    )

# =============================================================================
# MISSING KEYS WARNING
# =============================================================================

if not _keys_ok and not ss("_or_done", False):
    missing = []
    if not _anthropic_key or not _ANTHROPIC_AVAILABLE:
        missing.append("ANTHROPIC_API_KEY")
    if not _serper_key:
        missing.append("SERPER_API_KEY")
    if missing:
        st.warning(
            f"⚠ Missing secrets: {', '.join(missing)}. "
            "Add them to .streamlit/secrets.toml to run the scan."
        )

# =============================================================================
# STEP 2 — START / PROCESSING LOOP
# =============================================================================

_ready      = ss("_or_df_raw") is not None and _keys_ok
_processing = ss("_or_processing", False)
_done       = ss("_or_done", False)

if not _done and not _processing:
    start_btn = st.button(
        "▶ Start radar scan",
        type="primary",
        use_container_width=True,
        disabled=not _ready,
        key="or_start_btn",
    )

    if start_btn and _ready:
        df_raw = ss("_or_df_raw")
        company_list = _build_company_list(
            df_raw,
            ss("_or_name_col"),
            ss("_or_domain_col"),
            ss("_or_country_col"),
            ss("_or_score_col"),
            ss("_or_tier_col"),
            ss("_or_icp_col"),
            ss("_or_input_type", "simple_company_list"),
        )
        ss_set(
            _or_processing    = True,
            _or_done          = False,
            _or_process_index = 0,
            _or_company_list  = company_list,
            _or_results       = [],
            _or_raw_sources   = [],
            _or_excel_bytes   = None,
            _or_stop          = False,
        )
        st.rerun()

if _processing and not _done:
    company_list = ss("_or_company_list", [])
    results      = ss("_or_results", [])
    raw_sources  = ss("_or_raw_sources", [])
    idx          = ss("_or_process_index", 0)
    n_total      = len(company_list)

    # ── Stop button ───────────────────────────────────────────────────────────
    if st.button("⏹ Stop", key="or_stop_btn"):
        ss_set(_or_stop=True)

    if ss("_or_stop", False):
        ss_set(_or_processing=False, _or_done=True)
        st.rerun()

    # ── Progress display ──────────────────────────────────────────────────────
    if n_total:
        st.progress(idx / n_total, text=f"Scanning {idx} of {n_total} companies…")
        if idx < n_total:
            current_name = company_list[idx].get("company_name") or company_list[idx].get("domain", "")
            st.caption(f"Current company: **{current_name}**")

    # ── Process one company ───────────────────────────────────────────────────
    if idx < n_total:
        company     = company_list[idx]
        name        = company.get("company_name", "")
        domain      = company.get("domain", "")
        country     = company.get("country", "")
        fit_score   = company.get("fit_score", "")
        tier        = company.get("tier", "")
        icp_ev      = company.get("icp_evidence", "")
        c_itype     = company.get("input_type", "simple_company_list")
        c_fit_avail = company.get("commercial_fit_available", False)
        is_internal = company.get("internal", False)

        if is_internal:
            # Mark without any research
            record = {
                "company_name":             name,
                "domain":                   domain,
                "country":                  country,
                "fit_score":                "",
                "tier":                     "",
                "input_type":               c_itype,
                "commercial_fit_available": False,
                "claude": {
                    **_EMPTY_CLAUDE_RESULT,
                    "why_now": "Internal / exclude",
                    "trigger_evidence": "Internal company — excluded from radar.",
                },
                "scores": {
                    "trigger_score":       0,
                    "buying_window_score": 0,
                    "contact_route_score": 0,
                    "opportunity_score":   0,
                    "call_recommendation": "Internal / exclude",
                },
            }
            results.append(record)
        else:
            # Check cache first — key is scoped to input_type so enriched/simple never collide
            cached = _cache_load(name, domain, c_itype)
            if cached is not None:
                # Enforce correct fit data for this input type
                cached["input_type"]              = c_itype
                cached["commercial_fit_available"] = c_fit_avail
                if c_itype == "simple_company_list":
                    # Strip any enriched fit values that crept into the cache
                    cached["fit_score"] = ""
                    cached["tier"]      = ""
                # Re-apply window adjustment and recompute scores (in case window aged)
                adj_claude, fresh_scores = _compute_scores(
                    cached.get("claude", {}),
                    cached.get("fit_score", ""),
                    cached.get("tier", ""),
                    c_itype,
                )
                cached["claude"] = adj_claude
                cached["scores"] = fresh_scores
                results.append(cached)
                # Restore raw sources stored in cache (if any)
                raw_sources.extend(cached.get("raw_sources", []))
            else:
                # Run Serper searches
                grouped_results = _run_searches(name or domain, domain, _serper_key)
                sources         = _collect_raw_sources(name, grouped_results, c_itype)

                # Call Claude
                client = _anthropic_mod.Anthropic(api_key=_anthropic_key)
                raw_claude = _call_claude(
                    name, domain, country, fit_score, tier, icp_ev,
                    grouped_results, client,
                )

                # Adjust window + compute scores; formula depends on input type
                adj_claude, scores = _compute_scores(raw_claude, fit_score, tier, c_itype)

                # For simple lists: never carry commercial fit values
                out_fit_score = fit_score if c_itype == "enriched_export" else ""
                out_tier      = tier      if c_itype == "enriched_export" else ""

                record = {
                    "company_name":             name,
                    "domain":                   domain,
                    "country":                  country,
                    "fit_score":                out_fit_score,
                    "tier":                     out_tier,
                    "input_type":               c_itype,
                    "commercial_fit_available": c_fit_avail,
                    "claude":                   adj_claude,
                    "scores":                   scores,
                    "raw_sources":              sources,
                }
                _cache_save(name, domain, c_itype, record)
                results.append(record)
                raw_sources.extend(sources)

        ss_set(
            _or_results       = results,
            _or_raw_sources   = raw_sources,
            _or_process_index = idx + 1,
        )
        st.rerun()

    else:
        # All companies processed
        ss_set(_or_processing=False, _or_done=True)
        st.rerun()

# =============================================================================
# STEP 3 — RESULTS + DOWNLOAD
# =============================================================================

if _done:
    results     = ss("_or_results", [])
    raw_sources = ss("_or_raw_sources", [])
    processed   = len(results)
    n           = ss("_or_n_companies", 0)

    st.success(
        f"✅ Ready · **{processed:,}** "
        f"{'company' if processed == 1 else 'companies'} scanned"
    )

    # Build Excel once, cache bytes in session state
    if ss("_or_excel_bytes") is None:
        ss_set(_or_excel_bytes=_build_excel_bytes(results, raw_sources))

    st.download_button(
        label="⬇ Download opportunity radar",
        data=ss("_or_excel_bytes"),
        file_name=f"opportunity_radar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        type="primary",
    )

    st.divider()
    if st.button("↺ Start a new radar scan", use_container_width=True, key="or_restart_btn"):
        reset()
        st.rerun()
