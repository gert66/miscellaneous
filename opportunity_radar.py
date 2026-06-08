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

# Bump this string whenever the prompt or interpretation logic changes.
# It is included in cache keys so old Claude outputs are never reused after a prompt update.
CACHE_VERSION = "v3_myngle_20260608"

# 5 query groups — one Serper call each
QUERY_GROUPS = [
    (
        "Annual Report / Financial",
        '"{name}" annual report fiscal year results revenue 2024 2025',
    ),
    (
        "International Hiring / Growth",
        '"{name}" hiring international careers jobs "new office" global expansion 2024 2025',
    ),
    (
        "Language / Communication / L&D",
        '"{name}" "language training" OR "business English" OR "communication training" '
        'OR "learning and development" OR "talent development" OR training academy HR',
    ),
    (
        "Sales / Customer Success Expansion",
        '"{name}" "sales team" OR "customer success" OR "account management" '
        'OR "sales enablement" OR "client-facing" international expansion',
    ),
    (
        "M&A / Funding / Integration",
        '"{name}" acquisition OR merger OR integration OR funding OR investment OR "private equity"',
    ),
]

ALLOWED_TRIGGER_TYPES = [
    "International hiring",
    "Client-facing team expansion",
    "Sales / customer success growth",
    "New market or office expansion",
    "Multilingual workforce growth",
    "M&A / integration",
    "Funding / growth investment",
    "HR / L&D hiring",
    "Onboarding pressure",
    "Foreign HQ / group communication",
    "Employer branding / retention",
    "Annual planning / budget window",
    "No clear trigger",
    "Other",
]

ALLOWED_ROUTES = [
    "L&D / Talent Development",
    "HR / People",
    "International HR",
    "Sales Enablement",
    "Customer Success",
    "People Operations",
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
        _or_force_refresh=False,
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
    # CACHE_VERSION + input_type in key → old prompts never bleed into new runs
    raw = f"{CACHE_VERSION}|{name.lower().strip()}|{domain.lower().strip()}|{input_type}"
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
You are a B2B sales intelligence analyst working for mYngle, a company that sells \
online language training and business communication support to international companies.

mYngle's target buyers are companies that have:
- International or multilingual teams
- Foreign HQ or group structures
- Client-facing international roles (sales, customer success, account management)
- Fast hiring or onboarding of international employees
- Post-merger or cross-border communication challenges
- Expanding into new countries or markets

mYngle does NOT sell: generic L&D platforms, e-learning tools, or broad HR software.
mYngle sells: language training, Business English, business communication coaching.

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

ANALYSIS INSTRUCTIONS:

1. TRIGGER IDENTIFICATION
   Look for signals that a company is likely to need language or communication training:
   - International hiring, new foreign offices, global expansion
   - Growth of sales, customer success, or account management teams
   - M&A, integration activity, foreign group/HQ structure
   - Onboarding pressure from fast hiring
   - L&D or HR team growth suggesting new training budget
   - Annual report published (signals planning cycle timing)
   trigger_score: 0=no relevant signal, 1=weak/indirect, 2=clear signal, 3=strong + recent

   trigger_type MUST be exactly one of: {trigger_types}
   Choose the single strongest primary trigger. Do not combine or invent labels.
   If multiple triggers are present, pick the most commercially relevant one for mYngle
   and add context to trigger_evidence.
   Examples of correct mapping:
   - "International expansion + multinational workforce" → "Multilingual workforce growth"
   - "L&D / Onboarding Infrastructure + Annual Planning Cycle" → "HR / L&D hiring"
   - "Annual planning signal" → "Annual planning / budget window"
   - No relevant trigger found → "No clear trigger"

2. BUYING WINDOW
   - IMPORTANT: likely_buying_window must be FUTURE relative to today ({today})
   - If best evidence points to a past window, project forward one year and set
     buying_window_confidence to "Low"
   - Do NOT invent a specific quarter unless the fiscal year pattern and a planning
     logic chain clearly support it. Use hedged language:
       * "Possible Q3/Q4 planning conversation, assuming calendar-year budgeting"
       * "Possible annual planning window, confidence low"
       * "Possible fiscal-year planning conversation, timing not confirmed"
       * "Current trigger found, timing appears commercially relevant"
       * "No clear buying window found"
   - Only say "Q3 2026" if you can explain WHY Q3 — e.g. fiscal year ends Dec, so
     budget planning typically happens Aug/Sep. Otherwise use hedged wording.
   - If annual report found, it confirms a planning cycle exists but does NOT by itself
     confirm a specific quarter unless the fiscal year end is stated.
   buying_window_score: 0=no basis, 1=possible, 2=likely, 3=imminent

3. BUYER ROUTE — choose based on the dominant trigger signal:
   - "L&D / Talent Development": training, onboarding, people development, academy signals
   - "HR / People": general HR, people, workforce signals without strong L&D angle
   - "International HR": foreign HQ, global teams, multilingual workforce, cross-border structure
   - "Sales Enablement": sales expansion, account management, negotiation, international sales
   - "Customer Success": customer success growth, international client support
   - "People Operations": fast onboarding, operational employee growth, multi-site rollout
   - "Operations": only when operational coordination is the clearest angle
   - "Procurement": last resort only — never preferred first route
   - "Unknown": truly no signal to guide route choice
   preferred_buyer_route must be one of: {routes}

4. SUGGESTED TITLE SEARCHES (for LinkedIn Sales Navigator)
   Match to preferred and backup buyer routes. Output as a single string using OR syntax.
   Use ONLY the OR-syntax format. Do NOT use comma-separated lists.
   - L&D route: "Learning Development" OR "Talent Development" OR "L&D"
   - HR/People route: "HR Director" OR "People Director" OR "Head of People"
   - International HR route: "International HR" OR "Global HR" OR "People Operations"
   - Sales Enablement route: "Sales Enablement" OR "Revenue Enablement"
   - Customer Success route: "Customer Success Director" OR "VP Customer Success"
   - People Operations route: "Onboarding" OR "People Operations" OR "HR Operations"
   Combine preferred and backup routes: e.g.
   "Learning Development" OR "L&D" OR "HR Director" OR "Head of People"

5. WHY NOW — must connect evidence to mYngle's value proposition
   Focus on: language training, Business English, business communication, client-facing
   communication, international team communication, onboarding of international employees,
   multilingual workforce support, intercultural communication, foreign HQ communication,
   cross-border communication, business communication training.

   BANNED PHRASES — never write any of the following:
   "learning platform", "talent development tools", "upskilling solutions",
   "digital learning", "workforce solution", "talent tools", "e-learning",
   "learning tools", "broad L&D", "generic L&D", "training platform",
   "talent management", "HR platform", "learning management", "LMS",
   "skill development platform", "workforce training platform".

   Use instead: "language training", "Business English", "business communication training",
   "client-facing communication", "international team communication",
   "multilingual workforce support", "onboarding communication support",
   "cross-border communication", "intercultural communication".

   Example: "Capgemini has large international teams and active internal training
   infrastructure. Companies at that scale often review Business English and
   client-facing communication support during annual planning cycles."

6. CALLER OPENER — short, natural, specific
   - Mention one concrete signal from the search results
   - Connect it to language training or business communication
   - End with a soft discovery question
   - BANNED: "learning platform", "digital learning tools", "talent development tools",
     "upskilling", "workforce solutions", "HR software"
   - The question should feel like a natural cold-call opening, not a product pitch
   - Examples by trigger:
     * International hiring: "I noticed you're expanding internationally and hiring across
       new markets. Companies often use that moment to review language and communication
       support for new teams. Is business communication training already part of your
       L&D planning?"
     * Customer-facing growth: "I noticed growth in your international customer-facing
       teams. That often creates pressure around Business English and client communication.
       Is this something your team is already looking at?"
     * M&A/integration: "I noticed recent integration activity at {name}. Those transitions
       often bring communication and language alignment challenges across teams and countries.
       Is language training part of the integration plan?"
     * Annual planning: "I noticed your annual planning cycle may be coming up. Many
       companies review language and business communication training before finalising their
       L&D budget. Is this already on your agenda?"
     * Large international workforce: "I noticed {name} has extensive international
       operations. Companies at that scale often have ongoing needs around Business English
       and cross-border communication support. Is language training already part of your
       current L&D planning?"

7. CONFIDENCE AND EVIDENCE QUALITY
   - evidence_quality: Strong=multiple recent specific sources, Medium=1-2 relevant sources,
     Weak=snippets only or indirect signals, Insufficient=no relevant evidence
   - confidence_level: High only if Strong evidence + clear trigger + clear buyer route,
     Medium if some evidence present, Low/Unknown if mostly snippets or indirect
   - manual_review_needed: true if evidence is Weak/Insufficient OR buyer route unclear
     OR company seems relevant but signals are ambiguous

SCORING FIELD INSTRUCTIONS:
- trigger_score: 0-3 as above
- buying_window_score: 0-3 as above
- hiring_signal_score: 0-3 (overall hiring volume signal)
- international_hiring_signal: 0-3 (international/multilingual hiring specifically)
- lnd_hr_hiring_signal: 0-3 (L&D or HR hiring that signals training budget)
- sales_cs_hiring_signal: 0-3 (sales or customer success growth)
- onboarding_pressure_signal: 0-3 (fast hiring, headcount growth, new site openings)

Return ONLY the JSON object below — no markdown fences, no explanation text.

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
            max_tokens=1500,
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
    if any(k in route for k in ("l&d", "talent development", "sales enablement", "international hr")):
        return 3
    if any(k in route for k in ("hr / people", "customer success", "people operations")):
        return 2
    if any(k in route for k in ("hr", "people", "operations")):
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
    evidence_quality: str = "",
) -> str:
    eq = evidence_quality.lower()
    eq_strong = eq in ("strong", "medium")

    if input_type == "enriched_export":
        # ICP fit is known — use full decision matrix
        if fit == 0:
            return "Low priority"
        # Cool fit (1) with no trigger → just monitor; don't create false urgency
        if fit == 1 and trigger == 0 and window == 0:
            return "Monitor"
        # Call now only when fit is strong, trigger is strong, AND evidence is credible
        if trigger >= 3 and fit >= 2 and eq_strong:
            return "Call now"
        if fit >= 2 and trigger >= 2 and eq_strong:
            return "Call this month"
        if fit >= 2 and window >= 2:
            return "Call this month"
        if fit >= 2 and trigger >= 1:
            return "Call before budget cycle" if window >= 1 else "Monitor"
        # Hot/Warm fit but no trigger and manual flag → human should research next
        if fit >= 2 and manual:
            return "Manual research needed"
        if trigger == 0 and window == 0:
            return "Monitor"
        if manual:
            return "Manual research needed"
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


# Banned phrases that must never appear in caller-facing text fields
_BANNED_PHRASES = [
    "learning platform", "talent development tools", "upskilling solution",
    "digital learning tool", "workforce solution", "talent tools", "e-learning tool",
    "learning management", "talent management", "hr platform", "lms",
    "skill development platform", "workforce training platform",
    "broad l&d", "generic l&d", "learning tools", "training platform",
]

# Phrase-replacement map: banned generic phrase (lowercase) → mYngle-specific replacement
_PHRASE_REPLACEMENTS: list[tuple[str, str]] = [
    # Most specific multi-word first to prevent partial replacements
    ("l&d technology solutions",            "language and communication training"),
    ("learning and development technology", "language training"),
    ("talent development tools",            "language training"),
    ("talent development solutions",        "language training"),
    ("talent development platform",         "language training platform"),
    ("learning platforms",                  "language training"),
    ("learning platform",                   "language training"),
    ("upskilling solutions",                "language and communication training"),
    ("upskilling solution",                 "language and communication training"),
    ("digital learning tools",              "language training"),
    ("digital learning tool",               "language training"),
    ("digital learning",                    "language training"),
    ("workforce solutions",                 "language training"),
    ("workforce solution",                  "language training"),
    ("workforce training platform",         "language training"),
    ("talent tools",                        "language training"),
    ("e-learning tools",                    "language training"),
    ("e-learning tool",                     "language training"),
    ("learning management system",          "language training programme"),
    ("broad l&d",                           "language and communication training"),
    ("generic l&d",                         "language training"),
    ("l&d technology",                      "language and communication training"),
    ("training platform",                   "language training programme"),
    ("skill development platform",          "language training"),
    ("hr platform",                         "people development platform"),
    ("talent management",                   "people development"),
    ("evaluating new learning",             "reviewing language and communication training"),
    ("evaluating learning",                 "reviewing language training"),
    ("new learning tools",                  "language training"),
]


def _sanitize_text(text: str) -> str:
    """Replace banned generic phrases with mYngle-specific wording. Case-preserving."""
    if not text:
        return text
    result = text
    lower = result.lower()
    # Work on lowercase index to find replacement positions; rebuild preserving original case
    for banned, replacement in _PHRASE_REPLACEMENTS:
        start = 0
        while True:
            idx = lower.find(banned, start)
            if idx == -1:
                break
            result = result[:idx] + replacement + result[idx + len(banned):]
            lower  = result.lower()
            start  = idx + len(replacement)
    return result


def _sanitize_claude_result(cr: dict) -> dict:
    """Apply text sanitizer to all caller-facing text fields in a Claude result dict."""
    fields = ("why_now", "suggested_opener", "trigger_evidence",
              "buying_window_reason", "manual_review_reason")
    sanitized = dict(cr)
    for f in fields:
        if f in sanitized and isinstance(sanitized[f], str):
            sanitized[f] = _sanitize_text(sanitized[f])
    return sanitized


# Fuzzy mapping: substrings in Claude's free-text → canonical trigger type
_TRIGGER_MAP: list[tuple[str, str]] = [
    # Most specific first
    ("multilingual",              "Multilingual workforce growth"),
    ("foreign hq",                "Foreign HQ / group communication"),
    ("cross-border",              "Foreign HQ / group communication"),
    ("group communication",       "Foreign HQ / group communication"),
    ("m&a",                       "M&A / integration"),
    ("merger",                    "M&A / integration"),
    ("acquisition",               "M&A / integration"),
    ("integrat",                  "M&A / integration"),
    ("funding",                   "Funding / growth investment"),
    ("investment",                "Funding / growth investment"),
    ("private equity",            "Funding / growth investment"),
    ("client-facing",             "Client-facing team expansion"),
    ("customer success",          "Client-facing team expansion"),
    ("account manag",             "Client-facing team expansion"),
    ("sales",                     "Sales / customer success growth"),
    ("revenue",                   "Sales / customer success growth"),
    ("international hiring",      "International hiring"),
    ("international expansion",   "New market or office expansion"),
    ("new market",                "New market or office expansion"),
    ("new office",                "New market or office expansion"),
    ("global expansion",          "New market or office expansion"),
    ("onboard",                   "Onboarding pressure"),
    ("rapid growth",              "Onboarding pressure"),
    ("fast hiring",               "Onboarding pressure"),
    ("employer brand",            "Employer branding / retention"),
    ("retention",                 "Employer branding / retention"),
    ("l&d hiring",                "HR / L&D hiring"),
    ("hr hiring",                 "HR / L&D hiring"),
    ("learning and development",  "HR / L&D hiring"),
    ("annual plan",               "Annual planning / budget window"),
    ("budget",                    "Annual planning / budget window"),
    ("fiscal year",               "Annual planning / budget window"),
    ("annual report",             "Annual planning / budget window"),
    ("hiring",                    "International hiring"),
]

_ALLOWED_TRIGGER_SET = set(ALLOWED_TRIGGER_TYPES)


def _normalize_trigger_type(raw: str) -> str:
    """Map Claude's free-text trigger_type to the fixed taxonomy."""
    if raw in _ALLOWED_TRIGGER_SET:
        return raw
    raw_lower = raw.lower()
    for fragment, canonical in _TRIGGER_MAP:
        if fragment in raw_lower:
            return canonical
    if not raw or raw.lower() in ("none", "no trigger", "no clear trigger"):
        return "No clear trigger"
    return "Other"


_TITLE_SEARCH_ROUTE_MAP: dict[str, str] = {
    "l&d / talent development":  '"Learning Development" OR "Talent Development" OR "L&D"',
    "hr / people":               '"HR Director" OR "People Director" OR "Head of People"',
    "international hr":          '"International HR" OR "Global HR" OR "People Operations"',
    "sales enablement":          '"Sales Enablement" OR "Revenue Enablement"',
    "customer success":          '"Customer Success Director" OR "VP Customer Success"',
    "people operations":         '"Onboarding" OR "People Operations" OR "HR Operations"',
    "operations":                '"Operations Director" OR "Head of Operations"',
}


def _normalize_title_searches(raw: str, preferred_route: str, backup_route: str) -> str:
    """
    If Claude returned a comma-separated list instead of OR syntax, rebuild it
    from the preferred and backup routes.
    """
    if raw and " OR " in raw:
        return raw  # already in correct format

    # Rebuild from routes
    parts = []
    for route in (preferred_route, backup_route):
        key = str(route or "").lower()
        for route_key, titles in _TITLE_SEARCH_ROUTE_MAP.items():
            if route_key in key:
                parts.append(titles)
                break

    if parts:
        return " OR ".join(parts)

    # Fallback: if comma-separated, convert to OR syntax (keep as-is but add OR)
    if raw and "," in raw:
        pieces = [p.strip().strip('"') for p in raw.split(",") if p.strip()]
        return " OR ".join(f'"{p}"' for p in pieces[:6])

    return raw or ""


def _cap_opportunity_score(opp: float, rec: str, input_type: str, eq: str, trigger_type: str) -> float:
    """Apply post-processing caps to opportunity_score for simple company list."""
    if input_type != "simple_company_list":
        return opp
    eq_lower = eq.lower()
    if rec == "Manual research needed":
        opp = min(opp, 6.0)
    if eq_lower in ("medium", "weak", "insufficient") and trigger_type in ("No clear trigger", "Annual planning / budget window", "Other"):
        opp = min(opp, 5.5)
    if trigger_type == "No clear trigger":
        opp = min(opp, 4.5)
    return round(opp, 1)


def _infer_buyer_route_from_context(icp_evidence: str) -> str:
    """Infer a non-Unknown buyer route from ICP evidence text."""
    ev = (icp_evidence or "").lower()
    if any(k in ev for k in ("international", "multilingual", "global", "cross-border", "foreign hq")):
        return "International HR"
    if any(k in ev for k in ("l&d", "learning and development", "talent development", "training", "academy")):
        return "L&D / Talent Development"
    if any(k in ev for k in ("sales", "account manag", "revenue", "commercial team")):
        return "Sales Enablement"
    if any(k in ev for k in ("customer success", "client-facing", "client facing")):
        return "Customer Success"
    if any(k in ev for k in ("onboard", "people operations", "hr operations")):
        return "People Operations"
    return "HR / People"


def _fallback_opener(name: str, route: str) -> str:
    """Generate a cautious, mYngle-specific cold-call opener when Claude left it blank."""
    r = route.lower()
    if "international" in r:
        return (
            f"Given {name}'s international teams and cross-border operations, "
            "I wanted to ask whether Business English or cross-border communication training "
            "is currently part of your L&D planning."
        )
    if "l&d" in r or "talent development" in r:
        return (
            f"I noticed {name} has an active L&D function. "
            "I wanted to check whether language training or business communication support "
            "is currently part of your training agenda."
        )
    if "sales enablement" in r:
        return (
            f"Given {name}'s international sales and account teams, "
            "I wanted to ask whether Business English or client communication training "
            "is currently being reviewed."
        )
    if "customer success" in r:
        return (
            f"Given {name}'s customer-facing teams, "
            "I wanted to ask whether business communication or language training "
            "is currently on your L&D agenda."
        )
    return (
        f"Given {name}'s international presence, "
        "I wanted to ask whether Business English or cross-border communication training "
        "is currently part of your L&D planning."
    )


def _apply_enriched_fallback(
    adj: dict,
    fit_score_raw,
    tier_raw,
    icp_evidence: str,
    company_name: str,
    input_type: str,
) -> dict:
    """
    For enriched exports where Claude returned blank guidance, fill in fallback
    why_now, buyer route, opener, and buying_window from commercial fit context.
    Never called for simple company lists.
    """
    if input_type != "enriched_export":
        return adj

    fit = _fit_bucket(fit_score_raw, tier_raw)

    # Low-fit / Pass companies: ensure why_now explains the low priority
    if fit == 0:
        if not adj.get("why_now"):
            adj = dict(adj)
            adj["why_now"] = (
                f"{company_name} has a low commercial fit and no current trigger was found. "
                "No immediate language or communication training need is evident."
            )
        return adj

    # Infer buyer route from ICP evidence when Claude returned Unknown or blank
    if not adj.get("preferred_buyer_route") or adj.get("preferred_buyer_route") == "Unknown":
        adj = dict(adj)
        inferred = _infer_buyer_route_from_context(icp_evidence)
        adj["preferred_buyer_route"] = inferred
        adj["suggested_title_searches"] = _normalize_title_searches("", inferred, "")

    # Fill blank buying window
    if not adj.get("likely_buying_window"):
        adj = dict(adj)
        adj["likely_buying_window"] = "No clear buying window found"
        adj["buying_window_confidence"] = adj.get("buying_window_confidence") or "Unknown"

    # Fill blank why_now for commercially relevant companies
    if not adj.get("why_now"):
        adj = dict(adj)
        tier_str = str(tier_raw or "").strip()
        score_str = str(fit_score_raw or "").strip()
        if fit >= 3:
            fit_desc = "a very strong mYngle-fit company (tier: Hot)"
        elif fit >= 2:
            fit_desc = "a good mYngle-fit company (tier: Warm)"
        else:
            fit_desc = "a potential mYngle-fit company (tier: Cool)"
        adj["why_now"] = (
            f"{company_name} is {fit_desc}. "
            "No concrete recent timing trigger was found in available sources. "
            "Best next step: manually research current hiring, L&D planning, "
            "international team growth, or customer-facing expansion before calling."
        )

    # Fill blank opener
    if not adj.get("suggested_opener"):
        adj = dict(adj)
        adj["suggested_opener"] = _fallback_opener(
            company_name, adj.get("preferred_buyer_route", "")
        )

    return adj


def _compute_scores(
    claude_result: dict,
    fit_score_raw,
    tier_raw,
    input_type: str = "enriched_export",
    icp_evidence: str = "",
    company_name: str = "",
) -> tuple:
    """
    Returns (adjusted_claude_result, scores_dict).
    adjusted_claude_result has any expired buying window projected forward,
    banned phrases sanitized, and (for enriched exports) fallback guidance filled
    from commercial fit context when Claude returned no trigger.
    """
    adj = _adjust_past_buying_window(dict(claude_result))

    # Sanitize banned generic phrases — runs on cached AND fresh results
    adj = _sanitize_claude_result(adj)

    # Normalize trigger_type to fixed taxonomy
    adj["trigger_type"] = _normalize_trigger_type(adj.get("trigger_type", ""))

    # Normalize title searches to OR syntax
    adj["suggested_title_searches"] = _normalize_title_searches(
        adj.get("suggested_title_searches", ""),
        adj.get("preferred_buyer_route", ""),
        adj.get("backup_buyer_route", ""),
    )

    # For simple lists, never use commercial fit in scoring
    fit     = _fit_bucket(fit_score_raw, tier_raw) if input_type == "enriched_export" else 1
    trigger = int(adj.get("trigger_score", 0) or 0)
    window  = int(adj.get("buying_window_score", 0) or 0)
    route   = _contact_route_score(adj.get("preferred_buyer_route", ""))
    opp     = _opportunity_score(fit, trigger, window, route, input_type)
    eq      = str(adj.get("evidence_quality", ""))
    manual  = bool(adj.get("manual_review_needed", False))

    # For simple lists: cap confidence and force manual when evidence is weak
    if input_type == "simple_company_list":
        eq_low = eq.lower()
        if eq_low in ("weak", "insufficient"):
            manual = True
        cl = str(adj.get("confidence_level", "")).lower()
        if cl == "high":
            adj["confidence_level"] = "Medium"

    rec = _call_recommendation(fit, trigger, window, opp, manual, input_type, eq)

    # Cap opportunity_score for simple inputs that receive conservative recommendations
    opp = _cap_opportunity_score(opp, rec, input_type, eq, adj.get("trigger_type", ""))

    # For enriched exports: fill blank guidance from commercial fit context
    adj = _apply_enriched_fallback(adj, fit_score_raw, tier_raw, icp_evidence, company_name, input_type)
    # Re-normalise route score after possible fallback route assignment
    route = _contact_route_score(adj.get("preferred_buyer_route", ""))

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
                "company_name":         r.get("company_name", ""),
                "domain":               r.get("domain", ""),
                "call_recommendation":  s.get("call_recommendation", ""),
                "why_now":              c.get("why_now", ""),
                "trigger_type":         c.get("trigger_type", ""),
                "buying_window":        c.get("likely_buying_window", ""),
                "preferred_buyer_route": c.get("preferred_buyer_route", ""),
                "backup_buyer_route":   c.get("backup_buyer_route", ""),
                "title_searches":       c.get("suggested_title_searches", ""),
                "opener":               c.get("suggested_opener", ""),
                "evidence_summary":     c.get("trigger_evidence", ""),
                "evidence_quality":     c.get("evidence_quality", ""),
                "confidence_level":     c.get("confidence_level", ""),
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
    force_refresh = st.checkbox(
        "Force fresh scan — ignore cached results",
        value=ss("_or_force_refresh", False),
        key="or_force_refresh_cb",
        help=(
            f"When checked, cached analysis is skipped and every company is "
            f"re-fetched and re-analysed from scratch. "
            f"Cache version: {CACHE_VERSION}"
        ),
    )
    ss_set(_or_force_refresh=force_refresh)

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
            # Check cache — skip if force refresh is requested
            force_refresh = ss("_or_force_refresh", False)
            cached = None if force_refresh else _cache_load(name, domain, c_itype)
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
                    icp_evidence=icp_ev,
                    company_name=name,
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
                adj_claude, scores = _compute_scores(
                    raw_claude, fit_score, tier, c_itype,
                    icp_evidence=icp_ev, company_name=name,
                )

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
