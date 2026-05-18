"""
Claude + Jina AI Two-Step Company Enrichment
=============================================
Upload a file with company names and URLs.
Each row gets TWO enrichment passes:

  Step 1 — Basic firmographics (Jina AI Reader → Claude extraction)
  Step 2 — Mingle ICP signals  (Claude with web_search tool)

Architecture
------------
- One row is processed per Streamlit rerun so the Stop button works at any point.
- All mutable run state lives in st.session_state (keys prefixed with _).
- Anthropic API key is read ONLY from st.secrets['ANTHROPIC_API_KEY'].
- Debug mode: toggled via sidebar checkbox.
"""

import base64
import io
import json
import os
import re
import time
import unicodedata
import zipfile
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import quote

import anthropic
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from bs4 import BeautifulSoup

try:
    from human_scraper import scrape_with_human_behaviour
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

JINA_READER_URL  = "https://r.jina.ai/"
JINA_SEARCH_URL  = "https://s.jina.ai/"
CACHE_DIR        = Path("claude_json_cache")
DEBUG_LOG_DIR    = Path("debug_logs")
SEARCH_OUTPUT_DIR = DEBUG_LOG_DIR / "search_outputs"
AUTOSAVE_PATH         = "/tmp/enrichment_autosave.csv"
LOCAL_SAVE_EVERY      = 5    # filesystem snapshot every N companies (local runs)
_AUTO_DL_EVERY        = 100  # auto browser-download every N companies
_DEFAULT_DOWNLOAD_DIR = os.path.expanduser("~/Downloads")
MODEL_STEP1      = "claude-haiku-4-5-20251001"
MODEL_STEP2      = "claude-haiku-4-5-20251001"
MODEL_ID         = MODEL_STEP1   # legacy alias used in a few places
WEB_SEARCH_TOOL  = {"type": "web_search_20250305", "name": "web_search"}

SERPER_SEARCH_URL    = "https://google.serper.dev/search"
STEP2_PROVIDER_CLAUDE = "Claude Web Search"
STEP2_PROVIDER_SERPER = "Serper Google Search"

AVAILABLE_MODELS = {
    "Haiku 4.5 (fast, cheap)":                   "claude-haiku-4-5-20251001",
    "Sonnet 4.5 (better reasoning, higher cost)": "claude-sonnet-4-5-20250929",
}
# Per-row cost estimates for the pre-run info banner
_COST_EST = {
    ("claude-haiku-4-5-20251001",  "claude-haiku-4-5-20251001"):  0.01,
    ("claude-haiku-4-5-20251001",  "claude-sonnet-4-5-20250929"): 0.05,
    ("claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001"):  0.04,
    ("claude-sonnet-4-5-20250929", "claude-sonnet-4-5-20250929"): 0.08,
}

# Pricing per million tokens (claude-haiku-4-5)
_COST_INPUT_PER_M  = 0.80
_COST_OUTPUT_PER_M = 4.00

# Step 1 extraction prompt
_STEP1_PROMPT = (
    "Extract company information from this webpage content. "
    "Return ONLY a raw JSON object with these exact fields: "
    "company_name, domain, description, founded_year, employee_range, revenue_range, "
    "main_industry, sub_industry, company_type, country, city, continent, "
    "linkedin_url, specialties (as comma-separated string), "
    "technologies (as comma-separated string), "
    "total_funding_amount, total_funding_rounds, last_round_type, last_round_amount, "
    "last_round_date, ipo_status. "
    "Use empty string for any field not found."
)

# ── Step 2 prompt — static cacheable prefix + dynamic per-company suffix ──────
#
# The static prefix is sent with cache_control so Anthropic caches it after the
# first company.  Subsequent companies pay ~10 % of normal input cost for it.
# Minimum cacheable size: 1 024 tokens (Sonnet) / 2 048 tokens (Haiku).
# The web_search tool definition (~300 tokens) counts toward the threshold too.

STEP2_STATIC_PREFIX = """\
You are analyzing companies to identify whether they may have a potential interest in \
language training, business English, communication training, leadership training, \
negotiation training, team-building, onboarding, or broader employee development.

Your task is to evaluate each company based on the 10 buying signals below. \
Prioritize the signals in this order:

1. International footprint - The company has offices, teams, subsidiaries, clients, \
production sites, or operations in multiple countries or regions.

2. Foreign headquarters, parent company, or group structure - The company operates in \
one country but has its headquarters, parent company, group ownership, regional HQ, or \
reporting lines in another country.

3. Competitor signal - Check whether the company is mentioned together with any of the \
following competitors: GoFluent, Learnlight, Learnship, Voxy, Speexx, Preply, \
Fluentify, Twenix, Cambly, Berlitz, EF. This includes mentions on the company website, \
competitor case studies, testimonials, client lists, press releases, supplier pages, \
training pages, procurement documents, webinars, articles, events, job ads, or any page \
where the company name and competitor name appear together. If the company is mentioned \
together with one of these competitors, this is a very strong signal.

4. Merger, acquisition, integration, or new group ownership.

5. Explicit learning and development focus.

6. International customer base or client-facing international work.

7. Multicultural or multilingual workforce.

8. Employer branding and employee satisfaction.

9. Rapid growth, hiring, or expansion.

10. Leadership, management, sales, or negotiation-heavy roles.

Return ONLY a raw JSON object with exactly these fields and no others:
{"lead_score": "High or Medium or Low", \
"buying_signals": "comma-separated list of signal names actually supported by evidence", \
"competitor_signal": "name of competitor if found, otherwise empty string", \
"evidence": "brief description of what was found and source types", \
"likely_training_interest": \
"comma-separated list from: Language training / Business English, \
Intercultural communication, Leadership training, Negotiation training, \
Sales or client communication, Team collaboration / team-building, \
Onboarding / employee development, Broader professional training", \
"why_relevant": "brief explanation", \
"potential_buyer_function": \
"most likely buyer such as HR, Learning and Development, Talent Development, \
People and Culture, Leadership Development, Sales Enablement, Customer Success, \
Operations, Procurement"}

Do not invent evidence. Do not wrap in markdown.\
"""



# ── Field lists ───────────────────────────────────────────────────────────────

# Step 1: full Lusha-equivalent firmographics
STEP1_FIELDS = [
    "lusha_company_name",
    "lusha_domain",
    "lusha_description",
    "lusha_founded_year",
    "lusha_employee_range",
    "lusha_revenue",
    "lusha_industry",
    "lusha_sub_industry",
    "lusha_company_type",
    "lusha_country",
    "lusha_city",
    "lusha_continent",
    "lusha_linkedin_url",
    "lusha_specialties",
    "lusha_technologies",
    "lusha_total_funding_amount",
    "lusha_total_funding_rounds",
    "lusha_last_round_type",
    "lusha_last_round_amount",
    "lusha_last_round_date",
    "lusha_ipo_status",
]

# Step 2: Mingo ICP buying signals
ICP_FIELDS = [
    "icp_lead_score",
    "icp_buying_signals",
    "icp_competitor_signal",
    "icp_evidence",
    "icp_likely_training_interest",
    "icp_why_relevant",
    "icp_potential_buyer_function",
]

# Metadata added per row
META_FIELDS = [
    "enrichment_status",
    "step1_status",
    "step2_status",
    "needs_manual_review",
    "match_notes",
    "error_message",
    "step1_tokens_in",
    "step1_tokens_out",
    "step1_cost_usd",
    "step2_tokens_in",
    "step2_tokens_out",
    "step2_cost_usd",
    "total_tokens_in",
    "total_tokens_out",
    "total_cost_usd",
]

# Real Lusha API enrichment fields (prefix "lusha_api_")
LUSHA_API_FIELDS = [
    "lusha_api_company_name",
    "lusha_api_domain",
    "lusha_api_description",
    "lusha_api_founded_year",
    "lusha_api_employee_range",
    "lusha_api_revenue_range",
    "lusha_api_industry",
    "lusha_api_sub_industry",
    "lusha_api_company_type",
    "lusha_api_country",
    "lusha_api_city",
    "lusha_api_continent",
    "lusha_api_linkedin_url",
    "lusha_api_specialties",
    "lusha_api_technologies",
    "lusha_api_total_funding_amount",
    "lusha_api_total_funding_rounds",
    "lusha_api_last_round_type",
    "lusha_api_last_round_amount",
    "lusha_api_last_round_date",
    "lusha_api_ipo_status",
]

LUSHA_API_META_FIELDS = [
    "lusha_api_status",
    "lusha_api_error",
    "lusha_api_match_confidence",
    "lusha_api_needs_review",
    "lusha_api_match_notes",
]

ALL_ENRICHMENT_FIELDS = (
    LUSHA_API_FIELDS + LUSHA_API_META_FIELDS + STEP1_FIELDS + ICP_FIELDS + META_FIELDS
)

# ─────────────────────────────────────────────────────────────────────────────
# Extreme Light Mode (ELM) — zero-token, no API key, keyword-only extraction
# ─────────────────────────────────────────────────────────────────────────────

_ELM_SLUGS = ["", "/about", "/about-us", "/careers", "/jobs", "/locations", "/contact"]

_ELM_UA = "Mozilla/5.0 (compatible; CompanyResearchBot/1.0)"

_ELM_KW_INTERNATIONAL = [
    "international", "global", "worldwide", "multinational", "cross-border",
    "offices in", "presence in", "emea", "apac", "latam", "global team",
    "international team", "countries", "regions",
]
_ELM_KW_LANGUAGES = [
    "english", "french", "german", "spanish", "portuguese", "dutch", "italian",
    "chinese", "mandarin", "japanese", "korean", "arabic", "russian", "polish",
    "turkish", "swedish", "norwegian", "danish", "finnish", "hebrew",
]
_ELM_KW_HIRING = [
    "careers", "jobs", "hiring", "join us", "join our team", "open positions",
    "vacancies", "we're growing", "recruitment", "apply now", "job openings",
    "we are hiring", "current openings",
]
_ELM_KW_GROWTH = [
    "funding", "raised", "series a", "series b", "series c", "ipo",
    "acquisition", "acquired", "merger", "expansion", "hypergrowth",
    "fast-growing", "scaling",
]
_ELM_KW_TRAINING = [
    "training", "learning", "development", "upskilling", "reskilling",
    "e-learning", "coaching", "mentoring", "academy", "bootcamp",
    "certification", "corporate training", "language training",
]
_ELM_KW_OFFICES = [
    "offices", "headquarters", "locations", "branches", "regional office",
    "hub", "campus", "sites", "hq",
]
_ELM_KW_TECHNOLOGY = [
    "saas", "cloud", "platform", "software", "machine learning", "artificial intelligence",
    "automation", "digital", "engineering", "devops", "api", "data-driven",
]
_ELM_COUNTRIES = [
    "united states", "usa", "united kingdom", "uk", "germany", "france",
    "netherlands", "spain", "italy", "portugal", "belgium", "switzerland",
    "sweden", "norway", "denmark", "finland", "poland", "czech republic",
    "australia", "canada", "india", "china", "japan", "singapore", "brazil",
    "mexico", "south africa", "uae", "israel", "ireland", "austria",
]

ELM_STATUS_FIELDS = [
    "elm_fetch_status",   # "ok" / "partial" / "failed"
    "elm_pages_fetched",  # comma-separated slugs that returned 200
    "elm_pages_failed",   # comma-separated slugs that failed/non-200
    "elm_total_chars",    # total chars fetched across all pages
    "elm_error",          # any fetch-level error message
]
ELM_KEYWORD_FIELDS = [
    "elm_kw_international",
    "elm_kw_languages",
    "elm_kw_hiring",
    "elm_kw_growth",
    "elm_kw_training",
    "elm_kw_offices",
    "elm_kw_technology",
    "elm_kw_countries_found",
]
ELM_SCORE_FIELDS = [
    "elm_score_international",
    "elm_score_hiring",
    "elm_score_growth",
    "elm_score_training",
    "elm_score_technology",
    "elm_score_overall_icp",
]
ELM_ALL_FIELDS = ELM_STATUS_FIELDS + ELM_KEYWORD_FIELDS + ELM_SCORE_FIELDS

# Fields checked to decide if Step 1 returned usable data
_STEP1_DATA_FIELDS = [
    "lusha_company_name", "lusha_domain", "lusha_industry",
    "lusha_country", "lusha_description",
]

_COMPANY_HINTS = ["company", "account", "organisation", "organization", "name", "naam", "bedrijf"]
_DOMAIN_HINTS  = ["domain", "website", "url", "web", "site", "domein"]

_STATUS_LABELS = {
    "enriched_jina":                 "Enriched via Jina",
    "enriched_search":               "Enriched via Google",
    "enriched_jina_step1_only":      "Jina — Step 1 only",
    "enriched_search_step1_only":    "Google — Step 1 only",
    "enriched":                      "Enriched (both steps)",
    "step1_only":                    "Step 1 only",
    "no_data":                       "No data returned",
    "api_error":                     "API error",
    "jina_error":                    "Page fetch error",
    "web_search_fallback":           "Enriched via web search fallback",
    "cached_fallback":               "From cache (web search fallback)",
    "skipped_resume":                "Skipped (resumed)",
    "enriched_playwright":           "Enriched via browser scrape",
    "playwright_blocked":            "Browser blocked (bot detection)",
    "zero_cost_preview":             "Zero-cost preview",
}


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def clean_domain(raw: str) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    d = raw.strip().lower()
    d = re.sub(r"^https?://", "", d)
    d = re.sub(r"^www\.", "", d)
    d = d.split("/")[0].strip()
    return "" if d in {"nan", "none", ""} or " " in d else d


def normalize_url(raw: str) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    raw = raw.strip()
    if raw.lower() in {"nan", "none", ""} or " " in raw:
        return ""
    return raw if raw.startswith(("http://", "https://")) else f"https://{raw}"


def safe_filename(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = re.sub(r"[^\w\s\-.]", "", text)
    text = re.sub(r"\s+", "_", text).strip("_")
    return text[:120] or "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 debug logging helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_debug_log_dir() -> None:
    DEBUG_LOG_DIR.mkdir(exist_ok=True)


def append_to_debug_file(path: Path, content: str) -> None:
    """Append content to an existing debug file (creates it if missing)."""
    if not path or not _is_safe_debug_path(str(path)):
        return
    with path.open("a", encoding="utf-8") as fh:
        fh.write(content)


def write_debug_log(company_name: str, content: str, prefix: str = "step2_prompt") -> Path:
    ensure_debug_log_dir()
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fname = DEBUG_LOG_DIR / f"{prefix}_{safe_filename(company_name)}_{stamp}.txt"
    fname.write_text(content, encoding="utf-8")
    return fname


def append_debug_log(message: str) -> None:
    """Append a timestamped message to the in-session debug log."""
    stamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{stamp}] {message}\n"
    current = st.session_state.get("_step2_debug_log", "")
    st.session_state["_step2_debug_log"] = current + entry


def format_step2_debug_content(
    company_name: str,
    model: str,
    timestamp: str,
    provider: str,
    search_prompt: str,
    full_prompt: str,
    notes: list,
) -> str:
    sep  = "=" * 60
    thin = "-" * 40
    note_block = "\n".join(notes) if notes else "(none)"
    return (
        f"{sep}\n"
        f"COMPANY:             {company_name}\n"
        f"MODEL:               {model}\n"
        f"TIMESTAMP:           {timestamp}\n"
        f"WEB SEARCH PROVIDER: {provider}\n"
        f"\nGENERATED SEARCH PROMPT\n{thin}\n{search_prompt}\n"
        f"\nFULL CLAUDE PROMPT\n{thin}\n{full_prompt}\n"
        f"\nSTATUS / NOTES\n{thin}\n{note_block}\n"
        f"{sep}\n"
    )


def safe_json_dump(obj) -> str:
    """Serialize API response objects to indented JSON without exposing secrets."""
    if obj is None:
        return "null"
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        pass
    if hasattr(obj, "model_dump"):
        try:
            return json.dumps(obj.model_dump(), ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return json.dumps(obj.dict(), ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass
    return str(obj)


def ensure_search_output_dir() -> None:
    SEARCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def write_search_debug_file(
    company_name: str,
    provider_label: str,
    content: str,
    ext: str = "txt",
    index: int = 1,
) -> Path:
    """Write one search debug file; return the path."""
    ensure_search_output_dir()
    stamp    = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_co  = safe_filename(company_name)
    fname    = SEARCH_OUTPUT_DIR / (
        f"step2_search_{safe_co}_{provider_label}_{stamp}_search{index:02d}.{ext}"
    )
    fname.write_text(content, encoding="utf-8")
    return fname


_SAFE_DEBUG_DIRS = (
    str(DEBUG_LOG_DIR.resolve()),
    str(SEARCH_OUTPUT_DIR.resolve()),
)


def _is_safe_debug_path(p: str) -> bool:
    """Return True only when p resolves inside debug_logs/ or its subdirectories."""
    try:
        return str(Path(p).resolve()).startswith(_SAFE_DEBUG_DIRS)
    except Exception:
        return False


def build_debug_zip(file_records: list) -> bytes:
    """
    Build an in-memory ZIP of all debug files in file_records.
    Only includes files that resolve inside the debug_logs/ tree.
    Returns raw bytes suitable for st.download_button.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        seen_arcnames: set = set()
        for rec in file_records:
            fpath = rec.get("path", "")
            if not fpath or not _is_safe_debug_path(fpath):
                continue
            p = Path(fpath)
            if not p.exists():
                continue
            # Use the path relative to the parent of debug_logs/ as the archive name
            try:
                arcname = str(p.relative_to(Path(".")))
            except ValueError:
                arcname = p.name
            # Deduplicate archive names
            base, suffix = arcname, ""
            counter = 1
            while arcname + suffix in seen_arcnames:
                suffix = f"_{counter}"
                counter += 1
            arcname = arcname + suffix
            seen_arcnames.add(arcname)
            try:
                zf.write(p, arcname=arcname)
            except Exception:
                pass
    return buf.getvalue()


def _read_debug_file_safe(fpath: str, max_chars: int = 8000) -> tuple[str, bool]:
    """
    Read a debug file for preview.  Returns (content, truncated).
    Only reads files inside the safe debug dirs.
    """
    if not fpath or not _is_safe_debug_path(fpath):
        return "(file not accessible)", False
    try:
        text = Path(fpath).read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            return text[:max_chars], True
        return text, False
    except Exception as exc:
        return f"(could not read file: {exc})", False


def _debug_file_download_button(rec: dict, key_suffix: str) -> None:
    """Render a download button for one debug file record."""
    fpath = rec.get("path", "")
    if not fpath or not _is_safe_debug_path(fpath):
        return
    p = Path(fpath)
    if not p.exists():
        st.caption(f"_(file no longer on disk: `{p.name}`)_")
        return
    company = rec.get("company", "")
    label = f"⬇ Download {company} debug file" if company else f"⬇ Download {p.name}"
    try:
        data = p.read_bytes()
        st.download_button(
            label=label,
            data=data,
            file_name=p.name,
            mime="text/plain",
            key=f"dl_dbg_{key_suffix}",
        )
    except Exception:
        st.caption(f"_(download unavailable for `{p.name}`)_")


def _debug_file_preview_expander(rec: dict, key_suffix: str) -> None:
    """Render a collapsed preview expander for one debug file record."""
    fpath = rec.get("path", "")
    if not fpath:
        return
    p = Path(fpath)
    with st.expander(f"Preview: {p.name}", expanded=False):
        text, truncated = _read_debug_file_safe(fpath)
        st.code(text, language=None)
        if truncated:
            st.caption("_(preview truncated to 8 000 chars — download the full file above)_")


def _format_serper_query_debug(
    company_name: str,
    model: str,
    timestamp: str,
    query: str,
    index: int,
    total_queries: int,
    results: list,
    http_status: int,
    raw_json,
    error_str,
    dry_run: bool = False,
) -> str:
    sep  = "=" * 60
    thin = "-" * 40
    payload_safe = safe_json_dump({
        "q": query, "gl": "us", "hl": "en", "num": 10,
        # API key is intentionally excluded
    })
    if dry_run:
        result_block = "[DRY RUN: no web search output because no API call was made]"
        http_block   = "N/A (dry run)"
        raw_block    = "[DRY RUN: no raw response]"
    else:
        http_block = str(http_status) if http_status else "0 (network error)"
        if error_str:
            result_block = f"ERROR: {error_str}"
            raw_block    = safe_json_dump(raw_json) if raw_json else "(no response body)"
        else:
            raw_block = safe_json_dump(raw_json) if raw_json else "(not captured)"
            lines = []
            for r in results:
                lines.append(f"[{r.get('position', '?')}] {r.get('title', '(no title)')}")
                if r.get("date"):
                    lines.append(f"    Date: {r['date']}")
                lines.append(f"    URL:  {r.get('link', '')}")
                lines.append(f"    {r.get('snippet', '')}")
                sl = r.get("sitelinks", [])
                if sl:
                    lines.append(f"    Sitelinks: {sl}")
                lines.append("")
            result_block = "\n".join(lines).strip() or "(no results)"

    mode_note = "DRY RUN — no API call was made." if dry_run else ""
    return (
        f"{sep}\n"
        f"STEP 2 SERPER SEARCH DEBUG{' — DRY RUN' if dry_run else ''}\n"
        f"{sep}\n"
        f"COMPANY:              {company_name}\n"
        f"MODEL:                {model}\n"
        f"PROVIDER:             Serper Google Search\n"
        f"TIMESTAMP:            {timestamp} UTC\n"
        f"QUERY INDEX:          {index} of {total_queries}\n"
        f"\nSERPER QUERY\n{thin}\n{query}\n"
        f"\nSERPER REQUEST PAYLOAD (API key excluded)\n{thin}\n{payload_safe}\n"
        f"\nHTTP STATUS CODE\n{thin}\n{http_block}\n"
        f"\nRAW SERPER RESPONSE\n{thin}\n{raw_block}\n"
        f"\nEXTRACTED ORGANIC RESULTS ({len(results)} result(s))\n{thin}\n{result_block}\n"
        + (f"\nSTATUS / NOTES\n{thin}\n{mode_note or '(none)'}\n" if mode_note or error_str else "")
        + f"{sep}\n"
    )


def _format_claude_pre_debug(
    company_name: str,
    model: str,
    timestamp: str,
    prompt: str,
    dry_run: bool = False,
) -> str:
    sep  = "=" * 60
    thin = "-" * 40
    tools_safe = safe_json_dump([WEB_SEARCH_TOOL])
    mode = "DRY RUN — no API call will be made." if dry_run else ""
    return (
        f"{sep}\n"
        f"STEP 2 CLAUDE WEB SEARCH — PRE-CALL DEBUG{' (DRY RUN)' if dry_run else ''}\n"
        f"{sep}\n"
        f"COMPANY:              {company_name}\n"
        f"MODEL:                {model}\n"
        f"PROVIDER:             Claude Web Search\n"
        f"TIMESTAMP:            {timestamp} UTC\n"
        f"\nNOTE ON INTERNAL SEARCH QUERY\n{thin}\n"
        "The web_search_20250305 tool is a server-side built-in Anthropic tool.\n"
        "Claude generates the actual search query internally; it is NOT exposed\n"
        "by the API response.\n"
        "\n\"Exact internal Claude Web Search query was not exposed by the API response.\"\n"
        f"\nFULL STEP 2 PROMPT SENT TO CLAUDE\n{thin}\n{prompt}\n"
        f"\nTOOLS CONFIGURATION (secrets excluded)\n{thin}\n{tools_safe}\n"
        + (f"\nSTATUS / NOTES\n{thin}\n{mode}\n" if mode else "")
        + f"{sep}\n"
    )


def _format_claude_post_debug(
    company_name: str,
    model: str,
    timestamp: str,
    resp,
    raw_text: str,
    error_str: str = "",
    parsed_json=None,
    dry_run_skip: bool = False,
) -> str:
    sep  = "=" * 60
    thin = "-" * 40

    if dry_run_skip:
        return (
            f"\n\n{sep}\n"
            f"STEP 2 CLAUDE WEB SEARCH — POST-CALL DEBUG (NOT EXECUTED)\n"
            f"{sep}\n"
            f"COMPANY:   {company_name}\n"
            f"MODEL:     {model}\n"
            f"TIMESTAMP: {timestamp} UTC\n"
            f"\nPOST-CALL DEBUG: skipped because dry run / zero-cost preview was active.\n"
            f"No Anthropic API call was made.\n"
            f"{sep}\n"
        )

    if error_str:
        return (
            f"\n\n{sep}\n"
            f"STEP 2 CLAUDE WEB SEARCH — POST-CALL DEBUG\n"
            f"{sep}\n"
            f"COMPANY:   {company_name}\n"
            f"MODEL:     {model}\n"
            f"TIMESTAMP: {timestamp} UTC\n"
            f"\nERROR\n{thin}\n{error_str}\n"
            f"{sep}\n"
        )

    stop_reason = getattr(resp, "stop_reason", "unknown") if resp else "unknown"
    usage       = getattr(resp, "usage", None)
    in_tok      = getattr(usage, "input_tokens",  "?") if usage else "?"
    out_tok     = getattr(usage, "output_tokens", "?") if usage else "?"

    tool_use_blocks    = []
    tool_result_blocks = []
    web_search_blocks  = []
    citation_blocks    = []
    other_blocks       = []

    if resp and hasattr(resp, "content"):
        for blk in resp.content:
            btype = getattr(blk, "type", "")
            if btype == "tool_use":
                tool_use_blocks.append(blk)
            elif btype == "tool_result":
                tool_result_blocks.append(blk)
            elif btype in ("web_search_result", "server_tool_use"):
                web_search_blocks.append(blk)
            elif btype == "text":
                pass  # already in raw_text
            else:
                other_blocks.append(blk)
        # Collect citations if present on text blocks
        for blk in resp.content:
            if getattr(blk, "type", "") == "text":
                cits = getattr(blk, "citations", []) or []
                citation_blocks.extend(cits)

    def _blk_section(label, blocks):
        if not blocks:
            return f"\n{label}\n{thin}\n(none)\n"
        lines = [f"\n{label}\n{thin}"]
        for b in blocks:
            lines.append(safe_json_dump(b))
        return "\n".join(lines) + "\n"

    citation_section = ""
    if citation_blocks:
        citation_section = f"\nCITATIONS / SOURCE REFERENCES\n{thin}\n"
        for c in citation_blocks:
            citation_section += safe_json_dump(c) + "\n"
    else:
        citation_section = f"\nCITATIONS / SOURCE REFERENCES\n{thin}\n(none)\n"

    raw_resp_dump = safe_json_dump(resp) if resp else "(not available)"

    parsed_section = ""
    if parsed_json is not None:
        parsed_section = (
            f"\nPARSED JSON OUTPUT\n{thin}\n"
            + safe_json_dump(parsed_json) + "\n"
        )
    else:
        parsed_section = f"\nPARSED JSON OUTPUT\n{thin}\n(not available or parse failed)\n"

    return (
        f"\n\n{sep}\n"
        f"STEP 2 CLAUDE WEB SEARCH — POST-CALL DEBUG\n"
        f"{sep}\n"
        f"COMPANY:              {company_name}\n"
        f"MODEL:                {model}\n"
        f"PROVIDER:             Claude Web Search\n"
        f"TIMESTAMP:            {timestamp} UTC\n"
        f"STOP REASON:          {stop_reason}\n"
        f"USAGE:                input_tokens={in_tok}, output_tokens={out_tok}\n"
        f"\nRESPONSE TEXT\n{thin}\n{raw_text or '(empty)'}\n"
        + _blk_section("TOOL USE BLOCKS", tool_use_blocks)
        + _blk_section("TOOL RESULT BLOCKS", tool_result_blocks)
        + _blk_section("WEB SEARCH / SOURCE BLOCKS", web_search_blocks)
        + citation_section
        + parsed_section
        + f"\nRAW RESPONSE OBJECT\n{thin}\n{raw_resp_dump}\n"
        + f"{sep}\n"
    )


def str_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


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


def detect_columns(df: pd.DataFrame) -> tuple:
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


def calc_cost(in_tok: int, out_tok: int) -> float:
    return (in_tok * _COST_INPUT_PER_M + out_tok * _COST_OUTPUT_PER_M) / 1_000_000


def _parse_json_response(text: str) -> dict:
    """
    Extract and parse JSON from Claude's response.
    1. Try stripping markdown fences and parsing directly.
    2. Fall back to pulling the first {...} block via regex.
    Raises ValueError/JSONDecodeError when no valid JSON is found.
    """
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass
    # Extract first {...} block from raw text (handles prose wrappers)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group())
    raise ValueError(f"No JSON object found in response (first 200 chars): {text[:200]!r}")


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


def _delete_cache(cache_key: str) -> None:
    p = CACHE_DIR / f"{safe_filename(cache_key)}.json"
    p.unlink(missing_ok=True)


def get_cache_count() -> int:
    return len(list(CACHE_DIR.glob("*.json"))) if CACHE_DIR.exists() else 0


def list_cache_files() -> list:
    return sorted(CACHE_DIR.glob("*.json")) if CACHE_DIR.exists() else []


# ─────────────────────────────────────────────────────────────────────────────
# Jina AI  ← page / search fetching
# ─────────────────────────────────────────────────────────────────────────────

_JINA_HEADERS = {
    "Accept": "text/plain",
    "X-Return-Format": "text",
    "x-respond-with": "text",
}
_JINA_CHAR_LIMIT = 6_000   # cap Jina content to save tokens
_JINA_ABOUT_SLUGS  = ("/about-us", "/about")
_JINA_MIN_CONTENT  = 500   # fewer chars than this → treat as failed Jina fetch
_JINA_RETRY_WAITS  = (30, 60, 120)  # backoff seconds on 429

_BOT_DETECTION_KEYWORDS = (
    "automatically identified", "security system", "bot", "captcha",
    "cloudflare", "datadome",
)


class JinaRateLimitRetry(Exception):
    """Raised to signal the UI that we're waiting on a 429 before retrying."""
    def __init__(self, wait: int, company: str):
        self.wait    = wait
        self.company = company
        super().__init__(f"Rate limited — waiting {wait}s for {company}")


def _jina_get(url: str, company_hint: str = "") -> str:
    """
    GET a single URL via Jina Reader.
    On 429: waits 30 s → 60 s → 120 s (3 retries).
    Raises JinaRateLimitRetry to let the UI display the wait message.
    Raises requests.HTTPError on non-429 failures.
    Returns up to _JINA_CHAR_LIMIT characters.
    """
    for attempt, wait in enumerate(_JINA_RETRY_WAITS):
        resp = requests.get(
            f"{JINA_READER_URL}{url}",
            headers=_JINA_HEADERS,
            timeout=30,
        )
        if resp.status_code != 429:
            break
        raise JinaRateLimitRetry(wait, company_hint)
    resp.raise_for_status()
    return resp.text[:_JINA_CHAR_LIMIT]


def _jina_get_with_retry(url: str, company_hint: str = "") -> str:
    """Wrap _jina_get to actually sleep and retry when JinaRateLimitRetry is raised."""
    for attempt, wait in enumerate(_JINA_RETRY_WAITS):
        try:
            return _jina_get(url, company_hint)
        except JinaRateLimitRetry as exc:
            # Track retry count for the live counter
            st.session_state["_jina_retry_count"] = (
                st.session_state.get("_jina_retry_count", 0) + 1
            )
            st.session_state["_last_retry_msg"] = (
                f"⏳ Rate limit — waiting {exc.wait}s for {company_hint or url}…"
            )
            time.sleep(wait)
    # Final attempt — let HTTPError propagate
    return _jina_get(url, company_hint)


def fetch_via_jina_reader(url: str, company_hint: str = "") -> str:
    """
    Fetch the homepage AND about-us page via Jina Reader; return whichever has
    more content.  Raises requests.HTTPError when content is below
    _JINA_MIN_CONTENT so the caller can fall through to the next tier.
    """
    base = url.rstrip("/")
    best = ""

    # Try homepage
    try:
        text = _jina_get_with_retry(url, company_hint)
        if len(text) > len(best):
            best = text
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else 0
        if code not in (403, 503):
            raise

    # Try about-us slugs
    for slug in _JINA_ABOUT_SLUGS:
        try:
            text = _jina_get_with_retry(base + slug, company_hint)
            if len(text) > len(best):
                best = text
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if code not in (403, 503, 404):
                raise

    if len(best) < _JINA_MIN_CONTENT:
        # Simulate a 404 so the caller falls through to the search tier
        raise requests.HTTPError(
            f"Jina returned only {len(best)} chars (< {_JINA_MIN_CONTENT})",
            response=None,
        )
    return best


def fetch_via_jina_search(query: str) -> str:
    for attempt, wait in enumerate(_JINA_RETRY_WAITS):
        resp = requests.get(
            f"{JINA_SEARCH_URL}{quote(query)}",
            headers=_JINA_HEADERS,
            timeout=30,
        )
        if resp.status_code != 429:
            break
        time.sleep(wait)
    resp.raise_for_status()
    return resp.text[:_JINA_CHAR_LIMIT]


# ─────────────────────────────────────────────────────────────────────────────
# Extreme Light Mode — page fetching and keyword extraction
# ─────────────────────────────────────────────────────────────────────────────

def _elm_fetch_pages(base_url: str, domain: str) -> tuple[dict, list, list]:
    """
    Fetch _ELM_SLUGS pages via requests + BeautifulSoup.
    Returns (pages_dict, fetched_slugs, failed_slugs).
    Caches by domain so repeated runs skip HTTP calls.
    """
    ck = f"elm_{domain}"
    cached = load_cache(ck)
    if cached and "pages" in cached:
        return (
            cached["pages"],
            cached.get("fetched", []),
            cached.get("failed", []),
        )

    base    = base_url.rstrip("/")
    headers = {"User-Agent": _ELM_UA}
    pages: dict   = {}
    fetched: list = []
    failed: list  = []

    for slug in _ELM_SLUGS:
        target = base if slug == "" else f"{base}{slug}"
        label  = slug if slug else "/"
        try:
            resp = requests.get(target, headers=headers, timeout=12, allow_redirects=True)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                text = soup.get_text(separator=" ", strip=True)
                pages[label] = text[:8_000]
                fetched.append(label)
            else:
                failed.append(f"{label}({resp.status_code})")
        except Exception as exc:
            failed.append(f"{label}(err:{type(exc).__name__})")

    save_cache(ck, {"pages": pages, "fetched": fetched, "failed": failed})
    return pages, fetched, failed


def _elm_count(text: str, keywords: list) -> int:
    tl = text.lower()
    return sum(1 for kw in keywords if kw in tl)


def _elm_find(text: str, keywords: list) -> list:
    tl = text.lower()
    return [kw for kw in keywords if kw in tl]


def _elm_score(count: int, per_point: float = 2.0) -> int:
    """Convert raw keyword count to 0–10 score."""
    return min(round(count / per_point * 10), 10)


def _elm_extract_signals(pages: dict, fetched: list, failed: list) -> dict:
    all_text = " ".join(pages.values())

    kw_intl     = _elm_count(all_text, _ELM_KW_INTERNATIONAL)
    langs       = _elm_find(all_text, _ELM_KW_LANGUAGES)
    kw_hiring   = _elm_count(all_text, _ELM_KW_HIRING)
    kw_growth   = _elm_count(all_text, _ELM_KW_GROWTH)
    kw_training = _elm_count(all_text, _ELM_KW_TRAINING)
    kw_offices  = _elm_count(all_text, _ELM_KW_OFFICES)
    kw_tech     = _elm_count(all_text, _ELM_KW_TECHNOLOGY)
    countries   = _elm_find(all_text, _ELM_COUNTRIES)

    s_intl     = min(_elm_score(kw_intl, 1.5) + min(len(countries), 4), 10)
    s_hiring   = _elm_score(kw_hiring,   1.5)
    s_growth   = _elm_score(kw_growth,   1.5)
    s_training = _elm_score(kw_training, 1.5)
    s_tech     = _elm_score(kw_tech,     2.0)
    s_overall  = round(
        s_intl     * 0.30
        + s_hiring   * 0.20
        + s_training * 0.30
        + s_growth   * 0.10
        + s_tech     * 0.10,
        1,
    )

    n_fetched = len(fetched)
    n_failed  = len(failed)
    if n_fetched == 0:
        fetch_status = "failed"
    elif n_failed > n_fetched:
        fetch_status = "partial"
    else:
        fetch_status = "ok"

    return {
        # Status
        "elm_fetch_status":  fetch_status,
        "elm_pages_fetched": ", ".join(fetched),
        "elm_pages_failed":  ", ".join(failed),
        "elm_total_chars":   sum(len(v) for v in pages.values()),
        "elm_error":         "",
        # Keywords
        "elm_kw_international":  kw_intl,
        "elm_kw_languages":      ", ".join(sorted(set(langs))),
        "elm_kw_hiring":         kw_hiring,
        "elm_kw_growth":         kw_growth,
        "elm_kw_training":       kw_training,
        "elm_kw_offices":        kw_offices,
        "elm_kw_technology":     kw_tech,
        "elm_kw_countries_found": ", ".join(sorted(set(countries))),
        # Scores
        "elm_score_international": s_intl,
        "elm_score_hiring":        s_hiring,
        "elm_score_growth":        s_growth,
        "elm_score_training":      s_training,
        "elm_score_technology":    s_tech,
        "elm_score_overall_icp":   s_overall,
    }


def enrich_one_row_light(company_name: str, raw_url: str) -> tuple:
    """
    Extreme Light Mode row enrichment — no Claude API, no tokens.
    Returns (elm_fields_dict, debug_record_dict).
    """
    empty = {f: "" for f in ELM_ALL_FIELDS}
    url   = normalize_url(raw_url) if raw_url else ""

    if not url:
        row = {**empty,
               "elm_fetch_status": "failed",
               "elm_error": "No URL provided"}
        return row, {"company": company_name, "url": raw_url, "status": "no_url"}

    domain = clean_domain(url)
    try:
        pages, fetched, failed = _elm_fetch_pages(url, domain)
    except Exception as exc:
        row = {**empty,
               "elm_fetch_status": "failed",
               "elm_error": str(exc)[:200]}
        return row, {"company": company_name, "url": url, "status": "fetch_error", "error": str(exc)}

    signals = _elm_extract_signals(pages, fetched, failed)
    dbg = {
        "company":       company_name,
        "url":           url,
        "domain":        domain,
        "status":        signals["elm_fetch_status"],
        "pages_fetched": signals["elm_pages_fetched"],
        "total_chars":   signals["elm_total_chars"],
    }
    return signals, dbg


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Basic extraction via Claude (Jina page → Claude)
# ─────────────────────────────────────────────────────────────────────────────

def _claude_extract(webpage_text: str, api_key: str,
                    model_id: str = MODEL_STEP1) -> tuple:
    """
    Send page text to Claude for structured extraction.
    Returns (raw_fields_dict, input_tokens, output_tokens).
    """
    client    = anthropic.Anthropic(api_key=api_key)
    truncated = webpage_text[:_JINA_CHAR_LIMIT]
    msg = client.messages.create(
        model=model_id,
        max_tokens=1024,
        messages=[{"role": "user", "content": f"{_STEP1_PROMPT}\n\n{truncated}"}],
    )
    return (
        _parse_json_response(msg.content[0].text),
        msg.usage.input_tokens,
        msg.usage.output_tokens,
    )


def _map_step1_fields(raw: dict, source_url: str) -> dict:
    domain  = clean_domain(source_url) if source_url else ""
    # Prefer the domain Claude extracted over the source URL when available
    if not domain:
        domain = clean_domain(str(raw.get("domain") or ""))
    country = str(raw.get("country")   or "").strip()
    city    = str(raw.get("city")      or "").strip()

    def s(key):
        return str(raw.get(key) or "").strip()

    return {
        "lusha_company_name":         s("company_name"),
        "lusha_domain":               domain,
        "lusha_description":          s("description"),
        "lusha_founded_year":         s("founded_year"),
        "lusha_employee_range":       s("employee_range"),
        "lusha_revenue":              s("revenue_range"),
        "lusha_industry":             s("main_industry"),
        "lusha_sub_industry":         s("sub_industry"),
        "lusha_company_type":         s("company_type"),
        "lusha_country":              country,
        "lusha_city":                 city,
        "lusha_continent":            s("continent"),
        "lusha_linkedin_url":         s("linkedin_url"),
        "lusha_specialties":          s("specialties"),
        "lusha_technologies":         s("technologies"),
        "lusha_total_funding_amount": s("total_funding_amount"),
        "lusha_total_funding_rounds": s("total_funding_rounds"),
        "lusha_last_round_type":      s("last_round_type"),
        "lusha_last_round_amount":    s("last_round_amount"),
        "lusha_last_round_date":      s("last_round_date"),
        "lusha_ipo_status":           s("ipo_status"),
    }


def _step1_has_data(fields: dict) -> bool:
    return any(fields.get(f, "") for f in _STEP1_DATA_FIELDS)


_STEP1_FALLBACK_PROMPT_TMPL = (
    "Find company information about {company_name} ({url}). "
    "Extract: industry, employee count, founding year, headquarters location, "
    "description, international presence, specialties. "
    "Return ONLY JSON with fields: company_name, description, main_industry, "
    "sub_industry, employee_range, revenue_range, founded_year, company_type, "
    "country, city, linkedin_url, specialties, continent, technologies, "
    "total_funding_amount, total_funding_rounds, last_round_type, last_round_amount, "
    "last_round_date, ipo_status. Use empty string for any field not found."
)


_PW_DBG_SKIP = {"playwright_attempted": False, "playwright_result": "skipped"}


def run_step1(
    url: str,
    company_name: str,
    api_key: str,
    delay: float,
    use_playwright: bool = True,
    model_step1: str = MODEL_STEP1,
) -> tuple:
    """
    Three-tier Step 1 enrichment:
      Tier 1a — Jina direct scrape       → status 'enriched_jina'
      Tier 1b — human Playwright scrape  → status 'enriched_playwright'
      Tier 2  — Claude web_search        → status 'enriched_search'
    Returns (step1_fields, raw_json, in_tok, out_tok, status, error_msg, pw_debug).
    pw_debug: {"playwright_attempted": bool, "playwright_result": str}
    """
    source_url = normalize_url(url) if url else ""
    jina_err   = ""
    total_in   = total_out = 0

    # ── Tier 1a: Jina direct scrape ───────────────────────────────────────────
    if source_url:
        ck = f"step1_url_{source_url}"
        cached = load_cache(ck)
        if cached is not None:
            fields = _map_step1_fields(cached.get("claude_data", {}), source_url)
            if _step1_has_data(fields):
                return (fields, cached,
                        int(cached.get("tokens_in",  0) or 0),
                        int(cached.get("tokens_out", 0) or 0),
                        "enriched_jina", "", _PW_DBG_SKIP)
            _delete_cache(ck)

        try:
            time.sleep(delay)
            text = fetch_via_jina_reader(source_url, company_hint=company_name)
            raw_fields, in_t, out_t = _claude_extract(text, api_key, model_id=model_step1)
            total_in  += in_t
            total_out += out_t
            payload = {"claude_data": raw_fields, "tokens_in": in_t, "tokens_out": out_t}
            save_cache(ck, payload)
            fields = _map_step1_fields(raw_fields, source_url)
            if _step1_has_data(fields):
                return (fields, payload, total_in, total_out, "enriched_jina", "", _PW_DBG_SKIP)
            jina_err = "Jina page fetched but Claude found no usable data"
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            jina_err = f"Jina HTTP {code}: {str(e)[:120]}"
        except (json.JSONDecodeError, ValueError) as e:
            jina_err = f"Jina parse error: {e}"
        except anthropic.APIError as e:
            return ({}, {}, total_in, total_out, "api_error", f"Claude API: {e}", _PW_DBG_SKIP)
        except Exception as e:
            jina_err = str(e)

    # ── Tier 1b: Human Playwright scrape ──────────────────────────────────────
    _pw_attempted = False
    _pw_result    = "skipped"
    if source_url and jina_err and use_playwright and _PLAYWRIGHT_AVAILABLE:
        _pw_attempted = True
        ck_pw  = f"step1_playwright_{source_url}"
        cached_pw = load_cache(ck_pw)
        if cached_pw is not None:
            fields = _map_step1_fields(cached_pw.get("claude_data", {}), source_url)
            if _step1_has_data(fields):
                _pw_result = "success"
                return (fields, cached_pw,
                        int(cached_pw.get("tokens_in",  0) or 0),
                        int(cached_pw.get("tokens_out", 0) or 0),
                        "enriched_playwright", "",
                        {"playwright_attempted": True, "playwright_result": "success"})
            _delete_cache(ck_pw)

        try:
            pw_res = scrape_with_human_behaviour(source_url, max_chars=_JINA_CHAR_LIMIT)
            if pw_res.get("success") and len(pw_res.get("text", "")) >= _JINA_MIN_CONTENT:
                pw_text  = pw_res["text"]
                pw_lower = pw_text[:2000].lower()
                if any(kw in pw_lower for kw in _BOT_DETECTION_KEYWORDS):
                    _pw_result = "blocked"
                    jina_err  += " | playwright: bot-detected"
                else:
                    raw_fields, in_t, out_t = _claude_extract(pw_text, api_key, model_id=model_step1)
                    total_in  += in_t
                    total_out += out_t
                    payload = {"claude_data": raw_fields, "tokens_in": in_t, "tokens_out": out_t}
                    save_cache(ck_pw, payload)
                    fields = _map_step1_fields(raw_fields, source_url)
                    if _step1_has_data(fields):
                        _pw_result = "success"
                        return (fields, payload, total_in, total_out, "enriched_playwright", "",
                                {"playwright_attempted": True, "playwright_result": "success"})
                    _pw_result = "failed"
            else:
                _pw_result = "failed"
        except anthropic.APIError as e:
            return ({}, {}, total_in, total_out, "api_error", f"Claude API: {e}",
                    {"playwright_attempted": True, "playwright_result": "failed"})
        except Exception as e:
            _pw_result = "failed"
            jina_err  += f" | playwright error: {str(e)[:80]}"

    _pw_dbg = {"playwright_attempted": _pw_attempted, "playwright_result": _pw_result}

    # ── Tier 2: Claude web_search fallback ────────────────────────────────────
    target = source_url or company_name
    if not target:
        return ({}, {}, total_in, total_out, "no_data", "No URL or company name provided", _pw_dbg)

    ck = f"step1_fallback_{target}"
    cached = load_cache(ck)
    if cached is not None:
        fields = _map_step1_fields(cached.get("claude_data", {}), url)
        if _step1_has_data(fields):
            return (fields, cached,
                    int(cached.get("tokens_in",  0) or 0),
                    int(cached.get("tokens_out", 0) or 0),
                    "enriched_search", "", _pw_dbg)
        _delete_cache(ck)

    try:
        prompt = _STEP1_FALLBACK_PROMPT_TMPL.format(
            company_name=company_name or target,
            url=target,
        )
        raw_text, in_t, out_t = _claude_web_search_loop(prompt, api_key, model_id=model_step1)
        total_in  += in_t
        total_out += out_t
        raw_fields = _parse_json_response(raw_text)
        payload    = {"claude_data": raw_fields, "tokens_in": in_t, "tokens_out": out_t}
        save_cache(ck, payload)
        fields = _map_step1_fields(raw_fields, url)
        if _step1_has_data(fields):
            return (fields, payload, total_in, total_out, "enriched_search", "", _pw_dbg)
        return ({}, {}, total_in, total_out, "no_data",
                f"Google search returned no usable data. Jina: {jina_err}", _pw_dbg)
    except (json.JSONDecodeError, ValueError) as e:
        return ({}, {}, total_in, total_out, "no_data",
                f"Search parse error: {e}. Jina: {jina_err}", _pw_dbg)
    except anthropic.APIError as e:
        return ({}, {}, total_in, total_out, "api_error", f"Claude API: {e}", _pw_dbg)
    except Exception as e:
        return ({}, {}, total_in, total_out, "no_data",
                f"Search error: {e}. Jina: {jina_err}", _pw_dbg)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — ICP signals via Claude web_search
# ─────────────────────────────────────────────────────────────────────────────

_ICP_EMPTY = {f: "" for f in ICP_FIELDS}


def _claude_web_search_loop(prompt: str, api_key: str, model_id: str = None) -> tuple:
    """
    Run Claude with web_search_20250305 (server-side built-in tool).
    Anthropic executes the search automatically — no tool_result needed.
    Returns (final_text, total_input_tokens, total_output_tokens).
    """
    if model_id is None:
        raise ValueError("model_id must be provided to _claude_web_search_loop")

    client = anthropic.Anthropic(api_key=api_key)

    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=model_id,
                max_tokens=2048,
                tools=[WEB_SEARCH_TOOL],
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(
                getattr(b, "text", "")
                for b in resp.content
                if getattr(b, "type", "") == "text"
            ).strip()
            return text, resp.usage.input_tokens, resp.usage.output_tokens

        except anthropic.RateLimitError as e:
            wait = 30
            if hasattr(e, "response") and e.response is not None:
                wait = int(e.response.headers.get("retry-after", 30))
            time.sleep(wait)

        except anthropic.APIStatusError as e:
            if e.status_code == 529:
                time.sleep(60)
            else:
                raise

    return "", 0, 0


def _claude_web_search_full(prompt: str, api_key: str, model_id: str) -> tuple:
    """
    Like _claude_web_search_loop but also returns the raw response object.
    Returns (text, in_t, out_t, resp_or_None).
    """
    client = anthropic.Anthropic(api_key=api_key)
    for _ in range(3):
        try:
            resp = client.messages.create(
                model=model_id,
                max_tokens=2048,
                tools=[WEB_SEARCH_TOOL],
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(
                getattr(b, "text", "")
                for b in resp.content
                if getattr(b, "type", "") == "text"
            ).strip()
            return text, resp.usage.input_tokens, resp.usage.output_tokens, resp
        except anthropic.RateLimitError as e:
            wait = 30
            if hasattr(e, "response") and e.response is not None:
                wait = int(e.response.headers.get("retry-after", 30))
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            if e.status_code == 529:
                time.sleep(60)
            else:
                raise
    return "", 0, 0, None


# ─────────────────────────────────────────────────────────────────────────────
# Serper Google Search helpers (Step 2 alternative provider)
# ─────────────────────────────────────────────────────────────────────────────

def _build_serper_queries(company_name: str, target: str) -> list:
    """Return 2 focused Google-style queries covering the main Step 2 ICP signals."""
    name = company_name or target
    return [
        f'"{name}" international offices global headquarters language training',
        f'"{name}" learning development corporate training language employees hiring',
    ]


def _call_serper(query: str, serper_key: str, timeout: int = 15):
    """
    POST one query to the Serper API.
    Returns (organic_results, http_status, raw_json_or_None, error_str_or_None).
    organic_results is a list of dicts; empty on error.
    Never logs or returns the serper_key.
    """
    raw_json    = None
    http_status = 0
    try:
        resp = requests.post(
            SERPER_SEARCH_URL,
            headers={"X-API-KEY": serper_key, "Content-Type": "application/json"},
            json={"q": query, "gl": "us", "hl": "en", "num": 10},
            timeout=timeout,
        )
        http_status = resp.status_code
        resp.raise_for_status()
        raw_json = resp.json()
    except requests.Timeout:
        return [], 0, None, "Serper API timed out"
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else 0
        http_status = code
        try:
            raw_json = e.response.json() if e.response is not None else None
        except Exception:
            pass
        if code == 403:
            return [], code, raw_json, "Serper API key rejected (403)"
        if code == 429:
            return [], code, raw_json, "Serper quota exceeded (429)"
        return [], code, raw_json, f"Serper HTTP {code}: {e}"
    except (json.JSONDecodeError, ValueError) as e:
        return [], http_status, None, f"Serper returned invalid JSON: {e}"
    except Exception as e:
        return [], 0, None, f"Serper error: {e}"

    results = [
        {
            "title":     item.get("title", ""),
            "link":      item.get("link", ""),
            "snippet":   item.get("snippet", ""),
            "position":  item.get("position", ""),
            "date":      item.get("date", ""),
            "sitelinks": item.get("sitelinks", []),
        }
        for item in (raw_json or {}).get("organic", [])
    ]
    return results, http_status, raw_json, None


def _format_serper_results(results: list) -> str:
    """Format Serper organic results into a readable text block for Claude."""
    if not results:
        return "(No web search results were found.)"
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.get('title', '(no title)')}")
        if r.get("date"):
            lines.append(f"    Date: {r['date']}")
        lines.append(f"    URL: {r.get('link', '')}")
        lines.append(f"    {r.get('snippet', '')}")
        lines.append("")
    return "\n".join(lines).strip()


def run_step2_serper(
    url: str,
    company_name: str,
    api_key: str,
    serper_key: str,
    delay: float,
    model_step2: str = MODEL_STEP2,
    _debug_callback=None,
    dry_run: bool = False,
) -> tuple:
    """
    Step 2 via Serper Google Search + Claude analysis (no web_search tool).
    Returns the same 8-tuple as run_step2.
    dry_run=True: generate queries and Claude prompt without calling any API.
    """
    def _dlog(msg: str) -> None:
        if _debug_callback:
            _debug_callback("status", msg=msg)

    target = normalize_url(url) if url else company_name
    if not target:
        return (_ICP_EMPTY.copy(), {}, 0, 0, "no_input", "No URL or company name", 0, 0)

    if not dry_run:
        ck = f"step2_serper_{target}"
        cached = load_cache(ck)
        if cached is not None:
            icp = cached.get("icp_data", {})
            if any(icp.get(f, "") for f in ICP_FIELDS[:3]):
                in_t  = int(cached.get("tokens_in", 0) or 0)
                out_t = int(cached.get("tokens_out", 0) or 0)
                _dlog(f"Using cached Serper Step 2 result for {company_name}")
                return (_extract_icp_fields(icp), cached, in_t, out_t, "cached", "", 0, 0)
            _delete_cache(ck)

    # ── Serper searches ───────────────────────────────────────────────────────
    queries = _build_serper_queries(company_name, target)
    _dlog(f"Generating Serper queries for {company_name}")

    # DRY RUN GUARD: do not call Serper or Anthropic in prompt preview mode
    if dry_run:
        _dry_placeholder = "[DRY RUN: Serper results would be inserted here]"
        _dry_instruction = (
            f"Now analyze this company based on the web search results provided below.\n\n"
            f"Company: {target}\n\n"
            f"Web search results (retrieved via Serper Google Search):\n"
            f"{_dry_placeholder}\n\n"
            "Base your analysis ONLY on the search results and company information above. "
            "Do not claim to have searched the web yourself."
        )
        _dry_full_prompt = STEP2_STATIC_PREFIX + f"\n\n{_dry_instruction}"
        _dlog(f"DRY RUN: Serper queries that would be sent: {queries}")
        _dlog(f"DRY RUN: skipping Serper + Anthropic calls for {company_name}")
        if _debug_callback:
            _ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            for _qi, _q in enumerate(queries, 1):
                try:
                    _dry_content = _format_serper_query_debug(
                        company_name, model_step2, _ts, _q, _qi, len(queries),
                        [], 0, None, None, dry_run=True,
                    )
                    _fpath = write_search_debug_file(
                        company_name, "serper_google_search", _dry_content, index=_qi,
                    )
                    _debug_callback(
                        "search_output",
                        company=company_name,
                        provider=STEP2_PROVIDER_SERPER,
                        query=_q,
                        result_count=0,
                        top_results=[],
                        debug_file=str(_fpath),
                        dry_run=True,
                    )
                except Exception:
                    pass
            _debug_callback(
                "prompt",
                company=company_name,
                model=model_step2,
                provider=STEP2_PROVIDER_SERPER,
                search_prompt=(
                    "DRY RUN — Serper queries that would be sent:\n"
                    + "\n".join(f"  • {q}" for q in queries)
                ),
                full_prompt=_dry_full_prompt,
                notes=[
                    "DRY RUN ACTIVE: no Anthropic or Serper API calls are being made.",
                    f"Serper queries that would be sent: {queries}",
                    f"Selected model: {model_step2}",
                ],
                dry_run=True,
                queries=queries,
            )
        return (_ICP_EMPTY.copy(), {}, 0, 0, "dry_run", "DRY RUN: no API call made", 0, 0)

    all_hits: list = []
    for qi, q in enumerate(queries, 1):
        _dlog(f"Serper query: {q}")
        hits, http_status, raw_json, err_str = _call_serper(q, serper_key)
        if err_str:
            _dlog(f"Serper warning — {err_str}")
        else:
            all_hits.extend(hits)
            _dlog(f"Serper returned {len(hits)} results")
        if _debug_callback:
            _ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            try:
                _content = _format_serper_query_debug(
                    company_name, model_step2, _ts, q, qi, len(queries),
                    hits, http_status, raw_json, err_str,
                )
                _fpath = write_search_debug_file(
                    company_name, "serper_google_search", _content, index=qi,
                )
                _debug_callback(
                    "search_output",
                    company=company_name,
                    provider=STEP2_PROVIDER_SERPER,
                    query=q,
                    result_count=len(hits),
                    top_results=hits[:3],
                    debug_file=str(_fpath),
                    dry_run=False,
                )
            except Exception:
                pass

    # Deduplicate by URL
    seen: set = set()
    deduped: list = []
    for r in all_hits:
        lnk = r.get("link", "")
        if lnk and lnk not in seen:
            seen.add(lnk)
            deduped.append(r)

    _dlog(f"Total unique Serper results for {company_name}: {len(deduped)}")

    # ── Build Claude prompt ────────────────────────────────────────────────────
    results_text = _format_serper_results(deduped)
    search_instruction = (
        f"Now analyze this company based on the web search results provided below.\n\n"
        f"Company: {target}\n\n"
        f"Web search results (retrieved via Serper Google Search):\n"
        f"{results_text}\n\n"
        "Base your analysis ONLY on the search results and company information above. "
        "Do not claim to have searched the web yourself."
    )
    full_prompt = STEP2_STATIC_PREFIX + f"\n\n{search_instruction}"

    _dlog(f"Selected model: {model_step2}")

    if _debug_callback:
        _debug_callback(
            "prompt",
            company=company_name,
            model=model_step2,
            provider=STEP2_PROVIDER_SERPER,
            search_prompt=f"Queries: {queries}\n\nResults count: {len(deduped)}",
            full_prompt=full_prompt,
            notes=[
                f"Serper queries used: {queries}",
                f"Unique results retrieved: {len(deduped)}",
                f"Selected model: {model_step2}",
            ],
        )

    # ── Claude analysis (no web_search tool — we supply the context) ──────────
    _STRICT_SUFFIX = (
        "\n\nReply with ONLY a JSON object, no explanation, no markdown, no backticks."
    )
    client = anthropic.Anthropic(api_key=api_key)

    _dlog(f"Calling Claude for Serper analysis of {company_name}")
    try:
        time.sleep(delay)
        resp = client.messages.create(
            model=model_step2,
            max_tokens=2048,
            messages=[{"role": "user", "content": full_prompt}],
        )
        raw_text = "".join(
            getattr(b, "text", "") for b in resp.content
            if getattr(b, "type", "") == "text"
        ).strip()
        in_t  = resp.usage.input_tokens
        out_t = resp.usage.output_tokens
        _dlog(f"Received Claude response for {company_name}")

        # Parse (with one retry on failure)
        _icp_raw = None
        try:
            _dlog(f"Parsing Step 2 Serper response for {company_name}")
            _icp_raw = _parse_json_response(raw_text)
        except (json.JSONDecodeError, ValueError):
            _dlog(f"Parse failed — retrying with strict suffix for {company_name}")
            time.sleep(delay)
            resp2 = client.messages.create(
                model=model_step2,
                max_tokens=2048,
                messages=[{"role": "user", "content": full_prompt + _STRICT_SUFFIX}],
            )
            raw_text2 = "".join(
                getattr(b, "text", "") for b in resp2.content
                if getattr(b, "type", "") == "text"
            ).strip()
            in_t  += resp2.usage.input_tokens
            out_t += resp2.usage.output_tokens
            _dlog(f"Parsing Step 2 Serper retry response for {company_name}")
            _icp_raw = _parse_json_response(raw_text2)

        # Write Claude analysis post-call debug file (after parsing so it includes parsed JSON)
        if _debug_callback:
            _ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            try:
                _post_content = _format_claude_post_debug(
                    company_name, model_step2, _ts, resp, raw_text,
                    parsed_json=_icp_raw,
                )
                _serper_resp_path = write_search_debug_file(
                    company_name, "serper_google_search_claude_response",
                    _post_content, index=len(queries) + 1,
                )
                _debug_callback(
                    "search_output",
                    company=company_name,
                    provider=STEP2_PROVIDER_SERPER,
                    query="[Claude analysis of Serper results]",
                    result_count=0,
                    top_results=[],
                    debug_file=str(_serper_resp_path),
                    dry_run=False,
                )
            except Exception:
                pass

        payload = {"icp_data": _icp_raw, "tokens_in": in_t, "tokens_out": out_t}
        save_cache(ck, payload)
        _dlog(f"Finished Step 2 (Serper) for {company_name}")
        return (_extract_icp_fields(_icp_raw), payload, in_t, out_t, "ok", "", 0, 0)

    except (json.JSONDecodeError, ValueError) as e:
        _dlog(f"Step 2 Serper parse error for {company_name}: {e}")
        return (_ICP_EMPTY.copy(), {}, 0, 0, "parse_error", f"Serper parse error: {type(e).__name__}: {e}", 0, 0)
    except anthropic.APIError as e:
        _dlog(f"Step 2 Serper API error for {company_name}: {e}")
        return (_ICP_EMPTY.copy(), {}, 0, 0, "api_error", f"Claude API {type(e).__name__}: {e}", 0, 0)
    except Exception as e:
        _dlog(f"Step 2 Serper error for {company_name}: {e}")
        return (_ICP_EMPTY.copy(), {}, 0, 0, "api_error", f"{type(e).__name__}: {e}", 0, 0)


def run_step2(
    url: str,
    company_name: str,
    api_key: str,
    delay: float,
    model_step2: str = MODEL_STEP2,
    _debug_callback=None,
    search_provider: str = STEP2_PROVIDER_CLAUDE,
    serper_key: str = "",
    dry_run: bool = False,
) -> tuple:
    """
    Research ICP signals — dispatches to either the Claude web_search route
    or the Serper Google Search route depending on search_provider.
    Returns (icp_fields_dict, raw_json, in_tok, out_tok, status, error_msg,
             cache_creation_tokens, cache_read_tokens).

    dry_run=True: generate and log prompts/queries without calling any API.
    _debug_callback: optional callable(event, **kwargs).
      Events: "status" (msg=str), "prompt" (company, model, provider, search_prompt,
              full_prompt, notes, dry_run, queries).
    """
    if search_provider == STEP2_PROVIDER_SERPER:
        # In dry run mode the Serper key is not needed — skip the key guard.
        if not dry_run and not serper_key:
            return (
                _ICP_EMPTY.copy(), {}, 0, 0, "api_error",
                "SERPER_API_KEY is missing from .streamlit/secrets.toml", 0, 0,
            )
        return run_step2_serper(
            url, company_name, api_key, serper_key, delay,
            model_step2=model_step2, _debug_callback=_debug_callback,
            dry_run=dry_run,
        )

    def _dlog(msg: str) -> None:
        if _debug_callback:
            _debug_callback("status", msg=msg)

    target = normalize_url(url) if url else company_name
    if not target:
        return (_ICP_EMPTY.copy(), {}, 0, 0, "no_input", "No URL or company name", 0, 0)

    ck = f"step2_{target}"
    cached = load_cache(ck)
    if cached is not None:
        icp = cached.get("icp_data", {})
        if any(icp.get(f, "") for f in ICP_FIELDS[:3]):  # basic sanity check
            in_t  = int(cached.get("tokens_in", 0) or 0)
            out_t = int(cached.get("tokens_out", 0) or 0)
            _dlog(f"Using cached Step 2 result for {company_name}")
            return (_extract_icp_fields(icp), cached, in_t, out_t, "cached", "", 0, 0)
        _delete_cache(ck)

    _STRICT_SUFFIX = (
        "\n\nReply with ONLY a JSON object, no explanation, no markdown, no backticks."
    )
    search_prompt = f"Now research this company: {target}"
    full_prompt   = STEP2_STATIC_PREFIX + f"\n\n{search_prompt}"

    _dlog(f"Generating Step 2 prompt for {company_name}")
    _dlog(f"Selected model: {model_step2}")

    _ts_pre    = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    _pre_fpath = None   # path of the pre-call file; post-call section is appended here

    if _debug_callback:
        _debug_callback(
            "prompt",
            company=company_name,
            model=model_step2,
            provider=STEP2_PROVIDER_CLAUDE,
            search_prompt=search_prompt,
            full_prompt=full_prompt,
            notes=[
                "DRY RUN ACTIVE: no Anthropic API calls are being made."
                if dry_run else f"Generating Step 2 prompt for {company_name}",
                f"Selected model: {model_step2}",
            ],
            dry_run=dry_run,
            queries=[],
        )
        try:
            _pre_content = _format_claude_pre_debug(
                company_name, model_step2, _ts_pre, full_prompt, dry_run=dry_run,
            )
            _pre_fpath = write_search_debug_file(
                company_name, "claude_web_search", _pre_content, index=1,
            )
            _debug_callback(
                "search_output",
                company=company_name,
                provider=STEP2_PROVIDER_CLAUDE,
                query=search_prompt,
                result_count=0,
                top_results=[],
                debug_file=str(_pre_fpath),
                dry_run=dry_run,
            )
        except Exception:
            pass

    # DRY RUN GUARD: do not call Anthropic in prompt preview mode
    if dry_run:
        _dlog(f"DRY RUN: skipping Anthropic call for {company_name}")
        if _pre_fpath:
            try:
                append_to_debug_file(
                    _pre_fpath,
                    _format_claude_post_debug(
                        company_name, model_step2,
                        datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                        None, "", dry_run_skip=True,
                    ),
                )
            except Exception:
                pass
        return (_ICP_EMPTY.copy(), {}, 0, 0, "dry_run", "DRY RUN: no API call made", 0, 0)

    try:
        _dlog(f"Calling Claude web search for {company_name}")
        time.sleep(delay)
        raw_text, in_t, out_t, _resp = _claude_web_search_full(
            full_prompt, api_key, model_id=model_step2,
        )
        _dlog(f"Received Claude response for {company_name}")

        # Parse (with one retry on failure)
        _icp_raw = None
        try:
            _dlog(f"Parsing Step 2 response for {company_name}")
            _icp_raw = _parse_json_response(raw_text)
        except (json.JSONDecodeError, ValueError):
            _dlog(f"Parse failed — retrying with strict suffix for {company_name}")
            time.sleep(delay)
            raw_text2, in_t2, out_t2, _ = _claude_web_search_full(
                full_prompt + _STRICT_SUFFIX, api_key, model_id=model_step2,
            )
            in_t  += in_t2
            out_t += out_t2
            _dlog(f"Parsing Step 2 retry response for {company_name}")
            _icp_raw = _parse_json_response(raw_text2)  # raises if still bad

        # Append post-call section (with parsed JSON) to the pre-call file
        if _pre_fpath:
            try:
                append_to_debug_file(
                    _pre_fpath,
                    _format_claude_post_debug(
                        company_name, model_step2,
                        datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                        _resp, raw_text, parsed_json=_icp_raw,
                    ),
                )
            except Exception:
                pass

        payload = {"icp_data": _icp_raw, "tokens_in": in_t, "tokens_out": out_t}
        save_cache(ck, payload)
        _dlog(f"Finished Step 2 for {company_name}")
        return (_extract_icp_fields(_icp_raw), payload, in_t, out_t, "ok", "", 0, 0)

    except (json.JSONDecodeError, ValueError) as e:
        _dlog(f"Step 2 parse error for {company_name}: {e}")
        if _pre_fpath:
            try:
                append_to_debug_file(
                    _pre_fpath,
                    _format_claude_post_debug(
                        company_name, model_step2,
                        datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                        None, "", error_str=f"Parse error: {type(e).__name__}: {e}",
                    ),
                )
            except Exception:
                pass
        return (_ICP_EMPTY.copy(), {}, 0, 0, "parse_error", f"Claude parse error: {type(e).__name__}: {e}", 0, 0)

    except anthropic.APIError as e:
        _dlog(f"Step 2 API error for {company_name}: {e}")
        if _pre_fpath:
            try:
                append_to_debug_file(
                    _pre_fpath,
                    _format_claude_post_debug(
                        company_name, model_step2,
                        datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                        None, "", error_str=f"Claude API {type(e).__name__}: {e}",
                    ),
                )
            except Exception:
                pass
        return (_ICP_EMPTY.copy(), {}, 0, 0, "api_error", f"Claude API {type(e).__name__}: {e}", 0, 0)

    except Exception as e:
        _dlog(f"Step 2 error for {company_name}: {e}")
        return (_ICP_EMPTY.copy(), {}, 0, 0, "api_error", f"{type(e).__name__}: {e}", 0, 0)


def _extract_icp_fields(raw: dict) -> dict:
    return {
        "icp_lead_score":                str(raw.get("lead_score")                or "").strip(),
        "icp_buying_signals":            str(raw.get("buying_signals")            or "").strip(),
        "icp_competitor_signal":         str(raw.get("competitor_signal")         or "").strip(),
        "icp_evidence":                  str(raw.get("evidence")                  or "").strip(),
        "icp_likely_training_interest":  str(raw.get("likely_training_interest")  or "").strip(),
        "icp_why_relevant":              str(raw.get("why_relevant")              or "").strip(),
        "icp_potential_buyer_function":  str(raw.get("potential_buyer_function")  or "").strip(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Review flagging
# ─────────────────────────────────────────────────────────────────────────────

def flag_review(row: dict, input_company_name: str) -> dict:
    reasons: list[str] = []
    status   = row.get("enrichment_status", "")
    returned = row.get("lusha_company_name", "")
    inp      = (input_company_name or "").strip()

    _bad_statuses = ("no_data", "api_error", "jina_error", "parse_error", "no_input")
    if status in _bad_statuses or any(status.endswith(s) for s in _bad_statuses):
        reasons.append(f"enrichment status: {status}")

    if returned and inp:
        inp_core = _strip_legal(inp).strip() or inp
        ret_core = _strip_legal(returned).strip() or returned
        sim = str_similarity(inp_core, ret_core)
        if sim < 0.70:
            reasons.append(
                f"Returned name '{returned}' differs from input '{inp}' ({sim:.0%})"
            )

    if inp and returned:
        inp_sfx = _legal_suffix(inp)
        ret_sfx = _legal_suffix(returned)
        if inp_sfx and ret_sfx and inp_sfx != ret_sfx:
            reasons.append(
                f"Legal entity mismatch: input '{inp_sfx.upper()}' vs returned '{ret_sfx.upper()}'"
            )

    row["needs_manual_review"] = "TRUE" if reasons else "FALSE"
    row["match_notes"]         = "; ".join(reasons) if reasons else ""
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Real Lusha API enrichment (optional additional layer)
# ─────────────────────────────────────────────────────────────────────────────

_LUSHA_API_BASE = "https://api.lusha.com/company"
_LUSHA_TIMEOUT  = 15


def _map_lusha_api_fields(raw: dict, source_url: str) -> dict:
    """
    Map a raw Lusha API company response dict to LUSHA_API_FIELDS keys.
    Uses .get() throughout — tolerates missing or differently-shaped responses.
    """
    def s(key, *fallback_keys):
        for k in (key, *fallback_keys):
            v = raw.get(k)
            if v is not None and str(v).strip() not in ("", "None", "null"):
                return str(v).strip()
        return ""

    # Lusha nests some data under sub-dicts; handle both flat and nested
    company = raw if not raw.get("company") else raw.get("company", raw)

    domain = clean_domain(s("domain", "website")) or clean_domain(source_url)

    specialties = company.get("specialties") or company.get("specialties_list") or ""
    if isinstance(specialties, list):
        specialties = ", ".join(str(x) for x in specialties if x)

    technologies = company.get("technologies") or company.get("technology_stack") or ""
    if isinstance(technologies, list):
        technologies = ", ".join(str(x) for x in technologies if x)

    return {
        "lusha_api_company_name":         s("name", "company_name"),
        "lusha_api_domain":               domain,
        "lusha_api_description":          s("description", "about"),
        "lusha_api_founded_year":         s("founded_year", "founded", "year_founded"),
        "lusha_api_employee_range":       s("employee_range", "employees", "company_size", "size"),
        "lusha_api_revenue_range":        s("revenue_range", "revenue", "annual_revenue"),
        "lusha_api_industry":             s("industry", "main_industry"),
        "lusha_api_sub_industry":         s("sub_industry", "sub_category"),
        "lusha_api_company_type":         s("company_type", "type"),
        "lusha_api_country":              s("country", "hq_country"),
        "lusha_api_city":                 s("city", "hq_city"),
        "lusha_api_continent":            s("continent"),
        "lusha_api_linkedin_url":         s("linkedin_url", "linkedin"),
        "lusha_api_specialties":          specialties,
        "lusha_api_technologies":         technologies,
        "lusha_api_total_funding_amount": s("total_funding_amount", "total_funding"),
        "lusha_api_total_funding_rounds": s("total_funding_rounds", "funding_rounds"),
        "lusha_api_last_round_type":      s("last_round_type", "last_funding_type"),
        "lusha_api_last_round_amount":    s("last_round_amount", "last_funding_amount"),
        "lusha_api_last_round_date":      s("last_round_date", "last_funding_date"),
        "lusha_api_ipo_status":           s("ipo_status", "ipo"),
    }


def flag_lusha_api_review(
    lusha_fields: dict,
    input_company_name: str,
    input_url: str,
) -> dict:
    """
    Evaluate match quality for a real Lusha API result.
    Returns dict with keys: lusha_api_match_confidence, lusha_api_needs_review,
    lusha_api_match_notes.
    Kept separate from flag_review() which evaluates Step 1 fields.
    """
    reasons: list = []
    confidence = "high"

    api_name   = lusha_fields.get("lusha_api_company_name", "")
    api_domain = lusha_fields.get("lusha_api_domain",       "")
    api_status = lusha_fields.get("lusha_api_status",       "")

    inp_name   = (input_company_name or "").strip()
    inp_domain = clean_domain(input_url or "")

    # ── No useful data returned ───────────────────────────────────────────────
    useful_fields = [v for k, v in lusha_fields.items()
                     if k in LUSHA_API_FIELDS and v and v not in ("", "N/A")]
    if len(useful_fields) < 3:
        reasons.append("No useful Lusha API data returned")
        confidence = "low"

    # ── Company name similarity ───────────────────────────────────────────────
    if inp_name and api_name:
        inp_core = _strip_legal(inp_name).strip() or inp_name
        api_core = _strip_legal(api_name).strip()  or api_name
        sim = str_similarity(inp_core, api_core)
        if sim < 0.60:
            reasons.append(
                f"Name mismatch: input '{inp_name}' vs Lusha API '{api_name}' ({sim:.0%})"
            )
            confidence = "low"
        elif sim < 0.80:
            reasons.append(
                f"Possible name mismatch: '{inp_name}' vs '{api_name}' ({sim:.0%})"
            )
            if confidence == "high":
                confidence = "medium"

    # ── Domain comparison ─────────────────────────────────────────────────────
    if inp_domain and api_domain:
        if inp_domain != api_domain:
            reasons.append(
                f"Domain conflict: input '{inp_domain}' vs Lusha API '{api_domain}'"
            )
            if confidence != "low":
                confidence = "medium"

    # ── API-level error ───────────────────────────────────────────────────────
    if api_status and api_status not in ("ok", "cached"):
        reasons.append(f"Lusha API status: {api_status}")
        if confidence == "high":
            confidence = "medium"

    needs_review = "TRUE" if (confidence == "low" or reasons) else "FALSE"
    return {
        "lusha_api_match_confidence": confidence,
        "lusha_api_needs_review":     needs_review,
        "lusha_api_match_notes":      "; ".join(reasons) if reasons else "",
    }


def run_lusha_api_enrichment(
    company_name: str,
    raw_url: str,
    api_key: str,
) -> tuple:
    """
    Call the real Lusha Company API and return:
    (lusha_fields_dict, raw_json_dict, status, error_message)

    Uses domain when available, falls back to company name.
    Results are cached under lusha_api_{domain_or_name}.
    Never raises — all errors are returned as status/error strings.
    """
    _empty = {f: "" for f in LUSHA_API_FIELDS}

    if not api_key:
        return _empty, {}, "no_key", "Lusha API key not provided"

    domain = clean_domain(raw_url)
    cache_key = f"lusha_api_{domain or safe_filename(company_name or 'unknown')}"
    cached = load_cache(cache_key)
    if cached is not None:
        fields = _map_lusha_api_fields(cached.get("raw", {}), raw_url)
        return fields, cached.get("raw", {}), "cached", ""

    # ── Build request ─────────────────────────────────────────────────────────
    params: dict = {}
    if domain:
        params["domain"] = domain
    elif company_name:
        params["name"] = company_name
    else:
        return _empty, {}, "no_input", "No domain or company name available"

    try:
        resp = requests.get(
            _LUSHA_API_BASE,
            headers={"api_key": api_key, "Accept": "application/json"},
            params=params,
            timeout=_LUSHA_TIMEOUT,
        )

        if resp.status_code == 404:
            save_cache(cache_key, {"raw": {}, "status": "not_found"})
            return _empty, {}, "not_found", f"Lusha API: company not found (404)"

        if resp.status_code == 401:
            return _empty, {}, "auth_error", "Lusha API: invalid or missing API key (401)"

        if resp.status_code == 429:
            return _empty, {}, "rate_limit", "Lusha API: rate limit exceeded (429)"

        resp.raise_for_status()

        raw = resp.json()
        save_cache(cache_key, {"raw": raw, "status": "ok"})
        fields = _map_lusha_api_fields(raw, raw_url)
        return fields, raw, "ok", ""

    except requests.Timeout:
        return _empty, {}, "timeout", f"Lusha API timed out after {_LUSHA_TIMEOUT}s"
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else 0
        return _empty, {}, f"http_{code}", f"Lusha API HTTP {code}: {str(e)[:120]}"
    except (json.JSONDecodeError, ValueError) as e:
        return _empty, {}, "parse_error", f"Lusha API invalid JSON: {str(e)[:120]}"
    except Exception as e:
        return _empty, {}, "error", f"Lusha API error: {type(e).__name__}: {str(e)[:120]}"


# ─────────────────────────────────────────────────────────────────────────────
# Per-row enrichment  ← orchestrates both steps
# ─────────────────────────────────────────────────────────────────────────────

def enrich_one_row(
    company_name: str,
    raw_url: str,
    api_key: str,
    delay: float,
    use_playwright: bool = True,
    model_step1: str = MODEL_STEP1,
    model_step2: str = MODEL_STEP2,
    _debug_callback=None,
    search_provider: str = STEP2_PROVIDER_CLAUDE,
    serper_key: str = "",
    dry_run: bool = False,
    enable_lusha_api: bool = False,
    lusha_api_key: str = "",
) -> tuple:
    """
    Run optional Lusha API enrichment, then Step 1 (Jina + Claude extraction),
    then Step 2 (Claude web_search ICP).
    Returns (combined_fields_dict, debug_record_dict).
    """
    url          = raw_url.strip() if raw_url else ""
    company_name = company_name.strip() if company_name else ""

    row = {f: "" for f in ALL_ENRICHMENT_FIELDS}

    # 8-second pause between companies to stay under the token/min rate limit
    time.sleep(8)

    # ── Optional: Lusha API enrichment ────────────────────────────────────────
    _lusha_raw_json = {}
    if enable_lusha_api and lusha_api_key:
        la_fields, _lusha_raw_json, la_status, la_err = run_lusha_api_enrichment(
            company_name, url, lusha_api_key,
        )
        row.update(la_fields)
        row["lusha_api_status"] = la_status
        row["lusha_api_error"]  = la_err
        review_meta = flag_lusha_api_review(la_fields, company_name, url)
        row.update(review_meta)
    elif enable_lusha_api and not lusha_api_key:
        row["lusha_api_status"] = "no_key"
        row["lusha_api_error"]  = "Lusha API key not provided"

    # ── Step 1 (three-tier: Jina → Playwright → web_search → no_data) ──────────
    s1_fields, s1_raw, s1_in, s1_out, s1_status, s1_err, s1_pw_dbg = run_step1(
        url, company_name, api_key, delay,
        use_playwright=use_playwright, model_step1=model_step1,
    )

    row.update(s1_fields)
    row["step1_status"]     = s1_status
    row["step1_tokens_in"]  = str(s1_in)
    row["step1_tokens_out"] = str(s1_out)
    row["step1_cost_usd"]   = f"{calc_cost(s1_in, s1_out):.6f}"

    # ── Step 2 ────────────────────────────────────────────────────────────────
    s2_fields, s2_raw, s2_in, s2_out, s2_status, s2_err, s2_cache_create, s2_cache_read = run_step2(
        url, company_name, api_key, delay, model_step2=model_step2,
        _debug_callback=_debug_callback,
        search_provider=search_provider, serper_key=serper_key,
        dry_run=dry_run,
    )
    row.update(s2_fields)
    row["step2_status"]   = s2_status
    row["step2_tokens_in"]  = str(s2_in)
    row["step2_tokens_out"] = str(s2_out)
    row["step2_cost_usd"]   = f"{calc_cost(s2_in, s2_out):.6f}"

    # ── Combined metadata ─────────────────────────────────────────────────────
    total_in  = s1_in  + s2_in
    total_out = s1_out + s2_out
    row["total_tokens_in"]  = str(total_in)
    row["total_tokens_out"] = str(total_out)
    row["total_cost_usd"]   = f"{calc_cost(total_in, total_out):.6f}"

    has_s1 = _step1_has_data(s1_fields)
    has_s2 = any(s2_fields.get(f, "") for f in ICP_FIELDS[:3])

    if has_s1:
        # Preserve tier status (enriched_jina / enriched_search); append _step1_only if no ICP
        row["enrichment_status"] = s1_status if has_s2 else f"{s1_status}_step1_only"
    else:
        row["enrichment_status"] = "no_data"

    err_parts = [p for p in [s1_err, s2_err] if p]
    row["error_message"] = " | ".join(err_parts)

    flag_review(row, company_name)

    # Debug record
    dbg = {
        "input_company_name":       company_name,
        "input_url":                raw_url,
        "normalized_url":           normalize_url(url),
        "lusha_api_status":         row.get("lusha_api_status", ""),
        "lusha_api_raw_json":       _lusha_raw_json,
        "step1_status":             s1_status,
        "step1_raw_json":           s1_raw,
        "step1_tokens_in":          s1_in,
        "step1_tokens_out":         s1_out,
        "step1_playwright_attempted": s1_pw_dbg.get("playwright_attempted", False),
        "step1_playwright_result":    s1_pw_dbg.get("playwright_result", "skipped"),
        "step2_status":                  s2_status,
        "step2_raw_json":                s2_raw,
        "step2_tokens_in":               s2_in,
        "step2_tokens_out":              s2_out,
        "step2_cache_creation_tokens":   s2_cache_create,
        "step2_cache_read_tokens":       s2_cache_read,
        "total_cost":                    calc_cost(total_in, total_out),
        "enrichment_status":        row["enrichment_status"],
        "error_message":            row["error_message"],
        "needs_manual_review":      row["needs_manual_review"],
        "match_notes":              row["match_notes"],
    }
    return row, dbg


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering placeholder (not implemented yet)
# ─────────────────────────────────────────────────────────────────────────────

def build_model_features(df_enriched: pd.DataFrame) -> pd.DataFrame:
    """
    TODO:
    Convert raw enrichment data into model-ready scalar/discrete features.
    This should later create fields such as:
    - multi_country_presence: 0/1
    - foreign_hq_signal: 0/1
    - competitor_signal: 0/1
    - employee_size_score: 1-5
    - international_presence_score: 1-5
    - language_need_score: 1-5
    - industry_fit_score: 1-5
    - data_quality_score: 1-5

    This should stay separate from raw enrichment.
    """
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────────────────

def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Enriched")
    return buf.getvalue()


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def _js_auto_download(df: pd.DataFrame, filename: str) -> None:
    """
    Inject a tiny JS snippet that silently downloads the DataFrame as Excel.
    Executes inside a zero-height iframe — does NOT trigger a Streamlit rerun.
    Note: browsers may prompt once to allow multiple automatic downloads.
    """
    b64 = base64.b64encode(df_to_excel_bytes(df)).decode()
    components.html(
        f"""<script>
        (function(){{
            var a = document.createElement('a');
            a.href = 'data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                     + ';base64,{b64}';
            a.download = '{filename}';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }})();
        </script>""",
        height=1,   # must be ≥ 1; script still runs at 1px
    )


def _html_dl_buttons(partial_df: pd.DataFrame, n_done: int, stamp: str) -> None:
    """
    Render Excel + CSV download anchors as plain HTML — clicking them triggers
    a browser download WITHOUT causing a Streamlit rerun, so the processing
    loop is never interrupted.
    """
    xl_bytes  = df_to_excel_bytes(partial_df)
    csv_bytes = df_to_csv_bytes(partial_df)
    xl_b64    = base64.b64encode(xl_bytes).decode()
    csv_b64   = base64.b64encode(csv_bytes).decode()
    xl_kb     = max(len(xl_bytes)  // 1024, 1)
    csv_kb    = max(len(csv_bytes) // 1024, 1)
    _style = (
        "display:inline-block;padding:7px 16px;border-radius:6px;"
        "font-size:13px;font-family:sans-serif;font-weight:600;"
        "text-decoration:none;color:#fff;background:#0068c9;"
    )
    components.html(
        f"""<div style="display:flex;gap:10px;margin:2px 0;">
            <a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{xl_b64}"
               download="enriched_partial_{stamp}.xlsx" style="{_style}">
               ⬇ Excel ({n_done} rows, ~{xl_kb} KB)
            </a>
            <a href="data:text/csv;base64,{csv_b64}"
               download="enriched_partial_{stamp}.csv" style="{_style}">
               ⬇ CSV ({n_done} rows, ~{csv_kb} KB)
            </a>
        </div>""",
        height=48,
    )


def make_log_df(debug_records: list, elm_mode: bool = False) -> pd.DataFrame:
    rows = []
    for d in debug_records:
        if elm_mode:
            rows.append({
                "company":        d.get("company", d.get("input_company_name", "")),
                "url":            d.get("url", d.get("input_url", "")),
                "domain":         d.get("domain", ""),
                "fetch_status":   d.get("status", ""),
                "pages_fetched":  d.get("pages_fetched", ""),
                "total_chars":    d.get("total_chars", ""),
            })
        else:
            rows.append({
                "input_company_name":  d.get("input_company_name", ""),
                "input_url":           d.get("input_url", ""),
                "lusha_api_status":    d.get("lusha_api_status", ""),
                "step1_status":        d.get("step1_status", ""),
                "step2_status":        d.get("step2_status", ""),
                "enrichment_status":   d.get("enrichment_status", ""),
                "step1_tokens_in":     d.get("step1_tokens_in", ""),
                "step1_tokens_out":    d.get("step1_tokens_out", ""),
                "step2_tokens_in":     d.get("step2_tokens_in", ""),
                "step2_tokens_out":    d.get("step2_tokens_out", ""),
                "total_cost_usd":      f"{d.get('total_cost', 0):.6f}",
                "needs_manual_review": d.get("needs_manual_review", ""),
                "match_notes":         d.get("match_notes", ""),
                "error_message":       d.get("error_message", ""),
            })
    return pd.DataFrame(rows)


def cache_to_zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in list_cache_files():
            zf.write(f, f.name)
    return buf.getvalue()


def build_partial_df(results: list, df_work: pd.DataFrame,
                     field_list: list | None = None) -> pd.DataFrame:
    """Build an enriched DataFrame from however many rows have been processed so far."""
    flist       = field_list if field_list is not None else ALL_ENRICHMENT_FIELDS
    df_out      = df_work.head(len(results)).copy().reset_index(drop=True)
    enriched_df = pd.DataFrame(results)
    for col in flist:
        df_out[col] = enriched_df[col].values if col in enriched_df.columns else ""
    return df_out


def ts() -> str:
    """Compact UTC timestamp for filenames: 20250516_143022"""
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def get_provider_code(provider_name: str) -> str:
    """Return a short filename-safe code for the Step 2 search provider."""
    if provider_name == STEP2_PROVIDER_CLAUDE:
        return "cws"
    if provider_name == STEP2_PROVIDER_SERPER:
        return "sg"
    return "unk"


def get_model_code(model_name: str) -> str:
    """Return a short filename-safe code for the Step 2 model."""
    m = (model_name or "").lower()
    if "haiku" in m:
        return "hq"
    if "sonnet" in m:
        return "sn"
    if "opus" in m:
        return "op"
    return "unk"


def build_run_tag() -> str:
    """
    Return a filename-safe tag for the current run, including provider code,
    model code, and 'lusha' suffix when Lusha API enrichment is enabled.
    Safe to call from both the processing loop and the results section.
    """
    prov  = st.session_state.get("_step2_provider",   STEP2_PROVIDER_CLAUDE)
    model = st.session_state.get("_model_step2",       MODEL_STEP2)
    lusha = st.session_state.get("_enable_lusha_api",  False)
    tag   = f"{get_provider_code(prov)}_{get_model_code(model)}"
    if lusha:
        tag = f"{tag}_lusha"
    return tag


def save_to_local_folder(df: pd.DataFrame, folder: str, run_tag: str = "") -> tuple[str, str]:
    """
    Write Excel + CSV to *folder* with timestamped filenames.
    Returns (excel_path, csv_path). Raises OSError on permission / path errors.
    Only usable when the app runs locally — not on Streamlit Cloud.
    """
    folder_path = Path(folder.strip())
    folder_path.mkdir(parents=True, exist_ok=True)
    stamp      = ts()
    tag        = f"_{run_tag}" if run_tag else ""
    excel_path = folder_path / f"enriched_results{tag}_{stamp}.xlsx"
    csv_path   = folder_path / f"enriched_results{tag}_{stamp}.csv"
    df_to_excel_bytes_write(df, str(excel_path))
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return str(excel_path), str(csv_path)


def df_to_excel_bytes_write(df: pd.DataFrame, path: str) -> None:
    """Write DataFrame to an Excel file at *path* on disk."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Enriched")


# ─────────────────────────────────────────────────────────────────────────────
# Auto-save helpers
# ─────────────────────────────────────────────────────────────────────────────

def autosave_append(row_fields: dict, input_row: pd.Series) -> None:
    """Append one enriched row to the autosave CSV."""
    record = {**input_row.to_dict(), **row_fields}
    df_row = pd.DataFrame([record])
    write_header = not os.path.exists(AUTOSAVE_PATH)
    df_row.to_csv(AUTOSAVE_PATH, mode="a", header=write_header, index=False)


def autosave_load() -> pd.DataFrame | None:
    """Return the autosave DataFrame, or None if it doesn't exist / is unreadable."""
    if not os.path.exists(AUTOSAVE_PATH):
        return None
    try:
        df = pd.read_csv(AUTOSAVE_PATH)
        return df if len(df) > 0 else None
    except Exception:
        return None


def autosave_clear() -> None:
    try:
        os.remove(AUTOSAVE_PATH)
    except FileNotFoundError:
        pass


def autosave_already_done(df_saved: pd.DataFrame, name_col: str, domain_col: str | None,
                          company_name: str, raw_url: str) -> bool:
    """Return True if this company already appears in the autosave file."""
    if df_saved is None or df_saved.empty:
        return False
    # Match by domain first (more reliable), fall back to company name
    if domain_col and domain_col in df_saved.columns and raw_url:
        url_clean = clean_domain(raw_url)
        saved_domains = df_saved[domain_col].astype(str).apply(clean_domain)
        if url_clean and (saved_domains == url_clean).any():
            return True
    if name_col in df_saved.columns and company_name:
        saved_names = df_saved[name_col].astype(str).str.strip().str.lower()
        if company_name.lower() in saved_names.values:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Session-state helpers
# ─────────────────────────────────────────────────────────────────────────────

def ss(key, default=None):
    return st.session_state.get(key, default)


def ss_set(**kwargs):
    for k, v in kwargs.items():
        st.session_state[k] = v


def reset_processing(clear_autosave: bool = False):
    if clear_autosave:
        autosave_clear()
    ss_set(
        processing=False, stop_requested=False,
        process_index=0, results=[], debug_records=[],
        enrichment_done=False, df_enriched=None,
        total_tokens_in=0, total_tokens_out=0, total_cost_usd=0.0,
        total_cache_read_tokens=0, total_cache_create_tokens=0,
        autosave_last_name="",
        _jina_retry_count=0, _last_retry_msg="",
        _local_save_enabled=False, _final_auto_saved=False,
        _auto_dl_count=0, _auto_dl_last_msg="",
        _step2_debug_log="", _step2_prompt_records=[],
        _dry_run_records=[], _search_output_records=[], _step2_debug_files=[],
        _zero_cost_preview=False, _dry_run_preview_count=0,
    )


def build_and_finish(results: list, debug_records: list, df_work: pd.DataFrame,
                     field_list: list | None = None) -> None:
    flist       = field_list if field_list is not None else ALL_ENRICHMENT_FIELDS
    df_out      = df_work.head(len(results)).copy().reset_index(drop=True)
    enriched_df = pd.DataFrame(results)
    for col in flist:
        df_out[col] = enriched_df[col].values if col in enriched_df.columns else ""
    ss_set(
        processing=False, stop_requested=False,
        enrichment_done=True, df_enriched=df_out,
        debug_records=debug_records,
    )
    st.rerun()


# =============================================================================
# UI
# =============================================================================

st.set_page_config(
    page_title="Claude Company Enrichment",
    page_icon="🏢",
    layout="wide",
)
st.title("🏢 Claude Company Enrichment")
st.caption(
    "Upload a file with company names and URLs. "
    "Each row is enriched in two passes: basic firmographics (Jina AI + Claude) "
    "and Mingle ICP signals (Claude web search)."
)

# =============================================================================
# API KEY — secrets only, no sidebar input
# =============================================================================

api_key = ""
_api_key_error = ""
try:
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "") or ""
except Exception:
    pass

if not api_key:
    _api_key_error = (
        "**ANTHROPIC_API_KEY not found in Streamlit secrets.** "
        "Add it to your app's secrets: Settings → Secrets → "
        "`ANTHROPIC_API_KEY = \"sk-ant-...\"`"
    )

# Serper key — optional, only needed when Step 2 provider is Serper Google Search.
# Required entry in .streamlit/secrets.toml: SERPER_API_KEY = "your-key"
serper_key = ""
try:
    serper_key = st.secrets.get("SERPER_API_KEY", "") or ""
except Exception:
    pass

# Lusha API key — optional, only needed when Lusha API enrichment is enabled.
# Required entry in .streamlit/secrets.toml: LUSHA_API_KEY = "your-key"
lusha_api_key = ""
try:
    lusha_api_key = st.secrets.get("LUSHA_API_KEY", "") or ""
except Exception:
    pass

# =============================================================================
# SIDEBAR  — debug toggle only (no API key input)
# =============================================================================

with st.sidebar:
    # ── Enrichment mode ───────────────────────────────────────────────────────
    enrichment_mode = st.radio(
        "Enrichment mode",
        options=["Full Claude enrichment", "Extreme Light Mode (no API)"],
        index=1,
        key="enrichment_mode_radio",
        help=(
            "**Full Claude**: Jina AI + Claude API — requires ANTHROPIC_API_KEY.\n\n"
            "**Extreme Light Mode**: fetches company pages with requests/BeautifulSoup, "
            "extracts keyword signals and normalized scores — no API key, zero tokens."
        ),
    )
    _elm_mode = enrichment_mode == "Extreme Light Mode (no API)"
    st.divider()

    st.header("Settings")

    # ── API key statuses — always shown ──────────────────────────────────────
    if api_key:
        st.success("✓ Anthropic API key loaded")
    else:
        st.error("⚠ Anthropic API key missing")
    if serper_key:
        st.success("✓ Serper API key loaded")
    else:
        st.warning("⚠ Serper API key not set (only needed for Serper provider in real runs)")
    if lusha_api_key:
        st.success("✓ Lusha API key loaded")
    else:
        st.caption("ⓘ Lusha API key not set (only needed when Lusha API enrichment is enabled)")

    st.divider()

    # ── Lusha API enrichment toggle ───────────────────────────────────────────
    enable_lusha_api = st.checkbox(
        "Enable Lusha API enrichment",
        value=False,
        key="enable_lusha_api_checkbox",
        help=(
            "Calls the real Lusha Company API to enrich each row with verified firmographic data. "
            "Requires LUSHA_API_KEY in .streamlit/secrets.toml.\n\n"
            "Results appear as new columns prefixed lusha_api_. "
            "Existing Step 1 / Step 2 columns are not affected."
        ),
    )
    if enable_lusha_api and not lusha_api_key:
        st.warning(
            "⚠️ LUSHA_API_KEY is missing from .streamlit/secrets.toml. "
            "Add it or disable Lusha API enrichment."
        )
    st.session_state["_enable_lusha_api"] = enable_lusha_api

    st.divider()
    model_step1_label = st.selectbox(
        "Model — Step 1 (firmographics)",
        options=list(AVAILABLE_MODELS.keys()),
        index=0,
        help="Used for extracting structured company data from scraped pages. Haiku is sufficient here.",
    )
    model_step2_label = st.selectbox(
        "Model — Step 2 (ICP web search)",
        options=list(AVAILABLE_MODELS.keys()),
        index=0,
        help="Used for the agentic web search. Sonnet gives better signal detection but costs ~5x more.",
    )
    selected_model_step1 = AVAILABLE_MODELS[model_step1_label]
    selected_model_step2 = AVAILABLE_MODELS[model_step2_label]
    st.session_state["_model_step1"] = selected_model_step1
    st.session_state["_model_step2"] = selected_model_step2

    st.divider()

    step2_provider = st.selectbox(
        "Step 2 web search provider",
        options=[STEP2_PROVIDER_CLAUDE, STEP2_PROVIDER_SERPER],
        index=0,
        help=(
            f"**{STEP2_PROVIDER_CLAUDE}** (default): uses Anthropic's built-in "
            "web_search tool — no extra API key needed.\n\n"
            f"**{STEP2_PROVIDER_SERPER}**: calls the Serper API for Google results, "
            "then Claude analyzes the snippets. "
            "Requires `SERPER_API_KEY` in `.streamlit/secrets.toml`."
        ),
    )
    st.session_state["_step2_provider"] = step2_provider

    step2_dry_run = st.checkbox(
        "Step 2 dry run: generate prompts only",
        value=False,
        help=(
            "Generate and display Step 2 prompts/search queries without calling "
            "Anthropic or Serper. Useful for inspecting what would be sent before "
            "spending tokens or API credits."
        ),
    )
    st.session_state["_step2_dry_run"] = step2_dry_run
    if step2_dry_run:
        st.info("Dry run active — Step 2 will not call any API.")

    _zero_cost_default = step2_dry_run
    zero_cost_preview = st.checkbox(
        "Zero-cost preview: skip Step 1 API calls",
        value=_zero_cost_default,
        help=(
            "When enabled alongside dry run, skips ALL external API calls — "
            "Jina, Anthropic, Serper, and browser scraping. "
            "Uses only uploaded row data and existing cache to build Step 2 prompt previews. "
            "Cost stays $0.00."
        ),
    )
    st.session_state["_zero_cost_preview"] = zero_cost_preview
    if zero_cost_preview and step2_dry_run:
        st.info("Zero-cost preview active — no Step 1 or Step 2 API calls will be made.")

    st.divider()

    debug_mode = st.checkbox(
        "Enable debug mode",
        value=False,
        help="Shows per-row JSON responses, cache tools, and additional downloads.",
    )

    st.markdown("**Step 2 debug logging**")
    show_step2_debug = st.checkbox(
        "Show Step 2 debug logs",
        value=False,
        help=(
            "Shows a live debug/log window in the app during Step 2 processing. "
            "Reveals the exact prompt sent to Claude and status messages per company."
        ),
    )
    save_step2_debug = st.checkbox(
        "Save Step 2 debug logs to files",
        value=True,
        help=(
            f"Saves one .txt file per company to the `{DEBUG_LOG_DIR}/` folder. "
            "Includes model, prompt, provider, and status notes."
        ),
    )
    st.session_state["_show_step2_debug"] = show_step2_debug
    st.session_state["_save_step2_debug"] = save_step2_debug
    if save_step2_debug:
        st.caption(
            f"Prompt files → `{DEBUG_LOG_DIR}/`  \n"
            f"Search I/O files → `{SEARCH_OUTPUT_DIR}/`"
        )

    st.divider()
    if _PLAYWRIGHT_AVAILABLE:
        use_playwright = st.checkbox(
            "Use browser scraping for blocked sites",
            value=True,
            help=(
                "Uses headless Chrome with human behaviour to scrape sites that block Jina. "
                "Slower but more thorough."
            ),
        )
    else:
        st.caption(
            "⚠ Browser scraping unavailable — run "
            "`pip install playwright && playwright install chromium` to enable."
        )
        use_playwright = False
    st.session_state["_use_playwright"] = use_playwright

    st.divider()
    _cache_n = get_cache_count()
    st.caption(f"Enrichment cache: **{_cache_n}** file(s)")
    if st.button("Clear enrichment cache", use_container_width=True, key="clear_cache_always"):
        if CACHE_DIR.exists():
            for f in CACHE_DIR.glob("*.json"):
                f.unlink(missing_ok=True)
        st.rerun()

    # ── Auto-save status ──────────────────────────────────────────────────────
    _last_name = ss("autosave_last_name", "")
    if ss("processing", False) and _last_name:
        st.divider()
        st.caption(f"💾 Auto-save active — last saved: **{_last_name}**")
        _auto_dl_msg = ss("_auto_dl_last_msg", "")
        if _auto_dl_msg:
            st.caption(f"📥 {_auto_dl_msg}")
    elif os.path.exists(AUTOSAVE_PATH):
        _saved_df = autosave_load()
        if _saved_df is not None:
            st.divider()
            st.info(
                f"⚠️ Interrupted session found — "
                f"**{len(_saved_df)}** companies already processed."
            )
            _rb, _fb = st.columns(2)
            if _rb.button("▶ Resume", use_container_width=True, key="resume_btn"):
                ss_set(_resume_mode=True)
                st.rerun()
            if _fb.button("✕ Start fresh", use_container_width=True, key="fresh_btn"):
                autosave_clear()
                ss_set(_resume_mode=False)
                st.rerun()

    # ── Local auto-save ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("📁 Local auto-save")
    local_save_enabled = st.checkbox(
        "Enable local auto-save",
        value=ss("local_save_enabled", False),
        key="local_save_enabled",
        help=(
            f"Saves an Excel snapshot every {LOCAL_SAVE_EVERY} companies and once more "
            "on completion. Only works when the app runs locally."
        ),
    )
    if local_save_enabled:
        _default_dir = ss("_local_save_path", "") or _DEFAULT_DOWNLOAD_DIR
        local_save_path = st.text_input(
            "Download directory",
            value=_default_dir,
            placeholder=_DEFAULT_DOWNLOAD_DIR,
            key="local_save_path_input",
        )
        _eff_path = (local_save_path or "").strip() or _DEFAULT_DOWNLOAD_DIR
        ss_set(_local_save_path=_eff_path, _local_save_enabled=True)
        _last_local = ss("_last_local_save", "")
        if _last_local:
            st.caption(f"📁 Local auto-save active — saving to: **{_eff_path}**")
            st.caption(f"Last snapshot: {_last_local}")
        else:
            st.caption(f"Will save to: **{_eff_path}**")
    else:
        ss_set(_local_save_path="", _local_save_enabled=False)
        st.caption(
            f"When disabled, the final results file is automatically saved to "
            f"**{_DEFAULT_DOWNLOAD_DIR}** when processing completes."
        )

    if debug_mode:
        st.divider()
        st.subheader("Debug settings")
        delay_sec = st.slider(
            "Delay between API calls (sec)",
            min_value=0.0, max_value=3.0, value=1.0, step=0.1,
        )
        st.divider()
        st.metric("Cached entries", get_cache_count())
        st.caption(f"Cache: `{CACHE_DIR.resolve()}`")
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
    ss_set(_file_key=new_file_key, df_raw=None, file_name=None, file_error=None)
    reset_processing()
    if uploaded is not None:
        try:
            fname = uploaded.name
            df_loaded = (
                pd.read_csv(uploaded)
                if fname.lower().endswith(".csv")
                else pd.read_excel(uploaded)
            )
            ss_set(df_raw=df_loaded, file_name=fname)
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
    if _elm_mode:
        st.info(
            "💡 **Extreme Light Mode** — no API calls. "
            "Fetches company pages with requests/BeautifulSoup and extracts keyword signals."
        )
    else:
        st.info(
            "💡 Each row makes **two** Claude API calls (Step 1 + Step 2). "
            "Use the row limiter below to test with a small batch first."
        )

# =============================================================================
# STEP 2 — Preview
# =============================================================================

name_col     = None
domain_col   = None
n_to_process = 0

if df_raw is not None:

    st.divider()
    st.subheader("Step 2 · Preview")
    st.dataframe(df_raw.head(), use_container_width=True)
    st.caption(f"{len(df_raw):,} rows · {len(df_raw.columns)} columns")

    # ── Column selection ──────────────────────────────────────────────────────

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
                "Used for Jina Reader (Step 1) and Claude web search (Step 2). "
                "Falls back to company name search when URL is absent or unreachable."
            ),
        )
    domain_col = dom_choice if dom_choice != _NO_DOMAIN else None

    note_parts = []
    if auto_name_col:
        note_parts.append(f"company name → **{auto_name_col}**")
    if auto_domain_col:
        note_parts.append(f"URL → **{auto_domain_col}**")
    st.caption(
        ("Auto-detected: " + ",  ".join(note_parts))
        if note_parts
        else "Could not auto-detect columns — please select them manually."
    )

    # ── Processing scope ──────────────────────────────────────────────────────

    st.divider()
    st.subheader("Step 4 · Processing scope")

    limit_rows = st.checkbox("Limit rows for testing", value=False)
    if limit_rows:
        row_limit = st.number_input(
            "Number of rows to process",
            min_value=1, max_value=len(df_raw),
            value=min(5, len(df_raw)), step=1,
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
_active_provider     = ss("_step2_provider",    STEP2_PROVIDER_CLAUDE)
_active_dry_run      = ss("_step2_dry_run",     False)
_active_zero_cost    = ss("_zero_cost_preview", False)
_is_preview_mode     = _active_dry_run or _active_zero_cost

# (1) Claude real run → Anthropic key required
# (2) Serper real run → both Anthropic + Serper keys required
# (3) Dry run only   → no API keys required
# (4) Zero-cost      → no API keys required
if _api_key_error and not _elm_mode and not _is_preview_mode:
    blocking.append(_api_key_error)
if (
    _active_provider == STEP2_PROVIDER_SERPER
    and not serper_key
    and not _elm_mode
    and not _is_preview_mode
):
    blocking.append("SERPER_API_KEY is missing from .streamlit/secrets.toml")
_active_lusha_api = ss("_enable_lusha_api", False)
if _active_lusha_api and not lusha_api_key:
    blocking.append("LUSHA_API_KEY is missing from .streamlit/secrets.toml")
if uploaded is None:
    blocking.append("No file uploaded yet.")
if file_error:
    blocking.append(f"File could not be read: {file_error}")
if df_raw is not None and name_col is None:
    blocking.append("No company name column selected.")
if df_raw is not None and n_to_process == 0:
    blocking.append("Zero rows selected for processing.")

if blocking and not currently_processing:
    for reason in blocking:
        st.warning(f"⚠️ {reason}")
elif not blocking and not currently_processing and not enrichment_done:
    _s1 = ss("_model_step1", MODEL_STEP1)
    _s2 = ss("_model_step2", MODEL_STEP2)
    if _is_preview_mode or _elm_mode:
        st.info(
            f"Ready to preview **{n_to_process:,}** rows. "
            "Estimated cost: **$0.00** — no API calls will be made."
        )
    else:
        _cost_per_row = _COST_EST.get((_s1, _s2), 0.05)
        est = n_to_process * _cost_per_row
        st.info(
            f"Ready to enrich **{n_to_process:,}** rows with two enrichment steps each. "
            f"Rough estimated cost: ~${est:.2f} "
            f"(~${_cost_per_row:.2f}/company with current model selection)."
        )

start_btn = st.button(
    "▶ Start enrichment",
    type="primary",
    use_container_width=True,
    disabled=(bool(blocking) or currently_processing),
    key="start_button",
)

if start_btn and not blocking and not currently_processing:
    df_work      = df_raw.head(n_to_process).copy()
    resume_mode  = ss("_resume_mode", False)
    if not resume_mode:
        autosave_clear()   # wipe any previous autosave on a fresh start
    ss_set(
        processing=True, stop_requested=False,
        process_index=0, results=[], debug_records=[],
        enrichment_done=False, df_enriched=None,
        _df_work=df_work, _name_col=name_col, _domain_col=domain_col,
        _n_to_process=n_to_process, _api_key=api_key, _delay=delay_sec,
        total_tokens_in=0, total_tokens_out=0, total_cost_usd=0.0,
        total_cache_read_tokens=0, total_cache_create_tokens=0,
        _resume_mode=resume_mode, autosave_last_name="",
        _elm_mode=_elm_mode,
        _active_fields=ELM_ALL_FIELDS if _elm_mode else ALL_ENRICHMENT_FIELDS,
        _local_save_enabled=ss("_local_save_enabled", False),
        _final_auto_saved=False,
        _use_playwright=ss("_use_playwright", True),
        _model_step1=ss("_model_step1", MODEL_STEP1),
        _model_step2=ss("_model_step2", MODEL_STEP2),
        _step2_provider=ss("_step2_provider", STEP2_PROVIDER_CLAUDE),
        _serper_key=serper_key,
        _step2_dry_run=ss("_step2_dry_run", False),
        _zero_cost_preview=ss("_zero_cost_preview", False),
        _enable_lusha_api=ss("_enable_lusha_api", False),
        _lusha_api_key=lusha_api_key,
        _dry_run_records=[], _search_output_records=[], _step2_debug_files=[],
        _dry_run_preview_count=0,
    )
    st.rerun()

# =============================================================================
# PROCESSING LOOP — one row per Streamlit rerun
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
    _elm_mode_run  = ss("_elm_mode", False)
    _active_fields = ss("_active_fields", ALL_ENRICHMENT_FIELDS)
    _use_playwright_run  = ss("_use_playwright", True)
    _model_step1_run     = ss("_model_step1", MODEL_STEP1)
    _model_step2_run     = ss("_model_step2", MODEL_STEP2)
    _step2_provider_run  = ss("_step2_provider", STEP2_PROVIDER_CLAUDE)
    _serper_key_run      = ss("_serper_key", "")
    _dry_run_run         = ss("_step2_dry_run", False)
    _zero_cost_run       = ss("_zero_cost_preview", False)
    _enable_lusha_api_run = ss("_enable_lusha_api", False)
    _lusha_api_key_run    = ss("_lusha_api_key", "")
    total_in          = ss("total_tokens_in", 0)
    total_out         = ss("total_tokens_out", 0)
    total_cost        = ss("total_cost_usd", 0.0)
    total_cache_read  = ss("total_cache_read_tokens", 0)
    total_cache_create = ss("total_cache_create_tokens", 0)

    if st.button("⏹ Stop after current row", key="stop_button"):
        ss_set(stop_requested=True)
        st.rerun()

    st.progress(idx / _n if _n else 1.0, text=f"Row {idx} of {_n}")

    if _elm_mode_run:
        cnt_ok      = sum(1 for r in results if r.get("elm_fetch_status") == "ok")
        cnt_partial = sum(1 for r in results if r.get("elm_fetch_status") == "partial")
        cnt_failed  = sum(1 for r in results if r.get("elm_fetch_status") == "failed")
        avg_score   = (
            sum(float(r.get("elm_score_overall_icp", 0) or 0) for r in results) / len(results)
            if results else 0.0
        )
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Fetched OK",  cnt_ok)
        mc2.metric("Partial",     cnt_partial)
        mc3.metric("Failed",      cnt_failed)
        mc4.metric("Processed",   len(results))
        mc5.metric("Avg ICP score", f"{avg_score:.1f}/10")
    else:
        cnt_jina       = sum(1 for r in results if "enriched_jina"       in r.get("enrichment_status", ""))
        cnt_playwright = sum(1 for r in results if "enriched_playwright" in r.get("enrichment_status", ""))
        cnt_google     = sum(1 for r in results if "enriched_search"     in r.get("enrichment_status", ""))
        cnt_nodata     = sum(1 for r in results if r.get("enrichment_status") == "no_data")
        cnt_error      = sum(1 for r in results
                             if r.get("enrichment_status") not in
                             ("enriched_jina", "enriched_jina_step1_only",
                              "enriched_playwright", "enriched_playwright_step1_only",
                              "enriched_search", "enriched_search_step1_only",
                              "no_data", "skipped_resume", "zero_cost_preview", ""))
        cnt_retries    = ss("_jina_retry_count", 0)
        cnt_previews   = ss("_dry_run_preview_count", 0)

        if _zero_cost_run and _dry_run_run:
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Dry-run previews generated", cnt_previews)
            mc2.metric("Errors",                     cnt_error)
            mc3.metric("Est. cost",                  "$0.00")
        else:
            mc1, mc2, mc3, mc4, mc5, mc6, mc7 = st.columns(7)
            mc1.metric("Enriched (Jina)",    cnt_jina)
            mc2.metric("Enriched (Browser)", cnt_playwright)
            mc3.metric("Enriched (Google)",  cnt_google)
            mc4.metric("429 Retries",        cnt_retries)
            mc5.metric("No data",            cnt_nodata)
            mc6.metric("Errors",             cnt_error)
            mc7.metric("Est. cost",          f"${total_cost:.4f}")

        _retry_msg = ss("_last_retry_msg", "")
        if _retry_msg:
            st.info(_retry_msg)

    if _zero_cost_run and _dry_run_run and not _elm_mode_run:
        st.warning(
            "⚠️ **ZERO-COST PREVIEW ACTIVE**: no Step 1 or Step 2 API calls are being made. "
            "Using only uploaded row data and existing cache to generate Step 2 prompt previews."
        )
    elif _dry_run_run and not _elm_mode_run:
        st.warning(
            "⚠️ **DRY RUN ACTIVE**: no Anthropic or Serper API calls are being made "
            "for Step 2. Prompts and search queries are generated and displayed only."
        )

    # ── Intermediate download buttons (visible whenever ≥1 row is done) ───────
    # Uses HTML anchors so clicking does NOT trigger a Streamlit rerun / freeze.
    if results:
        _partial_df = build_partial_df(results, df_work, _active_fields)
        _n_done     = len(_partial_df)
        _stamp      = ts()
        with st.expander(
            f"⬇ Download intermediate results ({_n_done} rows so far)", expanded=True
        ):
            st.caption("These links download via the browser without interrupting processing.")
            _html_dl_buttons(_partial_df, _n_done, _stamp)

    # ── Step 2 dry run preview ────────────────────────────────────────────────
    if _dry_run_run and not _elm_mode_run:
        _dry_recs = ss("_dry_run_records", [])
        with st.expander("Step 2 Dry Run Preview", expanded=True):
            if not _dry_recs:
                st.caption("Dry run preview will appear here as companies are processed.")
            for _dr in _dry_recs:
                with st.expander(f"Company: {_dr['company']}", expanded=False):
                    st.markdown(
                        f"**Provider:** {_dr['provider']}  |  **Model:** `{_dr['model']}`"
                    )
                    if _dr.get("queries"):
                        st.markdown("**Serper queries that would be sent:**")
                        for _q in _dr["queries"]:
                            st.code(_q, language=None)
                    st.markdown("**Generated search instruction / prompt suffix:**")
                    st.code(_dr.get("search_prompt", ""), language=None)
                    st.markdown("**Full Step 2 Claude prompt:**")
                    st.code(_dr.get("full_prompt", ""), language=None)

    # ── Step 2 search output records (UI) ────────────────────────────────────
    if ss("_show_step2_debug", False) and not _elm_mode_run:
        _srecs = ss("_search_output_records", [])
        if _srecs:
            with st.expander(
                f"Search outputs ({len(_srecs)} action(s) so far)", expanded=False
            ):
                st.caption(
                    f"Detailed files saved in `{SEARCH_OUTPUT_DIR}/`. "
                    "Use the download buttons below to access them from the browser."
                )
            for _sri, _sr in enumerate(_srecs):
                _sr_label = (
                    f"Search output: {_sr['company']} "
                    f"({'DRY RUN' if _sr.get('dry_run') else _sr.get('provider', '')})"
                )
                with st.expander(_sr_label, expanded=False):
                    st.markdown(
                        f"**Provider:** {_sr.get('provider', '')}  |  "
                        f"**Dry run:** {_sr.get('dry_run', False)}"
                    )
                    st.markdown("**Query / search instruction:**")
                    st.code(_sr.get("query", ""), language=None)
                    _rc = _sr.get("result_count", 0)
                    st.markdown(f"**Results returned:** {_rc}")
                    _tops = _sr.get("top_results", [])
                    if _tops:
                        st.markdown("**Top results:**")
                        for _t in _tops:
                            st.markdown(
                                f"- [{_t.get('title','(no title)')}]({_t.get('link','')})"
                                + (f"  — {_t.get('snippet','')[:120]}" if _t.get('snippet') else "")
                            )
                    _df = _sr.get("debug_file", "")
                    if _df:
                        st.caption(f"`{_df}`")
                        _rec_wrap = {
                            "path":    _df,
                            "company": _sr.get("company", ""),
                            "dry_run": _sr.get("dry_run", False),
                        }
                        _debug_file_download_button(_rec_wrap, f"run_{idx}_{_sri}")
                        _debug_file_preview_expander(_rec_wrap, f"prev_{idx}_{_sri}")

    # ── Step 2 debug log window ───────────────────────────────────────────────
    if ss("_show_step2_debug", False) and not _elm_mode_run:
        _log_text     = ss("_step2_debug_log", "")
        _prompt_recs  = ss("_step2_prompt_records", [])
        with st.expander("Step 2 Debug Log", expanded=True):
            if _log_text:
                st.code(_log_text, language=None)
            else:
                st.caption("No log entries yet — will appear as companies are processed.")
        for _pr in _prompt_recs:
            with st.expander(f"Prompt sent for {_pr['company']}", expanded=False):
                st.code(_pr.get("prompt", ""), language=None)

    _resume_mode = ss("_resume_mode", False)
    _saved_df    = autosave_load() if _resume_mode else None

    if ss("stop_requested", False) or idx >= _n:
        build_and_finish(results, debug_records, df_work, _active_fields)
    else:
        input_row    = df_work.iloc[idx]
        company_name = str(input_row.get(_name_col, "")).strip()
        raw_url      = str(input_row.get(_domain_col, "")).strip() if _domain_col else ""

        # ── Resume: skip rows already in autosave ─────────────────────────────
        if _resume_mode and autosave_already_done(
            _saved_df, _name_col, _domain_col, company_name, raw_url
        ):
            st.caption(
                f"⏭ Skipping row {idx + 1} / {_n}: "
                f"**{company_name or '(empty)'}** — already in autosave"
            )
            # Represent the skipped row with a minimal placeholder so build_and_finish
            # still has the right row count; the full data lives in the autosave CSV.
            skip_fields = {f: "" for f in _active_fields}
            if not _elm_mode_run:
                skip_fields["enrichment_status"] = "skipped_resume"
            results.append(skip_fields)
            debug_records.append({"input_company_name": company_name, "skipped": True})
            ss_set(results=results, debug_records=debug_records, process_index=idx + 1)
            st.rerun()

        # ── Build Step 2 debug callback if either debug option is enabled ────────
        _show_debug_ui = ss("_show_step2_debug", False)
        _save_debug_fs = ss("_save_step2_debug", True)

        def _make_step2_callback(cname: str, dry_run_mode: bool = False):
            def _cb(event: str, **kwargs) -> None:
                if event == "status":
                    append_debug_log(kwargs.get("msg", ""))
                elif event == "search_output":
                    _so_rec = {
                        "company":      kwargs.get("company", cname),
                        "provider":     kwargs.get("provider", ""),
                        "query":        kwargs.get("query", ""),
                        "result_count": kwargs.get("result_count", 0),
                        "top_results":  kwargs.get("top_results", []),
                        "debug_file":   kwargs.get("debug_file", ""),
                        "dry_run":      kwargs.get("dry_run", False),
                    }
                    if _show_debug_ui:
                        _srecs = st.session_state.get("_search_output_records", [])
                        _srecs.append(_so_rec)
                        st.session_state["_search_output_records"] = _srecs
                    # Always track search output files for download (when saving is on)
                    _so_path = kwargs.get("debug_file", "")
                    if _so_path and _save_debug_fs:
                        try:
                            _all_files = st.session_state.get("_step2_debug_files", [])
                            _all_files.append({
                                "path":     _so_path,
                                "filename": Path(_so_path).name,
                                "company":  kwargs.get("company", cname),
                                "provider": kwargs.get("provider", ""),
                                "kind":     "search",
                                "dry_run":  kwargs.get("dry_run", False),
                            })
                            st.session_state["_step2_debug_files"] = _all_files
                        except Exception:
                            pass
                elif event == "prompt":
                    _is_dry   = kwargs.get("dry_run", False)
                    _file_pfx = "step2_dry_run" if _is_dry else "step2_prompt"
                    if _save_debug_fs:
                        _ts   = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                        _body = format_step2_debug_content(
                            company_name=kwargs.get("company", cname),
                            model=kwargs.get("model", ""),
                            timestamp=_ts,
                            provider=kwargs.get("provider", STEP2_PROVIDER_CLAUDE),
                            search_prompt=kwargs.get("search_prompt", ""),
                            full_prompt=kwargs.get("full_prompt", ""),
                            notes=kwargs.get("notes", []),
                        )
                        try:
                            _prompt_path = write_debug_log(cname, _body, prefix=_file_pfx)
                            _all_files = st.session_state.get("_step2_debug_files", [])
                            _all_files.append({
                                "path":     str(_prompt_path),
                                "filename": _prompt_path.name,
                                "company":  kwargs.get("company", cname),
                                "provider": kwargs.get("provider", STEP2_PROVIDER_CLAUDE),
                                "kind":     "prompt_dry_run" if _is_dry else "prompt",
                                "dry_run":  _is_dry,
                            })
                            st.session_state["_step2_debug_files"] = _all_files
                        except Exception:
                            pass
                    if _show_debug_ui:
                        _recs = st.session_state.get("_step2_prompt_records", [])
                        _recs.append({
                            "company":       kwargs.get("company", cname),
                            "prompt":        kwargs.get("full_prompt", ""),
                            "search_prompt": kwargs.get("search_prompt", ""),
                        })
                        st.session_state["_step2_prompt_records"] = _recs
                    # Always accumulate dry-run records for the preview section
                    if dry_run_mode or _is_dry:
                        _drecs = st.session_state.get("_dry_run_records", [])
                        _drecs.append({
                            "company":       kwargs.get("company", cname),
                            "provider":      kwargs.get("provider", ""),
                            "model":         kwargs.get("model", ""),
                            "queries":       kwargs.get("queries", []),
                            "search_prompt": kwargs.get("search_prompt", ""),
                            "full_prompt":   kwargs.get("full_prompt", ""),
                        })
                        st.session_state["_dry_run_records"] = _drecs
            return _cb

        _debug_cb = (
            _make_step2_callback(company_name, dry_run_mode=_dry_run_run)
            if (_show_debug_ui or _save_debug_fs or _dry_run_run) and not _elm_mode_run
            else None
        )

        with st.status(
            f"Row {idx + 1} / {_n}: **{company_name or '(empty)'}**",
            expanded=False,
        ) as status_box:
            if _elm_mode_run:
                status_box.write("⚡ Extreme Light Mode — fetching pages…")
                fields, dbg = enrich_one_row_light(company_name, raw_url)
                _fetch_status = fields.get("elm_fetch_status", "?")
                status_box.write(
                    f"✅ Done — fetch: {_fetch_status} | "
                    f"{int(fields.get('elm_total_chars', 0) or 0):,} chars | "
                    f"score: {fields.get('elm_score_overall_icp', '?')}/10"
                )
                row_cost = 0.0
            else:
                status_box.write(f"🤖 Step 2 model: `{_model_step2_run}`")
                _prov_label = (
                    f"🔍 Step 2 search provider: {_step2_provider_run}"
                    + (" **(ZERO-COST PREVIEW)**" if (_zero_cost_run and _dry_run_run) else
                       " **(DRY RUN)**" if _dry_run_run else "")
                )
                status_box.write(_prov_label)

                # ── ZERO-COST PREVIEW GUARD ───────────────────────────────────
                # Do NOT call Jina, Claude, Serper, browser scraping, or any
                # external API. Use only uploaded row data and existing cache.
                if _zero_cost_run and _dry_run_run:
                    status_box.write(
                        "⚡ Zero-cost preview — skipping all Step 1 API calls, "
                        "running Step 2 dry run from row data only…"
                    )
                    # Build a minimal debug record (no real API calls)
                    dbg = {"input_company_name": company_name, "zero_cost_preview": True}
                    # Call run_step2 with dry_run=True; no Jina/browser scraping
                    icp_fields, _step2_cache, _s2_ti, _s2_to, _s2_status, _s2_msg, _s2_cr, _s2_cc = run_step2(
                        url=raw_url,
                        company_name=company_name,
                        api_key="",
                        delay=_delay,
                        model_step2=_model_step2_run,
                        _debug_callback=_debug_cb,
                        search_provider=_step2_provider_run,
                        serper_key="",
                        dry_run=True,
                    )
                    fields = {f: "" for f in ALL_ENRICHMENT_FIELDS}
                    fields["enrichment_status"] = "zero_cost_preview"
                    fields["step2_status"]      = _s2_status
                    fields.update(icp_fields)
                    # Increment the preview counter
                    ss_set(_dry_run_preview_count=ss("_dry_run_preview_count", 0) + 1)
                    row_cost = 0.0
                    status_box.write("✅ Zero-cost preview done — no tokens used.")
                # ── END ZERO-COST PREVIEW GUARD ──────────────────────────────
                else:
                    status_box.write("⏳ Step 1 — Fetching page + extracting firmographics…")
                    fields, dbg = enrich_one_row(
                        company_name, raw_url, _api_key, _delay,
                        use_playwright=_use_playwright_run,
                        model_step1=_model_step1_run,
                        model_step2=_model_step2_run,
                        _debug_callback=_debug_cb,
                        search_provider=_step2_provider_run,
                        serper_key=_serper_key_run,
                        dry_run=_dry_run_run,
                        enable_lusha_api=_enable_lusha_api_run,
                        lusha_api_key=_lusha_api_key_run,
                    )
                if not (_zero_cost_run and _dry_run_run):
                    s1_tok   = int(fields.get("step1_tokens_in",  0) or 0) + int(fields.get("step1_tokens_out", 0) or 0)
                    s2_tok   = int(fields.get("step2_tokens_in",  0) or 0) + int(fields.get("step2_tokens_out", 0) or 0)
                    row_cost = float(fields.get("total_cost_usd", 0) or 0)
                    _retry_note = ""
                    if "enriched_search" in fields.get("enrichment_status", ""):
                        _retry_note = " | ⚡ Google fallback used"
                    elif ss("_last_retry_msg", ""):
                        _retry_note = " | ⏳ Had Jina 429 retry"
                    if fields.get("step2_status") == "api_error":
                        status_box.write(
                            f"⚠️ Step 2 API error: {fields.get('error_message', '(no detail)')}"
                        )
                    status_box.write(
                        f"✅ Done — Step 1: {s1_tok} tokens | Step 2: {s2_tok} tokens | "
                        f"Row cost: ${row_cost:.5f}{_retry_note}"
                    )

        results.append(fields)
        debug_records.append(dbg)

        # ── Auto-save (crash recovery) — skipped in dry run / zero-cost preview ──
        if not _dry_run_run:
            try:
                autosave_append(fields, input_row)
                ss_set(autosave_last_name=company_name or raw_url or f"row {idx + 1}")
            except Exception:
                pass  # never let autosave failure abort processing

        _new_idx = len(results)  # results already includes the row appended above

        # ── Filesystem snapshot every LOCAL_SAVE_EVERY companies (local runs) ─
        if ss("_local_save_enabled", False) and _new_idx % LOCAL_SAVE_EVERY == 0:
            _local_path = ss("_local_save_path", "") or _DEFAULT_DOWNLOAD_DIR
            try:
                _snap_df = build_partial_df(results, df_work, _active_fields)
                _xl, _csv = save_to_local_folder(_snap_df, _local_path, run_tag=build_run_tag())
                ss_set(_last_local_save=f"{_new_idx} rows → {Path(_xl).name}")
            except Exception as _e:
                ss_set(_last_local_save=f"⚠ Save failed: {_e}")

        # ── Auto browser-download every _AUTO_DL_EVERY companies ─────────────
        _auto_dl_done = ss("_auto_dl_count", 0)
        if _new_idx % _AUTO_DL_EVERY == 0 and _new_idx // _AUTO_DL_EVERY > _auto_dl_done:
            _dl_snap = build_partial_df(results, df_work, _active_fields)
            _dl_name = f"enriched_snapshot_{_new_idx}.xlsx"
            _js_auto_download(_dl_snap, _dl_name)
            ss_set(
                _auto_dl_count=_new_idx // _AUTO_DL_EVERY,
                _auto_dl_last_msg=f"Auto-downloaded at row {_new_idx} → {_dl_name}",
            )

        try:
            total_in   += int(fields.get("total_tokens_in",  0) or 0)
            total_out  += int(fields.get("total_tokens_out", 0) or 0)
            total_cost += float(fields.get("total_cost_usd", 0) or 0)
        except (ValueError, TypeError):
            pass

        # Accumulate cache token counts from the debug record of this row
        _dbg_last = debug_records[-1] if debug_records else {}
        try:
            total_cache_read   += int(_dbg_last.get("step2_cache_read_tokens",   0) or 0)
            total_cache_create += int(_dbg_last.get("step2_cache_creation_tokens", 0) or 0)
        except (ValueError, TypeError):
            pass

        ss_set(
            results=results, debug_records=debug_records, process_index=idx + 1,
            total_tokens_in=total_in, total_tokens_out=total_out, total_cost_usd=total_cost,
            total_cache_read_tokens=total_cache_read,
            total_cache_create_tokens=total_cache_create,
        )
        st.rerun()

# =============================================================================
# RESULTS
# =============================================================================

if ss("enrichment_done", False):
    df_enriched: pd.DataFrame = ss("df_enriched")
    debug_records_done: list  = ss("debug_records", [])
    processed = len(df_enriched)
    _elm_done = ss("_elm_mode", False)
    _done_fields = ELM_ALL_FIELDS if _elm_done else ALL_ENRICHMENT_FIELDS

    st.divider()
    if ss("stop_requested", False):
        st.warning(f"Enrichment stopped after **{processed}** rows. Partial results below.")
    else:
        st.success(f"✅ Enrichment complete — **{processed:,}** rows processed.")

    _done_dry_run   = ss("_step2_dry_run",     False)
    _done_zero_cost = ss("_zero_cost_preview", False)
    if _done_zero_cost and _done_dry_run and not _elm_done:
        _preview_count = ss("_dry_run_preview_count", 0)
        st.info(
            f"ℹ️ **Zero-cost preview completed.** No Step 1 or Step 2 API calls were made — "
            f"**{_preview_count}** dry-run previews generated. "
            "Disable zero-cost preview and dry run, then re-run to perform real enrichment."
        )
    elif _done_dry_run and not _elm_done:
        st.info(
            "ℹ️ **Dry run completed.** No Step 2 enrichment results were written — "
            "Step 2 ICP columns are empty. Disable dry run and re-run to perform real enrichment."
        )

    # ── Auto-save final file (runs exactly once per completed run) ────────────
    if not ss("_final_auto_saved", False):
        _save_enabled = ss("_local_save_enabled", False)
        _save_dir     = ss("_local_save_path", "") if _save_enabled else _DEFAULT_DOWNLOAD_DIR
        _save_dir     = _save_dir or _DEFAULT_DOWNLOAD_DIR
        try:
            _final_xl, _ = save_to_local_folder(df_enriched, _save_dir, run_tag=build_run_tag())
            ss_set(_final_auto_saved=True, _final_save_path=_final_xl)
        except Exception as _save_err:
            ss_set(_final_auto_saved=True, _final_save_path="",
                   _final_save_error=str(_save_err))

    _final_xl_path  = ss("_final_save_path", "")
    _final_xl_error = ss("_final_save_error", "")
    if _final_xl_path:
        st.info(f"📥 Results also saved locally to **{_final_xl_path}**")
    elif _final_xl_error:
        st.warning(f"⚠ Local auto-save failed: {_final_xl_error}")

    # ── Step 2 debug files — download + preview ───────────────────────────────
    if not _elm_done:
        _all_dbg_files = ss("_step2_debug_files", [])
        if _all_dbg_files:
            with st.expander(
                f"🔍 Step 2 debug files ({len(_all_dbg_files)} file(s))",
                expanded=True,
            ):
                # ── ZIP download (all files in one click) ─────────────────────
                try:
                    _zip_bytes = build_debug_zip(_all_dbg_files)
                    st.download_button(
                        label="⬇ Download all Step 2 debug files as ZIP",
                        data=_zip_bytes,
                        file_name=f"step2_debug_{build_run_tag()}_{ts()}.zip",
                        mime="application/zip",
                        use_container_width=True,
                        key="dl_all_debug_zip",
                    )
                except Exception as _ze:
                    st.warning(f"Could not build ZIP: {_ze}")

                st.divider()

                # ── Per-file: download button + preview ───────────────────────
                for _fi, _frec in enumerate(_all_dbg_files):
                    _tag  = " [DRY RUN]" if _frec.get("dry_run") else ""
                    _kind = _frec.get("kind", "")
                    _kind_label = {
                        "prompt":          "Prompt file",
                        "prompt_dry_run":  "Prompt file (dry run)",
                        "search":          "Search I/O file",
                    }.get(_kind, "Debug file")
                    st.markdown(
                        f"**{_kind_label}{_tag}** — {_frec.get('company', '')} "
                        f"· {_frec.get('provider', '')}"
                    )
                    st.caption(f"`{_frec.get('filename','')}`")
                    _debug_file_download_button(_frec, f"done_{_fi}")
                    _debug_file_preview_expander(_frec, f"done_{_fi}")
                    if _fi < len(_all_dbg_files) - 1:
                        st.divider()

    # ── Primary browser download ──────────────────────────────────────────────
    if _elm_done:
        _fname_dl = f"elm_results_{ts()}.xlsx"
    else:
        _fname_dl = f"enriched_results_{build_run_tag()}_{ts()}.xlsx"
    st.download_button(
        label="⬇ Download results to your local Downloads folder",
        data=df_to_excel_bytes(df_enriched),
        file_name=_fname_dl,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        type="primary",
    )

    # ── Status summary ────────────────────────────────────────────────────────
    status_counts  = (
        df_enriched["elm_fetch_status"].value_counts().to_dict()
        if _elm_done and "elm_fetch_status" in df_enriched.columns
        else df_enriched.get("enrichment_status", pd.Series(dtype=str)).value_counts().to_dict()
    )
    needs_review_n = (
        int((df_enriched["needs_manual_review"] == "TRUE").sum())
        if "needs_manual_review" in df_enriched.columns else 0
    )
    all_sc = list(status_counts.items())
    web_search_n = (
        int((df_enriched["step2_status"] == "ok").sum())
        if "step2_status" in df_enriched.columns else 0
    )
    competitor_n = (
        int(df_enriched["icp_competitor_signal"].astype(str).str.strip().ne("").sum())
        if "icp_competitor_signal" in df_enriched.columns else 0
    )
    # In ELM mode there is no "Needs review" concept
    extra_metrics = [] if _elm_done else [("⚑ Needs review", needs_review_n)]
    extra_metrics += [("🔍 Web search used", web_search_n), ("🏁 Competitor found", competitor_n)]
    n_cols = min(len(all_sc) + len(extra_metrics), 6)
    if all_sc:
        rcols = st.columns(max(n_cols, 1))
        for i, (s, c) in enumerate(all_sc):
            rcols[i % len(rcols)].metric(_STATUS_LABELS.get(s, s), c)
        for j, (label, val) in enumerate(extra_metrics):
            rcols[(len(all_sc) + j) % len(rcols)].metric(label, val)

    # ── Token usage ───────────────────────────────────────────────────────────
    t_in   = ss("total_tokens_in", 0)
    t_out  = ss("total_tokens_out", 0)
    t_cost = ss("total_cost_usd", 0.0)

    if not _elm_done:
        with st.expander("💰 Token usage & cost", expanded=True):
            tc1, tc2, tc3 = st.columns(3)
            tc1.metric("Total input tokens",   f"{t_in:,}")
            tc2.metric("Total output tokens",  f"{t_out:,}")
            tc3.metric("Estimated total cost", f"${t_cost:.4f}")

            _used_s1 = ss("_model_step1", MODEL_STEP1)
            _used_s2 = ss("_model_step2", MODEL_STEP2)
            st.caption(
                f"Step 1 model: `{_used_s1}` · Step 2 model: `{_used_s2}`. "
                "Two API calls per row (Step 1 + Step 2). "
                "Verify charges in your Anthropic dashboard."
            )

            _cache_read   = ss("total_cache_read_tokens",   0)
            _cache_create = ss("total_cache_create_tokens", 0)
            if _cache_read > 0 or _cache_create > 0:
                st.divider()
                # Input price per M for the Step 2 model (approximate)
                _s2_input_price_per_m = (
                    0.80 if "haiku"  in _used_s2 else
                    3.00 if "sonnet" in _used_s2 else 1.00
                )
                # Cache reads cost 10 % of normal input price; savings = 90 %
                _savings_usd = _cache_read * _s2_input_price_per_m * 0.90 / 1_000_000
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("Cache write tokens (Step 2)",  f"{_cache_create:,}",
                           help="Tokens written to Anthropic's prompt cache on the first company.")
                cc2.metric("Cache read tokens (Step 2)",   f"{_cache_read:,}",
                           help="Tokens served from cache at 10 % of normal input cost.")
                cc3.metric("Est. prompt-cache savings",    f"${_savings_usd:.4f}",
                           help="90 % discount on cache-read tokens vs full input price.")
                st.caption(
                    f"Prompt caching active on Step 2 ({_used_s2}). "
                    f"Static prefix cached once; each subsequent company reads it at "
                    f"~${_s2_input_price_per_m * 0.10:.3f}/M tokens instead of "
                    f"${_s2_input_price_per_m:.2f}/M."
                )
    else:
        st.info("⚡ Extreme Light Mode — no API calls, no tokens, no cost.")

    # ── Results table ─────────────────────────────────────────────────────────
    st.subheader("Results")
    orig_cols = [c for c in df_enriched.columns if c not in _done_fields]

    if _elm_done:
        summary_cols = orig_cols + [c for c in ELM_ALL_FIELDS if c in df_enriched.columns]
        st.dataframe(df_enriched[summary_cols], use_container_width=True, height=400)
        tab1, tab2, tab3 = st.tabs(
            ["Status & fetch info", "Keyword counts", "Normalized scores"]
        )
        with tab1:
            st.dataframe(
                df_enriched[[c for c in ELM_STATUS_FIELDS if c in df_enriched.columns]],
                use_container_width=True,
            )
        with tab2:
            st.dataframe(
                df_enriched[[c for c in ELM_KEYWORD_FIELDS if c in df_enriched.columns]],
                use_container_width=True,
            )
        with tab3:
            st.dataframe(
                df_enriched[[c for c in ELM_SCORE_FIELDS if c in df_enriched.columns]],
                use_container_width=True,
            )
    else:
        _lusha_done = ss("_enable_lusha_api", False)
        summary_cols = orig_cols + [
            c for c in [
                # Metadata
                "enrichment_status", "step1_status", "step2_status",
                "needs_manual_review", "match_notes",
                # Lusha API (real) — key fields only in summary
                "lusha_api_status", "lusha_api_match_confidence", "lusha_api_needs_review",
                "lusha_api_company_name", "lusha_api_domain", "lusha_api_industry",
                "lusha_api_employee_range", "lusha_api_country",
                # Step 1 — firmographics
                "lusha_company_name", "lusha_domain", "lusha_industry", "lusha_sub_industry",
                "lusha_company_type", "lusha_employee_range", "lusha_revenue",
                "lusha_country", "lusha_city", "lusha_continent",
                "lusha_founded_year", "lusha_description",
                "lusha_linkedin_url", "lusha_specialties", "lusha_technologies",
                "lusha_total_funding_amount", "lusha_total_funding_rounds",
                "lusha_last_round_type", "lusha_last_round_amount", "lusha_last_round_date",
                "lusha_ipo_status",
                # Step 2 — ICP buying signals
                "icp_lead_score", "icp_buying_signals", "icp_competitor_signal",
                "icp_evidence", "icp_likely_training_interest",
                "icp_why_relevant", "icp_potential_buyer_function",
                # Cost
                "total_tokens_in", "total_tokens_out", "total_cost_usd",
                "error_message",
            ]
            if c in df_enriched.columns
        ]
        st.dataframe(df_enriched[summary_cols], use_container_width=True, height=400)
        _tabs = ["Step 1 — All firmographic columns", "Step 2 — All ICP columns"]
        if _lusha_done:
            _tabs.append("Lusha API fields")
        _tab_objs = st.tabs(_tabs)
        with _tab_objs[0]:
            st.dataframe(df_enriched[[c for c in STEP1_FIELDS if c in df_enriched.columns]],
                         use_container_width=True)
        with _tab_objs[1]:
            st.dataframe(df_enriched[[c for c in ICP_FIELDS if c in df_enriched.columns]],
                         use_container_width=True)
        if _lusha_done and len(_tab_objs) > 2:
            with _tab_objs[2]:
                _lusha_display_cols = [
                    c for c in LUSHA_API_FIELDS + LUSHA_API_META_FIELDS
                    if c in df_enriched.columns
                ]
                st.dataframe(df_enriched[_lusha_display_cols], use_container_width=True)

    # ── Downloads ─────────────────────────────────────────────────────────────
    st.subheader("Download results")
    log_df = make_log_df(debug_records_done, elm_mode=_elm_done)

    if _elm_done:
        _fname_prefix = "elm_results"
        _log_fname    = "elm_fetch_log.csv"
    else:
        _run_tag      = build_run_tag()
        _fname_prefix = f"enriched_results_{_run_tag}_{ts()}"
        _log_fname    = f"processing_log_{_run_tag}_{ts()}.csv"
    _xl_help      = (
        "All original columns + keyword counts + normalized scores."
        if _elm_done else
        "All original columns + Step 1 firmographics + Step 2 ICP signals + metadata."
    )
    _log_help     = (
        "One row per company: fetch status, pages fetched, total chars."
        if _elm_done else
        "One row per company: step statuses, token counts, costs, review flags."
    )

    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button(
            "⬇ Results Excel (.xlsx)",
            data=df_to_excel_bytes(df_enriched),
            file_name=f"{_fname_prefix}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            help=_xl_help,
        )
    with dl2:
        st.download_button(
            "⬇ Results CSV",
            data=df_to_csv_bytes(df_enriched),
            file_name=f"{_fname_prefix}.csv",
            mime="text/csv",
            use_container_width=True,
            help="Same data as Excel, in CSV format.",
        )
    with dl3:
        st.download_button(
            "⬇ Processing log CSV",
            data=df_to_csv_bytes(log_df),
            file_name=_log_fname,
            mime="text/csv",
            use_container_width=True,
            help=_log_help,
        )

    # ── Debug section ─────────────────────────────────────────────────────────
    if debug_mode and debug_records_done:
        st.divider()
        st.subheader("🐛 Per-row debug details")

        if _elm_done:
            debug_df = pd.DataFrame([
                {
                    "row":          i + 1,
                    "company":      d.get("company", ""),
                    "url":          d.get("url", ""),
                    "domain":       d.get("domain", ""),
                    "fetch_status": d.get("status", ""),
                    "pages_fetched": d.get("pages_fetched", ""),
                    "total_chars":  d.get("total_chars", ""),
                }
                for i, d in enumerate(debug_records_done)
            ])
        else:
            debug_df = pd.DataFrame([
                {
                    "row":               i + 1,
                    "company":           d.get("input_company_name", ""),
                    "url":               d.get("input_url", ""),
                    "lusha_api_status":  d.get("lusha_api_status", ""),
                    "step1_status":      d.get("step1_status", ""),
                    "step2_status":      d.get("step2_status", ""),
                    "enrichment_status": d.get("enrichment_status", ""),
                    "step1_tok_in":      d.get("step1_tokens_in", ""),
                    "step1_tok_out":     d.get("step1_tokens_out", ""),
                    "step2_tok_in":      d.get("step2_tokens_in", ""),
                    "step2_tok_out":     d.get("step2_tokens_out", ""),
                    "row_cost":          f"${d.get('total_cost', 0):.5f}",
                    "error":             d.get("error_message", ""),
                }
                for i, d in enumerate(debug_records_done)
            ])
        st.dataframe(debug_df, use_container_width=True)

        if not _elm_done:
            st.subheader("🐛 Raw JSON responses")
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
                d = debug_records_done[sel_idx]
                col_lusha, col_a, col_b = st.columns(3)
                with col_lusha:
                    st.markdown("**Lusha API — real company data**")
                    st.caption(f"Status: `{d.get('lusha_api_status', '(not run)')}`")
                    if d.get("lusha_api_raw_json"):
                        st.json(d["lusha_api_raw_json"])
                    else:
                        st.info("No Lusha API response (disabled or no data).")
                with col_a:
                    st.markdown("**Step 1 — Jina + Claude extraction**")
                    if d.get("step1_raw_json"):
                        st.json(d["step1_raw_json"])
                    else:
                        st.info("No Step 1 response.")
                with col_b:
                    st.markdown("**Step 2 — Claude web_search ICP signals**")
                    if d.get("step2_raw_json"):
                        st.json(d["step2_raw_json"])
                    else:
                        st.info("No Step 2 response.")

        st.subheader("🐛 Additional debug downloads")
        dbg_dl1, dbg_dl2 = st.columns(2)
        with dbg_dl1:
            debug_enriched = df_enriched.copy()
            if not _elm_done:
                debug_enriched["step1_json_preview"] = [
                    json.dumps(d.get("step1_raw_json"), ensure_ascii=False)[:1500]
                    if d.get("step1_raw_json") else ""
                    for d in debug_records_done
                ] + [""] * max(0, len(debug_enriched) - len(debug_records_done))
                debug_enriched["step2_json_preview"] = [
                    json.dumps(d.get("step2_raw_json"), ensure_ascii=False)[:1500]
                    if d.get("step2_raw_json") else ""
                    for d in debug_records_done
                ] + [""] * max(0, len(debug_enriched) - len(debug_records_done))
            _dbg_fname = "elm_debug.xlsx" if _elm_done else "claude_enriched_debug.xlsx"
            st.download_button(
                "⬇ Debug Excel",
                data=df_to_excel_bytes(debug_enriched),
                file_name=_dbg_fname,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with dbg_dl2:
            cc = get_cache_count()
            if cc > 0:
                st.download_button(
                    f"⬇ Cache ZIP ({cc} files)",
                    data=cache_to_zip_bytes(),
                    file_name="claude_cache.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
            else:
                st.info("Cache is empty.")

        st.subheader("🐛 Cache viewer")
        st.caption(f"{CACHE_DIR.resolve()} — {get_cache_count()} file(s)")
        cache_files = list_cache_files()
        if cache_files:
            sel_cache = st.selectbox(
                "Select cached file:",
                options=[f.stem for f in cache_files],
                key="cache_file_selector",
            )
            if sel_cache:
                try:
                    st.json(json.loads(
                        (CACHE_DIR / f"{sel_cache}.json").read_text(encoding="utf-8")
                    ))
                except Exception as exc:
                    st.error(f"Could not read cache file: {exc}")
        else:
            st.info("Cache is empty. Run an enrichment first.")

    # ── Restart ───────────────────────────────────────────────────────────────
    st.divider()
    if st.button("↺ Start a new enrichment", use_container_width=True, key="restart_btn"):
        reset_processing(clear_autosave=True)
        st.rerun()
