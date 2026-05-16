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
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

JINA_READER_URL  = "https://r.jina.ai/"
JINA_SEARCH_URL  = "https://s.jina.ai/"
CACHE_DIR        = Path("claude_json_cache")
AUTOSAVE_PATH    = "/tmp/enrichment_autosave.csv"
LOCAL_SAVE_EVERY = 5   # write to user's local folder every N companies
MODEL_ID         = "claude-haiku-4-5-20251001"
WEB_SEARCH_TOOL  = {"type": "web_search_20250305", "name": "web_search"}

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

# Step 2 ICP research prompt
_STEP2_PROMPT_TMPL = (
    "Research the company at {url}. "
    "Find signals that indicate whether this company would benefit from corporate language training. "
    "Look for: international presence, languages used, global teams, hiring activity, "
    "recent growth or funding. "
    "Return ONLY raw JSON (no markdown, no code fences) with exactly these fields: "
    "international_presence (number of countries with offices, e.g. '5' or 'global' or '1'), "
    "languages_mentioned (comma-separated languages found on site, e.g. 'English, French, German'), "
    "hiring_activity ('yes - [brief evidence]' or 'no'), "
    "recent_funding ('yes - [amount]' or 'no'), "
    "recent_news ('yes - [brief summary of expansion/acquisition/IPO in last 12 months]' or 'none'), "
    "global_team_signals ('yes - [brief details]' or 'no'), "
    "training_signals ('yes - [brief details about training/learning/upskilling mentions]' or 'no'), "
    "multi_office ('yes - [count] offices' or 'no'), "
    "language_training_fit_score (integer 1-10, where 10 = highest fit for language training)."
)

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

# Step 2: Mingle ICP signals
ICP_FIELDS = [
    "icp_international_presence",
    "icp_languages_mentioned",
    "icp_hiring_activity",
    "icp_recent_funding",
    "icp_recent_news",
    "icp_global_team_signals",
    "icp_training_signals",
    "icp_multi_office",
    "icp_language_training_fit_score",
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

ALL_ENRICHMENT_FIELDS = STEP1_FIELDS + ICP_FIELDS + META_FIELDS

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
    Fetch the homepage AND about-us page; return whichever has more content.
    Both must be ≥ _JINA_MIN_CONTENT chars to count — shorter responses are
    treated as blocked/empty and raise requests.HTTPError so the caller can
    fall through to the Google tier.
    """
    base    = url.rstrip("/")
    best    = ""

    # Try homepage
    try:
        text = _jina_get_with_retry(url, company_hint)
        if len(text) > len(best):
            best = text
    except requests.HTTPError:
        pass

    # Try about-us slugs
    for slug in _JINA_ABOUT_SLUGS:
        try:
            text = _jina_get_with_retry(base + slug, company_hint)
            if len(text) > len(best):
                best = text
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if code not in (403, 404):
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

def _claude_extract(webpage_text: str, api_key: str) -> tuple:
    """
    Send page text to Claude for structured extraction.
    Returns (raw_fields_dict, input_tokens, output_tokens).
    """
    client    = anthropic.Anthropic(api_key=api_key)
    truncated = webpage_text[:_JINA_CHAR_LIMIT]
    msg = client.messages.create(
        model=MODEL_ID,
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


def run_step1(url: str, company_name: str, api_key: str, delay: float) -> tuple:
    """
    Three-tier Step 1 enrichment:
      Tier 1 — Jina direct scrape       → status 'enriched_jina'
      Tier 2 — Claude web_search        → status 'enriched_search'
      Tier 3 — Both failed              → status 'no_data'
    Returns (step1_fields, raw_json, in_tok, out_tok, status, error_msg).
    """
    source_url = normalize_url(url) if url else ""
    jina_err   = ""
    total_in   = total_out = 0

    # ── Tier 1: Jina direct scrape ────────────────────────────────────────────
    if source_url:
        ck = f"step1_url_{source_url}"
        cached = load_cache(ck)
        if cached is not None:
            fields = _map_step1_fields(cached.get("claude_data", {}), source_url)
            if _step1_has_data(fields):
                return (fields, cached,
                        int(cached.get("tokens_in",  0) or 0),
                        int(cached.get("tokens_out", 0) or 0),
                        "enriched_jina", "")
            _delete_cache(ck)

        try:
            time.sleep(delay)
            text = fetch_via_jina_reader(source_url, company_hint=company_name)
            raw_fields, in_t, out_t = _claude_extract(text, api_key)
            total_in  += in_t
            total_out += out_t
            payload = {"claude_data": raw_fields, "tokens_in": in_t, "tokens_out": out_t}
            save_cache(ck, payload)
            fields = _map_step1_fields(raw_fields, source_url)
            if _step1_has_data(fields):
                return (fields, payload, total_in, total_out, "enriched_jina", "")
            jina_err = "Jina page fetched but Claude found no usable data"
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            jina_err = f"Jina HTTP {code}: {str(e)[:120]}"
        except (json.JSONDecodeError, ValueError) as e:
            jina_err = f"Jina parse error: {e}"
        except anthropic.APIError as e:
            return ({}, {}, total_in, total_out, "api_error", f"Claude API: {e}")
        except Exception as e:
            jina_err = str(e)

    # ── Tier 2: Claude web_search fallback ────────────────────────────────────
    target = source_url or company_name
    if not target:
        return ({}, {}, total_in, total_out, "no_data", "No URL or company name provided")

    ck = f"step1_fallback_{target}"
    cached = load_cache(ck)
    if cached is not None:
        fields = _map_step1_fields(cached.get("claude_data", {}), url)
        if _step1_has_data(fields):
            return (fields, cached,
                    int(cached.get("tokens_in",  0) or 0),
                    int(cached.get("tokens_out", 0) or 0),
                    "enriched_search", "")
        _delete_cache(ck)

    try:
        prompt = _STEP1_FALLBACK_PROMPT_TMPL.format(
            company_name=company_name or target,
            url=target,
        )
        raw_text, in_t, out_t = _claude_web_search_loop(prompt, api_key)
        total_in  += in_t
        total_out += out_t
        raw_fields = _parse_json_response(raw_text)
        payload    = {"claude_data": raw_fields, "tokens_in": in_t, "tokens_out": out_t}
        save_cache(ck, payload)
        fields = _map_step1_fields(raw_fields, url)
        if _step1_has_data(fields):
            return (fields, payload, total_in, total_out, "enriched_search", "")
        # Tier 3 — web search returned data but nothing useful
        return ({}, {}, total_in, total_out, "no_data",
                f"Google search returned no usable data. Jina: {jina_err}")
    except (json.JSONDecodeError, ValueError) as e:
        return ({}, {}, total_in, total_out, "no_data",
                f"Search parse error: {e}. Jina: {jina_err}")
    except anthropic.APIError as e:
        return ({}, {}, total_in, total_out, "api_error", f"Claude API: {e}")
    except Exception as e:
        return ({}, {}, total_in, total_out, "no_data",
                f"Search error: {e}. Jina: {jina_err}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — ICP signals via Claude web_search
# ─────────────────────────────────────────────────────────────────────────────

_ICP_EMPTY = {f: "" for f in ICP_FIELDS}


def _claude_web_search_loop(prompt: str, api_key: str) -> tuple:
    """
    Run Claude with web_search_20250305 tool in an agentic loop.
    Returns (final_text, total_input_tokens, total_output_tokens).
    """
    client   = anthropic.Anthropic(api_key=api_key)
    messages = [{"role": "user", "content": prompt}]
    total_in = total_out = 0

    for _iteration in range(10):
        resp = client.messages.create(
            model=MODEL_ID,
            max_tokens=2048,
            tools=[WEB_SEARCH_TOOL],
            messages=messages,
        )
        total_in  += resp.usage.input_tokens
        total_out += resp.usage.output_tokens

        if resp.stop_reason == "end_turn":
            text = "".join(getattr(b, "text", "") for b in resp.content).strip()
            return text, total_in, total_out

        if resp.stop_reason == "tool_use":
            # Add assistant turn; API executed the search server-side
            messages.append({"role": "assistant", "content": resp.content})
            tool_results = []
            for b in resp.content:
                if getattr(b, "type", "") == "tool_use":
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": b.id,
                        "content": "",
                    })
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
        else:
            # Unexpected stop — return whatever text exists
            text = "".join(getattr(b, "text", "") for b in resp.content).strip()
            return text, total_in, total_out

    return "", total_in, total_out


def run_step2(url: str, company_name: str, api_key: str, delay: float) -> tuple:
    """
    Research ICP signals using Claude with web_search.
    Returns (icp_fields_dict, raw_json, in_tok, out_tok, status, error_msg).
    """
    target = normalize_url(url) if url else company_name
    if not target:
        return (_ICP_EMPTY.copy(), {}, 0, 0, "no_input", "No URL or company name")

    ck = f"step2_{target}"
    cached = load_cache(ck)
    if cached is not None:
        icp = cached.get("icp_data", {})
        if any(icp.get(f, "") for f in ICP_FIELDS[:3]):  # basic sanity check
            in_t  = int(cached.get("tokens_in", 0) or 0)
            out_t = int(cached.get("tokens_out", 0) or 0)
            return (_extract_icp_fields(icp), cached, in_t, out_t, "cached", "")
        _delete_cache(ck)

    _STRICT_SUFFIX = (
        "\n\nReply with ONLY a JSON object, no explanation, no markdown, no backticks."
    )
    try:
        time.sleep(delay)
        prompt   = _STEP2_PROMPT_TMPL.format(url=target)
        raw_text, in_t, out_t = _claude_web_search_loop(prompt, api_key)
        try:
            icp_raw = _parse_json_response(raw_text)
        except (json.JSONDecodeError, ValueError):
            # Retry once with a stricter prompt appended
            time.sleep(delay)
            raw_text2, in_t2, out_t2 = _claude_web_search_loop(prompt + _STRICT_SUFFIX, api_key)
            in_t  += in_t2
            out_t += out_t2
            icp_raw = _parse_json_response(raw_text2)   # raises if still bad
        payload = {"icp_data": icp_raw, "tokens_in": in_t, "tokens_out": out_t}
        save_cache(ck, payload)
        return (_extract_icp_fields(icp_raw), payload, in_t, out_t, "ok", "")
    except (json.JSONDecodeError, ValueError) as e:
        return (_ICP_EMPTY.copy(), {}, 0, 0, "parse_error", f"Claude parse error: {e}")
    except anthropic.APIError as e:
        return (_ICP_EMPTY.copy(), {}, 0, 0, "api_error", f"Claude API: {e}")
    except Exception as e:
        return (_ICP_EMPTY.copy(), {}, 0, 0, "api_error", str(e))


def _extract_icp_fields(raw: dict) -> dict:
    return {
        "icp_international_presence":     str(raw.get("international_presence")     or "").strip(),
        "icp_languages_mentioned":        str(raw.get("languages_mentioned")        or "").strip(),
        "icp_hiring_activity":            str(raw.get("hiring_activity")            or "").strip(),
        "icp_recent_funding":             str(raw.get("recent_funding")             or "").strip(),
        "icp_recent_news":                str(raw.get("recent_news")                or "").strip(),
        "icp_global_team_signals":        str(raw.get("global_team_signals")        or "").strip(),
        "icp_training_signals":           str(raw.get("training_signals")           or "").strip(),
        "icp_multi_office":               str(raw.get("multi_office")               or "").strip(),
        "icp_language_training_fit_score": str(raw.get("language_training_fit_score") or "").strip(),
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
# Per-row enrichment  ← orchestrates both steps
# ─────────────────────────────────────────────────────────────────────────────

def enrich_one_row(
    company_name: str,
    raw_url: str,
    api_key: str,
    delay: float,
) -> tuple:
    """
    Run Step 1 (Jina + Claude extraction) then Step 2 (Claude web_search ICP).
    Returns (combined_fields_dict, debug_record_dict).
    """
    url          = raw_url.strip() if raw_url else ""
    company_name = company_name.strip() if company_name else ""

    row = {f: "" for f in ALL_ENRICHMENT_FIELDS}

    # 8-second pause between companies to stay under the token/min rate limit
    time.sleep(8)

    # ── Step 1 (three-tier: Jina → web_search → no_data) ─────────────────────
    s1_fields, s1_raw, s1_in, s1_out, s1_status, s1_err = run_step1(
        url, company_name, api_key, delay
    )

    row.update(s1_fields)
    row["step1_status"]     = s1_status
    row["step1_tokens_in"]  = str(s1_in)
    row["step1_tokens_out"] = str(s1_out)
    row["step1_cost_usd"]   = f"{calc_cost(s1_in, s1_out):.6f}"

    # ── Step 2 ────────────────────────────────────────────────────────────────
    s2_fields, s2_raw, s2_in, s2_out, s2_status, s2_err = run_step2(
        url, company_name, api_key, delay
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
        "input_company_name": company_name,
        "input_url":          raw_url,
        "normalized_url":     normalize_url(url),
        "step1_status":       s1_status,
        "step1_raw_json":     s1_raw,
        "step1_tokens_in":    s1_in,
        "step1_tokens_out":   s1_out,
        "step2_status":       s2_status,
        "step2_raw_json":     s2_raw,
        "step2_tokens_in":    s2_in,
        "step2_tokens_out":   s2_out,
        "total_cost":         calc_cost(total_in, total_out),
        "enrichment_status":  row["enrichment_status"],
        "error_message":      row["error_message"],
        "needs_manual_review": row["needs_manual_review"],
        "match_notes":        row["match_notes"],
    }
    return row, dbg


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


def make_log_df(debug_records: list) -> pd.DataFrame:
    rows = []
    for d in debug_records:
        rows.append({
            "input_company_name":  d.get("input_company_name", ""),
            "input_url":           d.get("input_url", ""),
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


def save_to_local_folder(df: pd.DataFrame, folder: str) -> tuple[str, str]:
    """
    Write Excel + CSV to *folder* with timestamped filenames.
    Returns (excel_path, csv_path). Raises OSError on permission / path errors.
    Only usable when the app runs locally — not on Streamlit Cloud.
    """
    folder_path = Path(folder.strip())
    folder_path.mkdir(parents=True, exist_ok=True)
    stamp      = ts()
    excel_path = folder_path / f"enriched_results_{stamp}.xlsx"
    csv_path   = folder_path / f"enriched_results_{stamp}.csv"
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
        autosave_last_name="",
        _jina_retry_count=0, _last_retry_msg="",
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

# =============================================================================
# SIDEBAR  — debug toggle only (no API key input)
# =============================================================================

with st.sidebar:
    # ── Enrichment mode ───────────────────────────────────────────────────────
    enrichment_mode = st.radio(
        "Enrichment mode",
        options=["Full Claude enrichment", "Extreme Light Mode (no API)"],
        index=0,
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

    if api_key:
        st.success("✓ Anthropic API key loaded")
    else:
        st.error("⚠ API key missing")

    st.divider()

    debug_mode = st.checkbox(
        "Enable debug mode",
        value=False,
        help="Shows per-row JSON responses, cache tools, and additional downloads.",
    )

    # ── Auto-save status ──────────────────────────────────────────────────────
    _last_name = ss("autosave_last_name", "")
    if ss("processing", False) and _last_name:
        st.divider()
        st.caption(f"💾 Auto-save active — last saved: **{_last_name}**")
        _last_local = ss("_last_local_save", "")
        if _last_local:
            st.caption(f"📁 Local save: {_last_local}")
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

    # ── Local save folder ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("📁 Local auto-save")
    st.caption(
        f"Saves an Excel snapshot every {LOCAL_SAVE_EVERY} companies to a folder "
        "on this machine. Only works when the app runs locally."
    )
    local_save_enabled = st.checkbox("Enable local folder save", value=False,
                                     key="local_save_enabled")
    local_save_path = st.text_input(
        "Folder path",
        value=ss("_local_save_path", ""),
        placeholder=r"e.g. C:\Users\Gert\Downloads\ or /home/user/Downloads/",
        key="local_save_path_input",
        disabled=not local_save_enabled,
    )
    if local_save_enabled and local_save_path:
        ss_set(_local_save_path=local_save_path)
    elif not local_save_enabled:
        ss_set(_local_save_path="")

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
if _api_key_error and not _elm_mode:
    blocking.append(_api_key_error)
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
    est = n_to_process * 0.002  # rough: ~$0.002/row for 2 calls
    st.info(
        f"Ready to enrich **{n_to_process:,}** rows with two enrichment steps each. "
        f"Rough estimated cost: ~${est:.2f}."
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
        _resume_mode=resume_mode, autosave_last_name="",
        _elm_mode=_elm_mode,
        _active_fields=ELM_ALL_FIELDS if _elm_mode else ALL_ENRICHMENT_FIELDS,
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
    _elm_mode_run = ss("_elm_mode", False)
    _active_fields = ss("_active_fields", ALL_ENRICHMENT_FIELDS)
    total_in      = ss("total_tokens_in", 0)
    total_out     = ss("total_tokens_out", 0)
    total_cost    = ss("total_cost_usd", 0.0)

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
        cnt_jina   = sum(1 for r in results if "enriched_jina"   in r.get("enrichment_status", ""))
        cnt_google = sum(1 for r in results if "enriched_search" in r.get("enrichment_status", ""))
        cnt_nodata = sum(1 for r in results if r.get("enrichment_status") == "no_data")
        cnt_error  = sum(1 for r in results
                         if r.get("enrichment_status") not in
                         ("enriched_jina", "enriched_search",
                          "enriched_jina_step1_only", "enriched_search_step1_only",
                          "no_data", "skipped_resume", ""))
        cnt_retries = ss("_jina_retry_count", 0)

        mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
        mc1.metric("Enriched (Jina)",   cnt_jina)
        mc2.metric("Enriched (Google)", cnt_google)
        mc3.metric("429 Retries",       cnt_retries)
        mc4.metric("No data",           cnt_nodata)
        mc5.metric("Errors",            cnt_error)
        mc6.metric("Est. cost",         f"${total_cost:.4f}")

        _retry_msg = ss("_last_retry_msg", "")
        if _retry_msg:
            st.info(_retry_msg)

    # ── Intermediate download buttons (visible whenever ≥1 row is done) ───────
    if results:
        _partial_df = build_partial_df(results, df_work, _active_fields)
        _n_done     = len(_partial_df)
        _stamp      = ts()
        with st.expander(
            f"⬇ Download intermediate results ({_n_done} rows so far)", expanded=False
        ):
            _xl_bytes  = df_to_excel_bytes(_partial_df)
            _csv_bytes = df_to_csv_bytes(_partial_df)
            _xl_kb  = len(_xl_bytes)  // 1024 or 1
            _csv_kb = len(_csv_bytes) // 1024 or 1
            _ic1, _ic2 = st.columns(2)
            _ic1.download_button(
                f"⬇ Excel ({_n_done} rows, ~{_xl_kb} KB)",
                data=_xl_bytes,
                file_name=f"enriched_results_{_stamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key=f"dl_xl_{_n_done}",
            )
            _ic2.download_button(
                f"⬇ CSV ({_n_done} rows, ~{_csv_kb} KB)",
                data=_csv_bytes,
                file_name=f"enriched_results_{_stamp}.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"dl_csv_{_n_done}",
            )

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

        with st.status(
            f"Row {idx + 1} / {_n}: **{company_name or '(empty)'}**",
            expanded=False,
        ) as status_box:
            if _elm_mode_run:
                status_box.write("⚡ Extreme Light Mode — fetching pages…")
                fields, dbg = enrich_one_row_light(company_name, raw_url)
                _fetch_status = fields.get("elm_fetch_status", "?")
                _chars        = fields.get("elm_total_chars", 0)
                status_box.write(
                    f"✅ Done — fetch: {_fetch_status} | "
                    f"{_chars:,} chars | score: {fields.get('elm_score_overall_icp', '?')}/10"
                )
                row_cost = 0.0
            else:
                status_box.write("⏳ Step 1 — Fetching page + extracting firmographics…")
                fields, dbg = enrich_one_row(company_name, raw_url, _api_key, _delay)
                s1_tok   = int(fields.get("step1_tokens_in",  0) or 0) + int(fields.get("step1_tokens_out", 0) or 0)
                s2_tok   = int(fields.get("step2_tokens_in",  0) or 0) + int(fields.get("step2_tokens_out", 0) or 0)
                row_cost = float(fields.get("total_cost_usd", 0) or 0)
                _retry_note = ""
                if "enriched_search" in fields.get("enrichment_status", ""):
                    _retry_note = " | ⚡ Google fallback used"
                elif ss("_last_retry_msg", ""):
                    _retry_note = " | ⏳ Had Jina 429 retry"
                status_box.write(
                    f"✅ Done — Step 1: {s1_tok} tokens | Step 2: {s2_tok} tokens | "
                    f"Row cost: ${row_cost:.5f}{_retry_note}"
                )

        results.append(fields)
        debug_records.append(dbg)

        # ── Auto-save (crash recovery) ────────────────────────────────────────
        try:
            autosave_append(fields, input_row)
            ss_set(autosave_last_name=company_name or raw_url or f"row {idx + 1}")
        except Exception:
            pass  # never let autosave failure abort processing

        # ── Local folder save every N companies ───────────────────────────────
        _local_path = ss("_local_save_path", "")
        _new_idx    = len(results)  # results already includes the row appended above
        if _local_path and _new_idx % LOCAL_SAVE_EVERY == 0:
            try:
                _snap_df = build_partial_df(results, df_work, _active_fields)
                _xl, _csv = save_to_local_folder(_snap_df, _local_path)
                ss_set(_last_local_save=f"{_new_idx} rows → {Path(_xl).name}")
            except Exception as _e:
                ss_set(_last_local_save=f"⚠ Save failed: {_e}")

        try:
            total_in   += int(fields.get("total_tokens_in",  0) or 0)
            total_out  += int(fields.get("total_tokens_out", 0) or 0)
            total_cost += float(fields.get("total_cost_usd", 0) or 0)
        except (ValueError, TypeError):
            pass

        ss_set(
            results=results, debug_records=debug_records, process_index=idx + 1,
            total_tokens_in=total_in, total_tokens_out=total_out, total_cost_usd=total_cost,
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

    # ── Status summary ────────────────────────────────────────────────────────
    status_counts  = (
        df_enriched["elm_fetch_status"].value_counts().to_dict()
        if _elm_done and "elm_fetch_status" in df_enriched.columns
        else df_enriched.get("enrichment_status", pd.Series(dtype=str)).value_counts().to_dict()
    )
    needs_review_n = int((df_enriched["needs_manual_review"] == "TRUE").sum()) if "needs_manual_review" in df_enriched.columns else 0
    all_sc = list(status_counts.items())
    n_cols = min(len(all_sc) + 1, 6)
    if all_sc:
        rcols = st.columns(n_cols)
        for i, (s, c) in enumerate(all_sc):
            rcols[i % n_cols].metric(_STATUS_LABELS.get(s, s), c)
        rcols[len(all_sc) % n_cols].metric("⚑ Needs review", needs_review_n)

    # ── Token usage ───────────────────────────────────────────────────────────
    t_in   = ss("total_tokens_in", 0)
    t_out  = ss("total_tokens_out", 0)
    t_cost = ss("total_cost_usd", 0.0)

    if not _elm_done:
        with st.expander("💰 Token usage & cost", expanded=True):
            tc1, tc2, tc3 = st.columns(3)
            tc1.metric("Total input tokens",  f"{t_in:,}")
            tc2.metric("Total output tokens", f"{t_out:,}")
            tc3.metric("Estimated total cost", f"${t_cost:.4f}")
            st.caption(
                f"Model: `{MODEL_ID}` · "
                f"${_COST_INPUT_PER_M}/M input · ${_COST_OUTPUT_PER_M}/M output. "
                "Two API calls per row (Step 1 + Step 2). "
                "Verify charges in your Anthropic dashboard."
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
        summary_cols = orig_cols + [
            c for c in [
                # Metadata
                "enrichment_status", "step1_status", "step2_status",
                "needs_manual_review", "match_notes",
                # Step 1 — firmographics
                "lusha_company_name", "lusha_domain", "lusha_industry", "lusha_sub_industry",
                "lusha_company_type", "lusha_employee_range", "lusha_revenue",
                "lusha_country", "lusha_city", "lusha_continent",
                "lusha_founded_year", "lusha_description",
                "lusha_linkedin_url", "lusha_specialties", "lusha_technologies",
                "lusha_total_funding_amount", "lusha_total_funding_rounds",
                "lusha_last_round_type", "lusha_last_round_amount", "lusha_last_round_date",
                "lusha_ipo_status",
                # Step 2 — ICP signals
                "icp_language_training_fit_score",
                "icp_international_presence", "icp_languages_mentioned",
                "icp_hiring_activity", "icp_recent_funding", "icp_recent_news",
                "icp_global_team_signals", "icp_training_signals", "icp_multi_office",
                # Cost
                "total_tokens_in", "total_tokens_out", "total_cost_usd",
                "error_message",
            ]
            if c in df_enriched.columns
        ]
        st.dataframe(df_enriched[summary_cols], use_container_width=True, height=400)
        tab1, tab2 = st.tabs(["Step 1 — All firmographic columns", "Step 2 — All ICP columns"])
        with tab1:
            st.dataframe(df_enriched[[c for c in STEP1_FIELDS if c in df_enriched.columns]],
                         use_container_width=True)
        with tab2:
            st.dataframe(df_enriched[[c for c in ICP_FIELDS if c in df_enriched.columns]],
                         use_container_width=True)

    # ── Downloads ─────────────────────────────────────────────────────────────
    st.subheader("Download results")
    log_df = make_log_df(debug_records_done)

    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button(
            "⬇ Enriched Excel (.xlsx)",
            data=df_to_excel_bytes(df_enriched),
            file_name="claude_enriched.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            help="All original columns + Step 1 firmographics + Step 2 ICP signals + metadata.",
        )
    with dl2:
        st.download_button(
            "⬇ Enriched CSV",
            data=df_to_csv_bytes(df_enriched),
            file_name="claude_enriched.csv",
            mime="text/csv",
            use_container_width=True,
            help="Same data as Excel, in CSV format.",
        )
    with dl3:
        st.download_button(
            "⬇ Processing log CSV",
            data=df_to_csv_bytes(log_df),
            file_name="claude_processing_log.csv",
            mime="text/csv",
            use_container_width=True,
            help="One row per company: step statuses, token counts, costs, review flags.",
        )

    # ── Debug section ─────────────────────────────────────────────────────────
    if debug_mode and debug_records_done:
        st.divider()
        st.subheader("🐛 Per-row debug details")

        debug_df = pd.DataFrame([
            {
                "row":               i + 1,
                "company":           d.get("input_company_name", ""),
                "url":               d.get("input_url", ""),
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
            col_a, col_b = st.columns(2)
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
            st.download_button(
                "⬇ Debug Excel (+ JSON preview columns)",
                data=df_to_excel_bytes(debug_enriched),
                file_name="claude_enriched_debug.xlsx",
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
