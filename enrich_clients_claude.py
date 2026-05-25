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

try:
    from commercial_fit_scoring import score_dataframe as _score_dataframe, SCORE_OUTPUT_COLS as _SCORE_OUTPUT_COLS
    _SCORING_AVAILABLE = True
except ImportError:
    _score_dataframe = None  # type: ignore[assignment]
    _SCORE_OUTPUT_COLS = []
    _SCORING_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

JINA_READER_URL  = "https://r.jina.ai/"
JINA_SEARCH_URL  = "https://s.jina.ai/"
CACHE_DIR        = Path("claude_json_cache")
DEBUG_LOG_DIR    = Path("debug_logs")
SEARCH_OUTPUT_DIR = DEBUG_LOG_DIR / "search_outputs"
AUTOSAVE_PATH         = "/tmp/enrichment_autosave.csv"
CHECKPOINT_EVERY      = 50   # write checkpoint_NNN.xlsx every N companies
_AUTO_DL_EVERY        = 100  # auto browser-download every N companies
_DEFAULT_DOWNLOAD_DIR = os.path.expanduser("~/Downloads")
_PER_COMPANY_AUTOSAVE_DEFAULT_DIR = os.path.expanduser(
    "~/Downloads/company_enrichment_runs"
)
MODEL_STEP1      = "claude-haiku-4-5-20251001"
MODEL_STEP2      = "claude-haiku-4-5-20251001"
MODEL_ID         = MODEL_STEP1   # legacy alias used in a few places
WEB_SEARCH_TOOL  = {"type": "web_search_20250305", "name": "web_search"}

# Set to True (or via SHOW_ADVANCED_SETTINGS env var / Streamlit secret) to show
# the full sidebar, column picker, preview, debug sections, and technical logs.
# Normal users should always see False (the minimal flow).
SHOW_ADVANCED_SETTINGS: bool = False

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

3. Competitor signal — check all three categories below and populate the exact matching \
output field. Each provider belongs to EXACTLY ONE category; do not move it to another.

  Category 1 — Direct corporate language training competitors (STRONG signal). \
These providers must ONLY go into competitor_signal and direct_language_competitor_signal. \
Do NOT put them into online_language_learning_signal or broader_lnd_platform_signal. \
Providers: goFLUENT, Learnlight, Speexx, Voxy, Learnship, Berlitz, \
EF Corporate Solutions, Babbel for Business, Rosetta Stone Enterprise, Preply Business, \
Talaera, Busuu for Business, Lingoda for Business, Fluentify, Twenix, Cambly.

  Category 2 — Online language learning brands (MEDIUM signal). \
These must ONLY go into online_language_learning_signal, and only when found in a \
corporate, HR, L&D, employee benefit, or company-wide training context. \
Do NOT put them into competitor_signal, direct_language_competitor_signal, \
or broader_lnd_platform_signal. \
Providers: Duolingo, Babbel, Busuu, Rosetta Stone, Preply, Memrise, Mondly, ELSA Speak, \
FluentU, italki, Lingoda, Open English, Mango Languages, Pimsleur, Drops, HelloTalk, \
Tandem.

  Category 3 — Broader corporate learning / L&D platforms (L&D maturity signal). \
These must ONLY go into broader_lnd_platform_signal. \
Do NOT put them into competitor_signal, direct_language_competitor_signal, \
or online_language_learning_signal. \
Providers: OpenSesame, Coursera for Business, Udemy Business, LinkedIn Learning, \
Skillsoft, Docebo, Degreed, Cornerstone, 360Learning, Moodle Workplace, Absorb LMS, \
TalentLMS, LearnUpon, Pluralsight.

  mYngle — mYngle is the company running this analysis and is NOT a competitor. \
Do NOT place mYngle in any competitor or provider signal field under any circumstances. \
If mYngle is mentioned in the search results, note it only in the evidence field as a \
reference or client signal.

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
"competitor_signal": "comma-separated Category 1 provider names found, otherwise empty string — Category 3 platforms such as LinkedIn Learning or Skillsoft must never appear here", \
"direct_language_competitor_signal": "comma-separated Category 1 provider names found, otherwise empty string", \
"online_language_learning_signal": "comma-separated Category 2 provider names found in corporate/HR/L&D context only, otherwise empty string — Category 3 platforms must never appear here", \
"broader_lnd_platform_signal": "comma-separated Category 3 provider names found, otherwise empty string", \
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


# ── Model-signal extraction prompt ────────────────────────────────────────────
#
# This prompt is sent AFTER Step 1 + Step 2 enrichment to extract structured
# numeric signals for logistic-regression readiness.  It uses only evidence
# already gathered — no additional web searches are performed.

MODEL_SIGNAL_PROMPT_TEMPLATE = """\
You are extracting structured model signals for a company from available enrichment context.
Your ONLY task is to return a valid JSON object with exactly the fields listed below.

Rules:
- Use ONLY evidence from the provided enrichment context (website, search snippets, firmographic data).
- Do NOT invent facts, infer company size, or estimate employee count.
- Do NOT generate or fill any employee_size_score field.
- If evidence is weak or ambiguous, score 1 or 0.
- Use score 3 ONLY for explicit or very strong evidence.
- Keep each evidence field to one short sentence. Use empty string if score is 0.
- Return ONLY raw JSON — no markdown, no backticks, no explanation.
- All score fields must be integers.
- All binary fields must be 0 or 1.

Scoring rubric (0–3):
  0 = no evidence found
  1 = weak or indirect evidence
  2 = clear evidence
  3 = strong or explicit evidence

Provider category rules:
  Category 1 direct language-training competitors (goes into has_language_competitor):
    goFLUENT, Learnlight, Speexx, Voxy, Learnship, Berlitz, EF Corporate Solutions,
    Babbel for Business, Rosetta Stone Enterprise, Preply Business, Talaera,
    Busuu for Business, Lingoda for Business, Fluentify, Twenix, Cambly.
  Category 2 online language-learning brands (goes into has_online_learning_signal ONLY
    when found in a corporate/HR/L&D/employee-benefit/company-wide training context):
    Duolingo, Babbel, Busuu, Rosetta Stone, Preply, Memrise, Mondly, ELSA Speak,
    FluentU, italki, Lingoda, Open English, Mango Languages, Pimsleur, Drops,
    HelloTalk, Tandem.
  Category 3 broader L&D platforms (goes into has_lnd_platform_signal):
    OpenSesame, Coursera for Business, Udemy Business, LinkedIn Learning, Skillsoft,
    Docebo, Degreed, Cornerstone, 360Learning, Moodle Workplace, Absorb LMS,
    TalentLMS, LearnUpon, Pluralsight.
  has_competitor_signal = 1 if ANY provider from Cat 1, Cat 2 (in corporate context), or Cat 3 is found.
  mYngle is NOT a competitor and must never appear in any signal field.

Binary field rules:
  is_public = 1 if the company appears publicly listed or has clear public company evidence.
  has_funding = 1 if funding rounds, venture backing, private equity, acquisition funding, or similar evidence is found.

Company: __COMPANY_NAME__
Domain: __DOMAIN__

Enrichment context:
__ENRICHMENT_CONTEXT__

Return a JSON object with EXACTLY these fields and no others:
{
  "sig_intl_footprint_score": <int 0-3>,
  "sig_intl_footprint_evidence": <str>,
  "sig_foreign_hq_score": <int 0-3>,
  "sig_foreign_hq_evidence": <str>,
  "sig_explicit_lnd_score": <int 0-3>,
  "sig_explicit_lnd_evidence": <str>,
  "sig_multicultural_score": <int 0-3>,
  "sig_multicultural_evidence": <str>,
  "sig_employer_branding_score": <int 0-3>,
  "sig_employer_branding_evidence": <str>,
  "sig_rapid_growth_score": <int 0-3>,
  "sig_rapid_growth_evidence": <str>,
  "sig_merger_acq_score": <int 0-3>,
  "sig_merger_acq_evidence": <str>,
  "sig_lnd_onboarding_score": <int 0-3>,
  "sig_lnd_onboarding_evidence": <str>,
  "ti_language_english_score": <int 0-3>,
  "ti_language_english_evidence": <str>,
  "ti_onboarding_score": <int 0-3>,
  "ti_onboarding_evidence": <str>,
  "ti_leadership_score": <int 0-3>,
  "ti_leadership_evidence": <str>,
  "ti_broader_professional_score": <int 0-3>,
  "ti_broader_professional_evidence": <str>,
  "ti_team_collab_score": <int 0-3>,
  "ti_team_collab_evidence": <str>,
  "ti_intercultural_score": <int 0-3>,
  "ti_intercultural_evidence": <str>,
  "ti_negotiation_sales_score": <int 0-3>,
  "ti_negotiation_sales_evidence": <str>,
  "has_competitor_signal": <0 or 1>,
  "has_competitor_signal_evidence": <str>,
  "has_language_competitor": <0 or 1>,
  "has_language_competitor_evidence": <str>,
  "has_online_learning_signal": <0 or 1>,
  "has_online_learning_signal_evidence": <str>,
  "has_lnd_platform_signal": <0 or 1>,
  "has_lnd_platform_signal_evidence": <str>,
  "is_public": <0 or 1>,
  "is_public_evidence": <str>,
  "has_funding": <0 or 1>,
  "has_funding_evidence": <str>,
  "competitor_signal_strength_score": <int 0-3>,
  "competitor_signal_strength_evidence": <str>,
  "language_competitor_strength_score": <int 0-3>,
  "language_competitor_strength_evidence": <str>,
  "online_learning_signal_strength_score": <int 0-3>,
  "online_learning_signal_strength_evidence": <str>,
  "lnd_platform_signal_strength_score": <int 0-3>,
  "lnd_platform_signal_strength_evidence": <str>,
  "model_signal_overall_confidence_score": <int 0-3>,
  "model_signal_needs_manual_review": <0 or 1>,
  "model_signal_manual_review_reason": <str>,
  "model_signal_sources_used": <str>,
  "model_signal_search_quality": <"good" or "partial" or "weak" or "failed">
}
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
    "icp_direct_language_competitor_signal",
    "icp_online_language_learning_signal",
    "icp_broader_lnd_platform_signal",
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
    "step2_provider_used",
    "lucia_data_status",
    "lucia_api_called",
    "step1_run_status",
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
    "lusha_api_raw_keys",
]

# Model-signal fields — added by the structured signal-extraction layer
# Ordinal score fields (integer 0–3)
MODEL_SIGNAL_SCORE_FIELDS = [
    "sig_intl_footprint_score",
    "sig_foreign_hq_score",
    "sig_explicit_lnd_score",
    "sig_multicultural_score",
    "sig_employer_branding_score",
    "sig_rapid_growth_score",
    "sig_merger_acq_score",
    "sig_lnd_onboarding_score",
    "ti_language_english_score",
    "ti_onboarding_score",
    "ti_leadership_score",
    "ti_broader_professional_score",
    "ti_team_collab_score",
    "ti_intercultural_score",
    "ti_negotiation_sales_score",
    "competitor_signal_strength_score",
    "language_competitor_strength_score",
    "online_learning_signal_strength_score",
    "lnd_platform_signal_strength_score",
    "model_signal_overall_confidence_score",
]

# Binary fields (0 or 1)
MODEL_SIGNAL_BINARY_FIELDS = [
    "has_competitor_signal",
    "has_language_competitor",
    "has_online_learning_signal",
    "has_lnd_platform_signal",
    "is_public",
    "has_funding",
    "model_signal_needs_manual_review",
]

# Evidence columns (one per score/binary field, using same base name + _evidence)
_MODEL_SIGNAL_SCORED_BASES = [
    "sig_intl_footprint",
    "sig_foreign_hq",
    "sig_explicit_lnd",
    "sig_multicultural",
    "sig_employer_branding",
    "sig_rapid_growth",
    "sig_merger_acq",
    "sig_lnd_onboarding",
    "ti_language_english",
    "ti_onboarding",
    "ti_leadership",
    "ti_broader_professional",
    "ti_team_collab",
    "ti_intercultural",
    "ti_negotiation_sales",
    "competitor_signal_strength",
    "language_competitor_strength",
    "online_learning_signal_strength",
    "lnd_platform_signal_strength",
    "has_competitor_signal",
    "has_language_competitor",
    "has_online_learning_signal",
    "has_lnd_platform_signal",
    "is_public",
    "has_funding",
]
MODEL_SIGNAL_EVIDENCE_FIELDS = [f"{b}_evidence" for b in _MODEL_SIGNAL_SCORED_BASES]

# QA / metadata fields
MODEL_SIGNAL_QA_FIELDS = [
    "model_signal_manual_review_reason",
    "model_signal_sources_used",
    "model_signal_search_quality",
]

MODEL_SIGNAL_FIELDS = (
    MODEL_SIGNAL_SCORE_FIELDS
    + MODEL_SIGNAL_BINARY_FIELDS
    + MODEL_SIGNAL_EVIDENCE_FIELDS
    + MODEL_SIGNAL_QA_FIELDS
)

ALL_ENRICHMENT_FIELDS = (
    LUSHA_API_FIELDS + LUSHA_API_META_FIELDS + STEP1_FIELDS + ICP_FIELDS
    + META_FIELDS + MODEL_SIGNAL_FIELDS
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
_DOMAIN_HINTS  = ["domain", "website", "url", "web", "site", "domein",
                   "lusha_domain", "company_domain", "company_url"]

# Prefixes and field names used to detect existing Lusha/Lucia enrichment columns
_LUSHA_COL_PREFIXES = ("lusha_", "lusha_api_", "lucia_")
_LUSHA_COL_NAMES_EXACT = frozenset([
    "company_size", "employee_range", "employee_size_score",
    "revenue", "revenue_range", "industry", "sub_industry",
    "founded_year", "country", "city", "linkedin_url",
])

# Lucia/Lusha contact export column signatures (company-level fields)
_LUCIA_EXPORT_COMPANY_COLS = frozenset([
    "Company Name", "Company Domain", "Company Description",
    "Company Year Founded", "Company Website",
    "Company Number of Employees", "Company Revenue",
    "Company linkedin URL", "Total Funding Amount",
    "Total Number of Rounds", "Last Round/Event Amount",
    "Last Round/Event Type", "Last Round/Event Date",
    "IPO Status", "Company Main Industry", "Company Sub Industry",
    "Company Technologies", "Company Specialties",
    "Company Continent", "Company Country", "Company State",
    "Company City", "Company Country ISO",
])

# Mapping: Lucia/Lusha CSV column name → internal lusha_api_* field name
_LUCIA_COL_MAP = {
    "Company Name":                "lusha_api_company_name",
    "Company Domain":              "lusha_api_domain",
    "Company Website":             "lusha_api_domain",   # fallback if no Company Domain
    "Company Description":         "lusha_api_description",
    "Company Year Founded":        "lusha_api_founded_year",
    "Company Number of Employees": "lusha_api_employee_range",
    "Company Revenue":             "lusha_api_revenue_range",
    "Company Main Industry":       "lusha_api_industry",
    "Company Sub Industry":        "lusha_api_sub_industry",
    "Company Country":             "lusha_api_country",
    "Company City":                "lusha_api_city",
    "Company Continent":           "lusha_api_continent",
    "Company linkedin URL":        "lusha_api_linkedin_url",
    "Company Specialties":         "lusha_api_specialties",
    "Company Technologies":        "lusha_api_technologies",
    "Total Funding Amount":        "lusha_api_total_funding_amount",
    "Total Number of Rounds":      "lusha_api_total_funding_rounds",
    "Last Round/Event Type":       "lusha_api_last_round_type",
    "Last Round/Event Amount":     "lusha_api_last_round_amount",
    "Last Round/Event Date":       "lusha_api_last_round_date",
    "IPO Status":                  "lusha_api_ipo_status",
}

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


def is_lucia_contact_export(df: pd.DataFrame) -> bool:
    """Return True if df is a pre-enriched Lucia/Lusha contact export (Type 2).

    Requires: 'Company Name' AND ('Company Domain' OR 'Company Website')
    plus at least 2 additional Lucia-specific company-level columns.
    """
    cols = set(df.columns.tolist())
    if "Company Name" not in cols:
        return False
    if not (cols & {"Company Domain", "Company Website"}):
        return False
    extra = _LUCIA_EXPORT_COMPANY_COLS - {"Company Name", "Company Domain", "Company Website"}
    return len(cols & extra) >= 2


def get_lucia_name_col(df: pd.DataFrame) -> str | None:
    """Return 'Company Name' if present, else None."""
    return "Company Name" if "Company Name" in df.columns else None


def get_lucia_domain_col(df: pd.DataFrame) -> str | None:
    """Return 'Company Domain' or 'Company Website' for Lucia exports, else None."""
    if "Company Domain" in df.columns:
        return "Company Domain"
    if "Company Website" in df.columns:
        return "Company Website"
    return None


def map_lucia_export_row(row_dict: dict) -> dict:
    """Map a Lucia/Lusha CSV export row to internal lusha_api_* field names.

    'Company Domain' takes priority over 'Company Website' for lusha_api_domain.
    Always adds status fields marking the data as reused (no API call needed).
    """
    mapped: dict = {}
    domain_set = False
    for src_col, tgt_field in _LUCIA_COL_MAP.items():
        val = row_dict.get(src_col)
        if val is None or str(val).strip().lower() in ("", "nan", "none"):
            continue
        val_str = str(val).strip()
        if tgt_field == "lusha_api_domain":
            if src_col == "Company Domain":
                mapped[tgt_field] = val_str
                domain_set = True
            elif src_col == "Company Website" and not domain_set:
                mapped[tgt_field] = val_str
        else:
            mapped[tgt_field] = val_str
    mapped["lusha_api_status"]    = "reused_existing_lucia_data"
    mapped["lucia_data_status"]   = "existing_lucia_export_reused"
    mapped["lucia_api_called"]    = "False"
    mapped["lusha_api_match_notes"] = (
        "Existing Lucia/Lusha export data reused; API call skipped."
    )
    return mapped


def deduplicate_lucia_export(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate a Lucia contact export to one row per unique company.

    Primary dedup key: cleaned Company Domain.
    Fallback: cleaned Company Name.
    Adds 'source_contact_count' column.
    """
    df = df.copy()
    domain_col = (
        "Company Domain" if "Company Domain" in df.columns else
        "Company Website" if "Company Website" in df.columns else None
    )
    name_col = "Company Name" if "Company Name" in df.columns else None

    if domain_col:
        df["_dedup_key"] = df[domain_col].apply(
            lambda x: clean_domain(str(x)) if pd.notna(x) else ""
        )
        if name_col:
            df["_dedup_key"] = df.apply(
                lambda r: r["_dedup_key"] if r["_dedup_key"]
                else str(r.get(name_col, "")).strip().lower(),
                axis=1,
            )
    elif name_col:
        df["_dedup_key"] = df[name_col].apply(lambda x: str(x).strip().lower())
    else:
        df["_dedup_key"] = df.index.astype(str)

    counts = df.groupby("_dedup_key").size().rename("source_contact_count")
    df_dedup = df.drop_duplicates(subset=["_dedup_key"], keep="first").copy()
    df_dedup = df_dedup.merge(counts, on="_dedup_key", how="left")
    df_dedup = df_dedup.drop(columns=["_dedup_key"])
    return df_dedup.reset_index(drop=True)


def detect_columns(df: pd.DataFrame) -> tuple:
    """Detect company name and domain/URL columns in df.

    For Lucia/Lusha contact exports the exact 'Company Name' and
    'Company Domain'/'Company Website' columns are returned directly —
    fuzzy matching is bypassed so person fields like 'First Name' or
    'LinkedIn URL' can never be chosen.
    """
    # Exact match for Lucia/Lusha contact exports
    if is_lucia_contact_export(df):
        return get_lucia_name_col(df), get_lucia_domain_col(df)

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


def detect_lusha_columns(df: pd.DataFrame) -> list:
    """Return list of column names in df that look like existing Lusha/Lucia enrichment fields."""
    # Lucia/Lusha contact export: return all recognised company-level columns
    if is_lucia_contact_export(df):
        return [c for c in df.columns if c in _LUCIA_EXPORT_COMPANY_COLS]

    found = []
    for col in df.columns:
        col_l = col.lower().strip()
        if any(col_l.startswith(p) for p in _LUSHA_COL_PREFIXES):
            found.append(col)
        elif col_l in _LUSHA_COL_NAMES_EXACT:
            found.append(col)
    return found


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
    "Find company information about __COMPANY_NAME__ (__URL__). "
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
        prompt = (
            _STEP1_FALLBACK_PROMPT_TMPL
            .replace("__COMPANY_NAME__", company_name or target)
            .replace("__URL__", target)
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
    """Return 5 feature-driven queries for ICP signal extraction.

    Each query targets a distinct buying-signal dimension so Claude receives
    evidence that is grouped by intent rather than by brand co-mention.
    """
    name   = company_name or clean_domain(target) or target
    domain = clean_domain(target) if target else ""
    site_q = f'site:{domain} OR ' if domain else ""
    return [
        # Q1 — General company context (anchor to official site when possible)
        f'{site_q}"{name}" about company overview headquarters',
        # Q2 — International footprint / HQ structure
        f'"{name}" headquarters OR offices OR countries OR "international operations" OR "global presence" OR "regional HQ"',
        # Q3 — L&D / employee training
        f'"{name}" "learning and development" OR training OR academy OR onboarding OR "talent development" OR L&D',
        # Q4 — Language / global-team signals
        f'"{name}" English OR "language training" OR "global teams" OR multilingual OR "language program" OR intercultural',
        # Q5 — Competitor / online learning tool co-mention
        (
            f'"{name}" Preply OR Learnlight OR Speexx OR goFLUENT OR Learnship OR Voxy'
            f' OR "online learning" OR LMS OR Berlitz OR Talaera OR Busuu OR Duolingo'
        ),
    ]

# Human-readable label for each query position (1-to-1 with _build_serper_queries)
_SERPER_QUERY_LABELS = [
    "General company context",
    "International footprint / HQ",
    "L&D / employee training",
    "Language / global teams",
    "Competitor / online learning signal",
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
            json={"q": query, "gl": "us", "hl": "en", "num": 5},
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


_SERPER_CAREER_SIGNALS = frozenset([
    "career", "/jobs", "glassdoor", "indeed", ".jobs", "workable",
    "greenhouse.io", "lever.co", "smartrecruiters", "bamboohr",
    "jobvite", "icims", "taleo", "recruiting",
])
_SERPER_NEWS_SIGNALS = frozenset([
    "reuters", "bloomberg", "wsj.com", "ft.com", "forbes",
    "techcrunch", "businesswire", "prnewswire", "globenewswire",
    "marketwatch", "apnews.com", "cnbc", "wired", "theguardian",
    "economist", "fortune", "venturebeat",
])
_SERPER_THIRD_PARTY_SIGNALS = frozenset([
    "linkedin.com", "crunchbase", "zoominfo", "pitchbook",
    "dnb.com", "hoovers", "owler", "manta", "trustpilot",
    "clutch.co", "g2.com", "capterra", "wellfound", "angellist",
])


def _classify_serper_source(link: str, title: str) -> str:
    """Return a short source-type label for a Serper result URL."""
    u = link.lower()
    if any(s in u for s in _SERPER_CAREER_SIGNALS):
        return "careers"
    if any(s in u for s in _SERPER_NEWS_SIGNALS):
        return "news"
    if any(s in u for s in _SERPER_THIRD_PARTY_SIGNALS):
        return "third_party"
    return "company_site"


def _format_serper_results(query_groups: list) -> str:
    """Format per-query Serper results as compact annotated text for Claude.

    Args:
        query_groups: list of (query_label, hits) tuples where hits is a list
                      of result dicts returned by _call_serper.
    """
    if not query_groups or not any(hits for _, hits in query_groups):
        return "(No web search results were found.)"
    blocks: list[str] = []
    for label, hits in query_groups:
        if not hits:
            continue
        lines = [f"=== {label} ==="]
        for i, r in enumerate(hits, 1):
            title   = (r.get("title", "") or "(no title)").strip()
            link    = r.get("link", "")
            snippet = (r.get("snippet", "") or "").strip()[:180]
            stype   = _classify_serper_source(link, title)
            lines.append(f"[{i}] {title}")
            lines.append(f"    URL: {link}")
            lines.append(f"    source_type: {stype}")
            if snippet:
                lines.append(f"    snippet: {snippet}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "(No web search results were found.)"


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
        _dry_placeholder = "\n".join(
            f"=== {lbl} ===\n[DRY RUN: results would appear here]"
            for lbl in _SERPER_QUERY_LABELS
        )
        _dry_instruction = (
            f"Now analyze this company based on the web search results provided below.\n\n"
            f"Company: {target}\n\n"
            f"Web search results (retrieved via Serper Google Search, grouped by signal type):\n"
            f"{_dry_placeholder}\n\n"
            "Evidence quality rules:\n"
            "- Only mark a buying signal as present when a result contains company-specific, "
            "contextual evidence — not just a keyword in a URL or a generic snippet.\n"
            "- A competitor signal requires the provider name to appear in a meaningful context.\n"
            "- Set lead_score to High only when two or more clearly distinct strong signals appear.\n"
            "- Base your analysis ONLY on the search results above. Do not invent evidence."
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

    # ── Collect results per-query, preserving query-label grouping ───────────
    query_groups: list = []   # [(label, hits), ...]
    labels = _SERPER_QUERY_LABELS
    for qi, (q, label) in enumerate(zip(queries, labels), 1):
        _dlog(f"Serper query [{label}]: {q}")
        hits, http_status, raw_json, err_str = _call_serper(q, serper_key)
        if err_str:
            _dlog(f"Serper warning [{label}] — {err_str}")
            hits = []
        else:
            _dlog(f"Serper returned {len(hits)} results for [{label}]")
        query_groups.append((label, hits))
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

    total_hits = sum(len(h) for _, h in query_groups)
    _dlog(f"Total Serper results for {company_name}: {total_hits} across {len(query_groups)} queries")

    # ── Build Claude prompt ────────────────────────────────────────────────────
    results_text = _format_serper_results(query_groups)
    search_instruction = (
        f"Now analyze this company based on the web search results provided below.\n\n"
        f"Company: {target}\n\n"
        f"Web search results (retrieved via Serper Google Search, grouped by signal type):\n"
        f"{results_text}\n\n"
        "Evidence quality rules:\n"
        "- Only mark a buying signal as present when a result contains company-specific, "
        "contextual evidence — not just a keyword in a URL or a generic snippet.\n"
        "- A competitor signal requires the provider name to appear in a meaningful context "
        "(HR case study, employee benefit page, vendor review) not just a search result title.\n"
        "- Set lead_score to High only when two or more clearly distinct strong signals appear.\n"
        "- Base your analysis ONLY on the search results above. Do not invent or infer evidence."
    )
    full_prompt = STEP2_STATIC_PREFIX + f"\n\n{search_instruction}"

    _dlog(f"Selected model: {model_step2}")

    if _debug_callback:
        _debug_callback(
            "prompt",
            company=company_name,
            model=model_step2,
            provider=STEP2_PROVIDER_SERPER,
            search_prompt=f"Queries: {queries}\n\nTotal results: {total_hits}",
            full_prompt=full_prompt,
            notes=[
                f"Serper queries used: {queries}",
                f"Total results retrieved: {total_hits} across {len(query_groups)} queries",
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

    ck = f"step2_claude_{target}"
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


# ── Provider category membership — used by the post-processing sanitizer ─────
# Each provider is listed in exactly one set; membership is checked case-insensitively.

_CAT1_PROVIDERS: frozenset = frozenset({
    "gofluent", "learnlight", "speexx", "voxy", "learnship", "berlitz",
    "ef corporate solutions", "babbel for business", "rosetta stone enterprise",
    "preply business", "talaera", "busuu for business", "lingoda for business",
    "fluentify", "twenix", "cambly",
})

_CAT2_PROVIDERS: frozenset = frozenset({
    "duolingo", "babbel", "busuu", "rosetta stone", "preply", "memrise",
    "mondly", "elsa speak", "fluentu", "italki", "lingoda", "open english",
    "mango languages", "pimsleur", "drops", "hellotalk", "tandem",
})

_CAT3_PROVIDERS: frozenset = frozenset({
    "opensesame", "coursera for business", "udemy business", "linkedin learning",
    "skillsoft", "docebo", "degreed", "cornerstone", "360learning",
    "moodle workplace", "absorb lms", "talentlms", "learnupon", "pluralsight",
})

# mYngle must never appear in any competitor/provider signal field.
_MYNGLE_VARIANTS: frozenset = frozenset({"myngle", "mYngle"})


def _sanitize_provider_list(raw_value: str, allowed: frozenset) -> str:
    """Return only items from raw_value (comma-separated) whose lowercase name
    matches a member of allowed, stripping mYngle and cross-category stragglers."""
    if not raw_value or not raw_value.strip():
        return ""
    kept = []
    for item in raw_value.split(","):
        name = item.strip()
        if not name:
            continue
        name_lc = name.lower()
        if name_lc in {v.lower() for v in _MYNGLE_VARIANTS}:
            continue  # never a competitor/provider
        if name_lc in allowed:
            kept.append(name)
    return ", ".join(kept)


def _sanitize_icp_provider_fields(fields: dict) -> dict:
    """Deterministic post-processing: enforce category membership rules and
    remove mYngle from all competitor/provider signal fields.

    icp_competitor_signal and icp_direct_language_competitor_signal both
    represent Category 1 providers; after individual sanitization their union
    is written back to both fields so neither is accidentally left empty when
    the other has a valid value.
    """
    cs = _sanitize_provider_list(fields.get("icp_competitor_signal", ""), _CAT1_PROVIDERS)
    dl = _sanitize_provider_list(fields.get("icp_direct_language_competitor_signal", ""), _CAT1_PROVIDERS)

    # Build deduplicated union preserving first-seen order
    seen: dict = {}
    for name in [n.strip() for n in (cs + ("," if cs and dl else "") + dl).split(",") if n.strip()]:
        seen.setdefault(name.lower(), name)
    merged = ", ".join(seen.values())

    fields["icp_competitor_signal"] = merged
    fields["icp_direct_language_competitor_signal"] = merged
    fields["icp_online_language_learning_signal"] = _sanitize_provider_list(
        fields.get("icp_online_language_learning_signal", ""), _CAT2_PROVIDERS,
    )
    fields["icp_broader_lnd_platform_signal"] = _sanitize_provider_list(
        fields.get("icp_broader_lnd_platform_signal", ""), _CAT3_PROVIDERS,
    )
    return fields


def _extract_icp_fields(raw: dict) -> dict:
    fields = {
        "icp_lead_score":                          str(raw.get("lead_score")                          or "").strip(),
        "icp_buying_signals":                      str(raw.get("buying_signals")                      or "").strip(),
        "icp_competitor_signal":                   str(raw.get("competitor_signal")                   or "").strip(),
        "icp_direct_language_competitor_signal":   str(raw.get("direct_language_competitor_signal")   or "").strip(),
        "icp_online_language_learning_signal":     str(raw.get("online_language_learning_signal")     or "").strip(),
        "icp_broader_lnd_platform_signal":         str(raw.get("broader_lnd_platform_signal")         or "").strip(),
        "icp_evidence":                            str(raw.get("evidence")                            or "").strip(),
        "icp_likely_training_interest":            str(raw.get("likely_training_interest")            or "").strip(),
        "icp_why_relevant":                        str(raw.get("why_relevant")                        or "").strip(),
        "icp_potential_buyer_function":            str(raw.get("potential_buyer_function")            or "").strip(),
    }
    return _sanitize_icp_provider_fields(fields)


# ─────────────────────────────────────────────────────────────────────────────
# Model-signal extraction — Step 3
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_SIGNAL_EMPTY: dict = {}  # populated after field list is known at import time

def _build_model_signal_empty() -> dict:
    empty: dict = {}
    for f in MODEL_SIGNAL_SCORE_FIELDS:
        empty[f] = 0
    for f in MODEL_SIGNAL_BINARY_FIELDS:
        empty[f] = 0
    for f in MODEL_SIGNAL_EVIDENCE_FIELDS:
        empty[f] = ""
    for f in MODEL_SIGNAL_QA_FIELDS:
        empty[f] = ""
    return empty


def _build_enrichment_context(row: dict) -> str:
    """Format the existing Step 1 + Step 2 enrichment data into a concise text context."""
    lines: list[str] = []

    # Step 1 firmographics
    s1_parts = []
    for key, label in [
        ("lusha_description",    "Description"),
        ("lusha_industry",       "Industry"),
        ("lusha_sub_industry",   "Sub-industry"),
        ("lusha_company_type",   "Company type"),
        ("lusha_country",        "Country"),
        ("lusha_city",           "City"),
        ("lusha_continent",      "Continent"),
        ("lusha_founded_year",   "Founded"),
        ("lusha_employee_range", "Employee range"),
        ("lusha_revenue",        "Revenue"),
        ("lusha_specialties",    "Specialties"),
        ("lusha_technologies",   "Technologies"),
        ("lusha_total_funding_amount", "Total funding"),
        ("lusha_total_funding_rounds", "Funding rounds"),
        ("lusha_last_round_type",  "Last round type"),
        ("lusha_last_round_amount","Last round amount"),
        ("lusha_ipo_status",       "IPO status"),
    ]:
        v = str(row.get(key, "") or "").strip()
        if v:
            s1_parts.append(f"{label}: {v}")
    if s1_parts:
        lines.append("=== Step 1 firmographic data ===")
        lines.extend(s1_parts)

    # Step 2 ICP signals
    s2_parts = []
    for key, label in [
        ("icp_lead_score",                        "Lead score"),
        ("icp_buying_signals",                    "Buying signals"),
        ("icp_competitor_signal",                 "Competitor signal (Cat 1)"),
        ("icp_direct_language_competitor_signal", "Direct language competitor"),
        ("icp_online_language_learning_signal",   "Online language learning signal (Cat 2)"),
        ("icp_broader_lnd_platform_signal",       "Broader L&D platform signal (Cat 3)"),
        ("icp_evidence",                          "ICP evidence"),
        ("icp_likely_training_interest",          "Likely training interest"),
        ("icp_why_relevant",                      "Why relevant"),
        ("icp_potential_buyer_function",          "Potential buyer function"),
    ]:
        v = str(row.get(key, "") or "").strip()
        if v:
            s2_parts.append(f"{label}: {v}")
    if s2_parts:
        lines.append("=== Step 2 ICP signal data ===")
        lines.extend(s2_parts)

    return "\n".join(lines) if lines else "(No enrichment context available)"


def _coerce_model_signals(raw: dict) -> dict:
    """Validate and coerce parsed model-signal JSON into expected types."""
    out: dict = _build_model_signal_empty()

    for f in MODEL_SIGNAL_SCORE_FIELDS:
        try:
            v = int(raw.get(f, 0) or 0)
            out[f] = max(0, min(3, v))
        except (TypeError, ValueError):
            out[f] = 0

    for f in MODEL_SIGNAL_BINARY_FIELDS:
        try:
            v = int(raw.get(f, 0) or 0)
            out[f] = 1 if v else 0
        except (TypeError, ValueError):
            out[f] = 0

    for f in MODEL_SIGNAL_EVIDENCE_FIELDS:
        out[f] = str(raw.get(f, "") or "").strip()

    # QA text fields
    out["model_signal_manual_review_reason"] = str(
        raw.get("model_signal_manual_review_reason", "") or ""
    ).strip()
    out["model_signal_sources_used"] = str(
        raw.get("model_signal_sources_used", "") or ""
    ).strip()
    sq = str(raw.get("model_signal_search_quality", "") or "").strip().lower()
    out["model_signal_search_quality"] = sq if sq in ("good", "partial", "weak", "failed") else "weak"

    return out


def run_model_signal_extraction(
    company_name: str,
    raw_url: str,
    enrichment_row: dict,
    api_key: str,
    model_id: str = MODEL_STEP2,
    include_evidence: bool = True,
    search_provider: str = STEP2_PROVIDER_SERPER,
) -> dict:
    """
    Extract structured model signals from already-fetched enrichment context.
    Returns a dict with all MODEL_SIGNAL_FIELDS populated.
    Never calls Jina, Serper, or the web_search tool — uses only provided context.
    Cache key is provider-specific so Serper and Claude results don't overwrite each other.
    """
    empty = _build_model_signal_empty()

    domain = clean_domain(raw_url) or company_name
    _prov_code = "sg" if search_provider == STEP2_PROVIDER_SERPER else "cws"
    cache_key = f"model_signals_{_prov_code}_{domain or safe_filename(company_name or 'unknown')}"

    cached = load_cache(cache_key)
    if cached is not None and cached.get("version") == 1:
        try:
            return _coerce_model_signals(cached.get("signals", {}))
        except Exception:
            _delete_cache(cache_key)

    context_text = _build_enrichment_context(enrichment_row)

    prompt = (
        MODEL_SIGNAL_PROMPT_TEMPLATE
        .replace("__COMPANY_NAME__", company_name or "(unknown)")
        .replace("__DOMAIN__", domain or "(unknown)")
        .replace("__ENRICHMENT_CONTEXT__", context_text)
    )

    _STRICT_SUFFIX = "\n\nReturn ONLY raw JSON. No markdown, no backticks, no explanation."

    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model_id,
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = "".join(
            getattr(b, "text", "") for b in resp.content
            if getattr(b, "type", "") == "text"
        ).strip()

        try:
            raw_json = _parse_json_response(raw_text)
        except (json.JSONDecodeError, ValueError):
            resp2 = client.messages.create(
                model=model_id,
                max_tokens=2500,
                messages=[{"role": "user", "content": prompt + _STRICT_SUFFIX}],
            )
            raw_text2 = "".join(
                getattr(b, "text", "") for b in resp2.content
                if getattr(b, "type", "") == "text"
            ).strip()
            raw_json = _parse_json_response(raw_text2)

        signals = _coerce_model_signals(raw_json)
        save_cache(cache_key, {"version": 1, "signals": signals})

        if not include_evidence:
            for f in MODEL_SIGNAL_EVIDENCE_FIELDS:
                signals[f] = ""

        return signals

    except (json.JSONDecodeError, ValueError) as e:
        err_empty = _build_model_signal_empty()
        err_empty["model_signal_needs_manual_review"] = 1
        err_empty["model_signal_manual_review_reason"] = f"JSON parse error: {str(e)[:150]}"
        err_empty["model_signal_search_quality"] = "failed"
        return err_empty
    except anthropic.APIError as e:
        err_empty = _build_model_signal_empty()
        err_empty["model_signal_needs_manual_review"] = 1
        err_empty["model_signal_manual_review_reason"] = f"API error: {str(e)[:150]}"
        err_empty["model_signal_search_quality"] = "failed"
        return err_empty
    except Exception as e:
        err_empty = _build_model_signal_empty()
        err_empty["model_signal_needs_manual_review"] = 1
        err_empty["model_signal_manual_review_reason"] = f"Error: {type(e).__name__}: {str(e)[:150]}"
        err_empty["model_signal_search_quality"] = "failed"
        return err_empty


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


def _lusha_raw_keys_summary(raw: dict) -> str:
    """
    Return a compact key-path summary of the top two levels of a dict.
    Used only for debugging — contains no secret values.
    Example: "top: data,meta; data: name,domain,industry"
    """
    if not isinstance(raw, dict):
        return f"(not a dict: {type(raw).__name__})"
    top_keys = list(raw.keys())
    parts = [f"top: {','.join(str(k) for k in top_keys)}"]
    for k in top_keys[:5]:
        v = raw.get(k)
        if isinstance(v, dict) and v:
            parts.append(f"{k}: {','.join(str(sk) for sk in list(v.keys())[:15])}")
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            parts.append(f"{k}[0]: {','.join(str(sk) for sk in list(v[0].keys())[:15])}")
    return "; ".join(parts)


def _resolve_lusha_company_node(raw: dict) -> dict:
    """
    Try all known Lusha response nesting paths and return the dict that most
    likely contains the actual company record.

    Lusha API v2 wraps company data under raw["data"]; some versions use
    raw["company"], raw["data"]["company"], raw["companies"][0], or raw["results"][0].
    Fall back to raw itself if nothing better is found.
    """
    if not isinstance(raw, dict):
        return {}

    # Ordered list of extraction strategies
    candidates = []

    # raw["data"] — most common v2 envelope
    d = raw.get("data")
    if isinstance(d, dict) and d:
        # raw["data"]["company"] — double-wrapped
        dd = d.get("company")
        if isinstance(dd, dict) and dd:
            candidates.append(dd)
        else:
            candidates.append(d)

    # raw["company"]
    c = raw.get("company")
    if isinstance(c, dict) and c:
        candidates.append(c)

    # raw["companies"][0]
    clist = raw.get("companies")
    if isinstance(clist, list) and clist and isinstance(clist[0], dict):
        candidates.append(clist[0])

    # raw["results"][0]
    rlist = raw.get("results")
    if isinstance(rlist, list) and rlist and isinstance(rlist[0], dict):
        candidates.append(rlist[0])

    # raw itself as last resort
    candidates.append(raw)

    # Score each candidate by how many recognisable company fields it has
    _score_keys = {
        "name", "company_name", "industry", "description", "size",
        "employee_range", "employees", "country", "city", "domain",
        "linkedin", "founded", "type",
    }
    def _score(node):
        return sum(1 for k in node if k.lower() in _score_keys)

    best = max(candidates, key=_score, default=raw)
    return best if isinstance(best, dict) else raw


def _map_lusha_api_fields(raw: dict, source_url: str) -> dict:
    """
    Map a raw Lusha API response to LUSHA_API_FIELDS keys.
    Probes all known nesting structures defensively.
    lusha_api_raw_keys contains only key names for debugging — no values.
    """
    raw_keys_summary = _lusha_raw_keys_summary(raw)

    company = _resolve_lusha_company_node(raw)

    def _sv(node: dict, *keys) -> str:
        """Extract the first non-empty string value for any of the given keys."""
        for k in keys:
            v = node.get(k)
            if v is not None and str(v).strip() not in ("", "None", "null", "0"):
                return str(v).strip()
        return ""

    # Location — Lusha often nests under company["location"]
    loc = company.get("location") or {}
    if not isinstance(loc, dict):
        loc = {}

    def _loc(key: str) -> str:
        v = loc.get(key)
        if v is not None and str(v).strip() not in ("", "None", "null"):
            return str(v).strip()
        return ""

    country   = _sv(company, "country", "hq_country") or _loc("country") or _loc("countryCode")
    city      = _sv(company, "city", "hq_city")       or _loc("city")
    continent = _sv(company, "continent")              or _loc("continent")

    # Domain
    domain = (
        clean_domain(_sv(company, "domain", "website"))
        or clean_domain(_sv(raw,     "domain"))
        or clean_domain(source_url)
    )

    # Lists: specialties and technologies
    def _join_list(node: dict, *keys) -> str:
        for k in keys:
            v = node.get(k)
            if isinstance(v, list):
                return ", ".join(str(x) for x in v if x)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    specialties  = _join_list(company, "specialties",  "specialties_list",  "expertises")
    technologies = _join_list(company, "technologies", "technology_stack",  "tech_stack")

    # Funding — Lusha often nests under company["funding"]
    funding = company.get("funding") or {}
    if not isinstance(funding, dict):
        funding = {}

    def _fund(key: str, *fallback_keys) -> str:
        v = _sv(funding, key, *fallback_keys)
        if v:
            return v
        return _sv(company, key, *fallback_keys)

    return {
        "lusha_api_company_name":         _sv(company, "name", "company_name", "companyName"),
        "lusha_api_domain":               domain,
        "lusha_api_description":          _sv(company, "description", "about", "summary"),
        "lusha_api_founded_year":         _sv(company, "founded", "founded_year", "year_founded", "foundedYear"),
        "lusha_api_employee_range":       _sv(company, "size", "employee_range", "employees", "company_size",
                                              "employeeRange", "employeeCount", "headcount"),
        "lusha_api_revenue_range":        _sv(company, "revenue_range", "revenue", "annual_revenue",
                                              "revenueRange", "annualRevenue"),
        "lusha_api_industry":             _sv(company, "industry", "main_industry", "primaryIndustry"),
        "lusha_api_sub_industry":         _sv(company, "sub_industry", "sub_category", "subIndustry"),
        "lusha_api_company_type":         _sv(company, "type", "company_type", "companyType"),
        "lusha_api_country":              country,
        "lusha_api_city":                 city,
        "lusha_api_continent":            continent,
        "lusha_api_linkedin_url":         _sv(company, "linkedin", "linkedin_url", "linkedinUrl"),
        "lusha_api_specialties":          specialties,
        "lusha_api_technologies":         technologies,
        "lusha_api_total_funding_amount": _fund("totalAmount",    "total_funding_amount", "total_funding"),
        "lusha_api_total_funding_rounds": _fund("totalRounds",    "total_funding_rounds", "funding_rounds"),
        "lusha_api_last_round_type":      _fund("lastRoundType",  "last_round_type",      "last_funding_type"),
        "lusha_api_last_round_amount":    _fund("lastRoundAmount","last_round_amount",     "last_funding_amount"),
        "lusha_api_last_round_date":      _fund("lastRoundDate",  "last_round_date",       "last_funding_date"),
        "lusha_api_ipo_status":           _sv(company, "ipo", "ipo_status", "ipoStatus"),
        # Debug-only key summary (key names only, no values)
        "lusha_api_raw_keys":             raw_keys_summary,
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
    if not _lusha_has_useful_data(lusha_fields):
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


# Fields that determine whether a Lusha API response contains useful company data
_LUSHA_USEFUL_FIELDS = [
    "lusha_api_company_name",
    "lusha_api_industry",
    "lusha_api_employee_range",
    "lusha_api_country",
    "lusha_api_description",
]


def _lusha_has_useful_data(fields: dict) -> bool:
    return any(fields.get(f, "").strip() for f in _LUSHA_USEFUL_FIELDS)


def run_lusha_api_enrichment(
    company_name: str,
    raw_url: str,
    api_key: str,
) -> tuple:
    """
    Call the real Lusha Company API and return:
    (lusha_fields_dict, raw_json_dict, status, error_message)

    Status values:
      "ok"             — HTTP 200 AND at least one useful company field was mapped
      "empty_response" — HTTP 200 but no useful company data found in the response
      "not_found"      — Lusha returned 404 (no company match)
      "auth_error"     — 401 invalid key
      "rate_limit"     — 429 quota exceeded
      "timeout"        — request timed out
      "http_{N}"       — other HTTP error
      "parse_error"    — response body was not valid JSON
      "no_key"         — API key not provided
      "no_input"       — neither domain nor company name available
      "cached"         — result served from local file cache
    """
    _empty = {f: "" for f in LUSHA_API_FIELDS + ["lusha_api_raw_keys"]}

    api_key = (api_key or "").strip()
    if not api_key:
        return _empty, {}, "no_key", "Lusha API key not provided"

    domain = clean_domain(raw_url)
    cache_key = f"lusha_api_{domain or safe_filename(company_name or 'unknown')}"
    cached = load_cache(cache_key)
    if cached is not None:
        cached_status = cached.get("status", "")
        # Avoid returning "ok" for a previously-cached empty response
        if cached_status == "not_found":
            return _empty, cached.get("raw", {}), "not_found", "Lusha API: company not found (cached)"
        fields = _map_lusha_api_fields(cached.get("raw", {}), raw_url)
        effective_status = "cached" if _lusha_has_useful_data(fields) else "empty_response"
        return fields, cached.get("raw", {}), effective_status, ""

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
            return _empty, {}, "not_found", "Lusha API: company not found (404)"

        if resp.status_code == 401:
            return _empty, {}, "auth_error", "Lusha API: invalid or missing API key (401)"

        if resp.status_code == 429:
            return _empty, {}, "rate_limit", "Lusha API: rate limit exceeded (429)"

        resp.raise_for_status()

        raw = resp.json()
        fields = _map_lusha_api_fields(raw, raw_url)

        if _lusha_has_useful_data(fields):
            status = "ok"
            error  = ""
        else:
            status = "empty_response"
            error  = (
                f"Lusha returned HTTP 200 but no useful company fields were mapped. "
                f"Response keys: {fields.get('lusha_api_raw_keys', '(unknown)')}"
            )

        save_cache(cache_key, {"raw": raw, "status": status})
        return fields, raw, status, error

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
    extract_model_signals: bool = True,
    include_signal_evidence: bool = True,
    run_step1_enrichment: bool = True,
    run_step2_enrichment: bool = True,
    existing_lusha_data: dict | None = None,
) -> tuple:
    """
    Run optional Lusha API enrichment, then Step 1 (Jina + Claude extraction),
    then Step 2 (Claude web_search ICP), then model-signal extraction (Step 3).
    Returns (combined_fields_dict, debug_record_dict).
    """
    url          = raw_url.strip() if raw_url else ""
    company_name = company_name.strip() if company_name else ""

    row = {f: "" for f in ALL_ENRICHMENT_FIELDS}

    # Populate row with any pre-existing Lusha/Lucia data from the input file
    # so downstream steps (Step 2, Step 3) can use it as context.
    _lusha_fields_set = set(LUSHA_API_FIELDS + LUSHA_API_META_FIELDS + STEP1_FIELDS)
    if existing_lusha_data:
        for _k, _v in existing_lusha_data.items():
            if _k in _lusha_fields_set:
                row[_k] = _v

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
    if run_step1_enrichment:
        s1_fields, s1_raw, s1_in, s1_out, s1_status, s1_err, s1_pw_dbg = run_step1(
            url, company_name, api_key, delay,
            use_playwright=use_playwright, model_step1=model_step1,
        )
        # Only overwrite existing non-empty Lusha fields if the new value is non-empty
        if existing_lusha_data:
            for _sf, _sv in s1_fields.items():
                if _sv or not row.get(_sf):
                    row[_sf] = _sv
        else:
            row.update(s1_fields)
        row["step1_status"]     = s1_status
        row["step1_run_status"] = s1_status
        row["step1_tokens_in"]  = str(s1_in)
        row["step1_tokens_out"] = str(s1_out)
        row["step1_cost_usd"]   = f"{calc_cost(s1_in, s1_out):.6f}"
    else:
        # Step 1 skipped — use existing data already loaded into row above
        s1_fields = {}
        s1_raw    = {}
        s1_in = s1_out = 0
        s1_status = "skipped_existing_data"
        s1_err    = ""
        s1_pw_dbg = {"playwright_attempted": False, "playwright_result": "skipped"}
        row["step1_status"]     = "skipped_existing_data"
        row["step1_run_status"] = "skipped_existing_data"
        row["step1_cost_usd"]   = "0.000000"
        row["lucia_api_called"] = 0
        if existing_lusha_data:
            row["lucia_data_status"] = "existing_preserved"
        else:
            row["lucia_data_status"] = "missing_not_requested"

    # ── Step 2 ────────────────────────────────────────────────────────────────
    if run_step2_enrichment:
        s2_fields, s2_raw, s2_in, s2_out, s2_status, s2_err, s2_cache_create, s2_cache_read = run_step2(
            url, company_name, api_key, delay, model_step2=model_step2,
            _debug_callback=_debug_callback,
            search_provider=search_provider, serper_key=serper_key,
            dry_run=dry_run,
        )
        row.update(s2_fields)
        row["step2_status"]        = s2_status
        row["step2_provider_used"] = search_provider
        row["step2_tokens_in"]     = str(s2_in)
        row["step2_tokens_out"]    = str(s2_out)
        row["step2_cost_usd"]      = f"{calc_cost(s2_in, s2_out):.6f}"
    else:
        s2_fields = _ICP_EMPTY.copy()
        s2_raw    = {}
        s2_in = s2_out = 0
        s2_status = "skipped"
        s2_err    = ""
        s2_cache_create = s2_cache_read = 0
        row["step2_status"]        = "skipped"
        row["step2_provider_used"] = search_provider
        row["step2_cost_usd"]      = "0.000000"

    # ── Combined metadata ─────────────────────────────────────────────────────
    total_in  = s1_in  + s2_in
    total_out = s1_out + s2_out
    row["total_tokens_in"]  = str(total_in)
    row["total_tokens_out"] = str(total_out)
    row["total_cost_usd"]   = f"{calc_cost(total_in, total_out):.6f}"

    has_s1 = _step1_has_data(s1_fields) if run_step1_enrichment else bool(existing_lusha_data)
    has_s2 = any(s2_fields.get(f, "") for f in ICP_FIELDS[:3])

    if not run_step1_enrichment and existing_lusha_data:
        row["enrichment_status"] = "existing_lusha_preserved" if has_s2 else "existing_lusha_only"
    elif has_s1:
        row["enrichment_status"] = s1_status if has_s2 else f"{s1_status}_step1_only"
    else:
        row["enrichment_status"] = "no_data"

    err_parts = [p for p in [s1_err, s2_err] if p]
    row["error_message"] = " | ".join(err_parts)

    flag_review(row, company_name)

    # ── Step 3 — Model-signal extraction ─────────────────────────────────────
    if extract_model_signals and api_key and not dry_run:
        try:
            ms_fields = run_model_signal_extraction(
                company_name=company_name,
                raw_url=raw_url,
                enrichment_row=row,
                api_key=api_key,
                model_id=model_step2,
                include_evidence=include_signal_evidence,
                search_provider=search_provider,
            )
            row.update(ms_fields)
        except Exception as _ms_exc:
            # Never let Step 3 failures abort the run
            _ms_err = _build_model_signal_empty()
            _ms_err["model_signal_needs_manual_review"] = 1
            _ms_err["model_signal_manual_review_reason"] = (
                f"Step 3 extraction failed: {type(_ms_exc).__name__}: {str(_ms_exc)[:120]}"
            )
            _ms_err["model_signal_search_quality"] = "failed"
            row.update(_ms_err)
    else:
        # Fill defaults when signal extraction is disabled or dry-run
        row.update(_build_model_signal_empty())

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

def _build_model_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a trimmed model_features sheet for the Excel export.
    Contains: original input columns + all model signal score/binary columns.
    Evidence columns are excluded here (they live in the main Enriched sheet).
    employee_size_score is included only when it already exists in df.
    """
    keep: list[str] = []

    # Preserve all original input columns (anything not in the enrichment field list)
    enrichment_col_set = set(ALL_ENRICHMENT_FIELDS)
    input_cols = [c for c in df.columns if c not in enrichment_col_set]
    keep.extend(input_cols)

    # Add score and binary signal columns (no evidence columns)
    signal_cols = [
        c for c in MODEL_SIGNAL_SCORE_FIELDS + MODEL_SIGNAL_BINARY_FIELDS
        if c in df.columns
    ]
    keep.extend(signal_cols)

    # Include employee_size_score only when already present
    if "employee_size_score" in df.columns:
        keep.append("employee_size_score")

    keep = list(dict.fromkeys(keep))  # deduplicate, preserve order
    available = [c for c in keep if c in df.columns]
    return df[available].copy()


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Enriched")
        try:
            mf_df = _build_model_features_df(df)
            if not mf_df.empty:
                mf_df.to_excel(writer, index=False, sheet_name="model_features")
        except Exception:
            pass
        try:
            ev_cols = [c for c in df.columns if c.endswith("_evidence")]
            if ev_cols:
                # Include company name/domain columns plus all evidence columns
                id_cols = [c for c in df.columns if c not in set(ALL_ENRICHMENT_FIELDS)][:3]
                qa_cols = list(dict.fromkeys(id_cols + ev_cols))
                qa_df = df[[c for c in qa_cols if c in df.columns]]
                qa_df.to_excel(writer, index=False, sheet_name="qa_evidence")
        except Exception:
            pass
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
    prov  = st.session_state.get("_step2_provider",   STEP2_PROVIDER_SERPER)
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
    """Write DataFrame to an Excel file at *path* on disk (two sheets)."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Enriched")
        try:
            mf_df = _build_model_features_df(df)
            if not mf_df.empty:
                mf_df.to_excel(writer, index=False, sheet_name="model_features")
        except Exception:
            pass
        try:
            ev_cols = [c for c in df.columns if c.endswith("_evidence")]
            if ev_cols:
                # Include company name/domain columns plus all evidence columns
                id_cols = [c for c in df.columns if c not in set(ALL_ENRICHMENT_FIELDS)][:3]
                qa_cols = list(dict.fromkeys(id_cols + ev_cols))
                qa_df = df[[c for c in qa_cols if c in df.columns]]
                qa_df.to_excel(writer, index=False, sheet_name="qa_evidence")
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Rich Excel report builder
# Creates: Input | Summary | Company Profiles (visible)
#          Advanced Evidence | Scoring Settings | Enriched |
#          model_features | qa_evidence (hidden)
# ─────────────────────────────────────────────────────────────────────────────

def _xl_get(row_dict: dict, *keys: str, default: str = "") -> str:
    """Return first non-blank string value from row_dict for any of *keys."""
    _null = {"", "nan", "none", "n/a", "unknown", "null", "nat"}
    for k in keys:
        v = row_dict.get(k, "")
        if v is not None and str(v).strip().lower() not in _null:
            return str(v).strip()
    return default


def _xl_write_df(ws, df: pd.DataFrame) -> None:
    """Write DataFrame to worksheet with a styled header row."""
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter

    hdr_fill = PatternFill(start_color="0B4A92", end_color="0B4A92", fill_type="solid")
    hdr_font = Font(bold=True, color="FFFFFF", size=10)
    for ci, col in enumerate(df.columns, 1):
        cell = ws.cell(row=1, column=ci, value=str(col))
        cell.fill = hdr_fill
        cell.font = hdr_font
        cell.alignment = Alignment(horizontal="left", vertical="center")

    for ri, record in enumerate(df.to_dict("records"), 2):
        for ci, col in enumerate(df.columns, 1):
            val = record[col]
            if val is None or (isinstance(val, float) and pd.isna(val)):
                val = ""
            ws.cell(row=ri, column=ci, value=val)

    for col_cells in ws.columns:
        try:
            max_len = max((len(str(c.value or "")) for c in col_cells), default=8)
            ws.column_dimensions[get_column_letter(col_cells[0].column)].width = min(
                max(max_len + 2, 10), 60
            )
        except Exception:
            pass


def _xl_write_scoring_settings(ws) -> None:
    """Write scoring constants to the Scoring Settings sheet."""
    from openpyxl.styles import Font, PatternFill, Alignment
    try:
        from commercial_fit_scoring import (
            INTERCEPT as _INT, LEAN_COEFFICIENTS as _LC,
            SIZE_BANDS as _SB, TIER_THRESHOLDS as _TT,
        )
    except ImportError:
        ws.cell(row=1, column=1, value="Scoring module not available")
        return

    hdr_fill = PatternFill(start_color="0B4A92", end_color="0B4A92", fill_type="solid")
    hdr_font = Font(bold=True, color="FFFFFF", size=10)
    bold = Font(bold=True, size=10)
    norm = Font(size=10)

    r = 1
    ws.cell(row=r, column=1, value="Scoring Settings").font = Font(bold=True, size=13)
    r += 2

    ws.cell(row=r, column=1, value="Intercept").font = bold
    ws.cell(row=r, column=2, value=_INT).font = norm
    r += 2

    ws.cell(row=r, column=1, value="Lean Model Coefficients").font = bold
    r += 1
    for h, c in (("Field", 1), ("Coefficient", 2)):
        cell = ws.cell(row=r, column=c, value=h)
        cell.fill = hdr_fill
        cell.font = hdr_font
    r += 1
    for field, coef in sorted(_LC.items(), key=lambda x: -x[1]):
        ws.cell(row=r, column=1, value=field).font = norm
        ws.cell(row=r, column=2, value=coef).font = norm
        r += 1
    r += 1

    ws.cell(row=r, column=1, value="Size Bands").font = bold
    r += 1
    for h, c in (("Max Employees", 1), ("Score", 2)):
        cell = ws.cell(row=r, column=c, value=h)
        cell.fill = hdr_fill
        cell.font = hdr_font
    r += 1
    for upper, score in _SB:
        ws.cell(row=r, column=1, value=str(upper)).font = norm
        ws.cell(row=r, column=2, value=score).font = norm
        r += 1
    r += 1

    ws.cell(row=r, column=1, value="Tier Thresholds").font = bold
    r += 1
    for h, c in (("Min Score", 1), ("Tier", 2)):
        cell = ws.cell(row=r, column=c, value=h)
        cell.fill = hdr_fill
        cell.font = hdr_font
    r += 1
    for threshold, label in _TT:
        ws.cell(row=r, column=1, value=threshold).font = norm
        ws.cell(row=r, column=2, value=label).font = norm
        r += 1

    ws.column_dimensions["A"].width = 45
    ws.column_dimensions["B"].width = 20


def _xl_write_company_profiles(ws, df: pd.DataFrame,
                                name_col: str | None,
                                domain_col: str | None) -> dict:
    """
    Write one formatted block per company.
    Returns {df_row_index: profile_start_row} (0-based index → 1-based Excel row).
    """
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter

    # Column widths: A=28, B=80, C=14, D=18, E=70, F=75
    for ci, w in enumerate([28, 80, 14, 18, 70, 75], 1):
        ws.column_dimensions[get_column_letter(ci)].width = w

    tier_fills = {
        "Tier 1": PatternFill(start_color="0B4A92", end_color="0B4A92", fill_type="solid"),
        "Tier 2": PatternFill(start_color="1F7AC4", end_color="1F7AC4", fill_type="solid"),
        "Tier 3": PatternFill(start_color="E47228", end_color="E47228", fill_type="solid"),
        "Pass":   PatternFill(start_color="7F7F7F", end_color="7F7F7F", fill_type="solid"),
    }
    default_fill = PatternFill(start_color="2B4C7E", end_color="2B4C7E", fill_type="solid")
    label_font  = Font(bold=True,  size=10, color="1A1A1A")
    value_font  = Font(bold=False, size=10, color="1A1A1A")
    header_font = Font(bold=True,  size=12, color="FFFFFF")
    wrap_align  = Alignment(horizontal="left", vertical="top",  wrap_text=True)
    left_align  = Alignment(horizontal="left", vertical="center")
    right_align = Alignment(horizontal="right", vertical="center")
    alt_fill    = PatternFill(start_color="F7F9FC", end_color="F7F9FC", fill_type="solid")

    profile_start_rows: dict = {}
    cur = 1

    for df_idx, (_, row) in enumerate(df.iterrows()):
        rd = row.to_dict()
        profile_start_rows[df_idx] = cur

        company = _xl_get(rd, name_col or "", "lusha_company_name", "lusha_api_company_name")
        domain  = _xl_get(rd, domain_col or "", "lusha_domain", "lusha_api_domain")
        tier    = _xl_get(rd, "commercial_tier")
        try:
            score_val = float(rd.get("final_commercial_fit_score", 0) or 0)
            score_str = f"{score_val:.1f}"
        except (ValueError, TypeError):
            score_str = ""
        industry  = _xl_get(rd, "lusha_industry",       "lusha_api_industry")
        country   = _xl_get(rd, "lusha_country",        "lusha_api_country")
        employees = _xl_get(rd, "lusha_employee_range", "lusha_api_employee_range")
        why       = _xl_get(rd, "icp_why_relevant")
        signals   = _xl_get(rd, "icp_buying_signals",   "top_score_drivers")
        gaps      = _xl_get(rd, "weak_score_drivers",   "missing_scoring_fields")
        evidence  = _xl_get(rd, "icp_evidence")
        interp    = _xl_get(rd, "scoring_notes")

        hdr_text = company
        if tier:
            hdr_text = f"{company}  [{tier}]"
        if domain:
            hdr_text += f"  —  {domain}"
        fill = tier_fills.get(tier, default_fill)

        # ── Header row (merged A:F) ──────────────────────────────────────────
        ws.cell(row=cur, column=1, value=hdr_text)
        ws.merge_cells(start_row=cur, start_column=1, end_row=cur, end_column=6)
        for ci in range(1, 7):
            ws.cell(row=cur, column=ci).fill = fill
        ws.cell(row=cur, column=1).font = header_font
        ws.cell(row=cur, column=1).alignment = Alignment(
            horizontal="left", vertical="center", indent=1
        )
        ws.row_dimensions[cur].height = 24
        cur += 1

        # ── Score / Tier / Employees row ─────────────────────────────────────
        _compact = [
            (1, "Score",     label_font, right_align),
            (2, score_str,   value_font, left_align),
            (3, "Tier",      label_font, right_align),
            (4, tier,        value_font, left_align),
            (5, "Employees", label_font, right_align),
            (6, employees,   value_font, left_align),
        ]
        for ci, val, fnt, aln in _compact:
            c = ws.cell(row=cur, column=ci, value=val)
            c.font = fnt
            c.alignment = aln
            c.fill = alt_fill
        ws.row_dimensions[cur].height = 20
        cur += 1

        # ── Industry / Country row ────────────────────────────────────────────
        _ic = [
            (1, "Industry", label_font, right_align),
            (2, industry,   value_font, left_align),
            (3, "Country",  label_font, right_align),
            (4, country,    value_font, left_align),
        ]
        for ci, val, fnt, aln in _ic:
            c = ws.cell(row=cur, column=ci, value=val)
            c.font = fnt
            c.alignment = aln
        ws.row_dimensions[cur].height = 20
        cur += 1

        # ── Long-text rows ────────────────────────────────────────────────────
        long_rows = [
            ("Why Relevant",             why,      55),
            ("Top Positive Signals",     signals,  55),
            ("Gaps / Missing Signals",   gaps,     45),
            ("Evidence",                 evidence, 70),
            ("Commercial Interpretation", interp,  50),
        ]
        for label, content, height in long_rows:
            lc = ws.cell(row=cur, column=1, value=label)
            lc.font = label_font
            lc.alignment = Alignment(horizontal="left", vertical="top")

            vc = ws.cell(row=cur, column=2, value=content)
            vc.font = value_font
            vc.alignment = wrap_align
            try:
                ws.merge_cells(start_row=cur, start_column=2,
                               end_row=cur, end_column=6)
            except Exception:
                pass
            ws.row_dimensions[cur].height = height
            cur += 1

        # ── Separator ─────────────────────────────────────────────────────────
        ws.row_dimensions[cur].height = 10
        cur += 1

    return profile_start_rows


def _xl_write_summary(ws, df: pd.DataFrame,
                      name_col: str | None,
                      domain_col: str | None,
                      profile_start_rows: dict) -> None:
    """Write the Summary sheet with company scores and Open Profile hyperlinks."""
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.formatting.rule import DataBarRule
    from openpyxl.utils import get_column_letter

    headers = [
        "Company Name",
        "Company Domain / URL",
        "Final Commercial Fit Score",
        "Commercial Tier",
        "Open Profile",
    ]
    widths = [30, 35, 24, 16, 18]

    hdr_fill = PatternFill(start_color="0B4A92", end_color="0B4A92", fill_type="solid")
    hdr_font = Font(bold=True, color="FFFFFF", size=11)
    link_font = Font(color="0B4A92", underline="single", size=10)
    tier_colors = {
        "Tier 1": "D6E4F7",
        "Tier 2": "D9EAD3",
        "Tier 3": "FCE5CD",
        "Pass":   "F4CCCC",
    }

    for ci, (hdr, w) in enumerate(zip(headers, widths), 1):
        c = ws.cell(row=1, column=ci, value=hdr)
        c.fill = hdr_fill
        c.font = hdr_font
        c.alignment = Alignment(horizontal="center", vertical="center")
        ws.column_dimensions[get_column_letter(ci)].width = w
    ws.row_dimensions[1].height = 24
    ws.freeze_panes = "A2"

    for df_idx, (_, row) in enumerate(df.iterrows()):
        rd   = row.to_dict()
        xrow = df_idx + 2   # Excel row (1-indexed header + offset)

        company = _xl_get(rd, name_col or "", "lusha_company_name", "lusha_api_company_name")
        domain  = _xl_get(rd, domain_col or "", "lusha_domain", "lusha_api_domain")
        tier    = _xl_get(rd, "commercial_tier")
        try:
            score = float(rd.get("final_commercial_fit_score", "") or "")
        except (ValueError, TypeError):
            score = ""

        row_fill = PatternFill(
            start_color=tier_colors.get(tier, "FFFFFF"),
            end_color=tier_colors.get(tier, "FFFFFF"),
            fill_type="solid",
        ) if tier else None

        for ci, val in enumerate([company, domain, score, tier], 1):
            c = ws.cell(row=xrow, column=ci, value=val)
            if row_fill:
                c.fill = row_fill
            c.alignment = Alignment(horizontal="left", vertical="center")

        # Open Profile hyperlink — jump a few rows below the profile header
        prof_start = profile_start_rows.get(df_idx, 1)
        link_target_row = prof_start + 4   # lands near "Why Relevant"
        lc = ws.cell(row=xrow, column=5, value="Open Profile")
        lc.hyperlink = f"#'Company Profiles'!A{link_target_row}"
        lc.font = link_font
        lc.alignment = Alignment(horizontal="center", vertical="center")

        ws.row_dimensions[xrow].height = 20

    # Blue data bar on score column (C)
    n = len(df)
    if n > 0:
        try:
            rule = DataBarRule(
                start_type="num", start_value=0,
                end_type="num", end_value=10,
                color="0070C0",
            )
            ws.conditional_formatting.add(f"C2:C{n + 1}", rule)
        except Exception:
            pass


def build_rich_excel_bytes(
    df: pd.DataFrame,
    name_col: str | None = None,
    domain_col: str | None = None,
    df_input_original: pd.DataFrame | None = None,
) -> bytes:
    """
    Build a fully formatted multi-sheet Excel workbook.

    Visible sheets  : Input | Summary | Company Profiles
    Hidden sheets   : Advanced Evidence | Scoring Settings |
                      Enriched | model_features | qa_evidence

    name_col / domain_col: when provided (e.g. from detect_columns), these are
    used directly so person-level columns in Lucia exports are never chosen.
    df_input_original: original contact-level df to write to the Input sheet
    (for Lucia exports where df is already deduplicated to company level).
    """
    import openpyxl
    from openpyxl import Workbook

    wb = Workbook()
    wb.remove(wb.active)   # discard the default empty sheet

    # Identify original input columns (not added by enrichment pipeline)
    _all_enrich = set(ALL_ENRICHMENT_FIELDS + list(_SCORE_OUTPUT_COLS or []))
    input_cols_list = [c for c in df.columns if c not in _all_enrich]

    # Determine name / domain columns for Summary and Company Profiles
    if name_col is not None:
        # Explicit columns passed in — use them directly
        _name_guess   = name_col
        _domain_guess = domain_col
    else:
        # Prefer exact Lucia/Lusha company columns before falling back to heuristics
        _col_set = set(input_cols_list)
        if "Company Name" in _col_set:
            _name_guess = "Company Name"
        elif "lusha_api_company_name" in _col_set:
            _name_guess = "lusha_api_company_name"
        else:
            _name_guess = None

        if "Company Domain" in _col_set:
            _domain_guess = "Company Domain"
        elif "Company Website" in _col_set:
            _domain_guess = "Company Website"
        elif "lusha_api_domain" in _col_set:
            _domain_guess = "lusha_api_domain"
        else:
            _domain_guess = None

        # Heuristic fallback: substring match, skipping person-level column names
        _person_prefixes = ("first ", "last ", "middle ", "contact ")
        if _name_guess is None:
            for c in input_cols_list:
                cl = c.lower()
                if any(h in cl for h in _COMPANY_HINTS) and not cl.startswith(_person_prefixes):
                    _name_guess = c
                    break
        if _domain_guess is None:
            for c in input_cols_list:
                cl = c.lower()
                if any(h in cl for h in _DOMAIN_HINTS) and "linkedin" not in cl:
                    _domain_guess = c
                    break

        if _name_guess is None and input_cols_list:
            _name_guess = input_cols_list[0]
        if _domain_guess is None and len(input_cols_list) > 1:
            _domain_guess = input_cols_list[1]

    # ── Input (visible) ───────────────────────────────────────────────────────
    ws_input = wb.create_sheet("Input")
    if df_input_original is not None:
        # Lucia exports: write the original contact-level CSV unchanged
        _xl_write_df(ws_input, df_input_original)
    else:
        _xl_write_df(ws_input, df[input_cols_list] if input_cols_list else df)

    # ── Company Profiles (visible) ────────────────────────────────────────────
    ws_profiles = wb.create_sheet("Company Profiles")
    profile_rows = _xl_write_company_profiles(
        ws_profiles, df, _name_guess, _domain_guess
    )

    # ── Summary (visible) ─────────────────────────────────────────────────────
    ws_summary = wb.create_sheet("Summary")
    _xl_write_summary(ws_summary, df, _name_guess, _domain_guess, profile_rows)

    # Re-order: Input → Summary → Company Profiles
    wb._sheets = [ws_input, ws_summary, ws_profiles]

    # ── Advanced Evidence (hidden) ────────────────────────────────────────────
    try:
        ev_cols = [c for c in df.columns if c.endswith("_evidence")]
        if ev_cols:
            id_cols = input_cols_list[:3]
            qa_cols = list(dict.fromkeys(id_cols + ev_cols))
            qa_df   = df[[c for c in qa_cols if c in df.columns]]
            ws_ev   = wb.create_sheet("Advanced Evidence")
            _xl_write_df(ws_ev, qa_df)
            ws_ev.sheet_state = "hidden"
    except Exception:
        pass

    # ── Scoring Settings (hidden) ─────────────────────────────────────────────
    try:
        ws_sc = wb.create_sheet("Scoring Settings")
        _xl_write_scoring_settings(ws_sc)
        ws_sc.sheet_state = "hidden"
    except Exception:
        pass

    # ── Enriched (hidden) ─────────────────────────────────────────────────────
    try:
        ws_en = wb.create_sheet("Enriched")
        _xl_write_df(ws_en, df)
        ws_en.sheet_state = "hidden"
    except Exception:
        pass

    # ── model_features (hidden) ───────────────────────────────────────────────
    try:
        mf = _build_model_features_df(df)
        if not mf.empty:
            ws_mf = wb.create_sheet("model_features")
            _xl_write_df(ws_mf, mf)
            ws_mf.sheet_state = "hidden"
    except Exception:
        pass

    # ── qa_evidence (hidden) ──────────────────────────────────────────────────
    try:
        ev_cols = [c for c in df.columns if c.endswith("_evidence")]
        if ev_cols:
            id_cols = input_cols_list[:3]
            qa_cols = list(dict.fromkeys(id_cols + ev_cols))
            qa_df   = df[[c for c in qa_cols if c in df.columns]]
            ws_qa   = wb.create_sheet("qa_evidence")
            _xl_write_df(ws_qa, qa_df)
            ws_qa.sheet_state = "hidden"
    except Exception:
        pass

    # ── Validate and return bytes ─────────────────────────────────────────────
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.getvalue()


def _validate_rich_excel(xl_bytes: bytes) -> dict:
    """
    Reload workbook and verify sheet visibility.
    Returns {"valid": bool, "visible": [...], "hidden": [...], "issues": [...]}.
    """
    try:
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(xl_bytes))
    except Exception as exc:
        return {"valid": False, "visible": [], "hidden": [], "issues": [str(exc)]}

    visible = [ws.title for ws in wb.worksheets if ws.sheet_state == "visible"]
    hidden  = [ws.title for ws in wb.worksheets if ws.sheet_state != "visible"]
    issues  = []

    expected_visible = {"Input", "Summary", "Company Profiles"}
    for s in expected_visible - set(visible):
        issues.append(f"'{s}' should be visible but is missing or hidden")
    for s in set(visible) - expected_visible:
        issues.append(f"'{s}' is visible but should be hidden")

    return {"valid": not issues, "visible": visible, "hidden": hidden, "issues": issues}


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


# ─────────────────────────────────────────────────────────────────────────────
# Per-company run-folder autosave
# ─────────────────────────────────────────────────────────────────────────────

def create_run_folder(base_folder: str, run_tag: str) -> str:
    """
    Create and return the path of a new timestamped run folder.
    Structure:
      {base_folder}/run_{YYYYMMDD_HHMMSS}_{run_tag}/
        rows/
        logs/
    Raises OSError on permission / path errors (caller must catch).
    """
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    name  = f"run_{stamp}_{run_tag}" if run_tag else f"run_{stamp}"
    run_path = Path(base_folder.strip()) / name
    (run_path / "rows").mkdir(parents=True, exist_ok=True)
    (run_path / "logs").mkdir(parents=True, exist_ok=True)
    (run_path / "cache").mkdir(parents=True, exist_ok=True)
    return str(run_path)


def save_company_result_to_run_folder(
    row_index: int,
    company_name: str,
    input_row: pd.Series,
    enriched_fields: dict,
    debug_record: dict,
    run_dir: str,
) -> tuple:
    """
    Write one processed company as JSON to {run_dir}/rows/row_{NNNN}_{name}.json.
    Returns (success: bool, message: str).
    Does not include API keys or request headers.
    """
    try:
        safe_name = safe_filename(company_name or f"row_{row_index:04d}")
        fname = Path(run_dir) / "rows" / f"row_{row_index:04d}_{safe_name}.json"

        # Sanitise the debug record — strip any raw API response objects but keep
        # scalar metadata; the full raw JSON is already available in the cache files.
        _safe_debug: dict = {}
        for k, v in (debug_record or {}).items():
            if isinstance(v, (str, int, float, bool, type(None))):
                _safe_debug[k] = v
            elif isinstance(v, dict):
                # Truncate large nested dicts (e.g. raw API responses) to key list
                _safe_debug[k] = (
                    v if len(json.dumps(v, default=str)) < 4096
                    else {"_truncated": True, "keys": list(v.keys())}
                )
            else:
                _safe_debug[k] = str(v)[:500]

        record = {
            "row_index":       row_index,
            "company_name":    company_name,
            "saved_at_utc":    datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "input_row":       {str(k): str(v) for k, v in input_row.to_dict().items()},
            "enriched_fields": {k: str(v) for k, v in (enriched_fields or {}).items()},
            "debug_record":    _safe_debug,
        }
        fname.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return True, str(fname)
    except Exception as exc:
        return False, f"Per-company save failed for '{company_name}': {exc}"


def save_partial_outputs_to_run_folder(
    results: list,
    debug_records: list,
    df_work: pd.DataFrame,
    active_fields: list,
    run_dir: str,
    elm_mode: bool = False,
    row_count: int = 0,
) -> tuple:
    """
    Write cumulative files to run_dir:
      latest_results.csv    — always overwritten
      latest_results.xlsx   — always overwritten
      processing_log.csv    — always overwritten
      checkpoint_NNN.xlsx   — when row_count is a multiple of CHECKPOINT_EVERY
    Returns (success: bool, message: str).
    """
    try:
        rdir = Path(run_dir)
        partial_df = build_partial_df(results, df_work, active_fields)
        log_df     = make_log_df(debug_records, elm_mode=elm_mode)

        partial_df.to_csv(rdir / "latest_results.csv", index=False, encoding="utf-8-sig")
        log_df.to_csv(rdir / "processing_log.csv",     index=False, encoding="utf-8-sig")
        df_to_excel_bytes_write(partial_df, str(rdir / "latest_results.xlsx"))

        if row_count > 0 and row_count % CHECKPOINT_EVERY == 0:
            df_to_excel_bytes_write(partial_df, str(rdir / f"checkpoint_{row_count:04d}.xlsx"))

        return True, f"{len(results)} rows written to {run_dir}"
    except Exception as exc:
        return False, f"Partial output save failed: {exc}"


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
        _final_auto_saved=False, _last_local_save="",
        _auto_dl_count=0, _auto_dl_last_msg="",
        _step2_debug_log="", _step2_prompt_records=[],
        _dry_run_records=[], _search_output_records=[], _step2_debug_files=[],
        _zero_cost_preview=False, _dry_run_preview_count=0,
        _per_company_autosave_run_dir="",
        _per_company_autosave_last_saved="",
        _per_company_autosave_last_error="",
        _final_save_path="", _final_save_error="",
    )


def build_and_finish(results: list, debug_records: list, df_work: pd.DataFrame,
                     field_list: list | None = None) -> None:
    flist       = field_list if field_list is not None else ALL_ENRICHMENT_FIELDS
    df_out      = df_work.head(len(results)).copy().reset_index(drop=True)
    enriched_df = pd.DataFrame(results)
    for col in flist:
        df_out[col] = enriched_df[col].values if col in enriched_df.columns else ""
    if _SCORING_AVAILABLE and not st.session_state.get("_elm_mode", False):
        try:
            df_out = _score_dataframe(df_out)
        except Exception:
            pass
    ss_set(
        processing=False, stop_requested=False,
        enrichment_done=True, df_enriched=df_out,
        debug_records=debug_records,
    )
    st.rerun()


# =============================================================================
# UI
# =============================================================================

if not os.environ.get("_STREAMLIT_ENTRYPOINT"):
    # When run directly (streamlit run enrich_clients_claude.py) set up the page.
    # When launched via streamlit_app.py the entrypoint already called these.
    import pathlib as _pl
    import base64 as _b64
    st.set_page_config(
        page_title="mYngle · Company Enrichment",
        page_icon="🏢",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 880px;
            padding-top: 2.5rem;
            padding-bottom: 3rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    _logo = _pl.Path(__file__).parent / "Mynglelogofinal.jpg"
    _logo_src = (
        "data:image/jpeg;base64," + _b64.b64encode(_logo.read_bytes()).decode()
        if _logo.exists() else ""
    )
    st.markdown(
        f"""
        <style>
        div[data-testid="stMarkdownContainer"]:has(.brand-header) {{
            overflow: visible;
        }}
        .brand-header {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 12px;
            padding-top: 6px;
            overflow: visible;
        }}
        .brand-logo {{
            width: 145px;
            height: auto;
            display: block;
            flex-shrink: 0;
            overflow: visible;
            object-fit: contain;
        }}
        .brand-title {{
            font-size: 34px;
            font-weight: 700;
            color: #0B1F3A;
            line-height: 1.05;
            white-space: nowrap;
            margin: 0;
            padding: 0;
        }}
        </style>
        <div class="brand-header">
            {'<img src="' + _logo_src + '" class="brand-logo" alt="mYngle">' if _logo_src else ''}
            <div class="brand-title">company enrichment</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Upload a company file. "
        "The app will enrich and score the companies, "
        "then generate an Excel report."
    )

# =============================================================================
# API KEY — secrets only, no sidebar input
# =============================================================================

api_key = ""
_api_key_error = ""
try:
    api_key = (st.secrets.get("ANTHROPIC_API_KEY", "") or "").strip()
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
    serper_key = (st.secrets.get("SERPER_API_KEY", "") or "").strip()
except Exception:
    pass

# Lusha API key — optional, only needed when Lusha API enrichment is enabled.
# Required entry in .streamlit/secrets.toml: LUSHA_API_KEY = "your-key"
lusha_api_key = ""
try:
    lusha_api_key = (st.secrets.get("LUSHA_API_KEY", "") or "").strip()
except Exception:
    pass

# =============================================================================
# SIDEBAR — shown only when SHOW_ADVANCED_SETTINGS is True
# =============================================================================

# Re-check at runtime so the Streamlit secret can override the module constant.
_show_adv: bool = SHOW_ADVANCED_SETTINGS
try:
    _show_adv = bool(st.secrets.get("SHOW_ADVANCED_SETTINGS", SHOW_ADVANCED_SETTINGS))
except Exception:
    pass

if _show_adv:
 with st.sidebar:
    # ── 1. Enrichment mode ────────────────────────────────────────────────────
    enrichment_mode = st.radio(
        "Enrichment mode",
        options=["Full Claude enrichment", "Extreme Light Mode (no API)"],
        index=0,                        # default: Full Claude enrichment
        key="enrichment_mode_radio",
        help=(
            "**Full Claude**: Jina AI + Claude API — requires ANTHROPIC_API_KEY.\n\n"
            "**Extreme Light Mode**: fetches company pages with requests/BeautifulSoup, "
            "extracts keyword signals and normalised scores — no API key, zero tokens."
        ),
    )
    _elm_mode = enrichment_mode == "Extreme Light Mode (no API)"
    st.divider()

    st.header("Settings")

    # ── 2. API key statuses ───────────────────────────────────────────────────
    if api_key:
        st.success("✓ Anthropic API key loaded")
    else:
        st.error("⚠ Anthropic API key missing")
    _current_provider = st.session_state.get("_step2_provider", STEP2_PROVIDER_SERPER)
    if serper_key:
        st.success("✓ Serper API key loaded")
    elif _current_provider == STEP2_PROVIDER_SERPER:
        st.error("⚠ Serper API key missing — required for Serper Google Search")
    else:
        st.caption("ⓘ Serper API key not set (only needed for Serper Google Search)")
    if lusha_api_key:
        st.success("✓ Lusha API key loaded")
    else:
        st.caption("ⓘ Lusha API key not set (needed when Lusha enrichment is enabled)")

    st.divider()

    # ── 3. Lusha API enrichment ───────────────────────────────────────────────
    enable_lusha_api = st.checkbox(
        "Enable Lusha API enrichment",
        value=True,                     # default: enabled
        key="enable_lusha_api_checkbox",
        help=(
            "Calls the real Lusha Company API to enrich each row with verified "
            "firmographic data. Requires LUSHA_API_KEY in .streamlit/secrets.toml.\n\n"
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

    # ── 3b. Step selection ────────────────────────────────────────────────────
    _has_lusha_input = ss("_has_lusha_input", False)
    if _has_lusha_input:
        st.info(
            "ℹ️ Existing Lucia/Lusha fields detected in the uploaded file. "
            "Step 1 firmographic enrichment is disabled by default — "
            "existing values will be preserved."
        )

    run_step1_enrichment = st.checkbox(
        "Run Step 1 firmographic enrichment",
        value=not _has_lusha_input,
        key="run_step1_enrichment_checkbox",
        help=(
            "Runs Jina AI + Claude extraction to fill firmographic fields. "
            "Disable when the uploaded file already contains Lusha/Lucia enrichment. "
            "When disabled, existing Lusha/Lucia values are preserved unchanged."
        ),
    )
    if run_step1_enrichment and _has_lusha_input:
        st.warning(
            "⚠️ Step 1 is enabled but Lusha/Lucia fields already exist in the file. "
            "Existing non-empty values will be preserved."
        )
    st.session_state["_run_step1_enrichment"] = run_step1_enrichment

    run_step2_enrichment = st.checkbox(
        "Run Step 2 ICP/web enrichment",
        value=True,
        key="run_step2_enrichment_checkbox",
        help="Runs web search + Claude analysis to fill ICP buying signal fields.",
    )
    st.session_state["_run_step2_enrichment"] = run_step2_enrichment

    st.divider()

    # ── 4. Model selection ────────────────────────────────────────────────────
    model_step1_label = st.selectbox(
        "Model — Step 1 (firmographics)",
        options=list(AVAILABLE_MODELS.keys()),
        index=0,                        # default: Haiku 4.5
        help="Used for extracting structured company data from scraped pages. Haiku is sufficient here.",
    )
    model_step2_label = st.selectbox(
        "Model — Step 2 (ICP web search)",
        options=list(AVAILABLE_MODELS.keys()),
        index=0,                        # default: Haiku 4.5
        help="Used for the agentic web search. Sonnet gives better signal detection but costs ~5x more.",
    )
    selected_model_step1 = AVAILABLE_MODELS[model_step1_label]
    selected_model_step2 = AVAILABLE_MODELS[model_step2_label]
    st.session_state["_model_step1"] = selected_model_step1
    st.session_state["_model_step2"] = selected_model_step2

    st.divider()

    # ── 5. Step 2 web search provider ─────────────────────────────────────────
    _provider_options = [STEP2_PROVIDER_SERPER, STEP2_PROVIDER_CLAUDE]
    step2_provider = st.selectbox(
        "Step 2 web search provider",
        options=_provider_options,
        index=0,                        # default: Serper Google Search
        key="step2_provider_selectbox",
        help=(
            f"**{STEP2_PROVIDER_SERPER}** (default): calls the Serper API for Google "
            "results, then Claude analyzes the snippets. "
            "Requires `SERPER_API_KEY` in `.streamlit/secrets.toml`.\n\n"
            f"**{STEP2_PROVIDER_CLAUDE}**: uses Anthropic's built-in web_search tool — "
            "no extra API key needed, but costs more tokens per company."
        ),
    )
    st.session_state["_step2_provider"] = step2_provider
    if step2_provider == STEP2_PROVIDER_SERPER and not serper_key:
        st.error(
            "⚠️ SERPER_API_KEY missing — add it to `.streamlit/secrets.toml` "
            "or switch to Claude Web Search."
        )
    elif step2_provider == STEP2_PROVIDER_CLAUDE:
        st.info("ℹ️ Claude Web Search: uses Anthropic web_search tool, no Serper key needed.")

    st.divider()

    # ── 6. Per-company local autosave ─────────────────────────────────────────
    st.subheader("💾 Per-company local autosave")
    pca_enabled = st.checkbox(
        "Enable per-company local autosave",
        value=ss("_per_company_autosave_enabled", True),  # default: enabled
        key="pca_enabled_checkbox",
        help=(
            "Writes one JSON file per company immediately after processing, "
            "plus cumulative CSV and Excel files in a timestamped run folder. "
            "Nothing is lost if the app crashes or the browser refreshes."
        ),
    )
    if pca_enabled:
        _pca_default = (
            ss("_per_company_autosave_base_folder", "") or _PER_COMPANY_AUTOSAVE_DEFAULT_DIR
        )
        pca_folder = st.text_input(
            "Autosave base folder",
            value=_pca_default,
            placeholder=_PER_COMPANY_AUTOSAVE_DEFAULT_DIR,
            key="pca_folder_input",
        )
        _pca_folder_eff = (pca_folder or "").strip() or _PER_COMPANY_AUTOSAVE_DEFAULT_DIR
        ss_set(
            _per_company_autosave_enabled=True,
            _per_company_autosave_base_folder=_pca_folder_eff,
        )
        st.caption(
            "Only works when the app runs locally. "
            "On Streamlit Cloud this saves to the cloud container, not your PC."
        )
        _pca_run_dir = ss("_per_company_autosave_run_dir", "")
        if _pca_run_dir and ss("processing", False):
            st.caption(f"📂 Run folder: `{_pca_run_dir}`")
        _pca_last = ss("_per_company_autosave_last_saved", "")
        if _pca_last:
            st.caption(f"✔ Last saved: {_pca_last}")
        _pca_err = ss("_per_company_autosave_last_error", "")
        if _pca_err:
            st.warning(f"⚠ Autosave error: {_pca_err}")
    else:
        ss_set(
            _per_company_autosave_enabled=False,
            _per_company_autosave_base_folder=ss(
                "_per_company_autosave_base_folder", _PER_COMPANY_AUTOSAVE_DEFAULT_DIR
            ),
        )

    st.divider()

    # ── 7. Model signal extraction ────────────────────────────────────────────
    st.markdown("**Model signal extraction**")

    extract_model_signals = st.checkbox(
        "Extract model signals (Step 3)",
        value=True,
        key="extract_model_signals_checkbox",
        help=(
            "After Step 1 + Step 2, runs a structured signal-extraction pass that "
            "produces discrete ordinal scores (0–3) and binary columns ready for "
            "logistic-regression training.  Adds ~1 extra Claude call per company."
        ),
    )
    st.session_state["_extract_model_signals"] = extract_model_signals

    include_signal_evidence = st.checkbox(
        "Include QA evidence columns",
        value=True,
        key="include_signal_evidence_checkbox",
        help=(
            "Include the *_evidence columns alongside each score/binary field. "
            "Useful for manual QA. Disable to reduce Excel column count."
        ),
    )
    st.session_state["_include_signal_evidence"] = include_signal_evidence

    st.divider()

    # ── 8. Advanced options ───────────────────────────────────────────────────
    st.markdown("**Advanced options**")

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

    # ── Crash-recovery autosave status ────────────────────────────────────────
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

    # ── Local auto-save (snapshot every N rows) ───────────────────────────────
    st.divider()
    st.subheader("📁 Local auto-save")
    local_save_enabled = st.checkbox(
        "Enable local auto-save",
        value=ss("local_save_enabled", True),   # default: enabled
        key="local_save_enabled",
        help=(
            "Overwrites latest_results.xlsx/csv after every company. "
            f"Writes checkpoint_NNN.xlsx every {CHECKPOINT_EVERY} companies. "
            "Only works when the app runs locally."
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
        st.caption(f"📁 Saving to: **{_eff_path}**")
        _last_local = ss("_last_local_save", "")
        if _last_local:
            st.caption(f"Last save: {_last_local}")
        else:
            st.caption(
                f"Saves **latest_results.xlsx** after every company + "
                f"**checkpoint_NNN.xlsx** every {CHECKPOINT_EVERY} rows."
            )
    else:
        ss_set(_local_save_path="", _local_save_enabled=False)
        st.caption("When disabled, only the in-browser download button is available.")

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

else:
    # ── Simplified mode: no sidebar shown — use sensible defaults ─────────────
    _elm_mode  = False
    debug_mode = False
    delay_sec  = 1.0
    ss_set(
        _enable_lusha_api          = True,
        _run_step1_enrichment      = not ss("_has_lusha_input", False),
        _run_step2_enrichment      = True,
        _extract_model_signals     = True,
        _include_signal_evidence   = True,
        _step2_dry_run             = False,
        _zero_cost_preview         = False,
        _show_step2_debug          = False,
        _save_step2_debug          = True,
        _use_playwright            = _PLAYWRIGHT_AVAILABLE,
        _local_save_enabled        = False,
        _local_save_path           = _DEFAULT_DOWNLOAD_DIR,
        _model_step1               = MODEL_STEP1,
        _model_step2               = MODEL_STEP2,
        _step2_provider            = STEP2_PROVIDER_SERPER,
        _elm_mode                  = False,
        _per_company_autosave_enabled = False,
    )

# =============================================================================
# INPUT MODE
# =============================================================================

_sc_df: pd.DataFrame | None = None
_sc_name_col   = "company_name"
_sc_domain_col = "domain"

if _show_adv:
    _mode_col, _ = st.columns([2, 3])
    with _mode_col:
        _app_mode = st.radio(
            "Input mode",
            ["Batch Upload", "Single Company"],
            horizontal=True,
            key="app_mode_radio",
        )

    if _app_mode == "Single Company":
        st.divider()
        st.subheader("Single company enrichment & scoring")
        st.caption(
            "Enter a company name and optional domain. "
            "The app will run the full enrichment pipeline and compute the commercial fit score."
        )
        _sc_f1, _sc_f2 = st.columns(2)
        with _sc_f1:
            _sc_name_input = st.text_input(
                "Company name *", key="sc_company_name",
                placeholder="e.g. Acme Corp",
            )
        with _sc_f2:
            _sc_url_input = st.text_input(
                "Domain or URL (optional)", key="sc_company_url",
                placeholder="e.g. acme.com",
            )
        if _sc_name_input:
            _sc_df = pd.DataFrame([{
                "company_name": _sc_name_input.strip(),
                "domain": (_sc_url_input or "").strip(),
            }])
        else:
            st.info("Enter a company name above to begin.")
else:
    _app_mode = "Batch Upload"

# =============================================================================
# STEP 1 — Upload file  (Batch mode only)
# =============================================================================

uploaded = None
if _app_mode == "Batch Upload":
    st.divider()
    if _show_adv:
        st.subheader("Step 1 · Upload your file")
    uploaded = st.file_uploader(
        "Drag and drop here, or click to browse  (.xlsx · .xls · .csv)",
        type=["xlsx", "xls", "csv"],
        label_visibility="collapsed" if not _show_adv else "visible",
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
            # Detect existing Lusha/Lucia enrichment columns
            _detected_lusha  = detect_lusha_columns(df_loaded)
            _is_lucia_loaded = is_lucia_contact_export(df_loaded)
            ss_set(
                _lusha_cols_in_input=_detected_lusha,
                _has_lusha_input=bool(_detected_lusha),
                _is_lucia_export=_is_lucia_loaded,
            )
        except Exception as exc:
            ss_set(
                file_error=str(exc),
                _lusha_cols_in_input=[], _has_lusha_input=False,
                _is_lucia_export=False,
            )

df_raw: pd.DataFrame | None = ss("df_raw")
file_error: str | None      = ss("file_error")

if file_error:
    st.error(f"Could not read the file: {file_error}")
elif uploaded and df_raw is not None:
    if ss("_is_lucia_export", False):
        # Count unique companies for the friendly message
        _l_domain_col = get_lucia_domain_col(df_raw)
        _l_name_col   = get_lucia_name_col(df_raw)
        if _l_domain_col:
            _dedup_keys = df_raw[_l_domain_col].apply(
                lambda x: clean_domain(str(x)) if pd.notna(x) else ""
            )
            if _l_name_col:
                _dedup_keys = _dedup_keys.where(
                    _dedup_keys != "",
                    df_raw[_l_name_col].apply(lambda x: str(x).strip().lower()),
                )
        elif _l_name_col:
            _dedup_keys = df_raw[_l_name_col].apply(lambda x: str(x).strip().lower())
        else:
            _dedup_keys = pd.Series(range(len(df_raw))).astype(str)
        _n_contacts   = len(df_raw)
        _n_companies  = _dedup_keys.nunique()
        if _n_contacts == _n_companies:
            st.success(
                f"✅ **{ss('file_name')}** loaded — "
                f"{_n_contacts:,} unique companies ready"
            )
        else:
            st.success(
                f"✅ **{ss('file_name')}** loaded — "
                f"{_n_contacts:,} contact rows · {_n_companies:,} unique companies ready"
            )
    else:
        # Type 1 simple company list: count unique non-empty company names
        _t1_name_col, _ = detect_columns(df_raw)
        if _t1_name_col and _t1_name_col in df_raw.columns:
            _t1_count = int(
                df_raw[_t1_name_col]
                .dropna()
                .astype(str)
                .str.strip()
                .replace("", pd.NA)
                .dropna()
                .nunique()
            )
        else:
            _t1_count = len(df_raw)
        st.success(
            f"✅ **{ss('file_name')}** loaded — "
            f"{_t1_count:,} {'company' if _t1_count == 1 else 'companies'} ready"
        )
    if _show_adv:
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

# ── Column detection and processing scope ─────────────────────────────────────

name_col     = _sc_name_col if _app_mode == "Single Company" else None
domain_col   = _sc_domain_col if _app_mode == "Single Company" else None
n_to_process = 1 if (_app_mode == "Single Company" and _sc_df is not None) else 0

if df_raw is not None:
    if _show_adv:
        st.divider()
        st.subheader("Step 2 · Preview")
        st.dataframe(df_raw.head(), use_container_width=True)
        st.caption(f"{len(df_raw):,} rows · {len(df_raw.columns)} columns")

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
    else:
        # Auto-detect columns silently
        auto_name_col, auto_domain_col = detect_columns(df_raw)
        name_col   = auto_name_col
        domain_col = auto_domain_col
        n_to_process = len(df_raw)

# =============================================================================
# STEP 5 — Start enrichment
# =============================================================================

st.divider()
currently_processing = ss("processing", False)
enrichment_done      = ss("enrichment_done", False)

blocking: list = []
_active_provider     = ss("_step2_provider",    STEP2_PROVIDER_SERPER)
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
if _app_mode == "Batch Upload" and uploaded is None:
    blocking.append("No file uploaded yet.")
if _app_mode == "Single Company" and (_sc_df is None or _sc_df.empty):
    blocking.append("Enter a company name to proceed.")
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
    if _show_adv:
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

_start_label = "▶ Enrich & Score" if _app_mode == "Single Company" else "▶ Start enrichment"
start_btn = st.button(
    _start_label,
    type="primary",
    use_container_width=True,
    disabled=(bool(blocking) or currently_processing),
    key="start_button",
)

if start_btn and not blocking and not currently_processing:
    if _app_mode == "Single Company" and _sc_df is not None:
        df_work    = _sc_df.copy()
        name_col   = _sc_name_col
        domain_col = _sc_domain_col
        _df_raw_for_input = None
    else:
        _is_lucia_run = ss("_is_lucia_export", False)
        if _is_lucia_run and df_raw is not None:
            # Deduplicate to company level; keep original for Input sheet
            _df_deduped = deduplicate_lucia_export(df_raw)
            df_work = _df_deduped.head(n_to_process).copy()
            _df_raw_for_input = df_raw.copy()
            # Ensure correct company columns are used
            name_col   = get_lucia_name_col(df_raw) or name_col
            domain_col = get_lucia_domain_col(df_raw) or domain_col
        else:
            df_work = df_raw.head(n_to_process).copy()
            _df_raw_for_input = None
    resume_mode  = ss("_resume_mode", False)
    if not resume_mode:
        autosave_clear()   # wipe any previous autosave on a fresh start

    # ── Create per-company autosave run folder if feature is enabled ──────────
    _pca_run_dir_new = ""
    _pca_enabled_now = ss("_per_company_autosave_enabled", False)
    if _pca_enabled_now:
        _pca_base = (
            ss("_per_company_autosave_base_folder", "") or _PER_COMPANY_AUTOSAVE_DEFAULT_DIR
        )
        try:
            # Build a temporary run tag from current sidebar selections
            _tmp_prov  = ss("_step2_provider", STEP2_PROVIDER_SERPER)
            _tmp_model = ss("_model_step2",    MODEL_STEP2)
            _tmp_lusha = ss("_enable_lusha_api", False)
            _tmp_tag   = f"{get_provider_code(_tmp_prov)}_{get_model_code(_tmp_model)}"
            if _tmp_lusha:
                _tmp_tag = f"{_tmp_tag}_lusha"
            _pca_run_dir_new = create_run_folder(_pca_base, _tmp_tag)
        except Exception as _pca_err:
            st.warning(f"⚠ Could not create autosave run folder: {_pca_err}")
            _pca_run_dir_new = ""

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
        _local_save_enabled=ss("_local_save_enabled", True),
        _final_auto_saved=False, _last_local_save="",
        _use_playwright=ss("_use_playwright", True),
        _model_step1=ss("_model_step1", MODEL_STEP1),
        _model_step2=ss("_model_step2", MODEL_STEP2),
        _step2_provider=ss("_step2_provider", STEP2_PROVIDER_SERPER),
        _serper_key=serper_key,
        _step2_dry_run=ss("_step2_dry_run", False),
        _zero_cost_preview=ss("_zero_cost_preview", False),
        _enable_lusha_api=ss("_enable_lusha_api", False),
        _lusha_api_key=lusha_api_key,
        _extract_model_signals=ss("_extract_model_signals", True),
        _include_signal_evidence=ss("_include_signal_evidence", True),
        _run_step1_enrichment=ss("_run_step1_enrichment", True),
        _run_step2_enrichment=ss("_run_step2_enrichment", True),
        _dry_run_records=[], _search_output_records=[], _step2_debug_files=[],
        _dry_run_preview_count=0,
        # Lucia export: original (contact-level) df for the Input sheet
        _df_raw_original=_df_raw_for_input,
        # Per-company autosave
        _per_company_autosave_run_dir=_pca_run_dir_new,
        _per_company_autosave_last_saved="",
        _per_company_autosave_last_error="",
        _final_save_path="", _final_save_error="",
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
    _step2_provider_run  = ss("_step2_provider", STEP2_PROVIDER_SERPER)
    _serper_key_run      = ss("_serper_key", "")
    _dry_run_run         = ss("_step2_dry_run", False)
    _zero_cost_run       = ss("_zero_cost_preview", False)
    _enable_lusha_api_run      = ss("_enable_lusha_api", False)
    _lusha_api_key_run         = ss("_lusha_api_key", "")
    _extract_model_signals_run = ss("_extract_model_signals", True)
    _include_signal_evidence_run = ss("_include_signal_evidence", True)
    _run_step1_enrichment_run  = ss("_run_step1_enrichment", True)
    _run_step2_enrichment_run  = ss("_run_step2_enrichment", True)
    _pca_enabled_run  = ss("_per_company_autosave_enabled", False)
    _pca_run_dir_run  = ss("_per_company_autosave_run_dir", "")
    total_in          = ss("total_tokens_in", 0)
    total_out         = ss("total_tokens_out", 0)
    total_cost        = ss("total_cost_usd", 0.0)
    total_cache_read  = ss("total_cache_read_tokens", 0)
    total_cache_create = ss("total_cache_create_tokens", 0)

    if _show_adv and st.button("⏹ Stop after current row", key="stop_button"):
        ss_set(stop_requested=True)
        st.rerun()

    # Look ahead to get the current company name for the status line
    _cur_company = ""
    if idx < _n and df_work is not None:
        try:
            _cur_company = str(df_work.iloc[idx].get(_name_col, "")).strip()
        except Exception:
            pass
    _progress_text = (
        f"Processing {idx + 1} of {_n}"
        + (f" · {_cur_company}" if _cur_company else "")
    )
    st.progress(idx / _n if _n else 1.0, text=_progress_text)

    if _show_adv:
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

    # ── Intermediate download buttons — advanced mode only ─────────────────────
    if _show_adv and results:
        _partial_df = build_partial_df(results, df_work, _active_fields)
        _n_done     = len(_partial_df)
        _stamp      = ts()
        with st.expander(
            f"⬇ Download intermediate results ({_n_done} rows so far)", expanded=True
        ):
            st.caption("These links download via the browser without interrupting processing.")
            _html_dl_buttons(_partial_df, _n_done, _stamp)

    # ── Step 2 dry run preview ────────────────────────────────────────────────
    if _show_adv and _dry_run_run and not _elm_mode_run:
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

        # Normalize domain-only values to a URL (prepend https:// if needed)
        if raw_url and not raw_url.startswith(("http://", "https://")):
            raw_url = normalize_url(raw_url)

        # Extract existing Lusha/Lucia field values from the input row
        _lusha_field_set = set(LUSHA_API_FIELDS + LUSHA_API_META_FIELDS + STEP1_FIELDS)
        _existing_lusha = {
            k: (str(v) if not isinstance(v, str) else v)
            for k, v in input_row.items()
            if k in _lusha_field_set and v is not None and str(v).strip() not in ("", "nan", "NaN", "None")
        }
        # For Lucia/Lusha contact exports map "Company X" columns to lusha_api_* names
        if ss("_is_lucia_export", False):
            _lucia_mapped = map_lucia_export_row(input_row.to_dict())
            _existing_lusha.update(_lucia_mapped)

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
            f"**{company_name or '(empty)'}**  ({idx + 1} of {_n})",
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
                    fields["enrichment_status"]   = "zero_cost_preview"
                    fields["step2_status"]        = _s2_status
                    fields["step2_provider_used"] = _step2_provider_run
                    fields.update(icp_fields)
                    # Fill model-signal defaults (no API call in zero-cost mode)
                    fields.update(_build_model_signal_empty())
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
                        extract_model_signals=_extract_model_signals_run,
                        include_signal_evidence=_include_signal_evidence_run,
                        run_step1_enrichment=_run_step1_enrichment_run,
                        run_step2_enrichment=_run_step2_enrichment_run,
                        existing_lusha_data=_existing_lusha or None,
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

        # ── Per-company run-folder autosave ───────────────────────────────────
        if _pca_enabled_run and _pca_run_dir_run:
            # 1. Per-company JSON
            _pca_ok, _pca_msg = save_company_result_to_run_folder(
                row_index=_new_idx,
                company_name=company_name,
                input_row=input_row,
                enriched_fields=fields,
                debug_record=dbg,
                run_dir=_pca_run_dir_run,
            )
            if _pca_ok:
                ss_set(
                    _per_company_autosave_last_saved=(
                        f"{company_name or f'row {_new_idx}'} → "
                        f"row_{_new_idx:04d}_{safe_filename(company_name or 'row')}.json"
                    ),
                    _per_company_autosave_last_error="",
                )
            else:
                ss_set(_per_company_autosave_last_error=_pca_msg[:200])

            # 2. Cumulative partial files (latest_results.* + checkpoint on multiples)
            _pca_ok2, _pca_msg2 = save_partial_outputs_to_run_folder(
                results=results,
                debug_records=debug_records,
                df_work=df_work,
                active_fields=_active_fields,
                run_dir=_pca_run_dir_run,
                elm_mode=_elm_mode_run,
                row_count=_new_idx,
            )
            if _pca_ok2:
                _save_label = f"{_new_idx} rows → latest_results.xlsx"
                if _new_idx > 0 and _new_idx % CHECKPOINT_EVERY == 0:
                    _save_label += f" + checkpoint_{_new_idx:04d}.xlsx"
                ss_set(_last_local_save=_save_label)
            else:
                ss_set(
                    _per_company_autosave_last_error=_pca_msg2[:200],
                    _last_local_save=f"⚠ Save failed: {_pca_msg2[:120]}",
                )

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



def _render_advanced_results(
    df_enriched, debug_records_done, processed, _elm_done, debug_mode
):
    """Render advanced result sections — only shown when SHOW_ADVANCED_SETTINGS is True.

    Sections: status metrics, token cost, results table, commercial fit scoring,
    download buttons, debug details.
    """
    _done_fields = ELM_ALL_FIELDS if _elm_done else ALL_ENRICHMENT_FIELDS
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
        # Scoring columns first (if available)
        _sc_summary = [c for c in (_SCORE_OUTPUT_COLS or []) if c in df_enriched.columns]
        summary_cols = orig_cols + _sc_summary + [
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
                "icp_direct_language_competitor_signal",
                "icp_online_language_learning_signal",
                "icp_broader_lnd_platform_signal",
                "icp_evidence", "icp_likely_training_interest",
                "icp_why_relevant", "icp_potential_buyer_function",
                # Cost
                "total_tokens_in", "total_tokens_out", "total_cost_usd",
                "error_message",
            ]
            if c in df_enriched.columns
        ]
        st.dataframe(df_enriched[summary_cols], use_container_width=True, height=400)
        _tabs = [
            "Step 1 — All firmographic columns",
            "Step 2 — All ICP columns",
            "Model signals — scores & binaries",
            "Model signals — QA evidence",
        ]
        if _lusha_done:
            _tabs.append("Lusha API fields")
        _tab_objs = st.tabs(_tabs)
        with _tab_objs[0]:
            st.dataframe(df_enriched[[c for c in STEP1_FIELDS if c in df_enriched.columns]],
                         use_container_width=True)
        with _tab_objs[1]:
            st.dataframe(df_enriched[[c for c in ICP_FIELDS if c in df_enriched.columns]],
                         use_container_width=True)
        with _tab_objs[2]:
            _score_bin_cols = [
                c for c in MODEL_SIGNAL_SCORE_FIELDS + MODEL_SIGNAL_BINARY_FIELDS
                + MODEL_SIGNAL_QA_FIELDS
                if c in df_enriched.columns
            ]
            if _score_bin_cols:
                st.dataframe(df_enriched[_score_bin_cols], use_container_width=True)
            else:
                st.info("Model signal extraction was not run or is disabled.")
        with _tab_objs[3]:
            _evid_cols = [c for c in MODEL_SIGNAL_EVIDENCE_FIELDS if c in df_enriched.columns]
            if _evid_cols:
                st.dataframe(df_enriched[_evid_cols], use_container_width=True)
            else:
                st.info("No evidence columns found (evidence may be disabled or extraction not run).")
        if _lusha_done and len(_tab_objs) > 4:
            with _tab_objs[4]:
                _lusha_display_cols = [
                    c for c in LUSHA_API_FIELDS + LUSHA_API_META_FIELDS
                    if c in df_enriched.columns
                ]
                st.dataframe(df_enriched[_lusha_display_cols], use_container_width=True)

    # ── Commercial Fit Scoring ────────────────────────────────────────────────
    if not _elm_done and _SCORING_AVAILABLE:
        _score_key_col = "final_commercial_fit_score"
        _score_cols    = _SCORE_OUTPUT_COLS
        if _score_key_col in df_enriched.columns:
            st.divider()

            # ── Single company: score card ────────────────────────────────────
            if processed == 1:
                _sc_row = df_enriched.iloc[0]
                _fit   = _sc_row.get(_score_key_col, 0)
                _tier  = _sc_row.get("commercial_tier", "—")
                _icp   = _sc_row.get("icp_similarity_score", 0)
                _sz    = _sc_row.get("company_size_score", 0)
                _prob  = _sc_row.get("model_probability", 0)
                _top_d = _sc_row.get("top_score_drivers", "")
                _wk_d  = _sc_row.get("weak_score_drivers", "")
                _notes = _sc_row.get("scoring_notes", "")
                _dqf   = _sc_row.get("data_quality_flag", "")
                _gc    = _sc_row.get("global_complexity_score", "")
                _pd_s  = _sc_row.get("people_development_score", "")
                _cc    = _sc_row.get("commercial_complexity_score", "")

                _tier_emoji = {"Tier 1": "🟢", "Tier 2": "🟡", "Tier 3": "🟠", "Pass": "🔴"}.get(str(_tier), "⚪")
                _dq_label   = {"high": "✅ High", "medium": "⚠️ Medium", "low": "🔴 Low"}.get(str(_dqf), str(_dqf))

                st.subheader("🎯 Commercial Fit Score")
                st.caption(
                    "The score combines ICP similarity and company size. "
                    "ICP similarity is based on enriched buying signals. "
                    "Company size is used as a commercial weighting factor."
                )
                _m1, _m2, _m3, _m4 = st.columns(4)
                _m1.metric("Commercial Fit Score", f"{float(_fit):.2f} / 10")
                _m2.metric("Commercial Tier", f"{_tier_emoji} {_tier}")
                _m3.metric("ICP Similarity", f"{float(_icp):.2f} / 10")
                _m4.metric("Company Size", f"{_sz} / 10")

                _d1, _d2, _d3 = st.columns(3)
                _d1.metric("Global Complexity", f"{_gc} / 10" if _gc != "" else "—")
                _d2.metric("People Development", f"{_pd_s} / 10" if _pd_s != "" else "—")
                _d3.metric("Commercial Complexity", f"{_cc} / 10" if _cc != "" else "—")

                if _top_d and str(_top_d) != "none":
                    st.markdown(f"**Top score drivers:** {_top_d}")
                if _wk_d and str(_wk_d) != "none":
                    st.markdown(f"**Weak / missing drivers:** {_wk_d}")

                # Suggested cold-call opening angle from top driver
                _top_driver_field = (str(_top_d) or "").split(";")[0].split("=")[0].strip()
                _opening_map = {
                    "sig_explicit_lnd_score":    "I saw that {name} has a strong L&D programme — we help companies like yours scale language training across global teams.",
                    "sig_lnd_onboarding_score":  "Your structured onboarding process caught my attention — we specialise in language-ready onboarding for international hires.",
                    "sig_intl_footprint_score":  "{name}'s international presence is exactly the profile we work with — multilingual communication across offices is our core focus.",
                    "sig_foreign_hq_score":      "With {name}'s cross-border structure, language alignment between HQ and local teams is often a real friction point — that's where we come in.",
                    "sig_rapid_growth_score":    "{name}'s growth trajectory often brings language challenges — we help scaling companies maintain communication quality across markets.",
                    "sig_multicultural_score":   "The diverse workforce at {name} is a strong fit for our language learning programmes tailored for multicultural teams.",
                    "language_competitor_strength_score": "I noticed a competitor in your space is already investing in language training — this is a strong signal we can help {name} stay ahead.",
                }
                _company_display = str(df_enriched.iloc[0].get("company_name", "your company"))
                _opening = _opening_map.get(
                    _top_driver_field,
                    "Based on {name}'s profile, your team could benefit from targeted language and communication training across international operations.",
                ).replace("{name}", _company_display)
                with st.expander("💬 Suggested cold-call opening angle", expanded=True):
                    st.markdown(f"_{_opening}_")
                    st.caption("Generated from top score driver. Customise before use.")

                if _notes:
                    with st.expander("ℹ️ Scoring notes", expanded=False):
                        st.caption(str(_notes))
                if _dqf:
                    st.caption(f"Data quality: {_dq_label} · Model probability: {float(_prob):.1%}")

            else:
                # ── Batch: filtered + sorted score table ──────────────────────
                st.subheader("🎯 Commercial Fit Scoring")
                st.caption(
                    "The score combines ICP similarity and company size. "
                    "ICP similarity is based on enriched buying signals. "
                    "Company size is used as a commercial weighting factor. "
                    "Final Commercial Fit Score = 0.75 × ICP Similarity + 0.25 × Company Size."
                )

                with st.expander("ℹ️ About these scores", expanded=False):
                    st.markdown(
                        "1. **Model probability** — lean logistic regression on 7 key model-signal fields.\n"
                        "2. **ICP Similarity Score [1–10]** — rescaled sigmoid output.\n"
                        "3. **Company Size Score [1–10]** — 5-band employee-count mapping.\n"
                        "4. **Final Commercial Fit Score** = 0.75 × ICP Similarity + 0.25 × Company Size.\n"
                        "5. **Tier** — Tier 1 ≥ 7.5 · Tier 2 ≥ 6.0 · Tier 3 ≥ 4.5 · Pass < 4.5.\n"
                        "6. **Composite scores** — global complexity, people development, commercial complexity "
                        "(each 0–10, from signal groupings).\n\n"
                        "⚠️ Coefficients are placeholder values — update `LEAN_COEFFICIENTS` and `INTERCEPT` "
                        "in `commercial_fit_scoring.py` with fitted values from Results(3).xlsx."
                    )

                _tier_order  = ["Tier 1", "Tier 2", "Tier 3", "Pass"]
                _tier_counts = df_enriched["commercial_tier"].value_counts()
                _tier_colors = ["🟢", "🟡", "🟠", "🔴"]
                _tc = st.columns(4)
                for _i, (_tier, _emoji) in enumerate(zip(_tier_order, _tier_colors)):
                    _cnt = int(_tier_counts.get(_tier, 0))
                    _pct = f"{_cnt / max(processed, 1):.0%}"
                    _tc[_i].metric(f"{_emoji} {_tier}", _cnt, delta=_pct, delta_color="off")

                _fc1, _fc2, _fc3 = st.columns([2, 2, 2])
                with _fc1:
                    _tier_filter = st.selectbox(
                        "Show tiers",
                        ["All", "Tier 1 only", "Tier 1 & 2"],
                        key="score_tier_filter",
                    )
                with _fc2:
                    _min_score = st.slider(
                        "Minimum fit score",
                        min_value=0.0, max_value=10.0,
                        value=0.0, step=0.5,
                        key="score_min_slider",
                    )
                with _fc3:
                    _sort_by = st.selectbox(
                        "Sort by",
                        ["final_commercial_fit_score", "model_probability", "company_size_score"],
                        key="score_sort_col",
                    )

                _id_cols_sc = [
                    c for c in df_enriched.columns
                    if c not in set(ALL_ENRICHMENT_FIELDS + list(_score_cols))
                ][:2]
                _disp_core = [
                    "final_commercial_fit_score", "commercial_tier",
                    "model_probability", "icp_similarity_score", "company_size_score",
                    "top_score_drivers", "scoring_notes",
                    "global_complexity_score", "people_development_score",
                    "commercial_complexity_score", "data_quality_flag",
                ]
                _disp_cols = _id_cols_sc + [c for c in _disp_core if c in df_enriched.columns]
                _score_disp_df = df_enriched[[c for c in _disp_cols if c in df_enriched.columns]].copy()

                if _tier_filter == "Tier 1 only":
                    _score_disp_df = _score_disp_df[_score_disp_df["commercial_tier"] == "Tier 1"]
                elif _tier_filter == "Tier 1 & 2":
                    _score_disp_df = _score_disp_df[_score_disp_df["commercial_tier"].isin(["Tier 1", "Tier 2"])]
                if _min_score > 0 and "final_commercial_fit_score" in _score_disp_df.columns:
                    _score_disp_df = _score_disp_df[_score_disp_df["final_commercial_fit_score"] >= _min_score]
                if _sort_by in _score_disp_df.columns:
                    _score_disp_df = _score_disp_df.sort_values(_sort_by, ascending=False)

                st.dataframe(_score_disp_df, use_container_width=True, height=420)
                st.caption(f"Showing {len(_score_disp_df)} of {processed} companies.")

        elif not ss("_extract_model_signals", True):
            st.info(
                "ℹ️ Commercial fit scoring requires Step 3 model signal extraction. "
                "Enable **Extract model signals (Step 3)** in the sidebar and re-run."
            )

    # ── Downloads ─────────────────────────────────────────────────────────────
    st.subheader("Download results")
    log_df = make_log_df(debug_records_done, elm_mode=_elm_done)

    if _elm_done:
        _fname_prefix = "elm_results"
        _log_fname    = "elm_fetch_log.csv"
    else:
        _run_tag      = build_run_tag()
        _fname_prefix = f"enrichedResults_{ts()}"
        _log_fname    = f"processing_log_{_run_tag}_{ts()}.csv"
    _xl_help      = (
        "All original columns + keyword counts + normalized scores."
        if _elm_done else
        "Sheet 1 (Enriched): all enrichment columns. "
        "Sheet 2 (model_features): input columns + model signal scores + binary columns only."
    )
    _log_help     = (
        "One row per company: fetch status, pages fetched, total chars."
        if _elm_done else
        "One row per company: step statuses, token counts, costs, review flags."
    )

    _rich_xl = (
        build_rich_excel_bytes(
            df_enriched,
            name_col=ss("_name_col"),
            domain_col=ss("_domain_col"),
            df_input_original=ss("_df_raw_original"),
        )
        if not _elm_done else df_to_excel_bytes(df_enriched)
    )
    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button(
            "⬇ Results Excel (.xlsx)",
            data=_rich_xl,
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

    if not _elm_done:
        _mf_df = _build_model_features_df(df_enriched)
        if not _mf_df.empty:
            _mf_fname = f"model_features_{_run_tag}_{ts()}.csv"
            st.download_button(
                "⬇ Model features CSV (scores + binaries only)",
                data=df_to_csv_bytes(_mf_df),
                file_name=_mf_fname,
                mime="text/csv",
                use_container_width=True,
                help=(
                    "Input columns + all model signal score/binary columns. "
                    "No evidence columns. Ready for logistic regression."
                ),
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
                debug_enriched["lusha_api_raw_json_preview"] = [
                    json.dumps(d.get("lusha_api_raw_json"), ensure_ascii=False)[:1500]
                    if d.get("lusha_api_raw_json") else ""
                    for d in debug_records_done
                ] + [""] * max(0, len(debug_enriched) - len(debug_records_done))
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



if ss("enrichment_done", False):
    df_enriched: pd.DataFrame = ss("df_enriched")
    debug_records_done: list  = ss("debug_records", [])
    processed = len(df_enriched)
    _elm_done = ss("_elm_mode", False)
    _done_fields = ELM_ALL_FIELDS if _elm_done else ALL_ENRICHMENT_FIELDS

    st.divider()
    _done_dry_run   = ss("_step2_dry_run",     False)
    _done_zero_cost = ss("_zero_cost_preview", False)
    _processed_word = "company" if processed == 1 else "companies"
    if ss("stop_requested", False):
        st.warning(f"Enrichment stopped — **{processed:,}** {_processed_word} processed (partial).")
    else:
        st.success(f"✅ Ready · **{processed:,}** {_processed_word} processed")

    if _show_adv:
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

    # ── Auto-save final file into run folder (runs exactly once per completed run) ─
    _pca_done_enabled = ss("_per_company_autosave_enabled", False)
    _pca_done_dir     = ss("_per_company_autosave_run_dir", "")
    if not ss("_final_auto_saved", False):
        if _pca_done_enabled and _pca_done_dir:
            try:
                _pca_rdir = Path(_pca_done_dir)
                df_to_excel_bytes_write(df_enriched, str(_pca_rdir / "final_results.xlsx"))
                df_enriched.to_csv(_pca_rdir / "final_results.csv", index=False, encoding="utf-8-sig")
                # overwrite latest_results.* with the complete dataset too
                df_to_excel_bytes_write(df_enriched, str(_pca_rdir / "latest_results.xlsx"))
                df_enriched.to_csv(_pca_rdir / "latest_results.csv", index=False, encoding="utf-8-sig")
                ss_set(_final_auto_saved=True, _final_save_path=str(_pca_rdir / "final_results.xlsx"))
                st.info(f"📂 Final results saved to **{_pca_done_dir}**")
            except Exception as _fin_err:
                ss_set(_final_auto_saved=True, _final_save_path="",
                       _final_save_error=str(_fin_err))
        else:
            ss_set(_final_auto_saved=True)

    _final_xl_error = ss("_final_save_error", "")
    if _final_xl_error:
        st.warning(f"⚠ Final auto-save failed: {_final_xl_error}")

    # ── Step 2 debug files — download + preview ───────────────────────────────
    if _show_adv and not _elm_done:
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
        _fname_dl  = f"elm_results_{ts()}.xlsx"
        _dl_bytes  = df_to_excel_bytes(df_enriched)
    else:
        _fname_dl  = f"enrichedResults_{ts()}.xlsx"
        _dl_bytes  = build_rich_excel_bytes(
            df_enriched,
            name_col=ss("_name_col"),
            domain_col=ss("_domain_col"),
            df_input_original=ss("_df_raw_original"),
        )
    st.download_button(
        label="⬇ Download results",
        data=_dl_bytes,
        file_name=_fname_dl,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        type="primary",
    )

    if _show_adv:
        _render_advanced_results(
            df_enriched, debug_records_done, processed, _elm_done, debug_mode
        )

    # ── Restart ───────────────────────────────────────────────────────────────
    st.divider()
    if st.button("↺ Start a new enrichment", use_container_width=True, key="restart_btn"):
        reset_processing(clear_autosave=True)
        st.rerun()

# =============================================================================
# STANDALONE COMMERCIAL FIT SCORER
# Score a previously enriched file without running enrichment again.
# =============================================================================

if not ss("processing", False) and _SCORING_AVAILABLE and _show_adv:
    st.divider()
    with st.expander("🎯 Score an existing enrichment file", expanded=False):
        st.caption(
            "Upload a previously enriched Excel or CSV file to apply commercial fit scoring "
            "without re-running the enrichment pipeline.  The file must contain the Step 3 "
            "model-signal columns (`sig_*`, `ti_*`, `has_*`, `is_public`, `has_funding`)."
        )
        _sa_upload = st.file_uploader(
            "Upload enriched file (.xlsx · .xls · .csv)",
            type=["xlsx", "xls", "csv"],
            key="standalone_scorer_upload",
        )
        if _sa_upload is not None:
            try:
                _sa_fname = _sa_upload.name
                _sa_df = (
                    pd.read_csv(_sa_upload)
                    if _sa_fname.lower().endswith(".csv")
                    else pd.read_excel(_sa_upload)
                )
                st.success(
                    f"**{_sa_fname}** loaded — "
                    f"{len(_sa_df):,} rows, {len(_sa_df.columns)} columns"
                )

                # Check for required signal columns
                from commercial_fit_scoring import LEAN_COEFFICIENTS as _sa_lean_coeffs
                _req_sig = [c for c in _sa_lean_coeffs if c in _sa_df.columns]
                if not _req_sig:
                    st.warning(
                        "⚠️ No model-signal columns found (`sig_*`, `ti_*`, `has_*`). "
                        "Run the enrichment pipeline with **Extract model signals (Step 3)** "
                        "enabled first, then upload that output here."
                    )
                else:
                    st.caption(f"Signal columns found: {len(_req_sig)} / {len(_sa_lean_coeffs)}")
                    _sa_df_scored = _score_dataframe(_sa_df.copy())

                    _sa_tier_order  = ["Tier 1", "Tier 2", "Tier 3", "Pass"]
                    _sa_tier_counts = _sa_df_scored["commercial_tier"].value_counts()
                    _sa_tc = st.columns(4)
                    _sa_emojis = ["🟢", "🟡", "🟠", "🔴"]
                    for _si, (_st_tier, _se) in enumerate(zip(_sa_tier_order, _sa_emojis)):
                        _sa_tc[_si].metric(
                            f"{_se} {_st_tier}",
                            int(_sa_tier_counts.get(_st_tier, 0)),
                        )

                    _sa_score_cols = list(_SCORE_OUTPUT_COLS)
                    _sa_id_cols    = [c for c in _sa_df_scored.columns
                                      if c not in set(ALL_ENRICHMENT_FIELDS + _sa_score_cols)][:2]
                    _sa_disp_cols  = _sa_id_cols + _sa_score_cols
                    _sa_disp_df    = (
                        _sa_df_scored[[c for c in _sa_disp_cols if c in _sa_df_scored.columns]]
                        .copy()
                        .sort_values("final_commercial_fit_score", ascending=False)
                    )
                    st.dataframe(_sa_disp_df, use_container_width=True, height=400)

                    _sa_stamp = ts()
                    st.download_button(
                        "⬇ Download scored file (.xlsx)",
                        data=df_to_excel_bytes(_sa_df_scored),
                        file_name=f"scored_{_sa_fname.rsplit('.', 1)[0]}_{_sa_stamp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key="sa_dl_xlsx",
                    )
                    st.download_button(
                        "⬇ Download scored file (.csv)",
                        data=df_to_csv_bytes(_sa_df_scored),
                        file_name=f"scored_{_sa_fname.rsplit('.', 1)[0]}_{_sa_stamp}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="sa_dl_csv",
                    )
            except Exception as _sa_exc:
                st.error(f"Could not process file: {_sa_exc}")
