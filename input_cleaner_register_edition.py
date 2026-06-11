"""
input_cleaner_register_edition.py — Layer 0: mYngle Input Cleaner · Register Edition
======================================================================================
Cleans and enriches Italian Business Register exports before Lead Prioritizer.
Handles missing websites (common in register data), PEC email detection,
multi-website fields, and location-aware Serper search queries.

Website Discovery Upgrade v2:
- Multi-variant brand name extraction
- 8 Serper query strategies with configurable cap
- Aggressive email-domain usage with Serper confirmation
- Richer scoring (rank, title/snippet signals, .it TLD, email match, location)
- New diagnostic output columns
- Expanded blacklist

Entry point:  streamlit run input_cleaner_register_edition.py
"""

import hashlib
import io
import json
import re
import shutil
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st

try:
    import anthropic as _anthropic_sdk
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _anthropic_sdk = None
    _ANTHROPIC_AVAILABLE = False

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Input Cleaner · Register Edition",
    page_icon="🇮🇹",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# CONSTANTS
# =============================================================================

SERPER_URL      = "https://google.serper.dev/search"
_AUTOSAVE_DIR   = Path("autosave")
_AUTOSAVE_EVERY = 10   # write checkpoint every N processed rows

# Generic / directory / social / database domains to skip (global + Italian-specific)
_GENERIC_DOMAINS: frozenset = frozenset({
    # Social networks
    "linkedin.com", "facebook.com", "twitter.com", "x.com", "instagram.com",
    "youtube.com", "xing.com",
    # Global business directories / data providers
    "bloomberg.com", "crunchbase.com", "zoominfo.com", "dnb.com",
    "glassdoor.com", "indeed.com", "angel.co", "pitchbook.com",
    "opencorporates.com", "companieshouse.gov.uk",
    "rocketreach.co", "signalhire.com", "apollo.io", "hunter.io",
    "trustpilot.com", "yelp.com", "reuters.com", "ft.com",
    "github.com", "amazon.com", "app.lusha.com", "wikipedia.org",
    "google.com", "bing.com", "yahoo.com",
    # Job boards
    "jobrapido.it", "monster.it", "infojobs.it", "jobbydoo.it",
    "lavoro.corriere.it", "subito.it", "kijiji.it",
    # Italian company registers / directories / data sources
    "registroimprese.it", "infocamere.it", "imprese.it",
    "ufficiocamerale.it", "companywall.it", "reportaziende.it",
    "companyreports.it", "atoka.io",
    "paginegialle.it", "paginebianche.it",
    "europages.it", "europages.com",
    "kompass.com", "kompass.it",
    "cerved.com", "cervedgroup.it",
    "aziende.it", "icecat.it", "businessit.it",
    "italianmade.com", "viesus.com",
    "madeintaly.com", "italyexport.com",
    "nixonpowerseo.it", "dnbItaly.com",
    # News aggregators, price comparison, marketplaces
    "corriere.it", "repubblica.it", "ilsole24ore.com", "sole24ore.com",
    "trovaprezzi.it", "idealo.it", "amazon.it",
})

# Subdomain / path prefix check — any domain that contains these base domains is generic too
_GENERIC_DOMAIN_BASES: tuple = (
    "linkedin.com", "facebook.com", "twitter.com", "x.com", "instagram.com",
    "registroimprese.it", "infocamere.it", "atoka.io", "kompass.com", "kompass.it",
    "europages.com", "europages.it", "paginegialle.it", "paginebianche.it",
    "cerved.com", "cervedgroup.it", "dnb.com", "zoominfo.com",
    "bloomberg.com", "crunchbase.com", "glassdoor.com", "indeed.com",
)

# PEC (Posta Elettronica Certificata) domains — never use as company website
_PEC_DOMAIN_PATTERNS: tuple = (
    "pec.it", "pec.com", "pec.eu", "legalmail.it", "legalmail.com",
    "postecert.it", "arubapec.it", "arubapec.eu", "pecactually.it",
    "cert.it", "pecimprese.it", "certificata.it", "pecaziende.it",
    "pecmail.it", "pec.tiscali.it", "pecimprese.com",
    "ordineavvocati", "ordinedottori", "caf", "patronato",
    "libero.it", "yahoo.it", "gmail.com", "hotmail.it",
    "alice.it", "tin.it", "virgilio.it", "live.com", "outlook.com",
    "tiscali.it", "hotmail.com", "icloud.com", "protonmail.com",
)

# Expected column names for an Italian Business Register export
_REG_COL_COMPANY   = "Company Name"
_REG_COL_WEBSITE   = "Website"
_REG_COL_EMAIL     = "Email address"
_REG_COL_CITY      = "City"
_REG_COL_PROVINCE  = "National statistical institute Province"
_REG_COL_POSTCODE  = "Postal Code"
_REG_COL_PHONE     = "Phone number"
_REG_COL_SERIAL    = "Serial Number"
_REG_COL_REGOFFICE = "Registered Office"

# Fallback detection candidates (normalised lowercase)
_NAME_CANDIDATES = (
    "company name", "company_name", "ragione sociale", "denominazione",
    "nome", "company", "name",
)
_WEBSITE_CANDIDATES = (
    "website", "sito web", "sito", "url", "web", "homepage",
    "website_url", "domain",
)
_EMAIL_CANDIDATES = (
    "email address", "email", "e-mail", "posta elettronica",
    "indirizzo email",
)
_CITY_CANDIDATES = (
    "city", "città", "comune", "citta",
)
_PROVINCE_CANDIDATES = (
    "national statistical institute province", "province", "provincia",
    "prov",
)
_POSTCODE_CANDIDATES = (
    "postal code", "cap", "postcode", "zip", "codice postale",
)
_PHONE_CANDIDATES = (
    "phone number", "telefono", "tel", "phone",
)

# Legal suffix patterns (Italian + common European)
_LEGAL_TOKENS = re.compile(
    r"\b(s\.?r\.?l\.?|s\.?p\.?a\.?|s\.?a\.?s?|snc|s\.?n\.?c\.?|"
    r"s\.?a\.?p\.?a?\.?|ltd|limited|b\.?v\.?|n\.?v\.?|gmbh|ag|"
    r"llc|inc|corp|plc|holding|holdings|group|co|company|pty|"
    r"se|pte|bhd|sarl|eurl|scs|cv|impresa|ditta|studio)\b\.?",
    re.IGNORECASE,
)

# Italian descriptor words that are NOT part of the brand name
_ITALIAN_DESCRIPTORS = re.compile(
    r"\b(societ[aà]|societa|aziend[ae]|azienda|impres[ae]|impresa|"
    r"industri[ae]|industria|industriale|commerciale|agricol[ae]|agricola|"
    r"gruppo|gruppi|holding|cooperativ[ae]|cooperativa|manifattur[ae]|"
    r"manifatturiero|costruzioni|costruttori|distribuzione|lavorazione|"
    r"produzione|prodotti|fratelli|f\.lli|flli|figli|eredi|successori|"
    r"succ\.?|consorzio|consorzi|associazione|fondazione|istituto)\b\.?",
    re.IGNORECASE,
)

_NOISE_TOKENS: frozenset = frozenset({
    "the", "and", "for", "global", "international", "services", "solutions",
    "consulting", "management", "technology", "technologies", "systems",
    "software", "digital", "enterprise", "enterprises", "partners",
    "italia", "italy", "italian", "europe", "european",
    "snc", "srl", "spa", "sas", "del", "della", "degli", "dei",
    "di", "da", "in", "con", "su", "per", "tra", "fra",
    "group", "holding", "co", "ltd", "inc", "bv",
})

_TLDS: frozenset = frozenset({
    "com", "net", "org", "it", "eu", "nl", "de", "fr", "be", "uk", "co",
    "io", "biz", "info", "at", "ch", "es", "pl", "cz", "se", "no",
    "dk", "fi", "pt", "hu", "ro", "hr", "gr", "gov", "edu",
})

# Keywords that signal an official/home page in title/snippet
_OFFICIAL_SIGNALS = frozenset({
    "official", "sito ufficiale", "home page", "homepage",
    "benvenuti", "welcome", "chi siamo", "about us",
    "sito web ufficiale", "official website", "official site",
})

# Government domain patterns — always reject as company website
_GOVT_PATTERNS = re.compile(
    r"\.gov\.it$|\.gov\b|agenziaentrate|"
    r"(?:^|\.)comune\.|(?:^|\.)regione\.|(?:^|\.)provincia\.|"
    r"prefettura|questura|tribunale|ministero|"
    r"inps\.it$|inail\.it$|agenziademanio|"
    r"camera\.it$|senato\.it$|governo\.it$|quirinale\.it$|mef\.gov",
    re.IGNORECASE,
)

# Religious institution patterns — reject as company website
_RELIGIOUS_PATTERNS = re.compile(
    r"basilica|diocesi|parrocchia|chiesa(?:cattolica)?|santuario|"
    r"abbazia|convento|vescovado|cattedrale|arcidiocesi|"
    r"seminario|oratorio|vaticano|pontific|caritas|"
    r"cappella|pieve|fraternita|confraternita",
    re.IGNORECASE,
)

# Directory / hours / aggregator patterns not already in _GENERIC_DOMAINS
_DIRECTORY_EXTRA_PATTERNS = re.compile(
    r"oraridiapertura|aperturenegozi|tuttopmi|impresaitalia|"
    r"businessfinder|b2bnetwork|catalogoimprese|trovimprese|"
    r"ioimpresa|businessregister|italiabusiness|infobel",
    re.IGNORECASE,
)

# Academic / university patterns
_ACADEMIC_PATTERNS = re.compile(
    r"\.edu$|\.ac\.[a-z]{2,}$|universit[aà]|polimi|polito|"
    r"unimi|unibo|unitn|luiss|bocconi|sapienza|unipd|unifi|politecnico",
    re.IGNORECASE,
)

# Brand similarity gate thresholds
_MIN_BRAND_SIM_TO_SCORE    = 0.15  # below this → score × 0.10 (near-rejection)
_WEAK_BRAND_SIM_MULTIPLIER = 0.35  # between _MIN and 0.25 → score × this
_WEAK_BRAND_SIM_THRESHOLD  = 0.25
_HIGH_CONF_BRAND_THRESHOLD = 0.60  # brand must reach this for High-confidence rule A

# Row colours (openpyxl ARGB hex)
_ACTION_COLORS = {
    "OK":                   "C6EFCE",
    "LIKELY_OK":            "E2EFDA",
    "REVIEW":               "FFEB9C",
    "SUGGEST_REPLACE":      "FCE4D6",
    "MISSING_DOMAIN_FIXED": "FCE4D6",
    "NO_CONFIDENT_MATCH":   "FFC7CE",
    "MISSING_DOMAIN":       "FFC7CE",
    "EMAIL_DERIVED":        "DDEEFF",
}

# Source labels for domain_source column
SRC_ORIGINAL              = "original_website"
SRC_EMAIL                 = "email_domain"
SRC_SERPER                = "serper_search"
SRC_SERPER_EMAIL          = "serper_confirmed_email_domain"
SRC_NONE                  = ""

# Claude Haiku review mode constants
_DEFAULT_HAIKU_MODEL  = "claude-haiku-4-5-20251001"
_HAIKU_MODE_PYTHON    = "Python only"
_HAIKU_MODE_UNCERTAIN = "Haiku for uncertain rows only"
_HAIKU_MODE_ALL       = "Haiku for all rows"
_HAIKU_MODES          = [_HAIKU_MODE_PYTHON, _HAIKU_MODE_UNCERTAIN, _HAIKU_MODE_ALL]

_HAIKU_SYSTEM_PROMPT = (
    "You are a B2B sales intelligence assistant specialising in Italian companies. "
    "Your task: given a company name and candidate websites from Google search results, "
    "identify the company's official website.\n\n"
    "The search results include two kinds of entries:\n"
    "  [SCORED]   — passed the Python brand-similarity filter; scored and ranked\n"
    "  [FILTERED] — removed by Python heuristics (generic site, low similarity, "
    "government/religious/directory/academic category). You may override a FILTERED "
    "result if you are confident it is the correct official site.\n\n"
    "Return ONLY a JSON object with exactly these fields:\n"
    "  decision   — \"accept\" | \"replace\" | \"reject\" | \"uncertain\"\n"
    "               accept=python suggestion is correct; replace=a different domain is better;\n"
    "               reject=none of the results are the real site; uncertain=cannot tell\n"
    "  domain     — the domain you recommend (empty string for reject/uncertain)\n"
    "  confidence — \"High\" | \"Medium\" | \"Low\"\n"
    "  reason     — brief explanation, max 120 chars\n"
    "  risk_flags — JSON array of strings, e.g. [\"directory_site\",\"name_mismatch\"]\n\n"
    "Do not output any text outside the JSON object. No markdown fences."
)

_HAIKU_USER_TEMPLATE = (
    "Company: {company_name}\n"
    "Location: {city}, {province} (Italy)\n"
    "Email domain: {email_domain}\n"
    "Original website in register: {original_website}\n"
    "Python-suggested domain: {python_domain} (confidence: {python_confidence})\n\n"
    "Top search results:\n{results_block}\n\n"
    "Which is the correct official website for this Italian company? "
    "Reply with JSON only."
)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def _normalize_col_key(col: str) -> str:
    return re.sub(r"[\s_\-]+", " ", str(col).strip().lower())


def normalize_domain(raw: str) -> str:
    """Strip protocol, www, path, query. Return root domain lowercase."""
    if not raw or not isinstance(raw, str):
        return ""
    d = raw.strip().lower()
    d = re.sub(r"^https?://", "", d)
    d = re.sub(r"^www\.", "", d)
    d = d.split("/")[0].split("?")[0].split("#")[0].strip()
    if not d or " " in d or d in ("nan", "none", "n/a", "-", "—"):
        return ""
    if "." not in d:
        return ""
    return d


def split_multi_website(raw: str) -> list[str]:
    """
    Split a website field that may contain multiple URLs separated by
    commas, semicolons, spaces, or double spaces. Return normalised domains.
    """
    if not raw or not isinstance(raw, str):
        return []
    parts = re.split(r"[,;\s]+", raw.strip())
    domains = []
    for p in parts:
        d = normalize_domain(p)
        if d:
            domains.append(d)
    return domains


def best_website_domain(raw: str) -> str:
    """
    Parse a multi-website field and return the best single domain.
    Prefers non-generic, then .it TLD.
    """
    domains = split_multi_website(raw)
    if not domains:
        return ""
    non_generic = [d for d in domains if not is_generic(d)]
    if not non_generic:
        return ""
    it_domains = [d for d in non_generic if d.endswith(".it")]
    return it_domains[0] if it_domains else non_generic[0]


def strip_legal(name: str) -> str:
    cleaned = _LEGAL_TOKENS.sub(" ", name)
    return re.sub(r"\s+", " ", cleaned).strip(" .,/-")


def strip_descriptors(name: str) -> str:
    """Remove legal suffixes AND Italian descriptor words."""
    cleaned = _LEGAL_TOKENS.sub(" ", name)
    cleaned = _ITALIAN_DESCRIPTORS.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip(" .,/-")


def extract_name_variants(name: str) -> dict:
    """
    Build multiple name variants for search query generation.

    Returns dict with keys:
      full         — original name
      no_legal     — name with legal suffix removed
      no_desc      — name with legal suffix + Italian descriptors removed
      brand        — shortest meaningful token(s): the 'real' brand name
    """
    full = name.strip()
    no_legal = strip_legal(full)
    no_desc = strip_descriptors(full)

    # Extract brand: split no_desc into meaningful tokens, pick the longest
    # or the last/most-distinctive ones (often the brand is the last proper noun)
    raw_toks = [
        t for t in re.split(r"[\s\-_/&,]+", no_desc)
        if len(t) >= 2 and t.lower() not in _NOISE_TOKENS
        and not re.match(r"^\d+$", t)
    ]

    if not raw_toks:
        brand = no_desc or no_legal or full
    elif len(raw_toks) == 1:
        brand = raw_toks[0]
    else:
        # If any single token is ≥5 chars and not a noise word, treat it as brand
        # Prefer later tokens (brand name often at end of Italian company names)
        # but also consider longest token
        long_toks = [t for t in raw_toks if len(t) >= 4]
        if long_toks:
            # Heuristic: the last long token is often the most unique brand name
            brand = long_toks[-1]
        else:
            brand = raw_toks[-1]

    return {
        "full":     full,
        "no_legal": no_legal,
        "no_desc":  no_desc,
        "brand":    brand,
    }


def company_tokens(name: str) -> set:
    clean = strip_legal(name)
    clean = re.sub(r"[^\w\s\-]", " ", clean)
    toks = {t.lower() for t in re.split(r"[\s\-_]+", clean) if len(t) >= 2}
    return toks - _NOISE_TOKENS


def domain_tokens(domain: str) -> set:
    if not domain:
        return set()
    parts = domain.split(".")
    while len(parts) > 1 and parts[-1].lower() in _TLDS:
        parts = parts[:-1]
    base = ".".join(parts)
    toks = {t for t in re.split(r"[-.]", base.lower()) if t and len(t) >= 2}
    return toks - _TLDS


def token_overlap(name: str, domain: str) -> float:
    """Overlap between company name tokens and domain tokens."""
    ctok = company_tokens(name)
    dtok = domain_tokens(domain)
    if not ctok or not dtok:
        return 0.0
    overlap: set = ctok & dtok
    for c in ctok:
        for d in dtok:
            if c in d or d in c:
                overlap.add(c)
    return len(overlap) / min(len(ctok), len(dtok))


def brand_overlap(brand: str, domain: str) -> float:
    """
    Direct brand-name / domain overlap.
    Returns 1.0 if brand (lowercased, stripped) appears literally in domain base.
    """
    if not brand or not domain:
        return 0.0
    b = re.sub(r"[^\w]", "", brand.lower())
    # Get domain base (strip TLD)
    parts = domain.split(".")
    while len(parts) > 1 and parts[-1].lower() in _TLDS:
        parts = parts[:-1]
    base = re.sub(r"[^\w]", "", ".".join(parts).lower())
    if not b or not base:
        return 0.0
    if b == base:
        return 1.0
    if b in base or base in b:
        return 0.8
    # Token-level check
    b_toks = set(re.split(r"[-.]", b)) - _TLDS
    base_toks = set(re.split(r"[-.]", base)) - _TLDS
    if b_toks and base_toks:
        hit = b_toks & base_toks
        return len(hit) / min(len(b_toks), len(base_toks))
    return 0.0


def is_generic(domain: str) -> bool:
    """Return True if domain is in the generic/directory blacklist (incl. subdomains)."""
    if not domain:
        return False
    dl = domain.lower()
    if dl in _GENERIC_DOMAINS:
        return True
    # Check subdomain containment for known bad base domains
    for base in _GENERIC_DOMAIN_BASES:
        if dl == base or dl.endswith("." + base):
            return True
    return False


def is_pec_or_personal_email(email_domain: str) -> bool:
    """Return True if the email domain is a PEC provider or personal mailbox."""
    if not email_domain:
        return True
    dl = email_domain.lower()
    return any(dl == pat or dl.endswith("." + pat) for pat in _PEC_DOMAIN_PATTERNS)


def extract_email_domain(email: str) -> str:
    """Extract the domain part of an email address."""
    if not email or "@" not in email:
        return ""
    parts = email.strip().split("@")
    if len(parts) < 2:
        return ""
    domain = parts[-1].strip().lower()
    return domain if "." in domain else ""


def location_in_text(text: str, city: str, province: str) -> bool:
    """Return True if city or province appears in the text snippet/title."""
    t = text.lower()
    if city and len(city) >= 3 and city.lower() in t:
        return True
    if province and len(province) >= 2 and province.lower() in t:
        return True
    return False


def has_official_signal(text: str) -> bool:
    """Return True if text contains official-page keywords."""
    tl = text.lower()
    return any(sig in tl for sig in _OFFICIAL_SIGNALS)


def classify_domain(domain: str, title: str = "", snippet: str = "") -> str | None:
    """
    Return a rejection-category string if the domain belongs to a known
    non-commercial category, or None if the domain looks acceptable.

    Categories: "government" | "religious" | "directory" | "academic" | None

    Checks domain string first; for religious also checks the page title
    because a domain like 'sannicola.it' is ambiguous without title context.
    """
    dl = domain.lower()

    if _GOVT_PATTERNS.search(dl):
        return "government"

    # Religious: domain match OR title match (e.g. "Basilica di San Nicola" in title)
    if _RELIGIOUS_PATTERNS.search(dl) or _RELIGIOUS_PATTERNS.search(title.lower()):
        return "religious"

    if _DIRECTORY_EXTRA_PATTERNS.search(dl):
        return "directory"

    if _ACADEMIC_PATTERNS.search(dl):
        return "academic"

    return None


def _conf_label(conf: float) -> str:
    if conf >= 0.70:
        return "High"
    if conf >= 0.40:
        return "Medium"
    return "Low"


# =============================================================================
# COLUMN DETECTION
# =============================================================================


def detect_columns(df: pd.DataFrame) -> dict:
    """
    Detect register column names. Returns dict of role → actual_col_name.
    Tries exact match first, then normalised-key fallback.
    """
    cols_norm = {_normalize_col_key(c): c for c in df.columns}

    def _find(candidates, exact_first=None):
        if exact_first and exact_first in df.columns:
            return exact_first
        for cand in candidates:
            key = _normalize_col_key(cand)
            if key in cols_norm:
                return cols_norm[key]
        return None

    return {
        "company":   _find(_NAME_CANDIDATES,     _REG_COL_COMPANY),
        "website":   _find(_WEBSITE_CANDIDATES,  _REG_COL_WEBSITE),
        "email":     _find(_EMAIL_CANDIDATES,    _REG_COL_EMAIL),
        "city":      _find(_CITY_CANDIDATES,     _REG_COL_CITY),
        "province":  _find(_PROVINCE_CANDIDATES, _REG_COL_PROVINCE),
        "postcode":  _find(_POSTCODE_CANDIDATES, _REG_COL_POSTCODE),
        "phone":     _find(_PHONE_CANDIDATES,    _REG_COL_PHONE),
    }


# =============================================================================
# SERPER SEARCH
# =============================================================================


def _call_serper(query: str, serper_key: str, timeout: int = 12) -> tuple:
    try:
        resp = requests.post(
            SERPER_URL,
            headers={"X-API-KEY": serper_key, "Content-Type": "application/json"},
            json={"q": query, "gl": "it", "hl": "it", "num": 5},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("organic", []), None
    except requests.Timeout:
        return [], "Serper timeout"
    except Exception as e:
        return [], str(e)


def _extract_domain(url: str) -> str:
    try:
        p = urlparse(url if url.startswith("http") else f"https://{url}")
        host = p.hostname or ""
        return re.sub(r"^www\.", "", host.lower())
    except Exception:
        return ""


def _build_search_queries(
    name_variants: dict,
    city: str,
    province: str,
    postcode: str,
    max_queries: int = 5,
) -> list[str]:
    """
    Build up to max_queries Serper search queries using 8 strategy templates.
    Uses name variants: full clean name + short brand name.
    """
    clean_name = name_variants.get("no_desc") or name_variants.get("no_legal") or name_variants["full"]
    brand      = name_variants.get("brand") or clean_name

    # Use brand only if meaningfully shorter than clean_name
    use_brand_queries = (brand.lower() != clean_name.lower() and len(brand) >= 3)

    queries = []

    # Strategy 1-4: full clean name variants
    queries.append(f'"{clean_name}" official website')
    queries.append(f'"{clean_name}" sito ufficiale')
    queries.append(f'"{clean_name}" company website')
    queries.append(f'"{clean_name}" Italy')

    # Strategy 5-6: location-refined
    if city:
        queries.append(f'"{clean_name}" "{city}" Italy')
    elif province:
        queries.append(f'"{clean_name}" "{province}" Italy')

    if province and city:
        queries.append(f'"{clean_name}" "{province}" Italy')

    # Strategy 7: site:.it search
    queries.append(f'site:.it "{clean_name}"')

    # Strategy 8+: brand name fallback queries
    if use_brand_queries:
        queries.append(f'"{brand}" Italy official website')
        queries.append(f'site:.it "{brand}"')

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)

    return unique[:max_queries]


def _score_candidate(
    domain: str,
    rank: int,
    title: str,
    snippet: str,
    name_variants: dict,
    email_domain: str,
    city: str,
    province: str,
) -> float:
    """
    Score a candidate domain on a 0–3+ scale.
    Higher is better.
    """
    score = 0.0

    # 1. Position weight (rank 0 = 1.0, rank 4 = 0.2)
    position_w = 1.0 / (rank + 1)

    # 2. Name overlap signals (use best across variants)
    full_overlap  = token_overlap(name_variants["full"], domain)
    desc_overlap  = token_overlap(name_variants.get("no_desc", ""), domain)
    brand_ov      = brand_overlap(name_variants.get("brand", ""), domain)

    best_name_overlap = max(full_overlap, desc_overlap, brand_ov)
    score += position_w * (0.5 + best_name_overlap * 1.5)

    # 3. Brand name directly in domain (strong signal)
    if brand_ov >= 0.8:
        score += 0.4

    # Brand similarity gate — penalise domains that have very little to do with
    # the company name. This prevents high-ranking directory pages or unrelated
    # sites from winning purely on search position.
    if best_name_overlap < _MIN_BRAND_SIM_TO_SCORE:
        score *= 0.10   # near-rejection: keeps domain in evidence but won't win
    elif best_name_overlap < _WEAK_BRAND_SIM_THRESHOLD:
        score *= _WEAK_BRAND_SIM_MULTIPLIER

    # 4. Title / snippet contains official-page keywords
    combined_text = (title + " " + snippet).lower()
    if has_official_signal(combined_text):
        score += 0.25

    # 5. Title or snippet contains company / brand name (any variant)
    brand_lower = (name_variants.get("brand") or "").lower()
    if brand_lower and brand_lower in combined_text:
        score += 0.2

    # 6. Location signal
    if location_in_text(combined_text, city, province):
        score += 0.3

    # 7. Email domain match (strong confirmation)
    if email_domain and domain == email_domain:
        score += 0.5

    # 8. .it TLD bonus (official company domain for Italian businesses)
    if domain.endswith(".it"):
        score += 0.15

    return round(score, 4)


def search_official_domain_register(
    company_name: str,
    city: str,
    province: str,
    postcode: str,
    email_domain: str,
    serper_key: str,
    max_queries: int = 5,
) -> tuple[str, float, str, list, str, str, list, dict]:
    """
    Run up to max_queries Serper queries with multi-variant brand scoring.

    Returns:
      (suggested_domain, confidence, reason, evidence_rows,
       query_used, name_variant_used, top_3_domains, rejection_counts)

    rejection_counts: dict with keys directory/government/religious/academic/low_similarity
    """
    name_variants = extract_name_variants(company_name)
    queries = _build_search_queries(name_variants, city, province, postcode, max_queries)

    candidates: dict[str, float] = {}    # domain → best score seen
    domain_variant: dict[str, str] = {}  # domain → which name variant matched best
    evidence: list[dict] = []
    query_used = queries[0] if queries else ""
    rejection_notes: list[str] = []
    rejection_counts: dict[str, int] = {
        "directory": 0, "government": 0, "religious": 0,
        "academic": 0, "low_similarity": 0,
    }

    for query in queries:
        results, err = _call_serper(query, serper_key)
        if err:
            rejection_notes.append(f"Serper error: {err}")
            break
        for rank, item in enumerate(results):
            url     = item.get("link", "")
            title   = item.get("title", "")
            snippet = item.get("snippet", "")
            domain  = _extract_domain(url)

            if not domain or is_generic(domain):
                evidence.append({
                    "query": query, "title": title[:80], "url": url,
                    "snippet": snippet[:200],
                    "domain": domain, "used": False,
                    "skip_reason": "generic/blacklisted", "score": 0,
                })
                rejection_notes.append(f"{domain}: blacklisted")
                if domain:
                    rejection_counts["directory"] += 1
                continue

            # Category check — reject government, religious, directory, academic
            cat = classify_domain(domain, title, snippet)
            if cat:
                evidence.append({
                    "query": query, "title": title[:80], "url": url,
                    "snippet": snippet[:200],
                    "domain": domain, "used": False,
                    "skip_reason": f"category:{cat}", "score": 0,
                    "rejection_category": cat,
                })
                rejection_counts[cat] = rejection_counts.get(cat, 0) + 1
                rejection_notes.append(f"{domain}: rejected ({cat})")
                continue

            score = _score_candidate(
                domain, rank, title, snippet,
                name_variants, email_domain, city, province,
            )

            # Very low score after brand gate — note it but don't include in candidates
            if score < 0.08:
                evidence.append({
                    "query": query, "title": title[:80], "url": url,
                    "snippet": snippet[:200],
                    "domain": domain, "used": False,
                    "skip_reason": f"low_similarity_score({score:.3f})", "score": score,
                })
                rejection_counts["low_similarity"] += 1
                rejection_notes.append(f"{domain}: low similarity ({score:.3f})")
                continue

            if domain not in candidates or score > candidates[domain]:
                candidates[domain] = score
                bov = brand_overlap(name_variants.get("brand", ""), domain)
                dov = token_overlap(name_variants.get("no_desc", ""), domain)
                fov = token_overlap(name_variants["full"], domain)
                if bov >= dov and bov >= fov:
                    domain_variant[domain] = f"brand:{name_variants.get('brand','')}"
                elif dov >= fov:
                    domain_variant[domain] = f"no_desc:{name_variants.get('no_desc','')}"
                else:
                    domain_variant[domain] = f"full:{name_variants['full']}"
            else:
                candidates[domain] = max(candidates[domain], score)

            evidence.append({
                "query": query, "title": title[:120], "url": url,
                "snippet": snippet[:200],
                "domain": domain, "score": round(score, 3),
                "brand_overlap": round(brand_overlap(name_variants.get("brand", ""), domain), 3),
                "full_overlap":  round(token_overlap(name_variants["full"], domain), 3),
                "location_match": location_in_text(title + " " + snippet, city, province),
                "email_match": (domain == email_domain),
                "official_signal": has_official_signal(title + " " + snippet),
                "used": True,
            })
        time.sleep(0.25)

    if not candidates:
        return (
            "", 0.0,
            "No candidate domain found. " + "; ".join(rejection_notes[:4]),
            evidence, query_used, "", [],
            rejection_counts,
        )

    # Sort by score — compare all candidates, pick best
    sorted_cands = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    best, best_score = sorted_cands[0]
    top3 = [d for d, _ in sorted_cands[:3]]

    b_ov         = brand_overlap(name_variants.get("brand", ""), best)
    f_ov         = token_overlap(name_variants["full"], best)
    brand_lower  = (name_variants.get("brand") or "").lower()
    best_variant = domain_variant.get(best, "full")

    # Top evidence entry for supplementary signals
    top_ev = next(
        (e for e in evidence if e.get("domain") == best and e.get("used")), {}
    )
    loc_match = top_ev.get("location_match", False)
    official  = top_ev.get("official_signal", False)
    brand_in_title = bool(brand_lower and brand_lower in top_ev.get("title", "").lower())

    # ── High-confidence rules (must satisfy at least ONE) ───────────────────
    # A: Brand clearly in domain
    rule_A = b_ov >= _HIGH_CONF_BRAND_THRESHOLD
    # B: Domain matches email domain (external corroboration)
    rule_B = bool(email_domain and best == email_domain)
    # C: Brand name appears in the search result title
    rule_C = brand_in_title
    # D: Multiple independent signals agree
    rule_D = (
        best_score >= 1.0
        and sum([loc_match, official, rule_B, rule_A, rule_C]) >= 2
    )

    is_high = rule_A or rule_B or rule_C or rule_D

    # Assign confidence
    if rule_B and best_score >= 0.6:
        conf   = 0.88
        reason = f"Serper confirms email domain '{best}' as top result."
    elif is_high and best_score >= 1.2:
        conf   = 0.85
        reason = "Strong brand match in domain/title + search position."
    elif is_high and best_score >= 0.7:
        conf   = 0.78
        reason = "Brand confirmed + reasonable search position."
    elif is_high:
        conf   = 0.72
        reason = "At least one high-confidence signal (brand in domain/title or email match)."
    elif best_score >= 0.60:
        conf   = 0.52
        reason = "Reasonable position + partial name match, but brand not confirmed in domain or title."
    elif best_score >= 0.35:
        conf   = 0.38
        reason = "Weak brand-domain relationship. Likely needs manual review."
    else:
        conf   = 0.20
        reason = "Very weak match — high false-positive risk."

    extras = []
    if rule_A:
        extras.append(f"brand '{name_variants.get('brand','')}' in domain")
    if rule_B:
        extras.append(f"matches email domain ({best})")
    if rule_C:
        extras.append("brand in search result title")
    if loc_match:
        extras.append("city/province in result")
    if official:
        extras.append("official-page keyword in title/snippet")
    if best.endswith(".it"):
        extras.append(".it domain")
    if extras:
        reason += " — " + "; ".join(extras) + "."

    return best, conf, reason, evidence, query_used, best_variant, top3, rejection_counts


# =============================================================================
# CORE VALIDATION (register-aware)
# =============================================================================


def validate_register_row(
    company_name: str,
    raw_website: str,
    raw_email: str,
    city: str,
    province: str,
    postcode: str,
    serper_key: str | None,
    max_queries: int = 5,
) -> tuple[dict, list]:
    """
    Validate one register row. Returns result fields dict.

    Decision flow:
      1. Parse and clean website → normalized_input_website
      2. If website valid and non-generic → OK / LIKELY_OK
      3. If website missing/invalid → try email domain (aggressively)
      4. If email domain plausible → EMAIL_DERIVED, then try Serper to confirm
      5. If still missing → Serper search with multi-variant queries
      6. If Serper finds confident result → MISSING_DOMAIN_FIXED / SUGGEST_REPLACE
      7. Otherwise → MISSING_DOMAIN / NO_CONFIDENT_MATCH

    New diagnostic columns added v2:
      name_variant_used, candidate_domains_considered,
      best_candidate_score, top_3_candidate_domains,
      rejection_reason_if_missing, website_discovery_method
    """
    name     = str(company_name or "").strip()
    email    = str(raw_email or "").strip()
    city     = str(city or "").strip()
    province = str(province or "").strip()
    postcode = str(postcode or "").strip()

    _all_raw_evidence: list[dict] = []

    email_domain = extract_email_domain(email)
    email_is_pec = is_pec_or_personal_email(email_domain)

    norm_website = best_website_domain(raw_website)
    name_variants = extract_name_variants(name)

    result = {
        # Core output
        "cleaned_company_name":       name,
        "normalized_input_website":   norm_website,
        "email_domain":               email_domain,
        "validated_domain":           norm_website,
        "recommended_domain":         "",
        "domain_source":              SRC_ORIGINAL if norm_website else SRC_NONE,
        "domain_action":              "",
        "domain_confidence":          "",
        "domain_reason":              "",
        "manual_review_needed":       False,
        "search_query_used":          "",
        "serper_top_result_title":    "",
        "serper_top_result_url":      "",
        "serper_top_result_domain":   "",
        # v2 diagnostic columns
        "name_variant_used":            "",
        "candidate_domains_considered": "",
        "best_candidate_score":         "",
        "top_3_candidate_domains":      "",
        "rejection_reason_if_missing":  "",
        "website_discovery_method":     "",
        # v3 rejection counts (accumulated across Serper calls for this row)
        "rejected_directory":           0,
        "rejected_government":          0,
        "rejected_religious":           0,
        "rejected_academic":            0,
        "rejected_low_similarity":      0,
    }

    if not name:
        result.update(
            domain_action="NO_CONFIDENT_MATCH",
            domain_confidence="None",
            domain_reason="Company name is blank.",
            manual_review_needed=True,
            rejection_reason_if_missing="Company name is blank.",
            website_discovery_method="none",
        )
        return result, _all_raw_evidence

    # Helper: run Serper and fill result fields
    def _run_serper(existing_email_domain=""):
        sug, conf, reason, ev, query, variant, top3, rej = search_official_domain_register(
            name, city, province, postcode,
            existing_email_domain or email_domain,
            serper_key, max_queries,
        )
        _fill_serper_top(result, ev, query)
        result["name_variant_used"] = variant
        all_doms = [e.get("domain", "") for e in ev if e.get("domain")]
        result["candidate_domains_considered"] = ", ".join(dict.fromkeys(filter(None, all_doms)))
        result["top_3_candidate_domains"] = ", ".join(top3)
        if sug:
            result["best_candidate_score"] = str(round(conf, 3))
        # Accumulate rejection counts across multiple Serper calls for this row
        for cat, cnt in rej.items():
            key = f"rejected_{cat}"
            result[key] = result.get(key, 0) + cnt
        _all_raw_evidence.extend(ev)
        return sug, conf, reason, ev

    # ── Case 1: website present, non-generic ─────────────────────────────────
    if norm_website and not is_generic(norm_website):
        overlap = token_overlap(name, norm_website)
        b_ov    = brand_overlap(name_variants.get("brand", ""), norm_website)
        best_ov = max(overlap, b_ov)

        if best_ov >= 0.45:
            result.update(
                domain_action="OK",
                domain_confidence="High",
                domain_reason="Website domain matches company name / brand tokens closely.",
                manual_review_needed=False,
                website_discovery_method="original_website_accepted",
            )
            return result, _all_raw_evidence

        if best_ov >= 0.15:
            result.update(
                domain_action="LIKELY_OK",
                domain_confidence="Medium",
                domain_reason="Website present; partial name-domain overlap (group/abbreviation likely).",
                manual_review_needed=False,
                website_discovery_method="original_website_partial_match",
            )
            return result, _all_raw_evidence

        # Low overlap — search to confirm or find a better domain
        if serper_key:
            suggested, conf, reason, ev = _run_serper()
            if suggested and conf >= 0.40 and suggested != norm_website:
                result.update(
                    validated_domain=suggested,
                    recommended_domain=suggested,
                    domain_source=SRC_SERPER,
                    domain_action="SUGGEST_REPLACE",
                    domain_confidence=_conf_label(conf),
                    domain_reason=f"Low name-website overlap ({best_ov:.2f}). {reason}",
                    manual_review_needed=(conf < 0.70),
                    website_discovery_method="serper_replaced_low_overlap_website",
                )
                return result, _all_raw_evidence
            if suggested and suggested == norm_website:
                result.update(
                    domain_action="LIKELY_OK",
                    domain_confidence="Medium",
                    domain_reason=f"Search confirms website despite low token overlap ({best_ov:.2f}).",
                    manual_review_needed=False,
                    website_discovery_method="serper_confirmed_original_website",
                )
                return result, _all_raw_evidence

        result.update(
            domain_action="REVIEW",
            domain_confidence="Low",
            domain_reason=f"Website present but low name-domain overlap ({best_ov:.2f}). Manual check recommended.",
            manual_review_needed=True,
            website_discovery_method="original_website_low_confidence",
        )
        return result, _all_raw_evidence

    # ── Case 2: website is a generic/directory site ──────────────────────────
    if norm_website and is_generic(norm_website):
        result["rejection_reason_if_missing"] = f"Input website '{norm_website}' is a directory/blacklisted domain."
        if serper_key:
            suggested, conf, reason, ev = _run_serper()
            if suggested and conf >= 0.40:
                result.update(
                    validated_domain=suggested,
                    recommended_domain=suggested,
                    domain_source=SRC_SERPER,
                    domain_action="SUGGEST_REPLACE",
                    domain_confidence=_conf_label(conf),
                    domain_reason=f"Register website ({norm_website}) is a directory. {reason}",
                    manual_review_needed=(conf < 0.70),
                    website_discovery_method="serper_found_after_blacklisted_website",
                )
                return result, _all_raw_evidence
        result.update(
            validated_domain="",
            domain_action="REVIEW",
            domain_confidence="Low",
            domain_reason=f"Register website ({norm_website}) is a generic directory site.",
            manual_review_needed=True,
            website_discovery_method="none_website_blacklisted",
        )
        return result, _all_raw_evidence

    # ── Case 3: website missing — try email domain aggressively ──────────────
    # v2: Use email domain with much lower bar; Serper will confirm if needed.
    if email_domain and not email_is_pec and not is_generic(email_domain):
        email_name_overlap = token_overlap(name, email_domain)
        email_brand_overlap = brand_overlap(name_variants.get("brand", ""), email_domain)
        email_best_overlap = max(email_name_overlap, email_brand_overlap)

        # Any non-PEC, non-generic, non-personal email domain is a candidate
        # (v2: we no longer require 0.3 overlap — Serper will validate)
        email_plausible = (email_best_overlap >= 0.15) or (email_best_overlap >= 0.0 and serper_key)

        if email_plausible:
            # Try Serper to confirm or find better
            if serper_key:
                suggested, conf, reason, ev = _run_serper(email_domain)
                if suggested == email_domain:
                    result.update(
                        validated_domain=email_domain,
                        recommended_domain=email_domain,
                        domain_source=SRC_SERPER_EMAIL,
                        domain_action="MISSING_DOMAIN_FIXED",
                        domain_confidence=_conf_label(max(conf, 0.70)),
                        domain_reason=f"Website missing. Serper confirms email domain '{email_domain}': {reason}",
                        manual_review_needed=(conf < 0.70),
                        website_discovery_method="serper_confirmed_email_domain",
                    )
                    return result, _all_raw_evidence
                if suggested and conf >= 0.50:
                    # Serper found something better than the email domain
                    result.update(
                        validated_domain=suggested,
                        recommended_domain=suggested,
                        domain_source=SRC_SERPER,
                        domain_action="MISSING_DOMAIN_FIXED",
                        domain_confidence=_conf_label(conf),
                        domain_reason=f"Website missing. Email domain was proxy; Serper found better: {reason}",
                        manual_review_needed=(conf < 0.55),
                        website_discovery_method="serper_found_overrides_email_domain",
                    )
                    return result, _all_raw_evidence
                if suggested and conf >= 0.30:
                    # Weak Serper hit — fall back to email domain with Medium confidence
                    result.update(
                        validated_domain=email_domain,
                        recommended_domain=email_domain,
                        domain_source=SRC_EMAIL,
                        domain_action="EMAIL_DERIVED",
                        domain_confidence="Medium",
                        domain_reason=(
                            f"Website missing. Email domain '{email_domain}' used "
                            f"(overlap {email_best_overlap:.2f}); Serper inconclusive."
                        ),
                        manual_review_needed=True,
                        website_discovery_method="email_domain_serper_inconclusive",
                    )
                    return result, _all_raw_evidence

            # No Serper or Serper found nothing — use email domain if overlap reasonable
            if email_best_overlap >= 0.15:
                result.update(
                    validated_domain=email_domain,
                    recommended_domain=email_domain,
                    domain_source=SRC_EMAIL,
                    domain_action="EMAIL_DERIVED",
                    domain_confidence="Medium" if email_best_overlap >= 0.30 else "Low",
                    domain_reason=(
                        f"Website missing. Email domain '{email_domain}' used as proxy "
                        f"(overlap {email_best_overlap:.2f}). Verify manually."
                    ),
                    manual_review_needed=True,
                    website_discovery_method="email_domain_proxy",
                )
                return result, _all_raw_evidence

    # ── Case 4: website missing — Serper search (no email signal) ────────────
    if serper_key:
        suggested, conf, reason, ev = _run_serper()
        if suggested and conf >= 0.40:
            result.update(
                validated_domain=suggested,
                recommended_domain=suggested,
                domain_source=SRC_SERPER,
                domain_action="MISSING_DOMAIN_FIXED",
                domain_confidence=_conf_label(conf),
                domain_reason=f"Website missing in register. {reason}",
                manual_review_needed=(conf < 0.70),
                website_discovery_method="serper_search",
            )
        else:
            result.update(
                validated_domain="",
                domain_source=SRC_NONE,
                domain_action="MISSING_DOMAIN",
                domain_confidence="None",
                domain_reason="Website missing and no confident result found in search.",
                manual_review_needed=True,
                rejection_reason_if_missing="No sufficiently confident candidate in Serper results.",
                website_discovery_method="none_serper_failed",
            )
    else:
        # No Serper — email fallback with very low bar as last resort
        if email_domain and not email_is_pec and not is_generic(email_domain):
            result.update(
                validated_domain=email_domain,
                recommended_domain=email_domain,
                domain_source=SRC_EMAIL,
                domain_action="EMAIL_DERIVED",
                domain_confidence="Low",
                domain_reason=(
                    f"Website missing. No Serper key. Email domain '{email_domain}' "
                    "used as best guess — verify manually."
                ),
                manual_review_needed=True,
                website_discovery_method="email_domain_no_serper",
            )
        else:
            result.update(
                validated_domain="",
                domain_source=SRC_NONE,
                domain_action="MISSING_DOMAIN",
                domain_confidence="None",
                domain_reason="Website missing. No Serper key. No usable email domain.",
                manual_review_needed=True,
                rejection_reason_if_missing="No website, no Serper, no usable email domain.",
                website_discovery_method="none",
            )

    return result, _all_raw_evidence


def _fill_serper_top(result: dict, evidence: list, query: str) -> None:
    result["search_query_used"] = query
    used = [e for e in evidence if e.get("used")]
    top  = used or [e for e in evidence if e.get("domain")]
    if top:
        result["serper_top_result_title"]  = str(top[0].get("title", ""))[:120]
        result["serper_top_result_url"]    = top[0].get("url", "")
        result["serper_top_result_domain"] = top[0].get("domain", "")


# =============================================================================
# CLAUDE HAIKU REVIEW LAYER
# =============================================================================


def _build_haiku_results_block(raw_evidence: list[dict]) -> str:
    """
    Format ALL Serper evidence for the Haiku prompt, grouped by query.
    Shows up to 10 results per query: used results first, then filtered/rejected ones
    with their rejection reason so Haiku can override if appropriate.
    """
    if not raw_evidence:
        return "(no search results available)"

    # Group by query, preserving insertion order
    from collections import OrderedDict
    by_query: OrderedDict[str, list[dict]] = OrderedDict()
    for e in raw_evidence:
        q = e.get("query", "(unknown query)")
        by_query.setdefault(q, []).append(e)

    sections: list[str] = []
    for query, items in by_query.items():
        lines: list[str] = [f'Query: "{query}"']
        seen_domains: set[str] = set()
        count = 0
        for e in items:
            if count >= 10:
                break
            domain  = e.get("domain", "") or "(no domain)"
            title   = (e.get("title", "") or "")[:80]
            snippet = (e.get("snippet", "") or "")[:120]
            url     = e.get("url", "")
            used    = e.get("used", False)

            if domain in seen_domains:
                continue
            seen_domains.add(domain)
            count += 1

            if used:
                score   = e.get("score", "?")
                b_ov    = e.get("brand_overlap", "?")
                em      = "Yes" if e.get("email_match") else "No"
                lines.append(
                    f"  {count}. [SCORED] {domain}\n"
                    f"     Title:   {title}\n"
                    f"     Snippet: {snippet}\n"
                    f"     URL:     {url}\n"
                    f"     Score: {score} | Brand overlap: {b_ov} | Email match: {em}"
                )
            else:
                reason = e.get("skip_reason", "filtered")
                score  = e.get("score", "")
                score_str = f" | Score: {score}" if score else ""
                lines.append(
                    f"  {count}. [FILTERED: {reason}] {domain}\n"
                    f"     Title:   {title}\n"
                    f"     Snippet: {snippet}\n"
                    f"     URL:     {url}{score_str}"
                )
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def _haiku_review_domain(
    company_name: str,
    city: str,
    province: str,
    email_domain: str,
    original_website: str,
    python_result: dict,
    raw_evidence: list[dict],
    api_key: str,
    model: str = _DEFAULT_HAIKU_MODEL,
) -> dict:
    """
    Call Claude Haiku to validate the Python-suggested domain.
    Returns a dict with haiku_* fields.
    """
    out = {
        "haiku_used":       True,
        "haiku_decision":   "",
        "haiku_domain":     "",
        "haiku_confidence": "",
        "haiku_reason":     "",
        "haiku_risk_flags": "",
        "haiku_error":      "",
    }

    # Do not call Haiku if there is no Serper evidence to reason about
    if not raw_evidence:
        out["haiku_used"]     = False
        out["haiku_decision"] = "skipped_no_serper_evidence"
        out["haiku_error"]    = "no Serper results available for this row"
        return out

    if not _ANTHROPIC_AVAILABLE or not api_key:
        out["haiku_used"]  = False
        out["haiku_error"] = "anthropic SDK not installed or API key missing"
        return out

    python_domain     = str(python_result.get("validated_domain", "") or "")
    python_confidence = str(python_result.get("domain_confidence", "") or "")
    results_block     = _build_haiku_results_block(raw_evidence)

    user_msg = _HAIKU_USER_TEMPLATE.format(
        company_name=company_name,
        city=city,
        province=province,
        email_domain=email_domain or "(none)",
        original_website=original_website or "(none)",
        python_domain=python_domain or "(none)",
        python_confidence=python_confidence or "(none)",
        results_block=results_block,
    )

    try:
        client = _anthropic_sdk.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=256,
            system=_HAIKU_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw_text = resp.content[0].text.strip()
        # Strip optional markdown fences
        raw_text = re.sub(r"^```[a-z]*\n?", "", raw_text)
        raw_text = re.sub(r"\n?```$", "", raw_text)
        parsed = json.loads(raw_text)
        out["haiku_decision"]   = str(parsed.get("decision", "uncertain"))
        out["haiku_domain"]     = str(parsed.get("domain", ""))
        out["haiku_confidence"] = str(parsed.get("confidence", ""))
        out["haiku_reason"]     = str(parsed.get("reason", ""))[:200]
        flags = parsed.get("risk_flags", [])
        out["haiku_risk_flags"] = ", ".join(flags) if isinstance(flags, list) else str(flags)
    except Exception as exc:
        out["haiku_used"]     = True
        out["haiku_error"]    = str(exc)[:200]
        out["haiku_decision"] = "uncertain"

    return out


def _apply_haiku_decision(
    python_result: dict,
    haiku_result: dict,
    mode: str,
) -> dict:
    """
    Merge Python result and Haiku result into final_* fields.
    Returns dict with final_selected_domain, final_decision_source, final_confidence.
    """
    python_domain = str(python_result.get("validated_domain", "") or "")
    python_conf   = str(python_result.get("domain_confidence", "") or "")

    if mode == _HAIKU_MODE_PYTHON:
        return {
            "final_selected_domain": python_domain,
            "final_decision_source": "python",
            "final_confidence":      python_conf,
        }

    if not haiku_result.get("haiku_used"):
        decision_h = haiku_result.get("haiku_decision", "")
        source = "haiku_skipped" if decision_h == "skipped_no_serper_evidence" else "python"
        return {
            "final_selected_domain": python_domain,
            "final_decision_source": source,
            "final_confidence":      python_conf,
        }

    decision     = haiku_result.get("haiku_decision", "uncertain")
    haiku_domain = str(haiku_result.get("haiku_domain", "") or "")
    haiku_conf   = haiku_result.get("haiku_confidence", "")

    if decision == "accept":
        return {
            "final_selected_domain": python_domain,
            "final_decision_source": "haiku_accept",
            "final_confidence":      haiku_conf or python_conf,
        }
    if decision == "replace" and haiku_domain:
        return {
            "final_selected_domain": haiku_domain,
            "final_decision_source": "haiku_replace",
            "final_confidence":      haiku_conf or "Medium",
        }
    if decision == "reject":
        return {
            "final_selected_domain": "",
            "final_decision_source": "haiku_reject",
            "final_confidence":      "None",
        }
    # uncertain / skipped / error — keep python result
    source = "haiku_skipped" if decision == "skipped_no_serper_evidence" else "haiku_uncertain"
    return {
        "final_selected_domain": python_domain,
        "final_decision_source": source,
        "final_confidence":      python_conf,
    }


# =============================================================================
# AUTOSAVE / RESUME
# =============================================================================

_MODE_SAFE: dict[str, str] = {
    _HAIKU_MODE_PYTHON:    "pythononly",
    _HAIKU_MODE_UNCERTAIN: "haikuuncertain",
    _HAIKU_MODE_ALL:       "haikuall",
}


def _make_run_label(
    mode: str, batch_n: int, max_queries: int, debug: bool,
    ts: str | None = None,
) -> str:
    """Return a human-readable run label: YYYYMMDD_HHMM_mode_Nrows_Qq_debug."""
    if ts is None:
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    mode_safe = _MODE_SAFE.get(mode, "pythononly")
    debug_str = "debug" if debug else "nodebug"
    return f"{ts}_{mode_safe}_{batch_n}rows_{max_queries}q_{debug_str}"


def _make_filename(run_label: str, run_id: str) -> str:
    """Return a readable Excel output filename."""
    return f"register_cleaned_{run_label}_{run_id[:8]}.xlsx"


def _file_hash(data: bytes) -> str:
    """Return a short, stable identifier for a file's byte content."""
    return hashlib.sha1(data).hexdigest()[:16]


def _cp_dir(run_id: str, run_label: str = "") -> Path:
    folder = f"{run_label}_{run_id}" if run_label else run_id
    return _AUTOSAVE_DIR / folder


def _save_checkpoint(
    run_id: str,
    all_results: list[dict],
    all_evidence: list[dict],
    row_idx: int,        # number of rows completed so far
    total_rows: int,
    input_df: pd.DataFrame,
    cols: dict,
    settings: dict,
    run_label: str = "",
) -> None:
    """Persist current progress to autosave/{run_label}_{run_id}/ (or autosave/{run_id}/ if no label)."""
    d = _cp_dir(run_id, run_label)
    d.mkdir(parents=True, exist_ok=True)

    meta = {
        "run_id":     run_id,
        "run_label":  run_label,
        "folder_name": d.name,
        "row_idx":    row_idx,
        "total_rows": total_rows,
        "timestamp":  pd.Timestamp.now().isoformat(timespec="seconds"),
        "cols":       {k: v for k, v in cols.items() if v},
        "settings":   settings,
        "complete":   row_idx >= total_rows,
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    if all_results:
        pd.DataFrame(all_results).to_csv(d / "results.csv", index=False)

    if all_evidence:
        (d / "evidence.json").write_text(
            json.dumps(all_evidence, ensure_ascii=False, default=str), encoding="utf-8"
        )

    # Input snapshot — written once; never overwritten (needed for crash-resume)
    input_path = d / "input.csv"
    if not input_path.exists() and input_df is not None:
        input_df.to_csv(input_path, index=False)


def _load_checkpoint_from_dir(d: Path) -> dict | None:
    """Load checkpoint data from a specific directory."""
    try:
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        results: list[dict] = []
        results_path = d / "results.csv"
        if results_path.exists():
            results = pd.read_csv(results_path, dtype=str).fillna("").to_dict("records")
        evidence: list[dict] = []
        ev_path = d / "evidence.json"
        if ev_path.exists():
            evidence = json.loads(ev_path.read_text(encoding="utf-8"))
        input_df: pd.DataFrame | None = None
        input_path = d / "input.csv"
        if input_path.exists():
            input_df = pd.read_csv(input_path, dtype=str).fillna("")
        return {"meta": meta, "results": results, "evidence": evidence, "input_df": input_df}
    except Exception:
        return None


def _load_checkpoint(run_id: str) -> dict | None:
    """
    Load checkpoint by run_id.  Tries legacy hash-only folder first, then scans all
    checkpoint directories for a meta.json whose run_id matches.
    """
    legacy = _AUTOSAVE_DIR / run_id
    if legacy.is_dir() and (legacy / "meta.json").exists():
        return _load_checkpoint_from_dir(legacy)
    for cp_meta in _list_checkpoints():
        if cp_meta.get("run_id") == run_id:
            folder = _AUTOSAVE_DIR / cp_meta["_folder"]
            if folder.is_dir():
                return _load_checkpoint_from_dir(folder)
    return None


def _list_checkpoints() -> list[dict]:
    """Return all checkpoint meta dicts sorted newest-first.  Adds '_folder' key."""
    if not _AUTOSAVE_DIR.exists():
        return []
    out = []
    for d in _AUTOSAVE_DIR.iterdir():
        if not d.is_dir():
            continue
        mp = d / "meta.json"
        if not mp.exists():
            continue
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
            meta["_folder"] = d.name
            out.append(meta)
        except Exception:
            continue
    return sorted(out, key=lambda m: m.get("timestamp", ""), reverse=True)


def _delete_checkpoint(run_id: str) -> None:
    for cp_meta in _list_checkpoints():
        if cp_meta.get("run_id") == run_id:
            shutil.rmtree(_AUTOSAVE_DIR / cp_meta["_folder"], ignore_errors=True)
            return
    shutil.rmtree(_AUTOSAVE_DIR / run_id, ignore_errors=True)


def _checkpoint_excel_bytes(run_id: str, cols: dict) -> bytes | None:
    """Build and return an Excel file from a checkpoint's saved data."""
    cp = _load_checkpoint(run_id)
    if not cp or not cp["results"] or cp["input_df"] is None:
        return None
    try:
        input_df   = cp["input_df"]
        results    = cp["results"]
        evidence   = cp["evidence"]
        n_done     = len(results)

        result_df  = pd.DataFrame(results)
        partial_in = input_df.iloc[:n_done].copy().reset_index(drop=True)
        result_df  = result_df.reset_index(drop=True)

        enriched   = pd.concat([partial_in, result_df], axis=1)
        enriched   = enriched.loc[:, ~enriched.columns.duplicated()]

        return build_excel(enriched, input_df, evidence, cols,
                           debug_rows=None, debug_mode=False, run_meta=None)
    except Exception:
        return None


# =============================================================================
# DATAFRAME PROCESSOR
# =============================================================================

# Output columns added by this tool (v2 includes diagnostic columns)
_OUTPUT_COLS = [
    "cleaned_company_name",
    "normalized_input_website",
    "email_domain",
    "validated_domain",
    "recommended_domain",
    "domain_source",
    "domain_action",
    "domain_confidence",
    "domain_reason",
    "manual_review_needed",
    "search_query_used",
    "serper_top_result_title",
    "serper_top_result_url",
    "serper_top_result_domain",
    # v2 diagnostic
    "name_variant_used",
    "candidate_domains_considered",
    "best_candidate_score",
    "top_3_candidate_domains",
    "rejection_reason_if_missing",
    "website_discovery_method",
    # v3 rejection counts
    "rejected_directory",
    "rejected_government",
    "rejected_religious",
    "rejected_academic",
    "rejected_low_similarity",
    # v4 Haiku review columns
    "haiku_used",
    "haiku_decision",
    "haiku_domain",
    "haiku_confidence",
    "haiku_reason",
    "haiku_risk_flags",
    "haiku_error",
    "final_selected_domain",
    "final_decision_source",
    "final_confidence",
]


def process_dataframe(
    df: pd.DataFrame,
    cols: dict,
    serper_key: str | None,
    max_queries: int = 5,
    progress_cb=None,
    # Autosave / resume parameters
    run_id: str | None = None,
    resume_from: int = 0,
    prior_results: list[dict] | None = None,
    prior_evidence: list[dict] | None = None,
    settings: dict | None = None,
    run_label: str = "",
    # Claude Haiku review layer
    haiku_mode: str = _HAIKU_MODE_PYTHON,
    haiku_api_key: str | None = None,
    haiku_model: str = _DEFAULT_HAIKU_MODEL,
    haiku_max_rows: int = 0,   # 0 = no limit
    # Debug
    debug_mode: bool = False,
) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """
    Process rows resume_from..len(df)-1, prepending prior_results for already-done rows.
    Saves a checkpoint to disk every _AUTOSAVE_EVERY rows and on completion.

    Returns (enriched_df_for_all_rows, all_evidence_rows, all_debug_rows).
    """
    new_results:  list[dict] = []
    new_evidence: list[dict] = []
    new_debug:    list[dict] = []
    n = len(df)

    company_col  = cols.get("company") or ""
    website_col  = cols.get("website") or ""
    email_col    = cols.get("email") or ""
    city_col     = cols.get("city") or ""
    province_col = cols.get("province") or ""
    postcode_col = cols.get("postcode") or ""

    rows_list = list(df.iterrows())

    for local_i, (_, row) in enumerate(rows_list[resume_from:]):
        global_i = resume_from + local_i

        def _sv(col, _row=row):
            return str(_row.get(col, "") or "").strip() if col else ""

        name     = _sv(company_col)
        website  = _sv(website_col)
        email    = _sv(email_col)
        city     = _sv(city_col)
        province = _sv(province_col)
        postcode = _sv(postcode_col)

        res, raw_ev = validate_register_row(
            name, website, email, city, province, postcode, serper_key, max_queries
        )

        # Default Haiku fields (Python-only values)
        res.update({
            "haiku_used": False, "haiku_decision": "", "haiku_domain": "",
            "haiku_confidence": "", "haiku_reason": "", "haiku_risk_flags": "",
            "haiku_error": "",
        })

        # Determine if Haiku should run for this row
        _haiku_rows_done = global_i - resume_from + 1
        _haiku_limit_ok  = (haiku_max_rows <= 0 or _haiku_rows_done <= haiku_max_rows)
        _is_uncertain    = (
            str(res.get("manual_review_needed", "")).lower() in ("true", "1", "yes")
            or str(res.get("domain_confidence", "")).lower() in ("low", "medium", "none", "")
        )
        _run_haiku = (
            haiku_mode != _HAIKU_MODE_PYTHON
            and haiku_api_key
            and _haiku_limit_ok
            and (haiku_mode == _HAIKU_MODE_ALL or _is_uncertain)
        )

        if _run_haiku:
            email_domain_h = str(res.get("email_domain", "") or "")
            orig_website_h = str(res.get("normalized_input_website", "") or "")
            haiku_res = _haiku_review_domain(
                name, city, province, email_domain_h, orig_website_h,
                res, raw_ev, haiku_api_key, haiku_model,
            )
            res.update(haiku_res)
        else:
            haiku_res = {"haiku_used": False}

        final_fields = _apply_haiku_decision(res, haiku_res, haiku_mode)
        res.update(final_fields)

        new_results.append(res)

        query = res.get("search_query_used", "")
        if query:
            new_evidence.append({
                "company_name":      name,
                "city":              city,
                "province":          province,
                "search_query_used": query,
                "serper_top_title":  res.get("serper_top_result_title", ""),
                "serper_top_url":    res.get("serper_top_result_url", ""),
                "serper_top_domain": res.get("serper_top_result_domain", ""),
                "validated_domain":  res.get("validated_domain", ""),
                "domain_source":     res.get("domain_source", ""),
                "domain_action":     res.get("domain_action", ""),
                "domain_confidence": res.get("domain_confidence", ""),
                "name_variant_used": res.get("name_variant_used", ""),
                "top_3_candidates":  res.get("top_3_candidate_domains", ""),
                "rejection_reason":  res.get("rejection_reason_if_missing", ""),
                "discovery_method":  res.get("website_discovery_method", ""),
            })

        # Candidate Discovery Debug rows — one row per Serper result
        if debug_mode and raw_ev:
            query_counters: dict[str, int] = {}
            for e in raw_ev:
                q = e.get("query", "")
                query_counters[q] = query_counters.get(q, 0) + 1
                new_debug.append({
                    "company_name":        name,
                    "row_number":          global_i + 1,
                    "search_query":        q,
                    "result_rank":         query_counters[q],
                    "title":               e.get("title", ""),
                    "snippet":             e.get("snippet", ""),
                    "url":                 e.get("url", ""),
                    "extracted_domain":    e.get("domain", ""),
                    "score":               e.get("score", ""),
                    "used":                e.get("used", ""),
                    "skip_reason":         e.get("skip_reason", ""),
                    "rejection_category":  e.get("rejection_category", ""),
                    "brand_overlap":       e.get("brand_overlap", ""),
                    "full_overlap":        e.get("full_overlap", ""),
                    "location_match":      e.get("location_match", ""),
                    "email_match":         e.get("email_match", ""),
                    "official_signal":     e.get("official_signal", ""),
                    "final_python_domain": res.get("validated_domain", ""),
                    "haiku_mode":          haiku_mode,
                    "haiku_decision":      res.get("haiku_decision", ""),
                    "final_selected_domain": res.get("final_selected_domain", ""),
                })

        rows_done = global_i + 1
        # Checkpoint every N rows and on the final row
        if run_id and (rows_done % _AUTOSAVE_EVERY == 0 or rows_done == n):
            all_r = list(prior_results or []) + new_results
            all_e = list(prior_evidence or []) + new_evidence
            _save_checkpoint(
                run_id, all_r, all_e, rows_done, n,
                df, cols, settings or {}, run_label=run_label,
            )

        if progress_cb:
            progress_cb(rows_done, n)

    # Merge prior completed rows with newly processed rows
    all_results  = list(prior_results or []) + new_results
    all_evidence = list(prior_evidence or []) + new_evidence
    all_debug    = new_debug   # debug rows only cover the newly processed rows

    result_df = pd.DataFrame(all_results, index=df.index)
    enriched  = pd.concat([df.copy(), result_df], axis=1)
    enriched  = enriched.loc[:, ~enriched.columns.duplicated()]
    return enriched, all_evidence, all_debug


# =============================================================================
# EXCEL BUILDER
# =============================================================================


def _header_style():
    from openpyxl.styles import PatternFill, Font
    fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
    font = Font(bold=True, color="FFFFFF", size=10)
    return fill, font


def _action_fill(action: str):
    from openpyxl.styles import PatternFill
    hex_color = _ACTION_COLORS.get(str(action), "FFFFFF")
    return PatternFill(start_color=hex_color, end_color=hex_color, fill_type="solid")


def _write_sheet(ws, df: pd.DataFrame) -> None:
    from openpyxl.styles import Alignment
    from openpyxl.utils import get_column_letter

    hdr_fill, hdr_font = _header_style()
    cols = list(df.columns)

    for ci, col in enumerate(cols, 1):
        cell = ws.cell(row=1, column=ci, value=col)
        cell.fill = hdr_fill
        cell.font = hdr_font
        cell.alignment = Alignment(horizontal="left", vertical="center")
        ws.column_dimensions[get_column_letter(ci)].width = min(
            max(len(str(col)) + 2, 12), 52
        )

    action_idx   = (cols.index("domain_action") + 1)   if "domain_action"   in cols else None
    val_dom_idx  = (cols.index("validated_domain") + 1) if "validated_domain" in cols else None
    norm_web_idx = (cols.index("normalized_input_website") + 1) if "normalized_input_website" in cols else None

    for ri, (_, row) in enumerate(df.iterrows(), 2):
        action = str(row.get("domain_action", "") or "")
        fill   = _action_fill(action)
        for ci, col in enumerate(cols, 1):
            val = row[col]
            if isinstance(val, float) and val != val:
                val = ""
            cell = ws.cell(row=ri, column=ci, value=val)
            cell.fill = fill
            cell.alignment = Alignment(wrap_text=False, vertical="top")

        if val_dom_idx and norm_web_idx:
            v = str(row.get("validated_domain", "") or "")
            n = str(row.get("normalized_input_website", "") or "")
            if v and v != n:
                from openpyxl.styles import Font
                ws.cell(row=ri, column=val_dom_idx).font = Font(bold=True, color="C00000")

    ws.freeze_panes = "A2"
    if len(df) > 0:
        ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}1"
    ws.row_dimensions[1].height = 18


def _build_best_guess_df(
    enriched_df: pd.DataFrame,
    cols: dict,
) -> pd.DataFrame:
    """
    Best Guess Input — contains all fields useful for Lead Prioritizer:
    company_name, website_url (best guess), email, city, province, phone,
    plus key diagnostic columns.
    """
    company_col  = cols.get("company") or ""
    email_col    = cols.get("email") or ""
    city_col     = cols.get("city") or ""
    province_col = cols.get("province") or ""
    phone_col    = cols.get("phone") or ""

    rows = []
    for _, r in enriched_df.iterrows():
        def _sv(col):
            return str(r.get(col, "") or "").strip() if col else ""

        action = str(r.get("domain_action", "") or "")
        norm   = str(r.get("normalized_input_website", "") or "").strip()
        recom  = str(r.get("recommended_domain", "") or "").strip()
        final  = str(r.get("final_selected_domain", "") or "").strip()

        # If Haiku produced a final domain decision, use it; otherwise fall back to Python logic
        if final:
            url = final
        elif action in ("OK", "LIKELY_OK"):
            url = norm
        elif action in ("SUGGEST_REPLACE", "MISSING_DOMAIN_FIXED", "EMAIL_DERIVED"):
            url = recom or norm
        elif action == "REVIEW":
            url = norm or recom
        else:
            url = norm

        rows.append({
            "company_name":           _sv(company_col),
            "website_url":            url,
            "email":                  _sv(email_col),
            "city":                   _sv(city_col),
            "province":               _sv(province_col),
            "phone":                  _sv(phone_col),
            "domain_action":          action,
            "domain_confidence":      str(r.get("domain_confidence", "") or ""),
            "domain_source":          str(r.get("domain_source", "") or ""),
            "website_discovery_method": str(r.get("website_discovery_method", "") or ""),
            "manual_review_needed":   r.get("manual_review_needed", False),
        })
    return pd.DataFrame(rows)


def _write_best_guess_sheet(ws, bg_df: pd.DataFrame, enriched_df: pd.DataFrame) -> None:
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    hdr_fill, hdr_font = _header_style()
    cols = list(bg_df.columns)

    for ci, col in enumerate(cols, 1):
        cell = ws.cell(row=1, column=ci, value=col)
        cell.fill = hdr_fill
        cell.font = hdr_font
        cell.alignment = Alignment(horizontal="left", vertical="center")
        ws.column_dimensions[get_column_letter(ci)].width = min(
            max(len(str(col)) + 2, 16), 48
        )

    changed_fill = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")
    blank_fill   = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    ok_fill      = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
    email_fill   = PatternFill(start_color="DDEEFF", end_color="DDEEFF", fill_type="solid")

    orig_norms = enriched_df.get(
        "normalized_input_website",
        pd.Series("", index=enriched_df.index)
    )

    url_col_idx = (cols.index("website_url") + 1) if "website_url" in cols else 2

    for ri, ((_, bg_row), orig_norm) in enumerate(
        zip(bg_df.iterrows(), orig_norms), 2
    ):
        action = str(bg_row.get("domain_action", "") or "")
        url    = str(bg_row.get("website_url", "") or "").strip()
        orig   = str(orig_norm or "").strip()

        for ci, col in enumerate(cols, 1):
            val = bg_row[col]
            if isinstance(val, float) and val != val:
                val = ""
            cell = ws.cell(row=ri, column=ci, value=val)
            cell.alignment = Alignment(vertical="top")

        url_cell = ws.cell(row=ri, column=url_col_idx)
        src = str(bg_row.get("domain_source", "") or "")
        if not url:
            url_cell.fill = blank_fill
        elif src in (SRC_EMAIL, SRC_SERPER_EMAIL) or action == "EMAIL_DERIVED":
            url_cell.fill = email_fill
            url_cell.font = Font(italic=True, color="1F497D")
        elif url != orig:
            url_cell.fill = changed_fill
            url_cell.font = Font(bold=True, color="C00000")
        else:
            url_cell.fill = ok_fill

    ws.freeze_panes = "A2"
    if len(bg_df) > 0:
        ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}1"
    ws.row_dimensions[1].height = 18


def _write_run_summary_sheet(ws, run_meta: dict) -> None:
    """Write a two-column (Field / Value) run summary sheet."""
    from openpyxl.styles import Alignment, Font, PatternFill
    hdr_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
    hdr_font = Font(bold=True, color="FFFFFF", size=10)
    key_font  = Font(bold=True, size=10)
    for ci, header in enumerate(["Field", "Value"], 1):
        cell = ws.cell(row=1, column=ci, value=header)
        cell.fill = hdr_fill
        cell.font = hdr_font
        cell.alignment = Alignment(horizontal="left")
    ws.column_dimensions["A"].width = 36
    ws.column_dimensions["B"].width = 52
    for ri, (field, value) in enumerate(run_meta.items(), 2):
        a = ws.cell(row=ri, column=1, value=str(field))
        a.font = key_font
        a.alignment = Alignment(vertical="top")
        b = ws.cell(row=ri, column=2, value=str(value) if value is not None else "")
        b.alignment = Alignment(vertical="top", wrap_text=False)
    ws.freeze_panes = "A2"
    ws.row_dimensions[1].height = 18


def build_excel(
    enriched_df: pd.DataFrame,
    original_df: pd.DataFrame,
    evidence_rows: list[dict],
    cols: dict,
    debug_rows: list[dict] | None = None,
    debug_mode: bool = False,
    run_meta: dict | None = None,
) -> bytes:
    import openpyxl

    wb = openpyxl.Workbook()

    # Sheet 1: Best Guess Input
    ws0 = wb.active
    ws0.title = "Best Guess Input"
    bg_df = _build_best_guess_df(enriched_df, cols)
    _write_best_guess_sheet(ws0, bg_df, enriched_df)

    # Sheet 2: Cleaned Register Input
    ws1 = wb.create_sheet("Cleaned Register Input")
    _write_sheet(ws1, enriched_df)

    # Sheet 3: Review Needed
    ws2 = wb.create_sheet("Review Needed")
    review_mask = (
        enriched_df.get("manual_review_needed", pd.Series(False, index=enriched_df.index))
        .astype(str).str.lower().isin(["true", "1", "yes"])
    )
    review_df = enriched_df[review_mask] if review_mask.any() else enriched_df.iloc[:0]
    _write_sheet(ws2, review_df.copy())

    # Sheet 4: Original Input
    ws3 = wb.create_sheet("Original Input")
    _write_sheet(ws3, original_df)

    # Sheet 5: Raw Search Evidence
    ws4 = wb.create_sheet("Raw Search Evidence")
    ev_df = pd.DataFrame(evidence_rows) if evidence_rows else pd.DataFrame(columns=[
        "company_name", "city", "province", "search_query_used",
        "serper_top_title", "serper_top_url", "serper_top_domain",
        "validated_domain", "domain_source", "domain_action", "domain_confidence",
        "name_variant_used", "top_3_candidates", "rejection_reason", "discovery_method",
    ])
    _write_sheet(ws4, ev_df)

    # Sheet 6: Python vs Haiku Comparison (only shown when Haiku was run)
    haiku_cols = [
        cols.get("company"),
        "validated_domain", "domain_confidence", "domain_action",
        "haiku_used", "haiku_decision", "haiku_domain", "haiku_confidence",
        "haiku_reason", "haiku_risk_flags", "haiku_error",
        "final_selected_domain", "final_decision_source", "final_confidence",
    ]
    avail_haiku_cols = [c for c in haiku_cols if c and c in enriched_df.columns]
    haiku_ran = (
        "haiku_used" in enriched_df.columns
        and enriched_df["haiku_used"].astype(str).str.lower().isin(["true", "1"]).any()
    )
    ws5 = wb.create_sheet("Python vs Haiku Comparison")
    if haiku_ran and avail_haiku_cols:
        _write_sheet(ws5, enriched_df[avail_haiku_cols].copy())
    else:
        ws5.cell(row=1, column=1, value="Haiku review was not used in this run.")

    # Sheet 7: Candidate Discovery Debug (only when debug_mode is active)
    if debug_mode:
        ws6 = wb.create_sheet("Candidate Discovery Debug")
        _debug_cols = [
            "company_name", "row_number", "search_query", "result_rank",
            "title", "snippet", "url", "extracted_domain",
            "score", "used", "skip_reason", "rejection_category",
            "brand_overlap", "full_overlap", "location_match", "email_match",
            "official_signal", "final_python_domain",
            "haiku_mode", "haiku_decision", "final_selected_domain",
        ]
        if debug_rows:
            debug_df = pd.DataFrame(debug_rows)
            for c in _debug_cols:
                if c not in debug_df.columns:
                    debug_df[c] = ""
            _write_sheet(ws6, debug_df[_debug_cols])
        else:
            ws6.cell(row=1, column=1,
                     value="Debug mode was enabled but no Serper results were collected "
                           "(no Serper key, or no rows required search).")

    # Sheet 8 (or 7): Run Summary — always last
    ws_summary = wb.create_sheet("Run Summary")
    if run_meta:
        _write_run_summary_sheet(ws_summary, run_meta)
    else:
        ws_summary.cell(row=1, column=1, value="Run metadata not available.")

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# =============================================================================
# SUMMARY METRICS
# =============================================================================


def _summary_metrics(df: pd.DataFrame, cols: dict) -> None:
    actions  = df.get("domain_action",  pd.Series(dtype=str)).astype(str)
    sources  = df.get("domain_source",  pd.Series(dtype=str)).astype(str)
    confs    = df.get("domain_confidence", pd.Series(dtype=str)).astype(str)

    total     = len(df)
    has_web   = int(df.get(cols.get("website") or "_", pd.Series("")).astype(str)
                    .str.strip().replace("", pd.NA).notna().sum()) if cols.get("website") else 0
    has_email = int(df.get(cols.get("email") or "_", pd.Series("")).astype(str)
                    .str.strip().replace("", pd.NA).notna().sum()) if cols.get("email") else 0
    has_phone = int(df.get(cols.get("phone") or "_", pd.Series("")).astype(str)
                    .str.strip().replace("", pd.NA).notna().sum()) if cols.get("phone") else 0

    accepted        = int(actions.isin(["OK", "LIKELY_OK"]).sum())
    from_email      = int(sources.isin([SRC_EMAIL]).sum())
    serper_conf_email = int(sources.isin([SRC_SERPER_EMAIL]).sum())
    from_serper     = int(sources.isin([SRC_SERPER]).sum())
    no_match        = int(actions.isin(["MISSING_DOMAIN", "NO_CONFIDENT_MATCH"]).sum())

    # website found = any non-empty validated_domain
    has_domain_after = int(
        df.get("validated_domain", pd.Series(dtype=str)).astype(str)
        .str.strip().replace("", pd.NA).notna().sum()
    )
    review = int(
        df.get("manual_review_needed", pd.Series(dtype=str))
        .astype(str).str.lower().isin(["true", "1", "yes"]).sum()
    )
    coverage_pct = round(has_domain_after / total * 100) if total else 0

    def card(col, label, val, color, hint=""):
        col.markdown(
            f"<div style='border-left:4px solid {color};padding:8px 12px;"
            f"background:#f7f9fc;border-radius:4px;margin-bottom:6px'>"
            f"<div style='font-size:1.45em;font-weight:700;color:{color}'>{val}</div>"
            f"<div style='font-size:0.78em;color:#555'>{label}</div>"
            + (f"<div style='font-size:0.72em;color:#888'>{hint}</div>" if hint else "")
            + "</div>",
            unsafe_allow_html=True,
        )

    # Row 1 — input data
    row1 = st.columns(4)
    card(row1[0], "Total companies",       total,      "#0B4A92")
    card(row1[1], "With original website", has_web,    "#2E7D32",
         f"{round(has_web/total*100) if total else 0}% of rows")
    card(row1[2], "With email",            has_email,  "#1565C0")
    card(row1[3], "With phone",            has_phone,  "#37474F")

    st.markdown("")
    # Row 2 — discovery outcome
    row2 = st.columns(4)
    card(row2[0], "Website found after cleaning", has_domain_after, "#2E7D32",
         f"{coverage_pct}% coverage")
    card(row2[1], "Original website accepted",    accepted,          "#43A047",
         "OK + LIKELY_OK")
    card(row2[2], "Email domain used",            from_email + serper_conf_email, "#1565C0",
         f"{from_email} proxy · {serper_conf_email} Serper-confirmed")
    card(row2[3], "Found by Serper search",       from_serper,       "#E65100")

    st.markdown("")
    # Row 3 — review / gaps
    row3 = st.columns(4)
    card(row3[0], "Need manual review",    review,    "#B71C1C")
    card(row3[1], "No confident match",    no_match,  "#C62828",
         "MISSING_DOMAIN or NO_CONFIDENT_MATCH")
    card(row3[2], "Serper-confirmed email domain", serper_conf_email, "#6A1B9A",
         "strongest email signal")
    card(row3[3], "High confidence rows",
         int(confs.str.lower().eq("high").sum()),
         "#2E7D32",
         "manual_review_needed = False")

    # Row 3b — Haiku stats (only shown when Haiku was used)
    haiku_used_col = df.get("haiku_used", pd.Series(dtype=str)).astype(str).str.lower()
    n_haiku = int(haiku_used_col.isin(["true", "1"]).sum())
    if n_haiku > 0:
        st.markdown("")
        row3b = st.columns(4)
        decisions = df.get("haiku_decision", pd.Series(dtype=str)).astype(str)
        sources   = df.get("final_decision_source", pd.Series(dtype=str)).astype(str)
        card(row3b[0], "Rows reviewed by Haiku",      n_haiku,                             "#7B1FA2")
        card(row3b[1], "Haiku: accepted Python",       int(decisions.eq("accept").sum()),  "#2E7D32",
             "agreed with Python scoring")
        card(row3b[2], "Haiku: replaced domain",       int(decisions.eq("replace").sum()), "#E65100",
             "found better domain than Python")
        card(row3b[3], "Haiku: rejected / uncertain",
             int(decisions.isin(["reject", "uncertain"]).sum()),                            "#B71C1C",
             "no confident match")

    st.markdown("")
    # Row 4 — false-positive rejections (sum across all rows)
    def _sum_col(col_name):
        col = df.get(col_name, pd.Series(0, index=df.index))
        return int(pd.to_numeric(col, errors="coerce").fillna(0).sum())

    row4 = st.columns(5)
    card(row4[0], "Rejected: directory",      _sum_col("rejected_directory"),      "#5D4037",
         "oraridiapertura, pagine gialle, etc.")
    card(row4[1], "Rejected: government",     _sum_col("rejected_government"),     "#37474F",
         ".gov.it, agenziaentrate, comune, etc.")
    card(row4[2], "Rejected: religious",      _sum_col("rejected_religious"),      "#6A1B9A",
         "basilica, diocesi, parrocchia, etc.")
    card(row4[3], "Rejected: academic",       _sum_col("rejected_academic"),       "#1565C0",
         ".edu, università, politecnico, etc.")
    card(row4[4], "Rejected: low similarity", _sum_col("rejected_low_similarity"), "#E65100",
         "domain unrelated to company name")


# =============================================================================
# STREAMLIT UI
# =============================================================================


def _build_run_meta(
    enriched_df: pd.DataFrame,
    input_filename: str,
    run_id: str,
    run_label: str,
    total_rows_input: int,
    batch_n: int,
    max_queries: int,
    haiku_mode: str,
    haiku_model: str,
    haiku_max_rows: int,
    debug_mode: bool,
    serper_key_present: bool,
    anthropic_key_present: bool,
) -> dict:
    """Build the ordered dict that populates the Run Summary Excel sheet."""
    actions  = enriched_df.get("domain_action",  pd.Series(dtype=str)).astype(str)
    has_final = int(
        enriched_df.get("final_selected_domain", enriched_df.get("validated_domain",
            pd.Series(dtype=str))).astype(str).str.strip().replace("", pd.NA).notna().sum()
    )
    coverage = round(has_final / batch_n * 100) if batch_n else 0
    decisions = enriched_df.get("haiku_decision", pd.Series(dtype=str)).astype(str)
    n_haiku   = int(
        enriched_df.get("haiku_used", pd.Series(dtype=str)).astype(str)
        .str.lower().isin(["true", "1"]).sum()
    )
    review = int(
        enriched_df.get("manual_review_needed", pd.Series(dtype=str))
        .astype(str).str.lower().isin(["true", "1", "yes"]).sum()
    )
    no_match = int(actions.isin(["MISSING_DOMAIN", "NO_CONFIDENT_MATCH"]).sum())
    return {
        "timestamp":                 pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_filename":            input_filename,
        "input_file_hash":           run_id,
        "run_label":                 run_label,
        "total_rows_in_input":       total_rows_input,
        "processed_rows":            batch_n,
        "max_serper_queries":        max_queries,
        "haiku_mode":                haiku_mode,
        "haiku_model":               haiku_model,
        "haiku_max_rows":            haiku_max_rows if haiku_max_rows > 0 else "all",
        "candidate_debug_mode":      "on" if debug_mode else "off",
        "serper_key_present":        "yes" if serper_key_present else "no",
        "anthropic_key_present":     "yes" if anthropic_key_present else "no",
        "rows_with_final_domain":    has_final,
        "final_website_coverage_%":  f"{coverage}%",
        "rows_reviewed_by_haiku":    n_haiku,
        "haiku_accepted_python":     int(decisions.eq("accept").sum()),
        "haiku_replaced_python":     int(decisions.eq("replace").sum()),
        "haiku_rejected":            int(decisions.eq("reject").sum()),
        "haiku_uncertain":           int(decisions.eq("uncertain").sum()),
        "no_confident_match_count":  no_match,
        "manual_review_count":       review,
    }


def _show_run_info_block(
    run_label: str,
    run_id: str,
    batch_n,
    max_queries: int,
    haiku_mode: str,
    debug_mode: bool,
    filename: str,
) -> None:
    """Show a compact post-run settings summary in the main area."""
    mode_safe = _MODE_SAFE.get(haiku_mode, "pythononly")
    debug_str = "on" if debug_mode else "off"
    st.markdown(
        f"<div style='background:#f0f4f8;border-radius:6px;padding:10px 16px;"
        f"font-size:0.82em;margin:8px 0;line-height:1.7'>"
        f"<b>Run summary</b> · "
        f"Mode: <code>{mode_safe}</code> · "
        f"Rows: <code>{batch_n}</code> · "
        f"Queries: <code>{max_queries}</code> · "
        f"Debug: <code>{debug_str}</code><br>"
        f"File: <code>{filename}</code> · "
        f"Autosave: <code>{_AUTOSAVE_DIR}/{run_label}_{run_id[:8] if run_label else run_id}</code>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _load_secrets_key() -> str | None:
    try:
        return st.secrets.get("SERPER_API_KEY") or st.secrets.get("serper_api_key")
    except Exception:
        return None


def _parse_bytes(raw: bytes, filename: str) -> pd.DataFrame | None:
    """Parse CSV or Excel bytes into a DataFrame."""
    try:
        if filename.lower().endswith(".csv"):
            return pd.read_csv(io.BytesIO(raw), dtype=str).fillna("")
        else:
            return pd.read_excel(io.BytesIO(raw), dtype=str).fillna("")
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return None


def _show_results(enriched_df: pd.DataFrame, stored_cols: dict) -> None:
    """Render the results table and review expander."""
    show_cols = [c for c in [
        stored_cols.get("company"),
        stored_cols.get("website"),
        "normalized_input_website",
        "validated_domain",
        "domain_source",
        "domain_action",
        "domain_confidence",
        "website_discovery_method",
        "manual_review_needed",
    ] if c and c in enriched_df.columns]
    st.dataframe(enriched_df[show_cols], use_container_width=True, height=360)

    review_mask = (
        enriched_df.get("manual_review_needed", pd.Series(False))
        .astype(str).str.lower().isin(["true", "1", "yes"])
    )
    if review_mask.any():
        with st.expander(
            f"🔴 Rows needing manual review ({int(review_mask.sum())})", expanded=False
        ):
            st.dataframe(enriched_df[review_mask][show_cols], use_container_width=True)


def _download_section(
    enriched_df, original_df, evidence_rows, stored_cols,
    filename="register_cleaned_output.xlsx",
    debug_rows: list[dict] | None = None,
    debug_mode: bool = False,
    run_meta: dict | None = None,
) -> None:
    """Render the output-sheet legend + download button."""
    excel_bytes = build_excel(
        enriched_df, original_df, evidence_rows, stored_cols,
        debug_rows=debug_rows, debug_mode=debug_mode, run_meta=run_meta,
    )
    sheet_list = (
        "1. **Best Guess Input** — company, website, email, city, province, phone "
        "+ discovery method (ready for Lead Prioritizer)  \n"
        "2. **Cleaned Register Input** — all original columns + validation + diagnostic columns  \n"
        "3. **Review Needed** — rows requiring manual check  \n"
        "4. **Original Input** — unchanged source data  \n"
        "5. **Raw Search Evidence** — Serper queries, results, name variants, rejection reasons  \n"
        "6. **Python vs Haiku Comparison** — side-by-side comparison (populated when Haiku mode is active)"
    )
    if debug_mode:
        sheet_list += (
            "  \n7. **Candidate Discovery Debug** — every Serper result per company with full scoring detail"
        )
    sheet_list += "  \n8. **Run Summary** — run settings and outcome metrics" if debug_mode \
        else "  \n7. **Run Summary** — run settings and outcome metrics"
    st.markdown("**Output sheets:**  \n" + sheet_list)
    st.download_button(
        "⬇ Download cleaned register Excel",
        data=excel_bytes,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        type="primary",
    )


def main():
    st.title("🇮🇹 Input Cleaner · Register Edition")
    st.caption(
        "Layer 0 · mYngle Sales Intelligence · "
        "Cleans Italian Business Register exports before Lead Prioritizer enrichment  \n"
        "Website Discovery v2 — multi-variant brand extraction, 8 query strategies, "
        "category rejection, brand-similarity gate"
    )

    # ── Sidebar: API key + settings ───────────────────────────────────────────
    st.sidebar.header("Settings")

    serper_key = _load_secrets_key()
    if serper_key:
        st.sidebar.success("✓ Serper API key loaded from secrets.")
    else:
        st.sidebar.warning(
            "No Serper API key found in `.streamlit/secrets.toml`.\n\n"
            "Serper domain search will be skipped. "
            "Only website normalisation and email-domain fallback will run."
        )
        manual_key = st.sidebar.text_input(
            "Paste Serper API key (optional)", type="password", key="reg_serper"
        )
        if manual_key.strip():
            serper_key = manual_key.strip()

    # Anthropic API key for Haiku
    anthropic_key = None
    try:
        anthropic_key = st.secrets.get("ANTHROPIC_API_KEY") or st.secrets.get("anthropic_api_key")
    except Exception:
        pass

    st.sidebar.markdown("---")
    max_queries = st.sidebar.selectbox(
        "Max Serper queries per company",
        options=[3, 5, 8],
        index=1,
        help=(
            "3 = fast/cheap · 5 = default, good balance · 8 = maximum discovery.\n\n"
            "Each query costs 1 Serper credit. For 200 companies: "
            "3 queries ≈ up to 600 credits, 5 ≈ up to 1000, 8 ≈ up to 1600."
        ),
    )
    st.sidebar.caption(
        f"Each missing website tries up to {max_queries} search strategies."
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("Claude Haiku Review (Experiment)")
    if not _ANTHROPIC_AVAILABLE:
        st.sidebar.warning(
            "Install `anthropic` package to enable Haiku review:  \n"
            "`pip install anthropic`"
        )
    haiku_mode = st.sidebar.selectbox(
        "Haiku review mode",
        options=_HAIKU_MODES,
        index=0,
        help=(
            "Python only: use existing scoring only (no Haiku calls, no cost).  \n"
            "Uncertain rows: call Haiku only for rows where Python is not confident.  \n"
            "All rows: call Haiku for every row (most accurate, higher cost)."
        ),
    )
    if haiku_mode != _HAIKU_MODE_PYTHON:
        if anthropic_key:
            st.sidebar.success("✓ Anthropic API key loaded from secrets.")
        else:
            manual_anthropic = st.sidebar.text_input(
                "Paste Anthropic API key", type="password", key="reg_anthropic"
            )
            if manual_anthropic.strip():
                anthropic_key = manual_anthropic.strip()
            if not anthropic_key:
                st.sidebar.warning("Haiku review requires an Anthropic API key.")

        haiku_model = st.sidebar.text_input(
            "Haiku model ID",
            value=_DEFAULT_HAIKU_MODEL,
            key="reg_haiku_model",
        )
        haiku_max_rows = st.sidebar.number_input(
            "Max rows to send to Haiku (0 = all)",
            min_value=0, max_value=5000, value=50, step=10,
            key="reg_haiku_max_rows",
            help="Limit Haiku calls to keep costs controlled during testing.",
        )
        _est_calls = haiku_max_rows if haiku_max_rows > 0 else "all"
        st.sidebar.caption(
            f"Estimated Haiku calls: up to **{_est_calls}** rows.  \n"
            "Haiku input/output tokens ≈ 800/100 per row."
        )
    else:
        haiku_model    = _DEFAULT_HAIKU_MODEL
        haiku_max_rows = 0

    st.sidebar.markdown("---")
    debug_mode = st.sidebar.checkbox(
        "Candidate Discovery Debug Mode",
        value=False,
        key="reg_debug_mode",
        help=(
            "When enabled, exports an extra Excel sheet with every Serper result for "
            "every company: title, snippet, url, score, rejection reason, brand overlap, etc. "
            "Useful for diagnosing why certain companies fail. Increases file size."
        ),
    )
    if debug_mode:
        st.sidebar.caption(
            "🔍 Debug sheet will include all raw Serper candidates + rejection reasons."
        )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Autosave every **{_AUTOSAVE_EVERY} rows** → `{_AUTOSAVE_DIR}/`  \n"
        "If the app crashes or your browser refreshes, reopen the app and use "
        "**Resume previous run** to continue without reprocessing completed rows."
    )

    # ── Previous-run panel (shown even without a file uploaded) ──────────────
    checkpoints = _list_checkpoints()
    if checkpoints:
        n_cp = len(checkpoints)
        with st.expander(f"📂 Previous runs available ({n_cp})", expanded=False):
            for cp_meta in checkpoints[:8]:
                run_id_cp  = cp_meta.get("run_id", "?")
                row_idx    = cp_meta.get("row_idx", 0)
                total      = cp_meta.get("total_rows", "?")
                ts         = str(cp_meta.get("timestamp", "?"))[:19]
                complete   = cp_meta.get("complete", False)
                pct        = f"{row_idx / total * 100:.0f}%" if isinstance(total, int) and total else "?"
                run_label_cp = cp_meta.get("run_label", "") or cp_meta.get("_folder", run_id_cp[:8])
                label_str  = (
                    f"{'✅ Complete' if complete else '⏸ Partial'} · "
                    f"**{row_idx}/{total}** rows ({pct}) · saved {ts}  \n"
                    f"`{run_label_cp}`"
                )

                c1, c2, c3, c4 = st.columns([6, 2, 2, 2])
                c1.markdown(label_str)

                if c2.button("Resume", key=f"resume_{run_id_cp}", use_container_width=True):
                    cp_data = _load_checkpoint(run_id_cp)
                    if cp_data and cp_data.get("input_df") is not None:
                        st.session_state["reg_resume_data"] = cp_data
                        st.session_state["reg_run_id"]      = run_id_cp
                        st.rerun()
                    else:
                        st.error("Could not load checkpoint (input snapshot missing). Re-upload the file.")

                # Download partial Excel directly from checkpoint
                saved_cols = cp_meta.get("cols", {})
                partial_xl = _checkpoint_excel_bytes(run_id_cp, saved_cols)
                if partial_xl and c3.download_button(
                    "Download",
                    data=partial_xl,
                    file_name=f"partial_{run_id_cp[:8]}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"dl_{run_id_cp}",
                    use_container_width=True,
                ):
                    pass  # button handles download

                if c4.button("Delete", key=f"del_{run_id_cp}", use_container_width=True):
                    _delete_checkpoint(run_id_cp)
                    st.rerun()

    # ── Handle resume-from-checkpoint (no upload needed if snapshot present) ─
    resume_data = st.session_state.get("reg_resume_data")
    if resume_data is not None:
        cp_meta    = resume_data["meta"]
        run_id     = cp_meta["run_id"]
        resume_from = cp_meta["row_idx"]
        total_rows  = cp_meta["total_rows"]
        prior_results  = resume_data["results"]
        prior_evidence = resume_data["evidence"]
        resume_input_df = resume_data["input_df"]
        saved_cols = cp_meta.get("cols", {})

        st.info(
            f"⏸ **Resuming run `{run_id}`** — "
            f"{resume_from} of {total_rows} rows already completed.  \n"
            "Re-upload your file below to continue from where processing stopped, "
            "or click **Start fresh** to reprocess from the beginning."
        )

        uploaded = st.file_uploader(
            "Re-upload the same file to continue (or upload a new file for a fresh run)",
            type=["csv", "xlsx"],
            key="reg_upload_resume",
        )

        col_a, col_b = st.columns(2)
        if col_b.button("✖ Start fresh instead", use_container_width=True):
            st.session_state.pop("reg_resume_data", None)
            st.session_state.pop("reg_run_id", None)
            st.rerun()

        if uploaded is None:
            # Offer download of partial results from checkpoint while user locates file
            partial_xl = _checkpoint_excel_bytes(run_id, saved_cols)
            if partial_xl:
                st.download_button(
                    f"⬇ Download partial results ({resume_from} rows so far)",
                    data=partial_xl,
                    file_name=f"partial_{run_id[:8]}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            return

        raw_bytes  = uploaded.read()
        file_hash  = _file_hash(raw_bytes)
        upload_run_id = file_hash   # run_id based on file content

        if upload_run_id != run_id:
            # Different file uploaded — treat as fresh run
            st.warning(
                "The uploaded file does not match the previous run's input.  \n"
                "Starting a fresh run with this file."
            )
            st.session_state.pop("reg_resume_data", None)
            prior_results  = []
            prior_evidence = []
            resume_from    = 0
            run_id = upload_run_id
        else:
            st.success(
                f"✅ File matches previous run. Will continue from row **{resume_from + 1}**."
            )

        df = _parse_bytes(raw_bytes, uploaded.name)
        if df is None:
            return

        cols = detect_columns(df)
        # Apply saved column mapping
        for role, col_name in saved_cols.items():
            if col_name and col_name in df.columns:
                cols[role] = col_name

        run_df  = df.head(total_rows).copy()
        # Recompute a run label for this resume session (new timestamp + current settings)
        _resume_label = _make_run_label(haiku_mode, total_rows, int(max_queries), debug_mode)
        _resume_filename = _make_filename(_resume_label, run_id)
        settings_dict = {
            "max_queries": int(max_queries), "batch_n": total_rows,
            "haiku_mode": haiku_mode, "debug_mode": debug_mode,
            "run_label": _resume_label,
        }

        if col_a.button(
            f"▶ Continue from row {resume_from + 1}", type="primary", use_container_width=True
        ):
            progress_bar = st.progress(resume_from / total_rows if total_rows else 0.0)
            status_text  = st.empty()

            def progress_cb(i, total):
                progress_bar.progress(i / total)
                status_text.caption(f"Processing {i} / {total}…")

            enriched_df, evidence_rows, debug_rows = process_dataframe(
                run_df, cols, serper_key, int(max_queries),
                progress_cb=progress_cb,
                run_id=run_id,
                resume_from=resume_from,
                prior_results=prior_results,
                prior_evidence=prior_evidence,
                settings=settings_dict,
                run_label=_resume_label,
                haiku_mode=haiku_mode,
                haiku_api_key=anthropic_key,
                haiku_model=haiku_model,
                haiku_max_rows=int(haiku_max_rows),
                debug_mode=debug_mode,
            )

            progress_bar.progress(1.0)
            status_text.caption(f"Done — {total_rows} companies processed.")

            # Build run_meta for summary sheet
            _run_meta_resume = _build_run_meta(
                enriched_df=enriched_df,
                input_filename="(resumed run)",
                run_id=run_id, run_label=_resume_label,
                total_rows_input=total_rows, batch_n=total_rows,
                max_queries=int(max_queries), haiku_mode=haiku_mode,
                haiku_model=haiku_model, haiku_max_rows=int(haiku_max_rows),
                debug_mode=debug_mode,
                serper_key_present=bool(serper_key),
                anthropic_key_present=bool(anthropic_key),
            )

            st.session_state["reg_enriched"]   = enriched_df
            st.session_state["reg_evidence"]   = evidence_rows
            st.session_state["reg_debug"]      = debug_rows
            st.session_state["reg_original"]   = run_df
            st.session_state["reg_cols"]       = cols
            st.session_state["reg_run_id"]     = run_id
            st.session_state["reg_run_label"]  = _resume_label
            st.session_state["reg_filename"]   = _resume_filename
            st.session_state["reg_run_meta"]   = _run_meta_resume
            st.session_state["reg_debug_mode"] = debug_mode
            st.session_state.pop("reg_resume_data", None)

            # Mark checkpoint complete
            _save_checkpoint(
                run_id, list(prior_results or []) + evidence_rows,
                evidence_rows, total_rows, total_rows, run_df, cols, settings_dict,
                run_label=_resume_label,
            )

        # Show partial download while waiting for user to click Continue
        else:
            partial_xl = _checkpoint_excel_bytes(run_id, cols)
            if partial_xl:
                st.download_button(
                    f"⬇ Download partial results ({resume_from} rows so far)",
                    data=partial_xl,
                    file_name=f"partial_{run_id[:8]}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

        # Fall through to results section if enriched_df is available
        enriched_df = st.session_state.get("reg_enriched")
        if enriched_df is not None:
            st.markdown("---")
            st.markdown("### Results")
            stored_cols  = st.session_state.get("reg_cols", cols)
            _stored_label = st.session_state.get("reg_run_label", _resume_label)
            _stored_fn    = st.session_state.get("reg_filename", _resume_filename)
            _stored_dm    = st.session_state.get("reg_debug_mode", debug_mode)
            _summary_metrics(enriched_df, stored_cols)
            st.markdown("")
            _show_results(enriched_df, stored_cols)
            _show_run_info_block(
                run_label=_stored_label, run_id=run_id,
                batch_n=total_rows, max_queries=int(max_queries),
                haiku_mode=haiku_mode, debug_mode=_stored_dm,
                filename=_stored_fn,
            )
            st.markdown("---")
            _download_section(
                enriched_df,
                st.session_state.get("reg_original", run_df),
                st.session_state.get("reg_evidence", []),
                stored_cols,
                filename=_stored_fn,
                debug_rows=st.session_state.get("reg_debug", []),
                debug_mode=_stored_dm,
                run_meta=st.session_state.get("reg_run_meta"),
            )
        return   # ← resume path ends here

    # =========================================================================
    # NORMAL (fresh) UPLOAD PATH
    # =========================================================================

    uploaded = st.file_uploader(
        "Upload Italian Business Register export (CSV or Excel .xlsx)",
        type=["csv", "xlsx"],
        key="reg_upload",
    )

    if uploaded is None:
        st.info(
            "Upload a CSV or Excel export from the Italian Business Register.  \n"
            "Expected columns: **Company Name**, **Website**, **Email address**, "
            "**City**, **National statistical institute Province**, **Postal Code**, **Phone number**."
        )
        return

    raw_bytes = uploaded.read()
    file_hash = _file_hash(raw_bytes)
    run_id    = file_hash   # one run_id per unique file

    df = _parse_bytes(raw_bytes, uploaded.name)
    if df is None:
        return

    st.success(f"✅ Loaded **{len(df)} companies**, {len(df.columns)} columns from `{uploaded.name}`")

    # Offer resume if a checkpoint exists for this exact file
    existing_cp = _load_checkpoint(run_id)
    if existing_cp and not existing_cp["meta"].get("complete", False):
        row_idx   = existing_cp["meta"].get("row_idx", 0)
        total_rows_saved = existing_cp["meta"].get("total_rows", len(df))
        ts = str(existing_cp["meta"].get("timestamp", ""))[:19]
        st.warning(
            f"⏸ **Unfinished run found** for this file — "
            f"**{row_idx}/{total_rows_saved}** rows completed (saved {ts}).  \n"
            "Click **Resume** to continue, or **Start fresh** to reprocess from the beginning."
        )
        rc1, rc2 = st.columns(2)
        if rc1.button("⏩ Resume previous run", type="primary", use_container_width=True):
            st.session_state["reg_resume_data"] = existing_cp
            st.session_state["reg_run_id"]      = run_id
            st.rerun()
        if rc2.button("🔄 Start fresh (discard saved progress)", use_container_width=True):
            _delete_checkpoint(run_id)
            st.rerun()
        return

    # ── Column detection ──────────────────────────────────────────────────────
    cols = detect_columns(df)

    with st.expander("Column mapping (auto-detected)", expanded=False):
        col_options = ["(none)"] + list(df.columns)

        def _sel(label, role, default):
            cur = cols.get(role)
            idx = col_options.index(cur) if cur and cur in col_options else 0
            chosen = st.selectbox(label, col_options, index=idx, key=f"col_{role}")
            return None if chosen == "(none)" else chosen

        cols["company"]  = _sel("Company name column",  "company",  _REG_COL_COMPANY)
        cols["website"]  = _sel("Website column",        "website",  _REG_COL_WEBSITE)
        cols["email"]    = _sel("Email column",          "email",    _REG_COL_EMAIL)
        cols["city"]     = _sel("City column",           "city",     _REG_COL_CITY)
        cols["province"] = _sel("Province column",       "province", _REG_COL_PROVINCE)
        cols["postcode"] = _sel("Postal code column",    "postcode", _REG_COL_POSTCODE)
        cols["phone"]    = _sel("Phone column",          "phone",    _REG_COL_PHONE)

    if not cols.get("company"):
        st.error("Company name column could not be detected. Please select it above.")
        return

    # ── Detection summary ─────────────────────────────────────────────────────
    detected_info = {k: v for k, v in cols.items() if v}
    missing_info  = {k for k, v in cols.items() if not v}
    st.caption(
        "Detected: " + " · ".join(f"**{k}** → `{v}`" for k, v in detected_info.items())
        + (f"  \n⚠ Not found: {', '.join(missing_info)}" if missing_info else "")
    )

    # ── Input preview ─────────────────────────────────────────────────────────
    preview_cols = [v for v in [
        cols.get("company"), cols.get("website"), cols.get("email"),
        cols.get("city"), cols.get("province"), cols.get("phone"),
    ] if v and v in df.columns]
    st.dataframe(df[preview_cols].head(8), use_container_width=True)

    # ── Batch size option ─────────────────────────────────────────────────────
    max_rows = len(df)
    with st.expander("Batch options", expanded=False):
        run_all = st.checkbox("Process all rows", value=True, key="reg_run_all")
        if not run_all:
            batch_n = st.number_input(
                "Process top N rows (demo / credit-saving mode)",
                min_value=1, max_value=max_rows, value=min(10, max_rows), step=5,
                key="reg_batch_n",
            )
        else:
            batch_n = max_rows

    # ── Run ───────────────────────────────────────────────────────────────────
    if st.button("🧹 Clean and validate register data", type="primary", use_container_width=True):
        run_df = df.head(int(batch_n)).copy()
        n      = len(run_df)
        # Compute label + filename at run start
        _run_label    = _make_run_label(haiku_mode, n, int(max_queries), debug_mode)
        _run_filename = _make_filename(_run_label, run_id)
        settings_dict = {
            "max_queries": int(max_queries), "batch_n": n,
            "haiku_mode": haiku_mode, "debug_mode": debug_mode,
            "run_label": _run_label,
        }

        progress_bar = st.progress(0.0)
        status_text  = st.empty()

        def progress_cb(i, total):
            progress_bar.progress(i / total)
            status_text.caption(f"Processing {i} / {total}…")

        enriched_df, evidence_rows, debug_rows = process_dataframe(
            run_df, cols, serper_key, int(max_queries),
            progress_cb=progress_cb,
            run_id=run_id,
            resume_from=0,
            prior_results=[],
            prior_evidence=[],
            settings=settings_dict,
            run_label=_run_label,
            haiku_mode=haiku_mode,
            haiku_api_key=anthropic_key,
            haiku_model=haiku_model,
            haiku_max_rows=int(haiku_max_rows),
            debug_mode=debug_mode,
        )

        progress_bar.progress(1.0)
        status_text.caption(f"✅ Done — {n} companies processed.")

        # Build run_meta for summary sheet
        _run_meta = _build_run_meta(
            enriched_df=enriched_df,
            input_filename=uploaded.name,
            run_id=run_id, run_label=_run_label,
            total_rows_input=len(df), batch_n=n,
            max_queries=int(max_queries), haiku_mode=haiku_mode,
            haiku_model=haiku_model, haiku_max_rows=int(haiku_max_rows),
            debug_mode=debug_mode,
            serper_key_present=bool(serper_key),
            anthropic_key_present=bool(anthropic_key),
        )

        # Mark complete in checkpoint
        _save_checkpoint(run_id, [], evidence_rows, n, n, run_df, cols, settings_dict,
                         run_label=_run_label)

        st.session_state["reg_enriched"]   = enriched_df
        st.session_state["reg_evidence"]   = evidence_rows
        st.session_state["reg_debug"]      = debug_rows
        st.session_state["reg_original"]   = run_df
        st.session_state["reg_cols"]       = cols
        st.session_state["reg_run_id"]     = run_id
        st.session_state["reg_run_label"]  = _run_label
        st.session_state["reg_filename"]   = _run_filename
        st.session_state["reg_run_meta"]   = _run_meta
        st.session_state["reg_debug_mode"] = debug_mode

    # ── Results ───────────────────────────────────────────────────────────────
    enriched_df = st.session_state.get("reg_enriched")
    if enriched_df is None:
        return

    st.markdown("---")
    st.markdown("### Results")
    stored_cols   = st.session_state.get("reg_cols", cols)
    active_run_id = st.session_state.get("reg_run_id", run_id)
    _out_label    = st.session_state.get("reg_run_label", "")
    _out_filename = st.session_state.get("reg_filename",
                        f"register_cleaned_{active_run_id[:8]}.xlsx")
    _out_dm       = st.session_state.get("reg_debug_mode", debug_mode)

    _summary_metrics(enriched_df, stored_cols)
    st.markdown("")
    _show_results(enriched_df, stored_cols)
    _show_run_info_block(
        run_label=_out_label, run_id=active_run_id,
        batch_n=int(st.session_state.get("reg_run_meta", {}).get("processed_rows", "?")),
        max_queries=int(max_queries), haiku_mode=haiku_mode,
        debug_mode=_out_dm, filename=_out_filename,
    )

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown("---")
    _download_section(
        enriched_df,
        st.session_state.get("reg_original", df),
        st.session_state.get("reg_evidence", []),
        stored_cols,
        filename=_out_filename,
        debug_rows=st.session_state.get("reg_debug", []),
        debug_mode=_out_dm,
        run_meta=st.session_state.get("reg_run_meta"),
    )


if __name__ == "__main__":
    main()
