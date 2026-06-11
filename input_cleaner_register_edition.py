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

import io
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st

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

SERPER_URL = "https://google.serper.dev/search"

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
) -> tuple[str, float, str, list, str, str, list]:
    """
    Run up to max_queries Serper queries with multi-variant brand scoring.

    Returns:
      (suggested_domain, confidence, reason, evidence_rows,
       query_used, name_variant_used, top_3_domains)
    """
    name_variants = extract_name_variants(company_name)
    queries = _build_search_queries(name_variants, city, province, postcode, max_queries)

    candidates: dict[str, float] = {}   # domain → cumulative score
    domain_variant: dict[str, str] = {} # domain → which variant matched best
    evidence: list[dict] = []
    query_used = queries[0] if queries else ""
    rejection_notes: list[str] = []

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
                    "query": query, "title": title, "url": url,
                    "domain": domain, "used": False, "skip_reason": "generic/blacklisted",
                    "score": 0,
                })
                rejection_notes.append(f"{domain}: blacklisted")
                continue

            score = _score_candidate(
                domain, rank, title, snippet,
                name_variants, email_domain, city, province,
            )

            # Very low score — skip but note it
            if score < 0.1:
                evidence.append({
                    "query": query, "title": title[:120], "url": url,
                    "domain": domain, "used": False,
                    "skip_reason": f"score_too_low({score:.3f})",
                    "score": score,
                })
                rejection_notes.append(f"{domain}: score too low ({score:.3f})")
                continue

            if domain not in candidates or score > candidates[domain]:
                candidates[domain] = score
                # Track which variant drove the best match
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
        top3 = []
        return (
            "", 0.0,
            "No candidate domain found in search results. " + "; ".join(rejection_notes[:3]),
            evidence, query_used, "", top3,
        )

    # Sort by score
    sorted_cands = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    best, best_score = sorted_cands[0]
    top3 = [d for d, _ in sorted_cands[:3]]

    b_ov   = brand_overlap(name_variants.get("brand", ""), best)
    f_ov   = token_overlap(name_variants["full"], best)
    best_variant = domain_variant.get(best, "full")

    # Find top evidence entry for explanation
    top_ev = next(
        (e for e in evidence if e.get("domain") == best and e.get("used")), {}
    )

    # Confidence logic
    email_confirmed = (email_domain and best == email_domain)
    loc_match = top_ev.get("location_match", False)
    official  = top_ev.get("official_signal", False)

    if email_confirmed and best_score >= 0.6:
        conf = 0.88
        reason = f"Serper confirms email domain '{best}' as top result."
    elif best_score >= 1.2 and (b_ov >= 0.7 or f_ov >= 0.5):
        conf = 0.85
        reason = "Strong brand/name match + search position + supporting signals."
    elif best_score >= 0.8 and (b_ov >= 0.4 or f_ov >= 0.35):
        conf = 0.72
        reason = "Good name match with search confirmation."
    elif best_score >= 0.5 or b_ov >= 0.4:
        conf = 0.55
        reason = "Reasonable match; partial name-domain overlap."
    elif best_score >= 0.3:
        conf = 0.38
        reason = "Weak but plausible match. Manual review recommended."
    else:
        conf = 0.20
        reason = "Very weak domain match. High uncertainty."

    extras = []
    if loc_match:
        extras.append(f"city/province found in result")
    if email_confirmed:
        extras.append(f"matches email domain ({best})")
    if official:
        extras.append("official-page keyword in title/snippet")
    if best.endswith(".it"):
        extras.append(".it domain")
    if extras:
        reason += " — " + "; ".join(extras) + "."

    return best, conf, reason, evidence, query_used, best_variant, top3


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
) -> dict:
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
        return result

    # Helper: run Serper and fill result fields
    def _run_serper(existing_email_domain=""):
        sug, conf, reason, ev, query, variant, top3 = search_official_domain_register(
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
            result["best_candidate_score"] = str(round(
                next((s for d, s in {d: 0.0 for d in all_doms}.items() if d == sug), conf), 3
            ))
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
            return result

        if best_ov >= 0.15:
            result.update(
                domain_action="LIKELY_OK",
                domain_confidence="Medium",
                domain_reason="Website present; partial name-domain overlap (group/abbreviation likely).",
                manual_review_needed=False,
                website_discovery_method="original_website_partial_match",
            )
            return result

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
                return result
            if suggested and suggested == norm_website:
                result.update(
                    domain_action="LIKELY_OK",
                    domain_confidence="Medium",
                    domain_reason=f"Search confirms website despite low token overlap ({best_ov:.2f}).",
                    manual_review_needed=False,
                    website_discovery_method="serper_confirmed_original_website",
                )
                return result

        result.update(
            domain_action="REVIEW",
            domain_confidence="Low",
            domain_reason=f"Website present but low name-domain overlap ({best_ov:.2f}). Manual check recommended.",
            manual_review_needed=True,
            website_discovery_method="original_website_low_confidence",
        )
        return result

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
                return result
        result.update(
            validated_domain="",
            domain_action="REVIEW",
            domain_confidence="Low",
            domain_reason=f"Register website ({norm_website}) is a generic directory site.",
            manual_review_needed=True,
            website_discovery_method="none_website_blacklisted",
        )
        return result

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
                    return result
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
                    return result
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
                    return result

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
                return result

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

    return result


def _fill_serper_top(result: dict, evidence: list, query: str) -> None:
    result["search_query_used"] = query
    used = [e for e in evidence if e.get("used")]
    top  = used or [e for e in evidence if e.get("domain")]
    if top:
        result["serper_top_result_title"]  = str(top[0].get("title", ""))[:120]
        result["serper_top_result_url"]    = top[0].get("url", "")
        result["serper_top_result_domain"] = top[0].get("domain", "")


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
]


def process_dataframe(
    df: pd.DataFrame,
    cols: dict,
    serper_key: str | None,
    max_queries: int = 5,
    progress_cb=None,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Process all rows. Returns (enriched_df, evidence_rows).
    cols: dict from detect_columns().
    max_queries: max Serper queries per company.
    """
    results = []
    evidence_rows: list[dict] = []
    n = len(df)

    company_col  = cols.get("company") or ""
    website_col  = cols.get("website") or ""
    email_col    = cols.get("email") or ""
    city_col     = cols.get("city") or ""
    province_col = cols.get("province") or ""
    postcode_col = cols.get("postcode") or ""

    for i, (_, row) in enumerate(df.iterrows()):
        def _sv(col):
            return str(row.get(col, "") or "").strip() if col else ""

        name     = _sv(company_col)
        website  = _sv(website_col)
        email    = _sv(email_col)
        city     = _sv(city_col)
        province = _sv(province_col)
        postcode = _sv(postcode_col)

        res = validate_register_row(
            name, website, email, city, province, postcode, serper_key, max_queries
        )
        results.append(res)

        # Collect evidence for Raw Search Evidence sheet
        query = res.get("search_query_used", "")
        if query:
            evidence_rows.append({
                "company_name":          name,
                "city":                  city,
                "province":              province,
                "search_query_used":     query,
                "serper_top_title":      res.get("serper_top_result_title", ""),
                "serper_top_url":        res.get("serper_top_result_url", ""),
                "serper_top_domain":     res.get("serper_top_result_domain", ""),
                "validated_domain":      res.get("validated_domain", ""),
                "domain_source":         res.get("domain_source", ""),
                "domain_action":         res.get("domain_action", ""),
                "domain_confidence":     res.get("domain_confidence", ""),
                "name_variant_used":     res.get("name_variant_used", ""),
                "top_3_candidates":      res.get("top_3_candidate_domains", ""),
                "rejection_reason":      res.get("rejection_reason_if_missing", ""),
                "discovery_method":      res.get("website_discovery_method", ""),
            })

        if progress_cb:
            progress_cb(i + 1, n)

    result_df = pd.DataFrame(results, index=df.index)
    enriched  = pd.concat([df.copy(), result_df], axis=1)
    enriched  = enriched.loc[:, ~enriched.columns.duplicated()]
    return enriched, evidence_rows


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

        if action in ("OK", "LIKELY_OK"):
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


def build_excel(
    enriched_df: pd.DataFrame,
    original_df: pd.DataFrame,
    evidence_rows: list[dict],
    cols: dict,
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


# =============================================================================
# STREAMLIT UI
# =============================================================================


def _load_secrets_key() -> str | None:
    try:
        return st.secrets.get("SERPER_API_KEY") or st.secrets.get("serper_api_key")
    except Exception:
        return None


def _load_file(uploaded) -> pd.DataFrame | None:
    raw  = uploaded.read()
    name = uploaded.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(io.BytesIO(raw), dtype=str).fillna("")
        else:
            return pd.read_excel(io.BytesIO(raw), dtype=str).fillna("")
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return None


def main():
    st.title("🇮🇹 Input Cleaner · Register Edition")
    st.caption(
        "Layer 0 · mYngle Sales Intelligence · "
        "Cleans Italian Business Register exports before Lead Prioritizer enrichment  \n"
        "Website Discovery v2 — multi-variant brand extraction, 8 query strategies, "
        "aggressive email-domain usage"
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

    st.sidebar.markdown("---")
    max_queries = st.sidebar.selectbox(
        "Max Serper queries per company",
        options=[3, 5, 8],
        index=1,
        help=(
            "3 = fast/cheap · 5 = default, good balance · 8 = maximum discovery.\n\n"
            "Each query costs 1 Serper credit. For 200 companies: "
            "3 queries = up to 600 credits, 5 = up to 1000, 8 = up to 1600."
        ),
    )
    st.sidebar.caption(
        f"With {max_queries} queries/company, each missing website will try up to "
        f"{max_queries} search strategies (name variants, location, site:.it)."
    )

    # ── Upload ────────────────────────────────────────────────────────────────
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

    df = _load_file(uploaded)
    if df is None:
        return

    st.success(f"✅ Loaded **{len(df)} companies**, {len(df.columns)} columns from `{uploaded.name}`")

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
        run_df  = df.head(int(batch_n)).copy()
        n       = len(run_df)
        progress_bar = st.progress(0.0)
        status_text  = st.empty()

        def progress_cb(i, total):
            progress_bar.progress(i / total)
            status_text.caption(f"Processing {i} / {total}…")

        enriched_df, evidence_rows = process_dataframe(
            run_df, cols, serper_key, int(max_queries), progress_cb
        )

        progress_bar.progress(1.0)
        status_text.caption(f"Done — {n} companies processed.")

        st.session_state["reg_enriched"]  = enriched_df
        st.session_state["reg_evidence"]  = evidence_rows
        st.session_state["reg_original"]  = run_df
        st.session_state["reg_cols"]      = cols

    # ── Results ───────────────────────────────────────────────────────────────
    enriched_df = st.session_state.get("reg_enriched")
    if enriched_df is None:
        return

    st.markdown("---")
    st.markdown("### Results")
    stored_cols = st.session_state.get("reg_cols", cols)
    _summary_metrics(enriched_df, stored_cols)
    st.markdown("")

    # Results table — show key columns only
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

    # Review expander
    review_mask = (
        enriched_df.get("manual_review_needed", pd.Series(False))
        .astype(str).str.lower().isin(["true", "1", "yes"])
    )
    if review_mask.any():
        with st.expander(
            f"🔴 Rows needing manual review ({int(review_mask.sum())})", expanded=False
        ):
            st.dataframe(enriched_df[review_mask][show_cols], use_container_width=True)

    # ── Download ──────────────────────────────────────────────────────────────
    evidence_rows = st.session_state.get("reg_evidence", [])
    original_df   = st.session_state.get("reg_original", df)

    excel_bytes = build_excel(
        enriched_df, original_df, evidence_rows, stored_cols
    )

    st.markdown("---")
    st.markdown(
        "**Output sheets:**  \n"
        "1. **Best Guess Input** — company, website, email, city, province, phone + discovery method (ready for Lead Prioritizer)  \n"
        "2. **Cleaned Register Input** — all original columns + all validation + diagnostic columns  \n"
        "3. **Review Needed** — rows requiring manual check  \n"
        "4. **Original Input** — unchanged source data  \n"
        "5. **Raw Search Evidence** — Serper queries, top results, name variants, rejection reasons"
    )

    st.download_button(
        "⬇ Download cleaned register Excel",
        data=excel_bytes,
        file_name="register_cleaned_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        type="primary",
    )


if __name__ == "__main__":
    main()
