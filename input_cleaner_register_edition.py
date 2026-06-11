"""
input_cleaner_register_edition.py — Layer 0: mYngle Input Cleaner · Register Edition
======================================================================================
Cleans and enriches Italian Business Register exports before Lead Prioritizer.
Handles missing websites (common in register data), PEC email detection,
multi-website fields, and location-aware Serper search queries.

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

# Generic / directory domains to skip (global + Italian-specific)
_GENERIC_DOMAINS: frozenset = frozenset({
    # Global directories
    "linkedin.com", "facebook.com", "twitter.com", "x.com", "instagram.com",
    "youtube.com", "wikipedia.org", "bloomberg.com", "crunchbase.com",
    "glassdoor.com", "indeed.com", "xing.com", "angel.co", "pitchbook.com",
    "google.com", "bing.com", "yahoo.com", "reuters.com", "ft.com",
    "github.com", "amazon.com", "zoominfo.com", "dnb.com",
    "opencorporates.com", "companieshouse.gov.uk", "app.lusha.com",
    "rocketreach.co", "signalhire.com", "apollo.io", "hunter.io",
    "trustpilot.com", "yelp.com",
    # Italian business registers / directories
    "registroimprese.it", "infocamere.it", "imprese.it",
    "atoka.io", "nixonpowerseo.it", "paginegialle.it", "paginebianche.it",
    "europages.it", "europages.com", "kompass.com", "kompass.it",
    "dnbItaly.com", "cervedgroup.it", "cerved.com",
    "italianmade.com", "viesus.com",
    "aziende.it", "icecat.it", "businessit.it",
    "madeintaly.com", "italyexport.com",
})

# PEC (Posta Elettronica Certificata) domains — never use as company website
_PEC_DOMAIN_PATTERNS: tuple = (
    "pec.it", "pec.com", "pec.eu", "legalmail.it", "legalmail.com",
    "postecert.it", "arubapec.it", "arubapec.eu", "pecactually.it",
    "cert.it", "pecimprese.it", "certificata.it", "pecaziende.it",
    "pecmail.it", "pec.tiscali.it", "pecimprese.com",
    "ordineavvocati", "ordinedottori", "caf", "patronato",
    "libero.it", "yahoo.it", "gmail.com", "hotmail.it",
    "alice.it", "tin.it", "virgilio.it", "live.com", "outlook.com",
    "tiscali.it",
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

_NOISE_TOKENS: frozenset = frozenset({
    "the", "and", "for", "global", "international", "services", "solutions",
    "consulting", "management", "technology", "technologies", "systems",
    "software", "digital", "enterprise", "enterprises", "partners",
    "italia", "italy", "italian", "europe", "european",
    "snc", "srl", "spa", "sas", "del", "della", "degli", "dei",
    "di", "da", "in", "con", "su", "per", "tra", "fra",
})

_TLDS: frozenset = frozenset({
    "com", "net", "org", "it", "eu", "nl", "de", "fr", "be", "uk", "co",
    "io", "biz", "info", "at", "ch", "es", "pl", "cz", "se", "no",
    "dk", "fi", "pt", "hu", "ro", "hr", "gr", "gov", "edu",
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
SRC_ORIGINAL = "original_website"
SRC_EMAIL    = "email_domain"
SRC_SERPER   = "serper_search"
SRC_NONE     = ""

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
    # Must contain at least one dot to be a real domain
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
    # Split on common separators
    parts = re.split(r"[,;\s]+", raw.strip())
    domains = []
    for p in parts:
        d = normalize_domain(p)
        if d:
            domains.append(d)
    return domains


def best_website_domain(raw: str) -> str:
    """
    Parse a multi-website field and return the best single domain:
    - prefer non-generic domains
    - prefer .it TLD for Italian companies
    - return first valid one otherwise
    """
    domains = split_multi_website(raw)
    if not domains:
        return ""
    non_generic = [d for d in domains if not is_generic(d)]
    if not non_generic:
        return ""
    # Prefer .it domains
    it_domains = [d for d in non_generic if d.endswith(".it")]
    return it_domains[0] if it_domains else non_generic[0]


def strip_legal(name: str) -> str:
    cleaned = _LEGAL_TOKENS.sub(" ", name)
    return re.sub(r"\s+", " ", cleaned).strip(" .,/-")


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


def is_generic(domain: str) -> bool:
    return bool(domain) and domain.lower() in _GENERIC_DOMAINS


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


def _conf_label(conf: float) -> str:
    if conf >= 0.75:
        return "High"
    if conf >= 0.45:
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
    company_name: str, city: str, province: str, postcode: str
) -> list[str]:
    """Build up to 3 location-aware Serper queries for Italian company register context."""
    name = company_name.strip()
    queries = []

    # Primary: name + city
    if city:
        queries.append(f'"{name}" "{city}" Italy official website')
    # Secondary: name + province
    if province:
        queries.append(f'"{name}" "{province}" Italy company website')
    # Tertiary: name + postcode
    if postcode:
        queries.append(f'"{name}" "{postcode}" Italy')
    # Fallback: name only
    if not queries or len(queries) < 2:
        queries.append(f'"{name}" sito ufficiale')
        queries.append(f'"{name}" Italy official website')

    return queries[:3]


def search_official_domain_register(
    company_name: str,
    city: str,
    province: str,
    postcode: str,
    email_domain: str,
    serper_key: str,
) -> tuple[str, float, str, list, str]:
    """
    Run up to 3 location-aware Serper queries.
    Scoring bonuses:
      +1 vote if city/province appears in result title/snippet
      +1 vote if result domain matches email domain
    Returns (suggested_domain, confidence, reason, evidence_rows, query_used).
    """
    queries = _build_search_queries(company_name, city, province, postcode)

    candidates: dict[str, float] = {}  # domain → weighted score
    evidence: list[dict] = []
    query_used = queries[0] if queries else ""

    for query in queries:
        results, err = _call_serper(query, serper_key)
        if err:
            break
        for rank, item in enumerate(results):
            url     = item.get("link", "")
            title   = item.get("title", "")
            snippet = item.get("snippet", "")
            domain  = _extract_domain(url)

            if not domain or is_generic(domain):
                evidence.append({
                    "query": query, "title": title, "url": url,
                    "domain": domain, "used": False, "skip_reason": "generic",
                })
                continue

            name_overlap = token_overlap(company_name, domain)
            if name_overlap < 0.1:
                evidence.append({
                    "query": query, "title": title, "url": url,
                    "domain": domain, "used": False,
                    "skip_reason": f"low_overlap({name_overlap:.2f})",
                })
                continue

            # Base score = name overlap × position weight
            position_weight = 1.0 / (rank + 1)
            score = name_overlap * position_weight

            # Location bonus
            loc_match = location_in_text(title + " " + snippet, city, province)
            if loc_match:
                score += 0.3

            # Email domain bonus
            if email_domain and domain == email_domain:
                score += 0.4

            candidates[domain] = candidates.get(domain, 0.0) + score
            evidence.append({
                "query": query, "title": title[:120], "url": url,
                "domain": domain, "name_overlap": round(name_overlap, 3),
                "location_match": loc_match,
                "email_match": (domain == email_domain),
                "score": round(score, 3), "used": True,
            })
        time.sleep(0.3)

    if not candidates:
        return "", 0.0, "No candidate domain found in search results.", evidence, query_used

    best = max(candidates, key=lambda d: candidates[d])
    best_score = candidates[best]
    name_ov = token_overlap(company_name, best)

    # Confidence calibration
    if best_score >= 0.8 and name_ov >= 0.4:
        conf, reason = 0.85, "Strong name match + location/email signals confirmed."
    elif best_score >= 0.5 or name_ov >= 0.4:
        conf, reason = 0.65, "Reasonable name match with search confirmation."
    elif name_ov >= 0.2:
        conf, reason = 0.45, "Weak but plausible name-domain overlap. Review recommended."
    else:
        conf, reason = 0.25, "Domain found but name-domain overlap is very low."

    # Add explanation of why this domain was chosen
    top_ev = next((e for e in evidence if e.get("domain") == best and e.get("used")), {})
    extras = []
    if top_ev.get("location_match"):
        extras.append(f"city/province '{city or province}' found in result")
    if top_ev.get("email_match"):
        extras.append(f"matches email domain ({best})")
    if extras:
        reason += " " + "; ".join(extras).capitalize() + "."

    return best, conf, reason, evidence, query_used


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
) -> dict:
    """
    Validate one register row. Returns result fields dict.

    Decision flow:
      1. Parse and clean website → normalized_input_website
      2. If website valid and non-generic → OK / LIKELY_OK
      3. If website missing/invalid → try email domain
      4. If email domain looks like a real company website → EMAIL_DERIVED
      5. If still missing or generic → Serper search
      6. If Serper finds confident result → MISSING_DOMAIN_FIXED / SUGGEST_REPLACE
      7. Otherwise → MISSING_DOMAIN / NO_CONFIDENT_MATCH
    """
    name    = str(company_name or "").strip()
    email   = str(raw_email or "").strip()
    city    = str(city or "").strip()
    province = str(province or "").strip()
    postcode = str(postcode or "").strip()

    email_domain = extract_email_domain(email)
    email_is_pec = is_pec_or_personal_email(email_domain)

    # Parse website — may contain multiple URLs
    norm_website = best_website_domain(raw_website)

    result = {
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
    }

    if not name:
        result.update(
            domain_action="NO_CONFIDENT_MATCH",
            domain_confidence="None",
            domain_reason="Company name is blank.",
            manual_review_needed=True,
        )
        return result

    # ── Case 1: website present, non-generic ─────────────────────────────────
    if norm_website and not is_generic(norm_website):
        overlap = token_overlap(name, norm_website)
        if overlap >= 0.5:
            result.update(
                domain_action="OK",
                domain_confidence="High",
                domain_reason="Website domain matches company name tokens closely.",
                manual_review_needed=False,
            )
            return result

        if overlap >= 0.2:
            result.update(
                domain_action="LIKELY_OK",
                domain_confidence="Medium",
                domain_reason="Website present; partial name-domain overlap (group/abbreviation likely).",
                manual_review_needed=False,
            )
            return result

        # Low overlap — search to confirm or find a better domain
        if serper_key:
            suggested, conf, reason, ev, query = search_official_domain_register(
                name, city, province, postcode, email_domain, serper_key
            )
            _fill_serper_top(result, ev, query)
            if suggested and conf >= 0.45 and suggested != norm_website:
                result.update(
                    validated_domain=suggested,
                    recommended_domain=suggested,
                    domain_source=SRC_SERPER,
                    domain_action="SUGGEST_REPLACE",
                    domain_confidence=_conf_label(conf),
                    domain_reason=f"Low name-website overlap ({overlap:.2f}). {reason}",
                    manual_review_needed=True,
                )
                return result
            if suggested and suggested == norm_website:
                result.update(
                    domain_action="LIKELY_OK",
                    domain_confidence="Medium",
                    domain_reason=f"Search confirms website despite low token overlap ({overlap:.2f}).",
                    manual_review_needed=False,
                )
                return result

        result.update(
            domain_action="REVIEW",
            domain_confidence="Low",
            domain_reason=f"Website present but low name-domain overlap ({overlap:.2f}). Manual check recommended.",
            manual_review_needed=True,
        )
        return result

    # ── Case 2: website is a generic/directory site ──────────────────────────
    if norm_website and is_generic(norm_website):
        if serper_key:
            suggested, conf, reason, ev, query = search_official_domain_register(
                name, city, province, postcode, email_domain, serper_key
            )
            _fill_serper_top(result, ev, query)
            if suggested and conf >= 0.45:
                result.update(
                    validated_domain=suggested,
                    recommended_domain=suggested,
                    domain_source=SRC_SERPER,
                    domain_action="SUGGEST_REPLACE",
                    domain_confidence=_conf_label(conf),
                    domain_reason=f"Register website ({norm_website}) is a directory. {reason}",
                    manual_review_needed=True,
                )
                return result
        result.update(
            validated_domain="",
            domain_action="REVIEW",
            domain_confidence="Low",
            domain_reason=f"Register website ({norm_website}) is a generic directory site.",
            manual_review_needed=True,
        )
        return result

    # ── Case 3: website missing — try email domain ────────────────────────────
    if email_domain and not email_is_pec and not is_generic(email_domain):
        email_overlap = token_overlap(name, email_domain)
        if email_overlap >= 0.3:
            # Good enough — use email domain as suggested domain
            result.update(
                validated_domain=email_domain,
                recommended_domain=email_domain,
                domain_source=SRC_EMAIL,
                domain_action="EMAIL_DERIVED",
                domain_confidence="Medium",
                domain_reason=(
                    f"Website missing. Email domain '{email_domain}' matches company name "
                    f"(overlap {email_overlap:.2f}). Used as website proxy."
                ),
                manual_review_needed=True,
            )
            # Still run Serper to confirm or override
            if serper_key:
                suggested, conf, reason, ev, query = search_official_domain_register(
                    name, city, province, postcode, email_domain, serper_key
                )
                _fill_serper_top(result, ev, query)
                if suggested and conf >= 0.55:
                    result.update(
                        validated_domain=suggested,
                        recommended_domain=suggested,
                        domain_source=SRC_SERPER,
                        domain_action="MISSING_DOMAIN_FIXED",
                        domain_confidence=_conf_label(conf),
                        domain_reason=f"Website missing. Email domain was proxy; Serper confirmed: {reason}",
                    )
            return result

    # ── Case 4: website missing — Serper search ───────────────────────────────
    if serper_key:
        suggested, conf, reason, ev, query = search_official_domain_register(
            name, city, province, postcode, email_domain, serper_key
        )
        _fill_serper_top(result, ev, query)
        if suggested and conf >= 0.45:
            result.update(
                validated_domain=suggested,
                recommended_domain=suggested,
                domain_source=SRC_SERPER,
                domain_action="MISSING_DOMAIN_FIXED",
                domain_confidence=_conf_label(conf),
                domain_reason=f"Website missing in register. {reason}",
                manual_review_needed=True,
            )
        else:
            result.update(
                validated_domain="",
                domain_source=SRC_NONE,
                domain_action="MISSING_DOMAIN",
                domain_confidence="None",
                domain_reason="Website missing and no confident result found in search.",
                manual_review_needed=True,
            )
    else:
        # No Serper — try email domain even with low overlap as a last resort
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
            )
        else:
            result.update(
                validated_domain="",
                domain_source=SRC_NONE,
                domain_action="MISSING_DOMAIN",
                domain_confidence="None",
                domain_reason="Website missing. No Serper key. No usable email domain.",
                manual_review_needed=True,
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

# New output columns added by this tool
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
]


def process_dataframe(
    df: pd.DataFrame,
    cols: dict,
    serper_key: str | None,
    progress_cb=None,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Process all rows. Returns (enriched_df, evidence_rows).
    cols: dict from detect_columns().
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
            name, website, email, city, province, postcode, serper_key
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
            })

        if progress_cb:
            progress_cb(i + 1, n)

    result_df = pd.DataFrame(results, index=df.index)
    enriched  = pd.concat([df.copy(), result_df], axis=1)
    # Deduplicate columns (original df might already have some of these names)
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

        # Bold red on validated_domain when it differs from input website
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
    company_name, website_url (best guess), email, city, province, phone.
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
            url = norm  # blank if missing

        rows.append({
            "company_name": _sv(company_col),
            "website_url":  url,
            "email":        _sv(email_col),
            "city":         _sv(city_col),
            "province":     _sv(province_col),
            "phone":        _sv(phone_col),
            "domain_action":    action,
            "domain_confidence": str(r.get("domain_confidence", "") or ""),
            "domain_source":     str(r.get("domain_source", "") or ""),
            "manual_review_needed": r.get("manual_review_needed", False),
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
        if not url:
            url_cell.fill = blank_fill
        elif action == "EMAIL_DERIVED":
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
    ])
    _write_sheet(ws4, ev_df)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# =============================================================================
# SUMMARY METRICS
# =============================================================================


def _summary_metrics(df: pd.DataFrame, cols: dict) -> None:
    actions  = df.get("domain_action", pd.Series(dtype=str)).astype(str)
    sources  = df.get("domain_source", pd.Series(dtype=str)).astype(str)

    total    = len(df)
    has_web  = int(df.get(cols.get("website") or "_", pd.Series("")).astype(str)
                   .str.strip().replace("", pd.NA).notna().sum()) if cols.get("website") else 0
    has_email = int(df.get(cols.get("email") or "_", pd.Series("")).astype(str)
                    .str.strip().replace("", pd.NA).notna().sum()) if cols.get("email") else 0
    has_phone = int(df.get(cols.get("phone") or "_", pd.Series("")).astype(str)
                    .str.strip().replace("", pd.NA).notna().sum()) if cols.get("phone") else 0
    accepted  = int(actions.isin(["OK", "LIKELY_OK"]).sum())
    from_email = int(sources.isin([SRC_EMAIL]).sum())
    from_serper = int(sources.isin([SRC_SERPER]).sum())
    review   = int(
        df.get("manual_review_needed", pd.Series(dtype=str))
        .astype(str).str.lower().isin(["true", "1", "yes"]).sum()
    )

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

    row1 = st.columns(4)
    card(row1[0], "Total companies",         total,      "#0B4A92")
    card(row1[1], "With original website",   has_web,    "#2E7D32",
         f"{round(has_web/total*100) if total else 0}% of rows")
    card(row1[2], "With email",              has_email,  "#1565C0")
    card(row1[3], "With phone",              has_phone,  "#37474F")

    st.markdown("")
    row2 = st.columns(4)
    card(row2[0], "Accepted from website",   accepted,    "#2E7D32")
    card(row2[1], "Derived from email",      from_email,  "#1565C0", "used as domain proxy")
    card(row2[2], "Found by Serper",         from_serper, "#E65100")
    card(row2[3], "Need manual review",      review,      "#B71C1C")


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
        "Cleans Italian Business Register exports before Lead Prioritizer enrichment"
    )

    # ── API key ───────────────────────────────────────────────────────────────
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
            run_df, cols, serper_key, progress_cb
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
        "1. **Best Guess Input** — company, website, email, city, province, phone (ready for Lead Prioritizer)  \n"
        "2. **Cleaned Register Input** — all original columns + validation columns  \n"
        "3. **Review Needed** — rows requiring manual check  \n"
        "4. **Original Input** — unchanged source data  \n"
        "5. **Raw Search Evidence** — Serper queries and results"
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
