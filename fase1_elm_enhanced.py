"""
ELM Enhanced — Extreme Light Mode with metadata extraction
==========================================================
Extends the ELM pipeline from enrich_clients_claude.py with:
- hreflang tag extraction
- HTML lang attribute detection
- meta description / Open Graph tag presence
- LinkedIn company link detection
- Careers/jobs link detection
- A language-complexity score derived from the above
- Sitemap scan for relevant company paths
- Workforce keyword signals and sector-cluster detection

Only utility functions and constants are imported from the original file;
the fetch/extract/enrich pipeline is fully re-implemented here.
"""

import re
import requests
from bs4 import BeautifulSoup

from enrich_clients_claude import (
    clean_domain,
    normalize_url,
    load_cache,
    save_cache,
    _ELM_SLUGS,
    _ELM_UA,
    _ELM_KW_INTERNATIONAL,
    _ELM_KW_LANGUAGES,
    _ELM_KW_HIRING,
    _ELM_KW_GROWTH,
    _ELM_KW_TRAINING,
    _ELM_KW_OFFICES,
    _ELM_KW_TECHNOLOGY,
    _ELM_COUNTRIES,
    _elm_count,
    _elm_find,
    _elm_score,
)


_ELM_KW_WORKFORCE = [
    "global team", "customer service", "sales", "support", "operations",
    "engineers", "consultants", "client-facing", "communication",
    "remote team", "distributed team", "customer-facing", "field team",
]

_ELM_KW_SECTOR = {
    "financial_services": ["financial services", "banking", "insurance",
                           "fintech", "wealth management"],
    "manufacturing":      ["manufacturing", "industrial", "production",
                           "factory", "supply chain"],
    "technology":         ["saas", "software", "cloud", "platform",
                           "machine learning", "artificial intelligence"],
    "healthcare":         ["healthcare", "pharma", "pharmaceutical",
                           "medical", "biotech", "life sciences"],
    "logistics":          ["logistics", "transport", "freight",
                           "shipping", "warehousing"],
    "consulting":         ["consulting", "advisory",
                           "management consulting", "professional services"],
    "education":          ["education", "university",
                           "e-learning", "edtech"],
    "hospitality":        ["hospitality", "hotel", "travel", "tourism"],
    "engineering":        ["engineering", "construction", "infrastructure"],
    "retail":             ["retail", "e-commerce", "ecommerce",
                           "consumer goods"],
}

_SITEMAP_KEYWORDS = [
    "about", "company", "careers", "jobs", "locations", "offices", "global",
    "international", "academy", "training", "learning", "contact",
]


# ─────────────────────────────────────────────────────────────────────────────
# Function 1 — single-page fetch with metadata
# ─────────────────────────────────────────────────────────────────────────────

def fetch_page_with_metadata(url: str) -> dict:
    """Fetch one URL and return body text plus HTML metadata signals."""
    empty = {
        "text": "",
        "status_code": 0,
        "meta_lang": "",
        "hreflang_langs": [],
        "meta_description_found": False,
        "og_tags_found": False,
        "linkedin_link_found": False,
        "careers_link_found": False,
    }
    try:
        headers = {"User-Agent": _ELM_UA}
        resp = requests.get(url, headers=headers, timeout=12, allow_redirects=True)

        try:
            soup = BeautifulSoup(resp.text, "lxml")
        except Exception:
            soup = BeautifulSoup(resp.text, "html.parser")

        # ── Metadata extraction (before any decompose) ────────────────────────
        html_tag = soup.find("html")
        meta_lang = html_tag.get("lang", "") if html_tag else ""

        hreflang_langs = [
            tag["hreflang"]
            for tag in soup.find_all("link", rel="alternate")
            if tag.get("hreflang")
        ]

        meta_description_found = bool(
            soup.find("meta", attrs={"name": "description"})
        )

        og_tags_found = any(
            str(tag.get("property", "")).startswith("og:")
            for tag in soup.find_all("meta", property=True)
        )

        linkedin_link_found = any(
            "linkedin.com/company" in str(tag.get("href", ""))
            for tag in soup.find_all("a", href=True)
        )

        careers_link_found = any(
            str(tag.get("href", "")).rstrip("/").endswith(("/careers", "/jobs"))
            for tag in soup.find_all("a", href=True)
        )

        # ── Body text extraction ──────────────────────────────────────────────
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)[:8_000]

        return {
            "text": text,
            "status_code": resp.status_code,
            "meta_lang": meta_lang,
            "hreflang_langs": hreflang_langs,
            "meta_description_found": meta_description_found,
            "og_tags_found": og_tags_found,
            "linkedin_link_found": linkedin_link_found,
            "careers_link_found": careers_link_found,
        }
    except Exception:
        return empty


# ─────────────────────────────────────────────────────────────────────────────
# Sitemap scan
# ─────────────────────────────────────────────────────────────────────────────

def fetch_sitemap(base_url: str) -> dict:
    """Fetch /sitemap.xml and return relevant URL paths."""
    try:
        headers = {"User-Agent": _ELM_UA}
        resp = requests.get(
            f"{base_url.rstrip('/')}/sitemap.xml",
            headers=headers,
            timeout=12,
            allow_redirects=True,
        )
        if resp.status_code != 200:
            return {"sitemap_found": False, "sitemap_relevant_paths": ""}
        locs = re.findall(r"<loc>(.*?)</loc>", resp.text)
        matching = [
            url for url in locs
            if any(kw in url.lower() for kw in _SITEMAP_KEYWORDS)
        ]
        return {
            "sitemap_found": True,
            "sitemap_relevant_paths": ", ".join(matching[:10]),
        }
    except Exception:
        return {"sitemap_found": False, "sitemap_relevant_paths": ""}


# ─────────────────────────────────────────────────────────────────────────────
# Function 2 — multi-page fetch with caching
# ─────────────────────────────────────────────────────────────────────────────

def fetch_company_enhanced(base_url: str, domain: str) -> dict:
    """
    Fetch all ELM slug pages for a company, enriched with homepage metadata.
    Returns cached result on re-run.
    """
    ck = f"elm2_{domain}"
    cached = load_cache(ck)
    if cached and "pages" in cached:
        return cached

    base = base_url.rstrip("/")
    pages: dict = {}
    fetched: list = []
    failed: list = []
    metadata: dict = {}

    for slug in _ELM_SLUGS:
        target = base if slug == "" else f"{base}{slug}"
        label = slug if slug else "/"
        result = fetch_page_with_metadata(target)

        if result["status_code"] == 200:
            pages[label] = result["text"]
            fetched.append(label)
            if slug == "":
                metadata = {
                    "meta_lang": result["meta_lang"],
                    "hreflang_langs": result["hreflang_langs"],
                    "meta_description_found": result["meta_description_found"],
                    "og_tags_found": result["og_tags_found"],
                    "linkedin_link_found": result["linkedin_link_found"],
                    "careers_link_found": result["careers_link_found"],
                }
        else:
            failed.append(f"{label}({result['status_code']})")

    sitemap = fetch_sitemap(base)
    metadata.update(sitemap)

    out = {"pages": pages, "fetched": fetched, "failed": failed, "metadata": metadata}
    save_cache(ck, out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Function 3 — signal extraction with metadata fields
# ─────────────────────────────────────────────────────────────────────────────

def extract_signals_enhanced(pages: dict, metadata: dict) -> dict:
    """Extract keyword signals plus metadata fields and language-complexity score."""
    # Reconstruct fetched/failed from pages (needed for status field)
    fetched = list(pages.keys())
    failed: list = []

    all_text = " ".join(pages.values())

    kw_intl      = _elm_count(all_text, _ELM_KW_INTERNATIONAL)
    langs        = _elm_find(all_text, _ELM_KW_LANGUAGES)
    kw_hiring    = _elm_count(all_text, _ELM_KW_HIRING)
    kw_growth    = _elm_count(all_text, _ELM_KW_GROWTH)
    kw_training  = _elm_count(all_text, _ELM_KW_TRAINING)
    kw_offices   = _elm_count(all_text, _ELM_KW_OFFICES)
    kw_tech      = _elm_count(all_text, _ELM_KW_TECHNOLOGY)
    countries    = _elm_find(all_text, _ELM_COUNTRIES)
    kw_workforce = _elm_count(all_text, _ELM_KW_WORKFORCE)

    sector_scores = {
        sector: _elm_count(all_text, kws)
        for sector, kws in _ELM_KW_SECTOR.items()
    }
    best_sector = max(sector_scores, key=sector_scores.get)
    elm_sector_cluster = best_sector if sector_scores[best_sector] > 0 else "unknown"

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

    # ── Language-complexity score ─────────────────────────────────────────────
    n = len(metadata.get("hreflang_langs", []))
    if n == 0:
        score = 0
    elif n == 1:
        score = 2
    elif n <= 3:
        score = 5
    elif n <= 6:
        score = 7
    elif n <= 9:
        score = 9
    else:
        score = 10
    lang = metadata.get("meta_lang", "").lower()
    if lang and not lang.startswith("en"):
        score = min(score + 1, 10)

    return {
        # Status
        "elm_fetch_status":  fetch_status,
        "elm_pages_fetched": ", ".join(fetched),
        "elm_pages_failed":  ", ".join(failed),
        "elm_total_chars":   sum(len(v) for v in pages.values()),
        "elm_error":         "",
        # Keywords
        "elm_kw_international":   kw_intl,
        "elm_kw_languages":       ", ".join(sorted(set(langs))),
        "elm_kw_hiring":          kw_hiring,
        "elm_kw_growth":          kw_growth,
        "elm_kw_training":        kw_training,
        "elm_kw_offices":         kw_offices,
        "elm_kw_technology":      kw_tech,
        "elm_kw_countries_found": ", ".join(sorted(set(countries))),
        # Scores
        "elm_score_international": s_intl,
        "elm_score_hiring":        s_hiring,
        "elm_score_growth":        s_growth,
        "elm_score_training":      s_training,
        "elm_score_technology":    s_tech,
        "elm_score_overall_icp":   s_overall,
        # Metadata fields
        "elm_meta_lang":              metadata.get("meta_lang", ""),
        "elm_hreflang_count":         len(metadata.get("hreflang_langs", [])),
        "elm_hreflang_langs":         ", ".join(metadata.get("hreflang_langs", [])),
        "elm_meta_description_found": metadata.get("meta_description_found", False),
        "elm_og_tags_found":          metadata.get("og_tags_found", False),
        "elm_linkedin_link_found":    metadata.get("linkedin_link_found", False),
        "elm_careers_link_found":     metadata.get("careers_link_found", False),
        # New score
        "elm_score_language_complexity": score,
        # Workforce and sector
        "elm_kw_workforce":               kw_workforce,
        "elm_score_workforce":            _elm_score(kw_workforce, 1.5),
        "elm_sector_cluster":             elm_sector_cluster,
        # Sitemap
        "elm_sitemap_found":              metadata.get("sitemap_found", False),
        "elm_sitemap_relevant_paths":     metadata.get("sitemap_relevant_paths", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Function 4 — row-level entry point
# ─────────────────────────────────────────────────────────────────────────────

def enrich_one_row_enhanced(company_name: str, raw_url: str) -> tuple:
    """
    Enhanced ELM row enrichment with metadata extraction.
    Returns (signals_dict, debug_dict) matching the contract of
    enrich_one_row_light().
    """
    empty_status = {
        "elm_fetch_status": "failed",
        "elm_pages_fetched": "",
        "elm_pages_failed": "",
        "elm_total_chars": 0,
        "elm_error": "",
    }

    url = normalize_url(raw_url) if raw_url else ""
    if not url:
        row = {**empty_status, "elm_error": "No URL provided"}
        return row, {"company": company_name, "url": raw_url, "status": "no_url"}

    domain = clean_domain(url)
    try:
        result = fetch_company_enhanced(url, domain)
    except Exception as exc:
        row = {**empty_status, "elm_error": str(exc)[:200]}
        return row, {"company": company_name, "url": url, "status": "fetch_error", "error": str(exc)}

    pages    = result.get("pages", {})
    metadata = result.get("metadata", {})
    signals  = extract_signals_enhanced(pages, metadata)

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
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result, debug = enrich_one_row_enhanced("Test Company", "https://www.example.com")
    import json
    print(json.dumps(result, indent=2))
    new_fields = [
        "elm_kw_workforce",
        "elm_score_workforce",
        "elm_sector_cluster",
        "elm_sitemap_found",
        "elm_sitemap_relevant_paths",
    ]
    print("\n--- New fields ---")
    for f in new_fields:
        print(f"  {f}: {result.get(f)}")
