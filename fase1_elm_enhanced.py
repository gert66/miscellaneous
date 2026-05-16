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
import time
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
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
        "page_title": "",
        "schema_org_found": False,
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

        page_title = soup.title.string.strip() if soup.title else ""

        schema_org_found = bool(
            soup.find("script", attrs={"type": "application/ld+json"})
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
            "page_title": page_title,
            "schema_org_found": schema_org_found,
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
                    "page_title": result["page_title"],
                    "schema_org_found": result["schema_org_found"],
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

    # ── Website data-quality score ────────────────────────────────────────────
    website_access_ok    = fetch_status == "ok"
    about_page_found     = any(s in fetched for s in ["/about", "/about-us"])
    careers_page_found   = any(s in fetched for s in ["/careers", "/jobs"])
    meta_desc_found      = bool(metadata.get("meta_description_found", False))
    og_found             = bool(metadata.get("og_tags_found", False))
    sitemap_found_flag   = bool(metadata.get("sitemap_found", False))
    linkedin_found       = bool(metadata.get("linkedin_link_found", False))

    website_data_quality_score = (
        3 * int(website_access_ok)
        + 2 * int(about_page_found)
        + 1 * int(careers_page_found)
        + 1 * int(meta_desc_found)
        + 1 * int(og_found)
        + 1 * int(sitemap_found_flag)
        + 1 * int(linkedin_found)
    )
    data_found     = website_data_quality_score >= 3
    metadata_found = meta_desc_found or og_found

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
        # Page identity
        "elm_page_title":                 metadata.get("page_title", ""),
        "elm_schema_org_found":           metadata.get("schema_org_found", False),
        # Data quality
        "elm_website_data_quality_score": website_data_quality_score,
        "elm_data_found":                 data_found,
        "elm_metadata_found":             metadata_found,
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
# CLI helpers
# ─────────────────────────────────────────────────────────────────────────────

_NAME_HINTS = ["company", "account", "name", "naam", "bedrijf"]
_URL_HINTS  = ["domain", "website", "url", "web", "site"]


def detect_columns(df: pd.DataFrame):
    """Return (name_col, url_col) by fuzzy-matching column headers."""
    def best_match(hints, columns):
        best_col, best_ratio = None, 0.0
        for col in columns:
            col_lower = col.lower()
            for hint in hints:
                ratio = SequenceMatcher(None, hint, col_lower).ratio()
                if ratio > best_ratio:
                    best_ratio, best_col = ratio, col
        return best_col if best_ratio >= 0.45 else None

    cols = list(df.columns)
    return best_match(_NAME_HINTS, cols), best_match(_URL_HINTS, cols)


def process_csv(
    filepath: str,
    list_type: str,
    name_col,
    url_col,
    delay: float,
) -> pd.DataFrame:
    """Enrich every row in a CSV file. Returns a DataFrame with ELM fields appended."""
    df = pd.read_csv(filepath)

    if name_col is None or url_col is None:
        auto_name, auto_url = detect_columns(df)
        name_col = name_col or auto_name
        url_col  = url_col  or auto_url

    if name_col is None:
        raise ValueError(
            f"Could not detect a company-name column in {filepath}. "
            "Pass --name-col explicitly."
        )
    if url_col is None:
        raise ValueError(
            f"Could not detect a URL column in {filepath}. "
            "Pass --url-col explicitly."
        )

    total = len(df)
    records = []

    try:
        for i, row in enumerate(df.itertuples(index=False), start=1):
            company = str(getattr(row, name_col, "") or "")
            raw_url = str(getattr(row, url_col,  "") or "")

            signals, _ = enrich_one_row_enhanced(company, raw_url)
            status_label = signals.get("elm_fetch_status", "failed")
            print(f"[{i:>4}/{total}] {company[:50]:<50} — {status_label}")

            combined = {col: getattr(row, col) for col in df.columns}
            combined.update(signals)
            combined["list_type"] = list_type
            records.append(combined)

            if i < total:
                time.sleep(delay)

    except KeyboardInterrupt:
        print("\nInterrupted — saving partial results…")

    return pd.DataFrame(records)


# =============================================================================
# Streamlit UI helpers
# =============================================================================

def _ss(key, default=None):
    return st.session_state.get(key, default)


def _ss_set(**kwargs):
    for k, v in kwargs.items():
        st.session_state[k] = v


def _reset_run():
    _ss_set(
        _elm2_running=False,
        _elm2_stop=False,
        _elm2_idx=0,
        _elm2_results=[],
        _elm2_done=False,
        _elm2_df_out=None,
    )


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


# =============================================================================
# Streamlit app
# =============================================================================

st.set_page_config(
    page_title="Fase 1 — Enhanced ELM Enrichment",
    page_icon="⚡",
    layout="wide",
)

st.title("Fase 1 — Enhanced ELM Enrichment")
st.caption(
    "Upload a CSV with company names and URLs. "
    "Each row is enriched with ELM Enhanced: "
    "page fetch, metadata extraction, keyword signals, sitemap scan, and sector detection."
)

# =============================================================================
# Upload
# =============================================================================

uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

new_file_key = f"{uploaded.name}___{uploaded.size}" if uploaded else "__none__"
if new_file_key != _ss("_elm2_file_key"):
    _ss_set(_elm2_file_key=new_file_key, _elm2_df_raw=None,
            _elm2_file_name=None, _elm2_file_error=None)
    _reset_run()
    if uploaded is not None:
        try:
            _ss_set(_elm2_df_raw=pd.read_csv(uploaded), _elm2_file_name=uploaded.name)
        except Exception as exc:
            _ss_set(_elm2_file_error=str(exc))

df_raw: pd.DataFrame | None = _ss("_elm2_df_raw")
file_error: str | None      = _ss("_elm2_file_error")

if file_error:
    st.error(f"Could not read the file: {file_error}")
elif df_raw is not None:
    st.success(
        f"**{_ss('_elm2_file_name')}** — {len(df_raw):,} rows, {len(df_raw.columns)} columns"
    )

# =============================================================================
# Column detection + Start button
# =============================================================================

name_col     = None
url_col      = None
n_to_process = 0

if df_raw is not None:
    auto_name, auto_url = detect_columns(df_raw)
    name_col = auto_name
    url_col  = auto_url

    if name_col or url_col:
        parts = []
        if name_col:
            parts.append(f"company name → **{name_col}**")
        if url_col:
            parts.append(f"URL → **{url_col}**")
        st.info("Auto-detected columns: " + ",  ".join(parts))
    else:
        st.warning("Could not auto-detect columns. Make sure your CSV has recognisable headers.")

    n_to_process = len(df_raw)

st.divider()
currently_running = _ss("_elm2_running", False)

blocking: list = []
if uploaded is None:
    blocking.append("No file uploaded yet.")
if file_error:
    blocking.append(f"File could not be read: {file_error}")
if df_raw is not None and name_col is None:
    blocking.append("Could not detect a company name column.")
if df_raw is not None and url_col is None:
    blocking.append("Could not detect a URL column.")

if blocking and not currently_running:
    for reason in blocking:
        st.warning(f"⚠️ {reason}")

start_btn = st.button(
    "▶ Start enrichment",
    type="primary",
    use_container_width=True,
    disabled=(bool(blocking) or currently_running),
    key="elm2_start_btn",
)

if start_btn and not blocking and not currently_running:
    _ss_set(
        _elm2_running=True,
        _elm2_stop=False,
        _elm2_idx=0,
        _elm2_results=[],
        _elm2_done=False,
        _elm2_df_out=None,
        _elm2_df_work=df_raw.copy().reset_index(drop=True),
        _elm2_name_col=name_col,
        _elm2_url_col=url_col,
        _elm2_n=n_to_process,
    )
    st.rerun()

# =============================================================================
# Processing loop — one row per Streamlit rerun
# =============================================================================

if _ss("_elm2_running", False):
    idx     = _ss("_elm2_idx", 0)
    results = _ss("_elm2_results", [])
    df_work = _ss("_elm2_df_work")
    _nc     = _ss("_elm2_name_col")
    _uc     = _ss("_elm2_url_col")
    _n      = _ss("_elm2_n", 0)

    if st.button("⏹ Stop after current row", key="elm2_stop_btn"):
        _ss_set(_elm2_stop=True)
        st.rerun()

    st.progress(idx / _n if _n else 1.0, text=f"Row {idx} of {_n}")

    cnt_ok      = sum(1 for r in results if r.get("elm_fetch_status") == "ok")
    cnt_partial = sum(1 for r in results if r.get("elm_fetch_status") == "partial")
    cnt_failed  = sum(1 for r in results if r.get("elm_fetch_status") == "failed")
    avg_score   = (
        sum(float(r.get("elm_score_overall_icp", 0) or 0) for r in results) / len(results)
        if results else 0.0
    )
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Fetched OK",    cnt_ok)
    mc2.metric("Partial",       cnt_partial)
    mc3.metric("Failed",        cnt_failed)
    mc4.metric("Processed",     len(results))
    mc5.metric("Avg ICP score", f"{avg_score:.1f}/10")

    if results:
        _preview_cols = [
            _nc, _uc,
            "elm_fetch_status", "elm_score_overall_icp",
            "elm_sector_cluster", "elm_score_language_complexity",
            "elm_hreflang_count", "elm_sitemap_found",
            "elm_website_data_quality_score",
        ]
        partial_df = df_work.iloc[: len(results)].copy().reset_index(drop=True)
        for col in _preview_cols[2:]:
            partial_df[col] = [r.get(col, "") for r in results]
        show_cols = [c for c in _preview_cols if c in partial_df.columns]
        st.dataframe(partial_df[show_cols], use_container_width=True)

        st.download_button(
            label=f"⬇ Download partial results ({len(results)} rows)",
            data=_df_to_csv_bytes(
                df_work.iloc[: len(results)].assign(
                    **{k: [r.get(k, "") for r in results] for k in results[0]}
                )
            ),
            file_name=f"elm2_partial_{_ts()}.csv",
            mime="text/csv",
            key=f"elm2_dl_partial_{idx}",
        )

    if _ss("_elm2_stop", False) or idx >= _n:
        df_out = df_work.iloc[: len(results)].copy().reset_index(drop=True)
        for k in (results[0] if results else {}):
            df_out[k] = [r.get(k, "") for r in results]
        _ss_set(_elm2_running=False, _elm2_stop=False, _elm2_done=True, _elm2_df_out=df_out)
        st.rerun()
    else:
        input_row    = df_work.iloc[idx]
        company_name = str(input_row.get(_nc, "") or "").strip()
        raw_url      = str(input_row.get(_uc, "") or "").strip()

        with st.status(
            f"Row {idx + 1} / {_n}: **{company_name or '(empty)'}**",
            expanded=False,
        ) as status_box:
            status_box.write("⚡ Fetching pages and extracting signals…")
            fields, _ = enrich_one_row_enhanced(company_name, raw_url)
            status_box.write(
                f"Done — fetch: **{fields.get('elm_fetch_status', '?')}** | "
                f"{int(fields.get('elm_total_chars', 0) or 0):,} chars | "
                f"ICP: **{fields.get('elm_score_overall_icp', '?')}/10** | "
                f"sector: {fields.get('elm_sector_cluster', '?')}"
            )

        results.append(fields)
        _ss_set(_elm2_results=results, _elm2_idx=idx + 1)
        st.rerun()

# =============================================================================
# Results
# =============================================================================

if _ss("_elm2_done", False):
    df_out: pd.DataFrame = _ss("_elm2_df_out")
    processed = len(df_out)

    st.divider()
    if _ss("_elm2_stop", False):
        st.warning(f"Stopped after **{processed}** rows.")
    else:
        st.success(f"✅ Done — **{processed:,}** rows enriched.")

    st.download_button(
        label="⬇ Download enriched CSV",
        data=_df_to_csv_bytes(df_out),
        file_name=f"elm2_enriched_{_ts()}.csv",
        mime="text/csv",
        use_container_width=True,
        type="primary",
        key="elm2_dl_final",
    )

    if "elm_fetch_status" in df_out.columns:
        status_counts = df_out["elm_fetch_status"].value_counts()
        cols_m = st.columns(len(status_counts) + 1)
        for i, (status, count) in enumerate(status_counts.items()):
            cols_m[i].metric(status.capitalize(), count)
        avg_icp = df_out["elm_score_overall_icp"].apply(
            lambda x: float(x) if str(x).replace(".", "").isdigit() else 0.0
        ).mean()
        cols_m[-1].metric("Avg ICP score", f"{avg_icp:.1f}/10")

    st.divider()
    st.dataframe(df_out, use_container_width=True)

    st.divider()
    if st.button("↺ Start over", use_container_width=True, key="elm2_restart"):
        _reset_run()
        _ss_set(_elm2_file_key=None, _elm2_df_raw=None)
        st.rerun()

