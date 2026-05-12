"""
Lusha Batch Enrichment App
Reads an Excel file, enriches each row with Lusha company data,
and lets you download the result as CSV or Excel.
"""

import io
import json
import re
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# ── Constants ─────────────────────────────────────────────────────────────────
LUSHA_ENDPOINT = "https://api.lusha.com/company"
CACHE_DIR = Path("lusha_json_cache")

# Fields we extract from a Lusha company response (no personal data)
LUSHA_FIELDS = [
    "lusha_company_name",
    "lusha_domain",
    "lusha_industry",
    "lusha_sub_industry",
    "lusha_employee_count",
    "lusha_employee_range",
    "lusha_country",
    "lusha_city",
    "lusha_state",
    "lusha_revenue",
    "lusha_description",
    "lusha_linkedin_url",
    "lusha_specialties",
    "lusha_technologies",
    "lusha_founded_year",
    "enrichment_status",
    "match_confidence",
]

# Keywords used for auto-detecting the company-name column
COMPANY_NAME_HINTS = ["company", "bedrijf", "organisation", "organization", "naam", "name", "account"]
DOMAIN_HINTS = ["domain", "website", "url", "web", "site", "domein"]
# ─────────────────────────────────────────────────────────────────────────────


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_domain(raw: str) -> str:
    """Strip protocol, www and trailing slashes from a URL."""
    if not raw or not isinstance(raw, str):
        return ""
    d = raw.strip().lower()
    d = re.sub(r"^https?://", "", d)
    d = re.sub(r"^www\.", "", d)
    d = d.split("/")[0].strip()
    return d


def safe_filename(text: str) -> str:
    """Convert arbitrary text to a filesystem-safe filename stem."""
    text = unicodedata.normalize("NFKD", str(text))
    text = re.sub(r"[^\w\s\-.]", "", text)
    text = re.sub(r"\s+", "_", text).strip("_")
    return text[:120] or "unknown"


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def detect_columns(df: pd.DataFrame):
    """
    Return (name_col, domain_col) by scoring each column header.
    Returns None for domain_col if no good match is found.
    """
    cols = df.columns.tolist()
    col_lower = [c.lower() for c in cols]

    def best_match(hints):
        scores = []
        for i, cl in enumerate(col_lower):
            score = max(similarity(cl, h) for h in hints)
            scores.append((score, i))
        best_score, best_idx = max(scores)
        return (cols[best_idx], best_score)

    name_col, name_score = best_match(COMPANY_NAME_HINTS)
    domain_col, domain_score = best_match(DOMAIN_HINTS)

    # Require a minimum score; don't return same column for both
    detected_domain = domain_col if domain_score >= 0.55 and domain_col != name_col else None
    detected_name = name_col if name_score >= 0.45 else None
    return detected_name, detected_domain


def load_cache(cache_key: str) -> dict | None:
    path = CACHE_DIR / f"{safe_filename(cache_key)}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def save_cache(cache_key: str, data: dict) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{safe_filename(cache_key)}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Lusha API calls ───────────────────────────────────────────────────────────

def _api_get(params: dict, api_key: str) -> dict:
    """Raw GET against the Lusha /company endpoint."""
    resp = requests.get(
        LUSHA_ENDPOINT,
        headers={"api_key": api_key, "Content-Type": "application/json"},
        params=params,
        timeout=12,
    )
    resp.raise_for_status()
    return resp.json()


def lusha_by_domain(domain: str, api_key: str) -> dict:
    return _api_get({"domain": domain}, api_key)


def lusha_by_name(name: str, api_key: str) -> dict:
    return _api_get({"name": name}, api_key)


# ── Field extraction ──────────────────────────────────────────────────────────

def extract_company_fields(raw: dict) -> dict:
    """Pull only firmographic (non-personal) fields from a Lusha response."""
    company = raw.get("company", raw)

    # Location: try headquarters, then location dict, then flat fields
    hq = company.get("headquarters") or company.get("location") or {}
    if not isinstance(hq, dict):
        hq = {}

    country = (
        hq.get("country")
        or company.get("country")
        or ""
    )
    city = hq.get("city") or company.get("city") or ""
    state = hq.get("state") or company.get("state") or ""

    # Specialties and technologies may be lists or strings
    specialties = company.get("specialties") or company.get("specialities") or ""
    if isinstance(specialties, list):
        specialties = ", ".join(str(s) for s in specialties)

    technologies = company.get("technologies") or company.get("techStack") or ""
    if isinstance(technologies, list):
        technologies = ", ".join(str(t) for t in technologies)

    return {
        "lusha_company_name": company.get("name") or company.get("companyName") or "",
        "lusha_domain": company.get("domain") or company.get("website") or "",
        "lusha_industry": company.get("industry") or "",
        "lusha_sub_industry": company.get("subIndustry") or "",
        "lusha_employee_count": company.get("employeeCount") or "",
        "lusha_employee_range": company.get("employeeCountRange") or "",
        "lusha_country": country,
        "lusha_city": city,
        "lusha_state": state,
        "lusha_revenue": company.get("revenue") or company.get("revenueBand") or "",
        "lusha_description": company.get("description") or "",
        "lusha_linkedin_url": company.get("linkedinUrl") or company.get("linkedin") or "",
        "lusha_specialties": specialties,
        "lusha_technologies": technologies,
        "lusha_founded_year": company.get("foundedYear") or "",
    }


def empty_lusha_fields(status: str, confidence: str) -> dict:
    fields = {k: "" for k in LUSHA_FIELDS}
    fields["enrichment_status"] = status
    fields["match_confidence"] = confidence
    return fields


# ── Per-row enrichment logic ──────────────────────────────────────────────────

def enrich_row(
    company_name: str,
    raw_domain: str,
    api_key: str,
    delay: float,
) -> dict:
    """
    Enrich one row. Returns a dict with all LUSHA_FIELDS populated.
    Strategy:
      1. If domain present → try domain lookup (cache first)
      2. Else → try company-name lookup (cache first)
    """
    domain = clean_domain(raw_domain)
    company_name = str(company_name).strip() if company_name else ""

    # ── Try domain path ────────────────────────────────────────────────────
    if domain:
        cache_key = f"domain_{domain}"
        cached = load_cache(cache_key)
        if cached is not None:
            fields = extract_company_fields(cached)
            fields["enrichment_status"] = "cached"
            fields["match_confidence"] = "high"
            return fields

        try:
            time.sleep(delay)
            raw = lusha_by_domain(domain, api_key)
            save_cache(cache_key, raw)
            fields = extract_company_fields(raw)
            fields["enrichment_status"] = "enriched_by_domain"
            fields["match_confidence"] = "high"
            return fields
        except requests.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else 0
            if status_code == 404:
                pass  # fall through to name lookup
            else:
                fields = empty_lusha_fields("api_error", "no_match")
                fields["lusha_description"] = f"HTTP {status_code}"
                return fields
        except Exception as e:
            fields = empty_lusha_fields("api_error", "no_match")
            fields["lusha_description"] = str(e)
            return fields

    # ── Try company-name path ──────────────────────────────────────────────
    if not company_name:
        return empty_lusha_fields("no_match", "no_match")

    cache_key = f"name_{company_name}"
    cached = load_cache(cache_key)
    if cached is not None:
        fields = extract_company_fields(cached)
        returned_name = fields.get("lusha_company_name", "")
        confidence = "medium" if similarity(company_name, returned_name) >= 0.6 else "low"
        fields["enrichment_status"] = "cached"
        fields["match_confidence"] = confidence
        return fields

    try:
        time.sleep(delay)
        raw = lusha_by_name(company_name, api_key)
        save_cache(cache_key, raw)
        fields = extract_company_fields(raw)
        returned_name = fields.get("lusha_company_name", "")
        confidence = "medium" if similarity(company_name, returned_name) >= 0.6 else "low"
        fields["enrichment_status"] = "enriched_by_company_name"
        fields["match_confidence"] = confidence
        return fields
    except requests.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else 0
        if status_code == 404:
            return empty_lusha_fields("no_match", "no_match")
        fields = empty_lusha_fields("api_error", "no_match")
        fields["lusha_description"] = f"HTTP {status_code}"
        return fields
    except Exception as e:
        fields = empty_lusha_fields("api_error", "no_match")
        fields["lusha_description"] = str(e)
        return fields


# ── Download helpers ──────────────────────────────────────────────────────────

def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Enriched")
    return buf.getvalue()


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Lusha Batch Enrichment",
    page_icon="🏢",
    layout="wide",
)
st.title("🏢 Lusha Batch Enrichment")
st.caption("Upload een Excel-bestand, verrijk alle rijen met Lusha-bedrijfsdata en download het resultaat.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Instellingen")

    api_key = st.text_input(
        "Lusha API-key",
        type="password",
        help="Plak hier je Lusha API-key. Deze wordt niet opgeslagen.",
    )

    st.divider()

    max_rows = st.number_input(
        "Aantal rijen om te verwerken",
        min_value=1,
        max_value=10_000,
        value=10,
        step=1,
        help="Stel in op 10 om eerst te testen zonder credits te verspillen.",
    )

    delay_sec = st.slider(
        "Vertraging tussen API-calls (seconden)",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.1,
        help="Kleine pauze tussen calls om rate-limiting te voorkomen.",
    )

    st.divider()
    cache_count = len(list(CACHE_DIR.glob("*.json"))) if CACHE_DIR.exists() else 0
    st.metric("Gecachede bedrijven", cache_count)
    if st.button("Cache wissen", use_container_width=True):
        if CACHE_DIR.exists():
            for f in CACHE_DIR.glob("*.json"):
                f.unlink()
        st.success("Cache gewist.")
        st.rerun()

# ── File upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload Excel-bestand (.xlsx of .xls)",
    type=["xlsx", "xls"],
    help="Het bestand moet minimaal één kolom met bedrijfsnamen bevatten.",
)

if uploaded is None:
    st.info("Upload een Excel-bestand om te beginnen.")
    st.stop()

# ── Load and preview ──────────────────────────────────────────────────────────
try:
    df_raw = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Kan bestand niet lezen: {e}")
    st.stop()

st.success(f"Bestand geladen: **{len(df_raw)} rijen**, **{len(df_raw.columns)} kolommen**")

with st.expander("Voorbeeld van het bestand (eerste 5 rijen)"):
    st.dataframe(df_raw.head(), use_container_width=True)

# ── Column detection ──────────────────────────────────────────────────────────
auto_name_col, auto_domain_col = detect_columns(df_raw)
cols = df_raw.columns.tolist()

st.subheader("Kolomselectie")

col_left, col_right = st.columns(2)
with col_left:
    name_col = st.selectbox(
        "Kolom met bedrijfsnamen *",
        options=cols,
        index=cols.index(auto_name_col) if auto_name_col in cols else 0,
        help="Automatisch gedetecteerd. Pas aan als het niet klopt.",
    )
with col_right:
    domain_options = ["(geen)"] + cols
    default_domain_idx = (
        domain_options.index(auto_domain_col)
        if auto_domain_col and auto_domain_col in domain_options
        else 0
    )
    domain_col_choice = st.selectbox(
        "Kolom met domeinen / website-URLs (optioneel)",
        options=domain_options,
        index=default_domain_idx,
        help="Als beschikbaar, wordt het domein als eerste opzoekmethode gebruikt.",
    )

domain_col = domain_col_choice if domain_col_choice != "(geen)" else None

# Show detection confidence note
if auto_name_col:
    st.caption(f"Automatisch gedetecteerd: bedrijfsnaam = **{auto_name_col}**"
               + (f", domein = **{auto_domain_col}**" if auto_domain_col else ""))
else:
    st.caption("Kon geen bedrijfsnaamkolom automatisch detecteren. Selecteer er een hierboven.")

# ── Start enrichment ──────────────────────────────────────────────────────────
st.divider()

if not api_key:
    st.warning("Voer je Lusha API-key in de sidebar in om te beginnen.")
    st.stop()

n_to_process = min(int(max_rows), len(df_raw))
df_work = df_raw.head(n_to_process).copy()

st.info(
    f"**{n_to_process}** van de **{len(df_raw)}** rijen worden verwerkt. "
    "Verhoog 'Aantal rijen' in de sidebar voor meer."
)

run_button = st.button("▶ Start verrijking", type="primary", use_container_width=True)

if run_button:
    # ── Run enrichment ────────────────────────────────────────────────────
    progress_bar = st.progress(0, text="Bezig met verrijken…")
    status_placeholder = st.empty()
    results: list[dict] = []

    for i, (_, row) in enumerate(df_work.iterrows()):
        company_name = str(row.get(name_col, "")).strip()
        raw_domain = str(row.get(domain_col, "")).strip() if domain_col else ""

        status_placeholder.caption(
            f"Verwerken {i + 1}/{n_to_process}: **{company_name or '(leeg)'}**"
        )

        enriched = enrich_row(company_name, raw_domain, api_key, delay_sec)
        results.append(enriched)

        progress_bar.progress((i + 1) / n_to_process, text=f"{i + 1}/{n_to_process} rijen verwerkt")

    status_placeholder.empty()
    progress_bar.empty()

    # ── Merge results into original data ──────────────────────────────────
    df_enriched = df_work.copy()
    enriched_df = pd.DataFrame(results)
    for col in LUSHA_FIELDS:
        df_enriched[col] = enriched_df[col].values

    # ── Summary stats ─────────────────────────────────────────────────────
    status_counts = df_enriched["enrichment_status"].value_counts().to_dict()
    confidence_counts = df_enriched["match_confidence"].value_counts().to_dict()

    st.success(f"Verrijking voltooid voor **{n_to_process}** rijen.")

    stat_cols = st.columns(len(status_counts) or 1)
    status_labels = {
        "enriched_by_domain": "Via domein",
        "enriched_by_company_name": "Via naam",
        "cached": "Uit cache",
        "no_match": "Geen resultaat",
        "api_error": "API-fout",
    }
    for idx, (status, count) in enumerate(status_counts.items()):
        with stat_cols[idx % len(stat_cols)]:
            st.metric(status_labels.get(status, status), count)

    st.divider()

    # ── Results table ─────────────────────────────────────────────────────
    st.subheader("Verrijkte data")
    display_cols = list(df_work.columns) + ["enrichment_status", "match_confidence",
                                             "lusha_company_name", "lusha_domain",
                                             "lusha_industry", "lusha_country",
                                             "lusha_employee_range"]
    display_cols = [c for c in display_cols if c in df_enriched.columns]
    st.dataframe(df_enriched[display_cols], use_container_width=True, height=400)

    with st.expander("Alle verrijkte kolommen tonen"):
        st.dataframe(df_enriched, use_container_width=True)

    # ── Downloads ─────────────────────────────────────────────────────────
    st.subheader("Download resultaat")
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            label="⬇ Download als Excel (.xlsx)",
            data=df_to_excel_bytes(df_enriched),
            file_name="lusha_enriched.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with dl_col2:
        st.download_button(
            label="⬇ Download als CSV",
            data=df_to_csv_bytes(df_enriched),
            file_name="lusha_enriched.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.caption(f"JSON-cache opgeslagen in: `{CACHE_DIR.resolve()}`")
