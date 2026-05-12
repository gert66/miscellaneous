"""
Lusha Batch Enrichment App
──────────────────────────
Upload an Excel or CSV file, enrich each row with Lusha firmographic data,
and download the result as CSV or Excel.

Processing model: one row per Streamlit rerun so the Stop button works at any point.
All state lives in st.session_state.
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

# ── Constants ──────────────────────────────────────────────────────────────────
LUSHA_ENDPOINT = "https://api.lusha.com/company"
CACHE_DIR = Path("lusha_json_cache")

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

COMPANY_NAME_HINTS = ["company", "bedrijf", "organisation", "organization", "naam", "name", "account"]
DOMAIN_HINTS = ["domain", "website", "url", "web", "site", "domein"]


# ─────────────────────────────────────────────────────────────────────────────
# Pure utility functions  (all defined before any UI code)
# ─────────────────────────────────────────────────────────────────────────────

def clean_domain(raw: str) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    d = raw.strip().lower()
    d = re.sub(r"^https?://", "", d)
    d = re.sub(r"^www\.", "", d)
    d = d.split("/")[0].strip()
    if d in {"nan", "none", ""} or " " in d:
        return ""
    return d


def safe_filename(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = re.sub(r"[^\w\s\-.]", "", text)
    text = re.sub(r"\s+", "_", text).strip("_")
    return text[:120] or "unknown"


def str_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def detect_columns(df: pd.DataFrame):
    """Return (name_col, domain_col); domain_col may be None."""
    cols = df.columns.tolist()
    col_lower = [str(c).lower() for c in cols]

    def best_match(hints):
        scores = [(max(str_similarity(cl, h) for h in hints), i)
                  for i, cl in enumerate(col_lower)]
        best_score, best_idx = max(scores)
        return cols[best_idx], best_score

    name_col,   name_score   = best_match(COMPANY_NAME_HINTS)
    domain_col, domain_score = best_match(DOMAIN_HINTS)
    return (
        name_col   if name_score   >= 0.45 else None,
        domain_col if domain_score >= 0.55 and domain_col != name_col else None,
    )


# ── Cache ─────────────────────────────────────────────────────────────────────

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


def get_cache_count() -> int:
    return len(list(CACHE_DIR.glob("*.json"))) if CACHE_DIR.exists() else 0


# ── Lusha API ─────────────────────────────────────────────────────────────────
# [API CALL LOCATION] — _api_get / lusha_by_domain / lusha_by_name

def _api_get(params: dict, api_key: str) -> dict:
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


# ── Field extraction (firmographic only — no personal data) ───────────────────

def extract_company_fields(raw: dict) -> dict:
    company = raw.get("company", raw)
    hq = company.get("headquarters") or company.get("location") or {}
    if not isinstance(hq, dict):
        hq = {}

    def join_list(v):
        if isinstance(v, list):
            return ", ".join(str(x) for x in v)
        return v or ""

    return {
        "lusha_company_name":  company.get("name") or company.get("companyName") or "",
        "lusha_domain":        company.get("domain") or company.get("website") or "",
        "lusha_industry":      company.get("industry") or "",
        "lusha_sub_industry":  company.get("subIndustry") or "",
        "lusha_employee_count": company.get("employeeCount") or "",
        "lusha_employee_range": company.get("employeeCountRange") or "",
        "lusha_country":       hq.get("country") or company.get("country") or "",
        "lusha_city":          hq.get("city")    or company.get("city")    or "",
        "lusha_state":         hq.get("state")   or company.get("state")   or "",
        "lusha_revenue":       company.get("revenue") or company.get("revenueBand") or "",
        "lusha_description":   company.get("description") or "",
        "lusha_linkedin_url":  company.get("linkedinUrl") or company.get("linkedin") or "",
        "lusha_specialties":   join_list(company.get("specialties") or company.get("specialities")),
        "lusha_technologies":  join_list(company.get("technologies") or company.get("techStack")),
        "lusha_founded_year":  company.get("foundedYear") or "",
    }


def empty_fields(status: str, confidence: str, note: str = "") -> dict:
    fields = {k: "" for k in LUSHA_FIELDS}
    fields["enrichment_status"] = status
    fields["match_confidence"]  = confidence
    if note:
        fields["lusha_description"] = note
    return fields


# ── Per-row enrichment ────────────────────────────────────────────────────────
# [ENRICHMENT LOGIC LOCATION] — enrich_one_row

def enrich_one_row(company_name: str, raw_domain: str, api_key: str, delay: float) -> dict:
    domain       = clean_domain(raw_domain)
    company_name = str(company_name).strip() if company_name else ""

    # 1 — domain path
    if domain:
        cache_key = f"domain_{domain}"
        cached = load_cache(cache_key)
        if cached is not None:
            fields = extract_company_fields(cached)
            fields["enrichment_status"] = "cached"
            fields["match_confidence"]  = "high"
            return fields
        try:
            time.sleep(delay)
            raw = lusha_by_domain(domain, api_key)
            save_cache(cache_key, raw)
            fields = extract_company_fields(raw)
            fields["enrichment_status"] = "enriched_by_domain"
            fields["match_confidence"]  = "high"
            return fields
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if code != 404:
                return empty_fields("api_error", "no_match", f"HTTP {code}")
            # 404 → fall through to name lookup
        except Exception as e:
            return empty_fields("api_error", "no_match", str(e))

    # 2 — name path
    if not company_name:
        return empty_fields("no_match", "no_match")

    cache_key = f"name_{company_name}"
    cached = load_cache(cache_key)
    if cached is not None:
        fields = extract_company_fields(cached)
        conf   = "medium" if str_similarity(company_name, fields.get("lusha_company_name", "")) >= 0.6 else "low"
        fields["enrichment_status"] = "cached"
        fields["match_confidence"]  = conf
        return fields

    try:
        time.sleep(delay)
        raw = lusha_by_name(company_name, api_key)
        save_cache(cache_key, raw)
        fields = extract_company_fields(raw)
        conf   = "medium" if str_similarity(company_name, fields.get("lusha_company_name", "")) >= 0.6 else "low"
        fields["enrichment_status"] = "enriched_by_company_name"
        fields["match_confidence"]  = conf
        return fields
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else 0
        if code == 404:
            return empty_fields("no_match", "no_match")
        return empty_fields("api_error", "no_match", f"HTTP {code}")
    except Exception as e:
        return empty_fields("api_error", "no_match", str(e))


# ── Download helpers ──────────────────────────────────────────────────────────
# [OUTPUT FILE LOCATION] — df_to_excel_bytes / df_to_csv_bytes

def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Enriched")
    return buf.getvalue()


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


# ── Session-state helpers ─────────────────────────────────────────────────────

def ss(key, default=None):
    return st.session_state.get(key, default)


def ss_set(**kwargs):
    for k, v in kwargs.items():
        st.session_state[k] = v


def reset_processing():
    ss_set(processing=False, stop_requested=False, process_index=0,
           results=[], enrichment_done=False, df_enriched=None)


def build_and_finish(results: list, df_work: pd.DataFrame) -> None:
    """Assemble enriched DataFrame from raw results list and mark done."""
    df_out = df_work.copy().reset_index(drop=True)
    enriched_df = pd.DataFrame(results)
    for col in LUSHA_FIELDS:
        df_out[col] = enriched_df[col].values if col in enriched_df.columns else ""
    ss_set(processing=False, stop_requested=False,
           enrichment_done=True, df_enriched=df_out)
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Lusha Batch Enrichment", page_icon="🏢", layout="wide")
st.title("🏢 Lusha Batch Enrichment")
st.caption("Upload een Excel- of CSV-bestand, verrijk elke rij met Lusha-bedrijfsdata en download het resultaat.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Instellingen")
    api_key = st.text_input(
        "Lusha API-key",
        type="password",
        key="api_key_input",
        help="Plak hier je Lusha API-key. Wordt nergens opgeslagen.",
    )
    st.divider()
    max_rows = st.number_input(
        "Rijen om te verwerken",
        min_value=1, max_value=10_000, value=10, step=1,
        help="Standaard 10 — veilig om eerst te testen.",
    )
    delay_sec = st.slider(
        "Vertraging tussen calls (sec)",
        min_value=0.0, max_value=3.0, value=0.5, step=0.1,
    )
    st.divider()
    st.metric("Gecachede bedrijven", get_cache_count())
    if st.button("🗑 Cache wissen", use_container_width=True):
        if CACHE_DIR.exists():
            for f in CACHE_DIR.glob("*.json"):
                f.unlink()
        st.success("Cache gewist.")
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — File upload
# [FILE UPLOAD LOCATION] — st.file_uploader
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("Stap 1 · Bestand uploaden")
uploaded = st.file_uploader(
    "Sleep een bestand hierheen of klik om te uploaden (.xlsx · .xls · .csv)",
    type=["xlsx", "xls", "csv"],
    label_visibility="visible",
)

# Detect new / removed file and (re)load it
new_file_key = f"{uploaded.name}___{uploaded.size}" if uploaded else "__none__"
if new_file_key != ss("_file_key"):
    # New upload (or file removed) — reset everything
    ss_set(_file_key=new_file_key, df_raw=None, file_name=None,
           file_type=None, file_error=None)
    reset_processing()

    if uploaded is not None:
        try:
            fname = uploaded.name
            if fname.lower().endswith(".csv"):
                df_loaded = pd.read_csv(uploaded)
                ftype = "CSV"
            else:
                df_loaded = pd.read_excel(uploaded)
                ftype = "Excel"
            ss_set(df_raw=df_loaded, file_name=fname, file_type=ftype)
        except Exception as exc:
            ss_set(file_error=str(exc))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Status panel (always rendered, no st.stop())
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("Stap 2 · Status")

df_raw: pd.DataFrame | None = ss("df_raw")
file_error = ss("file_error")

c1, c2, c3, c4 = st.columns(4)
with c1:
    if uploaded:
        st.success("✅ Bestand geüpload")
        st.caption(ss("file_name", ""))
    else:
        st.error("❌ Geen bestand")

with c2:
    if file_error:
        st.error("❌ Leesfout")
        st.caption(file_error[:120])
    elif df_raw is not None:
        st.success("✅ Bestand gelezen")
        st.caption(f"{ss('file_type')} · {len(df_raw)} rijen · {len(df_raw.columns)} kol.")
    else:
        st.info("— Wacht op bestand")

with c3:
    if api_key:
        st.success("✅ API-key aanwezig")
    else:
        st.warning("⚠️ Geen API-key")
        st.caption("Voer in de sidebar in")

with c4:
    if df_raw is not None:
        n_sel = min(int(max_rows), len(df_raw))
        st.info(f"🔢 {n_sel} / {len(df_raw)} rijen")
    else:
        st.info("— Nog geen bestand")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Preview + column selection (only if file is loaded)
# ─────────────────────────────────────────────────────────────────────────────

name_col   = None
domain_col = None

if df_raw is not None:
    st.divider()
    st.subheader("Stap 3 · Voorbeeld & kolomselectie")

    with st.expander("Eerste 5 rijen van het bestand", expanded=True):
        st.dataframe(df_raw.head(), use_container_width=True)

    auto_name_col, auto_domain_col = detect_columns(df_raw)
    cols = df_raw.columns.tolist()

    sel_l, sel_r = st.columns(2)
    with sel_l:
        name_col = st.selectbox(
            "Kolom met bedrijfsnamen ✱",
            options=cols,
            index=cols.index(auto_name_col) if auto_name_col in cols else 0,
            help="Automatisch gedetecteerd — pas aan indien nodig.",
        )
    with sel_r:
        dom_opts = ["(geen)"] + cols
        def_dom  = (
            dom_opts.index(auto_domain_col)
            if auto_domain_col and auto_domain_col in dom_opts else 0
        )
        dom_choice = st.selectbox(
            "Kolom met website / domein (optioneel)",
            options=dom_opts,
            index=def_dom,
            help="Wordt gebruikt als eerste opzoekmethode.",
        )
    domain_col = dom_choice if dom_choice != "(geen)" else None

    note_parts = [f"bedrijfsnaam → **{auto_name_col}**"] if auto_name_col else ["bedrijfsnaamkolom niet automatisch gevonden"]
    if auto_domain_col:
        note_parts.append(f"domein → **{auto_domain_col}**")
    st.caption("Auto-detectie: " + ",  ".join(note_parts))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Start button (always rendered after file section)
# [START BUTTON LOCATION] — "Start Lusha enrichment now"
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.subheader("Stap 4 · Verrijking starten")

# Collect all blocking reasons so we can show them above the button
blocking: list[str] = []
if uploaded is None:
    blocking.append("Geen bestand geüpload.")
if file_error:
    blocking.append(f"Bestand kon niet worden gelezen: {file_error}")
if df_raw is not None and name_col is None:
    blocking.append("Geen bedrijfsnaamkolom geselecteerd.")
if not api_key:
    blocking.append("Geen Lusha API-key ingevoerd (voer in via de sidebar).")

n_to_process = min(int(max_rows), len(df_raw)) if df_raw is not None else 0
if df_raw is not None and n_to_process == 0:
    blocking.append("Nul rijen geselecteerd om te verwerken.")

currently_processing = ss("processing", False)
enrichment_done      = ss("enrichment_done", False)

if blocking and not currently_processing:
    for reason in blocking:
        st.warning(f"⚠️ {reason}")
elif not blocking and not currently_processing and not enrichment_done:
    st.info(f"Klaar om **{n_to_process}** rijen te verrijken. Klik op de knop hieronder.")

start_btn = st.button(
    "▶ Start Lusha enrichment now",
    type="primary",
    use_container_width=True,
    disabled=(bool(blocking) or currently_processing),
    key="start_button",
)

if start_btn and not blocking and not currently_processing:
    df_work = df_raw.head(n_to_process).copy()
    ss_set(
        processing=True,
        stop_requested=False,
        process_index=0,
        results=[],
        enrichment_done=False,
        df_enriched=None,
        _df_work=df_work,
        _name_col=name_col,
        _domain_col=domain_col,
        _n_to_process=n_to_process,
        _api_key=api_key,
        _delay=delay_sec,
    )
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Processing  (one row per rerun; Stop button works between rows)
# ─────────────────────────────────────────────────────────────────────────────

if ss("processing", False):
    idx         = ss("process_index", 0)
    results     = ss("results", [])
    df_work     = ss("_df_work")
    _name_col   = ss("_name_col")
    _domain_col = ss("_domain_col")
    _n          = ss("_n_to_process", 0)
    _api_key    = ss("_api_key", "")
    _delay      = ss("_delay", 0.5)

    # Stop button — rendered while processing; click triggers rerun with flag set
    if st.button("⏹ Stop after current row", key="stop_button"):
        ss_set(stop_requested=True)
        st.rerun()

    # Live progress
    pbar_text = f"Rij {idx} / {_n} verwerkt"
    st.progress(idx / _n if _n else 1.0, text=pbar_text)

    cnt_domain  = sum(1 for r in results if r.get("enrichment_status") == "enriched_by_domain")
    cnt_name    = sum(1 for r in results if r.get("enrichment_status") == "enriched_by_company_name")
    cnt_cached  = sum(1 for r in results if r.get("enrichment_status") == "cached")
    cnt_nomatch = sum(1 for r in results if r.get("enrichment_status") == "no_match")
    cnt_error   = sum(1 for r in results if r.get("enrichment_status") == "api_error")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Via domein",  cnt_domain)
    m2.metric("Via naam",    cnt_name)
    m3.metric("Uit cache",   cnt_cached)
    m4.metric("Geen match",  cnt_nomatch)
    m5.metric("API-fout",    cnt_error)

    if ss("stop_requested", False) or idx >= _n:
        # All rows done OR user pressed Stop → build result and finish
        build_and_finish(results, df_work)
    else:
        # Process exactly ONE row, then rerun to show updated progress
        row          = df_work.iloc[idx]
        company_name = str(row.get(_name_col, "")).strip()
        raw_domain   = str(row.get(_domain_col, "")).strip() if _domain_col else ""
        st.caption(f"⏳ Verwerken: rij {idx + 1}/{_n} — **{company_name or '(leeg)'}**")

        enriched = enrich_one_row(company_name, raw_domain, _api_key, _delay)
        results.append(enriched)
        ss_set(results=results, process_index=idx + 1)
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Results + downloads
# [OUTPUT FILE LOCATION] — df_to_excel_bytes / df_to_csv_bytes called here
# ─────────────────────────────────────────────────────────────────────────────

if ss("enrichment_done", False):
    df_enriched: pd.DataFrame = ss("df_enriched")
    processed = len(df_enriched)

    st.divider()
    stopped_early = ss("stop_requested", False)
    if stopped_early:
        st.warning(f"Verrijking gestopt na **{processed}** rijen. Gedeeltelijke resultaten beschikbaar.")
    else:
        st.success(f"✅ Verrijking voltooid — **{processed}** rijen verwerkt.")

    # Summary metrics
    status_counts = df_enriched["enrichment_status"].value_counts().to_dict()
    status_labels = {
        "enriched_by_domain":       "Via domein",
        "enriched_by_company_name": "Via naam",
        "cached":                   "Uit cache",
        "no_match":                 "Geen resultaat",
        "api_error":                "API-fout",
    }
    if status_counts:
        scols = st.columns(len(status_counts))
        for i, (s, c) in enumerate(status_counts.items()):
            scols[i].metric(status_labels.get(s, s), c)

    # Results table — compact view first, full view in expander
    orig_cols = [c for c in df_enriched.columns if c not in LUSHA_FIELDS]
    key_lusha = ["enrichment_status", "match_confidence",
                 "lusha_company_name", "lusha_domain",
                 "lusha_industry", "lusha_country", "lusha_employee_range"]
    preview_cols = orig_cols + [c for c in key_lusha if c in df_enriched.columns]

    st.subheader("Verrijkte data (samenvatting)")
    st.dataframe(df_enriched[preview_cols], use_container_width=True, height=400)

    with st.expander("Alle verrijkte kolommen"):
        st.dataframe(df_enriched, use_container_width=True)

    # Downloads
    st.subheader("Stap 5 · Download resultaat")
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "⬇ Download als Excel (.xlsx)",
            data=df_to_excel_bytes(df_enriched),
            file_name="lusha_enriched.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "⬇ Download als CSV",
            data=df_to_csv_bytes(df_enriched),
            file_name="lusha_enriched.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.caption(f"JSON-responses gecached in: `{CACHE_DIR.resolve()}`")

    if st.button("↺ Nieuwe verrijking starten", use_container_width=True, key="restart_btn"):
        reset_processing()
        st.rerun()
