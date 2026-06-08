"""
mYngle · Opportunity Radar
==========================
Identifies buying-window signals and contact routes for a list of companies.

Entry point:  streamlit run opportunity_radar.py
"""

import base64
import io
import pathlib

import pandas as pd
import streamlit as st

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="mYngle · Opportunity Radar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# HEADER  (logo + title — mirrors lead prioritizer layout)
# =============================================================================

_logo_path = pathlib.Path(__file__).parent / "mingle_local_final_fixed.png"
_logo_src  = (
    f"data:image/png;base64,{base64.b64encode(_logo_path.read_bytes()).decode()}"
    if _logo_path.exists() else ""
)
_img_tag = (
    f'<img src="{_logo_src}" class="brand-logo" alt="mYngle" />'
    if _logo_src else ""
)

st.markdown(
    f"""
    <style>
    .block-container {{
        max-width: 880px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }}

    div[data-testid="stMarkdownContainer"]:has(.brand-header) {{
        overflow: visible !important;
        margin-bottom: 1.0rem;
    }}

    .brand-header {{
        display: grid;
        grid-template-columns: 43% 57%;
        align-items: center;
        min-height: 140px;
        padding-top: 10px;
        padding-bottom: 6px;
        overflow: visible !important;
    }}

    .brand-title-block {{
        display: flex;
        align-items: center;
        justify-content: flex-start;
        overflow: visible !important;
    }}

    .brand-title {{
        font-size: 42px;
        font-weight: 700;
        color: #0B1F3A;
        line-height: 1.1;
        white-space: nowrap;
        margin: 0;
        padding: 0;
    }}

    .brand-logo-block {{
        display: flex;
        justify-content: flex-end;
        align-items: center;
        padding: 0;
        line-height: 0;
        overflow: visible !important;
    }}

    .brand-logo {{
        width: 430px;
        max-width: 100%;
        height: auto;
        display: block;
        object-fit: contain;
        object-position: center center;
        overflow: visible !important;
    }}
    </style>

    <div class="brand-header">
      <div class="brand-title-block">
        <span class="brand-title">Opportunity Radar</span>
      </div>
      <div class="brand-logo-block">
        {_img_tag}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# HELPERS
# =============================================================================

OUTPUT_COLUMNS = [
    "company_name",
    "domain",
    "country",
    "commercial_fit_score",
    "commercial_tier",
    "trigger_score",
    "buying_window_score",
    "contact_route_score",
    "opportunity_score",
    "call_recommendation",
    "why_now",
    "likely_buying_window",
    "preferred_buyer_route",
    "backup_buyer_route",
    "suggested_title_searches",
    "suggested_opener",
    "confidence_level",
]

# Candidate column names (lower-cased) → canonical key
_NAME_CANDIDATES = [
    "company_name", "company name", "company", "name", "organisation", "organization",
]
_DOMAIN_CANDIDATES = [
    "company_domain", "domain", "company website", "website", "url",
    "company url", "homepage",
]
_COUNTRY_CANDIDATES = [
    "country", "company_country", "company country", "hq country", "hq_country",
]
_SCORE_CANDIDATES = [
    "final_commercial_fit_score", "commercial_fit_score", "score",
]
_TIER_CANDIDATES = [
    "commercial_tier", "tier",
]


def _detect_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def _count_companies(df: pd.DataFrame, name_col: str | None) -> int:
    if name_col and name_col in df.columns:
        return int(
            df[name_col]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .nunique()
        )
    return len(df)


def _build_placeholder_output(df: pd.DataFrame, name_col: str | None, domain_col: str | None) -> pd.DataFrame:
    rows = []
    companies = (
        df[[c for c in [name_col, domain_col] if c]]
        .drop_duplicates()
        .reset_index(drop=True)
        if (name_col or domain_col)
        else df.head(len(df)).reset_index(drop=True)
    )
    for _, row in companies.iterrows():
        rec = {col: "" for col in OUTPUT_COLUMNS}
        rec["company_name"] = str(row[name_col]).strip() if name_col else ""
        rec["domain"]       = str(row[domain_col]).strip() if domain_col else ""
        rec["call_recommendation"] = "Pending scan"
        rec["confidence_level"]    = "—"
        rows.append(rec)
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def _to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Opportunity Radar")
    return buf.getvalue()


def _ts() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# =============================================================================
# SESSION STATE HELPERS
# =============================================================================

def ss(key: str, default=None):
    return st.session_state.get(key, default)


def ss_set(**kwargs):
    for k, v in kwargs.items():
        st.session_state[k] = v


def reset():
    ss_set(
        _or_df_raw=None,
        _or_file_name=None,
        _or_file_error=None,
        _or_file_key="__none__",
        _or_name_col=None,
        _or_domain_col=None,
        _or_n_companies=0,
        _or_done=False,
        _or_df_out=None,
    )


# =============================================================================
# STEP 1 — UPLOAD
# =============================================================================

st.divider()
st.subheader("Step 1 · Upload your file")

uploaded = st.file_uploader(
    "Drag and drop here, or click to browse  (.xlsx · .xls · .csv)",
    type=["xlsx", "xls", "csv"],
    label_visibility="collapsed",
)

new_key = f"{uploaded.name}___{uploaded.size}" if uploaded else "__none__"
if new_key != ss("_or_file_key", "__none__"):
    ss_set(
        _or_file_key=new_key,
        _or_df_raw=None,
        _or_file_name=None,
        _or_file_error=None,
        _or_name_col=None,
        _or_domain_col=None,
        _or_n_companies=0,
        _or_done=False,
        _or_df_out=None,
    )
    if uploaded is not None:
        try:
            fname = uploaded.name
            df_loaded = (
                pd.read_csv(uploaded)
                if fname.lower().endswith(".csv")
                else pd.read_excel(uploaded)
            )
            name_col   = _detect_col(df_loaded, _NAME_CANDIDATES)
            domain_col = _detect_col(df_loaded, _DOMAIN_CANDIDATES)
            n          = _count_companies(df_loaded, name_col)
            ss_set(
                _or_df_raw=df_loaded,
                _or_file_name=fname,
                _or_name_col=name_col,
                _or_domain_col=domain_col,
                _or_n_companies=n,
            )
        except Exception as exc:
            ss_set(_or_file_error=str(exc))

if ss("_or_file_error"):
    st.error(f"Could not read the file: {ss('_or_file_error')}")
elif ss("_or_df_raw") is not None:
    n = ss("_or_n_companies", 0)
    st.success(
        f"✓ **{ss('_or_file_name')}** loaded · "
        f"{n:,} {'company' if n == 1 else 'companies'} ready"
    )

# =============================================================================
# STEP 2 — START SCAN
# =============================================================================

_ready    = ss("_or_df_raw") is not None
_done     = ss("_or_done", False)
_df_out   = ss("_or_df_out")

if not _done:
    start_btn = st.button(
        "▶ Start radar scan",
        type="primary",
        use_container_width=True,
        disabled=not _ready,
        key="or_start_btn",
    )

    if start_btn and _ready:
        df_raw     = ss("_or_df_raw")
        name_col   = ss("_or_name_col")
        domain_col = ss("_or_domain_col")
        n          = ss("_or_n_companies", 0)

        with st.spinner("Scanning opportunities..."):
            df_out = _build_placeholder_output(df_raw, name_col, domain_col)

        ss_set(_or_done=True, _or_df_out=df_out)
        st.rerun()

# =============================================================================
# STEP 3 — RESULTS
# =============================================================================

if _done and _df_out is not None:
    processed = len(_df_out)
    st.success(
        f"✅ Ready · **{processed:,}** "
        f"{'company' if processed == 1 else 'companies'} scanned"
    )

    st.download_button(
        label="⬇ Download opportunity radar",
        data=_to_excel_bytes(_df_out),
        file_name=f"opportunity_radar_{_ts()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        type="primary",
    )

    st.divider()
    if st.button("↺ Start a new radar scan", use_container_width=True, key="or_restart_btn"):
        reset()
        st.rerun()
