"""
caller_prep_cockpit.py — Layer 3: mYngle Caller Prep Cockpit
Reads the 'Caller Prep Input' sheet from an Opportunity Radar export.
No LLM calls. No external API calls. No database. File-based prototype.
"""

import io
import re
import pandas as pd
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="mYngle Caller Prep Cockpit",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
SHEET_NAME = "Caller Prep Input"

CALL_OUTCOMES = [
    "Not called yet",
    "No answer",
    "Reached wrong person",
    "Reached right person",
    "Interested",
    "Not interested",
    "Call back later",
    "Meeting booked",
    "Bad data",
]

FIT_OPTIONS = ["Not sure", "Strong", "Medium", "Weak"]

DEFAULT_DQ1 = (
    "Are international communication or language skills currently part of "
    "your team development plans?"
)
DEFAULT_DQ2 = (
    "Which teams would benefit most from stronger Business English or "
    "cross-border communication support?"
)

REC_PRIORITY = [
    "Call now",
    "Call this month",
    "Call before budget cycle",
    "Manual research needed",
    "Monitor",
    "Low priority",
    "Internal / exclude",
]

TIER_COLORS = {
    "🥇 Hot": "#D6E4F7",
    "🥈 Warm": "#D9EAD3",
    "🥉 Cool": "#FCE5CD",
    "❄️ Pass": "#F4CCCC",
    "Hot": "#D6E4F7",
    "Warm": "#D9EAD3",
    "Cool": "#FCE5CD",
    "Pass": "#F4CCCC",
}

REC_BADGE = {
    "Call now":               "🔵",
    "Call this month":        "🟢",
    "Call before budget cycle": "🟡",
    "Manual research needed": "🟠",
    "Monitor":                "⚪",
    "Low priority":           "⚫",
    "Internal / exclude":     "🔴",
}


# ── Session state helpers ─────────────────────────────────────────────────────
def ss(key, default=None):
    return st.session_state.get(key, default)


def ss_set(key, value):
    st.session_state[key] = value


def get_edits() -> dict:
    """Return the edits dict, initialised if absent."""
    if "cpc_edits" not in st.session_state:
        st.session_state["cpc_edits"] = {}
    return st.session_state["cpc_edits"]


def get_edit(company_key: str, field: str, default=None):
    return get_edits().get(company_key, {}).get(field, default)


def set_edit(company_key: str, field: str, value):
    edits = get_edits()
    if company_key not in edits:
        edits[company_key] = {}
    edits[company_key][field] = value


# ── Data helpers ──────────────────────────────────────────────────────────────
def _safe(val, default=""):
    """Return a clean string or default for display."""
    if val is None:
        return default
    if isinstance(val, float) and val != val:  # NaN
        return default
    s = str(val).strip()
    return default if s.lower() in ("", "nan", "none") else s


def _num(val, default=None):
    """Return float or default."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _read_cpi_sheet(file_bytes: bytes) -> pd.DataFrame:
    xl = pd.ExcelFile(io.BytesIO(file_bytes))
    if SHEET_NAME not in xl.sheet_names:
        return None
    df = xl.parse(SHEET_NAME, dtype=str)
    df = df.fillna("")
    return df


def _company_key_col(df: pd.DataFrame) -> str:
    """Return best column to use as a unique row key."""
    for c in ("company_key", "lead_id", "company_name"):
        if c in df.columns:
            return c
    return df.columns[0]


_SEARCH_COLS = (
    "company_name", "domain", "preferred_buyer_route",
    "trigger_type", "call_recommendation",
)


def _filter_df(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    out = df.copy()

    # Standard column → value(s) filters
    for col, vals in filters.items():
        if col.startswith("_"):
            continue  # handled separately
        if col not in out.columns:
            continue
        if isinstance(vals, list) and vals:
            out = out[out[col].isin(vals)]
        elif isinstance(vals, str) and vals.strip():
            out = out[out[col].str.lower().str.contains(vals.strip().lower(), na=False)]

    # Caution filter
    caution_sel = filters.get("_caution", [])
    if caution_sel and "caution_note" in out.columns:
        has_caution = out["caution_note"].str.strip() != ""
        if "⚠️ Has caution" in caution_sel and "✅ No caution" not in caution_sel:
            out = out[has_caution]
        elif "✅ No caution" in caution_sel and "⚠️ Has caution" not in caution_sel:
            out = out[~has_caution]

    # Free-text search across multiple columns
    search = filters.get("_search", "").strip()
    if search:
        mask = pd.Series([False] * len(out), index=out.index)
        for col in _SEARCH_COLS:
            if col in out.columns:
                mask |= out[col].str.lower().str.contains(search.lower(), na=False)
        out = out[mask]

    return out


def _urls_from_text(text: str) -> list[str]:
    """Extract http/https URLs from a text blob."""
    return re.findall(r"https?://[^\s,;\"'<>]+", text)


def _build_export_df(df: pd.DataFrame, edits: dict) -> pd.DataFrame:
    """Merge original data with session-state edits."""
    out = df.copy()
    key_col = _company_key_col(df)
    for i, row in out.iterrows():
        ckey = str(row.get(key_col, i))
        row_edits = edits.get(ckey, {})
        for field, val in row_edits.items():
            out.at[i, field] = val
    return out


# ── Styled helpers ────────────────────────────────────────────────────────────
def tier_badge(tier: str) -> str:
    color = TIER_COLORS.get(tier, "#EEEEEE")
    return f'<span style="background:{color};padding:2px 8px;border-radius:4px;font-size:0.85em">{tier or "—"}</span>'


def rec_badge(rec: str) -> str:
    icon = REC_BADGE.get(rec, "•")
    return f"{icon} {rec}" if rec else "—"


def field_row(label: str, value: str, mono: bool = False):
    """Render a label/value pair."""
    if not value:
        return
    style = "font-family:monospace;font-size:0.9em" if mono else ""
    st.markdown(
        f"<div style='margin-bottom:4px'><span style='color:#666;font-size:0.8em'>{label}</span><br>"
        f"<span style='{style}'>{value}</span></div>",
        unsafe_allow_html=True,
    )


def section_header(title: str):
    st.markdown(f"### {title}")
    st.markdown("<hr style='margin:4px 0 12px 0;border-color:#ddd'>", unsafe_allow_html=True)


# ── KPI Cards ─────────────────────────────────────────────────────────────────
def _kpi_card(label: str, value, color: str = "#0B4A92"):
    st.markdown(
        f"""<div style='background:#f7f9fc;border-left:4px solid {color};
        padding:10px 14px;border-radius:4px;margin-bottom:4px'>
        <div style='font-size:1.6em;font-weight:700;color:{color}'>{value}</div>
        <div style='font-size:0.78em;color:#555'>{label}</div></div>""",
        unsafe_allow_html=True,
    )


# ── Sidebar: upload only ──────────────────────────────────────────────────────
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 📞 Caller Prep Cockpit")
        st.markdown("**mYngle · Layer 3**")
        st.markdown("---")

        uploaded = st.file_uploader(
            "Upload Opportunity Radar export (.xlsx)",
            type=["xlsx"],
            key="cpc_upload",
        )
        if uploaded is not None:
            raw = uploaded.read()
            loaded = _read_cpi_sheet(raw)
            if loaded is None:
                st.error(
                    f'"{SHEET_NAME}" sheet not found. '
                    "Please upload an Opportunity Radar export."
                )
            else:
                ss_set("cpc_df", loaded)
                ss_set("cpc_raw", raw)
                st.success(f"{len(loaded)} companies loaded.")

        df = ss("cpc_df")
        if df is not None:
            st.markdown("---")
            st.caption(f"**{len(df)} companies** in this export.")
            active = ss("cpc_filter_epoch", 0)
            if active:
                st.caption("⚙️ Filters active")


# ── Inline filter bar ─────────────────────────────────────────────────────────
def render_filters(df: pd.DataFrame) -> dict:
    """Render compact filter controls; return active filters dict."""
    epoch = ss("cpc_filter_epoch", 0)

    def _opts(col):
        if col not in df.columns:
            return []
        vals = sorted(df[col].dropna().unique().tolist())
        return [v for v in vals if v not in ("", "nan")]

    with st.container():
        st.markdown("#### Filters")
        row1 = st.columns([2, 2, 2, 2, 1])
        row2 = st.columns([2, 2, 2, 4])

        with row1[0]:
            rec_sel = st.multiselect(
                "Call recommendation", _opts("call_recommendation"),
                key=f"f_rec_{epoch}",
            )
        with row1[1]:
            tier_sel = st.multiselect(
                "Commercial tier", _opts("commercial_tier"),
                key=f"f_tier_{epoch}",
            )
        with row1[2]:
            bucket_sel = st.multiselect(
                "Recency bucket", _opts("recency_bucket"),
                key=f"f_bucket_{epoch}",
            )
        with row1[3]:
            route_sel = st.multiselect(
                "Preferred buyer route", _opts("preferred_buyer_route"),
                key=f"f_route_{epoch}",
            )
        with row1[4]:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button("Reset", key=f"f_reset_{epoch}", use_container_width=True):
                ss_set("cpc_filter_epoch", epoch + 1)
                st.rerun()

        with row2[0]:
            conf_sel = st.multiselect(
                "Domain confidence", _opts("domain_match_confidence"),
                key=f"f_conf_{epoch}",
            )
        with row2[1]:
            caution_sel = st.multiselect(
                "Has caution",
                ["⚠️ Has caution", "✅ No caution"],
                key=f"f_caution_{epoch}",
            )
        with row2[2]:
            review_sel = st.multiselect(
                "Needs domain review", _opts("needs_domain_review"),
                key=f"f_review_{epoch}",
            )
        with row2[3]:
            search = st.text_input(
                "Search company, domain, trigger, recommendation, buyer route",
                key=f"f_search_{epoch}",
                placeholder="Type to search…",
            )

    filters: dict = {}
    if rec_sel:
        filters["call_recommendation"] = rec_sel
    if tier_sel:
        filters["commercial_tier"] = tier_sel
    if bucket_sel:
        filters["recency_bucket"] = bucket_sel
    if route_sel:
        filters["preferred_buyer_route"] = route_sel
    if conf_sel:
        filters["domain_match_confidence"] = conf_sel
    if review_sel:
        filters["needs_domain_review"] = review_sel
    if caution_sel:
        filters["_caution"] = caution_sel
    if search.strip():
        filters["_search"] = search.strip()
    return filters


# ── Dashboard ─────────────────────────────────────────────────────────────────
def render_dashboard(df: pd.DataFrame):
    total = len(df)
    recs = df["call_recommendation"] if "call_recommendation" in df.columns else pd.Series(dtype=str)

    def _count(val):
        return int((recs == val).sum()) if len(recs) else 0

    call_now      = _count("Call now")
    call_month    = _count("Call this month")
    call_budget   = _count("Call before budget cycle")
    manual        = _count("Manual research needed")
    monitor       = _count("Monitor")
    low_prio      = _count("Low priority")

    fresh = 0
    if "recency_bucket" in df.columns:
        fresh = int(df["recency_bucket"].isin(["Fresh", "Recent-ish"]).sum())

    caution = 0
    if "caution_note" in df.columns:
        caution = int((df["caution_note"].str.strip() != "").sum())

    st.markdown("### Pipeline overview")
    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    with c1: _kpi_card("Total companies", total, "#0B4A92")
    with c2: _kpi_card("Call now", call_now, "#1565C0")
    with c3: _kpi_card("Call this month", call_month, "#2E7D32")
    with c4: _kpi_card("Before budget cycle", call_budget, "#F57F17")
    with c5: _kpi_card("Manual research", manual, "#E65100")
    with c6: _kpi_card("Monitor", monitor, "#607D8B")
    with c7: _kpi_card("Fresh / Recent triggers", fresh, "#00695C")
    with c8: _kpi_card("Has caution note", caution, "#B71C1C")
    st.markdown("")


# ── Company list ──────────────────────────────────────────────────────────────
LIST_COLS = [
    "company_name", "domain", "commercial_tier", "commercial_fit_score",
    "call_recommendation", "opportunity_score", "trigger_type",
    "recency_bucket", "preferred_buyer_route",
]


def render_company_list(df: pd.DataFrame, filters: dict) -> str | None:
    """Render filtered list inside expander; return selected company_key or None."""
    filtered = _filter_df(df, filters)
    n_filtered, n_total = len(filtered), len(df)

    expander_label = (
        f"Company list — {n_filtered} of {n_total} companies"
        + (" (filters active)" if n_filtered < n_total else "")
    )

    with st.expander(expander_label, expanded=True):
        if filtered.empty:
            st.warning("No companies match the current filters. Try resetting the filters.")
        else:
            show_cols = [c for c in LIST_COLS if c in filtered.columns]
            if "caution_note" in filtered.columns:
                filtered = filtered.copy()
                filtered["⚠"] = filtered["caution_note"].apply(
                    lambda x: "⚠️" if str(x).strip() not in ("", "nan") else ""
                )
                show_cols.append("⚠")

            st.dataframe(
                filtered[show_cols].reset_index(drop=True),
                use_container_width=True,
                height=min(40 * n_filtered + 40, 360),
            )

    if filtered.empty:
        return None

    key_col = _company_key_col(df)
    name_col = "company_name" if "company_name" in filtered.columns else key_col
    options = filtered[name_col].tolist()

    selected_name = st.selectbox(
        "Select company for call prep",
        options,
        key="cpc_selected_name",
    )
    match = filtered[filtered[name_col] == selected_name]
    if match.empty:
        return None
    return str(match.iloc[0].get(key_col, selected_name))


# ── Company prep view ─────────────────────────────────────────────────────────
def render_company_prep(row: pd.Series, ckey: str):
    name = _safe(row.get("company_name"), "Unknown company")
    st.markdown(f"## 📋 {name}")

    tab_a, tab_b, tab_c, tab_d, tab_e, tab_f, tab_li, tab_contacts, tab_call = st.tabs([
        "A · Summary",
        "B · Why this company",
        "C · Why now",
        "D · Evidence",
        "E · Buyer route",
        "F · Caution",
        "LinkedIn helper",
        "Contacts",
        "Call prep & outcome",
    ])

    # ── A. Company summary ────────────────────────────────────────────────────
    with tab_a:
        section_header("Company summary")
        c1, c2, c3 = st.columns(3)
        with c1:
            field_row("Company", _safe(row.get("company_name")))
            field_row("Domain", _safe(row.get("domain")))
            field_row("Country", _safe(row.get("country")))
            field_row("City", _safe(row.get("city")))
            field_row("Industry", _safe(row.get("industry")))
            field_row("Employees", _safe(row.get("employee_range")))
        with c2:
            tier = _safe(row.get("commercial_tier"))
            rec  = _safe(row.get("call_recommendation"))
            st.markdown(f"**Commercial tier:** {tier_badge(tier)}", unsafe_allow_html=True)
            st.markdown(f"**Call recommendation:** {rec_badge(rec)}")
            score = _num(row.get("commercial_fit_score"))
            if score is not None:
                st.metric("Commercial fit score", f"{score:.2f}")
            opp = _num(row.get("opportunity_score"))
            if opp is not None:
                st.metric("Opportunity score", f"{opp:.2f}")
        with c3:
            field_row("Lead ID",      _safe(row.get("lead_id")))
            field_row("Company key",  _safe(row.get("company_key")))
            field_row("Domain confidence", _safe(row.get("domain_match_confidence")))
            field_row("Domain mismatch?",  _safe(row.get("possible_domain_mismatch")))
            field_row("Needs review?",     _safe(row.get("needs_domain_review")))

    # ── B. Why this company ───────────────────────────────────────────────────
    with tab_b:
        section_header("Why this company")
        field_row("Why relevant for mYngle",        _safe(row.get("icp_why_relevant")))
        field_row("Likely training interest",        _safe(row.get("icp_likely_training_interest")))
        field_row("Top positive signals",            _safe(row.get("top_positive_signals")))
        field_row("Gaps / missing signals",          _safe(row.get("gaps_missing_signals")))
        field_row("ICP buying signals",              _safe(row.get("icp_buying_signals")))
        field_row("Likely buyer function",           _safe(row.get("icp_potential_buyer_function")))

    # ── C. Why now ────────────────────────────────────────────────────────────
    with tab_c:
        section_header("Why now")
        c1, c2 = st.columns(2)
        with c1:
            field_row("Trigger type",          _safe(row.get("trigger_type")))
            field_row("Trigger date",          _safe(row.get("trigger_date")))
            age = _safe(row.get("trigger_age_days"))
            field_row("Trigger age (days)",    age)
            field_row("Recency bucket",        _safe(row.get("recency_bucket")))
            field_row("Likely buying window",  _safe(row.get("likely_buying_window")))
            field_row("Date confidence",       _safe(row.get("date_confidence")))
        with c2:
            field_row("Why now",               _safe(row.get("why_now")))
            field_row("Evidence summary",      _safe(row.get("evidence_summary")))
            field_row("Recency note",          _safe(row.get("recency_note")))

    # ── D. Evidence ───────────────────────────────────────────────────────────
    with tab_d:
        section_header("Evidence")
        field_row("ICP evidence",         _safe(row.get("icp_evidence")))
        field_row("Raw source summary",   _safe(row.get("raw_source_summary")))

        urls_text = _safe(row.get("top_source_urls"))
        if urls_text:
            urls = _urls_from_text(urls_text)
            if urls:
                st.markdown("**Top sources:**")
                for u in urls:
                    st.markdown(f"- [{u}]({u})")
            else:
                field_row("Top source URLs", urls_text)

        c1, c2 = st.columns(2)
        with c1:
            field_row("Source count",      _safe(row.get("source_count")))
        with c2:
            field_row("Latest source date", _safe(row.get("latest_source_date")))

    # ── E. Buyer route ────────────────────────────────────────────────────────
    with tab_e:
        section_header("Buyer route")
        field_row("Preferred buyer route",   _safe(row.get("preferred_buyer_route")))
        field_row("Backup buyer route",      _safe(row.get("backup_buyer_route")))
        field_row("Suggested title searches", _safe(row.get("suggested_title_searches")))
        field_row("Suggested opener",        _safe(row.get("suggested_opener")))

    # ── F. Caution ────────────────────────────────────────────────────────────
    with tab_f:
        section_header("Caution")
        caution = _safe(row.get("caution_note"))
        reason  = _safe(row.get("reason_not_to_call_now"))
        missing = _safe(row.get("missing_evidence"))
        no_over = _safe(row.get("what_not_to_overclaim"))

        if caution:
            st.warning(f"**Caution:** {caution}")
        else:
            st.success("No caution notes for this company.")

        if reason:
            st.error(f"**Reason not to call now:** {reason}")
        if missing:
            st.info(f"**Missing evidence:** {missing}")
        if no_over:
            st.warning(f"**Do not overclaim:** {no_over}")

        if not any([caution, reason, missing, no_over]):
            st.caption("No caution data available.")

    # ── LinkedIn helper ───────────────────────────────────────────────────────
    with tab_li:
        section_header("LinkedIn / Sales Navigator helper")
        st.info(
            "Search manually in LinkedIn or Sales Navigator. "
            "Do not rely on automatic extraction."
        )

        company  = _safe(row.get("company_name"))
        domain   = _safe(row.get("domain"))
        route    = _safe(row.get("preferred_buyer_route"))

        # Build ordered search suggestion list
        _standard_functions = [
            "HR", "People & Culture", "Learning & Development",
            "Talent", "International HR", "Training",
        ]
        search_suggestions: list[str] = []
        if route:
            search_suggestions.append(f"{company} {route}")
        for fn in _standard_functions:
            candidate = f"{company} {fn}"
            if candidate not in search_suggestions:
                search_suggestions.append(candidate)

        st.markdown("**Manual search suggestions** — copy each into LinkedIn / Sales Navigator:")
        for s in search_suggestions:
            st.code(s, language=None)

        titles_raw = _safe(row.get("suggested_title_searches"))
        if titles_raw:
            st.markdown("**AI-suggested title searches from Opportunity Radar:**")
            for t in re.split(r"[,;\n]+", titles_raw):
                t = t.strip()
                if t:
                    st.code(f"{company} {t}", language=None)

        st.markdown("---")
        st.markdown(
            "**[Open LinkedIn Sales Navigator people search]"
            "(https://www.linkedin.com/sales/search/people)**  "
            "*(log in first)*"
        )
        if domain:
            st.markdown(
                f"**[Find {company} on LinkedIn]"
                f"(https://www.linkedin.com/search/results/companies/?keywords={domain})**"
            )

    # ── Contacts ──────────────────────────────────────────────────────────────
    with tab_contacts:
        section_header("Manual contact capture")
        st.caption(
            "Enter contacts found via LinkedIn or Sales Navigator. "
            "Data is stored in this session and included in the export."
        )

        for n in (1, 2, 3):
            with st.expander(f"Contact {n}", expanded=(n == 1)):
                c1, c2 = st.columns(2)
                with c1:
                    name_val = st.text_input(
                        "Name", key=f"c{n}_name_{ckey}",
                        value=get_edit(ckey, f"contact_{n}_name", ""),
                    )
                    title_val = st.text_input(
                        "Title", key=f"c{n}_title_{ckey}",
                        value=get_edit(ckey, f"contact_{n}_title", ""),
                    )
                    fit_val = st.selectbox(
                        "Fit", FIT_OPTIONS, key=f"c{n}_fit_{ckey}",
                        index=FIT_OPTIONS.index(get_edit(ckey, f"contact_{n}_fit", "Not sure")),
                    )
                with c2:
                    li_val = st.text_input(
                        "LinkedIn URL", key=f"c{n}_li_{ckey}",
                        value=get_edit(ckey, f"contact_{n}_linkedin_url", ""),
                    )
                    email_val = st.text_input(
                        "Email", key=f"c{n}_email_{ckey}",
                        value=get_edit(ckey, f"contact_{n}_email", ""),
                    )
                    notes_val = st.text_area(
                        "Notes", key=f"c{n}_notes_{ckey}",
                        value=get_edit(ckey, f"contact_{n}_notes", ""),
                        height=68,
                    )
                # Persist every interaction
                set_edit(ckey, f"contact_{n}_name",         name_val)
                set_edit(ckey, f"contact_{n}_title",        title_val)
                set_edit(ckey, f"contact_{n}_linkedin_url", li_val)
                set_edit(ckey, f"contact_{n}_email",        email_val)
                set_edit(ckey, f"contact_{n}_fit",          fit_val)
                set_edit(ckey, f"contact_{n}_notes",        notes_val)

    # ── Call prep & outcome ───────────────────────────────────────────────────
    with tab_call:
        section_header("Final call prep")

        # Read-only context
        opener_default   = _safe(row.get("suggested_opener"))
        dq1_default      = _safe(row.get("discovery_question_1"), DEFAULT_DQ1)
        dq2_default      = _safe(row.get("discovery_question_2"), DEFAULT_DQ2)
        evidence_default = _safe(row.get("evidence_summary"))
        caution_default  = _safe(row.get("caution_note"))
        overclaim_default= _safe(row.get("what_not_to_overclaim"))

        c1_name  = get_edit(ckey, "contact_1_name", "")
        c1_title = get_edit(ckey, "contact_1_title", "")

        with st.container(border=True):
            st.markdown("**Company angle**")
            st.markdown(f"> {_safe(row.get('icp_why_relevant'), '—')}")
            if c1_name:
                st.markdown(f"**Contact:** {c1_name}" + (f" · {c1_title}" if c1_title else ""))
            st.markdown(f"**Opener:** {opener_default or '—'}")
            st.markdown(f"**Discovery Q1:** {dq1_default}")
            st.markdown(f"**Discovery Q2:** {dq2_default}")
            if evidence_default:
                st.markdown(f"**Evidence to mention:** {evidence_default}")
            if caution_default:
                st.warning(f"⚠️ {caution_default}")
            if overclaim_default:
                st.error(f"🚫 Do not overclaim: {overclaim_default}")

        st.markdown("---")
        st.markdown("**Editable call fields**")

        final_opener = st.text_area(
            "Final opener (edit if needed)",
            key=f"final_opener_{ckey}",
            value=get_edit(ckey, "final_opener", opener_default),
            height=80,
        )
        set_edit(ckey, "final_opener", final_opener)

        call_notes = st.text_area(
            "Call notes",
            key=f"call_notes_{ckey}",
            value=get_edit(ckey, "call_notes", ""),
            height=100,
        )
        set_edit(ckey, "call_notes", call_notes)

        outcome_default = get_edit(ckey, "call_outcome", "Not called yet")
        outcome_idx = CALL_OUTCOMES.index(outcome_default) if outcome_default in CALL_OUTCOMES else 0
        call_outcome = st.selectbox(
            "Call outcome",
            CALL_OUTCOMES,
            index=outcome_idx,
            key=f"call_outcome_{ckey}",
        )
        set_edit(ckey, "call_outcome", call_outcome)

        next_step = st.text_input(
            "Next step",
            key=f"next_step_{ckey}",
            value=get_edit(ckey, "next_step", ""),
        )
        set_edit(ckey, "next_step", next_step)

        sales_feedback = st.text_area(
            "Sales feedback",
            key=f"sales_feedback_{ckey}",
            value=get_edit(ckey, "sales_feedback", ""),
            height=80,
        )
        set_edit(ckey, "sales_feedback", sales_feedback)


# ── Export ────────────────────────────────────────────────────────────────────
def render_export(df: pd.DataFrame):
    section_header("Export")
    edits = get_edits()
    export_df = _build_export_df(df, edits)

    c1, c2 = st.columns(2)
    with c1:
        csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇ Download CSV",
            data=csv_bytes,
            file_name="caller_prep_export.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            export_df.to_excel(writer, index=False, sheet_name="Caller Prep Export")
        buf.seek(0)
        st.download_button(
            "⬇ Download Excel",
            data=buf.getvalue(),
            file_name="caller_prep_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Sidebar handles upload only; re-read df after so first-upload rerun works.
    render_sidebar()
    df: pd.DataFrame | None = ss("cpc_df")

    if df is None:
        st.markdown("# 📞 mYngle Caller Prep Cockpit")
        st.markdown(
            "Upload an **Opportunity Radar export** (.xlsx) in the sidebar to get started.\n\n"
            "This tool helps cold callers prepare Business English and language training "
            "conversations with international companies. It reads the **Caller Prep Input** "
            "sheet produced by the Opportunity Radar (Layer 2)."
        )
        st.info("No file uploaded yet.")
        return

    render_dashboard(df)
    st.markdown("---")

    filters = render_filters(df)
    st.markdown("")

    selected_key = render_company_list(df, filters)
    st.markdown("---")

    if selected_key:
        key_col = _company_key_col(df)
        match = df[df[key_col] == selected_key]
        if not match.empty:
            row = match.iloc[0]
            render_company_prep(row, selected_key)
            st.markdown("---")

    render_export(df)


if __name__ == "__main__":
    main()
