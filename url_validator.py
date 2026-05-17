"""
URL Validator & Auto-Corrector
==============================
Upload an Excel or CSV file with company names and URLs.
The app validates each URL, attempts auto-correction, and produces a
cleaned file with six new url_ columns ready for enrichment.

Processing model: one row per Streamlit rerun so the Stop button works.
"""

import io
import re
import time
from difflib import SequenceMatcher
from urllib.parse import urlparse, urlunparse

import pandas as pd
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_COMPANY_HINTS = ["company", "account", "organisation", "organization", "name", "naam", "bedrijf"]
_DOMAIN_HINTS  = ["domain", "website", "url", "web", "site", "domein"]

_UA = "Mozilla/5.0 (compatible; URLValidatorBot/1.0)"
_TIMEOUT = 10
_DDGO_API = "https://api.duckduckgo.com/?q={query}+official+website&format=json&no_redirect=1"

URL_FIELDS = [
    "url_status",
    "final_url",
    "redirect_target",
    "http_status_code",
    "correction_source",
    "correction_notes",
]

_STATUS_COLOR = {
    "ok":         "#1a7a3c",  # deep green
    "redirected": "#b35c00",  # burnt orange
    "corrected":  "#7a5c00",  # dark amber
    "dead":       "#8b1a1a",  # deep red
}

_STATUS_TEXT_COLOR = {
    "ok":         "#e6ffe6",
    "redirected": "#ffe8cc",
    "corrected":  "#fff4cc",
    "dead":       "#ffe6e6",
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def detect_columns(df: pd.DataFrame):
    cols = df.columns.tolist()
    col_lower = [str(c).lower() for c in cols]

    def best(hints):
        scores = [(max(_similarity(cl, h) for h in hints), i)
                  for i, cl in enumerate(col_lower)]
        score, idx = max(scores)
        return cols[idx], score

    name_col, ns = best(_COMPANY_HINTS)
    domain_col, ds = best(_DOMAIN_HINTS)
    return (
        name_col   if ns >= 0.45 else None,
        domain_col if ds >= 0.55 and domain_col != name_col else None,
    )


def _normalise_url(raw: str) -> str:
    """Ensure the URL has a scheme so requests can handle it."""
    raw = raw.strip()
    if not raw:
        return ""
    if not re.match(r"^https?://", raw, re.IGNORECASE):
        return "https://" + raw
    return raw


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().lstrip("www.")
    except Exception:
        return ""


def _get(url: str):
    """GET url; return (response, error_str). Follows redirects."""
    try:
        resp = requests.get(
            url,
            timeout=_TIMEOUT,
            headers={"User-Agent": _UA},
            allow_redirects=True,
        )
        return resp, None
    except requests.exceptions.SSLError:
        # retry without SSL verification as last resort
        try:
            resp = requests.get(
                url,
                timeout=_TIMEOUT,
                headers={"User-Agent": _UA},
                allow_redirects=True,
                verify=False,
            )
            return resp, None
        except Exception as exc:
            return None, str(exc)
    except Exception as exc:
        return None, str(exc)


def _duckduckgo_url(company_name: str) -> str | None:
    """Ask DuckDuckGo for the official website of company_name."""
    try:
        query = requests.utils.quote(company_name)
        resp = requests.get(
            f"https://api.duckduckgo.com/?q={query}+official+website&format=json&no_redirect=1",
            timeout=_TIMEOUT,
            headers={"User-Agent": _UA},
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        # AbstractURL is the best signal
        url = data.get("AbstractURL") or data.get("Redirect") or ""
        if url:
            return url
        # Fall back to first RelatedTopic URL
        for topic in data.get("RelatedTopics", []):
            u = topic.get("FirstURL", "")
            if u and "duckduckgo.com" not in u:
                return u
        return None
    except Exception:
        return None


def validate_url(original_url: str, company_name: str) -> dict:
    """
    Validate one URL through the four-step waterfall.
    Returns a dict with the six url_ fields.
    """
    result = {
        "url_status":       "dead",
        "final_url":        original_url,
        "redirect_target":  "",
        "http_status_code": "",
        "correction_source":"dead",
        "correction_notes": "",
    }

    raw = (original_url or "").strip()
    if not raw:
        result["correction_notes"] = "no URL provided"
        return result

    # ── Step 1: try URL as-is (normalised to have a scheme) ──────────────────
    url_as_is = _normalise_url(raw)
    resp, err = _get(url_as_is)
    if resp is not None and resp.status_code < 400:
        final = resp.url
        orig_dom = _domain(url_as_is)
        final_dom = _domain(final)
        result["http_status_code"] = resp.status_code
        result["final_url"] = final
        if orig_dom and final_dom and orig_dom != final_dom:
            result["url_status"] = "redirected"
            result["redirect_target"] = final
            result["correction_source"] = "original"
            result["correction_notes"] = f"redirected to {final_dom}"
        else:
            result["url_status"] = "ok"
            result["correction_source"] = "original"
        return result

    # ── Step 2: try adding https:// if it was missing ─────────────────────────
    if not re.match(r"^https?://", raw, re.IGNORECASE):
        url_https = "https://" + raw
        resp, err = _get(url_https)
        if resp is not None and resp.status_code < 400:
            final = resp.url
            result["http_status_code"] = resp.status_code
            result["final_url"] = final
            result["url_status"] = "corrected"
            result["correction_source"] = "https_added"
            result["correction_notes"] = "added https:// prefix"
            if _domain(url_https) != _domain(final):
                result["redirect_target"] = final
            return result

    # ── Step 3: try www. variant ──────────────────────────────────────────────
    parsed = urlparse(url_as_is)
    if not parsed.netloc.lower().startswith("www."):
        www_netloc = "www." + parsed.netloc
        url_www = urlunparse(parsed._replace(netloc=www_netloc))
        resp, err = _get(url_www)
        if resp is not None and resp.status_code < 400:
            final = resp.url
            result["http_status_code"] = resp.status_code
            result["final_url"] = final
            result["url_status"] = "corrected"
            result["correction_source"] = "www_added"
            result["correction_notes"] = "added www. prefix"
            if _domain(url_www) != _domain(final):
                result["redirect_target"] = final
            return result

    # ── Step 4: DuckDuckGo search ─────────────────────────────────────────────
    if company_name:
        ddg_url = _duckduckgo_url(company_name)
        if ddg_url:
            resp, err = _get(ddg_url)
            if resp is not None and resp.status_code < 400:
                final = resp.url
                result["http_status_code"] = resp.status_code
                result["final_url"] = final
                result["url_status"] = "corrected"
                result["correction_source"] = "search_corrected"
                result["correction_notes"] = f"found via DuckDuckGo: {_domain(final)}"
                if _domain(ddg_url) != _domain(final):
                    result["redirect_target"] = final
                return result

    # ── All steps failed ──────────────────────────────────────────────────────
    result["url_status"] = "dead"
    result["correction_source"] = "dead"
    result["correction_notes"] = "no working URL found"
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="URL Validator", page_icon="🔗", layout="wide")
st.title("🔗 URL Validator & Auto-Corrector")
st.caption("Validate and fix company URLs before enrichment.")

# ── Session-state initialisation ─────────────────────────────────────────────
for key, default in {
    "_running":      False,
    "_stop":         False,
    "_current_idx":  0,
    "_results":      [],   # list of dicts (one per processed row)
    "_df_raw":       None,
    "_name_col":     None,
    "_url_col":      None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def _reset():
    st.session_state["_running"]     = False
    st.session_state["_stop"]        = False
    st.session_state["_current_idx"] = 0
    st.session_state["_results"]     = []


# ── File upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "xls", "csv"])

if uploaded:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df_raw = pd.read_csv(uploaded)
        else:
            df_raw = pd.read_excel(uploaded)
        df_raw = df_raw.reset_index(drop=True)
    except Exception as exc:
        st.error(f"Could not read file: {exc}")
        st.stop()

    # Reset if a new file is uploaded
    if st.session_state["_df_raw"] is None or not df_raw.equals(st.session_state["_df_raw"]):
        _reset()
        st.session_state["_df_raw"] = df_raw

    df_raw = st.session_state["_df_raw"]
    auto_name, auto_url = detect_columns(df_raw)

    st.subheader("Column selection")
    col_a, col_b = st.columns(2)
    with col_a:
        name_col = st.selectbox(
            "Company name column",
            options=df_raw.columns.tolist(),
            index=df_raw.columns.tolist().index(auto_name) if auto_name else 0,
        )
    with col_b:
        url_col = st.selectbox(
            "URL column",
            options=df_raw.columns.tolist(),
            index=df_raw.columns.tolist().index(auto_url) if auto_url else 0,
        )

    st.session_state["_name_col"] = name_col
    st.session_state["_url_col"]  = url_col

    st.subheader("Data preview")
    st.dataframe(df_raw.head(10), use_container_width=True)

    # ── Controls ──────────────────────────────────────────────────────────────
    total = len(df_raw)
    ctrl_l, ctrl_r = st.columns([1, 1])
    with ctrl_l:
        start_btn = st.button(
            "▶ Start validation",
            disabled=st.session_state["_running"],
            type="primary",
        )
    with ctrl_r:
        stop_btn = st.button(
            "⏹ Stop",
            disabled=not st.session_state["_running"],
        )

    if start_btn:
        _reset()
        st.session_state["_running"] = True
        st.rerun()

    if stop_btn:
        st.session_state["_stop"] = True

    # ── Processing loop (one row per rerun) ───────────────────────────────────
    if st.session_state["_running"]:
        idx = st.session_state["_current_idx"]

        if st.session_state["_stop"] or idx >= total:
            st.session_state["_running"] = False
            if st.session_state["_stop"]:
                st.info(f"Stopped after {idx} row(s).")
            st.rerun()

        # Progress
        st.progress(idx / total, text=f"Processing row {idx + 1} of {total}…")

        # Process current row
        row = df_raw.iloc[idx]
        company = str(row.get(name_col, "") or "")
        url     = str(row.get(url_col, "")  or "")

        url_result = validate_url(url, company)
        st.session_state["_results"].append({**row.to_dict(), **url_result})
        st.session_state["_current_idx"] += 1

        # Live table
        if st.session_state["_results"]:
            live_df = pd.DataFrame(st.session_state["_results"])
            st.subheader(f"Live results ({len(live_df)} processed)")
            _display_cols = [url_col, "url_status", "final_url", "http_status_code",
                             "correction_source", "correction_notes"]
            _display_cols = [c for c in _display_cols if c in live_df.columns]

            def _row_style(row):
                status = row.get("url_status", "")
                bg = _STATUS_COLOR.get(status, "")
                fg = _STATUS_TEXT_COLOR.get(status, "")
                style = f"background-color: {bg}; color: {fg}" if bg else ""
                return [style for _ in row]

            st.dataframe(
                live_df[_display_cols].style.apply(_row_style, axis=1),
                use_container_width=True,
            )

        time.sleep(0.05)
        st.rerun()

    # ── Results panel (shown when not running and results exist) ──────────────
    results = st.session_state["_results"]
    if results and not st.session_state["_running"]:
        results_df = pd.DataFrame(results)

        # Metrics
        st.subheader("Summary")
        counts = results_df["url_status"].value_counts()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("✅ OK",         counts.get("ok", 0))
        m2.metric("↩️ Redirected", counts.get("redirected", 0))
        m3.metric("🔧 Corrected",  counts.get("corrected", 0))
        m4.metric("💀 Dead",       counts.get("dead", 0))

        dead_count = counts.get("dead", 0)
        if dead_count:
            st.warning(
                f"⚠️ {dead_count} URL(s) could not be resolved. "
                "These rows are highlighted in red — fix them manually before running enrichment."
            )

        # Full results table
        st.subheader("Full results")
        display_cols = [name_col, url_col] + URL_FIELDS
        display_cols = [c for c in display_cols if c in results_df.columns]

        def _row_style(row):
            status = row.get("url_status", "")
            bg = _STATUS_COLOR.get(status, "")
            fg = _STATUS_TEXT_COLOR.get(status, "")
            style = f"background-color: {bg}; color: {fg}" if bg else ""
            return [style for _ in row]

        st.dataframe(
            results_df[display_cols].style.apply(_row_style, axis=1),
            use_container_width=True,
            height=500,
        )

        # Dead URLs detail
        dead_df = results_df[results_df["url_status"] == "dead"]
        if not dead_df.empty:
            with st.expander(f"💀 Dead URLs ({len(dead_df)} rows) — needs manual fix"):
                st.dataframe(
                    dead_df[[name_col, url_col, "correction_notes"]],
                    use_container_width=True,
                )

        # Download
        st.subheader("Download")
        # Preserve original column order; append url_ columns
        orig_cols = df_raw.columns.tolist()
        new_cols  = [c for c in URL_FIELDS if c not in orig_cols]
        out_df    = results_df[orig_cols + new_cols] if all(c in results_df.columns for c in orig_cols) else results_df

        buf = io.BytesIO()
        out_df.to_excel(buf, index=False, engine="openpyxl")
        buf.seek(0)

        st.download_button(
            label="⬇️ Download cleaned Excel",
            data=buf,
            file_name="validated_urls.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

else:
    st.info("Upload an Excel or CSV file to get started.")
