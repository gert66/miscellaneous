"""
NNT Simulation v2 — interactive model playground
Proton vs Photon therapy: predicted vs true NNT

Run with:  streamlit run nnt_simulation/nnt_sim_v2.py
"""

import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── Chart config ──────────────────────────────────────────────────────────────

_CHART_CFG = {"displayModeBar": False}

# ── Default values ────────────────────────────────────────────────────────────

_DEFAULTS = {
    "n_patients":     5_000,
    "seed":           42,
    "gtv_mean":       50.0,
    "gtv_std":        20.0,
    "mhd_mean":       15.0,
    "mhd_std":         5.0,
    "gtv_mid":        50.0,
    "gtv_slope":      -0.1,
    "mhd_mid":        15.0,
    "mhd_slope":      -0.1,
    "proton_mode":    "Multiply by factor",
    "proton_delta":    5.0,
    "proton_factor":   0.5,
    "delta_thresh":    0.02,
    "hist_mode":      "Histogram",
    "noise_enabled":  False,
    "noise_sd":        1.0,
    "z_enabled":      False,
    "z_mean":          0.0,
    "z_sd":            1.0,
    "z_beta":          0.5,
}

# ── Math helpers ──────────────────────────────────────────────────────────────

def sigmoid(x, midpoint, slope):
    return 1.0 / (1.0 + np.exp(-slope * (x - midpoint)))


def logit_to_prob(logit):
    return 1.0 / (1.0 + np.exp(-logit))


def sample_truncnorm(rng, mean, std, n, lower=0.0):
    out = np.empty(n)
    filled = 0
    while filled < n:
        batch = rng.normal(loc=mean, scale=std, size=(n - filled) * 3 + 100)
        batch = batch[batch >= lower]
        take = min(len(batch), n - filled)
        out[filled : filled + take] = batch[:take]
        filled += take
    return out


def truncnorm_pdf(x_arr, mean, std, lower=0.0):
    alpha = (lower - mean) / std
    Z = 1.0 - 0.5 * (1.0 + math.erf(alpha / math.sqrt(2)))
    Z = max(Z, 1e-9)
    z = (x_arr - mean) / std
    pdf = np.exp(-0.5 * z ** 2) / (std * math.sqrt(2 * math.pi) * Z)
    return np.where(x_arr < lower, 0.0, pdf)


def apply_proton_mhd(mhd, mode, delta, factor):
    if mode == "Set to zero":
        return np.zeros_like(mhd)
    if mode == "Subtract fixed delta":
        return np.maximum(mhd - delta, 0.0)
    return mhd * factor


def _compute_auc(scores, labels):
    n_pos = float(labels.sum())
    n_neg = float(len(labels)) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(-scores)
    ls = labels[order].astype(float)
    tpr = np.concatenate([[0.0], np.cumsum(ls) / n_pos])
    fpr = np.concatenate([[0.0], np.cumsum(1.0 - ls) / n_neg])
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1])) / 2)


def _compute_roc(scores, labels):
    """Return (fpr, tpr) arrays sorted by ascending fpr, pure numpy."""
    n_pos = float(labels.sum())
    n_neg = float(len(labels)) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    order = np.argsort(-scores)
    ls = labels[order].astype(float)
    tpr = np.concatenate([[0.0], np.cumsum(ls) / n_pos, [1.0]])
    fpr = np.concatenate([[0.0], np.cumsum(1.0 - ls) / n_neg, [1.0]])
    return fpr, tpr


# ── Logistic regression fitting ───────────────────────────────────────────────

@st.cache_data
def fit_logistic_regression(gtv, mhd, outcomes):
    """
    Fit logistic regression on raw standardised GTV and MHD (zero mean, unit variance).
    Returns (model, scaler, error_string, debug_logs).
    debug_logs is always a dict; error_string is None on success.
    """
    import warnings as _warnings
    debug_logs = {}

    n_alive = int(outcomes.sum())
    n_dead  = int(len(outcomes) - n_alive)
    debug_logs["n_alive"] = n_alive
    debug_logs["n_dead"]  = n_dead

    if len(np.unique(outcomes)) < 2:
        msg = (
            f"Cannot fit: all {len(outcomes)} patients have the same outcome "
            f"({n_alive} alive, {n_dead} dead)."
        )
        debug_logs["error"] = msg
        return None, None, msg, debug_logs

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X = scaler.fit_transform(np.column_stack([gtv, mhd]))
        y = outcomes.astype(int)

        # ── Pre-fit diagnostics ───────────────────────────────────────────────
        debug_logs["X_shape"]    = tuple(X.shape)
        debug_logs["y_shape"]    = tuple(y.shape)
        debug_logs["n_zeros"]    = int((y == 0).sum())
        debug_logs["n_ones"]     = int((y == 1).sum())
        debug_logs["col0_mean"]  = float(X[:, 0].mean())
        debug_logs["col0_std"]   = float(X[:, 0].std())
        debug_logs["col1_mean"]  = float(X[:, 1].mean())
        debug_logs["col1_std"]   = float(X[:, 1].std())
        debug_logs["X_head"]     = X[:5].tolist()
        debug_logs["y_head"]     = y[:5].tolist()

        with _warnings.catch_warnings():
            _warnings.filterwarnings("error")
            lr = LogisticRegression(solver="lbfgs", max_iter=5000, C=1e6)
            lr.fit(X, y)

        b0    = float(lr.intercept_[0])
        b_gtv = float(lr.coef_[0][0])
        b_mhd = float(lr.coef_[0][1])

        # ── Post-fit diagnostics ──────────────────────────────────────────────
        fit_proba = lr.predict_proba(X)[:, 1]
        debug_logs["fit_proba_mean"] = float(fit_proba.mean())
        debug_logs["fit_proba_std"]  = float(fit_proba.std())
        debug_logs["b0"]    = b0
        debug_logs["b_gtv"] = b_gtv
        debug_logs["b_mhd"] = b_mhd

        # ── Manual single-patient verification (patient 0) ────────────────────
        ex_gtv_raw  = float(gtv[0])
        ex_mhd_raw  = float(mhd[0])
        ex_gtv_z    = float(X[0, 0])
        ex_mhd_z    = float(X[0, 1])
        ex_logit    = b0 + b_gtv * ex_gtv_z + b_mhd * ex_mhd_z
        ex_p_manual = float(1.0 / (1.0 + np.exp(-ex_logit)))
        ex_p_sklearn = float(lr.predict_proba(X[0:1, :])[0, 1])
        debug_logs["example"] = {
            "gtv_raw":    ex_gtv_raw,
            "mhd_raw":    ex_mhd_raw,
            "gtv_z":      ex_gtv_z,
            "mhd_z":      ex_mhd_z,
            "logit":      ex_logit,
            "p_manual":   ex_p_manual,
            "p_sklearn":  ex_p_sklearn,
        }

        return lr, scaler, None, debug_logs

    except Exception as e:
        debug_logs["exception"] = str(e)
        return None, None, f"Fitting failed: {e}", debug_logs


# ── Bin definitions ───────────────────────────────────────────────────────────

BIN_EDGES  = [0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.01]
BIN_LABELS = ["2–5 %", "5–10 %", "10–20 %", "20–40 %", "40–60 %", "60–80 %", "80–100 %"]


def bin_analysis(pred_delta, true_delta, threshold, fit_delta=None):
    sel   = pred_delta >= threshold
    n_sel = int(sel.sum())
    rows  = []
    for label, lo, hi in zip(BIN_LABELS, BIN_EDGES[:-1], BIN_EDGES[1:]):
        mask = sel & (pred_delta >= lo) & (pred_delta < hi)
        n    = int(mask.sum())
        if n == 0:
            nan_cols = ["Share (%)", "Mean Δ pred", "Pred NNT", "True ARR", "True NNT"]
            if fit_delta is not None:
                nan_cols.append("Fitted NNT")
            rows.append(dict(Bin=label, n=0, **{k: np.nan for k in nan_cols}))
            continue
        mean_pred = pred_delta[mask].mean()
        true_arr  = true_delta[mask].mean()
        row = {
            "Bin":         label,
            "n":           n,
            "Share (%)":   n / n_sel * 100 if n_sel > 0 else np.nan,
            "Mean Δ pred": mean_pred,
            "Pred NNT":    1.0 / mean_pred if mean_pred > 1e-9 else np.nan,
            "True ARR":    true_arr,
            "True NNT":    1.0 / true_arr  if true_arr  > 1e-9 else np.nan,
        }
        if fit_delta is not None:
            mean_fit = float(fit_delta[mask].mean())
            row["Fitted NNT"] = 1.0 / mean_fit if mean_fit > 1e-9 else np.nan
        rows.append(row)
    return pd.DataFrame(rows), n_sel


# ── Simulation ────────────────────────────────────────────────────────────────

@st.cache_data
def run_simulation(
    n, seed,
    gtv_mean, gtv_std,
    mhd_mean, mhd_std,
    gtv_mid, gtv_slope, mhd_mid, mhd_slope,
    proton_mode, proton_delta, proton_factor,
    noise_enabled=False, noise_sd=1.0,
    z_enabled=False, z_mean=0.0, z_sd=1.0, z_beta=0.5,
):
    rng    = np.random.default_rng(seed)
    gtv    = sample_truncnorm(rng, gtv_mean, gtv_std, n)
    mhd    = sample_truncnorm(rng, mhd_mean, mhd_std, n)
    mhd_pr = apply_proton_mhd(mhd, proton_mode, proton_delta, proton_factor)

    # ── Truth-world generation ────────────────────────────────────────────────
    # Step 1: direct survival probability per variable
    p_gtv    = sigmoid(gtv,    gtv_mid, gtv_slope)   # ranges 0–1
    p_mhd_ph = sigmoid(mhd,    mhd_mid, mhd_slope)
    p_mhd_pr = sigmoid(mhd_pr, mhd_mid, mhd_slope)

    # Step 2: combine multiplicatively
    p_ph = p_gtv * p_mhd_ph
    p_pr = p_gtv * p_mhd_pr

    # Noiseless, Z-free binary outcomes (AUC baseline reference)
    out_ph_base = (rng.random(n) < p_ph).astype(float)
    out_pr_base = (rng.random(n) < p_pr).astype(float)

    pred_delta = p_pr - p_ph  # always noiseless, always Z-free

    # Step 3: hidden factor Z (only if enabled)
    if z_enabled:
        z_rng         = np.random.default_rng(seed + 2000)
        z_vals        = z_rng.normal(z_mean, z_sd, n)
        p_z           = sigmoid(z_vals, 0.0, z_beta)   # expit(beta_z * Z)
        p_combined_ph = p_ph * p_z
        p_combined_pr = p_pr * p_z
    else:
        z_rng         = None
        z_vals        = None
        p_combined_ph = p_ph
        p_combined_pr = p_pr

    # Step 4: additive noise on logit (only if enabled)
    if noise_enabled:
        noise_rng    = np.random.default_rng(seed + 1000)
        eps_ph       = noise_rng.normal(0.0, noise_sd, n)
        eps_pr       = noise_rng.normal(0.0, noise_sd, n)
        p_clip_ph    = np.clip(p_combined_ph, 1e-9, 1.0 - 1e-9)
        p_clip_pr    = np.clip(p_combined_pr, 1e-9, 1.0 - 1e-9)
        logit_ph     = np.log(p_clip_ph / (1.0 - p_clip_ph))
        logit_pr     = np.log(p_clip_pr / (1.0 - p_clip_pr))
        p_final_ph   = logit_to_prob(logit_ph + eps_ph)
        p_final_pr   = logit_to_prob(logit_pr + eps_pr)
        draw_rng     = noise_rng
    else:
        noise_rng    = None
        p_final_ph   = p_combined_ph
        p_final_pr   = p_combined_pr
        draw_rng     = np.random.default_rng(seed + 2001) if z_enabled else None

    # Step 5: sample binary outcomes
    if z_enabled or noise_enabled:
        out_ph     = (draw_rng.random(n) < p_final_ph).astype(float)
        out_pr     = (draw_rng.random(n) < p_final_pr).astype(float)
        true_delta = out_pr - out_ph
    else:
        out_ph     = out_ph_base
        true_delta = out_pr_base - out_ph_base

    # out_ph_base: Z-free noiseless outcomes (AUC reference)
    # out_ph:      outcomes from full truth-world (Z + noise if enabled)
    # z_vals:      sampled Z values (None when Z is disabled)
    return gtv, mhd, p_ph, p_pr, pred_delta, true_delta, out_ph_base, out_ph, z_vals


# ── Dual-input widget ─────────────────────────────────────────────────────────

def _on_slider(key):
    v = st.session_state[f"_sl_{key}"]
    st.session_state[key]          = v
    st.session_state[f"_nu_{key}"] = v


def _on_num(key, lo, hi):
    v = float(st.session_state[f"_nu_{key}"])
    st.session_state[key]          = v
    st.session_state[f"_sl_{key}"] = float(np.clip(v, lo, hi))


def dual_param(label, key, lo, hi, step, fmt="%.2f", help_text=None):
    sl_key  = f"_sl_{key}"
    nu_key  = f"_nu_{key}"
    current = float(st.session_state[key])
    if sl_key not in st.session_state:
        st.session_state[sl_key] = float(np.clip(current, lo, hi))
    if nu_key not in st.session_state:
        st.session_state[nu_key] = current

    c_slider, c_num = st.columns([3, 1])
    with c_slider:
        st.slider(
            label,
            min_value=float(lo),
            max_value=float(hi),
            step=float(step),
            key=sl_key,
            help=help_text,
            on_change=_on_slider,
            args=(key,),
        )
    with c_num:
        st.number_input(
            label,
            min_value=float(lo),
            max_value=float(hi),
            step=float(step),
            format=fmt,
            key=nu_key,
            on_change=_on_num,
            args=(key, lo, hi),
            label_visibility="collapsed",
        )
    return float(st.session_state[key])


# ── Combined distribution + sigmoid figure ────────────────────────────────────

def make_combined_plot(
    data_a, mean, std, sig_mid, sig_slope,
    x_label, title,
    data_b=None, label_a="Photon", label_b="Proton",
    color_a="#4C8BF5", color_b="#00C9A7", color_sig="#E86510",
    hist_mode="Histogram", n_bins=40,
):
    x_lo    = max(0.0, mean - 4 * std)
    x_hi    = mean + 4 * std
    x_curve = np.linspace(x_lo, x_hi, 400)
    sig_y   = sigmoid(x_curve, sig_mid, sig_slope)

    fig = go.Figure()

    if hist_mode == "Histogram":
        fig.add_trace(go.Histogram(
            x=data_a, nbinsx=n_bins, name=label_a,
            marker_color=color_a, opacity=0.75, yaxis="y",
        ))
        if data_b is not None:
            fig.add_trace(go.Histogram(
                x=data_b, nbinsx=n_bins, name=label_b,
                marker_color=color_b, opacity=0.50, yaxis="y",
            ))
    else:
        x_den = np.linspace(x_lo, x_hi, 600)
        pdf   = truncnorm_pdf(x_den, mean, std)
        scale = len(data_a) * (x_hi - x_lo) / n_bins
        ra, ga, ba = int(color_a[1:3], 16), int(color_a[3:5], 16), int(color_a[5:7], 16)
        fig.add_trace(go.Scatter(
            x=x_den, y=pdf * scale, mode="lines", fill="tozeroy",
            line=dict(color=color_a, width=2.5),
            fillcolor=f"rgba({ra},{ga},{ba},0.15)",
            name=label_a, yaxis="y",
        ))
        if data_b is not None:
            fig.add_trace(go.Histogram(
                x=data_b, nbinsx=n_bins, name=label_b,
                marker_color=color_b, opacity=0.40, yaxis="y",
            ))

    fig.add_trace(go.Scatter(
        x=x_curve, y=sig_y, mode="lines",
        line=dict(color=color_sig, width=2.5),
        name="P(2-year survival)", yaxis="y2",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(title=x_label, range=[x_lo, x_hi]),
        yaxis=dict(
            title="Count" if hist_mode == "Histogram" else "Density (scaled)",
            side="left",
        ),
        yaxis2=dict(
            title="P(2-year survival)", side="right", overlaying="y",
            range=[0, 1], showgrid=False, tickformat=".1f",
        ),
        height=410,
        margin=dict(t=55, b=50, l=65, r=75),
        legend=dict(orientation="h", y=1.16, x=0, xanchor="left", font=dict(size=11)),
        barmode="overlay",
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.title("Simulation Setup")

        with st.expander("Population", expanded=True):
            st.number_input(
                "Number of patients",
                min_value=200,
                max_value=100_000,
                step=200,
                key="n_patients",
            )
            st.number_input(
                "Random seed",
                min_value=0,
                max_value=9_999,
                step=1,
                key="seed",
            )

        with st.expander("Proton effect", expanded=True):
            st.caption(
                "Choose how proton therapy reduces mean heart dose. "
                "Adjust the reduction magnitude in the main panel."
            )
            st.radio(
                "MHD reduction mode",
                ["Set to zero", "Subtract fixed delta", "Multiply by factor"],
                key="proton_mode",
            )

        with st.expander("Display", expanded=True):
            st.radio(
                "Distribution view",
                ["Histogram", "Density"],
                key="hist_mode",
            )

        with st.expander("Survival noise", expanded=False):
            st.checkbox(
                "Add random noise to survival mechanism",
                key="noise_enabled",
            )
            if st.session_state.get("noise_enabled", False):
                dual_param("Noise SD", "noise_sd", 0.1, 3.0, 0.1, "%.1f")

        with st.expander("Hidden prognostic factor Z", expanded=False):
            st.checkbox("Add hidden prognostic factor Z", key="z_enabled")
            if st.session_state.get("z_enabled", False):
                dual_param("Z mean",  "z_mean", -3.0, 3.0, 0.1, "%.1f")
                dual_param("Z SD",    "z_sd",    0.1, 3.0, 0.1, "%.1f")
                dual_param("β_Z",     "z_beta", -2.0, 2.0, 0.1, "%.1f")

    return (
        int(st.session_state["n_patients"]),
        int(st.session_state["seed"]),
        st.session_state["proton_mode"],
        st.session_state["hist_mode"],
        bool(st.session_state["noise_enabled"]),
        float(st.session_state["noise_sd"]),
        bool(st.session_state["z_enabled"]),
        float(st.session_state["z_mean"]),
        float(st.session_state["z_sd"]),
        float(st.session_state["z_beta"]),
    )


# ── Summary cards ─────────────────────────────────────────────────────────────

def render_summary_cards(n_patients, n_sel, p_ph, p_pr, pred_delta, true_delta, selected):
    ca, cb, cc, cd = st.columns(4)
    ca.metric("Mean P(OS2y) — photon",  f"{p_ph.mean():.3f}")
    cb.metric("Mean P(OS2y) — proton",  f"{p_pr.mean():.3f}")
    cc.metric("Mean predicted Δ (all)", f"{pred_delta.mean():.3f}")
    cd.metric("Mean true Δ (all)",      f"{true_delta.mean():.3f}")

    ce, cf, cg, ch = st.columns(4)
    ce.metric("Patients selected", f"{n_sel:,}",
              f"{n_sel / n_patients * 100:.1f} % of total")
    if n_sel > 0:
        mp = float(pred_delta[selected].mean())
        mt = float(true_delta[selected].mean())
        cf.metric("Mean predicted Δ (selected)", f"{mp:.3f}")
        cg.metric("Predicted NNT (selected)", f"{1/mp:.1f}" if mp > 1e-9 else "∞")
        ch.metric("True NNT (selected)",      f"{1/mt:.1f}" if mt > 1e-9 else "∞")
    else:
        cf.metric("Mean predicted Δ (selected)", "—")
        cg.metric("Predicted NNT (selected)", "—")
        ch.metric("True NNT (selected)", "—")


# ── Model playground ──────────────────────────────────────────────────────────

def render_playground(proton_mode, hist_mode, gtv, mhd, mhd_pr):
    st.subheader("Model Playground")
    st.markdown(
        "Adjust any parameter with the slider for quick exploration or type an "
        "exact value in the box on the right. All plots update on every change."
    )
    col_gtv, col_mhd = st.columns(2)

    with col_gtv:
        st.markdown("#### GTV")
        st.caption(
            "Gross tumour volume directly determines survival probability via a sigmoid. "
            "The orange curve shows P(2-year survival) as a function of GTV."
        )
        gtv_mean  = float(st.session_state["gtv_mean"])
        gtv_std   = float(st.session_state["gtv_std"])
        gtv_mid   = float(st.session_state["gtv_mid"])
        gtv_slope = float(st.session_state["gtv_slope"])
        st.plotly_chart(
            make_combined_plot(
                gtv, gtv_mean, gtv_std, gtv_mid, gtv_slope,
                x_label="GTV (cc)",
                title="GTV distribution & survival probability",
                color_a="#4C8BF5", color_sig="#E86510",
                hist_mode=hist_mode,
            ),
            use_container_width=True,
            config=_CHART_CFG,
        )
        st.markdown("**Distribution**")
        dual_param("Mean (cc)",  "gtv_mean",  0.0, 200.0, 1.0,  "%.1f")
        dual_param("Std  (cc)",  "gtv_std",   0.5,  80.0, 0.5,  "%.1f")
        st.markdown("**Sigmoid**")
        dual_param("Midpoint (cc)", "gtv_mid",   0.0, 200.0, 1.0,  "%.1f")
        dual_param(
            "Slope", "gtv_slope", -10.0, 10.0, 0.1, "%.3f",
            help_text="Negative → larger GTV reduces survival.",
        )
        st.caption("Midpoint controls where the transition happens. Slope controls how sharp it is.")

    with col_mhd:
        st.markdown("#### MHD")
        st.caption(
            "Mean heart dose is reduced by proton therapy. "
            "The teal overlay shows the proton MHD distribution."
        )
        mhd_mean  = float(st.session_state["mhd_mean"])
        mhd_std   = float(st.session_state["mhd_std"])
        mhd_mid   = float(st.session_state["mhd_mid"])
        mhd_slope = float(st.session_state["mhd_slope"])
        st.plotly_chart(
            make_combined_plot(
                mhd, mhd_mean, mhd_std, mhd_mid, mhd_slope,
                x_label="MHD (Gy)",
                title="MHD distribution & survival probability (photon + proton)",
                data_b=mhd_pr,
                label_a="Photon MHD", label_b="Proton MHD",
                color_a="#E8543A", color_b="#00C9A7", color_sig="#7B61FF",
                hist_mode=hist_mode,
            ),
            use_container_width=True,
            config=_CHART_CFG,
        )
        st.markdown("**Distribution**")
        dual_param("Mean (Gy)", "mhd_mean",  0.0, 60.0, 0.5,  "%.1f")
        dual_param("Std  (Gy)", "mhd_std",   0.5, 25.0, 0.5,  "%.1f")
        st.markdown("**Sigmoid**")
        dual_param("Midpoint (Gy)", "mhd_mid",   0.0, 60.0,  0.5,  "%.1f")
        dual_param(
            "Slope", "mhd_slope", -10.0, 10.0, 0.1, "%.3f",
            help_text="Negative → higher MHD reduces survival.",
        )
        st.caption("Midpoint controls where the transition happens. Slope controls how sharp it is.")

    st.markdown("**Proton MHD reduction**")
    if proton_mode == "Subtract fixed delta":
        dual_param("Reduction (Gy)", "proton_delta", 0.0, 40.0, 0.5, "%.1f")
    elif proton_mode == "Multiply by factor":
        dual_param(
            "Reduction factor", "proton_factor", 0.0, 1.0, 0.05, "%.2f",
            help_text="0 = abolish MHD entirely; 1 = no change.",
        )
    else:
        st.info("MHD is set to zero for all proton patients.")


# ── Patient selection ─────────────────────────────────────────────────────────

def render_selection(pred_delta, true_delta, delta_thresh, selected, n_sel, noise_enabled=False):
    st.subheader("Predicted Survival Benefit (Proton − Photon)")
    st.markdown(
        f"Δ = P_proton − P_photon per patient. Patients with Δ ≥ threshold are "
        f"selected for proton therapy. The current threshold is "
        f"**{delta_thresh:.3f}** ({delta_thresh * 100:.1f} %)."
    )
    dual_param(
        "Predicted Δ threshold",
        "delta_thresh",
        lo=0.0, hi=0.50, step=0.005, fmt="%.3f",
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Histogram(x=pred_delta, nbinsx=60, marker_color="#7B61FF"))
        fig.add_vline(x=delta_thresh, line_dash="dash", line_color="crimson",
                      annotation_text=f"Threshold {delta_thresh:.3f}",
                      annotation_position="top right")
        fig.update_layout(
            title="Predicted Δ — all patients",
            xaxis_title="Predicted Δ (P_proton − P_photon)", yaxis_title="Count",
            height=300, margin=dict(t=40, b=40, l=50, r=20),
        )
        st.plotly_chart(fig, use_container_width=True, config=_CHART_CFG)

    with col2:
        if n_sel > 0:
            fig = go.Figure(
                go.Histogram(x=pred_delta[selected], nbinsx=40, marker_color="#00C9A7")
            )
            fig.update_layout(
                title=f"Predicted Δ — selected patients (n = {n_sel:,})",
                xaxis_title="Predicted Δ", yaxis_title="Count",
                height=300, margin=dict(t=40, b=40, l=50, r=20),
            )
            st.plotly_chart(fig, use_container_width=True, config=_CHART_CFG)
        else:
            st.info(
                "No patients exceed the selection threshold. "
                "Lower the threshold to see selected patients."
            )

    if noise_enabled:
        st.caption(
            "True outcomes include additional random variation not visible to the model."
        )


# ── Bin analysis ──────────────────────────────────────────────────────────────

def render_bin_analysis(pred_delta, true_delta, delta_thresh, n_sel, fit_delta=None):
    st.subheader("Bin Analysis: Predicted NNT vs True NNT")
    st.markdown(
        "Selected patients are grouped into fixed bins by their predicted benefit. "
        "Within each bin, predicted NNT (1 / mean predicted Δ) is compared with "
        "true NNT (1 / mean true Δ from Monte Carlo outcomes). "
        "Close agreement indicates good calibration; divergence shows where the model "
        "over- or under-predicts benefit."
    )

    if n_sel == 0:
        st.info("Lower the selection threshold to populate the bin analysis.")
        return

    bin_df, _ = bin_analysis(pred_delta, true_delta, float(delta_thresh), fit_delta)

    def _fmt(v, spec):
        return format(v, spec) if pd.notna(v) else "—"

    display = bin_df.copy()
    display["Share (%)"]   = display["Share (%)"].map(lambda v: _fmt(v, ".1f"))
    display["Mean Δ pred"] = display["Mean Δ pred"].map(lambda v: _fmt(v, ".4f"))
    display["Pred NNT"]    = display["Pred NNT"].map(lambda v: _fmt(v, ".1f"))
    display["True ARR"]    = display["True ARR"].map(lambda v: _fmt(v, ".4f"))
    display["True NNT"]    = display["True NNT"].map(lambda v: _fmt(v, ".1f"))
    if "Fitted NNT" in display.columns:
        display["Fitted NNT"] = display["Fitted NNT"].map(lambda v: _fmt(v, ".1f"))
    st.dataframe(display, use_container_width=True, hide_index=True)

    chart_df = bin_df.dropna(subset=["Pred NNT", "True NNT"])
    if len(chart_df) >= 1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=chart_df["Bin"], y=chart_df["Pred NNT"],
            mode="lines+markers", name="Predicted NNT",
            line=dict(color="#7B61FF", width=2.5),
            marker=dict(size=9, symbol="circle"),
        ))
        fig.add_trace(go.Scatter(
            x=chart_df["Bin"], y=chart_df["True NNT"],
            mode="lines+markers", name="True NNT",
            line=dict(color="#00C9A7", width=2.5),
            marker=dict(size=9, symbol="diamond"),
        ))
        if fit_delta is not None and "Fitted NNT" in chart_df.columns:
            fit_chart = bin_df.dropna(subset=["Fitted NNT"])
            if len(fit_chart) >= 1:
                fig.add_trace(go.Scatter(
                    x=fit_chart["Bin"], y=fit_chart["Fitted NNT"],
                    mode="lines+markers", name="Fitted NNT",
                    line=dict(color="#E8543A", width=2.5, dash="dot"),
                    marker=dict(size=9, symbol="square"),
                ))
        fig.update_layout(
            title="Predicted NNT vs True NNT by predicted benefit bin",
            xaxis_title="Predicted benefit bin",
            yaxis_title="NNT",
            height=420,
            margin=dict(t=50, b=50, l=60, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True, config=_CHART_CFG)
    else:
        st.info(
            "Not enough populated bins to draw the comparison chart. "
            "Increase the number of patients or adjust the proton effect."
        )


# ── Noise: AUC summary panel ──────────────────────────────────────────────────

def render_noise_section(p_ph, out_ph_base, out_ph, noise_sd):
    st.subheader("Random Noise: Model Discrimination")

    auc_no_noise   = _compute_auc(p_ph, out_ph_base)
    auc_with_noise = _compute_auc(p_ph, out_ph)

    c1, c2, c3 = st.columns(3)
    c1.metric("Noise SD",          f"{noise_sd:.1f}")
    c2.metric("AUC without noise", f"{auc_no_noise:.3f}")
    c3.metric("AUC with noise",    f"{auc_with_noise:.3f}")

    st.info(
        "Noise represents unexplained variation outside the measured predictors. "
        "As noise increases, AUC falls. A good-looking predicted delta can still "
        "correspond to a much weaker true effect."
    )


# ── Hidden factor Z: diagnostics panel ───────────────────────────────────────

def render_z_section(z_vals, z_mean, z_sd, z_beta):
    st.info(
        "The true world includes an unmeasured prognostic factor Z. "
        "The fitted model does not see Z."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Z mean",  f"{z_mean:.2f}")
    c2.metric("Z SD",    f"{z_sd:.2f}")
    c3.metric("β_Z",     f"{z_beta:.2f}")

    with st.expander("Hidden factor diagnostics"):
        fig = go.Figure(go.Histogram(
            x=z_vals, nbinsx=40,
            marker_color="#A855F7", opacity=0.80,
        ))
        fig.update_layout(
            title="Distribution of hidden factor Z",
            xaxis_title="Z", yaxis_title="Count",
            height=300, margin=dict(t=45, b=45, l=55, r=20),
        )
        st.plotly_chart(fig, use_container_width=True, config=_CHART_CFG)

    st.caption(
        "Z affects true survival but is invisible to the prediction model. "
        "This can reduce AUC and increase the gap between predicted and true NNT."
    )


# ── Fitted model ─────────────────────────────────────────────────────────────

def render_fitted_model(
    fit_error,
    b0, b_gtv, b_mhd,
    p_ph, p_pr, pred_delta,
    p_fit_ph, p_fit_pr, fit_delta,
    true_delta,
    out_ph,
    gtv_slope=None,
    fit_corr=None,
    fit_debug=None,
    gtv=None,
    mhd=None,
    lr_scaler=None,
    proton_mode=None,
    proton_delta=None,
    proton_factor=None,
):
    st.subheader("Fitted Model")

    if fit_error is not None:
        st.error(fit_error)
    else:
        # Coefficients row
        c1, c2, c3 = st.columns(3)
        c1.metric("Intercept (β₀)", f"{b0:.3f}")
        c2.metric("β_GTV",          f"{b_gtv:.3f}")
        c3.metric("β_MHD",          f"{b_mhd:.3f}")

        st.markdown(
            f"**logit(P(OS2y)) = {b0:.3f}"
            f" {'+ ' if b_gtv >= 0 else '− '}{abs(b_gtv):.3f} × GTV"
            f" {'+ ' if b_mhd >= 0 else '− '}{abs(b_mhd):.3f} × MHD**"
        )
        st.caption(
            "GTV and MHD are standardised raw values (zero mean, unit variance). "
            "Logistic regression is fitted on binary photon survival outcomes."
        )

        # Correlation sanity check
        if fit_corr is not None and fit_corr < 0:
            st.error(
                f"Fitted predictions are anti-correlated with truth-generator "
                f"(r = {fit_corr:.3f}). The feature matrix may be inverted."
            )

        # Comparison table
        comp_rows = [
            {
                "Metric":          "Mean photon survival",
                "Truth-generator": f"{p_ph.mean():.3f}",
                "Fitted model":    f"{p_fit_ph.mean():.3f}",
            },
            {
                "Metric":          "Mean proton survival",
                "Truth-generator": f"{p_pr.mean():.3f}",
                "Fitted model":    f"{p_fit_pr.mean():.3f}",
            },
            {
                "Metric":          "Mean predicted Δ",
                "Truth-generator": f"{pred_delta.mean():.4f}",
                "Fitted model":    f"{fit_delta.mean():.4f}",
            },
            {
                "Metric":          "Mean true Δ (outcomes)",
                "Truth-generator": f"{true_delta.mean():.4f}",
                "Fitted model":    "—",
            },
        ]
        st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)

        # ── Reference patient plots ───────────────────────────────────────────
        if lr_scaler is not None and gtv is not None and mhd is not None:
            coef = np.array([b_gtv, b_mhd])
            gtv_med = float(np.median(gtv))
            mhd_med = float(np.median(mhd))

            col1, col2 = st.columns(2)

            # Plot 1: fixed median GTV, vary MHD
            with col1:
                mhd_x   = np.linspace(mhd.min(), mhd.max(), 300)
                mhd_x_pr = apply_proton_mhd(mhd_x, proton_mode, proton_delta, proton_factor)
                gtv_rep  = np.full_like(mhd_x, gtv_med)

                p_ph_mhd = logit_to_prob(
                    b0 + lr_scaler.transform(np.column_stack([gtv_rep, mhd_x])) @ coef
                )
                p_pr_mhd = logit_to_prob(
                    b0 + lr_scaler.transform(np.column_stack([gtv_rep, mhd_x_pr])) @ coef
                )

                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=mhd_x, y=p_ph_mhd, mode="lines", name="Photon",
                    line=dict(width=2.5),
                ))
                fig1.add_trace(go.Scatter(
                    x=mhd_x, y=p_pr_mhd, mode="lines", name="Proton",
                    line=dict(width=2.5, dash="dot"),
                ))
                fig1.update_layout(
                    title=f"Fitted model: OS2y vs MHD (at median GTV = {gtv_med:.1f} cc)",
                    xaxis_title="MHD (Gy)",
                    yaxis=dict(title="P(OS2y)", range=[0, 1]),
                    height=360,
                    margin=dict(t=55, b=50, l=60, r=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="right", x=1),
                )
                st.plotly_chart(fig1, use_container_width=True, config=_CHART_CFG)

            # Plot 2: fixed median MHD, vary GTV
            with col2:
                gtv_x    = np.linspace(gtv.min(), gtv.max(), 300)
                mhd_rep  = np.full_like(gtv_x, mhd_med)
                mhd_rep_pr = apply_proton_mhd(mhd_rep, proton_mode, proton_delta, proton_factor)

                p_ph_gtv = logit_to_prob(
                    b0 + lr_scaler.transform(np.column_stack([gtv_x, mhd_rep])) @ coef
                )
                p_pr_gtv = logit_to_prob(
                    b0 + lr_scaler.transform(np.column_stack([gtv_x, mhd_rep_pr])) @ coef
                )

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=gtv_x, y=p_ph_gtv, mode="lines", name="Photon",
                    line=dict(width=2.5),
                ))
                fig2.add_trace(go.Scatter(
                    x=gtv_x, y=p_pr_gtv, mode="lines", name="Proton",
                    line=dict(width=2.5, dash="dot"),
                ))
                fig2.update_layout(
                    title=f"Fitted model: OS2y vs GTV (at median MHD = {mhd_med:.1f} Gy)",
                    xaxis_title="GTV (cc)",
                    yaxis=dict(title="P(OS2y)", range=[0, 1]),
                    height=360,
                    margin=dict(t=55, b=50, l=60, r=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="right", x=1),
                )
                st.plotly_chart(fig2, use_container_width=True, config=_CHART_CFG)

    # Fitting diagnostics — always visible
    with st.expander("Fitting diagnostics"):
        n_alive = int(out_ph.sum())
        n_dead  = int(len(out_ph) - n_alive)
        n_total = len(out_ph)
        d1, d2, d3, d4, d5, d6 = st.columns(6)
        d1.metric("Alive (photon)",  f"{n_alive:,}")
        d2.metric("Dead (photon)",   f"{n_dead:,}")
        d3.metric("Mean P(OS2y)",    f"{p_ph.mean():.3f}")
        d4.metric("SD P(OS2y)",      f"{p_ph.std():.3f}")
        d5.metric("Min P(OS2y)",     f"{p_ph.min():.3f}")
        d6.metric("Max P(OS2y)",     f"{p_ph.max():.3f}")

        st.markdown(
            f"**Feature matrix:** shape ({n_total}, 2) — "
            f"column 0 = GTV (standardised), column 1 = MHD (standardised)"
        )
        if fit_corr is not None:
            corr_colour = "red" if fit_corr < 0 else "green"
            st.markdown(
                f"**Pearson r (fitted vs truth-generator):** "
                f":{corr_colour}[{fit_corr:.3f}]"
            )
            if fit_corr < 0:
                st.error("Anti-correlation detected — feature matrix may be inverted.")

        st.caption(
            "The summary above uses the truth-generator probability. "
            "The fitted logistic model is shown separately once the fit succeeds."
        )

    # Fitting debug logs — always visible
    with st.expander("Fitting debug logs"):
        d = fit_debug or {}

        if not d:
            st.info("No debug data available.")
        else:
            st.markdown("#### Pre-fit")
            st.write(f"**X shape:** {d.get('X_shape', 'N/A')}  |  "
                     f"**y shape:** {d.get('y_shape', 'N/A')}")
            st.write(f"**y=0 (dead):** {d.get('n_zeros', 'N/A')}  |  "
                     f"**y=1 (alive):** {d.get('n_ones', 'N/A')}")

            col_stats = {
                "Feature":   ["GTV (col 0)", "MHD (col 1)"],
                "Mean after scaling": [
                    f"{d['col0_mean']:.6f}" if "col0_mean" in d else "N/A",
                    f"{d['col1_mean']:.6f}" if "col1_mean" in d else "N/A",
                ],
                "SD after scaling": [
                    f"{d['col0_std']:.6f}" if "col0_std" in d else "N/A",
                    f"{d['col1_std']:.6f}" if "col1_std" in d else "N/A",
                ],
            }
            st.dataframe(pd.DataFrame(col_stats), hide_index=True, use_container_width=True)

            if "X_head" in d and "y_head" in d:
                head_df = pd.DataFrame(d["X_head"], columns=["GTV_z", "MHD_z"])
                head_df.insert(0, "y", d["y_head"])
                st.markdown("**First 5 rows of X and y:**")
                st.dataframe(head_df, hide_index=True, use_container_width=True)

            if "exception" in d:
                st.error(f"Exception during fitting: {d['exception']}")

            if "b0" in d:
                st.markdown("#### Post-fit")
                st.write(f"**Coefficients:** β₀ = {d['b0']:.4f} | "
                         f"β_GTV = {d['b_gtv']:.4f} | β_MHD = {d['b_mhd']:.4f}")
                st.write(f"**Fitted P̂ mean:** {d.get('fit_proba_mean', 'N/A'):.4f}  |  "
                         f"**SD:** {d.get('fit_proba_std', 'N/A'):.4f}")

                if "truth_mean" in d:
                    st.write(f"**Truth-generator P mean:** {d['truth_mean']:.4f}  |  "
                             f"**SD:** {d['truth_std']:.4f}")

                if "fit_truth_corr" in d:
                    r = d["fit_truth_corr"]
                    colour = "red" if r < 0.5 else "green"
                    st.markdown(f"**Pearson r (fitted vs truth):** :{colour}[{r:.4f}]")
                    if r < 0.5:
                        st.warning(f"Low correlation (r = {r:.4f}) — fitted predictions "
                                   "diverge from truth-generator.")

            if "example" in d:
                ex = d["example"]
                st.markdown("#### Manual single-patient check (patient 0)")
                st.write(f"Raw GTV = **{ex['gtv_raw']:.3f}** cc  |  "
                         f"Raw MHD = **{ex['mhd_raw']:.3f}** Gy")
                st.write(f"Standardised GTV_z = **{ex['gtv_z']:.4f}**  |  "
                         f"MHD_z = **{ex['mhd_z']:.4f}**")
                st.write(f"logit = {d['b0']:.4f} + "
                         f"{d['b_gtv']:.4f}×{ex['gtv_z']:.4f} + "
                         f"{d['b_mhd']:.4f}×{ex['mhd_z']:.4f} = **{ex['logit']:.4f}**")
                st.write(f"Manual P = 1/(1+exp(−logit)) = **{ex['p_manual']:.4f}**")
                st.write(f"sklearn predict_proba = **{ex['p_sklearn']:.4f}**")
                if abs(ex["p_manual"] - ex["p_sklearn"]) > 1e-4:
                    st.error("Manual and sklearn probabilities disagree — check implementation.")
                else:
                    st.success("Manual and sklearn probabilities agree ✓")


# ── Model diagnostics ────────────────────────────────────────────────────────

def render_model_diagnostics(
    p_ph, p_pr, out_ph_base, out_ph,
    noise_enabled, noise_sd,
    gtv_mid, gtv_slope, mhd_mid, mhd_slope,
    p_fit_ph=None,
):
    st.subheader("Model Diagnostics")

    # ── ROC curve — use fitted predictions when available ─────────────────────
    roc_scores = p_fit_ph if p_fit_ph is not None else p_ph
    score_label = "Fitted LR" if p_fit_ph is not None else "Truth-generator"

    auc_base = _compute_auc(roc_scores, out_ph_base)
    fpr_base, tpr_base = _compute_roc(roc_scores, out_ph_base)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr_base, y=tpr_base, mode="lines",
        name=f"{score_label} vs noiseless outcomes (AUC = {auc_base:.3f})",
        line=dict(width=2.5),
    ))

    if noise_enabled:
        auc_noise = _compute_auc(roc_scores, out_ph)
        fpr_noise, tpr_noise = _compute_roc(roc_scores, out_ph)
        fig_roc.add_trace(go.Scatter(
            x=fpr_noise, y=tpr_noise, mode="lines",
            name=f"{score_label} vs noisy outcomes SD={noise_sd:.1f} (AUC = {auc_noise:.3f})",
            line=dict(width=2.5, dash="dot"),
        ))

    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="grey", width=1, dash="dash"),
        name="Random (AUC = 0.50)", showlegend=True,
    ))
    fig_roc.update_layout(
        title="ROC Curve — Predicted vs Actual Photon Survival",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=420,
        margin=dict(t=50, b=50, l=60, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_roc, use_container_width=True, config=_CHART_CFG)

    col1, col2 = st.columns(2)

    # ── Prediction stats ───────────────────────────────────────────────────────
    with col1:
        st.markdown("**Predicted survival probability — summary**")
        stats_rows = [
            {"Arm": "Photon", "Mean": f"{p_ph.mean():.3f}", "SD": f"{p_ph.std():.3f}",
             "Min": f"{p_ph.min():.3f}", "Max": f"{p_ph.max():.3f}"},
            {"Arm": "Proton", "Mean": f"{p_pr.mean():.3f}", "SD": f"{p_pr.std():.3f}",
             "Min": f"{p_pr.min():.3f}", "Max": f"{p_pr.max():.3f}"},
        ]
        st.dataframe(pd.DataFrame(stats_rows), hide_index=True, use_container_width=True)

        # Histogram of predicted photon survival
        fig_hist = go.Figure(go.Histogram(
            x=p_ph, nbinsx=40, opacity=0.85,
        ))
        fig_hist.update_layout(
            title="Predicted photon survival probability",
            xaxis_title="Predicted P(survival)", yaxis_title="Count",
            height=280, margin=dict(t=45, b=45, l=55, r=20),
        )
        st.plotly_chart(fig_hist, use_container_width=True, config=_CHART_CFG)

    # ── Model parameter table ──────────────────────────────────────────────────
    with col2:
        st.markdown("**Survival model parameters**")
        param_rows = [
            {"Parameter": "GTV midpoint", "Value": f"{gtv_mid:.1f} cc"},
            {"Parameter": "GTV slope",    "Value": f"{gtv_slope:.4f}"},
            {"Parameter": "MHD midpoint", "Value": f"{mhd_mid:.1f} Gy"},
            {"Parameter": "MHD slope",    "Value": f"{mhd_slope:.4f}"},
        ]
        st.dataframe(pd.DataFrame(param_rows), hide_index=True, use_container_width=True)


# ── Extra exploration plots ───────────────────────────────────────────────────

def render_extra_plots(gtv, mhd, p_ph, p_pr, mhd_pr):
    st.divider()
    st.subheader("Additional Exploration")

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Histogram(x=gtv, nbinsx=40, marker_color="#4C8BF5", opacity=0.85))
        fig.update_layout(
            title="GTV distribution",
            xaxis_title="GTV (cc)", yaxis_title="Count",
            height=320, margin=dict(t=45, b=45, l=55, r=20),
        )
        st.plotly_chart(fig, use_container_width=True, config=_CHART_CFG)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=mhd, nbinsx=40, name="Photon MHD",
            marker_color="#E8543A", opacity=0.75,
        ))
        fig.add_trace(go.Histogram(
            x=mhd_pr, nbinsx=40, name="Proton MHD",
            marker_color="#00C9A7", opacity=0.55,
        ))
        fig.update_layout(
            title="MHD distribution (photon vs proton)",
            xaxis_title="MHD (Gy)", yaxis_title="Count",
            barmode="overlay",
            height=320, margin=dict(t=45, b=45, l=55, r=20),
            legend=dict(orientation="h", y=1.12, x=0, xanchor="left", font=dict(size=11)),
        )
        st.plotly_chart(fig, use_container_width=True, config=_CHART_CFG)

    # Survival probability vs MHD (10 equal-width bins)
    edges = np.linspace(mhd.min(), mhd.max(), 11)
    centers, p_ph_bin, p_pr_bin = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (mhd >= lo) & (mhd < hi)
        if mask.sum() > 0:
            centers.append(float((lo + hi) / 2))
            p_ph_bin.append(float(p_ph[mask].mean()))
            p_pr_bin.append(float(p_pr[mask].mean()))

    if centers:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=centers, y=p_ph_bin,
            mode="lines+markers", name="Photon",
            line=dict(color="#4C8BF5", width=2.5),
            marker=dict(size=8),
        ))
        fig.add_trace(go.Scatter(
            x=centers, y=p_pr_bin,
            mode="lines+markers", name="Proton",
            line=dict(color="#00C9A7", width=2.5),
            marker=dict(size=8),
        ))
        fig.update_layout(
            title="Mean survival probability vs MHD (10 equal-width bins)",
            xaxis_title="MHD bin centre (Gy)",
            yaxis_title="Mean P(OS2y)",
            height=360,
            margin=dict(t=50, b=50, l=60, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True, config=_CHART_CFG)


# ── App entry point ───────────────────────────────────────────────────────────

def main():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None:
            return
    except Exception:
        return

    st.set_page_config(
        page_title="NNT Simulation v2 — Proton vs Photon",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    for _k, _v in _DEFAULTS.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    n_patients, seed, proton_mode, hist_mode, noise_enabled, noise_sd, \
        z_enabled, z_mean, z_sd, z_beta = render_sidebar()

    gtv_mean      = float(st.session_state["gtv_mean"])
    gtv_std       = float(st.session_state["gtv_std"])
    mhd_mean      = float(st.session_state["mhd_mean"])
    mhd_std       = float(st.session_state["mhd_std"])
    gtv_mid       = float(st.session_state["gtv_mid"])
    gtv_slope     = float(st.session_state["gtv_slope"])
    mhd_mid       = float(st.session_state["mhd_mid"])
    mhd_slope     = float(st.session_state["mhd_slope"])
    proton_delta  = float(st.session_state["proton_delta"])
    proton_factor = float(st.session_state["proton_factor"])
    delta_thresh  = float(st.session_state["delta_thresh"])

    # ── Truth-world generation ────────────────────────────────────────────────
    gtv, mhd, p_ph, p_pr, pred_delta, true_delta, out_ph_base, out_ph, z_vals = run_simulation(
        n_patients, seed,
        gtv_mean, gtv_std, mhd_mean, mhd_std,
        gtv_mid, gtv_slope, mhd_mid, mhd_slope,
        proton_mode, proton_delta, proton_factor,
        noise_enabled, noise_sd,
        z_enabled, z_mean, z_sd, z_beta,
    )
    mhd_pr   = apply_proton_mhd(mhd, proton_mode, proton_delta, proton_factor)
    selected = pred_delta >= delta_thresh
    n_sel    = int(selected.sum())

    # ── Logistic regression fitting ───────────────────────────────────────────
    lr_model, lr_scaler, fit_error, fit_debug = fit_logistic_regression(gtv, mhd, out_ph)

    # ── Fitted photon and proton predictions ──────────────────────────────────
    if lr_model is not None:
        X_ph = lr_scaler.transform(np.column_stack([gtv,    mhd]))
        X_pr = lr_scaler.transform(np.column_stack([gtv,    mhd_pr]))
        p_fit_ph  = logit_to_prob(lr_model.intercept_[0] + X_ph @ lr_model.coef_[0])
        p_fit_pr  = logit_to_prob(lr_model.intercept_[0] + X_pr @ lr_model.coef_[0])
        fit_delta = p_fit_pr - p_fit_ph
        b0    = float(lr_model.intercept_[0])
        b_gtv = float(lr_model.coef_[0][0])
        b_mhd = float(lr_model.coef_[0][1])
        # Pearson r and truth stats appended to debug dict (copy to avoid mutating cache)
        fit_corr = float(np.corrcoef(p_fit_ph, p_ph)[0, 1])
        fit_debug = dict(fit_debug)
        fit_debug["fit_truth_corr"]  = fit_corr
        fit_debug["truth_mean"]      = float(p_ph.mean())
        fit_debug["truth_std"]       = float(p_ph.std())
    else:
        b0 = b_gtv = b_mhd = None
        p_fit_ph = p_fit_pr = fit_delta = fit_corr = None

    st.title("Predicted vs True NNT — Proton vs Photon")
    st.markdown(
        "This app simulates a patient population to explore how well the "
        "**predicted NNT** (from survival probability differences) matches the "
        "**true NNT** (from Monte Carlo binary outcomes) when selecting patients "
        "for proton therapy. All parameters update instantly."
    )

    render_summary_cards(n_patients, n_sel, p_ph, p_pr, pred_delta, true_delta, selected)
    st.divider()
    render_playground(proton_mode, hist_mode, gtv, mhd, mhd_pr)
    st.divider()
    render_selection(pred_delta, true_delta, delta_thresh, selected, n_sel, noise_enabled)
    st.divider()
    # ── NNT analysis ─────────────────────────────────────────────────────────
    render_bin_analysis(pred_delta, true_delta, delta_thresh, n_sel, fit_delta)
    if noise_enabled:
        st.divider()
        render_noise_section(p_ph, out_ph_base, out_ph, noise_sd)
    if z_enabled:
        st.divider()
        render_z_section(z_vals, z_mean, z_sd, z_beta)
    st.divider()
    render_fitted_model(
        fit_error,
        b0, b_gtv, b_mhd,
        p_ph, p_pr, pred_delta,
        p_fit_ph, p_fit_pr, fit_delta,
        true_delta,
        out_ph,
        gtv_slope=gtv_slope,
        fit_corr=fit_corr,
        fit_debug=fit_debug,
        gtv=gtv,
        mhd=mhd,
        lr_scaler=lr_scaler,
        proton_mode=proton_mode,
        proton_delta=proton_delta,
        proton_factor=proton_factor,
    )
    st.divider()
    render_model_diagnostics(
        p_ph, p_pr, out_ph_base, out_ph,
        noise_enabled, noise_sd,
        gtv_mid, gtv_slope, mhd_mid, mhd_slope,
        p_fit_ph=p_fit_ph,
    )
    render_extra_plots(gtv, mhd, p_ph, p_pr, mhd_pr)


main()
