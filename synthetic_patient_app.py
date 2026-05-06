"""
Synthetic patient dataset — confounding demo (mortality framing).

Core question:
  Can a model show an apparent MHD effect on 24-month mortality even when
  MHD is not truly causal, because MHD is correlated with GTV or hidden
  disease severity?

Causal structure:
  Scenario A:  GTV ──► mortality   GTV ──► MHD   (MHD not causal)
  Scenario B:  GTV ──► mortality   MHD ──► mortality   GTV ──► MHD
  Scenario C:  logit(p_mortality) = −1.3409 + 0.059·√GTV + 0.2635·√MHD
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.patches import Patch
from scipy import stats
from scipy.stats import norm as sp_norm, lognorm, gamma as sp_gamma
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc


# ── constants ─────────────────────────────────────────────────────────────────

DIST_SIMPLE  = "Eenvoudige normale verdelingen"
DIST_ARTICLE = "Artikel-achtige longkanker-verdelingen"

SCEN_A = "A — GTV causaal, MHD niet-causaal"
SCEN_B = "B — GTV en MHD causaal"
SCEN_C = "C — Artikel-stijl mortaliteitsmodel"

PROTON_MULT = "A — Multiplicatief"
PROTON_ABS  = "B — Absoluut"

SCALE_Z_RAW    = "Gestandaardiseerd ruw GTV/MHD"
SCALE_SQRT_RAW = "√GTV/MHD, ongestandaardiseerd"
SCALE_SQRT_Z   = "√GTV/MHD, gestandaardiseerd"

_GTV_MU    = np.log(69.0)
_GTV_SIGMA = np.sqrt(2 * (np.log(110.0) - np.log(69.0)))
_MHD_K     = (12.0 / 8.2) ** 2
_MHD_THETA = 8.2 ** 2 / 12.0

_BINS    = [0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40,
            0.50, 0.60, 0.70, 0.80, 0.90, 1.01]
_BLABELS = ["0–2 %", "2–5 %", "5–10 %", "10–20 %", "20–30 %", "30–40 %",
            "40–50 %", "50–60 %", "60–70 %", "70–80 %", "80–90 %", "90–100 %"]

_NOISE_LEVELS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]

_FIT_TARGETS = {
    "beta_gtv":  0.0590,
    "beta_mhd":  0.2635,
    "auc":       0.640,
    "mortality": 0.506,
}
_FIT_WEIGHTS = {"beta_gtv": 1.0, "beta_mhd": 1.0, "auc": 2.0, "mortality": 2.0}

# Copula target r used internally for the "estimate rho" advanced tool.
# Does NOT claim this is reported in the article.
_COPULA_ESTIMATE_TARGET_R = 0.45


# ── cached computation functions ──────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _calibrate_correlation(
    target_pearson_r: float = _COPULA_ESTIMATE_TARGET_R,
    n: int = 5000,
    seed: int = 0,
) -> tuple[float, float]:
    """Grid-search the Gaussian copula rho that best reproduces a target
    Pearson r(GTV, MHD).  Returns (best_rho, achieved_r).  Cached after first run."""
    rng0 = np.random.default_rng(seed)
    best_rho, best_err, best_r = 0.45, 1.0, 0.45
    for rho in np.linspace(0.0, 0.95, 96):
        Z  = rng0.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], size=n)
        U1 = np.clip(sp_norm.cdf(Z[:, 0]), 1e-9, 1 - 1e-9)
        U2 = np.clip(sp_norm.cdf(Z[:, 1]), 1e-9, 1 - 1e-9)
        gtv = lognorm.ppf(U1, s=_GTV_SIGMA, scale=np.exp(_GTV_MU)).clip(0.0, 1800.0)
        mhd = sp_gamma.ppf(U2, a=_MHD_K, scale=_MHD_THETA).clip(0.0, 45.0)
        r, _ = stats.pearsonr(gtv, mhd)
        err  = abs(r - target_pearson_r)
        if err < best_err:
            best_err, best_rho, best_r = err, float(rho), float(r)
    return best_rho, best_r


@st.cache_data(show_spinner=False)
def _fit_to_article(
    n: int = 500,
    test_size: float = 0.2,
    seed: int = 42,
    true_b2: float = _FIT_TARGETS["beta_mhd"],
) -> dict:
    """Grid search over (copula rho, noise) to minimise loss vs article targets.
    Generates data with SCALE_SQRT_RAW, fits an unstandardized logistic model on
    sqrt(GTV) and sqrt(MHD), and evaluates the four target metrics.
    true_b2=0 for Scenario A (MHD non-causal), true_b2=0.2635 for Scenario B/C.
    Returns the best parameter set and achieved metrics."""
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.model_selection import train_test_split as _tts
    from sklearn.metrics import roc_curve as _roc, auc as _auc
    from scipy.stats import norm as _norm

    rho_grid   = np.arange(0.0, 0.95, 0.05)
    noise_grid = np.arange(0.0, 2.25, 0.25)

    best_loss   = np.inf
    best_params = {"rho": 0.45, "noise": 1.0}
    best_metrics: dict = {}

    for rho in rho_grid:
        for noise in noise_grid:
            rng0 = np.random.default_rng(seed)
            Z  = rng0.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], size=n)
            eps = 1e-6
            U1 = np.clip(_norm.cdf(Z[:, 0]), eps, 1 - eps)
            U2 = np.clip(_norm.cdf(Z[:, 1]), eps, 1 - eps)
            gtv = lognorm.ppf(U1, s=_GTV_SIGMA, scale=np.exp(_GTV_MU)).clip(0.0, 1800.0)
            mhd = sp_gamma.ppf(U2, a=_MHD_K, scale=_MHD_THETA).clip(0.0, 45.0)

            tv_gen  = np.sqrt(gtv)
            mhd_gen = np.sqrt(mhd)
            gen_sl  = _FIT_TARGETS["beta_gtv"] * tv_gen + true_b2 * mhd_gen
            target_mort = _FIT_TARGETS["mortality"]
            try:
                def _mp(i): return (1/(1+np.exp(-(i+gen_sl)))).mean() - target_mort
                intercept = brentq(_mp, -30.0, 30.0)
            except ValueError:
                intercept = np.log(target_mort/(1-target_mort)) - gen_sl.mean()
            logit = intercept + gen_sl + noise * rng0.standard_normal(n)
            mortality = rng0.binomial(1, 1.0/(1.0+np.exp(-logit))).astype(int)

            if len(np.unique(mortality)) < 2:
                continue

            Xsq = np.column_stack([tv_gen, mhd_gen])
            Xtr, Xte, ytr, yte = _tts(Xsq, mortality,
                                       test_size=test_size, random_state=int(seed))
            if len(np.unique(ytr)) < 2:
                continue

            lr = _LR(max_iter=1000, fit_intercept=True)
            lr.fit(Xtr, ytr)
            b_gtv, b_mhd = lr.coef_[0]
            fp, tp, _ = _roc(yte, lr.predict_proba(Xte)[:, 1])
            achieved_auc  = _auc(fp, tp)
            achieved_mort = mortality.mean()

            loss = (
                _FIT_WEIGHTS["beta_gtv"]  * (b_gtv - _FIT_TARGETS["beta_gtv"])**2 +
                _FIT_WEIGHTS["beta_mhd"]  * (b_mhd - _FIT_TARGETS["beta_mhd"])**2 +
                _FIT_WEIGHTS["auc"]       * (achieved_auc  - _FIT_TARGETS["auc"])**2 +
                _FIT_WEIGHTS["mortality"] * (achieved_mort - _FIT_TARGETS["mortality"])**2
            )
            if loss < best_loss:
                best_loss   = loss
                best_params = {"rho": float(rho), "noise": float(noise)}
                best_metrics = {
                    "beta_gtv":  float(b_gtv),
                    "beta_mhd":  float(b_mhd),
                    "auc":       float(achieved_auc),
                    "mortality": float(achieved_mort),
                    "loss":      float(best_loss),
                }

    return {**best_params, **best_metrics}


# ── helpers ───────────────────────────────────────────────────────────────────

def wald_pvalues(lr: LogisticRegression, X_scaled: np.ndarray) -> np.ndarray:
    p = lr.predict_proba(X_scaled)[:, 1]
    W = p * (1 - p)
    Xa = np.column_stack([np.ones(len(X_scaled)), X_scaled])
    H  = Xa.T @ (W[:, None] * Xa)
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.full(X_scaled.shape[1], np.nan)
    se = np.sqrt(np.diag(cov)[1:])
    z  = lr.coef_[0] / se
    return 2 * (1 - stats.norm.cdf(np.abs(z)))


def wald_pvalues_pipe(pipe: Pipeline, X_feat: np.ndarray) -> np.ndarray:
    return wald_pvalues(
        pipe.named_steps["lr"],
        pipe.named_steps["sc"].transform(X_feat),
    )


def fmt_pval(p: float) -> str:
    if np.isnan(p): return "n/a"
    if p < 0.001:   return "< 0.001"
    return f"{p:.3f}"


def sig_stars(p: float) -> str:
    if np.isnan(p): return ""
    if p < 0.001:   return "***"
    if p < 0.01:    return "**"
    if p < 0.05:    return "*"
    return "ns"


def make_pipe() -> Pipeline:
    return Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])


def centered(fig, use_container_width: bool = False) -> None:
    _, mid, _ = st.columns([1, 3, 1])
    with mid:
        st.pyplot(fig, use_container_width=use_container_width)
    plt.close(fig)


def describe_arr(arr: np.ndarray) -> dict:
    return {
        "Mean":   f"{arr.mean():.2f}",
        "SD":     f"{arr.std(ddof=1):.2f}",
        "Median": f"{np.median(arr):.2f}",
        "Min":    f"{arr.min():.2f}",
        "Max":    f"{arr.max():.2f}",
    }


def fmt_f(v, decimals=1):
    try:
        return "—" if np.isnan(float(v)) else f"{v:.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def build_nnt_table(delta: np.ndarray, n_total: int) -> pd.DataFrame:
    """Original single-delta NNT table (kept for backward compatibility)."""
    rows = []
    for label, lo, hi in zip(_BLABELS, _BINS[:-1], _BINS[1:]):
        mask = (delta >= lo) & (delta < hi)
        d, n = delta[mask], int(mask.sum())
        if n == 0:
            rows.append({"Bin": label, "N": 0, "%": "0.0 %",
                         "Mean Δ": "—", "Median Δ": "—",
                         "Mean NNT": "—", "Median NNT": "—",
                         "1/Mean Δ": "—", "1/Median Δ": "—"})
            continue
        mean_d, med_d = d.mean(), np.median(d)
        nnt_pos = np.where(d > 0, 1.0 / d, np.nan)
        rows.append({
            "Bin":        label,
            "N":          n,
            "%":          f"{100 * n / n_total:.1f} %",
            "Mean Δ":     f"{mean_d * 100:.2f} %",
            "Median Δ":   f"{med_d  * 100:.2f} %",
            "Mean NNT":   fmt_f(np.nanmean(nnt_pos)),
            "Median NNT": fmt_f(np.nanmedian(nnt_pos)),
            "1/Mean Δ":   fmt_f(1 / mean_d) if mean_d > 0 else "—",
            "1/Median Δ": fmt_f(1 / med_d)  if med_d  > 0 else "—",
        })
    return pd.DataFrame(rows)


def build_nnt_comparison_table(
    delta_pred: np.ndarray,
    delta_true: np.ndarray | None,
    n_total: int,
) -> pd.DataFrame:
    """NNT comparison table binned on predicted delta.

    Shows predicted NNT, true causal NNT, and NNT inflation factor
    (true_NNT / predicted_NNT) for each bin.
    delta_true=None means no true benefit is available (Scenario A, no override).
    """
    rows = []
    for label, lo, hi in zip(_BLABELS, _BINS[:-1], _BINS[1:]):
        mask = (delta_pred >= lo) & (delta_pred < hi)
        n = int(mask.sum())
        dp = delta_pred[mask]
        dt = delta_true[mask] if delta_true is not None else None

        if n == 0:
            rows.append({
                "Predicted Δ bin": label, "N": 0,
                "Mean predicted Δ": "—", "Mean true Δ": "—",
                "Predicted NNT": "—", "True causal NNT": "—",
                "NNT inflation": "—",
            })
            continue

        mean_dp  = dp.mean()
        pred_nnt = (1.0 / mean_dp) if mean_dp > 0 else float("nan")

        if dt is not None:
            mean_dt   = dt.mean()
            true_nnt  = (1.0 / mean_dt) if mean_dt > 0 else float("nan")
            inflation = (true_nnt / pred_nnt
                         if (not np.isnan(true_nnt) and not np.isnan(pred_nnt) and pred_nnt > 0)
                         else float("nan"))
            mean_dt_str  = f"{mean_dt * 100:.2f} %"
            true_nnt_str = fmt_f(true_nnt)
            infl_str     = fmt_f(inflation)
        else:
            mean_dt_str  = "0 % (no true effect)"
            true_nnt_str = "No true benefit"
            infl_str     = "∞"

        rows.append({
            "Predicted Δ bin":   label,
            "N":                 n,
            "Mean predicted Δ":  f"{mean_dp * 100:.2f} %",
            "Mean true Δ":       mean_dt_str,
            "Predicted NNT":     fmt_f(pred_nnt),
            "True causal NNT":   true_nnt_str,
            "NNT inflation":     infl_str,
        })
    return pd.DataFrame(rows)


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="MHD Confounding Simulator", layout="wide")

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Simulation settings")

    # ── A. Data generation ────────────────────────────────────────────────────
    st.subheader("A. Data generation")

    dist_mode = st.selectbox(
        "Distribution mode",
        [DIST_ARTICLE, DIST_SIMPLE],
        help=(
            "**Article-like:** log-normal GTV and gamma-distributed MHD via Gaussian copula. "
            "**Simple:** normal distributions with adjustable parameters."
        ),
    )

    n_patients = st.slider("Sample size", 100, 3000, 500, step=50)
    seed       = st.number_input("Random seed", value=42, step=1)

    use_calibrated = False  # overridden below for DIST_ARTICLE

    if dist_mode == DIST_SIMPLE:
        tv_mu        = st.slider("Mean GTV (cc)",          10.0, 200.0, 45.0, step=1.0)
        tv_sigma     = st.slider("SD GTV (cc)",             5.0,  60.0, 15.0, step=1.0)
        corr_a       = st.slider("GTV → MHD coefficient",  0.0,   1.0,  0.5, step=0.05)
        mhd_noise_sd = st.slider("MHD noise SD (Gy)",      0.1,  10.0,  3.0, step=0.1)
        target_corr  = corr_a
    else:
        target_corr = st.slider(
            "GTV ↔ MHD copula correlation (ρ)", 0.0, 0.9, 0.45, step=0.05,
            help=(
                "Controls how strongly larger tumors also tend to receive higher mean heart dose. "
                "Higher ρ → stronger GTV-MHD correlation in the generated data."
            ),
        )

    # ── B. Mortality scenario ─────────────────────────────────────────────────
    st.subheader("B. Mortality scenario")

    survival_scenario = st.selectbox("Scenario", [SCEN_A, SCEN_B, SCEN_C])

    if survival_scenario == SCEN_A:
        st.info(
            "**Scenario A — GTV causal, MHD not causal.**  \n"
            "Mortality depends only on GTV. MHD correlates with GTV but has no "
            "direct effect on mortality. Any fitted MHD coefficient arises from "
            "correlation or noise alone."
        )
    elif survival_scenario == SCEN_B:
        st.info(
            "**Scenario B — GTV and MHD both causal.**  \n"
            "Both GTV and MHD have a true direct effect on mortality. "
            "The fitted MHD coefficient reflects a real causal pathway."
        )
    else:
        st.info(
            "**Scenario C — Article formula.**  \n"
            "`logit(p) = −1.3409 + 0.059·√GTV + 0.2635·√MHD`  \n"
            "Data generated directly from the Van Loon et al. model. "
            "Both GTV and MHD are causal in this scenario."
        )

    mhd_is_causal = survival_scenario in (SCEN_B, SCEN_C)

    # ── C. True effect controls ───────────────────────────────────────────────
    st.subheader("C. True effect controls")

    if survival_scenario in (SCEN_A, SCEN_B):
        _default_scale = SCALE_SQRT_RAW if dist_mode == DIST_ARTICLE else SCALE_Z_RAW
        gen_scale = st.selectbox(
            "True mortality formula scale",
            [SCALE_Z_RAW, SCALE_SQRT_RAW, SCALE_SQRT_Z],
            index=[SCALE_Z_RAW, SCALE_SQRT_RAW, SCALE_SQRT_Z].index(_default_scale),
            key=f"gen_scale_{dist_mode}_{survival_scenario}",
            help=(
                "**Standardised raw:** b1/b2 are standardised logit effects of raw values. "
                "**√ unstandardised:** same scale as the Van Loon article (0.059 / 0.264). "
                "**√ standardised:** b1/b2 are standardised logit effects of √-transformed values."
            ),
        )
        if gen_scale == SCALE_SQRT_RAW:
            _b1_def, _b1_max, _b1_step = 0.0590, 0.5,  0.005
            _b2_def, _b2_max, _b2_step = 0.2635, 1.0,  0.005
            _b1_label = "b1 — true √GTV effect (logit)"
            _b2_label = "b2 — true √MHD effect (logit)"
        elif gen_scale == SCALE_SQRT_Z:
            _b1_def, _b1_max, _b1_step = 2.0, 5.0, 0.1
            _b2_def, _b2_max, _b2_step = 1.0, 3.0, 0.1
            _b1_label = "b1 — true effect of standardised √GTV (logit)"
            _b2_label = "b2 — true effect of standardised √MHD (logit)"
        else:
            _b1_def, _b1_max, _b1_step = 2.0, 5.0, 0.1
            _b2_def, _b2_max, _b2_step = 1.0, 3.0, 0.1
            _b1_label = "b1 — true GTV effect, standardised (logit)"
            _b2_label = "b2 — true MHD effect, standardised (logit)"

        b1 = st.slider(_b1_label, 0.0, _b1_max, _b1_def, step=_b1_step,
                       key=f"b1_{gen_scale}")
        if survival_scenario == SCEN_B:
            b2_val = st.slider(_b2_label, 0.0, _b2_max, _b2_def, step=_b2_step,
                               key=f"b2_{gen_scale}")
        else:
            b2_val = 0.0

        surv_noise = st.slider(
            "Mortality noise level (σ)", 0.0, 3.0, 1.0, step=0.1,
            help="Random variation on the logit scale not explained by GTV or MHD.",
        )
    else:
        gen_scale  = SCALE_SQRT_RAW
        b1, b2_val, surv_noise = 0.059, 0.2635, 0.0

    baseline_survival = st.slider(
        "Baseline 2-year survival", 0.20, 0.90, 0.494, step=0.01,
        help="Target mean survival before individual risk differences. 49.4 % ≈ 50.6 % mortality.",
    )

    if survival_scenario == SCEN_C:
        calibrate_article = st.checkbox(
            "Calibrate article intercept to selected baseline",
            value=False,
            help="Adjusts only the intercept so mean simulated mortality matches the baseline. "
                 "Slopes (0.0590 and 0.2635) remain unchanged.",
        )
    else:
        calibrate_article = False

    # ── D. Proton simulation ──────────────────────────────────────────────────
    st.subheader("D. Proton simulation")

    proton_reduction_mode = st.selectbox(
        "MHD reduction mode",
        [PROTON_MULT, PROTON_ABS],
        help=(
            "**Multiplicative:** mhd_proton = mhd_photon × factor.  "
            "**Absolute:** mhd_proton = mhd_photon − reduction (Gy), minimum 0."
        ),
    )
    if proton_reduction_mode == PROTON_MULT:
        mhd_reduction_factor   = st.slider("MHD reduction factor", 0.0, 1.0, 0.6, step=0.05,
                                           help="0.6 = proton reduces MHD to 60 % of photon value.")
        mhd_absolute_reduction = 0.0
    else:
        mhd_absolute_reduction = st.slider("Absolute MHD reduction (Gy)", 0.0, 30.0, 5.0, step=0.5)
        mhd_reduction_factor   = 1.0

    # ── Advanced controls ─────────────────────────────────────────────────────
    st.divider()
    with st.expander("Advanced controls", expanded=False):

        st.markdown("**Model settings**")
        use_sqrt = st.checkbox(
            "Use √-transformed predictors in model",
            value=(dist_mode == DIST_ARTICLE),
            key=f"use_sqrt_{dist_mode}",
            help="Fits the logistic model on √GTV and √MHD. Affects model only, not data generation.",
        )
        test_size = st.slider("Test set fraction", 0.1, 0.5, 0.2, step=0.05)

        st.markdown("**Estimate copula ρ for a target Pearson r**")
        st.caption(
            "Grid-searches the copula ρ that best reproduces a chosen Pearson r(GTV, MHD) "
            "under the article-like distributions. Does not claim the article reports this value."
        )
        if dist_mode == DIST_ARTICLE:
            _adv_target_r = st.slider(
                "Target Pearson r", 0.10, 0.90, _COPULA_ESTIMATE_TARGET_R, step=0.05,
                key="adv_target_r",
            )
            use_calibrated = st.toggle("Apply estimated ρ to simulation", value=False)
            if use_calibrated:
                with st.spinner("Estimating copula ρ…"):
                    _cal_rho, _cal_r = _calibrate_correlation(target_pearson_r=_adv_target_r)
                target_corr = _cal_rho
                st.caption(
                    f"Best copula ρ = **{_cal_rho:.3f}** → achieved Pearson r ≈ {_cal_r:.3f} "
                    f"(target {_adv_target_r:.2f})"
                )
        else:
            st.caption("Only available with article-like distributions.")

        st.markdown("**True mortality gain from MHD reduction**")
        st.caption(
            "Off: proton MHD reduction only changes model-predicted risk. "
            "Actual simulated outcome does not improve unless MHD is truly causal.  \n"
            "On: forces a true treatment effect proportional to MHD reduction, "
            "regardless of scenario causal structure."
        )
        apply_true_proton = st.toggle("Enable forced true treatment effect", value=False)
        if apply_true_proton:
            proton_effect_strength = st.slider(
                "True MHD reduction effect strength", 0.0, 1.0, 0.1, step=0.01,
            )
        else:
            proton_effect_strength = 0.0

        st.markdown("**Fit to article targets**")
        _fit_eligible = (
            dist_mode == DIST_ARTICLE
            and survival_scenario in (SCEN_A, SCEN_B)
            and gen_scale == SCALE_SQRT_RAW
        )
        if not _fit_eligible:
            st.caption(
                "Available when: article-like distributions + Scenario A or B + √ unstandardised scale."
            )
            fit_to_article = False
        else:
            fit_to_article = st.toggle(
                "Optimise parameters to article targets",
                key="fit_to_article_toggle",
                value=False,
                help=(
                    "Grid search over copula ρ and noise to best reproduce: "
                    "β_√GTV ≈ 0.059, β_√MHD ≈ 0.264, AUC ≈ 0.64, mortality ≈ 50.6 %. "
                    "Result is cached — first run ~5 s."
                ),
            )
            if fit_to_article:
                with st.spinner("Running grid search…"):
                    _true_b2_for_optimizer = (
                        0.0 if survival_scenario == SCEN_A else _FIT_TARGETS["beta_mhd"]
                    )
                    _fit_result = _fit_to_article(
                        n=n_patients, test_size=test_size, seed=int(seed),
                        true_b2=_true_b2_for_optimizer,
                    )
                st.caption(
                    f"✅ **Optimum found** — "
                    f"ρ = **{_fit_result['rho']:.2f}**, σ = **{_fit_result['noise']:.2f}**  \n"
                    f"β_√GTV = {_fit_result['beta_gtv']:.4f} (target 0.0590)  \n"
                    f"β_√MHD = {_fit_result['beta_mhd']:.4f} (target 0.2635)  \n"
                    f"AUC = {_fit_result['auc']:.3f} (target 0.640)  \n"
                    f"Mortality = {_fit_result['mortality']*100:.1f} % (target 50.6 %)"
                )


# ── fit-to-article overrides ──────────────────────────────────────────────────

_fit_overridden = False
if fit_to_article and _fit_eligible:
    target_corr = _fit_result["rho"]
    surv_noise  = _fit_result["noise"]
    _fit_overridden = True

# ── data generation ───────────────────────────────────────────────────────────

rng = np.random.default_rng(int(seed))

if dist_mode == DIST_SIMPLE:
    tumor_volume    = rng.normal(loc=tv_mu, scale=tv_sigma, size=n_patients).clip(1.0, None)
    mhd_base        = 2.0 + corr_a * (tumor_volume / tv_mu) * 10.0
    mean_heart_dose = (mhd_base + rng.normal(0, mhd_noise_sd, n_patients)).clip(0.0, None)
else:
    rho = float(target_corr)
    Z   = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], size=n_patients)
    eps = 1e-6
    U1  = np.clip(sp_norm.cdf(Z[:, 0]), eps, 1 - eps)
    U2  = np.clip(sp_norm.cdf(Z[:, 1]), eps, 1 - eps)
    tumor_volume    = lognorm.ppf(U1, s=_GTV_SIGMA, scale=np.exp(_GTV_MU)).clip(0.0, 1800.0)
    mean_heart_dose = sp_gamma.ppf(U2, a=_MHD_K, scale=_MHD_THETA).clip(0.0, 45.0)

# ── mortality generation ──────────────────────────────────────────────────────

_baseline_mort = 1.0 - baseline_survival

if survival_scenario == SCEN_C:
    article_slopes = 0.0590 * np.sqrt(tumor_volume) + 0.2635 * np.sqrt(mean_heart_dose)
    if calibrate_article:
        def _mean_p(intercept):
            return (1.0 / (1.0 + np.exp(-(intercept + article_slopes)))).mean() - _baseline_mort
        try:
            _art_int = brentq(_mean_p, -30.0, 30.0)
        except ValueError:
            _art_int = np.log(_baseline_mort / (1.0 - _baseline_mort)) - article_slopes.mean()
    else:
        _art_int = -1.3409
    true_mort_logit = _art_int + article_slopes
    _display_int    = _art_int
else:
    if gen_scale == SCALE_SQRT_RAW:
        tv_gen  = np.sqrt(tumor_volume)
        mhd_gen = np.sqrt(mean_heart_dose)
        _gen_slopes = b1 * tv_gen + b2_val * mhd_gen
        def _mean_p_ab(intercept):
            return (1.0 / (1.0 + np.exp(-(intercept + _gen_slopes)))).mean() - _baseline_mort
        try:
            _mort_int = brentq(_mean_p_ab, -30.0, 30.0)
        except ValueError:
            _mort_int = np.log(_baseline_mort / (1.0 - _baseline_mort)) - _gen_slopes.mean()
    elif gen_scale == SCALE_SQRT_Z:
        _sq_gtv = np.sqrt(tumor_volume)
        _sq_mhd = np.sqrt(mean_heart_dose)
        tv_gen  = (_sq_gtv - _sq_gtv.mean()) / _sq_gtv.std()
        mhd_gen = (_sq_mhd - _sq_mhd.mean()) / _sq_mhd.std()
        _mort_int = np.log(_baseline_mort / (1.0 - _baseline_mort))
    else:  # SCALE_Z_RAW
        tv_gen  = (tumor_volume    - tumor_volume.mean())    / tumor_volume.std()
        mhd_gen = (mean_heart_dose - mean_heart_dose.mean()) / mean_heart_dose.std()
        _mort_int = np.log(_baseline_mort / (1.0 - _baseline_mort))
    _display_int    = _mort_int
    true_mort_logit = (_mort_int
                       + b1 * tv_gen
                       + b2_val * mhd_gen
                       + surv_noise * rng.standard_normal(n_patients))

p_mortality = 1.0 / (1.0 + np.exp(-true_mort_logit))
mortality   = rng.binomial(1, p_mortality).astype(int)
survival    = 1 - mortality

# ── dataframe ─────────────────────────────────────────────────────────────────

df = pd.DataFrame({
    "tumor_volume_cc":    tumor_volume.round(2),
    "mean_heart_dose_gy": mean_heart_dose.round(2),
    "mortality":          mortality,
})
y = df["mortality"].values

# ── proton MHD reduction ──────────────────────────────────────────────────────

if proton_reduction_mode == PROTON_MULT:
    mhd_proton    = (mean_heart_dose * mhd_reduction_factor).clip(0.0)
    _proton_label = f"factor × {mhd_reduction_factor:.2f}"
else:
    mhd_proton    = (mean_heart_dose - mhd_absolute_reduction).clip(0.0)
    _proton_label = f"− {mhd_absolute_reduction:.1f} Gy"

# ── statistics ────────────────────────────────────────────────────────────────

pearson_r, pearson_p = stats.pearsonr(tumor_volume, mean_heart_dose)
obs_mort_rate        = mortality.mean()

# ── feature preparation ───────────────────────────────────────────────────────

X_raw = df[["tumor_volume_cc", "mean_heart_dose_gy"]].values
if use_sqrt:
    X_feat = np.sqrt(X_raw)
    fl     = ["√GTV", "√MHD"]
else:
    X_feat = X_raw.copy()
    fl     = ["GTV", "MHD"]

if use_sqrt:
    X_feat_proton = np.column_stack([np.sqrt(tumor_volume), np.sqrt(mhd_proton)])
else:
    X_feat_proton = np.column_stack([tumor_volume, mhd_proton])

# ── train / test split ────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X_feat, y, test_size=test_size, random_state=int(seed)
)

# ── guard: need both classes in training set ──────────────────────────────────

if len(np.unique(y_train)) < 2:
    st.warning(
        "⚠️ Training set contains only one class. "
        "Increase sample size, adjust baseline mortality, or lower noise."
    )
    st.stop()

# ── three pipeline models ─────────────────────────────────────────────────────

pipe_A, pipe_B, pipe_C = make_pipe(), make_pipe(), make_pipe()
pipe_A.fit(X_train[:, [0]], y_train)
pipe_B.fit(X_train[:, [1]], y_train)
pipe_C.fit(X_train,         y_train)


def _roc(pipe, Xte, cols=None):
    Xi = Xte[:, cols] if cols is not None else Xte
    f, t, _ = roc_curve(y_test, pipe.predict_proba(Xi)[:, 1])
    return f, t, auc(f, t)


fpr_A, tpr_A, auc_A = _roc(pipe_A, X_test, cols=[0])
fpr_B, tpr_B, auc_B = _roc(pipe_B, X_test, cols=[1])
fpr_C, tpr_C, auc_C = _roc(pipe_C, X_test)

_lr      = lambda p: p.named_steps["lr"]
_coefstr = lambda p, labels: ", ".join(
    f"{l}: {c:.3f}" for l, c in zip(labels, _lr(p).coef_[0])
)

# ── combined model regression results ─────────────────────────────────────────

pvals_C    = wald_pvalues_pipe(pipe_C, X_feat)
coefs_C    = _lr(pipe_C).coef_[0]
interc_C   = _lr(pipe_C).intercept_[0]
y_pred_C   = pipe_C.predict(X_test)
accuracy_C = (y_pred_C == y_test).mean()

# Unstandardised logistic regression for article coefficient comparison
if use_sqrt and X_train.shape[1] == 2:
    _lr_unstd        = LogisticRegression(max_iter=1000, fit_intercept=True)
    _lr_unstd.fit(X_train, y_train)
    _beta_gtv_unstd  = float(_lr_unstd.coef_[0][0])
    _beta_mhd_unstd  = float(_lr_unstd.coef_[0][1])
    _intercept_unstd = float(_lr_unstd.intercept_[0])
    _fp_u, _tp_u, _  = roc_curve(y_test, _lr_unstd.predict_proba(X_test)[:, 1])
    _auc_unstd       = float(auc(_fp_u, _tp_u))
else:
    _beta_gtv_unstd = _beta_mhd_unstd = _intercept_unstd = _auc_unstd = float("nan")

# Coefficient table for core results tab
coef_df = pd.DataFrame({
    "Feature":               fl,
    "Std. coefficient":      coefs_C.round(4),
    "Unstd. coefficient":    [
        f"{_beta_gtv_unstd:.4f}" if not np.isnan(_beta_gtv_unstd) else "n/a",
        f"{_beta_mhd_unstd:.4f}" if not np.isnan(_beta_mhd_unstd) else "n/a",
    ],
    "p-value":               [fmt_pval(p) for p in pvals_C],
    "Sig.":                  [sig_stars(p) for p in pvals_C],
    "Causal in simulation?": ["Yes", "Yes" if mhd_is_causal else "No"],
})

roc_summary = pd.DataFrame({
    "Model":          ["GTV only", "MHD only", "GTV + MHD"],
    "Predictors":     [fl[0], fl[1], f"{fl[0]} + {fl[1]}"],
    "AUC (test set)": [f"{auc_A:.3f}", f"{auc_B:.3f}", f"{auc_C:.3f}"],
    "Std. coefs":     [_coefstr(pipe_A, [fl[0]]),
                       _coefstr(pipe_B, [fl[1]]),
                       _coefstr(pipe_C, fl)],
})

# ── proton benefit calculations ───────────────────────────────────────────────

p_mort_photon = pipe_C.predict_proba(X_feat)[:, 1]
p_mort_proton = pipe_C.predict_proba(X_feat_proton)[:, 1]
delta_model   = p_mort_photon - p_mort_proton
nnt_model     = np.where(delta_model > 0, 1.0 / delta_model, np.nan)

# True causal delta — only non-zero when forced true treatment effect is on
if apply_true_proton and proton_effect_strength > 0:
    mhd_diff             = mean_heart_dose - mhd_proton
    true_mort_logit_prot = true_mort_logit - proton_effect_strength * mhd_diff
    true_p_mort_photon   = 1.0 / (1.0 + np.exp(-true_mort_logit))
    true_p_mort_proton   = 1.0 / (1.0 + np.exp(-true_mort_logit_prot))
    true_delta           = true_p_mort_photon - true_p_mort_proton
else:
    true_delta = None

# Aggregate NNT metrics
_mean_pred_delta = delta_model.mean()
_pred_nnt_mean   = (1.0 / _mean_pred_delta) if _mean_pred_delta > 0 else float("nan")

if true_delta is not None:
    _mean_true_delta = true_delta.mean()
    _true_nnt_mean   = (1.0 / _mean_true_delta) if _mean_true_delta > 0 else float("nan")
    _nnt_inflation   = (
        _true_nnt_mean / _pred_nnt_mean
        if (not np.isnan(_true_nnt_mean) and not np.isnan(_pred_nnt_mean) and _pred_nnt_mean > 0)
        else float("nan")
    )
else:
    _mean_true_delta = 0.0
    _true_nnt_mean   = float("nan")
    _nnt_inflation   = float("nan")

# ── noise impact helper ───────────────────────────────────────────────────────

def _noise_auc_row(nl: float) -> dict:
    rng_n = np.random.default_rng(int(seed))
    if gen_scale == SCALE_SQRT_RAW:
        tv_n = np.sqrt(tumor_volume); mh_n = np.sqrt(mean_heart_dose)
    elif gen_scale == SCALE_SQRT_Z:
        _sg = np.sqrt(tumor_volume); _sm = np.sqrt(mean_heart_dose)
        tv_n = (_sg - _sg.mean()) / _sg.std()
        mh_n = (_sm - _sm.mean()) / _sm.std()
    else:
        tv_n = (tumor_volume    - tumor_volume.mean())    / tumor_volume.std()
        mh_n = (mean_heart_dose - mean_heart_dose.mean()) / mean_heart_dose.std()
    logit_n = _display_int + b1 * tv_n + b2_val * mh_n + nl * rng_n.standard_normal(n_patients)
    mort_n  = rng_n.binomial(1, 1.0 / (1.0 + np.exp(-logit_n))).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X_feat, mort_n,
                                           test_size=test_size, random_state=int(seed))
    pa, pb, pc = make_pipe(), make_pipe(), make_pipe()
    pa.fit(Xtr[:, [0]], ytr); pb.fit(Xtr[:, [1]], ytr); pc.fit(Xtr, ytr)
    fa, ta, _ = roc_curve(yte, pa.predict_proba(Xte[:, [0]])[:, 1])
    fb, tb, _ = roc_curve(yte, pb.predict_proba(Xte[:, [1]])[:, 1])
    fc, tc, _ = roc_curve(yte, pc.predict_proba(Xte)[:, 1])
    return {
        "Noise σ":              nl,
        "Obs. mortality":       f"{mort_n.mean()*100:.1f} %",
        f"AUC {fl[0]}":         f"{auc(fa, ta):.3f}",
        f"AUC {fl[1]}":         f"{auc(fb, tb):.3f}",
        f"AUC {fl[0]}+{fl[1]}": f"{auc(fc, tc):.3f}",
    }

# ── colour / style maps ───────────────────────────────────────────────────────

MC = {1: "#e74c3c", 0: "#2ecc71"}
ML = {1: "Died", 0: "Survived"}
MS = {"A": ("#2980b9", "solid"), "B": ("#e67e22", "dashed"), "C": ("#27ae60", "dashdot")}


# ── tabs ──────────────────────────────────────────────────────────────────────

(tab_core, tab_article, tab_proton, tab_dose,
 tab_datachk, tab_adv) = st.tabs([
    "Core results",
    "Article match",
    "Proton simulation",
    "Dose curves",
    "Data checks",
    "Advanced diagnostics",
])

# ── tab: Core results ─────────────────────────────────────────────────────────

with tab_core:
    st.markdown(
        "> **This simulator separates three things that are often mixed together: "
        "true causal effects, statistical prediction, and model-based treatment benefit. "
        "The key question is whether an apparent MHD coefficient can arise without MHD "
        "being the true driver.**"
    )

    st.subheader("True causal structure")
    if survival_scenario == SCEN_A:
        st.code(
            "GTV ──► mortality\n"
            "GTV ──► MHD\n"
            "\n"
            "MHD has NO direct path to mortality.",
            language=None,
        )
        st.info(
            "**Scenario A:** MHD has no direct causal effect. "
            "Any fitted MHD coefficient must come from its correlation with GTV, "
            "random noise, or other hidden structure."
        )
    elif survival_scenario == SCEN_B:
        st.code(
            "GTV ──► mortality\n"
            "MHD ──► mortality\n"
            "GTV ──► MHD",
            language=None,
        )
        st.success(
            "**Scenario B:** MHD has a true causal effect in the simulation. "
            "The fitted MHD coefficient reflects a real pathway."
        )
    else:
        st.code(
            "logit(p_mortality) = −1.3409 + 0.059·√GTV + 0.2635·√MHD\n"
            "\n"
            "Both GTV and MHD are causal (article formula).",
            language=None,
        )
        st.success(
            "**Scenario C:** Data generated directly from the Van Loon et al. formula. "
            "Both GTV and MHD are causal."
        )

    st.divider()

    st.subheader("Fitted combined model:  mortality ~ " + " + ".join(fl))
    r1, r2 = st.columns([3, 1])
    with r1:
        st.dataframe(coef_df, use_container_width=True, hide_index=True)
        st.caption(
            f"Predictors: {', '.join(fl)} (StandardScaler applied for std. coefficients). "
            "Outcome: 24-month mortality (1 = died). "
            "p-values via Wald test.  *** p<0.001  ** p<0.01  * p<0.05  ns p≥0.05"
        )
    with r2:
        st.metric("AUC (test set)",            f"{auc_C:.3f}")
        st.metric("Observed mortality",         f"{obs_mort_rate*100:.1f}%")
        st.metric("GTV-MHD correlation (r)",    f"{pearson_r:.3f}")

    if survival_scenario != SCEN_C:
        _gen_uses_sqrt = gen_scale in (SCALE_SQRT_RAW, SCALE_SQRT_Z)
        if _gen_uses_sqrt != use_sqrt:
            st.warning(
                "⚠️ **Scale mismatch:** the true mortality formula uses "
                f"{'√-transformed' if _gen_uses_sqrt else 'raw'} predictors, "
                f"but the fitted model uses {'√-transformed' if use_sqrt else 'raw'} predictors. "
                "Coefficients b1/b2 cannot be directly compared to fitted coefficients."
            )

    st.divider()

    with st.expander("True data-generating formula", expanded=False):
        if survival_scenario == SCEN_C:
            st.markdown(
                f"**Scenario C — Article formula**\n\n"
                f"```\nlogit(p) = {_display_int:.4f} + 0.0590·√GTV + 0.2635·√MHD\n```\n\n"
                f"{'*(intercept calibrated to selected baseline)*' if calibrate_article else '*(article intercept −1.3409)*'}"
            )
        else:
            _noise_line = f" + N(0, {surv_noise:.2f})" if surv_noise > 0 else ""
            if gen_scale == SCALE_SQRT_RAW:
                _pred_gtv, _pred_mhd = "√GTV", "√MHD"
                _scale_note = "*(unstandardised √-values — same scale as Van Loon article)*"
            elif gen_scale == SCALE_SQRT_Z:
                _pred_gtv, _pred_mhd = "z_√GTV", "z_√MHD"
                _scale_note = "z\\_√GTV = (√GTV − mean)/SD,  z\\_√MHD = (√MHD − mean)/SD"
            else:
                _pred_gtv, _pred_mhd = "z_GTV", "z_MHD"
                _scale_note = "z\\_GTV = (GTV − mean)/SD,  z\\_MHD = (MHD − mean)/SD"
            _b2_line = f" + {b2_val:.4f}·{_pred_mhd}" if survival_scenario == SCEN_B else ""
            st.markdown(
                f"**Scenario {'A' if survival_scenario == SCEN_A else 'B'}**\n\n"
                f"```\nlogit(p) = {_display_int:.4f}"
                f" + {b1:.4f}·{_pred_gtv}{_b2_line}{_noise_line}\n```\n\n"
                f"{_scale_note}"
            )
        param_rows = [
            ("Intercept (logit)", f"{_display_int:.4f}", f"= logit({_baseline_mort*100:.1f} %)"),
            ("b1 — GTV coefficient", f"{b1:.4f}",
             "fixed article value" if survival_scenario == SCEN_C else "user-set"),
        ]
        if survival_scenario != SCEN_A:
            param_rows.append((
                "b2 — MHD coefficient", f"{b2_val:.4f}",
                "fixed article value" if survival_scenario == SCEN_C else "user-set",
            ))
        if survival_scenario in (SCEN_A, SCEN_B):
            param_rows.append(("Noise σ", f"{surv_noise:.2f}", "N(0,σ) added to logit"))
        param_rows += [
            ("Target baseline mortality", f"{_baseline_mort*100:.1f} %", ""),
            ("Simulated mortality",       f"{obs_mort_rate*100:.1f} %",  ""),
        ]
        st.dataframe(
            pd.DataFrame(param_rows, columns=["Parameter", "Value", "Note"]),
            use_container_width=True, hide_index=True,
        )


# ── tab: Article match ────────────────────────────────────────────────────────

with tab_article:
    st.markdown(
        "Compare current simulation output to the Van Loon et al. article targets. "
        "These targets describe what the article reports, not what the true causal "
        "structure must be."
    )

    _targets = {
        "Mortality":       (obs_mort_rate,     _FIT_TARGETS["mortality"], ".1%"),
        "AUC":             (auc_C,             _FIT_TARGETS["auc"],       ".3f"),
        "β √GTV (unstd.)": (_beta_gtv_unstd,   _FIT_TARGETS["beta_gtv"], ".4f"),
        "β √MHD (unstd.)": (_beta_mhd_unstd,   _FIT_TARGETS["beta_mhd"], ".4f"),
    }

    match_rows = []
    for metric, (current, target, fmt) in _targets.items():
        if np.isnan(current):
            match_rows.append({
                "Metric":     metric,
                "Current":    "n/a (enable √ model)",
                "Target":     format(target, fmt),
                "Difference": "—",
                "Status":     "⚠️ unavailable",
            })
            continue
        diff    = current - target
        curr_str = f"{current:.1%}" if fmt == ".1%" else format(current, fmt)
        diff_str = f"{diff:+.1%}"   if fmt == ".1%" else f"{diff:+{fmt}}"
        pct_off  = abs(diff / target) if target != 0 else float("nan")
        status   = "✅ close" if pct_off < 0.10 else ("⚠️ moderate" if pct_off < 0.25 else "❌ far")
        match_rows.append({
            "Metric":     metric,
            "Current":    curr_str,
            "Target":     format(target, fmt),
            "Difference": diff_str,
            "Status":     status,
        })

    st.dataframe(pd.DataFrame(match_rows), use_container_width=True, hide_index=True)
    st.caption(
        "β values require √-transformed predictors (enable in Advanced controls). "
        "Status: ✅ within 10 % of target · ⚠️ within 25 % · ❌ more than 25 % off."
    )

    if not use_sqrt:
        st.info(
            "Enable **Use √-transformed predictors in model** (Advanced controls) "
            "to compute unstandardised β coefficients comparable to the article."
        )

    st.divider()
    st.subheader("Fit-to-article optimizer")

    if not _fit_overridden:
        st.info(
            "The optimizer is off. Enable **Optimise parameters to article targets** "
            "in Advanced controls to run a grid search over copula ρ and noise level."
        )
    else:
        oc1, oc2, oc3, oc4 = st.columns(4)
        oc1.metric("β_√GTV (fitted, unstd.)", f"{_beta_gtv_unstd:.4f}",
                   delta=f"{_beta_gtv_unstd - 0.059:+.4f} vs 0.059")
        oc2.metric("β_√MHD (fitted, unstd.)", f"{_beta_mhd_unstd:.4f}",
                   delta=f"{_beta_mhd_unstd - 0.2635:+.4f} vs 0.264")
        oc3.metric("AUC (test set)", f"{auc_C:.3f}",
                   delta=f"{auc_C - 0.64:+.3f} vs 0.640")
        oc4.metric("Mortality (simulated)", f"{obs_mort_rate*100:.1f}%",
                   delta=f"{(obs_mort_rate - 0.506)*100:+.1f}pp vs 50.6%")

        st.success(
            f"Optimizer found: copula ρ = **{_fit_result['rho']:.2f}**, "
            f"noise σ = **{_fit_result['noise']:.2f}**  \n"
            f"Optimizer loss = {_fit_result.get('loss', float('nan')):.6f}"
        )
        st.warning(
            "**Equal coefficients do not prove causality.**  \n"
            "Different data-generating mechanisms can produce the same fitted model. "
            "Matching the article's β values does not mean the causal structure matches."
        )

        if survival_scenario == SCEN_A:
            st.subheader("Scenario A — how can β_√MHD be large with b2 = 0?")
            _cols = st.columns(3)
            _cols[0].metric("True b2 (MHD causal effect)", "0.000")
            _cols[1].metric("Fitted β_√MHD (unstd.)",
                            f"{_beta_mhd_unstd:.4f}" if not np.isnan(_beta_mhd_unstd) else "n/a")
            _cols[2].metric("Observed GTV-MHD correlation (r)", f"{pearson_r:.3f}")

            if not np.isnan(_beta_mhd_unstd) and abs(_beta_mhd_unstd) > 0.05:
                if abs(pearson_r) > 0.30:
                    st.warning(
                        f"The fitted β_√MHD = {_beta_mhd_unstd:.4f} despite b2 = 0. "
                        f"The observed GTV-MHD correlation (r = {pearson_r:.3f}) is the driver: "
                        "MHD carries GTV signal into the model."
                    )
                else:
                    st.warning(
                        f"The fitted β_√MHD = {_beta_mhd_unstd:.4f} despite b2 = 0 "
                        f"and low correlation (r = {pearson_r:.3f}). "
                        "This article-like MHD coefficient requires strong correlation "
                        "or another hidden driver — check optimizer debug info in Advanced diagnostics."
                    )
            else:
                st.info(
                    f"With low GTV-MHD correlation (r = {pearson_r:.3f}) and b2 = 0, "
                    "the fitted β_√MHD stays near zero — confounding is weak."
                )


# ── tab: Proton simulation ────────────────────────────────────────────────────

with tab_proton:
    st.markdown(
        "This tab separates model-based treatment selection from true causal treatment benefit. "
        "The model may predict that reducing MHD lowers mortality, but if MHD is only partly "
        "causal, the true absolute benefit may be smaller and the true NNT much higher."
    )

    if proton_reduction_mode == PROTON_MULT:
        st.info(
            f"**Multiplicative mode:** mhd_proton = mhd_photon × {mhd_reduction_factor:.2f}  "
            f"(MHD reduced to {mhd_reduction_factor*100:.0f}% of photon value, min. 0 Gy)"
        )
    else:
        st.info(
            f"**Absolute mode:** mhd_proton = mhd_photon − {mhd_absolute_reduction:.1f} Gy  "
            f"(fixed reduction per patient, min. 0 Gy)"
        )

    # ── Section A: Model-predicted benefit ────────────────────────────────────
    st.subheader("A. Model-predicted proton benefit")
    st.caption(
        "Calculated from the fitted model after substituting reduced MHD. "
        "This is what the model says — not necessarily what would happen in reality."
    )

    pm1, pm2, pm3, pm4 = st.columns(4)
    pm1.metric("Mean mortality (photon)",   f"{p_mort_photon.mean()*100:.1f}%")
    pm2.metric("Mean mortality (proton)",   f"{p_mort_proton.mean()*100:.1f}%")
    pm3.metric("Mean predicted Δ",          f"{_mean_pred_delta*100:.2f}%")
    pm4.metric("Predicted NNT (1/mean Δ)", fmt_f(_pred_nnt_mean))

    # ── Section B: True simulated benefit ─────────────────────────────────────
    st.subheader("B. True simulated proton benefit")

    if not apply_true_proton:
        if survival_scenario == SCEN_A:
            st.warning(
                "**Scenario A — no forced true treatment effect.**  \n"
                "MHD is not causal in this simulation. Proton MHD reduction only changes "
                "model-predicted risk. The true simulated mortality does not improve.  \n"
                "Expected true benefit ≈ 0. If the model predicts benefit, it is "
                "**model-predicted benefit only** — not a real outcome improvement."
            )
        elif survival_scenario == SCEN_B:
            st.info(
                "**Scenario B — MHD is causal, but forced true treatment effect is off.**  \n"
                "MHD has a true causal effect (b2 > 0), but the true proton benefit is not "
                "separately computed here because 'Enable forced true treatment effect' is off. "
                "Enable it in Advanced controls to quantify the true benefit."
            )
        else:
            st.info(
                "**Scenario C — article formula.**  \n"
                "MHD is causal in the article formula. Enable the forced true treatment effect "
                "in Advanced controls to quantify the true simulated benefit."
            )
        pt1, pt2 = st.columns(2)
        pt1.metric("Mean true Δ", "0 % (no true effect active)")
        pt2.metric("True causal NNT", "No true benefit")
    else:
        st.success(
            f"**Forced true treatment effect is ON** (strength = {proton_effect_strength:.2f}).  \n"
            "The true simulated mortality is reduced proportionally to MHD reduction. "
            "This applies regardless of the scenario causal structure — it is a manual override."
        )
        pt1, pt2, pt3, pt4 = st.columns(4)
        pt1.metric("Mean true Δ",      f"{_mean_true_delta*100:.2f}%")
        pt2.metric("True causal NNT",  fmt_f(_true_nnt_mean))
        pt3.metric("Mean predicted Δ", f"{_mean_pred_delta*100:.2f}%")
        pt4.metric("Predicted NNT",    fmt_f(_pred_nnt_mean))

    # ── NNT inflation ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("NNT inflation factor")

    if true_delta is None:
        st.markdown(
            "**NNT inflation factor:** not computable — no true treatment effect is active.  \n"
            "Enable the forced true treatment effect in Advanced controls to see inflation."
        )
    else:
        ni1, ni2, ni3 = st.columns(3)
        ni1.metric("Predicted NNT",   fmt_f(_pred_nnt_mean))
        ni2.metric("True causal NNT", fmt_f(_true_nnt_mean))
        ni3.metric(
            "NNT inflation factor  (true NNT / predicted NNT)",
            fmt_f(_nnt_inflation),
            help="1.0 = no inflation. >1 = true NNT is higher than model predicts.",
        )

        if not np.isnan(_nnt_inflation):
            if _nnt_inflation > 3:
                st.error(
                    f"**NNT inflation factor = {_nnt_inflation:.1f}** — "
                    "the true NNT is much higher than the model-predicted NNT. "
                    "The model may substantially overstate the treatment benefit."
                )
            elif _nnt_inflation > 1.5:
                st.warning(
                    f"**NNT inflation factor = {_nnt_inflation:.1f}** — "
                    "the true NNT is higher than the model-predicted NNT."
                )
            else:
                st.success(
                    f"**NNT inflation factor = {_nnt_inflation:.1f}** — "
                    "model-predicted and true NNT are broadly consistent."
                )

    # ── Delta histogram ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("Distribution of absolute mortality reduction per patient")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(delta_model * 100, bins=40, color="#2980b9", edgecolor="none",
            alpha=0.8, label="Model-predicted Δ")
    if true_delta is not None:
        ax.hist(true_delta * 100, bins=40, color="#e67e22", edgecolor="none",
                alpha=0.6, label="True simulated Δ (forced effect)")
        ax.legend(fontsize=9)
    ax.set_xlabel("Absolute mortality reduction Δ (percentage points)")
    ax.set_ylabel("Number of patients")
    ax.set_title(f"Predicted Δ distribution  (MHD reduction: {_proton_label})")
    fig.tight_layout()
    centered(fig)

    # ── NNT comparison table ──────────────────────────────────────────────────
    st.subheader("NNT table by predicted benefit bin")
    nnt_cmp_df = build_nnt_comparison_table(delta_model, true_delta, n_patients)
    st.dataframe(nnt_cmp_df, use_container_width=True, hide_index=True)
    st.caption(
        "Binned on **predicted** Δ. "
        "Predicted NNT = 1 / mean predicted Δ within bin. "
        "True causal NNT = 1 / mean true Δ within bin. "
        "NNT inflation = true NNT / predicted NNT — higher means the model overstates benefit."
    )

    # ── Patients per bin bar chart ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(_BLABELS, nnt_cmp_df["N"].values,
                  color="#2980b9", edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, padding=2, fontsize=7)
    ax.set_xlabel("Predicted absolute mortality reduction Δ")
    ax.set_ylabel("Number of patients")
    ax.set_title(f"Patients per predicted Δ bin  (MHD reduction: {_proton_label})")
    ax.tick_params(axis="x", rotation=40, labelsize=7)
    fig.tight_layout()
    centered(fig)

    if survival_scenario == SCEN_A and not apply_true_proton:
        st.caption(
            "Scenario A, no forced true effect: patients with high predicted Δ have large GTV "
            "and correspondingly high MHD. The model sees MHD reduction as beneficial because "
            "MHD carries GTV signal. No true outcome improvement occurs."
        )


# ── tab: Dose curves ──────────────────────────────────────────────────────────

with tab_dose:
    st.markdown("### Mortality vs dose — combined model")
    st.caption("All curves show **mortality probability** predicted by the fitted combined model.")

    if survival_scenario == SCEN_A:
        st.info(
            "Scenario A: the MHD curves below reflect GTV-MHD correlation, "
            "not a direct causal effect of MHD on mortality."
        )

    def _X(gtv_arr, mhd_arr):
        if use_sqrt:
            return np.column_stack([np.sqrt(gtv_arr), np.sqrt(mhd_arr)])
        return np.column_stack([gtv_arr, mhd_arr])

    gtv_med              = np.median(tumor_volume)
    mhd_med              = np.median(mean_heart_dose)
    gtv_p10, gtv_p90     = np.percentile(tumor_volume,    [10, 90])
    mhd_p10, mhd_p90     = np.percentile(mean_heart_dose, [10, 90])

    tv_g  = np.linspace(tumor_volume.min(),    tumor_volume.max(),    300)
    mhd_g = np.linspace(mean_heart_dose.min(), mean_heart_dose.max(), 300)

    mort_vs_gtv = pipe_C.predict_proba(_X(tv_g,  np.full_like(tv_g,  mhd_med)))[:, 1]
    mort_vs_mhd = pipe_C.predict_proba(_X(np.full_like(mhd_g, gtv_med), mhd_g))[:, 1]

    dc1, dc2 = st.columns(2)
    with dc1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(tv_g, mort_vs_gtv, color="#e74c3c", lw=2)
        ax.axvline(gtv_med, color="gray", lw=1, linestyle=":",
                   label=f"Median GTV = {gtv_med:.0f} cc")
        ax.set_xlabel("GTV (cc)")
        ax.set_ylabel("Predicted 24-month mortality")
        ax.set_title(f"Mortality vs GTV\n(MHD fixed at median {mhd_med:.1f} Gy)")
        ax.set_ylim(0, 1); ax.legend(fontsize=8)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    with dc2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(mhd_g, mort_vs_mhd, color="#e74c3c", lw=2)
        ax.axvline(mhd_med, color="gray", lw=1, linestyle=":",
                   label=f"Median MHD = {mhd_med:.1f} Gy")
        ax.set_xlabel("Mean Heart Dose (Gy)")
        ax.set_ylabel("Predicted 24-month mortality")
        ax.set_title(f"Mortality vs MHD\n(GTV fixed at median {gtv_med:.0f} cc)")
        ax.set_ylim(0, 1); ax.legend(fontsize=8)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    show_stratified = st.checkbox("Show stratified curves (low / median / high)", value=False)

    if show_stratified:
        st.markdown("#### Mortality vs GTV at low / median / high MHD")
        fig, ax = plt.subplots(figsize=(7, 4))
        for mhd_val, label, color in [
            (mhd_p10, f"MHD p10 = {mhd_p10:.1f} Gy",  "#1a9641"),
            (mhd_med, f"MHD median = {mhd_med:.1f} Gy", "#fdae61"),
            (mhd_p90, f"MHD p90 = {mhd_p90:.1f} Gy",   "#d7191c"),
        ]:
            ax.plot(tv_g,
                    pipe_C.predict_proba(_X(tv_g, np.full_like(tv_g, mhd_val)))[:, 1],
                    color=color, lw=2, label=label)
        ax.set_xlabel("GTV (cc)"); ax.set_ylabel("Predicted 24-month mortality")
        ax.set_title("Mortality vs GTV — low / median / high MHD")
        ax.set_ylim(0, 1); ax.legend(fontsize=8)
        fig.tight_layout(); centered(fig)

        st.markdown("#### Mortality vs MHD at low / median / high GTV")
        fig, ax = plt.subplots(figsize=(7, 4))
        for gtv_val, label, color in [
            (gtv_p10, f"GTV p10 = {gtv_p10:.0f} cc",   "#1a9641"),
            (gtv_med, f"GTV median = {gtv_med:.0f} cc",  "#fdae61"),
            (gtv_p90, f"GTV p90 = {gtv_p90:.0f} cc",   "#d7191c"),
        ]:
            ax.plot(mhd_g,
                    pipe_C.predict_proba(_X(np.full_like(mhd_g, gtv_val), mhd_g))[:, 1],
                    color=color, lw=2, label=label)
        ax.set_xlabel("Mean Heart Dose (Gy)"); ax.set_ylabel("Predicted 24-month mortality")
        ax.set_title("Mortality vs MHD — low / median / high GTV")
        ax.set_ylim(0, 1); ax.legend(fontsize=8)
        fig.tight_layout(); centered(fig)

        if survival_scenario == SCEN_A:
            st.caption(
                "The MHD curves show model-predicted associations driven by GTV-MHD "
                "correlation — not a direct causal effect."
            )


# ── tab: Data checks ──────────────────────────────────────────────────────────

with tab_datachk:
    st.markdown("### Data summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients",               n_patients)
    c2.metric("24-month mortality",     f"{obs_mort_rate*100:.1f}%")
    c3.metric("Pearson r (GTV ↔ MHD)", f"{pearson_r:.3f}")
    c4.metric("p-value (correlation)",  fmt_pval(pearson_p))

    st.markdown("#### Descriptive statistics")
    vc1, vc2 = st.columns(2)
    with vc1:
        st.dataframe(
            pd.DataFrame({"GTV (cc)": describe_arr(tumor_volume),
                          "MHD (Gy)": describe_arr(mean_heart_dose)}),
            use_container_width=True,
        )
        if dist_mode == DIST_ARTICLE:
            st.caption(
                "Reference: GTV mean ≈ 110 cc, median ≈ 69 cc, SD ≈ 130 cc; "
                "MHD mean ≈ 12 Gy, median ≈ 11 Gy, SD ≈ 8.2 Gy."
            )
    with vc2:
        st.markdown("**Baseline vs simulated mortality**")
        st.dataframe(
            pd.DataFrame({
                "":                   ["Target", "Simulated"],
                "2-year survival":    [f"{baseline_survival*100:.1f}%",
                                       f"{(1-obs_mort_rate)*100:.1f}%"],
                "24-month mortality": [f"{_baseline_mort*100:.1f}%",
                                       f"{obs_mort_rate*100:.1f}%"],
            }),
            use_container_width=True, hide_index=True,
        )
        if survival_scenario == SCEN_C and not calibrate_article:
            st.caption("Scenario C uses fixed article intercept (−1.3409).")

    st.markdown("#### MHD reduction summary (proton therapy)")
    st.dataframe(
        pd.DataFrame({
            "":                ["Photon", "Proton", "Reduction"],
            "Mean MHD (Gy)":   [f"{mean_heart_dose.mean():.2f}",
                                 f"{mhd_proton.mean():.2f}",
                                 f"{(mean_heart_dose - mhd_proton).mean():.2f}"],
            "Median MHD (Gy)": [f"{np.median(mean_heart_dose):.2f}",
                                 f"{np.median(mhd_proton):.2f}",
                                 f"{np.median(mean_heart_dose - mhd_proton):.2f}"],
        }),
        use_container_width=True, hide_index=True,
    )
    st.caption(f"Proton MHD reduction mode: {proton_reduction_mode} ({_proton_label})")

    st.markdown("#### Distributions")
    show_cumul = st.checkbox("Show cumulative histograms", value=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, col, xlabel in zip(
        axes,
        ["tumor_volume_cc", "mean_heart_dose_gy"],
        ["GTV (cc)", "Mean Heart Dose (Gy)"],
    ):
        for val in [0, 1]:
            data = df.loc[df["mortality"] == val, col].values
            if show_cumul:
                ax.hist(data, bins=60, density=True, cumulative=True,
                        alpha=0.6, color=MC[val], label=ML[val], edgecolor="none")
                ax.set_ylabel("Cumulative fraction")
                ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
            else:
                ax.hist(data, bins=30, alpha=0.6, color=MC[val],
                        label=ML[val], edgecolor="none")
                ax.set_ylabel("Number of patients")
        ax.set_xlabel(xlabel); ax.legend(fontsize=8)
    fig.suptitle(
        ("Cumulative distribution" if show_cumul else "Distribution") + " by mortality status",
        fontsize=12,
    )
    fig.tight_layout(); centered(fig)


# ── tab: Advanced diagnostics ─────────────────────────────────────────────────

with tab_adv:
    st.markdown("### Advanced diagnostics")

    # ── ROC comparison ────────────────────────────────────────────────────────
    st.subheader("ROC comparison")
    st.markdown(
        "> A good predictor does not need to be a good treatment target. "
        "MHD-only can predict mortality better than chance even when MHD is not causal, "
        "because MHD correlates with GTV."
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    for key, name, f, t, a_val in [
        ("A", "GTV only",  fpr_A, tpr_A, auc_A),
        ("B", "MHD only",  fpr_B, tpr_B, auc_B),
        ("C", "GTV + MHD", fpr_C, tpr_C, auc_C),
    ]:
        color, ls = MS[key]
        ax.plot(f, t, color=color, lw=2, linestyle=ls,
                label=f"{name},  AUC = {a_val:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", lw=1, label="Chance (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC comparison — 24-month mortality (test set)")
    ax.legend(loc="lower right", fontsize=9); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); centered(fig)

    st.dataframe(roc_summary, use_container_width=True, hide_index=True)

    if not mhd_is_causal and auc_B > 0.55:
        st.warning(
            f"MHD-only model (AUC = {auc_B:.3f}) predicts mortality better than chance "
            f"despite MHD having no causal effect (r(GTV,MHD) = {pearson_r:.2f})."
        )

    st.markdown("#### Fitted mortality functions")
    tv_grid  = np.linspace(tumor_volume.min(), tumor_volume.max(), 200)
    mhd_grid = np.linspace(mean_heart_dose.min(), mean_heart_dose.max(), 200)
    tv_in    = np.sqrt(tv_grid)  if use_sqrt else tv_grid
    mhd_in   = np.sqrt(mhd_grid) if use_sqrt else mhd_grid
    _mhd_p10, _mhd_p50, _mhd_p90 = (np.percentile(mean_heart_dose, q) for q in [10, 50, 90])

    sv1, sv2, sv3 = st.columns(3)
    with sv1:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(tumor_volume, mortality, c=[MC[m] for m in mortality],
                   alpha=0.25, s=10, linewidths=0, zorder=1)
        ax.plot(tv_grid, pipe_A.predict_proba(tv_in.reshape(-1, 1))[:, 1],
                color=MS["A"][0], lw=2, zorder=2)
        ax.set_xlabel("GTV (cc)"); ax.set_ylabel("Predicted mortality")
        ax.set_title("GTV only"); ax.set_ylim(0, 1)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    with sv2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(mean_heart_dose, mortality, c=[MC[m] for m in mortality],
                   alpha=0.25, s=10, linewidths=0, zorder=1)
        ax.plot(mhd_grid, pipe_B.predict_proba(mhd_in.reshape(-1, 1))[:, 1],
                color=MS["B"][0], lw=2, zorder=2)
        ax.set_xlabel("Mean Heart Dose (Gy)"); ax.set_ylabel("Predicted mortality")
        ax.set_title("MHD only"); ax.set_ylim(0, 1)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    with sv3:
        fig, ax = plt.subplots(figsize=(5, 4))
        for mhd_val, label, color in [
            (_mhd_p10, f"MHD p10 = {_mhd_p10:.1f} Gy", "#1a9641"),
            (_mhd_p50, f"MHD med = {_mhd_p50:.1f} Gy",  "#fdae61"),
            (_mhd_p90, f"MHD p90 = {_mhd_p90:.1f} Gy",  "#d7191c"),
        ]:
            Xc = (np.column_stack([np.sqrt(tv_grid), np.full_like(tv_grid, np.sqrt(mhd_val))])
                  if use_sqrt else
                  np.column_stack([tv_grid, np.full_like(tv_grid, mhd_val)]))
            ax.plot(tv_grid, pipe_C.predict_proba(Xc)[:, 1], color=color, lw=2, label=label)
        ax.set_xlabel("GTV (cc)"); ax.set_ylabel("Predicted mortality")
        ax.set_title("GTV + MHD"); ax.set_ylim(0, 1)
        ax.legend(fontsize=6.5, loc="lower right")
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    st.caption(
        "Red = died, green = survived. "
        "In Scenario A the MHD curves reflect correlation with GTV, not a direct causal effect."
    )

    # ── Scatter plot ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Scatter: GTV ↔ MHD")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(tumor_volume, mean_heart_dose,
               c=[MC[m] for m in mortality], alpha=0.4, s=18, linewidths=0)
    m_r, b_r, *_ = stats.linregress(tumor_volume, mean_heart_dose)
    xl = np.array([tumor_volume.min(), tumor_volume.max()])
    ax.plot(xl, m_r * xl + b_r, color="#2c3e50", lw=1.5,
            label=f"y = {m_r:.2f}x + {b_r:.1f}  (r = {pearson_r:.2f}, p = {fmt_pval(pearson_p)})")
    h = [Patch(color=MC[v], alpha=0.7, label=ML[v]) for v in [1, 0]]
    l1 = ax.legend(handles=h, loc="upper left", fontsize=8)
    ax.add_artist(l1); ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("GTV (cc)"); ax.set_ylabel("Mean Heart Dose (Gy)")
    ax.set_title("GTV vs MHD — colour = mortality status")
    fig.tight_layout(); centered(fig)

    # ── Correlation matrix ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Correlation matrix")
    corr_df = df[["tumor_volume_cc", "mean_heart_dose_gy", "mortality"]].corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr_df, annot=True, fmt=".3f", cmap="RdYlGn",
                center=0, vmin=-1, vmax=1, mask=np.eye(3, dtype=bool),
                ax=ax, linewidths=0.5)
    ax.set_title("Pearson correlation matrix")
    fig.tight_layout(); centered(fig)
    st.caption(
        "MHD correlates with both GTV and mortality even in Scenario A "
        "(no causal MHD → mortality path)."
    )

    # ── Confusion matrix ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("Confusion matrix — combined model (test set)")
    cm = confusion_matrix(y_test, y_pred_C)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=["Survived", "Died"],
                yticklabels=["Survived", "Died"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix — 24-month mortality")
    fig.tight_layout(); centered(fig)

    # ── Noise impact ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Noise impact on AUC")
    if survival_scenario == SCEN_C:
        st.info(
            "Scenario C uses a fixed article formula without a noise parameter. "
            "Noise impact analysis is only available for Scenarios A and B."
        )
    else:
        st.markdown(
            "AUC across three models for increasing mortality noise levels, "
            "using the same GTV and MHD values."
        )
        noise_rows = [_noise_auc_row(nl) for nl in _NOISE_LEVELS]
        noise_df   = pd.DataFrame(noise_rows)
        st.dataframe(noise_df, use_container_width=True, hide_index=True)
        st.caption(f"Current setting: σ = {surv_noise:.2f}.")

        col_a = f"AUC {fl[0]}"
        col_b = f"AUC {fl[1]}"
        col_c = f"AUC {fl[0]}+{fl[1]}"
        fig, ax = plt.subplots(figsize=(7, 4))
        for col, color, ls, lab in [
            (col_a, MS["A"][0], MS["A"][1], f"{fl[0]} only"),
            (col_b, MS["B"][0], MS["B"][1], f"{fl[1]} only"),
            (col_c, MS["C"][0], MS["C"][1], f"{fl[0]} + {fl[1]}"),
        ]:
            ax.plot(_NOISE_LEVELS, [float(r[col]) for r in noise_rows],
                    color=color, linestyle=ls, lw=2, marker="o", ms=5, label=lab)
        ax.axvline(surv_noise, color="gray", linestyle=":", lw=1,
                   label=f"Current σ = {surv_noise:.2f}")
        ax.set_xlabel("Mortality noise level σ"); ax.set_ylabel("ROC AUC (test set)")
        ax.set_title("AUC vs mortality noise level")
        ax.legend(fontsize=8); ax.set_ylim(0.4, 1.0)
        fig.tight_layout(); centered(fig)

    # ── Formula expanders ─────────────────────────────────────────────────────
    st.divider()
    with st.expander("True logit function plot", expanded=False):
        if survival_scenario == SCEN_C:
            x_range = np.linspace(0, 15, 300)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(x_range, 0.059  * x_range, color=MS["A"][0], lw=2, label="0.0590 · √GTV")
            ax.plot(x_range, 0.2635 * x_range, color=MS["B"][0], lw=2,
                    linestyle="--", label="0.2635 · √MHD")
            ax.axhline(0, color="gray", lw=0.8, linestyle=":")
            ax.set_xlabel("Predictor on √-scale"); ax.set_ylabel("Logit contribution")
            ax.set_title("Logit contributions per predictor (Scenario C)")
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
            fig.tight_layout(); centered(fig)
        else:
            x_range = np.linspace(0, 15, 300) if gen_scale == SCALE_SQRT_RAW else np.linspace(-3, 3, 300)
            _xlabel = "Predictor on √-scale (unstandardised)" if gen_scale == SCALE_SQRT_RAW else "Standardised predictor"
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(x_range, b1 * x_range, color=MS["A"][0], lw=2, label=f"{b1:.4f} · GTV term")
            if survival_scenario == SCEN_B:
                ax.plot(x_range, b2_val * x_range, color=MS["B"][0], lw=2,
                        linestyle="--", label=f"{b2_val:.4f} · MHD term")
            ax.axhline(0, color="gray", lw=0.8, linestyle=":")
            ax.set_xlabel(_xlabel); ax.set_ylabel("Logit contribution")
            ax.set_title("True logit contributions by predictor")
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
            fig.tight_layout(); centered(fig)
        st.caption(
            "Logit contribution = linear effect on the logit scale. "
            "Probability follows p = 1 / (1 + exp(−logit)). "
            "A logit increase of +1 around logit = 0 raises probability from 50% to ~73%."
        )

    # ── Optimizer debug info ──────────────────────────────────────────────────
    with st.expander("Optimizer debug info", expanded=False):
        _opt_rho   = _fit_result["rho"]   if _fit_overridden else float(target_corr if dist_mode == DIST_ARTICLE else 0.0)
        _opt_noise = _fit_result["noise"] if _fit_overridden else surv_noise
        _debug_rows = [
            ("Active scenario",                     survival_scenario),
            ("Optimizer used b2 = 0 (Scenario A)?",
             "Yes" if survival_scenario == SCEN_A else "No — b2 = 0.2635"),
            ("Optimised copula ρ",                  f"{_opt_rho:.4f}"),
            ("Optimised noise σ",                   f"{_opt_noise:.4f}"),
            ("True b1 (GTV effect)",                f"{b1:.4f}"),
            ("True b2 (MHD effect)",                f"{b2_val:.4f}"),
            ("Observed Pearson r(GTV, MHD)",        f"{pearson_r:.4f}"),
            ("Fitted β_√GTV (unstd.)",
             f"{_beta_gtv_unstd:.4f}" if not np.isnan(_beta_gtv_unstd) else "n/a (use_sqrt=False)"),
            ("Fitted β_√MHD (unstd.)",
             f"{_beta_mhd_unstd:.4f}" if not np.isnan(_beta_mhd_unstd) else "n/a (use_sqrt=False)"),
            ("Fitted intercept (unstd.)",
             f"{_intercept_unstd:.4f}" if not np.isnan(_intercept_unstd) else "n/a"),
            ("AUC (test set, combined model)",      f"{auc_C:.4f}"),
            ("Simulated mortality rate",            f"{obs_mort_rate*100:.2f}%"),
        ]
        if _fit_overridden:
            _debug_rows += [
                ("Optimizer internal β_√GTV",    f"{_fit_result['beta_gtv']:.4f}"),
                ("Optimizer internal β_√MHD",    f"{_fit_result['beta_mhd']:.4f}"),
                ("Optimizer internal AUC",       f"{_fit_result['auc']:.4f}"),
                ("Optimizer internal mortality", f"{_fit_result['mortality']*100:.2f}%"),
                ("Optimizer loss",               f"{_fit_result.get('loss', float('nan')):.6f}"),
            ]
        st.dataframe(
            pd.DataFrame(_debug_rows, columns=["Parameter", "Value"]),
            use_container_width=True, hide_index=True,
        )
        if _fit_overridden and survival_scenario == SCEN_A:
            st.info(
                "**Scenario A — optimizer used b2 = 0.**  \n"
                "Fitted β_√MHD on the generated dataset can only be nonzero "
                "through GTV-MHD correlation. With ρ ≈ 0 and b2 = 0, "
                "β_√MHD (unstd.) should stay near zero."
            )

    # ── Raw data ──────────────────────────────────────────────────────────────
    with st.expander("Raw data (first 50 rows)", expanded=False):
        st.dataframe(df.head(50), use_container_width=True)




