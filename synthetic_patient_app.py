"""
Synthetic patient dataset — confounding demo (mortality framing).

Causal structure:
  Scenario A:  GTV ──► mortality   GTV ──► MHD   (MHD is confounder only)
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

# ── article calibration target ────────────────────────────────────────────────
# Van Loon et al.: GTV med ≈ 69 cc, mean ≈ 110 cc; MHD mean ≈ 12 Gy, SD ≈ 8.2 Gy
# Pearson r(GTV, MHD) ≈ 0.45 (reported in article)
_ARTICLE_TARGET_R = 0.45


@st.cache_data(show_spinner=False)
def _calibrate_correlation(
    target_pearson_r: float = _ARTICLE_TARGET_R,
    n: int = 5000,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Grid-search the Gaussian copula rho that best reproduces
    the article's observed Pearson r(GTV, MHD).
    Returns (best_rho, achieved_r).  Result is cached after first run.
    """
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


_FIT_TARGETS = {
    "beta_gtv":  0.0590,
    "beta_mhd":  0.2635,
    "auc":       0.640,
    "mortality": 0.506,
}
_FIT_WEIGHTS = {"beta_gtv": 1.0, "beta_mhd": 1.0, "auc": 2.0, "mortality": 2.0}


@st.cache_data(show_spinner=False)
def _fit_to_article(
    n: int = 500,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Grid search over (copula rho, noise) to minimise loss vs article targets.
    Generates data with SCEN_B causal structure + SCALE_SQRT_RAW, fits an
    unstandardized logistic model on sqrt(GTV) and sqrt(MHD), and evaluates the
    four target metrics. Returns the best parameter set and achieved metrics."""
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

            # mortality via SCEN_B / SCALE_SQRT_RAW with b1=0.059, b2=0.2635
            tv_gen  = np.sqrt(gtv)
            mhd_gen = np.sqrt(mhd)
            gen_sl  = _FIT_TARGETS["beta_gtv"] * tv_gen + _FIT_TARGETS["beta_mhd"] * mhd_gen
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
        "Gemiddelde": f"{arr.mean():.2f}",
        "SD":         f"{arr.std(ddof=1):.2f}",
        "Mediaan":    f"{np.median(arr):.2f}",
        "Min":        f"{arr.min():.2f}",
        "Max":        f"{arr.max():.2f}",
    }


def fmt_f(v, decimals=1):
    try:
        return "—" if np.isnan(float(v)) else f"{v:.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def build_nnt_table(delta: np.ndarray, n_total: int) -> pd.DataFrame:
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


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Mortaliteitsmodel — Confounding Demo", layout="wide")

# Read optimizer toggle from session state before the sidebar so that data
# generation (which runs before tabs) can see whether it is active.
fit_to_article = st.session_state.get("fit_to_article_toggle", False)

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Simulation settings")

    _fit_preview = fit_to_article   # alias used by slider captions below

    # ── A. Patient population ─────────────────────────────────────────────────
    st.markdown("### 👥 A — Patient population")
    st.caption("How synthetic patients are generated")

    dist_mode = st.selectbox(
        "Distribution mode",
        [DIST_ARTICLE, DIST_SIMPLE],
        help=(
            "**Article-like:** Log-normal GTV and gamma-distributed MHD "
            "(GTV mean≈110 cc, median≈69 cc; MHD mean≈12 Gy, median≈11 Gy). "
            "**Simple:** Normal distributions with adjustable parameters."
        ),
    )

    n_patients = st.slider(
        "Sample size", 100, 3000, 500, step=50,
        help="Number of synthetic patients.",
    )
    seed = st.number_input(
        "Random seed", value=42, step=1,
        help="Starting number for the pseudo-random generator.",
    )

    if dist_mode == DIST_SIMPLE:
        st.markdown("**Tumor volume (GTV)**")
        tv_mu    = st.slider("Mean GTV (cc)", 10.0, 200.0, 45.0, step=1.0)
        tv_sigma = st.slider("SD GTV (cc)",   5.0,  60.0, 15.0, step=1.0)
        st.markdown("**Mean Heart Dose = a·GTV + noise**")
        corr_a       = st.slider("Correlation coefficient a (GTV → MHD)", 0.0, 1.0, 0.5, step=0.05)
        mhd_noise_sd = st.slider("MHD noise level (Gy)", 0.1, 10.0, 3.0, step=0.1)
    else:
        target_pearson_r = st.slider(
            "Target GTV-MHD correlation (r)", 0.0, 0.9, 0.45, step=0.05,
            help=(
                "How strongly larger tumors tend to co-occur with higher heart dose. "
                "The simulation automatically generates a population with approximately "
                "this Pearson r between GTV and MHD."
            ),
        )
        if _fit_preview:
            st.caption("⚙️ *Overridden by article optimisation*")
        # Internally translate target Pearson r → copula ρ (grid search, cached).
        _cal_rho, _cal_r = _calibrate_correlation(target_pearson_r)
        target_corr = _cal_rho
        with st.expander("🔧 Correlation diagnostics", expanded=False):
            st.caption(
                f"**Target r:** {target_pearson_r:.2f}  \n"
                f"**Internal copula ρ:** {_cal_rho:.3f}  \n"
                f"**Expected achieved r:** {_cal_r:.3f}  \n"
                "Observed Pearson r in the simulated sample is shown in the "
                "**Data summary** tab."
            )

    # ── B. True causal mortality model ────────────────────────────────────────
    st.divider()
    st.markdown("### ⚙️ B — True causal mortality model")
    st.caption("What truly causes mortality in this simulation")

    survival_scenario = st.selectbox(
        "True causal mortality model",
        [SCEN_A, SCEN_B, SCEN_C],
        help=(
            "**A:** Mortality driven by GTV only. MHD correlates with GTV but has no causal effect. "
            "**B:** Both GTV and MHD have true causal effects on mortality. "
            "**C:** Article formula using √GTV and √MHD."
        ),
    )
    if survival_scenario in (SCEN_A, SCEN_B):
        _default_scale = SCALE_SQRT_RAW if dist_mode == DIST_ARTICLE else SCALE_Z_RAW
        gen_scale = st.selectbox(
            "Scale of the true mortality formula",
            [SCALE_Z_RAW, SCALE_SQRT_RAW, SCALE_SQRT_Z],
            index=[SCALE_Z_RAW, SCALE_SQRT_RAW, SCALE_SQRT_Z].index(_default_scale),
            key=f"gen_scale_{dist_mode}_{survival_scenario}",
            help=(
                "**Standardised raw GTV/MHD:** b1/b2 are standardised logit effects of raw values. "
                "**√GTV/MHD unstandardised:** b1/b2 on the same scale as the Van Loon article. "
                "**√GTV/MHD standardised:** b1/b2 are standardised logit effects of √-transformed values."
            ),
        )
        if gen_scale == SCALE_SQRT_RAW:
            _b1_label = "True causal √GTV effect on mortality"
            _b2_label = "True causal √MHD effect on mortality"
            _b1_def, _b1_max, _b1_step = 0.0590, 0.5,  0.005
            _b2_def, _b2_max, _b2_step = 0.2635, 1.0,  0.005
        elif gen_scale == SCALE_SQRT_Z:
            _b1_label = "True causal std. √GTV effect on mortality"
            _b2_label = "True causal std. √MHD effect on mortality"
            _b1_def, _b1_max, _b1_step = 2.0, 5.0, 0.1
            _b2_def, _b2_max, _b2_step = 1.0, 3.0, 0.1
        else:  # SCALE_Z_RAW
            _b1_label = "True causal GTV effect on mortality (logit, std.)"
            _b2_label = "True causal MHD effect on mortality (logit, std.)"
            _b1_def, _b1_max, _b1_step = 2.0, 5.0, 0.1
            _b2_def, _b2_max, _b2_step = 1.0, 3.0, 0.1
        b1 = st.slider(_b1_label, 0.0, _b1_max, _b1_def, step=_b1_step,
                       key=f"b1_{gen_scale}",
                       help="True causal effect of GTV on the logit scale.")
        if survival_scenario == SCEN_B:
            b2_val = st.slider(_b2_label, 0.0, _b2_max, _b2_def, step=_b2_step,
                               key=f"b2_{gen_scale}",
                               help="True causal effect of MHD on the logit scale.")
        else:
            b2_val = 0.0
        if gen_scale == SCALE_SQRT_RAW:
            st.caption("These coefficients are on the same scale as the Van Loon article formula.")
        surv_noise = st.slider(
            "Mortality complexity / noise", 0.0, 3.0, 1.0, step=0.1,
            help=(
                "Random variation on the logit scale not explained by GTV or MHD. "
                "Higher values make mortality harder to predict — reduces AUC."
            ),
        )
        if _fit_preview:
            st.caption("⚙️ *Overridden by article optimisation*")
    else:
        gen_scale = SCALE_SQRT_RAW  # SCEN_C uses sqrt; gen_scale not used for generation
        b1, b2_val, surv_noise = 0.059, 0.2635, 0.0
        st.info(
            "Mortality via article formula:\n\n"
            "`logit(p) = −1.3409 + 0.0590·√GTV + 0.2635·√MHD`\n\n"
            "Both GTV and MHD are causally active."
        )

    mhd_is_causal = survival_scenario in (SCEN_B, SCEN_C)

    st.markdown("**Baseline mortality**")
    baseline_survival = st.slider(
        "Baseline 2-year survival", 0.20, 0.90, 0.494, step=0.01,
        help=(
            "Average 2-year survival level before individual risk differences. "
            "The article reports ~49.4 % 2-year survival (= 50.6 % mortality)."
        ),
    )
    if survival_scenario == SCEN_C:
        calibrate_article = st.checkbox(
            "Calibrate article model to selected baseline",
            value=False,
            help=(
                "Adjusts only the intercept so that mean simulated mortality matches the selected baseline. "
                "The slopes (0.0590 and 0.2635) remain unchanged."
            ),
        )
    else:
        calibrate_article = False

    # ── C. Proton dose reduction model ────────────────────────────────────────
    st.divider()
    st.markdown("### ⚛️ C — Proton dose reduction model")
    st.caption("How proton therapy changes MHD — not mortality directly")

    st.info(
        "Proton therapy lowers MHD. True mortality benefit arises **only** through "
        "the true causal mortality model above.",
        icon="ℹ️",
    )

    proton_reduction_mode = st.selectbox(
        "Dose reduction mode",
        [PROTON_MULT, PROTON_ABS],
        help=(
            "**A — Multiplicative:** mhd_proton = mhd_photon × factor. "
            "**B — Absolute:** mhd_proton = mhd_photon − reduction (Gy), minimum 0."
        ),
    )
    if proton_reduction_mode == PROTON_MULT:
        mhd_reduction_factor = st.slider(
            "MHD reduction factor", 0.0, 1.0, 0.6, step=0.05,
            help="0.6 means proton therapy reduces MHD to 60 % of the photon value.",
        )
        mhd_absolute_reduction = 0.0
    else:
        mhd_absolute_reduction = st.slider(
            "Absolute MHD reduction (Gy)", 0.0, 30.0, 5.0, step=0.5,
            help="Subtracts a fixed number of Gy from each patient's photon MHD.",
        )
        mhd_reduction_factor = 1.0

    # ── D. Statistical regression model ──────────────────────────────────────
    st.divider()
    st.markdown("### 📊 D — Statistical regression model")
    st.caption("How the fitted prediction model is estimated")

    st.markdown(
        "<small>These settings affect fitted coefficients, predictions, and AUC. "
        "They do **not** affect true causal mortality.</small>",
        unsafe_allow_html=True,
    )

    use_sqrt = st.checkbox(
        "Use √-transformed predictors",
        value=(dist_mode == DIST_ARTICLE),
        key=f"use_sqrt_{dist_mode}",
        help="Fit models on √GTV and √MHD. Affects only the regression model, not data generation.",
    )
    test_size = st.slider(
        "Test set fraction", 0.1, 0.5, 0.2, step=0.05,
        help="Fraction of patients reserved for model validation.",
    )

    # Article reproduction mode banner
    if _fit_preview:
        st.warning(
            "⚠️ **Article reproduction mode active.**  \n"
            "The optimizer is controlling the active simulation.  \n"
            "Manual sliders below are **overridden**.  \n"
            "See the **Article reproduction** tab.",
            icon="🔬",
        )

    # ── E. Non-causal treatment-effect override (experimental) ───────────────
    with st.expander("⚗️ Non-causal treatment-effect override", expanded=False):
        st.warning(
            "**This experimental mode injects a manually forced treatment benefit "
            "that does not arise from the true causal mortality model.**  \n\n"
            "Use only for sensitivity analysis, debugging, or methodological experiments.",
        )
        st.caption(
            "⚠️ Enabling this intentionally breaks the normal causal logic of the simulator. "
            "The injected benefit has no causal interpretation."
        )
        apply_true_proton = st.toggle(
            "Inject non-causal treatment benefit",
            value=False,
            help=(
                "When ON: a manually scaled mortality reduction is subtracted directly from "
                "the true logit in proportion to the MHD dose reduction. "
                "This effect is NOT mediated by the causal mortality model — "
                "it is an artificial injection for sensitivity testing only."
            ),
        )
        if apply_true_proton:
            proton_effect_strength = st.slider(
                "Override strength", 0.0, 1.0, 0.1, step=0.01,
                help=(
                    "Scales how strongly the MHD reduction is subtracted from the true logit. "
                    "This parameter has no causal interpretation — it is a direct logit manipulation."
                ),
            )
        else:
            proton_effect_strength = 0.0


# ── article-reproduction eligibility & overrides ─────────────────────────────
# Runs before data generation so optimizer parameters feed into the simulation.

_fit_eligible = (
    dist_mode == DIST_ARTICLE
    and survival_scenario in (SCEN_A, SCEN_B)
    and gen_scale == SCALE_SQRT_RAW
)

_fit_overridden = False
_fit_result: dict = {}

if fit_to_article and _fit_eligible:
    with st.spinner("Article reproduction: running grid search…"):
        _fit_result = _fit_to_article(n=n_patients, test_size=test_size, seed=int(seed))
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
mortality   = rng.binomial(1, p_mortality).astype(int)   # 1 = died, 0 = survived
survival    = 1 - mortality

# ── dataframe ─────────────────────────────────────────────────────────────────

df = pd.DataFrame({
    "tumor_volume_cc":    tumor_volume.round(2),
    "mean_heart_dose_gy": mean_heart_dose.round(2),
    "mortality":          mortality,
})
y = df["mortality"].values   # 1 = died; models predict P(mortality=1)

# ── proton MHD reduction ──────────────────────────────────────────────────────

if proton_reduction_mode == PROTON_MULT:
    mhd_proton    = (mean_heart_dose * mhd_reduction_factor).clip(0.0)
    _proton_label = f"factor × {mhd_reduction_factor:.2f}"
else:
    mhd_proton    = (mean_heart_dose - mhd_absolute_reduction).clip(0.0)
    _proton_label = f"− {mhd_absolute_reduction:.1f} Gy"

# ── statistics ────────────────────────────────────────────────────────────────

pearson_r, pearson_p = stats.pearsonr(tumor_volume, mean_heart_dose)
obs_mort_rate = mortality.mean()

# ── feature preparation ───────────────────────────────────────────────────────

X_raw = df[["tumor_volume_cc", "mean_heart_dose_gy"]].values
if use_sqrt:
    X_feat = np.sqrt(X_raw)
    fl = ["√GTV", "√MHD"]
else:
    X_feat = X_raw.copy()
    fl = ["GTV", "MHD"]

if use_sqrt:
    X_feat_proton = np.column_stack([np.sqrt(tumor_volume), np.sqrt(mhd_proton)])
else:
    X_feat_proton = np.column_stack([tumor_volume, mhd_proton])

# ── train / test split ────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X_feat, y, test_size=test_size, random_state=int(seed)
)

# ── three pipeline models (predicting mortality) ──────────────────────────────

if len(np.unique(y_train)) < 2:
    st.warning(
        "⚠️ De trainingsset bevat slechts één klasse — alle patiënten hebben dezelfde uitkomst. "
        "Vergroot de steekproef, pas de baseline mortaliteit aan, of verlaag het ruisniveau."
    )
    st.stop()

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

# ── regression results (combined model) ───────────────────────────────────────

pvals_C    = wald_pvalues_pipe(pipe_C, X_feat)
coefs_C    = _lr(pipe_C).coef_[0]
interc_C   = _lr(pipe_C).intercept_[0]
y_pred_C   = pipe_C.predict(X_test)
accuracy_C = (y_pred_C == y_test).mean()

coef_df = pd.DataFrame({
    "Feature":            fl,
    "Coëfficiënt (std.)": coefs_C.round(4),
    "p-waarde":           [fmt_pval(p) for p in pvals_C],
    "Significantie":      [sig_stars(p) for p in pvals_C],
    "Causaal effect?":    ["Ja", "Ja" if mhd_is_causal else "Nee — confounder"],
})

roc_summary = pd.DataFrame({
    "Model":                [f"A — {fl[0]} only", f"B — {fl[1]} only", f"C — {fl[0]} + {fl[1]}"],
    "Predictoren":          [fl[0], fl[1], f"{fl[0]} + {fl[1]}"],
    "AUC (testset)":        [f"{auc_A:.3f}", f"{auc_B:.3f}", f"{auc_C:.3f}"],
    "Intercept":            [f"{_lr(pipe_A).intercept_[0]:.3f}",
                             f"{_lr(pipe_B).intercept_[0]:.3f}",
                             f"{interc_C:.3f}"],
    "Coëfficiënten (std.)": [_coefstr(pipe_A, [fl[0]]),
                             _coefstr(pipe_B, [fl[1]]),
                             _coefstr(pipe_C, fl)],
})

# ── proton benefit model outputs ──────────────────────────────────────────────

p_mort_photon = pipe_C.predict_proba(X_feat)[:, 1]
p_mort_proton = pipe_C.predict_proba(X_feat_proton)[:, 1]
delta_model   = p_mort_photon - p_mort_proton
nnt_model     = np.where(delta_model > 0, 1.0 / delta_model, np.nan)

if apply_true_proton and proton_effect_strength > 0:
    mhd_diff             = mean_heart_dose - mhd_proton
    true_mort_logit_prot = true_mort_logit - proton_effect_strength * mhd_diff
    true_p_mort_photon   = 1.0 / (1.0 + np.exp(-true_mort_logit))
    true_p_mort_proton   = 1.0 / (1.0 + np.exp(-true_mort_logit_prot))
    true_delta           = true_p_mort_photon - true_p_mort_proton
else:
    true_delta = None

# ── noise impact helper (scenarios A/B only) ─────────────────────────────────

def _noise_auc_row(nl: float) -> dict:
    rng_n = np.random.default_rng(int(seed))
    if gen_scale == SCALE_SQRT_RAW:
        tv_n  = np.sqrt(tumor_volume)
        mh_n  = np.sqrt(mean_heart_dose)
    elif gen_scale == SCALE_SQRT_Z:
        _sg = np.sqrt(tumor_volume); _sm = np.sqrt(mean_heart_dose)
        tv_n  = (_sg - _sg.mean()) / _sg.std()
        mh_n  = (_sm - _sm.mean()) / _sm.std()
    else:  # SCALE_Z_RAW
        tv_n  = (tumor_volume    - tumor_volume.mean())    / tumor_volume.std()
        mh_n  = (mean_heart_dose - mean_heart_dose.mean()) / mean_heart_dose.std()
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
        "Ruisniveau σ":         nl,
        "Obs. mortaliteit":     f"{mort_n.mean()*100:.1f} %",
        f"AUC {fl[0]}":         f"{auc(fa, ta):.3f}",
        f"AUC {fl[1]}":         f"{auc(fb, tb):.3f}",
        f"AUC {fl[0]}+{fl[1]}": f"{auc(fc, tc):.3f}",
    }

# ── colour / style maps ───────────────────────────────────────────────────────

MC = {1: "#e74c3c", 0: "#2ecc71"}   # 1=gestorven=red, 0=niet gestorven=green
ML = {1: "Gestorven", 0: "Niet gestorven"}
MS = {"A": ("#2980b9", "solid"), "B": ("#e67e22", "dashed"), "C": ("#27ae60", "dashdot")}

# ── tabs ──────────────────────────────────────────────────────────────────────

(tab_model, tab_article, tab_roc, tab_hist, tab_dose, tab_proton, tab_data,
 tab_noise, tab_scatter, tab_corr, tab_cm, tab_raw) = st.tabs([
    "Model results", "🔬 Article reproduction", "ROC vergelijking", "Histogrammen", "Dosis-effect",
    "Proton benefit", "Data summary", "Ruis-impact",
    "Scatter GTV ↔ MHD", "Correlatiemat.", "Verwarringsmatrix", "Ruwe data",
])

# ── tab: Model results ────────────────────────────────────────────────────────

with tab_model:
    st.title("24-maands mortaliteit: confounding in klinische data")
    st.markdown(
        "**Doel:** laat zien dat een variabele zonder causale werking "
        "(_mean heart dose_) toch statistisch significant kan lijken doordat ze "
        "correleert met de echte oorzaak (_tumorvolume_)."
    )

    st.subheader("Logistische regressie — gecombineerd mortaliteitsmodel")
    r1, r2 = st.columns([3, 1])
    with r1:
        st.dataframe(coef_df, use_container_width=True, hide_index=True)
        st.caption(
            f"Predictoren: {', '.join(fl)} (StandardScaler). "
            "Uitkomst: 24-maands mortaliteit (1 = gestorven). "
            "p-waarden via Wald-test.  *** p<0.001  ** p<0.01  * p<0.05  ns p≥0.05"
        )
    with r2:
        st.metric("ROC AUC (testset)", f"{auc_C:.3f}")
        st.metric("Nauwkeurigheid",    f"{accuracy_C*100:.1f}%")

    if not mhd_is_causal and pvals_C[1] < 0.05:
        st.warning(
            f"**Confounding waarschuwing:** {fl[1]} is statistisch significant "
            f"(p = {fmt_pval(pvals_C[1])}) terwijl het **geen causale invloed** heeft op mortaliteit. "
            f"Dit komt doordat MHD correleert met GTV (r = {pearson_r:.2f})."
        )
    elif not mhd_is_causal:
        st.info(f"{fl[1]} is niet significant (p = {fmt_pval(pvals_C[1])}). "
                f"Vergroot de correlatie of steekproef om confounding zichtbaar te maken.")
    else:
        st.success(f"MHD heeft een causaal effect op mortaliteit. "
                   f"Coëfficiënt {fl[1]}: {coefs_C[1]:.3f},  p = {fmt_pval(pvals_C[1])}.")

    if survival_scenario != SCEN_C:
        _gen_uses_sqrt = gen_scale in (SCALE_SQRT_RAW, SCALE_SQRT_Z)
        if _gen_uses_sqrt != use_sqrt:
            st.warning(
                "⚠️ **Schaalverschil:** De ware mortaliteitsformule gebruikt "
                f"{'√-getransformeerde' if _gen_uses_sqrt else 'ruwe (niet-√)'} predictoren, "
                f"terwijl het gefitte model {'√-getransformeerde' if use_sqrt else 'ruwe'} predictoren gebruikt. "
                "De coëfficiënten b1/b2 kunnen niet direct worden vergeleken met de gefitte coëfficiënten."
            )

    if _fit_overridden:
        st.warning(
            "⚠️ **Article reproduction mode active** — the optimizer is controlling "
            "the active simulation. Manual sidebar sliders are overridden.  \n"
            "→ See the **🔬 Article reproduction** tab for parameters, results, and caveats.",
            icon="🔬",
        )

    with st.expander("Ware mortaliteitsgenererende formule", expanded=False):
        if survival_scenario == SCEN_C:
            st.markdown(
                f"**Scenario C — Artikel-formule**\n\n"
                f"```\nlogit(p_mortaliteit) = {_display_int:.4f} + 0.0590 · √GTV + 0.2635 · √MHD\n```\n\n"
                f"{'*(intercept gekalibreerd op geselecteerde baseline)*' if calibrate_article else '*(artikel-intercept −1.3409)*'}"
            )
        else:
            _noise_line = f" + N(0, {surv_noise:.2f})" if surv_noise > 0 else ""
            if gen_scale == SCALE_SQRT_RAW:
                _pred_gtv  = "√GTV"
                _pred_mhd  = "√MHD"
                _scale_note = "*(ongestandaardiseerde √-waarden — dezelfde schaal als Van Loon artikel)*"
            elif gen_scale == SCALE_SQRT_Z:
                _pred_gtv  = "z_√GTV"
                _pred_mhd  = "z_√MHD"
                _scale_note = (
                    "z\\_√GTV = (√GTV − gem.) / SD,   z\\_√MHD = (√MHD − gem.) / SD  "
                    "*(gestandaardiseerde √-waarden)*"
                )
            else:  # SCALE_Z_RAW
                _pred_gtv  = "z_GTV"
                _pred_mhd  = "z_MHD"
                _scale_note = (
                    "z\\_GTV = (GTV − gem.) / SD,   z\\_MHD = (MHD − gem.) / SD  "
                    "*(gestandaardiseerde ruwe waarden)*"
                )
            _b2_line = f" + {b2_val:.4f} · {_pred_mhd}" if survival_scenario == SCEN_B else ""
            st.markdown(
                f"**Scenario {'A' if survival_scenario == SCEN_A else 'B'}**\n\n"
                f"```\nlogit(p_mortaliteit) = {_display_int:.4f}"
                f" + {b1:.4f} · {_pred_gtv}{_b2_line}{_noise_line}\n```\n\n"
                f"{_scale_note}"
            )
        param_rows = [
            ("Intercept (logit mortaliteit)", f"{_display_int:.4f}",
             f"= logit({_baseline_mort*100:.1f} %)"),
            ("b1 — GTV-coëfficiënt", f"{b1:.4f}",
             "vaste artikel-coëfficiënt" if survival_scenario == SCEN_C else "instelbaar"),
        ]
        if survival_scenario != SCEN_A:
            param_rows.append(
                ("b2 — MHD-coëfficiënt", f"{b2_val:.4f}",
                 "vaste artikel-coëfficiënt" if survival_scenario == SCEN_C else "instelbaar")
            )
        if survival_scenario in (SCEN_A, SCEN_B):
            param_rows.append(("Mortaliteitsruisniveau σ", f"{surv_noise:.2f}", "N(0,σ) toegevoegd aan logit"))
        param_rows += [
            ("Geselecteerde baseline mortaliteit", f"{_baseline_mort*100:.1f} %", ""),
            ("Gesimuleerde mortaliteit",           f"{obs_mort_rate*100:.1f} %", ""),
        ]
        st.dataframe(
            pd.DataFrame(param_rows, columns=["Parameter", "Waarde", "Toelichting"]),
            use_container_width=True, hide_index=True,
        )

    with st.expander("Toon ware mortaliteitslogitfunctie", expanded=False):
        if survival_scenario == SCEN_C:
            st.markdown(
                "**Scenario C — logit-bijdragen op √-schaal (niet gestandaardiseerd)**\n\n"
                f"```\nlogit(p) = {_display_int:.4f} + 0.0590 · √GTV + 0.2635 · √MHD\n```"
            )
            x_range = np.linspace(0, 15, 300)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(x_range, 0.059  * x_range, color=MS["A"][0], lw=2, label="0.0590 · √GTV")
            ax.plot(x_range, 0.2635 * x_range, color=MS["B"][0], lw=2, linestyle="--",
                    label="0.2635 · √MHD")
            ax.axhline(0, color="gray", lw=0.8, linestyle=":")
            ax.set_xlabel("Predictor op √-schaal"); ax.set_ylabel("Logit-bijdrage")
            ax.set_title("Logit-bijdragen per predictor (Scenario C)")
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
            fig.tight_layout()
            centered(fig)
        else:
            _noise_str = f" + N(0, {surv_noise:.2f})" if surv_noise > 0 else ""
            if gen_scale == SCALE_SQRT_RAW:
                _vp_gtv = "√GTV";    _vp_mhd = "√MHD"
                _xlabel = "Predictor op √-schaal (ongestand.)"
                _title  = "Logit-bijdragen op √-schaal (ongestandaardiseerd)"
                x_range = np.linspace(0, 15, 300)
            elif gen_scale == SCALE_SQRT_Z:
                _vp_gtv = "z_√GTV";  _vp_mhd = "z_√MHD"
                _xlabel = "Gestandaardiseerde √-predictor (z-score)"
                _title  = "Logit-bijdragen op gestandaardiseerde √-schaal"
                x_range = np.linspace(-3, 3, 300)
            else:  # SCALE_Z_RAW
                _vp_gtv = "z_GTV";   _vp_mhd = "z_MHD"
                _xlabel = "Gestandaardiseerde predictor (z-score)"
                _title  = "Logit-bijdragen op gestandaardiseerde schaal"
                x_range = np.linspace(-3, 3, 300)
            _b2_str = f" + {b2_val:.4f} · {_vp_mhd}" if survival_scenario == SCEN_B else ""
            st.markdown(
                f"**Scenario {'A' if survival_scenario == SCEN_A else 'B'}:**\n\n"
                f"```\nlogit(p) = {_display_int:.4f} + {b1:.4f} · {_vp_gtv}{_b2_str}{_noise_str}\n```"
            )
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(x_range, b1 * x_range, color=MS["A"][0], lw=2, label=f"{b1:.4f} · {_vp_gtv}")
            if survival_scenario == SCEN_B:
                ax.plot(x_range, b2_val * x_range, color=MS["B"][0], lw=2, linestyle="--",
                        label=f"{b2_val:.4f} · {_vp_mhd}")
            ax.axhline(0, color="gray", lw=0.8, linestyle=":")
            if gen_scale != SCALE_SQRT_RAW:
                ax.axvline(0, color="gray", lw=0.8, linestyle=":")
            ax.set_xlabel(_xlabel)
            ax.set_ylabel("Bijdrage aan logit(p_mortaliteit)")
            ax.set_title(_title)
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
            fig.tight_layout()
            centered(fig)
        st.caption(
            "De logit-bijdrage is het lineaire effect van de predictor op de logit-schaal, "
            "niet de mortaliteitskans direct. "
            "De kans volgt via p = 1 / (1 + exp(−logit)). "
            "Een logit-toename van +1 rondom logit = 0 verhoogt de kans van 50 % naar ~73 %."
        )

    st.divider()
    st.caption(
        f"Scenario {survival_scenario[:1]} · "
        "Causaal diagram (A): **GTV → mortaliteit** en **GTV → MHD**. "
        "In scenario A heeft MHD geen direct pad naar mortaliteit. "
        "In scenario B en C heeft MHD ook een causaal effect."
    )

# ── tab: Article reproduction experiment ─────────────────────────────────────

with tab_article:
    st.title("🔬 Article reproduction experiment")
    st.markdown(
        "**Scientific purpose:** Can alternative causal structures reproduce "
        "article-like regression outputs?  \n\n"
        "This is an **inverse parameter search** — the opposite of forward causal simulation. "
        "The optimizer searches for combinations of GTV-MHD correlation and prognostic noise "
        "that reproduce the regression coefficients, AUC, and mortality rate reported in "
        "Van Loon et al.  \n\n"
        'This workflow answers: *"Which causal worlds are consistent with the published numbers?"*'
    )

    st.divider()

    # ── Toggle ────────────────────────────────────────────────────────────────
    _art_eligible_msg = (
        "**Requires:** Article-like distribution + Scenario A or B "
        "+ Scale √GTV/MHD unstandardised."
    )
    if not _fit_eligible:
        st.info(
            "**Optimizer not available with current settings.**  \n" + _art_eligible_msg,
            icon="ℹ️",
        )
        # Render a disabled-looking toggle that can't be activated
        st.toggle(
            "Run article reproduction optimizer",
            key="fit_to_article_toggle",
            value=False,
            disabled=True,
            help=_art_eligible_msg,
        )
    else:
        _toggled = st.toggle(
            "Run article reproduction optimizer",
            key="fit_to_article_toggle",
            value=False,
            help=(
                "Grid-searches GTV-MHD correlation and noise level to minimise the weighted "
                "squared distance from article targets. Result is cached — first run ~5 s."
            ),
        )

    st.divider()

    if not _fit_overridden:
        st.info(
            "**Optimizer is OFF.** The forward causal simulation is running normally.  \n"
            "Enable the toggle above to start an article reproduction experiment.",
            icon="💡",
        )
    else:
        # ── Active-mode banner ────────────────────────────────────────────────
        st.error(
            "### ⚠️ Article reproduction mode active\n\n"
            "The optimizer is controlling the active simulation.  \n"
            "**Manual sidebar sliders are not controlling the active simulation.**  \n"
            "The parameters shown below are what is actually being used.",
            icon="🔬",
        )

        # ── Parameters used by the optimizer ─────────────────────────────────
        st.subheader("Parameters used by the optimizer")
        st.caption(
            "These values were chosen by grid search to best match article targets — "
            "they are not your manual slider settings."
        )

        _obs_r = pearson_r  # computed after data generation

        p1, p2, p3, p4 = st.columns(4)
        p1.metric(
            "True causal b1 (√GTV)",
            f"{_FIT_TARGETS['beta_gtv']:.4f}",
            help="Fixed to the article value (0.0590 · √GTV).",
        )
        p2.metric(
            "True causal b2 (√MHD)",
            f"{_FIT_TARGETS['beta_mhd']:.4f}",
            help="Fixed to the article value (0.2635 · √MHD).",
        )
        p3.metric(
            "GTV-MHD correlation (observed r)",
            f"{_obs_r:.3f}",
            help="Pearson r in the generated population with optimizer copula ρ.",
        )
        p4.metric(
            "Prognostic noise (σ)",
            f"{_fit_result['noise']:.2f}",
            help="Logit-scale noise chosen by the optimizer.",
        )

        # ── Achieved vs target metrics ────────────────────────────────────────
        st.subheader("Achieved vs. article targets")

        _art_rows = [
            ("β_√GTV (fitted)",  f"{_fit_result['beta_gtv']:.4f}",
             f"{_FIT_TARGETS['beta_gtv']:.4f}",
             f"{_fit_result['beta_gtv'] - _FIT_TARGETS['beta_gtv']:+.4f}"),
            ("β_√MHD (fitted)",  f"{_fit_result['beta_mhd']:.4f}",
             f"{_FIT_TARGETS['beta_mhd']:.4f}",
             f"{_fit_result['beta_mhd'] - _FIT_TARGETS['beta_mhd']:+.4f}"),
            ("AUC",              f"{_fit_result['auc']:.3f}",
             f"{_FIT_TARGETS['auc']:.3f}",
             f"{_fit_result['auc'] - _FIT_TARGETS['auc']:+.3f}"),
            ("Mortality rate",   f"{_fit_result['mortality']*100:.1f} %",
             f"{_FIT_TARGETS['mortality']*100:.1f} %",
             f"{(_fit_result['mortality'] - _FIT_TARGETS['mortality'])*100:+.1f} pp"),
        ]
        st.dataframe(
            pd.DataFrame(_art_rows, columns=["Metric", "Achieved", "Article target", "Δ"]),
            use_container_width=True,
            hide_index=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("β_√GTV (fitted)",  f"{_fit_result['beta_gtv']:.4f}",
                  delta=f"{_fit_result['beta_gtv'] - _FIT_TARGETS['beta_gtv']:+.4f} vs 0.059")
        m2.metric("β_√MHD (fitted)",  f"{_fit_result['beta_mhd']:.4f}",
                  delta=f"{_fit_result['beta_mhd'] - _FIT_TARGETS['beta_mhd']:+.4f} vs 0.264")
        m3.metric("AUC",              f"{_fit_result['auc']:.3f}",
                  delta=f"{_fit_result['auc'] - _FIT_TARGETS['auc']:+.3f} vs 0.640")
        m4.metric("Mortality",        f"{_fit_result['mortality']*100:.1f} %",
                  delta=f"{(_fit_result['mortality'] - _FIT_TARGETS['mortality'])*100:+.1f} pp vs 50.6 %")

        st.divider()

        # ── Scientific caveat ─────────────────────────────────────────────────
        st.subheader("⚠️ Important scientific caveat")
        st.warning(
            "**Matching published regression coefficients does not prove the underlying "
            "causal structure is correct.**  \n\n"
            "The optimizer found a causal world that produces similar fitted β values, "
            "AUC, and mortality rate. But many different causal structures can produce "
            "the same regression output.  \n\n"
            "Identical β values as Van Loon et al. do **not** mean:  \n"
            "- the true b1 and b2 are those values  \n"
            "- MHD is (or is not) causally linked to mortality  \n"
            "- the correlation structure matches the real patient population  \n\n"
            "This experiment demonstrates **observational equivalence**: published "
            "regression results are consistent with multiple causal stories."
        )

        st.info(
            "**What this optimizer does NOT do:**  \n"
            "- It does not search over causal structures (b1, b2 are fixed to article values)  \n"
            "- It does not test whether MHD is causal  \n"
            "- It does not validate the article's conclusions  \n\n"
            "It only searches over: *GTV-MHD correlation* and *prognostic noise*.",
            icon="ℹ️",
        )

# ── tab: ROC vergelijking ─────────────────────────────────────────────────────

with tab_roc:
    st.markdown(
        "> **Predictie vs. causaliteit** — Het MHD-only model kan 24-maands mortaliteit "
        "voorspellen wanneer MHD correleert met tumorvolume, ook als MHD **geen** direct "
        "causaal effect heeft op mortaliteit. Een goede predictor hoeft geen effectieve "
        "behandeldoelstelling te zijn."
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, name, f, t, a_val in [
        ("A", f"{fl[0]} only",       fpr_A, tpr_A, auc_A),
        ("B", f"{fl[1]} only",       fpr_B, tpr_B, auc_B),
        ("C", f"{fl[0]} + {fl[1]}", fpr_C, tpr_C, auc_C),
    ]:
        color, ls = MS[key]
        ax.plot(f, t, color=color, lw=2, linestyle=ls, label=f"{name},  AUC = {a_val:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", lw=1, label="Toeval  (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-vergelijking — 24-maands mortaliteit (testset)")
    ax.legend(loc="lower right", fontsize=9); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    centered(fig)

    st.markdown("**Samenvatting per model**")
    st.dataframe(roc_summary, use_container_width=True, hide_index=True)

    if not mhd_is_causal and auc_B > 0.55:
        st.warning(
            f"**Confounding zichtbaar:** Model B ({fl[1]} only, AUC = {auc_B:.3f}) "
            f"voorspelt mortaliteit beter dan toeval zonder causale werking "
            f"(r(GTV,MHD) = {pearson_r:.2f})."
        )
    elif not mhd_is_causal:
        st.info(f"Model B ({fl[1]} only, AUC = {auc_B:.3f}) presteert nauwelijks beter dan toeval.")
    else:
        st.success(f"MHD heeft causaal effect. Model B AUC = {auc_B:.3f}, C AUC = {auc_C:.3f}.")

    st.markdown("### Fitted mortaliteitsfuncties")
    tv_grid  = np.linspace(tumor_volume.min(), tumor_volume.max(), 200)
    mhd_grid = np.linspace(mean_heart_dose.min(), mean_heart_dose.max(), 200)
    tv_in    = np.sqrt(tv_grid)  if use_sqrt else tv_grid
    mhd_in   = np.sqrt(mhd_grid) if use_sqrt else mhd_grid
    mhd_p10, mhd_p50, mhd_p90 = (np.percentile(mean_heart_dose, q) for q in [10, 50, 90])

    sv1, sv2, sv3 = st.columns(3)
    with sv1:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(tumor_volume, mortality, c=[MC[m] for m in mortality],
                   alpha=0.25, s=10, linewidths=0, zorder=1)
        ax.plot(tv_grid, pipe_A.predict_proba(tv_in.reshape(-1, 1))[:, 1],
                color=MS["A"][0], lw=2, zorder=2)
        ax.set_xlabel("GTV (cc)"); ax.set_ylabel("Voorspelde 24-maands mortaliteit")
        ax.set_title(f"Model A — {fl[0]} only"); ax.set_ylim(0, 1)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    with sv2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(mean_heart_dose, mortality, c=[MC[m] for m in mortality],
                   alpha=0.25, s=10, linewidths=0, zorder=1)
        ax.plot(mhd_grid, pipe_B.predict_proba(mhd_in.reshape(-1, 1))[:, 1],
                color=MS["B"][0], lw=2, zorder=2)
        ax.set_xlabel("MHD (Gy)"); ax.set_ylabel("Voorspelde 24-maands mortaliteit")
        ax.set_title(f"Model B — {fl[1]} only"); ax.set_ylim(0, 1)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    with sv3:
        fig, ax = plt.subplots(figsize=(5, 4))
        for mhd_val, label, color in [
            (mhd_p10, f"MHD p10 = {mhd_p10:.1f} Gy", "#1a9641"),
            (mhd_p50, f"MHD med = {mhd_p50:.1f} Gy",  "#fdae61"),
            (mhd_p90, f"MHD p90 = {mhd_p90:.1f} Gy",  "#d7191c"),
        ]:
            Xc = (np.column_stack([np.sqrt(tv_grid), np.full_like(tv_grid, np.sqrt(mhd_val))])
                  if use_sqrt else
                  np.column_stack([tv_grid, np.full_like(tv_grid, mhd_val)]))
            ax.plot(tv_grid, pipe_C.predict_proba(Xc)[:, 1], color=color, lw=2, label=label)
        ax.set_xlabel("GTV (cc)"); ax.set_ylabel("Voorspelde 24-maands mortaliteit")
        ax.set_title(f"Model C — {fl[0]} + {fl[1]}"); ax.set_ylim(0, 1)
        ax.legend(fontsize=6.5, loc="lower right")
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    st.caption(
        "Rode punten = gestorven; groene punten = niet gestorven. "
        "In scenario A weerspiegelen de drie MHD-curves confounding, niet een direct effect."
    )

# ── tab: Histogrammen ─────────────────────────────────────────────────────────

with tab_hist:
    show_cumul = st.checkbox(
        "Toon cumulatieve histogrammen", value=False,
        help="Toon cumulatieve verdelingen gesplitst op mortaliteitsstatus.",
    )
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, col, xlabel in zip(axes,
        ["tumor_volume_cc", "mean_heart_dose_gy"], ["GTV (cc)", "Mean Heart Dose (Gy)"]):
        for val in [0, 1]:
            data = df.loc[df["mortality"] == val, col].values
            if show_cumul:
                ax.hist(data, bins=60, density=True, cumulative=True,
                        alpha=0.6, color=MC[val], label=ML[val], edgecolor="none")
                ax.set_ylabel("Cumulatief aandeel")
                ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
            else:
                ax.hist(data, bins=30, alpha=0.6, color=MC[val], label=ML[val], edgecolor="none")
                ax.set_ylabel("Aantal patiënten")
        ax.set_xlabel(xlabel); ax.legend()
    title = "Cumulatieve verdeling" if show_cumul else "Verdeling"
    fig.suptitle(f"{title} per mortaliteitsstatus", fontsize=12)
    fig.tight_layout()
    centered(fig)

# ── tab: Dosis-effect ─────────────────────────────────────────────────────────

with tab_dose:
    st.markdown("### Dosis-effect op 24-maands mortaliteit — gecombineerd model")

    gtv_med  = np.median(tumor_volume)
    mhd_med  = np.median(mean_heart_dose)
    gtv_p10, gtv_p90 = np.percentile(tumor_volume, [10, 90])
    mhd_p10_de, mhd_p90_de = np.percentile(mean_heart_dose, [10, 90])

    tv_g  = np.linspace(tumor_volume.min(),    tumor_volume.max(),    300)
    mhd_g = np.linspace(mean_heart_dose.min(), mean_heart_dose.max(), 300)

    def _X(gtv_arr, mhd_arr):
        if use_sqrt:
            return np.column_stack([np.sqrt(gtv_arr), np.sqrt(mhd_arr)])
        return np.column_stack([gtv_arr, mhd_arr])

    mort_vs_gtv = pipe_C.predict_proba(_X(tv_g,  np.full_like(tv_g,  mhd_med)))[:, 1]
    mort_vs_mhd = pipe_C.predict_proba(_X(np.full_like(mhd_g, gtv_med), mhd_g))[:, 1]

    de1, de2 = st.columns(2)
    with de1:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(tv_g, mort_vs_gtv, color="#e74c3c", lw=2)
        ax.axvline(gtv_med, color="gray", lw=1, linestyle=":",
                   label=f"Mediaan GTV = {gtv_med:.0f} cc")
        ax.set_xlabel("GTV (cc)"); ax.set_ylabel("Voorspelde 24-maands mortaliteit")
        ax.set_title(f"Plot A — mortaliteit vs GTV\n(MHD vast op mediaan {mhd_med:.1f} Gy)")
        ax.set_ylim(0, 1); ax.legend(fontsize=8)
        fig.tight_layout(); st.pyplot(fig, use_container_width=False); plt.close(fig)

    with de2:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(mhd_g, mort_vs_mhd, color="#e74c3c", lw=2)
        ax.axvline(mhd_med, color="gray", lw=1, linestyle=":",
                   label=f"Mediaan MHD = {mhd_med:.1f} Gy")
        ax.set_xlabel("Mean Heart Dose (Gy)"); ax.set_ylabel("Voorspelde 24-maands mortaliteit")
        ax.set_title(f"Plot B — mortaliteit vs MHD\n(GTV vast op mediaan {gtv_med:.0f} cc)")
        ax.set_ylim(0, 1); ax.legend(fontsize=8)
        fig.tight_layout(); st.pyplot(fig, use_container_width=False); plt.close(fig)

    st.markdown("#### Mortaliteit vs GTV voor drie MHD-niveaus (Plot C)")
    fig, ax = plt.subplots(figsize=(7, 5))
    for mhd_val, label, color in [
        (mhd_p10_de, f"MHD laag (p10) = {mhd_p10_de:.1f} Gy", "#1a9641"),
        (mhd_med,    f"MHD mediaan = {mhd_med:.1f} Gy",         "#fdae61"),
        (mhd_p90_de, f"MHD hoog (p90) = {mhd_p90_de:.1f} Gy", "#d7191c"),
    ]:
        ax.plot(tv_g, pipe_C.predict_proba(_X(tv_g, np.full_like(tv_g, mhd_val)))[:, 1],
                color=color, lw=2, label=label)
    ax.set_xlabel("GTV (cc)"); ax.set_ylabel("Voorspelde 24-maands mortaliteit")
    ax.set_title("Plot C — mortaliteit vs GTV voor laag/mediaan/hoog MHD")
    ax.set_ylim(0, 1); ax.legend(fontsize=8)
    fig.tight_layout(); centered(fig)

    st.markdown("#### Mortaliteit vs MHD voor drie GTV-niveaus (Plot D)")
    fig, ax = plt.subplots(figsize=(7, 5))
    for gtv_val, label, color in [
        (gtv_p10, f"GTV laag (p10) = {gtv_p10:.0f} cc", "#1a9641"),
        (gtv_med, f"GTV mediaan = {gtv_med:.0f} cc",      "#fdae61"),
        (gtv_p90, f"GTV hoog (p90) = {gtv_p90:.0f} cc", "#d7191c"),
    ]:
        ax.plot(mhd_g, pipe_C.predict_proba(_X(np.full_like(mhd_g, gtv_val), mhd_g))[:, 1],
                color=color, lw=2, label=label)
    ax.set_xlabel("Mean Heart Dose (Gy)"); ax.set_ylabel("Voorspelde 24-maands mortaliteit")
    ax.set_title("Plot D — mortaliteit vs MHD voor laag/mediaan/hoog GTV")
    ax.set_ylim(0, 1); ax.legend(fontsize=8)
    fig.tight_layout(); centered(fig)

    if not mhd_is_causal:
        st.info(
            "In scenario A (GTV causaal, MHD confounder): de MHD-curves (Plot B en D) "
            "tonen een associatie die **niet causaal** is. De scheiding in Plot C "
            "ontstaat doordat hogere MHD samenhangt met grotere GTV."
        )

# ── tab: Proton benefit ───────────────────────────────────────────────────────

with tab_proton:
    st.markdown(
        "Dit tabblad vergelijkt foton- en gesimuleerde protonenplannen door de MHD te verlagen. "
        "**De voorspelde absolute mortaliteitsreductie komt van het gefitte model.** "
        "Als MHD alleen correleert met GTV maar niet causaal is, kan het model "
        "mortaliteitsreductie voorspellen terwijl de werkelijke uitkomst niet verbetert."
    )

    if proton_reduction_mode == PROTON_MULT:
        st.info(
            f"**Modus A — Multiplicatief:** mhd_proton = mhd_photon × {mhd_reduction_factor:.2f}  "
            f"(MHD teruggebracht tot {mhd_reduction_factor*100:.0f} % van de fotonwaarde; min. 0 Gy)"
        )
    else:
        st.info(
            f"**Modus B — Absoluut:** mhd_proton = mhd_photon − {mhd_absolute_reduction:.1f} Gy  "
            f"(vaste reductie van {mhd_absolute_reduction:.1f} Gy per patiënt; min. 0 Gy)"
        )

    pm1, pm2, pm3, pm4 = st.columns(4)
    pm1.metric("Gem. mortaliteit (foton)",                f"{p_mort_photon.mean()*100:.1f}%")
    pm2.metric("Gem. mortaliteit (proton)",               f"{p_mort_proton.mean()*100:.1f}%")
    pm3.metric("Gem. absolute mortaliteitsreductie (Δ)",  f"{delta_model.mean()*100:.2f}%")
    pm4.metric("Gem. NNT (model)",                        fmt_f(np.nanmean(nnt_model)))

    if true_delta is not None:
        st.error(
            "⚗️ **Non-causal treatment-effect override is active.**  \n"
            "The 'true Δ' shown below is an artificially injected logit manipulation — "
            "it does **not** arise from the causal mortality model.  \n"
            "Results below are for sensitivity analysis only and should not be "
            "interpreted as a causal treatment effect.",
        )
        pt1, pt2, *_ = st.columns(4)
        pt1.metric(
            "Injected Δ (override, non-causal)",
            f"{true_delta.mean()*100:.2f}%",
            help="This is an artificially forced logit reduction, not a causal effect.",
        )
        pt2.metric(
            "Model Δ vs. injected Δ",
            f"{(delta_model.mean() - true_delta.mean())*100:.2f}%-pt",
            help="Positive = model predicts more reduction than the injected override.",
        )
        if not mhd_is_causal:
            st.warning(
                f"**Confounding note:** the model predicts {delta_model.mean()*100:.2f}% "
                f"absolute mortality reduction, while the injected override is "
                f"{true_delta.mean()*100:.2f}%. "
                f"MHD correlates with GTV but has no causal path to mortality."
            )

    st.markdown("---")
    st.markdown("#### Verdeling van voorspelde absolute mortaliteitsreductie (Δ) per patiënt")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(delta_model * 100, bins=40, color="#2980b9", edgecolor="none", alpha=0.8,
            label="Model-gebaseerde Δ")
    if true_delta is not None:
        ax.hist(true_delta * 100, bins=40, color="#e67e22", edgecolor="none",
                alpha=0.5, label="Injected Δ (non-causal override)")
        ax.legend(fontsize=8)
    ax.set_xlabel("Absolute mortaliteitsreductie Δ (%-punt)")
    ax.set_ylabel("Aantal patiënten")
    ax.set_title("Verdeling van model-gebaseerde mortaliteitsreductie")
    fig.tight_layout(); centered(fig)

    st.markdown("#### NNT-tabel per Δ-bin")
    nnt_df = build_nnt_table(delta_model, n_patients)
    st.dataframe(nnt_df, use_container_width=True, hide_index=True)
    st.caption(
        "Δ = mortaliteit(foton) − mortaliteit(proton); positief = proton reduceert mortaliteit. "
        "NNT = 1/Δ per patiënt. 1/Mean Δ en 1/Median Δ zijn NNT-schattingen op groepsniveau."
    )

    st.markdown("#### Patiënten per Δ-bin")
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(_BLABELS, nnt_df["N"].values, color="#2980b9", edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, padding=2, fontsize=7)
    ax.set_xlabel("Absolute mortaliteitsreductie Δ")
    ax.set_ylabel("Aantal patiënten")
    ax.set_title(f"Patiënten per Δ-bin  (MHD-reductie: {_proton_label})")
    ax.tick_params(axis="x", rotation=40, labelsize=7)
    fig.tight_layout(); centered(fig)

    if not mhd_is_causal:
        st.caption(
            "Scenario A: MHD heeft geen causale werking. Patiënten met hoge MHD hebben "
            "ook een grotere GTV — hun hogere 'voorspelde mortaliteitsreductie' is een "
            "artefact van confounding."
        )

# ── tab: Data summary ─────────────────────────────────────────────────────────

with tab_data:
    st.markdown("### Gegevenssamenvatting")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patiënten",              n_patients)
    c2.metric("24-maands mortaliteit",  f"{obs_mort_rate*100:.1f}%")
    c3.metric("Pearson r (GTV ↔ MHD)", f"{pearson_r:.3f}")
    c4.metric("p-waarde correlatie",    fmt_pval(pearson_p))

    st.markdown("#### Beschrijvende statistieken")
    vc1, vc2 = st.columns(2)
    with vc1:
        st.dataframe(
            pd.DataFrame({"GTV (cc)": describe_arr(tumor_volume),
                          "MHD (Gy)": describe_arr(mean_heart_dose)}),
            use_container_width=True,
        )
        if dist_mode == DIST_ARTICLE:
            st.caption("Referentie: GTV gem.≈110 cc, med.≈69 cc, SD≈130 cc; "
                       "MHD gem.≈12 Gy, med.≈11 Gy, SD≈8.2 Gy.")
    with vc2:
        st.markdown("**Baseline vs gesimuleerd**")
        st.dataframe(
            pd.DataFrame({
                "": ["Geselecteerd", "Gesimuleerd"],
                "2-jaars overleving":    [f"{baseline_survival*100:.1f} %",
                                          f"{(1-obs_mort_rate)*100:.1f} %"],
                "24-maands mortaliteit": [f"{_baseline_mort*100:.1f} %",
                                          f"{obs_mort_rate*100:.1f} %"],
            }),
            use_container_width=True, hide_index=True,
        )
        if survival_scenario == SCEN_C and not calibrate_article:
            st.caption("Scenario C gebruikt vaste artikel-intercept (−1.3409).")

    st.markdown("#### MHD-reductie samenvatting (protonentherapie)")
    pd_proton = pd.DataFrame({
        "": ["Fotontherapie", "Protonentherapie", "Reductie"],
        "Gem. MHD (Gy)": [
            f"{mean_heart_dose.mean():.2f}",
            f"{mhd_proton.mean():.2f}",
            f"{(mean_heart_dose - mhd_proton).mean():.2f}",
        ],
        "Mediaan MHD (Gy)": [
            f"{np.median(mean_heart_dose):.2f}",
            f"{np.median(mhd_proton):.2f}",
            f"{np.median(mean_heart_dose - mhd_proton):.2f}",
        ],
    })
    st.dataframe(pd_proton, use_container_width=True, hide_index=True)
    st.caption(f"Proton MHD reductie modus: {proton_reduction_mode} ({_proton_label})")

# ── tab: Ruwe data ────────────────────────────────────────────────────────────

with tab_raw:
    st.subheader("Ruwe data (eerste 50 rijen)")
    st.dataframe(df.head(50), use_container_width=True)

# ── tab: Scatter GTV ↔ MHD ───────────────────────────────────────────────────

with tab_scatter:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(tumor_volume, mean_heart_dose,
               c=[MC[m] for m in mortality], alpha=0.4, s=18, linewidths=0)
    m_r, b_r, *_ = stats.linregress(tumor_volume, mean_heart_dose)
    xl = np.array([tumor_volume.min(), tumor_volume.max()])
    ax.plot(xl, m_r * xl + b_r, color="#2c3e50", lw=1.5,
            label=f"y = {m_r:.2f}x + {b_r:.1f}  (r = {pearson_r:.2f}, p = {fmt_pval(pearson_p)})")
    h = [Patch(color=MC[v], alpha=0.7, label=ML[v]) for v in [1, 0]]
    l1 = ax.legend(handles=h, loc="upper left", fontsize=8)
    ax.add_artist(l1)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("GTV (cc)"); ax.set_ylabel("Mean Heart Dose (Gy)")
    ax.set_title("Scatter: GTV vs MHD — kleur = mortaliteitsstatus")
    fig.tight_layout()
    centered(fig)
    st.caption("De regressielijn toont GTV → MHD. Doordat mortaliteit afhangt van GTV, "
               "lijkt MHD ook te scheiden — dit is de confounder.")

# ── tab: Correlatiemat. ───────────────────────────────────────────────────────

with tab_corr:
    corr_df = df[["tumor_volume_cc", "mean_heart_dose_gy", "mortality"]].corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr_df, annot=True, fmt=".3f", cmap="RdYlGn",
                center=0, vmin=-1, vmax=1, mask=np.eye(3, dtype=bool),
                ax=ax, linewidths=0.5)
    ax.set_title("Pearson correlatiemat.")
    fig.tight_layout()
    centered(fig)
    st.caption("MHD correleert met zowel GTV als mortaliteit, "
               "ook zonder causaal pad MHD → mortaliteit (scenario A).")

# ── tab: Verwarringsmatrix ────────────────────────────────────────────────────

with tab_cm:
    cm = confusion_matrix(y_test, y_pred_C)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=["Niet gestorven", "Gestorven"],
                yticklabels=["Niet gestorven", "Gestorven"], ax=ax)
    ax.set_xlabel("Voorspeld"); ax.set_ylabel("Werkelijk")
    ax.set_title("Verwarringsmatrix — mortaliteit (testset)")
    fig.tight_layout()
    centered(fig)

# ── tab: Ruis-impact ──────────────────────────────────────────────────────────

with tab_noise:
    st.markdown("### Ruis-impact op mortaliteitsmodel")
    st.markdown(
        "Hieronder worden voor verschillende mortaliteitsruisniveaus de AUC-waarden "
        "van de drie modellen gesimuleerd op basis van dezelfde GTV- en MHD-waarden."
    )
    if survival_scenario == SCEN_C:
        st.info(
            "Scenario C gebruikt een vaste artikel-formule zonder ruisparameter. "
            "De ruis-impact-analyse is alleen beschikbaar voor scenario A en B."
        )
    else:
        noise_rows = [_noise_auc_row(nl) for nl in _NOISE_LEVELS]
        noise_df   = pd.DataFrame(noise_rows)
        st.dataframe(noise_df, use_container_width=True, hide_index=True)
        st.caption(
            f"Huidige instelling: σ = {surv_noise:.2f}. "
            "De AUC daalt naarmate het ruisniveau toeneemt doordat de relatie "
            "tussen GTV/MHD en mortaliteit minder zichtbaar wordt."
        )

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
                   label=f"Huidig σ = {surv_noise:.2f}")
        ax.set_xlabel("Mortaliteitsruisniveau σ")
        ax.set_ylabel("ROC AUC (testset)")
        ax.set_title("AUC vs mortaliteitsruisniveau")
        ax.legend(fontsize=8); ax.set_ylim(0.4, 1.0)
        fig.tight_layout(); centered(fig)

