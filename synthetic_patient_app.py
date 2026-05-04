"""
Synthetic patient dataset — confounding demo.

Causal structure:
  Scenario A:  GTV ──► survival   GTV ──► MHD   (MHD is confounder only)
  Scenario B:  GTV ──► survival   MHD ──► survival   GTV ──► MHD
  Scenario C:  logit(mortality) = −1.3409 + 0.059·√GTV + 0.2635·√MHD
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc


# ── constants ─────────────────────────────────────────────────────────────────

DIST_SIMPLE  = "Eenvoudige normale verdelingen"
DIST_ARTICLE = "Artikel-achtige longkanker-verdelingen"

SCEN_A = "A — GTV causaal, MHD alleen gecorreleerd"
SCEN_B = "B — GTV én MHD causaal"
SCEN_C = "C — Artikel-formule (vaste coëfficiënten)"

# Article-like distribution parameters
_GTV_MU    = np.log(69.0)
_GTV_SIGMA = np.sqrt(2 * (np.log(110.0) - np.log(69.0)))   # ≈ 0.965
_MHD_K     = (12.0 / 8.2) ** 2                              # shape ≈ 2.14
_MHD_THETA = 8.2 ** 2 / 12.0                                # scale ≈ 5.60

# NNT table bin edges and labels (delta in [0, 1])
_BINS   = [0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40,
           0.50, 0.60, 0.70, 0.80, 0.90, 1.01]
_BLABELS = ["0–2 %", "2–5 %", "5–10 %", "10–20 %", "20–30 %", "30–40 %",
            "40–50 %", "50–60 %", "60–70 %", "70–80 %", "80–90 %", "90–100 %"]


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
    if np.isnan(p):   return "n/a"
    if p < 0.001:     return "< 0.001"
    return f"{p:.3f}"


def sig_stars(p: float) -> str:
    if np.isnan(p):  return ""
    if p < 0.001:    return "***"
    if p < 0.01:     return "**"
    if p < 0.05:     return "*"
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


def fmt_f(v, decimals=1, suffix=""):
    return f"{v:.{decimals}f}{suffix}" if not (np.isnan(v) if isinstance(v, float) else False) else "—"


def build_nnt_table(delta: np.ndarray, n_total: int) -> pd.DataFrame:
    """Build NNT summary table binned by absolute risk reduction."""
    rows = []
    for label, lo, hi in zip(_BLABELS, _BINS[:-1], _BINS[1:]):
        mask = (delta >= lo) & (delta < hi)
        d    = delta[mask]
        n    = int(mask.sum())
        if n == 0:
            rows.append({
                "Bin": label, "N": 0, "%": "0.0 %",
                "Mean Δ": "—", "Median Δ": "—",
                "Mean NNT": "—", "Median NNT": "—",
                "1/Mean Δ": "—", "1/Median Δ": "—",
            })
            continue
        mean_d  = d.mean()
        med_d   = np.median(d)
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

st.set_page_config(page_title="Confounding Demo", layout="wide")
st.title("Confounding in klinische data")
st.markdown(
    "**Doel:** laat zien dat een variabele zonder causale werking "
    "(_mean heart dose_) toch statistisch significant kan lijken doordat ze "
    "correleert met de echte oorzaak (_tumorvolume_)."
)

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Simulatie-instellingen")

    dist_mode = st.selectbox(
        "Distribution mode",
        [DIST_ARTICLE, DIST_SIMPLE],        # article-like is default
        help=(
            "**Artikel-achtig (standaard):** log-normale GTV en gamma-verdeelde MHD "
            "via een Gaussian copula, passend bij gepubliceerde longkankercijfers "
            "(GTV gem.≈110 cc, med.≈69 cc; MHD gem.≈12 Gy, med.≈11 Gy). "
            "**Eenvoudig:** normale verdelingen met instelbare parameters."
        ),
    )

    st.subheader("Steekproef")
    n_patients = st.slider(
        "Steekproefgrootte", 100, 3000, 500, step=50,
        help="Aantal synthetische patiënten. Grotere steekproeven maken schattingen "
             "stabieler en maken zwakke confounding-effecten eerder statistisch significant.",
    )
    seed = st.number_input(
        "Random seed", value=42, step=1,
        help="Startnummer voor de pseudo-random generator. Verander dit voor andere realisaties.",
    )

    # distribution-specific controls
    if dist_mode == DIST_SIMPLE:
        st.subheader("Tumorvolume (GTV)")
        tv_mu = st.slider(
            "Gemiddelde GTV (cc)", 10.0, 200.0, 45.0, step=1.0,
            help="Gemiddelde van de normale verdeling voor tumorvolume.",
        )
        tv_sigma = st.slider(
            "Spreiding GTV (cc)", 5.0, 60.0, 15.0, step=1.0,
            help="Standaarddeviatie van de GTV-verdeling.",
        )
        st.subheader("Mean Heart Dose = a·GTV + ruis")
        corr_a = st.slider(
            "Correlatiecoëfficiënt a (GTV → MHD)", 0.0, 1.0, 0.5, step=0.05,
            help="Bepaalt hoe sterk GTV de MHD aandrijft. Hogere waarden = sterkere confounding.",
        )
        mhd_noise_sd = st.slider(
            "Ruisniveau MHD (Gy)", 0.1, 10.0, 3.0, step=0.1,
            help="Willekeurige variatie in MHD niet verklaard door GTV. Hogere ruis verzwakt de GTV-MHD correlatie.",
        )
    else:
        st.subheader("Correlatie (Gaussian copula)")
        target_corr = st.slider(
            "Target correlatie GTV ↔ MHD", 0.0, 0.9, 0.45, step=0.05,
            help="Controls how strongly larger tumors also tend to receive higher mean heart dose. "
                 "This creates confounding when MHD itself has no causal effect.",
        )

    # survival scenario
    st.subheader("Overlevingsscenario")
    survival_scenario = st.selectbox(
        "Scenario",
        [SCEN_A, SCEN_B, SCEN_C],
        help="**A:** alleen GTV causaal; MHD is confounder. "
             "**B:** GTV én MHD causaal. "
             "**C:** artikel-formule (aanbevolen met artikel-achtige verdelingen).",
    )
    if survival_scenario == SCEN_A:
        b1 = st.slider(
            "b1 — causaal effect GTV (logit, gestand.)", -5.0, 0.0, -2.0, step=0.1,
            help="Werkelijk causaal effect van GTV op overlevingskans. Negatiever = groter GTV → lagere kans.",
        )
        b2_val     = 0.0
        surv_noise = st.slider(
            "Ruisniveau overleving", 0.0, 3.0, 1.0, step=0.1,
            help="Adds random patient-level variation to survival risk that is not explained by GTV or MHD. "
                 "Higher values make the outcome more random and reduce model performance.",
        )
    elif survival_scenario == SCEN_B:
        b1 = st.slider(
            "b1 — causaal effect GTV (logit, gestand.)", -5.0, 0.0, -2.0, step=0.1,
            help="Werkelijk causaal effect van GTV op overlevingskans.",
        )
        b2_val = st.slider(
            "b2 — causaal effect MHD (logit, gestand.)", -3.0, 0.0, -1.0, step=0.1,
            help="Werkelijk causaal effect van MHD op overlevingskans.",
        )
        surv_noise = st.slider(
            "Ruisniveau overleving", 0.0, 3.0, 1.0, step=0.1,
            help="Adds random patient-level variation to survival risk that is not explained by GTV or MHD. "
                 "Higher values make the outcome more random and reduce model performance.",
        )
    else:
        b1, b2_val, surv_noise = -2.0, 0.0, 0.0
        st.info(
            "Mortaliteitskans via artikel-formule:\n\n"
            "`logit(p) = −1.3409 + 0.0590·√GTV + 0.2635·√MHD`\n\n"
            "GTV én MHD zijn beide causaal."
        )

    mhd_is_causal = survival_scenario in (SCEN_B, SCEN_C)

    # baseline survival
    st.subheader("Baseline overleving")
    baseline_survival = st.slider(
        "Baseline 2-jaars overleving", 0.20, 0.90, 0.494, step=0.01,
        help=(
            "Sets the average 2-year survival level in the synthetic dataset before "
            "individual risk differences from GTV and MHD are applied. "
            "The article reports approximately 49.4% 2-year survival, equivalent to "
            "50.6% 24-month mortality. "
            "Used as intercept in scenarios A and B; optionally applied to scenario C."
        ),
    )
    if survival_scenario == SCEN_C:
        calibrate_article = st.checkbox(
            "Kalibreer artikel-model op geselecteerde baseline",
            value=False,
            help=(
                "If enabled, adjust only the intercept of the article formula so that "
                "the average simulated 24-month mortality approximately matches the "
                "selected baseline mortality. "
                "The GTV (0.0590) and MHD (0.2635) coefficients remain unchanged."
            ),
        )
    else:
        calibrate_article = False

    # model settings
    st.subheader("Modelinstellingen")
    use_sqrt = st.checkbox(
        "Gebruik √-getransformeerde predictoren",
        value=(dist_mode == DIST_ARTICLE),
        key=f"use_sqrt_{dist_mode}",
        help="Als ingeschakeld: modellen gefit op √GTV en √MHD, vergelijkbaar met het gepubliceerde model. "
             "Beïnvloedt alleen het model, niet de datagen.",
    )
    test_size = st.slider(
        "Testset aandeel", 0.1, 0.5, 0.2, step=0.05,
        help="Fraction of patients held out for testing the model. A larger test set gives a more "
             "stable validation estimate but leaves fewer patients for model fitting.",
    )

    st.divider()
    st.subheader("Protonentherapie")
    mhd_reduction_factor = st.slider(
        "MHD reductiefactor met protonentherapie", 0.0, 1.0, 0.6, step=0.05,
        help="0.6 betekent dat protonentherapie de mean heart dose terugbrengt tot 60 % van de "
             "fotonwaarde. 0.0 = volledige eliminatie; 1.0 = geen reductie.",
    )
    apply_true_proton = st.toggle(
        "Echte overlevingswinst van MHD-reductie",
        value=False,
        help="Uit (standaard): protonreductie beïnvloedt alleen de modelvoorspelling, niet de "
             "werkelijke gesimuleerde uitkomst. "
             "Aan: verlaagde MHD verbetert ook de gesimuleerde werkelijke overleving.",
    )
    if apply_true_proton:
        proton_effect_strength = st.slider(
            "Sterkte echt MHD-reductieffect", 0.0, 1.0, 0.1, step=0.01,
            help="Controls whether lowering MHD truly improves survival in the simulated data. "
                 "0 = geen echt effect; hogere waarden = sterkere werkelijke winst.",
        )
    else:
        proton_effect_strength = 0.0

    st.divider()
    show_raw = st.checkbox(
        "Toon ruwe data", value=False,
        help="Toon de eerste 50 rijen van de gegenereerde dataset.",
    )

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

# ── survival generation (store true_mort_logit for proton counterfactual) ─────

_baseline_mort = 1.0 - baseline_survival

if survival_scenario == SCEN_C:
    article_slopes = 0.0590 * np.sqrt(tumor_volume) + 0.2635 * np.sqrt(mean_heart_dose)
    if calibrate_article:
        # Shift intercept so E[p_mortality] ≈ baseline_mortality
        _article_int = np.log(_baseline_mort / (1.0 - _baseline_mort)) - article_slopes.mean()
    else:
        _article_int = -1.3409
    true_mort_logit = _article_int + article_slopes
    p_mortality     = 1.0 / (1.0 + np.exp(-true_mort_logit))
    survival        = rng.binomial(1, 1.0 - p_mortality).astype(int)
else:
    tv_norm    = (tumor_volume    - tumor_volume.mean())    / tumor_volume.std()
    mhd_norm   = (mean_heart_dose - mean_heart_dose.mean()) / mean_heart_dose.std()
    # Intercept derived from baseline survival: logit(baseline_survival)
    _surv_int       = np.log(baseline_survival / _baseline_mort)
    surv_logit      = _surv_int + b1 * tv_norm + b2_val * mhd_norm + surv_noise * rng.standard_normal(n_patients)
    true_mort_logit = -surv_logit
    survival        = rng.binomial(1, 1.0 / (1.0 + np.exp(-surv_logit))).astype(int)

# ── dataframe ─────────────────────────────────────────────────────────────────

df = pd.DataFrame({
    "tumor_volume_cc":    tumor_volume.round(2),
    "mean_heart_dose_gy": mean_heart_dose.round(2),
    "survival":           survival,
})
y = df["survival"].values

# ── validation table ──────────────────────────────────────────────────────────

pearson_r, pearson_p = stats.pearsonr(tumor_volume, mean_heart_dose)
mortality_rate = 1.0 - survival.mean()

with st.expander("Validatietabel — beschrijvende statistieken", expanded=True):
    vc1, vc2 = st.columns(2)
    with vc1:
        st.dataframe(
            pd.DataFrame({"GTV (cc)": describe_arr(tumor_volume),
                          "MHD (Gy)": describe_arr(mean_heart_dose)}),
            use_container_width=True,
        )
        if dist_mode == DIST_ARTICLE:
            st.caption(
                "Referentie (artikel): GTV gem.≈110 cc, med.≈69 cc, SD≈130 cc; "
                "MHD gem.≈12 Gy, med.≈11 Gy, SD≈8.2 Gy."
            )
    with vc2:
        st.metric("Pearson r (GTV ↔ MHD)", f"{pearson_r:.3f}")
        st.metric("p-waarde correlatie",    fmt_pval(pearson_p))
        st.markdown("**Baseline vs gesimuleerd**")
        bv_df = pd.DataFrame({
            "": ["Geselecteerd (baseline)", "Gesimuleerd"],
            "2-jaars overleving":      [
                f"{baseline_survival * 100:.1f} %",
                f"{survival.mean() * 100:.1f} %",
            ],
            "24-maands mortaliteit":   [
                f"{_baseline_mort * 100:.1f} %",
                f"{mortality_rate * 100:.1f} %",
            ],
        })
        st.dataframe(bv_df, use_container_width=True, hide_index=True)
        if survival_scenario == SCEN_C and not calibrate_article:
            st.caption(
                "Scenario C gebruikt de vaste artikel-intercept (−1.3409). "
                "Schakel 'Kalibreer artikel-model' in om de gesimuleerde mortaliteit "
                "op de geselecteerde baseline af te stemmen."
            )

# ── raw data ──────────────────────────────────────────────────────────────────

if show_raw:
    st.subheader("Ruwe data (eerste 50 rijen)")
    st.dataframe(df.head(50), use_container_width=True)

# ── summary metrics ───────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
c1.metric("Patiënten",             n_patients)
c2.metric("Overlevingspercentage", f"{survival.mean()*100:.1f}%")
c3.metric("Pearson r (GTV ↔ MHD)", f"{pearson_r:.3f}")
c4.metric("p-waarde correlatie",   fmt_pval(pearson_p))

# ── feature preparation ───────────────────────────────────────────────────────

X_raw = df[["tumor_volume_cc", "mean_heart_dose_gy"]].values
if use_sqrt:
    X_feat = np.sqrt(X_raw)
    fl = ["√GTV", "√MHD"]
else:
    X_feat = X_raw.copy()
    fl = ["GTV", "MHD"]

# ── train / test split ────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X_feat, y, test_size=test_size, random_state=int(seed)
)

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
    "Intercept":            [
        f"{_lr(pipe_A).intercept_[0]:.3f}",
        f"{_lr(pipe_B).intercept_[0]:.3f}",
        f"{interc_C:.3f}",
    ],
    "Coëfficiënten (std.)": [
        _coefstr(pipe_A, [fl[0]]),
        _coefstr(pipe_B, [fl[1]]),
        _coefstr(pipe_C, fl),
    ],
})

st.subheader("Logistische regressie — gecombineerd model")
r1, r2 = st.columns([3, 1])
with r1:
    st.dataframe(coef_df, use_container_width=True, hide_index=True)
    st.caption(
        f"Predictoren: {', '.join(fl)} (StandardScaler). "
        "p-waarden via Wald-test.  *** p<0.001  ** p<0.01  * p<0.05  ns p≥0.05"
    )
with r2:
    st.metric("ROC AUC (testset)",       f"{auc_C:.3f}")
    st.metric("Nauwkeurigheid (testset)", f"{accuracy_C*100:.1f}%")

if not mhd_is_causal and pvals_C[1] < 0.05:
    st.warning(
        f"**Confounding waarschuwing:** {fl[1]} is statistisch significant "
        f"(p = {fmt_pval(pvals_C[1])}) terwijl het **geen causale invloed** heeft. "
        f"Dit komt doordat MHD correleert met GTV (r = {pearson_r:.2f})."
    )
elif not mhd_is_causal:
    st.info(
        f"{fl[1]} is niet significant (p = {fmt_pval(pvals_C[1])}). "
        f"Vergroot de correlatie of steekproef om confounding zichtbaar te maken."
    )
else:
    st.success(
        f"MHD heeft een causaal effect in dit scenario. "
        f"Coëfficiënt {fl[1]}: {coefs_C[1]:.3f},  p = {fmt_pval(pvals_C[1])}."
    )

# ── proton calculations (all patients) ───────────────────────────────────────

mhd_proton      = mean_heart_dose * mhd_reduction_factor
if use_sqrt:
    X_feat_proton = np.column_stack([np.sqrt(tumor_volume), np.sqrt(mhd_proton)])
else:
    X_feat_proton = np.column_stack([tumor_volume, mhd_proton])

p_mort_photon = 1.0 - pipe_C.predict_proba(X_feat)[:, 1]
p_mort_proton = 1.0 - pipe_C.predict_proba(X_feat_proton)[:, 1]
delta_model   = p_mort_photon - p_mort_proton          # positive = proton reduces mortality
nnt_model     = np.where(delta_model > 0, 1.0 / delta_model, np.nan)

# True counterfactual when causal toggle is on
if apply_true_proton and proton_effect_strength > 0:
    mhd_diff             = mean_heart_dose - mhd_proton          # positive (MHD reduced)
    true_mort_logit_prot = true_mort_logit - proton_effect_strength * mhd_diff
    true_p_mort_photon   = 1.0 / (1.0 + np.exp(-true_mort_logit))
    true_p_mort_proton   = 1.0 / (1.0 + np.exp(-true_mort_logit_prot))
    true_delta           = true_p_mort_photon - true_p_mort_proton
else:
    true_delta = None

# ── tabs ──────────────────────────────────────────────────────────────────────

st.subheader("Visualisaties")

SC = {0: "#e74c3c", 1: "#2ecc71"}
SL = {0: "Niet overleefd", 1: "Overleefd"}
MS = {"A": ("#2980b9", "solid"), "B": ("#e67e22", "dashed"), "C": ("#27ae60", "dashdot")}

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Histogrammen", "Scatter GTV ↔ MHD", "Correlatiemat.",
    "ROC-curve", "Verwarringsmatrix", "ROC vergelijking",
    "Dosis-effect", "Proton benefit",
])

# ── tab 1: histograms (+ optional cumulative) ─────────────────────────────────

with tab1:
    show_cumul = st.checkbox("Toon cumulatieve histogrammen", value=False,
                             help="Vervang standaard histogrammen door cumulatieve verdelingen per overlevingsstatus.")
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, col, xlabel in zip(
        axes,
        ["tumor_volume_cc", "mean_heart_dose_gy"],
        ["GTV (cc)", "Mean Heart Dose (Gy)"],
    ):
        for val in [0, 1]:
            data = df.loc[df["survival"] == val, col].values
            if show_cumul:
                ax.hist(data, bins=60, density=True, cumulative=True,
                        alpha=0.6, color=SC[val], label=SL[val], edgecolor="none")
                ax.set_ylabel("Cumulatief aandeel")
                ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
            else:
                ax.hist(data, bins=30, alpha=0.6, color=SC[val], label=SL[val], edgecolor="none")
                ax.set_ylabel("Aantal patiënten")
        ax.set_xlabel(xlabel)
        ax.legend()
    title = "Cumulatieve verdeling" if show_cumul else "Verdeling"
    fig.suptitle(f"{title} per overlevingsstatus", fontsize=12)
    fig.tight_layout()
    centered(fig)

# ── tab 2: scatter GTV ↔ MHD ─────────────────────────────────────────────────

with tab2:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(tumor_volume, mean_heart_dose,
               c=[SC[s] for s in survival], alpha=0.4, s=18, linewidths=0)
    m_r, b_r, *_ = stats.linregress(tumor_volume, mean_heart_dose)
    xl = np.array([tumor_volume.min(), tumor_volume.max()])
    ax.plot(xl, m_r * xl + b_r, color="#2c3e50", lw=1.5,
            label=f"y = {m_r:.2f}x + {b_r:.1f}  (r = {pearson_r:.2f}, p = {fmt_pval(pearson_p)})")
    h = [Patch(color=SC[v], alpha=0.7, label=SL[v]) for v in [0, 1]]
    l1 = ax.legend(handles=h, loc="upper left", fontsize=8)
    ax.add_artist(l1)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("GTV (cc)")
    ax.set_ylabel("Mean Heart Dose (Gy)")
    ax.set_title("Scatter: GTV vs MHD — kleur = overlevingsstatus")
    fig.tight_layout()
    centered(fig)
    st.caption("De regressielijn toont GTV → MHD. Doordat overleving afhangt van GTV, "
               "lijkt MHD ook te scheiden — dit is de confounder.")

# ── tab 3: correlation matrix ─────────────────────────────────────────────────

with tab3:
    corr_df = df[["tumor_volume_cc", "mean_heart_dose_gy", "survival"]].corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr_df, annot=True, fmt=".3f", cmap="RdYlGn",
                center=0, vmin=-1, vmax=1, mask=np.eye(3, dtype=bool),
                ax=ax, linewidths=0.5)
    ax.set_title("Pearson correlatiemat.")
    fig.tight_layout()
    centered(fig)
    st.caption("MHD correleert met zowel GTV als survival, ook zonder causaal pad MHD → survival (scenario A).")

# ── tab 4: ROC combined model ─────────────────────────────────────────────────

with tab4:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr_C, tpr_C, color=MS["C"][0], lw=2,
            label=f"{fl[0]} + {fl[1]}  (AUC = {auc_C:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Toeval")
    ax.fill_between(fpr_C, tpr_C, alpha=0.08, color=MS["C"][0])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-curve — gecombineerd model (testset)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    centered(fig)

# ── tab 5: confusion matrix ───────────────────────────────────────────────────

with tab5:
    cm = confusion_matrix(y_test, y_pred_C)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Niet overleefd", "Overleefd"],
                yticklabels=["Niet overleefd", "Overleefd"], ax=ax)
    ax.set_xlabel("Voorspeld"); ax.set_ylabel("Werkelijk")
    ax.set_title("Verwarringsmatrix (testset)")
    fig.tight_layout()
    centered(fig)

# ── tab 6: ROC comparison + fitted survival functions ─────────────────────────

with tab6:
    st.markdown(
        "> **Predictie vs. causaliteit** — Het MHD-only model kan voorspellende waarde "
        "vertonen wanneer MHD correleert met tumorvolume, ook als MHD in de simulatie "
        "**geen** direct causaal effect heeft. Dit illustreert het verschil tussen "
        "*voorspelling* en *causale behandeleffecten*: een goede predictor hoeft geen "
        "effectieve behandeldoelstelling te zijn."
    )

    st.markdown("### ROC-vergelijking")
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, name, f, t, a_val in [
        ("A", f"{fl[0]} only",        fpr_A, tpr_A, auc_A),
        ("B", f"{fl[1]} only",        fpr_B, tpr_B, auc_B),
        ("C", f"{fl[0]} + {fl[1]}", fpr_C, tpr_C, auc_C),
    ]:
        color, ls = MS[key]
        ax.plot(f, t, color=color, lw=2, linestyle=ls,
                label=f"{name},  AUC = {a_val:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", lw=1, label="Toeval  (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-vergelijking — drie modellen (testset)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    centered(fig)

    st.markdown("**Samenvatting per model**")
    st.dataframe(roc_summary, use_container_width=True, hide_index=True)

    if not mhd_is_causal and auc_B > 0.55:
        st.warning(
            f"**Confounding zichtbaar:** Model B ({fl[1]} only, AUC = {auc_B:.3f}) presteert "
            f"beter dan toeval terwijl MHD **geen causale werking** heeft "
            f"(r(GTV,MHD) = {pearson_r:.2f})."
        )
    elif not mhd_is_causal:
        st.info(f"Model B ({fl[1]} only, AUC = {auc_B:.3f}) presteert nauwelijks beter dan toeval.")
    else:
        st.success(f"MHD heeft een causaal effect. Model B AUC = {auc_B:.3f}, C AUC = {auc_C:.3f}.")

    st.markdown("### Fitted survival functions")
    tv_grid   = np.linspace(tumor_volume.min(),    tumor_volume.max(),    200)
    mhd_grid  = np.linspace(mean_heart_dose.min(), mean_heart_dose.max(), 200)
    tv_in     = np.sqrt(tv_grid)  if use_sqrt else tv_grid
    mhd_in    = np.sqrt(mhd_grid) if use_sqrt else mhd_grid
    mhd_p10   = np.percentile(mean_heart_dose, 10)
    mhd_p50   = np.percentile(mean_heart_dose, 50)
    mhd_p90   = np.percentile(mean_heart_dose, 90)

    sv1, sv2, sv3 = st.columns(3)
    with sv1:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(tumor_volume, survival, c=[SC[s] for s in survival],
                   alpha=0.25, s=10, linewidths=0, zorder=1)
        ax.plot(tv_grid, pipe_A.predict_proba(tv_in.reshape(-1, 1))[:, 1],
                color=MS["A"][0], lw=2, zorder=2)
        ax.set_xlabel("GTV (cc)"); ax.set_ylabel("Voorspelde overlevingskans")
        ax.set_title(f"Model A — {fl[0]} only"); ax.set_ylim(0, 1)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    with sv2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(mean_heart_dose, survival, c=[SC[s] for s in survival],
                   alpha=0.25, s=10, linewidths=0, zorder=1)
        ax.plot(mhd_grid, pipe_B.predict_proba(mhd_in.reshape(-1, 1))[:, 1],
                color=MS["B"][0], lw=2, zorder=2)
        ax.set_xlabel("MHD (Gy)"); ax.set_ylabel("Voorspelde overlevingskans")
        ax.set_title(f"Model B — {fl[1]} only"); ax.set_ylim(0, 1)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    with sv3:
        fig, ax = plt.subplots(figsize=(5, 4))
        for mhd_val, label, color in [
            (mhd_p10, f"MHD laag (p10) = {mhd_p10:.1f} Gy", "#1a9641"),
            (mhd_p50, f"MHD mediaan = {mhd_p50:.1f} Gy",     "#fdae61"),
            (mhd_p90, f"MHD hoog (p90) = {mhd_p90:.1f} Gy",  "#d7191c"),
        ]:
            if use_sqrt:
                Xc = np.column_stack([np.sqrt(tv_grid), np.full_like(tv_grid, np.sqrt(mhd_val))])
            else:
                Xc = np.column_stack([tv_grid, np.full_like(tv_grid, mhd_val)])
            ax.plot(tv_grid, pipe_C.predict_proba(Xc)[:, 1], color=color, lw=2, label=label)
        ax.set_xlabel("GTV (cc)"); ax.set_ylabel("Voorspelde overlevingskans")
        ax.set_title(f"Model C — {fl[0]} + {fl[1]}"); ax.set_ylim(0, 1)
        ax.legend(fontsize=6.5, loc="upper right")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    st.caption(
        "Model C: drie curves voor lage (p10), gemiddelde (p50) en hoge (p90) MHD. "
        "In scenario A weerspiegelen afwijkende curves confounding, niet een direct effect."
    )

# ── tab 7: dose-effect curves ─────────────────────────────────────────────────

with tab7:
    st.markdown("### Dosis-effect curves — gecombineerd model")

    gtv_med_val = np.median(tumor_volume)
    mhd_med_val = np.median(mean_heart_dose)
    tv_g  = np.linspace(tumor_volume.min(),    tumor_volume.max(),    300)
    mhd_g = np.linspace(mean_heart_dose.min(), mean_heart_dose.max(), 300)

    # GTV curves (MHD fixed at median)
    if use_sqrt:
        X_gtv_curve = np.column_stack([np.sqrt(tv_g), np.full_like(tv_g, np.sqrt(mhd_med_val))])
        X_mhd_curve = np.column_stack([np.full_like(mhd_g, np.sqrt(gtv_med_val)), np.sqrt(mhd_g)])
    else:
        X_gtv_curve = np.column_stack([tv_g, np.full_like(tv_g, mhd_med_val)])
        X_mhd_curve = np.column_stack([np.full_like(mhd_g, gtv_med_val), mhd_g])

    surv_vs_gtv = pipe_C.predict_proba(X_gtv_curve)[:, 1]
    mort_vs_gtv = 1.0 - surv_vs_gtv
    surv_vs_mhd = pipe_C.predict_proba(X_mhd_curve)[:, 1]
    mort_vs_mhd = 1.0 - surv_vs_mhd

    de1, de2 = st.columns(2)

    with de1:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(tv_g, surv_vs_gtv, color="#2ecc71", lw=2, label="2-jaars overleving")
        ax.plot(tv_g, mort_vs_gtv, color="#e74c3c", lw=2, linestyle="--", label="2-jaars mortaliteit")
        ax.axvline(gtv_med_val, color="gray", lw=1, linestyle=":", label=f"Mediaan GTV = {gtv_med_val:.0f} cc")
        ax.set_xlabel("GTV (cc)")
        ax.set_ylabel("Kans")
        ax.set_title(f"Dosis-effect GTV\n(MHD vastgehouden op mediaan {mhd_med_val:.1f} Gy)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False); plt.close(fig)

    with de2:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(mhd_g, surv_vs_mhd, color="#2ecc71", lw=2, label="2-jaars overleving")
        ax.plot(mhd_g, mort_vs_mhd, color="#e74c3c", lw=2, linestyle="--", label="2-jaars mortaliteit")
        ax.axvline(mhd_med_val, color="gray", lw=1, linestyle=":", label=f"Mediaan MHD = {mhd_med_val:.1f} Gy")
        ax.set_xlabel("Mean Heart Dose (Gy)")
        ax.set_ylabel("Kans")
        ax.set_title(f"Dosis-effect MHD\n(GTV vastgehouden op mediaan {gtv_med_val:.0f} cc)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False); plt.close(fig)

    if not mhd_is_causal:
        st.info(
            f"In scenario A (GTV causaal, MHD confounder) toont de MHD-curve een "
            f"associatie die **niet causaal** is. De curve weerspiegelt de confounding "
            f"via GTV, niet een direct effect van hartdosis op overleving."
        )

# ── tab 8: proton benefit ─────────────────────────────────────────────────────

with tab8:
    st.markdown(
        "Dit tabblad vergelijkt foton- en gesimuleerde protonenplannen door de MHD te verlagen. "
        "De voorspelde winst komt van het gefitte model. **Als MHD alleen gecorreleerd is met GTV "
        "maar niet causaal is, kan het model winst voorspellen terwijl de werkelijke gesimuleerde "
        "uitkomst niet verbetert.** Dit illustreert het verschil tussen model-gebaseerde "
        "voorspellingen en causale behandeleffecten."
    )

    # ── summary metrics ───────────────────────────────────────────────────────
    pm1, pm2, pm3, pm4 = st.columns(4)
    pm1.metric("Gem. mortaliteit (foton)",  f"{p_mort_photon.mean()*100:.1f}%")
    pm2.metric("Gem. mortaliteit (proton)", f"{p_mort_proton.mean()*100:.1f}%")
    pm3.metric("Gem. ARR (model)",          f"{delta_model.mean()*100:.2f}%")
    pm4.metric("Gem. NNT (model)",          fmt_f(np.nanmean(nnt_model), decimals=1))

    if true_delta is not None:
        pt1, pt2, pt3, _ = st.columns(4)
        pt1.metric("Gem. ARR (werkelijk, via logit)",  f"{true_delta.mean()*100:.2f}%")
        pt2.metric(
            "Overestimatie door confounding",
            f"{(delta_model.mean() - true_delta.mean())*100:.2f}%-punt",
            help="Positief = model voorspelt meer winst dan de werkelijke logit impliceert.",
        )
        if not mhd_is_causal:
            st.warning(
                f"**Confounding in de protonenwinst:** het model voorspelt gemiddeld "
                f"{delta_model.mean()*100:.2f}% absolute risicowinst, maar de werkelijke "
                f"gesimuleerde winst (via het causale logit-pad) is slechts "
                f"{true_delta.mean()*100:.2f}%. Het verschil wordt veroorzaakt door confounding: "
                f"MHD is gecorreleerd met GTV maar heeft geen direct causaal effect."
            )

    st.markdown("---")

    # ── delta histogram ───────────────────────────────────────────────────────
    st.markdown("#### Verdeling van voorspelde mortaliteitswinst (Δ) per patiënt")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(delta_model * 100, bins=40, color="#2980b9", edgecolor="none", alpha=0.8)
    ax.set_xlabel("Absolute risicoreductie Δ (%-punt)")
    ax.set_ylabel("Aantal patiënten")
    ax.set_title("Verdeling van model-gebaseerde Δ")
    if true_delta is not None:
        ax.hist(true_delta * 100, bins=40, color="#e67e22", edgecolor="none",
                alpha=0.5, label="Werkelijk Δ (via causaal logit)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    centered(fig)

    # ── NNT table ─────────────────────────────────────────────────────────────
    st.markdown("#### NNT-tabel per Δ-bin")
    nnt_df = build_nnt_table(delta_model, n_patients)
    st.dataframe(nnt_df, use_container_width=True, hide_index=True)
    st.caption(
        "Δ = mortaliteit(foton) − mortaliteit(proton); positief = proton gunstig. "
        "NNT = 1/Δ per patiënt. Mean/Median NNT = gemiddeld/mediaan van de individuele NNTs in de bin. "
        "1/Mean Δ en 1/Median Δ zijn alternatieve NNT-schattingen op basis van groepsgemiddelden."
    )

    # ── bar chart patients per bin ─────────────────────────────────────────────
    st.markdown("#### Patiënten per Δ-bin")
    bin_counts = nnt_df["N"].values
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(_BLABELS, bin_counts, color="#2980b9", edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, padding=2, fontsize=7)
    ax.set_xlabel("Absolute risicoreductie Δ")
    ax.set_ylabel("Aantal patiënten")
    ax.set_title(f"Patiënten per Δ-bin  (MHD reductiefactor = {mhd_reduction_factor})")
    ax.tick_params(axis="x", rotation=40, labelsize=7)
    fig.tight_layout()
    centered(fig)

    if not mhd_is_causal:
        st.caption(
            "In scenario A heeft MHD geen causale werking. De spreiding in Δ "
            "komt door de spreiding in MHD (en dus GTV). Patiënten met hoge MHD "
            "hebben ook een hoge GTV — hun hogere 'voorspelde winst' is een artefact "
            "van confounding."
        )

# ── footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    f"Scenario {survival_scenario[:1]} · "
    "Standaard causaal diagram: **GTV → overleving** en **GTV → MHD**. "
    "In scenario A heeft MHD geen direct pad naar overleving. "
    "In scenario B en C heeft MHD ook een causaal effect."
)
