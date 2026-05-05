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

_GTV_MU    = np.log(69.0)
_GTV_SIGMA = np.sqrt(2 * (np.log(110.0) - np.log(69.0)))
_MHD_K     = (12.0 / 8.2) ** 2
_MHD_THETA = 8.2 ** 2 / 12.0

_BINS    = [0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40,
            0.50, 0.60, 0.70, 0.80, 0.90, 1.01]
_BLABELS = ["0–2 %", "2–5 %", "5–10 %", "10–20 %", "20–30 %", "30–40 %",
            "40–50 %", "50–60 %", "60–70 %", "70–80 %", "80–90 %", "90–100 %"]

_NOISE_LEVELS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]


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
st.title("24-maands mortaliteit: confounding in klinische data")
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
        [DIST_ARTICLE, DIST_SIMPLE],
        help=(
            "**Artikel-achtig:** log-normale GTV en gamma-verdeelde MHD via Gaussian copula "
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
        help="Startnummer voor de pseudo-random generator.",
    )

    if dist_mode == DIST_SIMPLE:
        st.subheader("Tumorvolume (GTV)")
        tv_mu = st.slider("Gemiddelde GTV (cc)", 10.0, 200.0, 45.0, step=1.0,
                          help="Gemiddelde van de normale verdeling voor tumorvolume.")
        tv_sigma = st.slider("Spreiding GTV (cc)", 5.0, 60.0, 15.0, step=1.0,
                             help="Standaarddeviatie van de GTV-verdeling.")
        st.subheader("Mean Heart Dose = a·GTV + ruis")
        corr_a = st.slider(
            "Correlatiecoëfficiënt a (GTV → MHD)", 0.0, 1.0, 0.5, step=0.05,
            help="Controls how strongly larger tumors also tend to receive higher mean heart dose. "
                 "This creates confounding when MHD itself has no causal effect.",
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

    st.subheader("Mortaliteitsscenario")
    survival_scenario = st.selectbox(
        "Scenario",
        [SCEN_A, SCEN_B, SCEN_C],
        help=(
            "**A:** Mortaliteit gegenereerd vanuit GTV alleen. MHD kan toch voorspellend zijn "
            "omdat het correleert met GTV. "
            "**B:** Mortaliteit gegenereerd vanuit zowel GTV als MHD. "
            "**C:** Artikel-formule met √GTV en √MHD."
        ),
    )
    if survival_scenario == SCEN_A:
        b1 = st.slider(
            "b1 — causaal effect GTV op mortaliteit (logit, gestand.)", 0.0, 5.0, 2.0, step=0.1,
            help="Werkelijk causaal effect van GTV op mortaliteitskans. Groter = groter GTV → hogere mortaliteit.",
        )
        b2_val     = 0.0
        surv_noise = st.slider(
            "Mortaliteitsruisniveau", 0.0, 3.0, 1.0, step=0.1,
            help="Adds random patient-level variation to the mortality logit that is not explained "
                 "by GTV or MHD. Higher values make mortality more random, weaken the visible "
                 "dose-response relationship, and usually reduce model AUC.",
        )
    elif survival_scenario == SCEN_B:
        b1 = st.slider(
            "b1 — causaal effect GTV op mortaliteit (logit, gestand.)", 0.0, 5.0, 2.0, step=0.1,
            help="Werkelijk causaal effect van GTV op mortaliteitskans.",
        )
        b2_val = st.slider(
            "b2 — causaal effect MHD op mortaliteit (logit, gestand.)", 0.0, 3.0, 1.0, step=0.1,
            help="Werkelijk causaal effect van MHD op mortaliteitskans.",
        )
        surv_noise = st.slider(
            "Mortaliteitsruisniveau", 0.0, 3.0, 1.0, step=0.1,
            help="Adds random patient-level variation to the mortality logit that is not explained "
                 "by GTV or MHD. Higher values make mortality more random, weaken the visible "
                 "dose-response relationship, and usually reduce model AUC.",
        )
    else:
        b1, b2_val, surv_noise = 0.059, 0.2635, 0.0
        st.info(
            "Mortaliteit via artikel-formule:\n\n"
            "`logit(p) = −1.3409 + 0.0590·√GTV + 0.2635·√MHD`\n\n"
            "GTV én MHD zijn beide causaal."
        )

    mhd_is_causal = survival_scenario in (SCEN_B, SCEN_C)

    st.subheader("Baseline mortaliteit")
    baseline_survival = st.slider(
        "Baseline 2-jaars overleving", 0.20, 0.90, 0.494, step=0.01,
        help=(
            "Sets the average 2-year survival level before individual risk differences from GTV "
            "and MHD are applied. The article reports approximately 49.4% 2-year survival "
            "(= 50.6% 24-month mortality). Used as intercept in scenarios A and B."
        ),
    )
    if survival_scenario == SCEN_C:
        calibrate_article = st.checkbox(
            "Kalibreer artikel-model op geselecteerde baseline",
            value=False,
            help="Adjust only the intercept of the article formula so that the average simulated "
                 "24-month mortality matches the selected baseline. "
                 "GTV (0.0590) and MHD (0.2635) coefficients remain unchanged.",
        )
    else:
        calibrate_article = False

    st.subheader("Modelinstellingen")
    use_sqrt = st.checkbox(
        "Gebruik √-getransformeerde predictoren",
        value=(dist_mode == DIST_ARTICLE),
        key=f"use_sqrt_{dist_mode}",
        help="Als ingeschakeld: modellen gefit op √GTV en √MHD. "
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
        help="0.6 = protonentherapie brengt MHD terug tot 60 % van de fotonwaarde.",
    )
    apply_true_proton = st.toggle(
        "Echte mortaliteitswinst van MHD-reductie",
        value=False,
        help="Uit: protonreductie beïnvloedt alleen het model, niet de werkelijke uitkomst. "
             "Aan: verlaagde MHD vermindert ook de werkelijke gesimuleerde mortaliteit.",
    )
    if apply_true_proton:
        proton_effect_strength = st.slider(
            "Sterkte echt MHD-reductieffect", 0.0, 1.0, 0.1, step=0.01,
            help="Controls whether lowering MHD truly reduces mortality in the simulated data.",
        )
    else:
        proton_effect_strength = 0.0

    st.divider()
    show_raw = st.checkbox("Toon ruwe data", value=False,
                           help="Toon de eerste 50 rijen van de gegenereerde dataset.")

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
        _art_int = np.log(_baseline_mort / (1.0 - _baseline_mort)) - article_slopes.mean()
    else:
        _art_int = -1.3409
    true_mort_logit = _art_int + article_slopes
    _display_int    = _art_int
else:
    tv_norm  = (tumor_volume    - tumor_volume.mean())    / tumor_volume.std()
    mhd_norm = (mean_heart_dose - mean_heart_dose.mean()) / mean_heart_dose.std()
    _mort_int       = np.log(_baseline_mort / (1.0 - _baseline_mort))   # logit(baseline_mort)
    _display_int    = _mort_int
    # positive b1/b2 → higher GTV/MHD → higher mortality
    true_mort_logit = (_mort_int
                       + b1 * tv_norm
                       + b2_val * mhd_norm
                       + surv_noise * rng.standard_normal(n_patients))

p_mortality = 1.0 / (1.0 + np.exp(-true_mort_logit))
mortality   = rng.binomial(1, p_mortality).astype(int)   # 1 = died, 0 = survived
survival    = 1 - mortality                               # kept for coloring only

# ── dataframe ─────────────────────────────────────────────────────────────────

df = pd.DataFrame({
    "tumor_volume_cc":    tumor_volume.round(2),
    "mean_heart_dose_gy": mean_heart_dose.round(2),
    "mortality":          mortality,
})
y = df["mortality"].values   # 1 = died; models predict P(mortality=1)

# ── validation table ──────────────────────────────────────────────────────────

pearson_r, pearson_p = stats.pearsonr(tumor_volume, mean_heart_dose)
obs_mort_rate = mortality.mean()

with st.expander("Validatietabel — beschrijvende statistieken", expanded=True):
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
        st.metric("Pearson r (GTV ↔ MHD)", f"{pearson_r:.3f}")
        st.metric("p-waarde correlatie",    fmt_pval(pearson_p))
        st.markdown("**Baseline vs gesimuleerd**")
        st.dataframe(
            pd.DataFrame({
                "": ["Geselecteerd", "Gesimuleerd"],
                "2-jaars overleving":      [f"{baseline_survival*100:.1f} %",
                                            f"{(1-obs_mort_rate)*100:.1f} %"],
                "24-maands mortaliteit":   [f"{_baseline_mort*100:.1f} %",
                                            f"{obs_mort_rate*100:.1f} %"],
            }),
            use_container_width=True, hide_index=True,
        )
        if survival_scenario == SCEN_C and not calibrate_article:
            st.caption("Scenario C gebruikt vaste artikel-intercept (−1.3409).")

# ── raw data ──────────────────────────────────────────────────────────────────

if show_raw:
    st.subheader("Ruwe data (eerste 50 rijen)")
    st.dataframe(df.head(50), use_container_width=True)

# ── summary metrics ───────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
c1.metric("Patiënten",                n_patients)
c2.metric("24-maands mortaliteit",    f"{obs_mort_rate*100:.1f}%")
c3.metric("Pearson r (GTV ↔ MHD)",   f"{pearson_r:.3f}")
c4.metric("p-waarde correlatie",      fmt_pval(pearson_p))

# ── true mortality-generating formula ─────────────────────────────────────────

with st.expander("Ware mortaliteitsgenererende formule", expanded=False):
    if survival_scenario == SCEN_C:
        _int_label = f"{_display_int:.4f}"
        _coef_line = "0.0590 · √GTV + 0.2635 · √MHD"
        st.markdown(
            f"**Scenario C — Artikel-formule**\n\n"
            f"```\nlogit(p_mortaliteit) = {_int_label} + {_coef_line}\n```\n\n"
            f"{'*(intercept gekalibreerd op geselecteerde baseline)*' if calibrate_article else '*(artikel-intercept −1.3409)*'}"
        )
    else:
        _gtv_t = "√GTV" if use_sqrt else "GTV (gestand.)"
        _mhd_t = "√MHD" if use_sqrt else "MHD (gestand.)"
        _b2_line = f" + {b2_val:.2f} · {_mhd_t}" if survival_scenario == SCEN_B else ""
        _noise_line = f" + N(0, {surv_noise:.2f})" if surv_noise > 0 else ""
        st.markdown(
            f"**Scenario {'A' if survival_scenario == SCEN_A else 'B'}**\n\n"
            f"```\nlogit(p_mortaliteit) = {_display_int:.4f}"
            f" + {b1:.2f} · {_gtv_t}{_b2_line}{_noise_line}\n```"
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
        param_rows += [
            ("Mortaliteitsruisniveau σ", f"{surv_noise:.2f}", "N(0,σ) toegevoegd aan logit"),
        ]
    param_rows += [
        ("Geselecteerde baseline mortaliteit", f"{_baseline_mort*100:.1f} %", ""),
        ("Gesimuleerde mortaliteit",           f"{obs_mort_rate*100:.1f} %", ""),
    ]
    st.dataframe(
        pd.DataFrame(param_rows, columns=["Parameter", "Waarde", "Toelichting"]),
        use_container_width=True, hide_index=True,
    )

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

# ── three pipeline models (predicting mortality) ───────────────────────────────

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

# ── proton calculations ───────────────────────────────────────────────────────

mhd_proton      = mean_heart_dose * mhd_reduction_factor
if use_sqrt:
    X_feat_proton = np.column_stack([np.sqrt(tumor_volume), np.sqrt(mhd_proton)])
else:
    X_feat_proton = np.column_stack([tumor_volume, mhd_proton])

p_mort_photon = pipe_C.predict_proba(X_feat)[:, 1]          # predicted mortality, photon
p_mort_proton = pipe_C.predict_proba(X_feat_proton)[:, 1]   # predicted mortality, proton
delta_model   = p_mort_photon - p_mort_proton                # positive = proton reduces mortality
nnt_model     = np.where(delta_model > 0, 1.0 / delta_model, np.nan)

if apply_true_proton and proton_effect_strength > 0:
    mhd_diff             = mean_heart_dose - mhd_proton
    true_mort_logit_prot = true_mort_logit - proton_effect_strength * mhd_diff
    true_p_mort_photon   = 1.0 / (1.0 + np.exp(-true_mort_logit))
    true_p_mort_proton   = 1.0 / (1.0 + np.exp(-true_mort_logit_prot))
    true_delta           = true_p_mort_photon - true_p_mort_proton
else:
    true_delta = None

# ── noise impact (scenarios A/B only) ────────────────────────────────────────

def _noise_auc_row(nl: float) -> dict:
    """Recompute mortality + AUC for a given noise level using same GTV/MHD."""
    rng_n = np.random.default_rng(int(seed))
    tv_n  = (tumor_volume - tumor_volume.mean()) / tumor_volume.std()
    mh_n  = (mean_heart_dose - mean_heart_dose.mean()) / mean_heart_dose.std()
    logit_n = _display_int + b1 * tv_n + b2_val * mh_n + nl * rng_n.standard_normal(n_patients)
    mort_n  = rng_n.binomial(1, 1.0 / (1.0 + np.exp(-logit_n))).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X_feat, mort_n,
                                           test_size=test_size, random_state=int(seed))
    pa, pb, pc = make_pipe(), make_pipe(), make_pipe()
    pa.fit(Xtr[:, [0]], ytr); pb.fit(Xtr[:, [1]], ytr); pc.fit(Xtr, ytr)
    def _a(p, X, c=None):
        Xi = X[:, c] if c is not None else X
        _, _, a = _roc(p, Xi) if False else (None, None,
                   auc(*roc_curve(yte, p.predict_proba(Xi[:, c] if c else Xi)[:, 1])[:2]))
        return a
    fa, ta, aa = roc_curve(yte, pa.predict_proba(Xte[:, [0]])[:, 1])
    fb, tb, ab = roc_curve(yte, pb.predict_proba(Xte[:, [1]])[:, 1])
    fc, tc, ac = roc_curve(yte, pc.predict_proba(Xte)[:, 1])
    return {
        "Ruisniveau σ":      nl,
        "Obs. mortaliteit":  f"{mort_n.mean()*100:.1f} %",
        f"AUC {fl[0]}":      f"{auc(fa, ta):.3f}",
        f"AUC {fl[1]}":      f"{auc(fb, tb):.3f}",
        f"AUC {fl[0]}+{fl[1]}": f"{auc(fc, tc):.3f}",
    }

# ── tabs ──────────────────────────────────────────────────────────────────────

st.subheader("Visualisaties")

MC = {1: "#e74c3c", 0: "#2ecc71"}   # mortality colours: 1=gestorven=red, 0=niet gestorven=green
ML = {1: "Gestorven", 0: "Niet gestorven"}
MS = {"A": ("#2980b9", "solid"), "B": ("#e67e22", "dashed"), "C": ("#27ae60", "dashdot")}

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Histogrammen", "Scatter GTV ↔ MHD", "Correlatiemat.",
    "ROC-curve", "Verwarringsmatrix", "ROC vergelijking",
    "Dosis-effect", "Ruis-impact", "Proton benefit",
])

# ── tab 1: histograms ─────────────────────────────────────────────────────────

with tab1:
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

# ── tab 2: scatter GTV ↔ MHD ─────────────────────────────────────────────────

with tab2:
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

# ── tab 3: correlation matrix ─────────────────────────────────────────────────

with tab3:
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

# ── tab 4: ROC combined model ─────────────────────────────────────────────────

with tab4:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr_C, tpr_C, color=MS["C"][0], lw=2,
            label=f"{fl[0]} + {fl[1]}  (AUC = {auc_C:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Toeval")
    ax.fill_between(fpr_C, tpr_C, alpha=0.08, color=MS["C"][0])
    ax.set_xlabel("False Positive Rate (1 − specificiteit)")
    ax.set_ylabel("True Positive Rate (sensitiviteit)")
    ax.set_title("ROC-curve — gecombineerd mortaliteitsmodel (testset)")
    ax.legend(loc="lower right"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    centered(fig)

# ── tab 5: confusion matrix ───────────────────────────────────────────────────

with tab5:
    cm = confusion_matrix(y_test, y_pred_C)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=["Niet gestorven", "Gestorven"],
                yticklabels=["Niet gestorven", "Gestorven"], ax=ax)
    ax.set_xlabel("Voorspeld"); ax.set_ylabel("Werkelijk")
    ax.set_title("Verwarringsmatrix — mortaliteit (testset)")
    fig.tight_layout()
    centered(fig)

# ── tab 6: ROC comparison + fitted mortality functions ────────────────────────

with tab6:
    st.markdown(
        "> **Predictie vs. causaliteit** — Het MHD-only model kan 24-maands mortaliteit "
        "voorspellen wanneer MHD correleert met tumorvolume, ook als MHD **geen** direct "
        "causaal effect heeft op mortaliteit. Een goede predictor hoeft geen effectieve "
        "behandeldoelstelling te zijn."
    )

    st.markdown("### ROC-vergelijking")
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, name, f, t, a_val in [
        ("A", f"{fl[0]} only",         fpr_A, tpr_A, auc_A),
        ("B", f"{fl[1]} only",         fpr_B, tpr_B, auc_B),
        ("C", f"{fl[0]} + {fl[1]}",   fpr_C, tpr_C, auc_C),
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
        st.success(f"MHD heeft causaal effect op mortaliteit. "
                   f"Model B AUC = {auc_B:.3f}, C AUC = {auc_C:.3f}.")

    # ── fitted mortality functions ─────────────────────────────────────────────
    st.markdown("### Fitted mortaliteitsfuncties")
    tv_grid   = np.linspace(tumor_volume.min(), tumor_volume.max(), 200)
    mhd_grid  = np.linspace(mean_heart_dose.min(), mean_heart_dose.max(), 200)
    tv_in     = np.sqrt(tv_grid)  if use_sqrt else tv_grid
    mhd_in    = np.sqrt(mhd_grid) if use_sqrt else mhd_grid
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
        "Rode punten = gestorven (mortaliteit=1); groene punten = niet gestorven. "
        "In scenario A weerspiegelen de drie MHD-curves in plot C confounding, niet een direct effect."
    )

# ── tab 7: dose-effect curves (mortality only) ────────────────────────────────

with tab7:
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
        (mhd_p10_de,   f"MHD laag (p10) = {mhd_p10_de:.1f} Gy",   "#1a9641"),
        (mhd_med,      f"MHD mediaan = {mhd_med:.1f} Gy",           "#fdae61"),
        (mhd_p90_de,   f"MHD hoog (p90) = {mhd_p90_de:.1f} Gy",   "#d7191c"),
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
        (gtv_p10,  f"GTV laag (p10) = {gtv_p10:.0f} cc",   "#1a9641"),
        (gtv_med,  f"GTV mediaan = {gtv_med:.0f} cc",        "#fdae61"),
        (gtv_p90,  f"GTV hoog (p90) = {gtv_p90:.0f} cc",   "#d7191c"),
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

# ── tab 8: noise impact ───────────────────────────────────────────────────────

with tab8:
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

        # AUC vs noise plot
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

# ── tab 9: proton benefit ─────────────────────────────────────────────────────

with tab9:
    st.markdown(
        "Dit tabblad vergelijkt foton- en gesimuleerde protonenplannen door de MHD te verlagen. "
        "**De voorspelde absolute mortaliteitsreductie komt van het gefitte model.** "
        "Als MHD alleen correleert met GTV maar niet causaal is, kan het model "
        "mortaliteitsreductie voorspellen terwijl de werkelijke gesimuleerde uitkomst "
        "niet verbetert. Dit illustreert het verschil tussen model-gebaseerde "
        "voorspellingen en causale behandeleffecten."
    )

    pm1, pm2, pm3, pm4 = st.columns(4)
    pm1.metric("Gem. mortaliteit (foton)",                f"{p_mort_photon.mean()*100:.1f}%")
    pm2.metric("Gem. mortaliteit (proton)",               f"{p_mort_proton.mean()*100:.1f}%")
    pm3.metric("Gem. absolute mortaliteitsreductie (Δ)",  f"{delta_model.mean()*100:.2f}%")
    pm4.metric("Gem. NNT (model)",                        fmt_f(np.nanmean(nnt_model)))

    if true_delta is not None:
        pt1, pt2, *_ = st.columns(4)
        pt1.metric("Gem. werkelijke Δ (via causaal logit)", f"{true_delta.mean()*100:.2f}%")
        pt2.metric(
            "Overestimatie door confounding",
            f"{(delta_model.mean() - true_delta.mean())*100:.2f}%-punt",
            help="Positief = model voorspelt meer mortaliteitsreductie dan werkelijk optreedt.",
        )
        if not mhd_is_causal:
            st.warning(
                f"**Confounding in de mortaliteitsreductie:** het model voorspelt gemiddeld "
                f"{delta_model.mean()*100:.2f}% absolute mortaliteitsreductie, maar de werkelijke "
                f"gesimuleerde reductie is slechts {true_delta.mean()*100:.2f}%. "
                f"MHD is gecorreleerd met GTV maar heeft geen direct causaal effect."
            )

    st.markdown("---")
    st.markdown("#### Verdeling van voorspelde absolute mortaliteitsreductie (Δ) per patiënt")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(delta_model * 100, bins=40, color="#2980b9", edgecolor="none", alpha=0.8,
            label="Model-gebaseerde Δ")
    if true_delta is not None:
        ax.hist(true_delta * 100, bins=40, color="#e67e22", edgecolor="none",
                alpha=0.5, label="Werkelijke Δ (causaal logit)")
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
    ax.set_title(f"Patiënten per Δ-bin  (MHD reductiefactor = {mhd_reduction_factor})")
    ax.tick_params(axis="x", rotation=40, labelsize=7)
    fig.tight_layout(); centered(fig)

    if not mhd_is_causal:
        st.caption(
            "Scenario A: MHD heeft geen causale werking. Patiënten met hoge MHD hebben "
            "ook een grotere GTV — hun hogere 'voorspelde mortaliteitsreductie' is een "
            "artefact van confounding."
        )

# ── footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    f"Scenario {survival_scenario[:1]} · "
    "Causaal diagram (A): **GTV → mortaliteit** en **GTV → MHD**. "
    "In scenario A heeft MHD geen direct pad naar mortaliteit. "
    "In scenario B en C heeft MHD ook een causaal effect."
)
