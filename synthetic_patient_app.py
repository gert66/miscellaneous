"""
Synthetic patient dataset — confounding demo.

Causal structure:
  Scenario A (default):  GTV ──► survival   GTV ──► MHD   (MHD is confounder only)
  Scenario B:            GTV ──► survival   MHD ──► survival   GTV ──► MHD
  Scenario C:            article-style logistic formula: logit = -1.3409 + 0.059·√GTV + 0.2635·√MHD
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
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

# Log-normal GTV parameters → median ≈ 69 cc, mean ≈ 110 cc, SD ≈ 136 cc
_GTV_MU    = np.log(69.0)
_GTV_SIGMA = np.sqrt(2 * (np.log(110.0) - np.log(69.0)))   # ≈ 0.965

# Gamma MHD parameters → mean ≈ 12 Gy, SD ≈ 8.2 Gy, median ≈ 11 Gy
_MHD_K     = (12.0 / 8.2) ** 2   # shape ≈ 2.14
_MHD_THETA = 8.2 ** 2 / 12.0     # scale ≈ 5.60


# ── helpers ───────────────────────────────────────────────────────────────────

def wald_pvalues(lr: LogisticRegression, X_scaled: np.ndarray) -> np.ndarray:
    """Wald-test p-values from the Hessian of the log-likelihood."""
    p = lr.predict_proba(X_scaled)[:, 1]
    W = p * (1 - p)
    X_aug = np.column_stack([np.ones(len(X_scaled)), X_scaled])
    H = X_aug.T @ (W[:, None] * X_aug)
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
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def sig_stars(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def make_pipe() -> Pipeline:
    return Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])


def centered(fig, use_container_width: bool = False) -> None:
    """Render figure centred inside a 1-3-1 column layout."""
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
        [DIST_SIMPLE, DIST_ARTICLE],
        help=(
            "**Eenvoudig:** normale verdelingen met instelbare gemiddelde en spreiding. "
            "**Artikel-achtig:** log-normale GTV en gamma-verdeelde MHD via een Gaussian copula, "
            "passend bij gepubliceerde longkankercijfers "
            "(GTV gem.≈110 cc, med.≈69 cc; MHD gem.≈12 Gy, med.≈11 Gy)."
        ),
    )

    st.subheader("Steekproef")
    n_patients = st.slider(
        "Steekproefgrootte", 100, 3000, 500, step=50,
        help=(
            "Aantal synthetische patiënten. Grotere steekproeven maken schattingen "
            "stabieler en maken zwakke confounding-effecten eerder statistisch significant."
        ),
    )
    seed = st.number_input(
        "Random seed", value=42, step=1,
        help="Startnummer voor de pseudo-random generator. Verander dit voor andere realisaties.",
    )

    # ── distribution-specific controls ───────────────────────────────────────
    if dist_mode == DIST_SIMPLE:
        st.subheader("Tumorvolume (GTV)")
        tv_mu = st.slider(
            "Gemiddelde GTV (cc)", 10.0, 200.0, 45.0, step=1.0,
            help="Gemiddelde van de normale verdeling voor tumorvolume. Beïnvloedt de ligging maar niet de correlatie- of overlevingsstructuur.",
        )
        tv_sigma = st.slider(
            "Spreiding GTV (cc)", 5.0, 60.0, 15.0, step=1.0,
            help="Standaarddeviatie van de GTV-verdeling. Grotere waarden geven meer variatie in GTV en daarmee ook in MHD.",
        )
        st.subheader("Mean Heart Dose = a·GTV + ruis")
        corr_a = st.slider(
            "Correlatiecoëfficiënt a (GTV → MHD)", 0.0, 1.0, 0.5, step=0.05,
            help="Bepaalt hoe sterk GTV de MHD aandrijft. Hogere waarden = sterkere confounding: MHD wordt een betere proxy voor GTV.",
        )
        mhd_noise_sd = st.slider(
            "Ruisniveau MHD (Gy)", 0.1, 10.0, 3.0, step=0.1,
            help="Willekeurige variatie in MHD niet verklaard door GTV. Hogere ruis verzwakt de GTV-MHD correlatie.",
        )
    else:
        st.subheader("Correlatie (Gaussian copula)")
        target_corr = st.slider(
            "Target correlatie GTV ↔ MHD", 0.0, 0.9, 0.45, step=0.05,
            help=(
                "Bepaalt hoe sterk GTV en MHD gecorreleerd zijn. De Gaussian copula gebruikt "
                "dit als correlatie van de latente normaalvariabelen; de werkelijke Pearson-r "
                "tussen GTV en MHD kan iets afwijken door de niet-lineaire transformatie."
            ),
        )

    # ── survival scenario ─────────────────────────────────────────────────────
    st.subheader("Overlevingsscenario")
    survival_scenario = st.selectbox(
        "Scenario",
        [SCEN_A, SCEN_B, SCEN_C],
        help=(
            "**A:** alleen GTV heeft causaal effect; MHD is puur een confounder. "
            "**B:** zowel GTV als MHD zijn causaal. "
            "**C:** artikel-formule met vaste coëfficiënten "
            "(aanbevolen samen met artikel-achtige verdelingen)."
        ),
    )
    if survival_scenario == SCEN_A:
        b1 = st.slider(
            "b1 — causaal effect GTV (logit, gestand.)", -5.0, 0.0, -2.0, step=0.1,
            help="Werkelijk causaal effect van GTV op overlevingskans in de simulatie. Negatiever = groter GTV → lagere overlevingskans.",
        )
        b2_val = 0.0
        surv_noise = st.slider(
            "Ruisniveau overleving", 0.0, 3.0, 1.0, step=0.1,
            help="Ruis op de logit van de overlevingskans. Hogere waarden maken de relatie GTV → overleving zwakker.",
        )
    elif survival_scenario == SCEN_B:
        b1 = st.slider(
            "b1 — causaal effect GTV (logit, gestand.)", -5.0, 0.0, -2.0, step=0.1,
            help="Werkelijk causaal effect van GTV op overlevingskans.",
        )
        b2_val = st.slider(
            "b2 — causaal effect MHD (logit, gestand.)", -3.0, 0.0, -1.0, step=0.1,
            help="Werkelijk causaal effect van MHD op overlevingskans. Dit is het echte causale pad MHD → overleving.",
        )
        surv_noise = st.slider(
            "Ruisniveau overleving", 0.0, 3.0, 1.0, step=0.1,
            help="Ruis op de logit.",
        )
    else:  # SCEN_C
        b1, b2_val, surv_noise = -2.0, 0.0, 0.0  # unused
        st.info(
            "Mortaliteitskans via artikel-formule:\n\n"
            "`logit(p) = −1.3409 + 0.0590·√GTV + 0.2635·√MHD`\n\n"
            "GTV én MHD hebben beiden een causaal effect in dit scenario."
        )

    mhd_is_causal = survival_scenario in (SCEN_B, SCEN_C)

    # ── model settings ────────────────────────────────────────────────────────
    st.subheader("Modelinstellingen")
    use_sqrt = st.checkbox(
        "Gebruik √-getransformeerde predictoren",
        value=(dist_mode == DIST_ARTICLE),
        key=f"use_sqrt_{dist_mode}",
        help=(
            "Als ingeschakeld: modellen worden gefit op √GTV en √MHD, "
            "vergelijkbaar met het gepubliceerde model. "
            "Aanbevolen in artikel-achtige modus. "
            "Beïnvloedt alleen het model, niet de datagen."
        ),
    )
    test_size = st.slider(
        "Testset aandeel", 0.1, 0.5, 0.2, step=0.05,
        help="Fractie van de data gereserveerd voor evaluatie. ROC AUC en nauwkeurigheid worden berekend op de testset.",
    )

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

else:  # article-like — Gaussian copula
    rho = float(target_corr)
    Z   = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], size=n_patients)
    eps = 1e-6
    U1  = np.clip(sp_norm.cdf(Z[:, 0]), eps, 1 - eps)
    U2  = np.clip(sp_norm.cdf(Z[:, 1]), eps, 1 - eps)
    tumor_volume    = lognorm.ppf(U1, s=_GTV_SIGMA, scale=np.exp(_GTV_MU)).clip(0.0, 1800.0)
    mean_heart_dose = sp_gamma.ppf(U2, a=_MHD_K, scale=_MHD_THETA).clip(0.0, 45.0)

# ── survival / mortality generation ──────────────────────────────────────────

if survival_scenario == SCEN_C:
    logit_mort  = -1.3409 + 0.0590 * np.sqrt(tumor_volume) + 0.2635 * np.sqrt(mean_heart_dose)
    p_mortality = 1.0 / (1.0 + np.exp(-logit_mort))
    survival    = rng.binomial(1, 1.0 - p_mortality).astype(int)
else:
    tv_norm  = (tumor_volume    - tumor_volume.mean())    / tumor_volume.std()
    mhd_norm = (mean_heart_dose - mean_heart_dose.mean()) / mean_heart_dose.std()
    logit_p  = (b1 * tv_norm + b2_val * mhd_norm
                + surv_noise * rng.standard_normal(n_patients))
    survival = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit_p))).astype(int)

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

with st.expander("Validatietabel — beschrijvende statistieken", expanded=(dist_mode == DIST_ARTICLE)):
    vcol1, vcol2 = st.columns(2)
    with vcol1:
        st.dataframe(
            pd.DataFrame({
                "GTV (cc)": describe_arr(tumor_volume),
                "MHD (Gy)": describe_arr(mean_heart_dose),
            }),
            use_container_width=True,
        )
    with vcol2:
        st.metric("Pearson r (GTV ↔ MHD)", f"{pearson_r:.3f}")
        st.metric("p-waarde correlatie",    fmt_pval(pearson_p))
        st.metric("24-maands mortaliteit",  f"{mortality_rate*100:.1f}%")
        if dist_mode == DIST_ARTICLE:
            st.caption(
                "Referentie (artikel): GTV gem.≈110 cc, med.≈69 cc, SD≈130 cc; "
                "MHD gem.≈12 Gy, med.≈11 Gy, SD≈8.2 Gy."
            )

# ── raw data ──────────────────────────────────────────────────────────────────

if show_raw:
    st.subheader("Ruwe data (eerste 50 rijen)")
    st.dataframe(df.head(50), use_container_width=True)

# ── summary metrics ───────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
c1.metric("Patiënten",            n_patients)
c2.metric("Overlevingspercentage", f"{survival.mean()*100:.1f}%")
c3.metric("Pearson r (GTV ↔ MHD)", f"{pearson_r:.3f}")
c4.metric("p-waarde correlatie",   fmt_pval(pearson_p))

# ── feature preparation ───────────────────────────────────────────────────────

X_raw = df[["tumor_volume_cc", "mean_heart_dose_gy"]].values
if use_sqrt:
    X_feat  = np.sqrt(X_raw)
    fl = ["√GTV", "√MHD"]
else:
    X_feat  = X_raw.copy()
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
y_prob_C   = pipe_C.predict_proba(X_test)[:, 1]
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
        f"Predictoren: {', '.join(fl)} (StandardScaler toegepast). "
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

# ── tabs ──────────────────────────────────────────────────────────────────────

st.subheader("Visualisaties")

SC = {0: "#e74c3c", 1: "#2ecc71"}   # survival colours
SL = {0: "Niet overleefd", 1: "Overleefd"}
MS = {"A": ("#2980b9", "solid"), "B": ("#e67e22", "dashed"), "C": ("#27ae60", "dashdot")}

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Histogrammen", "Scatter GTV ↔ MHD", "Correlatiemat.",
    "ROC-curve", "Verwarringsmatrix", "ROC vergelijking",
])

# ── tab 1: histograms ─────────────────────────────────────────────────────────

with tab1:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, col, xlabel in zip(axes,
        ["tumor_volume_cc", "mean_heart_dose_gy"],
        ["GTV (cc)", "Mean Heart Dose (Gy)"]):
        for val in [0, 1]:
            ax.hist(df.loc[df["survival"] == val, col],
                    bins=30, alpha=0.6, color=SC[val], label=SL[val], edgecolor="none")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Aantal patiënten")
        ax.legend()
    fig.suptitle("Verdeling per overlevingsstatus", fontsize=12)
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
    st.caption(
        "De regressielijn toont de lineaire afhankelijkheid GTV → MHD. "
        "Doordat overleving afhangt van GTV, lijkt MHD ook te scheiden — dit is de confounder."
    )

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
            label=f"GTV + MHD  (AUC = {auc_C:.3f})")
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
    ax.set_xlabel("Voorspeld")
    ax.set_ylabel("Werkelijk")
    ax.set_title("Verwarringsmatrix (testset)")
    fig.tight_layout()
    centered(fig)

# ── tab 6: ROC comparison + fitted survival functions ─────────────────────────

with tab6:

    st.markdown(
        "> **Predictie vs. causaliteit** — "
        "Het MHD-only model kan voorspellende waarde vertonen wanneer MHD "
        "correleert met tumorvolume, ook als MHD in de simulatie **geen** "
        "direct causaal effect op overleving heeft. "
        "Dit illustreert het verschil tussen *voorspelling* en "
        "*causale behandeleffecten*: een goede predictor hoeft geen "
        "effectieve behandeldoelstelling te zijn."
    )

    # ── ROC comparison ────────────────────────────────────────────────────────
    st.markdown("### ROC-vergelijking")

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, name, f, t, a_val in [
        ("A", f"{fl[0]} only",         fpr_A, tpr_A, auc_A),
        ("B", f"{fl[1]} only",         fpr_B, tpr_B, auc_B),
        ("C", f"{fl[0]} + {fl[1]}",   fpr_C, tpr_C, auc_C),
    ]:
        color, ls = MS[key]
        ax.plot(f, t, color=color, lw=2, linestyle=ls,
                label=f"{name},  AUC = {a_val:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", lw=1,
            label="Toeval  (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-vergelijking — drie modellen (testset)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    centered(fig)

    # ── model summary table ───────────────────────────────────────────────────
    st.markdown("**Samenvatting per model**")
    st.dataframe(roc_summary, use_container_width=True, hide_index=True)

    if not mhd_is_causal and auc_B > 0.55:
        st.warning(
            f"**Confounding zichtbaar:** Model B ({fl[1]} only, AUC = {auc_B:.3f}) "
            f"presteert beter dan toeval terwijl MHD **geen causale werking** heeft. "
            f"Dit komt doordat MHD correleert met GTV (r = {pearson_r:.2f}). "
            f"Model A ({fl[0]} only) heeft AUC = {auc_A:.3f}."
        )
    elif not mhd_is_causal:
        st.info(
            f"Model B ({fl[1]} only, AUC = {auc_B:.3f}) presteert nauwelijks beter dan toeval. "
            f"Vergroot de correlatie of steekproef om confounding-predictie zichtbaar te maken."
        )
    else:
        st.success(
            f"MHD heeft een causaal effect in dit scenario. "
            f"Model B AUC = {auc_B:.3f},  Model C AUC = {auc_C:.3f}."
        )

    # ── fitted survival functions ─────────────────────────────────────────────
    st.markdown("### Fitted survival functions")

    tv_grid  = np.linspace(tumor_volume.min(),    tumor_volume.max(),    200)
    mhd_grid = np.linspace(mean_heart_dose.min(), mean_heart_dose.max(), 200)

    # Pipeline input: optionally sqrt-transformed
    tv_in  = np.sqrt(tv_grid)  if use_sqrt else tv_grid
    mhd_in = np.sqrt(mhd_grid) if use_sqrt else mhd_grid

    mhd_p10 = np.percentile(mean_heart_dose, 10)
    mhd_p50 = np.percentile(mean_heart_dose, 50)
    mhd_p90 = np.percentile(mean_heart_dose, 90)

    sv1, sv2, sv3 = st.columns(3)

    with sv1:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(tumor_volume, survival, c=[SC[s] for s in survival],
                   alpha=0.25, s=10, linewidths=0, zorder=1)
        ax.plot(tv_grid, pipe_A.predict_proba(tv_in.reshape(-1, 1))[:, 1],
                color=MS["A"][0], lw=2, zorder=2)
        ax.set_xlabel("GTV (cc)")
        ax.set_ylabel("Voorspelde overlevingskans")
        ax.set_title(f"Model A — {fl[0]} only")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with sv2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(mean_heart_dose, survival, c=[SC[s] for s in survival],
                   alpha=0.25, s=10, linewidths=0, zorder=1)
        ax.plot(mhd_grid, pipe_B.predict_proba(mhd_in.reshape(-1, 1))[:, 1],
                color=MS["B"][0], lw=2, zorder=2)
        ax.set_xlabel("MHD (Gy)")
        ax.set_ylabel("Voorspelde overlevingskans")
        ax.set_title(f"Model B — {fl[1]} only")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with sv3:
        fig, ax = plt.subplots(figsize=(5, 4))
        for mhd_val, label, color in [
            (mhd_p10, f"MHD laag (p10) = {mhd_p10:.1f} Gy",    "#1a9641"),
            (mhd_p50, f"MHD mediaan   = {mhd_p50:.1f} Gy",      "#fdae61"),
            (mhd_p90, f"MHD hoog (p90) = {mhd_p90:.1f} Gy",     "#d7191c"),
        ]:
            if use_sqrt:
                X_c = np.column_stack([np.sqrt(tv_grid), np.full_like(tv_grid, np.sqrt(mhd_val))])
            else:
                X_c = np.column_stack([tv_grid, np.full_like(tv_grid, mhd_val)])
            ax.plot(tv_grid, pipe_C.predict_proba(X_c)[:, 1],
                    color=color, lw=2, label=label)
        ax.set_xlabel("GTV (cc)")
        ax.set_ylabel("Voorspelde overlevingskans")
        ax.set_title(f"Model C — {fl[0]} + {fl[1]}")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=6.5, loc="upper right")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.caption(
        "Model C toont drie curves voor lage (p10), gemiddelde (p50) en hoge (p90) MHD-waarden. "
        "In scenario A (MHD = confounder) weerspiegelen afwijkende curves confounding: "
        "hogere MHD hangt samen met hoger GTV — niet met een direct effect op overleving."
    )

# ── footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    f"Scenario {survival_scenario[:1]} · "
    "Causaal diagram (standaard A): **GTV → overleving** en **GTV → MHD**. "
    "MHD heeft in scenario A geen direct pad naar overleving. "
    "In scenario B en C heeft MHD ook een causaal effect."
)
