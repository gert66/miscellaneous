"""
Synthetic patient dataset — confounding demo.

Causal structure (default):
    tumor_volume  ──►  survival
    tumor_volume  ──►  mean_heart_dose   (confound, NOT causal for survival)

With the bonus toggle enabled:
    tumor_volume  ──►  survival
    mean_heart_dose ──► survival         (additional causal path)
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc


# ── helpers ───────────────────────────────────────────────────────────────────

def wald_pvalues(model: LogisticRegression, X_scaled: np.ndarray) -> np.ndarray:
    p_hat = model.predict_proba(X_scaled)[:, 1]
    W = p_hat * (1 - p_hat)
    X_aug = np.column_stack([np.ones(len(X_scaled)), X_scaled])
    H = X_aug.T @ (W[:, None] * X_aug)
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.full(X_scaled.shape[1], np.nan)
    se = np.sqrt(np.diag(cov)[1:])
    z = model.coef_[0] / se
    return 2 * (1 - stats.norm.cdf(np.abs(z)))


def fmt_pval(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def sig_stars(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def centered(fig, use_container_width: bool = False):
    """Render a matplotlib figure centred inside a 1-3-1 column layout."""
    _, mid, _ = st.columns([1, 3, 1])
    with mid:
        st.pyplot(fig, use_container_width=use_container_width)
    plt.close(fig)


def make_lr_pipeline() -> Pipeline:
    return Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Confounding Demo", layout="wide")
st.title("Confounding in klinische data")
st.markdown(
    "**Doel:** laat zien dat een variabele zonder causale werking "
    "(_mean heart dose_) toch statistisch significant lijkt doordat ze "
    "correleert met de echte oorzaak (_tumorvolume_)."
)

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Data-generatie")
    n_patients = st.slider(
        "Steekproefgrootte", 100, 3000, 500, step=50,
        help=(
            "Aantal synthetische patiënten. Grotere steekproeven maken "
            "schattingen stabieler en maken zwakke confounding-effecten "
            "eerder statistisch significant."
        ),
    )
    seed = st.number_input(
        "Random seed", value=42, step=1,
        help=(
            "Startnummer voor de pseudo-random generator. Verander dit om "
            "andere realisaties van dezelfde verdeling te zien zonder de "
            "parameters te wijzigen."
        ),
    )

    st.subheader("Tumorvolume")
    tv_mu = st.slider(
        "Gemiddelde tumorvolume (cc)", 10.0, 100.0, 45.0, step=1.0,
        help=(
            "Gemiddelde van de normale verdeling waaruit tumorvolumes worden "
            "getrokken. Beïnvloedt de ligging van de data maar niet de "
            "correlatie- of overlevingsstructuur."
        ),
    )
    tv_sigma = st.slider(
        "Spreiding tumorvolume (cc)", 5.0, 40.0, 15.0, step=1.0,
        help=(
            "Standaarddeviatie van de tumorvolumeverdeling. Grotere waarden "
            "geven meer variatie in TV en daarmee ook meer variatie in MHD "
            "(omdat MHD = a·TV + ruis)."
        ),
    )

    st.subheader("Mean Heart Dose = a·TV + ruis")
    corr_strength = st.slider(
        "Correlatiecoëfficiënt a (TV → MHD)", 0.0, 1.0, 0.5, step=0.05,
        help=(
            "Bepaalt hoe sterk tumorvolume de mean heart dose aandrijft. "
            "Hogere waarden creëren sterkere confounding: MHD wordt een "
            "betere proxy voor TV en krijgt daardoor meer schijnbaar "
            "voorspellend vermogen voor overleving."
        ),
    )
    mhd_noise_sd = st.slider(
        "Ruisniveau MHD (Gy)", 0.1, 10.0, 3.0, step=0.1,
        help=(
            "Willekeurige variatie in MHD die niet verklaard wordt door TV. "
            "Hogere ruis verzwakt de TV-MHD correlatie en vermindert daarmee "
            "het confounding-effect."
        ),
    )

    st.subheader("Overlevingsmodel")
    b1 = st.slider(
        "b1 — effect tumorvolume op survival", -5.0, 0.0, -2.0, step=0.1,
        help=(
            "Werkelijk causaal effect van tumorvolume op overleving in de "
            "simulatie (logit-schaal, gestandaardiseerde TV). Negatiever = "
            "groter tumorvolume → lagere overlevingskans. Dit is het enige "
            "variabele dat de ware datagen beïnvloedt."
        ),
    )
    surv_noise_sd = st.slider(
        "Ruisniveau overleving", 0.0, 3.0, 1.0, step=0.1,
        help=(
            "Willekeurig ruis op de logit van de overlevingskans. Hogere "
            "waarden maken de relatie TV → survival zwakker, waardoor alle "
            "modellen een lagere AUC krijgen."
        ),
    )

    st.subheader("Model")
    test_size = st.slider(
        "Testset aandeel", 0.1, 0.5, 0.2, step=0.05,
        help=(
            "Fractie van de data gereserveerd voor evaluatie. De ROC-curves "
            "en AUC-waarden worden berekend op de testset; de modellen worden "
            "getraind op de resterende trainset."
        ),
    )

    st.divider()
    causal_mhd = st.toggle(
        "Echte causale werking van MHD inschakelen",
        value=False,
        help=(
            "Uit (standaard): MHD heeft geen direct causaal effect op "
            "overleving — het is puur een confounder. "
            "Aan: MHD wordt toegevoegd aan de overlevingsvergelijking met "
            "het hieronder in te stellen effect b2."
        ),
    )
    if causal_mhd:
        b2_causal = st.slider(
            "b2 — causaal effect MHD op survival", -3.0, 0.0, -1.0, step=0.1,
            help=(
                "Werkelijk causaal effect van MHD op overleving wanneer de "
                "causale modus actief is (logit-schaal, gestandaardiseerde MHD)."
            ),
        )
    else:
        b2_causal = 0.0

    st.divider()
    show_raw = st.checkbox(
        "Toon ruwe data", value=False,
        help="Toon de eerste 50 rijen van de gegenereerde dataset.",
    )

# ── data generation ───────────────────────────────────────────────────────────

rng = np.random.default_rng(int(seed))

tumor_volume = rng.normal(loc=tv_mu, scale=tv_sigma, size=n_patients).clip(1.0, None)

mhd_base = 2.0 + corr_strength * (tumor_volume / tv_mu) * 10.0
mean_heart_dose = (mhd_base + rng.normal(0, mhd_noise_sd, n_patients)).clip(0.0, None)

tv_norm  = (tumor_volume   - tumor_volume.mean())   / tumor_volume.std()
mhd_norm = (mean_heart_dose - mean_heart_dose.mean()) / mean_heart_dose.std()

logit_p = (
    b1 * tv_norm
    + b2_causal * mhd_norm
    + surv_noise_sd * rng.standard_normal(n_patients)
)
prob_survival = 1.0 / (1.0 + np.exp(-logit_p))
survival = rng.binomial(1, prob_survival).astype(int)

df = pd.DataFrame(
    {
        "tumor_volume_cc":    tumor_volume.round(2),
        "mean_heart_dose_gy": mean_heart_dose.round(2),
        "survival":           survival,
    }
)

# ── raw data ──────────────────────────────────────────────────────────────────

if show_raw:
    st.subheader("Ruwe data (eerste 50 rijen)")
    st.dataframe(df.head(50), use_container_width=True)

# ── summary metrics ───────────────────────────────────────────────────────────

pearson_r, pearson_p = stats.pearsonr(tumor_volume, mean_heart_dose)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Patiënten", n_patients)
c2.metric("Overlevingspercentage", f"{survival.mean()*100:.1f}%")
c3.metric("Pearson r (TV ↔ MHD)", f"{pearson_r:.3f}")
c4.metric("p-waarde correlatie", fmt_pval(pearson_p))

# ── modelling (combined model used for existing tabs) ─────────────────────────

X = df[["tumor_volume_cc", "mean_heart_dose_gy"]].values
y = df["survival"].values

scaler     = StandardScaler()
X_scaled   = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size, random_state=int(seed)
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

pvals = wald_pvalues(model, X_scaled)
coefs = model.coef_[0]

coef_df = pd.DataFrame(
    {
        "Feature": ["tumor_volume_cc", "mean_heart_dose_gy"],
        "Coëfficiënt (std.)": coefs.round(4),
        "p-waarde": [fmt_pval(p) for p in pvals],
        "Significantie": [sig_stars(p) for p in pvals],
        "Causaal effect?": [
            "Ja",
            "Ja (bonus)" if causal_mhd else "Nee — confounder",
        ],
    }
)

y_prob_test = model.predict_proba(X_test)[:, 1]
y_pred_test = model.predict(X_test)
fpr, tpr, _ = roc_curve(y_test, y_prob_test)
roc_auc_val = auc(fpr, tpr)

# ── three pipeline models for ROC comparison ──────────────────────────────────

# Raw split with identical random_state → same train/test rows as above
X_raw_train, X_raw_test, _, _ = train_test_split(
    X, y, test_size=test_size, random_state=int(seed)
)

pipe_A = make_lr_pipeline()
pipe_B = make_lr_pipeline()
pipe_C = make_lr_pipeline()

pipe_A.fit(X_raw_train[:, [0]], y_train)  # GTV only
pipe_B.fit(X_raw_train[:, [1]], y_train)  # MHD only
pipe_C.fit(X_raw_train,         y_train)  # GTV + MHD


def _roc(pipe, Xte, cols=None):
    Xi = Xte[:, cols] if cols is not None else Xte
    prob = pipe.predict_proba(Xi)[:, 1]
    f, t, _ = roc_curve(y_test, prob)
    return f, t, auc(f, t)


fpr_A, tpr_A, auc_A = _roc(pipe_A, X_raw_test, cols=[0])
fpr_B, tpr_B, auc_B = _roc(pipe_B, X_raw_test, cols=[1])
fpr_C, tpr_C, auc_C = _roc(pipe_C, X_raw_test)

_lr_step  = lambda p: p.named_steps["lr"]
_coef_str = lambda p, labels: ", ".join(
    f"{l}: {c:.3f}" for l, c in zip(labels, _lr_step(p).coef_[0])
)

roc_summary = pd.DataFrame(
    {
        "Model":          ["A — GTV only", "B — MHD only", "C — GTV + MHD"],
        "Predictoren":    ["tumor_volume", "mean_heart_dose", "tumor_volume + mean_heart_dose"],
        "AUC (testset)":  [f"{auc_A:.3f}", f"{auc_B:.3f}", f"{auc_C:.3f}"],
        "Intercept":      [
            f"{_lr_step(pipe_A).intercept_[0]:.3f}",
            f"{_lr_step(pipe_B).intercept_[0]:.3f}",
            f"{_lr_step(pipe_C).intercept_[0]:.3f}",
        ],
        "Coëfficiënten (std.)": [
            _coef_str(pipe_A, ["GTV"]),
            _coef_str(pipe_B, ["MHD"]),
            _coef_str(pipe_C, ["GTV", "MHD"]),
        ],
    }
)

# ── regression results section ────────────────────────────────────────────────

st.subheader("Logistische regressie — resultaten (gecombineerd model)")

res_col1, res_col2 = st.columns([3, 1])
with res_col1:
    st.dataframe(coef_df, use_container_width=True, hide_index=True)
    st.caption(
        "Coëfficiënten zijn gestandaardiseerd (StandardScaler). "
        "p-waarden via Wald-test.  *** p<0.001  ** p<0.01  * p<0.05  ns p≥0.05"
    )
with res_col2:
    st.metric("ROC AUC (testset)", f"{roc_auc_val:.3f}")
    accuracy = (y_pred_test == y_test).mean()
    st.metric("Nauwkeurigheid (testset)", f"{accuracy*100:.1f}%")

if not causal_mhd and pvals[1] < 0.05:
    st.warning(
        f"**Confounding waarschuwing:** mean_heart_dose is statistisch significant "
        f"(p = {fmt_pval(pvals[1])}) terwijl het **geen causale invloed** heeft. "
        f"Dit ontstaat doordat MHD correleert met TV (r = {pearson_r:.2f})."
    )
elif not causal_mhd and pvals[1] >= 0.05:
    st.info(
        f"Mean_heart_dose is niet significant (p = {fmt_pval(pvals[1])}). "
        f"Vergroot de correlatie (a) of steekproef om confounding zichtbaar te maken."
    )
else:
    st.success(
        f"Causale modus actief: MHD heeft een echt effect (b2 = {b2_causal}). "
        f"Coëfficiënt MHD: {coefs[1]:.3f}, p = {fmt_pval(pvals[1])}."
    )

# ── tabs ──────────────────────────────────────────────────────────────────────

st.subheader("Visualisaties")

SURV_COLORS = {0: "#e74c3c", 1: "#2ecc71"}
SURV_LABELS  = {0: "Niet overleefd", 1: "Overleefd"}
MODEL_STYLE  = {
    "A": ("#2980b9", "solid"),
    "B": ("#e67e22", "dashed"),
    "C": ("#27ae60", "dashdot"),
}

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Histogrammen", "Scatter TV ↔ MHD", "Correlatiemat.",
     "ROC-curve", "Verwarringsmatrix", "ROC vergelijking"]
)

# ── tab 1: histograms ─────────────────────────────────────────────────────────

with tab1:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, col, xlabel in zip(
        axes,
        ["tumor_volume_cc", "mean_heart_dose_gy"],
        ["Tumorvolume (cc)", "Mean Heart Dose (Gy)"],
    ):
        for val in [0, 1]:
            ax.hist(
                df.loc[df["survival"] == val, col],
                bins=30, alpha=0.6,
                color=SURV_COLORS[val], label=SURV_LABELS[val], edgecolor="none",
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Aantal patiënten")
        ax.legend()
    fig.suptitle("Verdeling per overlevingsstatus", fontsize=12)
    fig.tight_layout()
    centered(fig)

# ── tab 2: scatter TV ↔ MHD ──────────────────────────────────────────────────

with tab2:
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter_colors = [SURV_COLORS[s] for s in survival]
    ax.scatter(tumor_volume, mean_heart_dose, c=scatter_colors, alpha=0.4, s=18, linewidths=0)
    m_reg, b_reg, *_ = stats.linregress(tumor_volume, mean_heart_dose)
    x_line = np.array([tumor_volume.min(), tumor_volume.max()])
    ax.plot(
        x_line, m_reg * x_line + b_reg, color="#2c3e50", lw=1.5,
        label=f"y = {m_reg:.2f}x + {b_reg:.1f}  (r = {pearson_r:.2f}, p = {fmt_pval(pearson_p)})",
    )
    surv_handles = [Patch(color=SURV_COLORS[v], alpha=0.7, label=SURV_LABELS[v]) for v in [0, 1]]
    l1 = ax.legend(handles=surv_handles, loc="upper left", fontsize=8)
    ax.add_artist(l1)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("Tumorvolume (cc)")
    ax.set_ylabel("Mean Heart Dose (Gy)")
    ax.set_title("Scatter: TV vs MHD — kleur = overlevingsstatus")
    fig.tight_layout()
    centered(fig)
    st.caption(
        "De regressielijn toont MHD = a·TV + ruis. "
        "Doordat overleving afhangt van TV, lijkt MHD ook te scheiden — dit is de confounder."
    )

# ── tab 3: correlation matrix ─────────────────────────────────────────────────

with tab3:
    corr_df = df[["tumor_volume_cc", "mean_heart_dose_gy", "survival"]].corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    mask = np.eye(len(corr_df), dtype=bool)
    sns.heatmap(
        corr_df, annot=True, fmt=".3f", cmap="RdYlGn",
        center=0, vmin=-1, vmax=1, mask=mask, ax=ax, linewidths=0.5,
    )
    ax.set_title("Pearson correlatiemat.")
    fig.tight_layout()
    centered(fig)
    st.caption(
        "MHD correleert met zowel TV als survival, ook al is er geen causaal pad MHD → survival."
    )

# ── tab 4: single ROC (combined model) ────────────────────────────────────────

with tab4:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#2980b9", lw=2, label=f"GTV + MHD  (AUC = {roc_auc_val:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Toeval")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#2980b9")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-curve — gecombineerd model (testset)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    centered(fig)

# ── tab 5: confusion matrix ───────────────────────────────────────────────────

with tab5:
    cm = confusion_matrix(y_test, y_pred_test)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Niet overleefd", "Overleefd"],
        yticklabels=["Niet overleefd", "Overleefd"],
        ax=ax,
    )
    ax.set_xlabel("Voorspeld")
    ax.set_ylabel("Werkelijk")
    ax.set_title("Verwarringsmatrix (testset)")
    fig.tight_layout()
    centered(fig)

# ── tab 6: ROC comparison + fitted survival functions ─────────────────────────

with tab6:

    # ── explanatory text ─────────────────────────────────────────────────────
    st.markdown(
        "> **Predictie vs. causaliteit** — "
        "Het MHD-only model kan voorspellende waarde vertonen wanneer MHD "
        "correleert met tumorvolume, ook als MHD in de simulatie **geen** "
        "direct causaal effect op overleving heeft. "
        "Dit illustreert het verschil tussen *voorspelling* en "
        "*causale behandeleffecten*: een goede predictor hoeft geen "
        "effectieve behandeldoelstelling te zijn."
    )

    # ── ROC comparison plot ───────────────────────────────────────────────────
    st.markdown("### ROC-vergelijking")

    fig, ax = plt.subplots(figsize=(8, 5))
    roc_data = [
        ("A", "GTV only",     fpr_A, tpr_A, auc_A),
        ("B", "MHD only",     fpr_B, tpr_B, auc_B),
        ("C", "GTV + MHD",    fpr_C, tpr_C, auc_C),
    ]
    for key, name, f, t, a_val in roc_data:
        color, ls = MODEL_STYLE[key]
        ax.plot(f, t, color=color, lw=2, linestyle=ls,
                label=f"{name},  AUC = {a_val:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", lw=1, label="Toeval  (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-vergelijking — drie logistische regressiemodellen (testset)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    centered(fig)

    # ── model summary table ───────────────────────────────────────────────────
    st.markdown("**Samenvatting per model**")
    st.dataframe(roc_summary, use_container_width=True, hide_index=True)

    # context-aware callout
    if not causal_mhd and auc_B > 0.55:
        st.warning(
            f"**Confounding zichtbaar:** Model B (_MHD only_, AUC = {auc_B:.3f}) presteert "
            f"beter dan toeval, ook al heeft MHD **geen causale werking**. "
            f"Dit komt puur doordat MHD correleert met TV (r = {pearson_r:.2f}). "
            f"Model A (GTV only) heeft AUC = {auc_A:.3f}; het AUC-verschil met C is klein "
            f"omdat MHD na correctie voor TV weinig extra informatie toevoegt."
        )
    elif not causal_mhd:
        st.info(
            f"Model B (MHD only, AUC = {auc_B:.3f}) presteert nauwelijks beter dan toeval. "
            f"Vergroot de correlatie (a) of steekproef om confounding-predictie zichtbaar te maken."
        )
    else:
        st.success(
            f"Causale modus actief: MHD heeft nu een echt effect (b2 = {b2_causal}). "
            f"Model B AUC = {auc_B:.3f}, Model C AUC = {auc_C:.3f}."
        )

    # ── fitted survival functions ─────────────────────────────────────────────
    st.markdown("### Fitted survival functions")

    tv_grid  = np.linspace(tumor_volume.min(),    tumor_volume.max(),    200)
    mhd_grid = np.linspace(mean_heart_dose.min(), mean_heart_dose.max(), 200)

    # Percentiles for low / medium / high MHD in the combined-model plot
    mhd_low  = np.percentile(mean_heart_dose, 10)
    mhd_med  = np.percentile(mean_heart_dose, 50)
    mhd_high = np.percentile(mean_heart_dose, 90)

    sv_col1, sv_col2, sv_col3 = st.columns(3)

    # A — GTV only
    with sv_col1:
        prob_A = pipe_A.predict_proba(tv_grid.reshape(-1, 1))[:, 1]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(tumor_volume, survival, c=[SURV_COLORS[s] for s in survival],
                   alpha=0.25, s=10, linewidths=0, zorder=1)
        ax.plot(tv_grid, prob_A, color=MODEL_STYLE["A"][0], lw=2, zorder=2)
        ax.set_xlabel("Tumorvolume (cc)")
        ax.set_ylabel("Voorspelde overlevingskans")
        ax.set_title("Model A — GTV only")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # B — MHD only
    with sv_col2:
        prob_B = pipe_B.predict_proba(mhd_grid.reshape(-1, 1))[:, 1]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(mean_heart_dose, survival, c=[SURV_COLORS[s] for s in survival],
                   alpha=0.25, s=10, linewidths=0, zorder=1)
        ax.plot(mhd_grid, prob_B, color=MODEL_STYLE["B"][0], lw=2, zorder=2)
        ax.set_xlabel("Mean Heart Dose (Gy)")
        ax.set_ylabel("Voorspelde overlevingskans")
        ax.set_title("Model B — MHD only")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # C — GTV + MHD: three curves for low / med / high MHD
    with sv_col3:
        fig, ax = plt.subplots(figsize=(5, 4))
        mhd_levels = [
            (mhd_low,  "MHD laag (p10)",  "#1a9641"),
            (mhd_med,  "MHD mediaan",      "#fdae61"),
            (mhd_high, "MHD hoog (p90)",   "#d7191c"),
        ]
        for mhd_val, label, color in mhd_levels:
            X_curve = np.column_stack([tv_grid, np.full_like(tv_grid, mhd_val)])
            prob_curve = pipe_C.predict_proba(X_curve)[:, 1]
            ax.plot(tv_grid, prob_curve, color=color, lw=2, label=f"{label} = {mhd_val:.1f} Gy")
        ax.set_xlabel("Tumorvolume (cc)")
        ax.set_ylabel("Voorspelde overlevingskans")
        ax.set_title("Model C — GTV + MHD")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.caption(
        "Model C toont drie curves voor lage, gemiddelde en hoge MHD-waarden over het "
        "tumorvolumebereik. Als MHD puur een confounder is (toggle uit), lopen de curves "
        "uiteen doordat hoge MHD gepaard gaat met groter TV — niet door een causaal effect."
    )

# ── footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Causaal diagram (standaard): **TV → survival** en **TV → MHD**. "
    "MHD heeft geen direct pad naar survival maar correleert erdoor met TV. "
    "Met de bonus-toggle: **MHD → survival** is ook actief."
)
