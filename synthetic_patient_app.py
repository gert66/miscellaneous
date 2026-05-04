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
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc


# ── helpers ───────────────────────────────────────────────────────────────────

def wald_pvalues(model: LogisticRegression, X_scaled: np.ndarray) -> np.ndarray:
    """Compute Wald-test p-values for logistic regression coefficients."""
    p_hat = model.predict_proba(X_scaled)[:, 1]
    W = p_hat * (1 - p_hat)
    # Add intercept column
    X_aug = np.column_stack([np.ones(len(X_scaled)), X_scaled])
    H = X_aug.T @ (W[:, None] * X_aug)
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.full(X_scaled.shape[1], np.nan)
    se = np.sqrt(np.diag(cov)[1:])  # skip intercept SE
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
    n_patients = st.slider("Steekproefgrootte", 100, 3000, 500, step=50)
    seed = st.number_input("Random seed", value=42, step=1)

    st.subheader("Tumorvolume")
    tv_mu = st.slider("Gemiddelde tumorvolume (cc)", 10.0, 100.0, 45.0, step=1.0)
    tv_sigma = st.slider("Spreiding tumorvolume (cc)", 5.0, 40.0, 15.0, step=1.0)

    st.subheader("Mean Heart Dose = a·TV + ruis")
    corr_strength = st.slider(
        "Correlatiecoëfficiënt a (TV → MHD)", 0.0, 1.0, 0.5, step=0.05
    )
    mhd_noise_sd = st.slider("Ruisniveau MHD (Gy)", 0.1, 10.0, 3.0, step=0.1)

    st.subheader("Overlevingsmodel")
    b1 = st.slider(
        "b1 — effect tumorvolume op survival", -5.0, 0.0, -2.0, step=0.1
    )
    surv_noise_sd = st.slider("Ruisniveau overleving", 0.0, 3.0, 1.0, step=0.1)

    st.subheader("Model")
    test_size = st.slider("Testset aandeel", 0.1, 0.5, 0.2, step=0.05)

    st.divider()
    causal_mhd = st.toggle(
        "Echte causale werking van MHD inschakelen",
        value=False,
        help=(
            "Als aan: MHD krijgt een negatief causaal effect op overleving. "
            "Als uit: MHD is puur een confounder."
        ),
    )
    if causal_mhd:
        b2_causal = st.slider(
            "b2 — causaal effect MHD op survival", -3.0, 0.0, -1.0, step=0.1
        )
    else:
        b2_causal = 0.0

    st.divider()
    show_raw = st.checkbox("Toon ruwe data", value=False)

# ── data generation ───────────────────────────────────────────────────────────

rng = np.random.default_rng(int(seed))

# tumor_volume: clipped normal (realistic; all values > 0)
tumor_volume = rng.normal(loc=tv_mu, scale=tv_sigma, size=n_patients).clip(1.0, None)

# mean_heart_dose: linearly dependent on tumor_volume + independent noise
# Scale a so that the coefficient is per-unit of TV
mhd_base = 2.0 + corr_strength * (tumor_volume / tv_mu) * 10.0
mean_heart_dose = (mhd_base + rng.normal(0, mhd_noise_sd, n_patients)).clip(0.0, None)

# survival: ONLY depends on tumor_volume (+ optional causal MHD path)
tv_norm = (tumor_volume - tumor_volume.mean()) / tumor_volume.std()
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
        "tumor_volume_cc": tumor_volume.round(2),
        "mean_heart_dose_gy": mean_heart_dose.round(2),
        "survival": survival,
    }
)

# ── raw data ──────────────────────────────────────────────────────────────────

if show_raw:
    st.subheader("Ruwe data (eerste 50 rijen)")
    st.dataframe(df.head(50), use_container_width=True)

# ── summary metrics ───────────────────────────────────────────────────────────

pearson_r, pearson_p = stats.pearsonr(tumor_volume, mean_heart_dose)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Patiënten", n_patients)
col2.metric("Overlevingspercentage", f"{survival.mean()*100:.1f}%")
col3.metric("Pearson r (TV ↔ MHD)", f"{pearson_r:.3f}")
col4.metric("p-waarde correlatie", fmt_pval(pearson_p))

# ── modelling ─────────────────────────────────────────────────────────────────

X = df[["tumor_volume_cc", "mean_heart_dose_gy"]].values
y = df["survival"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

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

st.subheader("Logistische regressie — resultaten")

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

# Educational callout
if not causal_mhd and pvals[1] < 0.05:
    st.warning(
        f"**Confounding waarschuwing:** mean_heart_dose is statistisch significant "
        f"(p = {fmt_pval(pvals[1])}) terwijl het **geen causale invloed** heeft op "
        f"overleving. Dit ontstaat omdat MHD correleert met tumorvolume (r = {pearson_r:.2f}). "
        f"Vergroot de correlatie of steekproef om dit effect te versterken."
    )
elif not causal_mhd and pvals[1] >= 0.05:
    st.info(
        f"Mean_heart_dose is op dit moment **niet** significant (p = {fmt_pval(pvals[1])}). "
        f"Vergroot de correlatie (a) of steekproef om het confounding-effect te zien."
    )
elif causal_mhd:
    st.success(
        f"Causale modus actief: MHD heeft nu een echt effect (b2 = {b2_causal}). "
        f"Coëfficiënt MHD: {coefs[1]:.3f}, p = {fmt_pval(pvals[1])}."
    )

# ── visualisations ────────────────────────────────────────────────────────────

st.subheader("Visualisaties")
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Histogrammen", "Scatter TV ↔ MHD", "Correlatiemat.", "ROC-curve", "Verwarringsmatrix"]
)

COLORS = {0: "#e74c3c", 1: "#2ecc71"}
LABELS = {0: "Niet overleefd", 1: "Overleefd"}

with tab1:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, xlabel in zip(
        axes,
        ["tumor_volume_cc", "mean_heart_dose_gy"],
        ["Tumorvolume (cc)", "Mean Heart Dose (Gy)"],
    ):
        for val in [0, 1]:
            ax.hist(
                df.loc[df["survival"] == val, col],
                bins=30,
                alpha=0.6,
                color=COLORS[val],
                label=LABELS[val],
                edgecolor="none",
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Aantal patiënten")
        ax.legend()
    fig.suptitle("Verdeling per overlevingsstatus", fontsize=12)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter_colors = [COLORS[s] for s in survival]
    ax.scatter(
        tumor_volume, mean_heart_dose,
        c=scatter_colors, alpha=0.4, s=18, linewidths=0
    )
    # Regression line
    m, b_reg, *_ = stats.linregress(tumor_volume, mean_heart_dose)
    x_line = np.array([tumor_volume.min(), tumor_volume.max()])
    ax.plot(x_line, m * x_line + b_reg, color="#2c3e50", lw=1.5,
            label=f"y = {m:.2f}x + {b_reg:.1f}  (r = {pearson_r:.2f}, p = {fmt_pval(pearson_p)})")
    # Legend for survival colour
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=COLORS[0], alpha=0.7, label=LABELS[0]),
        Patch(color=COLORS[1], alpha=0.7, label=LABELS[1]),
    ]
    l1 = ax.legend(handles=legend_handles, loc="upper left", fontsize=8)
    ax.add_artist(l1)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("Tumorvolume (cc)")
    ax.set_ylabel("Mean Heart Dose (Gy)")
    ax.set_title("Scatter: TV vs MHD — kleur = overlevingsstatus")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "De regressielijn toont de lineaire afhankelijkheid MHD = a·TV + ruis. "
        "Doordat overleving afhankelijk is van TV, lijkt MHD ook te scheiden — dat is de confounder."
    )

with tab3:
    corr_df = df[["tumor_volume_cc", "mean_heart_dose_gy", "survival"]].corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    mask = np.zeros_like(corr_df, dtype=bool)
    np.fill_diagonal(mask, True)
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        vmin=-1, vmax=1,
        mask=mask,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Pearson correlatiemat.")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Merk op dat MHD correleert met zowel TV als survival, "
        "ook al is er geen causaal pad MHD → survival."
    )

with tab4:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2980b9", lw=2, label=f"AUC = {roc_auc_val:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Kans")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#2980b9")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-curve (testset)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab5:
    cm = confusion_matrix(y_test, y_pred_test)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Niet overleefd", "Overleefd"],
        yticklabels=["Niet overleefd", "Overleefd"],
        ax=ax,
    )
    ax.set_xlabel("Voorspeld")
    ax.set_ylabel("Werkelijk")
    ax.set_title("Verwarringsmatrix (testset)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Causaal diagram (standaard): **TV → survival** en **TV → MHD**. "
    "MHD heeft geen direct pad naar survival maar correleert erdoor met TV. "
    "Met de bonus-toggle: **MHD → survival** is ook actief."
)
