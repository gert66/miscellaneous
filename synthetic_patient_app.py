import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc

st.set_page_config(page_title="Synthetische Patiëntdata", layout="wide")
st.title("Synthetische Patiëntdata & Logistische Regressie")
st.markdown(
    "Tumorvolume correleert met overleving. Mean heart dose heeft **geen** echte invloed."
)

# --- Sidebar parameters ---
with st.sidebar:
    st.header("Parameters")
    n_patients = st.slider("Aantal patiënten", 50, 2000, 300, step=50)
    correlation_strength = st.slider(
        "Correlatiesterkte tumorvolume ↔ overleving", 0.5, 5.0, 2.0, step=0.1
    )
    noise_level = st.slider("Ruisniveau", 0.1, 3.0, 1.0, step=0.1)
    seed = st.number_input("Random seed", value=42, step=1)
    test_size = st.slider("Testset aandeel", 0.1, 0.5, 0.2, step=0.05)
    st.divider()
    show_raw = st.checkbox("Toon ruwe data", value=False)

rng = np.random.default_rng(int(seed))

# --- Data generatie ---
tumor_volume = rng.normal(loc=50, scale=20, size=n_patients).clip(5, 120)
mean_heart_dose = rng.normal(loc=10, scale=5, size=n_patients).clip(0, 30)

# Logit: alleen tumorvolume heeft echte invloed
tumor_norm = (tumor_volume - tumor_volume.mean()) / tumor_volume.std()
logit = -correlation_strength * tumor_norm + noise_level * rng.standard_normal(n_patients)
prob_survival = 1 / (1 + np.exp(-logit))
survival = rng.binomial(1, prob_survival).astype(int)

df = pd.DataFrame(
    {
        "tumor_volume_cc": tumor_volume.round(2),
        "mean_heart_dose_gy": mean_heart_dose.round(2),
        "survival": survival,
    }
)

if show_raw:
    st.subheader("Ruwe data (eerste 50 rijen)")
    st.dataframe(df.head(50), use_container_width=True)

st.subheader("Beschrijvende statistieken")
col1, col2, col3 = st.columns(3)
col1.metric("Totaal patiënten", n_patients)
col2.metric("Overlevers (%)", f"{survival.mean()*100:.1f}%")
col3.metric("Mediaan tumorvolume (cc)", f"{np.median(tumor_volume):.1f}")

# --- Logistische regressie ---
X = df[["tumor_volume_cc", "mean_heart_dose_gy"]].values
y = df["survival"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size, random_state=int(seed)
)

model = LogisticRegression()
model.fit(X_train, y_train)

coef_df = pd.DataFrame(
    {
        "Feature": ["tumor_volume_cc", "mean_heart_dose_gy"],
        "Coëfficiënt": model.coef_[0].round(4),
        "Interpretatie": [
            "Negatief → groter volume, lager overlevingskans",
            "Dicht bij 0 → geen echte invloed",
        ],
    }
)

st.subheader("Logistische regressie — coëfficiënten (gestandaardiseerd)")
st.dataframe(coef_df, use_container_width=True, hide_index=True)
st.caption(f"Intercept: {model.intercept_[0]:.4f}")

# --- Visualisaties ---
st.subheader("Visualisaties")
tab1, tab2, tab3, tab4 = st.tabs(
    ["Verdeling features", "Correlatie & scatter", "Verwarringsmatrix", "ROC-curve"]
)

with tab1:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, label in zip(
        axes,
        ["tumor_volume_cc", "mean_heart_dose_gy"],
        ["Tumorvolume (cc)", "Mean Heart Dose (Gy)"],
    ):
        for val, name, color in [(0, "Niet overleefd", "#e74c3c"), (1, "Overleefd", "#2ecc71")]:
            ax.hist(
                df.loc[df["survival"] == val, col],
                bins=25,
                alpha=0.6,
                label=name,
                color=color,
            )
        ax.set_xlabel(label)
        ax.set_ylabel("Aantal patiënten")
        ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    colors = df["survival"].map({0: "#e74c3c", 1: "#2ecc71"})
    axes[0].scatter(df["tumor_volume_cc"], df["survival"], c=colors, alpha=0.4, s=20)
    # Smooth logistic curve
    tv_range = np.linspace(df["tumor_volume_cc"].min(), df["tumor_volume_cc"].max(), 200)
    mean_hd = df["mean_heart_dose_gy"].mean()
    X_curve = scaler.transform(np.column_stack([tv_range, np.full_like(tv_range, mean_hd)]))
    axes[0].plot(tv_range, model.predict_proba(X_curve)[:, 1], color="#2980b9", lw=2, label="Voorspelde kans")
    axes[0].set_xlabel("Tumorvolume (cc)")
    axes[0].set_ylabel("Overlevingskans")
    axes[0].set_title("Tumorvolume vs overleving")
    axes[0].legend()

    axes[1].scatter(df["mean_heart_dose_gy"], df["survival"], c=colors, alpha=0.4, s=20)
    hd_range = np.linspace(df["mean_heart_dose_gy"].min(), df["mean_heart_dose_gy"].max(), 200)
    mean_tv = df["tumor_volume_cc"].mean()
    X_curve2 = scaler.transform(np.column_stack([np.full_like(hd_range, mean_tv), hd_range]))
    axes[1].plot(hd_range, model.predict_proba(X_curve2)[:, 1], color="#8e44ad", lw=2, label="Voorspelde kans")
    axes[1].set_xlabel("Mean Heart Dose (Gy)")
    axes[1].set_ylabel("Overlevingskans")
    axes[1].set_title("Mean Heart Dose vs overleving (geen invloed)")
    axes[1].legend()

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab3:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
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

    accuracy = (y_pred == y_test).mean()
    st.metric("Nauwkeurigheid op testset", f"{accuracy*100:.1f}%")

with tab4:
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2980b9", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-curve (testset)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.divider()
st.caption(
    "Synthetische data — tumorvolume bepaalt overleving via een logistisch model. "
    "Mean heart dose is puur ruis en heeft geen causale invloed."
)
