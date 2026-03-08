"""
app.py
Sprint FTP ML — Streamlit Application (MSIS 522 · UW Foster School of Business)

Required tabs (st.tabs):
    Tab 1 — Executive Summary
    Tab 2 — Descriptive Analytics
    Tab 3 — Model Performance
    Tab 4 — Explainability & Interactive Prediction

Run:
    streamlit run app.py
"""

import os, warnings, io, contextlib
os.environ["KERAS_BACKEND"] = "torch"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "outputs" / "models"
RESULTS_DIR = BASE_DIR / "outputs" / "results"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sprint FTP ML",
    page_icon=":material/bolt:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.sidebar.title("Sprint FTP ML")
st.sidebar.caption("Ultrathon.io")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data**: GoldenCheetah OpenData (n = 4,768)  \n"
    "**Best sprint-only model**: XGBoost / Random Forest  \n"
    "**Sprint-only R²**: 0.52 · **+2m R²**: 0.70  \n"
    "**GitHub**: [sprint_FTP_ML](https://github.com/APXAXN/sprint_FTP_ML)"
)

# ══════════════════════════════════════════════════════════════════════════════
# Cached loaders
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_clean_data():
    p = BASE_DIR / "data_inspection" / "athletes_clean.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_metrics():
    dfs = []
    for fs in ["sprint_only", "sprint_eng", "sprint_bio", "sprint_bio_v2", "full_submax"]:
        p = RESULTS_DIR / f"metrics_{fs}.csv"
        if p.exists():
            df = pd.read_csv(p)
            if "feature_set" not in df.columns:
                df.insert(0, "feature_set", fs)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


@st.cache_data
def load_sprint_bio_predictions():
    p = RESULTS_DIR / "predictions_sprint_bio.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_resource
def load_all_sklearn_models():
    """Load saved sprint_bio sklearn model objects."""
    out = {}
    for name in ["Linear", "Ridge", "Lasso", "DecisionTree", "RandomForest", "XGBoost"]:
        p = MODELS_DIR / f"{name}_sprint_bio.joblib"
        if p.exists():
            try:
                out[name] = joblib.load(p)
            except Exception:
                pass
    return out


@st.cache_resource
def load_xgb_scaler_features():
    """Separate scaler + feature list needed when predicting with XGBoost model directly."""
    try:
        scaler = joblib.load(MODELS_DIR / "scaler_sprint_bio.joblib")
        feats  = joblib.load(MODELS_DIR / "feature_names_sprint_bio.joblib")
        return scaler, feats
    except Exception:
        return None, None


@st.cache_data(show_spinner=False)
def load_shap_data():
    """
    Load pre-computed SHAP values from disk (fast path), or recompute if missing.
    Returns (shap_explanation, feat_names, y_test_w, y_pred_w), or (None, str_error, None, None).
    """
    try:
        import sys; sys.path.insert(0, str(BASE_DIR))
        import shap

        # ── Fast path: load pre-computed SHAP values (~instant) ───────────────
        sv_path = MODELS_DIR / "shap_values_sprint_bio.joblib"
        if sv_path.exists():
            saved = joblib.load(sv_path)
            sv = shap.Explanation(
                values=saved["values"],
                base_values=saved["base_values"],
                data=saved["data"],
                feature_names=saved["feat_names"],
            )
            return sv, saved["feat_names"], saved["y_test_w"], saved["y_pred_w"]

        # ── Slow path: recompute from scratch (~45s) ───────────────────────────
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            from pipeline.data_loader import load_and_prepare
            X_train, X_test, y_train, y_test, feat_names, _, _, back_transform = \
                load_and_prepare("sprint_bio", log_target=True)

        rf_pipe = joblib.load(MODELS_DIR / "RandomForest_sprint_bio.joblib")
        rf_est  = rf_pipe.named_steps["model"]
        X_test_sc  = rf_pipe.named_steps["scaler"].transform(X_test)
        X_train_sc = rf_pipe.named_steps["scaler"].transform(X_train)

        expl_path = MODELS_DIR / "shap_explainer.joblib"
        if expl_path.exists():
            explainer = joblib.load(expl_path)
        else:
            explainer = shap.TreeExplainer(rf_est, data=X_train_sc)

        sv = explainer(X_test_sc)
        sv.feature_names = feat_names

        y_pred_w = np.exp(rf_est.predict(X_test_sc))
        y_test_w = np.exp(y_test)
        return sv, feat_names, y_test_w, y_pred_w
    except Exception as e:
        import traceback
        return None, str(e) + "\n" + traceback.format_exc(), None, None


@st.cache_data(show_spinner=False)
def get_pop_medians():
    """
    Compute population median values for all sprint_bio features
    (used as defaults when the user hasn't specified a value).
    """
    try:
        import sys; sys.path.insert(0, str(BASE_DIR))
        from pipeline.features import SprintFeatureEngineer
        df = load_clean_data()
        if df.empty:
            return {}, []
        _, feat_names = load_xgb_scaler_features()
        if feat_names is None:
            return {}, []
        eng = SprintFeatureEngineer()
        df_eng = eng.transform(df)
        medians = {}
        for col in feat_names:
            if col in df_eng.columns:
                medians[col] = float(df_eng[col].median())
        return medians, list(feat_names)
    except Exception:
        return {}, []


# ══════════════════════════════════════════════════════════════════════════════
# Prediction helpers
# ══════════════════════════════════════════════════════════════════════════════

def predict_ftp_any_model(model_name, raw_inputs: dict, models_dict, feat_names):
    """
    Predict log(FTP) → exp() using any loaded model.
    - sklearn Pipelines (Linear, Ridge, Lasso, DT, RF): pipeline handles scaling internally.
    - XGBoost: raw model, scale-invariant trees.
    raw_inputs should include all raw sprint_bio columns (filled with pop medians for unset features).
    """
    import sys; sys.path.insert(0, str(BASE_DIR))
    from pipeline.features import SprintFeatureEngineer

    df_in  = pd.DataFrame([raw_inputs])
    eng    = SprintFeatureEngineer()
    df_eng = eng.transform(df_in)

    X = np.zeros((1, len(feat_names)))
    for i, col in enumerate(feat_names):
        if col in df_eng.columns:
            X[0, i] = float(df_eng[col].iloc[0])

    model    = models_dict[model_name]
    log_pred = model.predict(X)[0]
    return float(np.exp(log_pred)), X


def shap_waterfall_for_input(model_name, X_aligned, feat_names, models_dict):
    """
    Compute and return a SHAP waterfall matplotlib Figure for a single input row.
    Works for tree models (DT, RF, XGBoost). Returns None for linear models.
    """
    try:
        import shap
        model = models_dict[model_name]

        if hasattr(model, "named_steps"):
            raw_model = model.named_steps["model"]
            X_sc      = model.named_steps["scaler"].transform(X_aligned)
        else:
            raw_model = model
            X_sc      = X_aligned

        if not hasattr(raw_model, "feature_importances_"):
            return None  # linear model — no TreeExplainer

        explainer = shap.TreeExplainer(raw_model)
        sv        = explainer(X_sc)
        sv.feature_names = feat_names

        fig, _ = plt.subplots(figsize=(9, 5))
        shap.plots.waterfall(sv[0], max_display=12, show=False)
        fig = plt.gcf()
        fig.suptitle(f"SHAP Waterfall — {model_name}", fontsize=11, fontweight="bold", y=1.01)
        plt.tight_layout()
        return fig
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Shared plot style
# ══════════════════════════════════════════════════════════════════════════════
DARK_BG   = "#0e0e23"
DARK_AX   = "#1a1a2e"
TICK_CLR  = "white"

def _dark_fig(figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_AX)
    ax.tick_params(colors=TICK_CLR)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    return fig, ax


MODEL_CLR = {
    "Linear": "#4e79a7", "Ridge": "#59a14f", "Lasso": "#f28e2b",
    "DecisionTree": "#e15759", "RandomForest": "#76b7b2", "XGBoost": "#edc948",
}
MODEL_ORDER = ["Linear", "Ridge", "Lasso", "DecisionTree", "RandomForest", "XGBoost"]
SPRINT_COLS = [
    "1s_critical_power", "5s_critical_power", "10s_critical_power",
    "15s_critical_power", "20s_critical_power", "30s_critical_power",
]
SPRINT_LABELS = ["1s", "5s", "10s", "15s", "20s", "30s"]

# ══════════════════════════════════════════════════════════════════════════════
# TOP-LEVEL TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Summary",
    "Descriptive Analytics",
    "Model Performance",
    "Explainability & Prediction",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.title("Can a 30-Second Sprint Predict Your FTP?")
    st.markdown(
        "##### A machine learning study of 4,768 cyclists · Ultrathon.io"
    )
    st.markdown("---")

    # Key metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Athletes analyzed", "4,768", help="After quality filtering from 6,043 raw records")
    c2.metric("Sprint-only R² (XGBoost)", "0.52", delta="MAE ≈ ±33W", delta_color="off",
              help="XGBoost trained on 6 sprint features + 9 engineered + biometrics")
    c3.metric("Adding 2-min sub-max effort", "R² = 0.70", delta="+18 pp vs sprint-only",
              help="At 2 minutes, aerobic metabolism becomes the dominant FTP predictor")
    c4.metric("Full sub-max ceiling (30m)", "R² = 0.94",
              help="Theoretical upper bound with full sub-maximal curve data")

    st.markdown("---")

    # Hero chart
    switch_img = FIGURES_DIR / "60_physiological_switch.png"
    if switch_img.exists():
        st.image(str(switch_img), use_container_width=True,
                 caption="The physiological switch: at 2 minutes, R² crosses 0.70 — the aerobic transition zone.")
    st.markdown("---")

    st.subheader("About the Dataset")
    st.markdown("""
This project uses the **GoldenCheetah OpenData** dataset
([Kaggle: markliversedge/goldencheetah-opendata](https://www.kaggle.com/datasets/markliversedge/goldencheetahopendataathleteactivityandmmp)),
a large anonymized collection of cycling training files contributed by athletes who use
GoldenCheetah — an open-source cycling analytics platform used worldwide.
From each athlete's training history, GoldenCheetah computes a **Mean Maximal Power (MMP) curve**:
the highest average power ever recorded for every duration from 1 second to 60 minutes.
After removing physiologically implausible records (corrupted age values, power readings outside
credible ranges for cycling, and missing biometrics), the dataset retains **4,768 clean athletes**
from an original 6,043 (79% retention). The cohort is 97% male, ages 14–88, and spans the full
spectrum from recreational cyclists to competitive amateur racers.

The **prediction target** is `20m_critical_power`: the athlete's all-time best average power over
a 20-minute effort. This is the standard field proxy for **Functional Threshold Power (FTP)** —
the gold-standard metric of aerobic cycling performance, used to set training zones, evaluate
fitness, and guide race pacing strategy. FTP values in this dataset range from 57W to 587W,
with a mean of 273W (std = 68W), reflecting the wide ability range across the dataset.
The **features** available for prediction fall into three groups: (1) six raw sprint power readings
from the MMP curve (best 1s, 5s, 10s, 15s, 20s, and 30s average power), which are the primary
inputs for the sprint prediction thesis; (2) nine engineered features capturing the *shape* of the
sprint power-duration curve — fatigue index (30s/1s ratio), early/mid/late decay ratios,
anaerobic reserve (1s−30s absolute loss), decay curvature, and power-to-weight ratios at key
sprint durations; and (3) three biometric features (body weight in kg, age, gender).
""")

    st.subheader("Why This Problem Matters")
    st.markdown("""
Measuring FTP directly requires a demanding 20-minute all-out cycling effort: one that is
physically exhausting, psychologically challenging, requires accurate self-pacing (going out too
hard ruins the test), and demands access to a calibrated power meter and a safe testing
environment. This is a real barrier for many athletes and coaches. For **remote coaches** working
with athletes across time zones, for **sports scientists** conducting population-level fitness
screening, or for **recreational cyclists** without the experience to pace a 20-minute maximal
effort accurately, an FTP test is often impractical or unreliable.

If a **30-second maximal sprint** — a far simpler, faster, and less physically taxing effort —
could reliably predict FTP, it would meaningfully lower the barrier to aerobic fitness assessment.
A sprint takes under a minute, doesn't require careful pacing, can be performed on a trainer or
an empty road, and provides additional information (peak neuromuscular power, anaerobic capacity)
as a byproduct. From a physiological standpoint, the 30-second sprint engages both anaerobic
(PCr stores, rapid glycolysis) and aerobic metabolic pathways, with aerobic contribution growing
as duration increases. This raises the core research question: *how much* of the aerobic capacity
underlying FTP is already visible in a 30-second sprint — and at what effort duration does the
aerobic system become the *dominant* predictor of sustained 20-minute power?
""")

    st.subheader("Approach & Key Findings")
    st.markdown("""
Six machine learning models were trained on a primary 18-feature set (`sprint_bio`: six sprint
power readings + nine engineered decay/ratio features + three biometrics), using a stratified
70/30 train-test split (3,338 train / 1,430 test athletes) with 5-fold cross-validation for
hyperparameter tuning. Models ranged from OLS Linear Regression (baseline) through
Ridge/Lasso regularization, Decision Tree (GridSearchCV), Random Forest and XGBoost
(RandomizedSearchCV), and a Keras MLP neural network. All models predict `log(FTP)` during
training and back-transform to watts for evaluation — this normalization addresses the mild
right skew in FTP and prevents negative predictions.

The best sprint-only models (XGBoost and Random Forest, which effectively tie) achieve
**R² ≈ 0.52 and MAE ≈ 33W** — a statistically significant and practically useful
prediction, but not a replacement for an FTP test. A **SHAP analysis** identifies
`30s_critical_power` as the single most important feature by a factor of ~6× over any other,
confirming the thesis that longer sprint durations engage aerobic metabolism progressively more.
Pearson correlation increases monotonically from r = 0.44 (1s) to r = 0.68 (30s) — the
30-second effort is the best sprint predictor of FTP.

The central finding of the project is the **physiological switch point at 2 minutes**. A
separate ablation study tested adding progressively longer sub-maximal effort data to the sprint
feature set. Adding just the **2-minute critical power** to sprint features jumps R² from 0.52
to 0.70 — a **+18 percentage point gain** that represents the transition from anaerobic/sprint
physiology to aerobic metabolism as the dominant predictor of FTP. This aligns with exercise
science: at ~75 seconds, aerobic and anaerobic ATP contributions are roughly equal (Gastin 2001);
by 2 minutes, the aerobic system supplies ~65% of energy demand (Medbø & Tabata). A 2-minute
field test could achieve 70% accuracy in FTP estimation — significantly easier than a 20-minute
test — making it a viable screening tool for coaches performing large-group assessments or
remote athlete monitoring.
""")

    st.info(
        "**Bottom line for non-technical readers**: You can estimate a cyclist's FTP from a "
        "30-second sprint with 52% explained variance and ~33W average error. That's useful — "
        "but the real insight is that extending the effort to just 2 minutes more than halves "
        "the unexplained variance. The sprint captures neuromuscular power; the 2-minute effort "
        "captures the aerobic engine. Together, they locate each athlete on a sprint-to-endurance "
        "continuum that has direct implications for training design."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DESCRIPTIVE ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    import scipy.stats as stats

    st.title("Descriptive Analytics")
    st.markdown(
        "A deep look at the dataset before modeling — "
        "distributions, feature relationships, and the correlation structure that "
        "motivates the sprint-to-FTP prediction thesis."
    )
    st.markdown("---")

    df = load_clean_data()
    if df.empty:
        st.error("Could not load `data_inspection/athletes_clean.csv`.")
        st.stop()

    df["ftp_wpk"] = df["20m_critical_power"] / df["weightkg"]
    df["ftp_quartile"] = pd.qcut(
        df["20m_critical_power"], q=4,
        labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
    )

    # ── 1.1 Dataset Introduction ──────────────────────────────────────────────
    st.subheader("1.1 Dataset Introduction")

    with st.expander("About the dataset", expanded=True):
        st.markdown("""
**Source**: [GoldenCheetah OpenData](https://github.com/GoldenCheetah/OpenData) via Kaggle
(`markliversedge/goldencheetah-opendata-athlete-activity-and-mmp`)

GoldenCheetah is an open-source cycling analytics platform used worldwide by
amateur and professional cyclists. Athletes who opted in to data sharing contributed
anonymized training files, from which **Mean Maximal Power (MMP) curves** were computed —
the best average power achievable for every duration from 1 second to 60 minutes.

**Prediction target**: `20m_critical_power` — an athlete's highest mean power
sustained for 20 minutes. This is the standard field proxy for **Functional Threshold
Power (FTP)**, the cornerstone metric of aerobic cycling performance.

**Quality filters** removed physiologically implausible records (corrupted ages, power
readings outside credible ranges, missing biometrics), retaining **4,768 clean athletes**
from an original 6,043 (79% retention).
""")

    n_rows, n_cols_total = df.shape
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Athletes (rows)", f"{n_rows:,}")
    c2.metric("Total columns", f"{n_cols_total}")
    c3.metric("Numerical features", f"{len(num_cols)}")
    c4.metric("Categorical features", f"{len(cat_cols)}")

    feat_type_df = pd.DataFrame({
        "Group": [
            "Sprint power (raw)", "Sub-maximal power (reference)",
            "Biometrics", "Activity counts", "W/kg peaks",
            "Categorical", "Target",
        ],
        "Features": [
            "1s, 5s, 10s, 15s, 20s, 30s critical power (watts)",
            "2m, 3m, 5m, 8m, 10m, 30m critical power (watts)",
            "weight (kg), age (years)",
            "total activities, cycling, run, swim, other",
            "1m, 5m, 10m, 20m, 30m peak W/kg",
            "gender (M/F)",
            "20m_critical_power — FTP proxy (watts)",
        ],
        "Count": [6, 6, 2, 5, 5, 1, 1],
    })
    st.dataframe(feat_type_df, use_container_width=True, hide_index=True)

    key_cols = SPRINT_COLS + ["20m_critical_power", "weightkg", "age"]
    desc = df[key_cols].describe().T[["mean", "std", "min", "50%", "max"]].rename(
        columns={"50%": "median"}
    )
    desc.index = [c.replace("_critical_power", "").replace("s_", "s ").replace("kg", " (kg)")
                  for c in desc.index]
    st.markdown("**Descriptive statistics — key columns**")
    st.dataframe(desc.style.format("{:.1f}"), use_container_width=True)

    st.markdown("---")

    # ── 1.2 Target Distribution ───────────────────────────────────────────────
    st.subheader("1.2 Target Variable Distribution")

    ftp = df["20m_critical_power"]
    mean_ftp   = ftp.mean()
    median_ftp = ftp.median()
    std_ftp    = ftp.std()
    skew_ftp   = ftp.skew()

    fig_target, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig_target.patch.set_facecolor(DARK_BG)

    ax = axes[0]
    ax.set_facecolor(DARK_AX)
    ax.hist(ftp, bins=60, color="steelblue", alpha=0.7, edgecolor="white",
            density=True, label="Histogram")
    kde_x = np.linspace(ftp.min(), ftp.max(), 400)
    kde   = stats.gaussian_kde(ftp)
    ax.plot(kde_x, kde(kde_x), color="orangered", lw=2, label="KDE")
    ax.axvline(mean_ftp,   color="gold",  lw=1.8, ls="--", label=f"Mean  {mean_ftp:.0f}W")
    ax.axvline(median_ftp, color="white", lw=1.8, ls=":",  label=f"Median {median_ftp:.0f}W")
    ax.set_xlabel("20-min Critical Power (W)", fontsize=11, color="white")
    ax.set_ylabel("Density", fontsize=11, color="white")
    ax.set_title("FTP Distribution (all athletes)", fontsize=12, fontweight="bold", color="white")
    ax.legend(fontsize=9, facecolor=DARK_AX, labelcolor="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")

    ax2 = axes[1]
    ax2.set_facecolor(DARK_AX)
    for i, (gender, color) in enumerate([("M", "steelblue"), ("F", "orangered")]):
        sub = df[df["gender"] == gender]["20m_critical_power"]
        ax2.boxplot(sub, positions=[i], widths=0.4, patch_artist=True,
                    boxprops=dict(facecolor=color, alpha=0.6),
                    medianprops=dict(color="white", lw=2),
                    whiskerprops=dict(color="white"), capprops=dict(color="white"),
                    flierprops=dict(marker="o", color=color, alpha=0.3, markersize=3))
        ax2.scatter(np.random.normal(i, 0.06, size=min(300, len(sub))),
                    np.random.choice(sub, size=min(300, len(sub)), replace=False),
                    color=color, alpha=0.25, s=8)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(
        [f"Male (n={int((df['gender']=='M').sum()):,})",
         f"Female (n={int((df['gender']=='F').sum())})"],
        color="white"
    )
    ax2.set_ylabel("20-min Critical Power (W)", fontsize=11, color="white")
    ax2.set_title("FTP by Gender", fontsize=12, fontweight="bold", color="white")
    ax2.tick_params(colors="white")
    for sp in ax2.spines.values(): sp.set_edgecolor("#444")

    plt.tight_layout(pad=1.5)
    st.pyplot(fig_target, use_container_width=True)
    plt.close(fig_target)

    st.caption(
        f"**Left**: FTP follows a mildly right-skewed distribution (skew = {skew_ftp:.2f}), "
        f"with mean {mean_ftp:.0f}W and median {median_ftp:.0f}W. A small cohort of elite cyclists "
        f"pulls the tail rightward — a pattern that motivates using log(FTP) as the model target. "
        f"**Right**: Male athletes average {df[df['gender']=='M']['20m_critical_power'].mean():.0f}W "
        f"versus {df[df['gender']=='F']['20m_critical_power'].mean():.0f}W for females, reflecting "
        f"known physiological differences in absolute power output (not power-to-weight ratio, "
        f"which normalizes most of this gap)."
    )
    st.markdown("---")

    # ── 1.3 Feature Visualizations ────────────────────────────────────────────
    st.subheader("1.3 Feature Distributions and Relationships")

    tab_v1, tab_v2, tab_v3, tab_v4 = st.tabs([
        "Sprint Decay Curves",
        "Sprint → FTP Scatter Grid",
        "FTP Quartile Profiles",
        "Biometric Relationships",
    ])

    with tab_v1:
        st.markdown("#### Sprint Power Decay by FTP Quartile")
        fig_decay, ax_d = plt.subplots(figsize=(10, 5))
        fig_decay.patch.set_facecolor(DARK_BG)
        ax_d.set_facecolor(DARK_AX)
        palette = ["#e15759", "#f28e2b", "#4e79a7", "#59a14f"]
        durations = [1, 5, 10, 15, 20, 30]

        for i, quartile in enumerate(["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]):
            grp = df[df["ftp_quartile"] == quartile][SPRINT_COLS]
            medians = grp.median().values
            q25     = grp.quantile(0.25).values
            q75     = grp.quantile(0.75).values
            med_ftp = df[df["ftp_quartile"] == quartile]["20m_critical_power"].median()
            ax_d.plot(durations, medians, color=palette[i], lw=2.5, marker="o", ms=6,
                      label=f"{quartile} (FTP≈{med_ftp:.0f}W)")
            ax_d.fill_between(durations, q25, q75, color=palette[i], alpha=0.12)

        ax_d.set_xlabel("Sprint Duration (seconds)", fontsize=11, color="white")
        ax_d.set_ylabel("Mean Maximal Power (W)", fontsize=11, color="white")
        ax_d.set_title("Sprint Power Decay Curves — Median by FTP Quartile (shaded = IQR)",
                       fontsize=12, fontweight="bold", color="white")
        ax_d.set_xticks(durations)
        ax_d.legend(fontsize=9, facecolor=DARK_AX, labelcolor="white",
                    title="FTP Quartile", title_fontsize=9).get_title().set_color("white")
        ax_d.tick_params(colors="white")
        for sp in ax_d.spines.values(): sp.set_edgecolor("#444")
        plt.tight_layout()
        st.pyplot(fig_decay, use_container_width=True)
        plt.close(fig_decay)

        st.caption(
            "**Sprint power decay curves by FTP quartile.** Athletes in higher FTP quartiles "
            "produce substantially more power at every sprint duration, and the gap between "
            "quartiles *widens* as sprint duration increases. This widening is explained by "
            "greater aerobic contribution at 30 seconds for high-FTP (aerobically superior) "
            "athletes — the physiological basis of the prediction thesis. Shaded bands show "
            "the interquartile range (25th–75th percentile); no quartile curves cross, "
            "confirming a consistent and monotonic relationship between FTP level and sprint output."
        )

    with tab_v2:
        st.markdown("#### Sprint Power vs. 20-min FTP — All 6 Durations")
        sample = df.sample(n=min(1500, len(df)), random_state=42)
        fig_sc, axes_sc = plt.subplots(2, 3, figsize=(13, 8))
        fig_sc.patch.set_facecolor(DARK_BG)

        for idx, (col, lbl) in enumerate(zip(SPRINT_COLS, SPRINT_LABELS)):
            row, col_idx = divmod(idx, 3)
            ax = axes_sc[row][col_idx]
            ax.set_facecolor(DARK_AX)
            r  = df[col].corr(df["20m_critical_power"])
            ax.scatter(sample[col], sample["20m_critical_power"],
                       alpha=0.25, s=10, color="steelblue")
            m, b = np.polyfit(df[col].dropna(), df["20m_critical_power"][df[col].notna()], 1)
            x_line = np.linspace(df[col].min(), df[col].max(), 100)
            ax.plot(x_line, m * x_line + b, color="orangered", lw=1.8)
            ax.set_xlabel(f"{lbl} power (W)", fontsize=9, color="white")
            ax.set_ylabel("20-min FTP (W)" if col_idx == 0 else "", fontsize=9, color="white")
            ax.set_title(f"{lbl} sprint  |  r = {r:.2f}", fontsize=10,
                         fontweight="bold", color="white")
            ax.tick_params(colors="white")
            for sp in ax.spines.values(): sp.set_edgecolor("#444")

        plt.suptitle("Sprint Duration vs. FTP: Pearson r increases monotonically (1s→30s)",
                     fontsize=12, fontweight="bold", color="white", y=1.01)
        plt.tight_layout()
        st.pyplot(fig_sc, use_container_width=True)
        plt.close(fig_sc)

        r_vals = {lbl: df[col].corr(df["20m_critical_power"])
                  for col, lbl in zip(SPRINT_COLS, SPRINT_LABELS)}
        r_df = pd.DataFrame({"Sprint Duration": list(r_vals.keys()),
                              "Pearson r with FTP": list(r_vals.values())})
        st.dataframe(r_df.style.format({"Pearson r with FTP": "{:.3f}"}),
                     use_container_width=True, hide_index=True)
        st.caption(
            "**Scatter plots of each sprint duration vs. 20-minute FTP** (1,500-athlete sample; "
            "full dataset regression lines). Pearson correlation increases **monotonically** from "
            "r = 0.44 at 1 second to r = 0.68 at 30 seconds — the central empirical finding "
            "motivating the thesis. The positive slope is clear at all durations, but the scatter "
            "is substantial (especially at 1s), meaning sprint power alone is informative but "
            "far from deterministic. The increasing r values reflect growing aerobic contribution "
            "at longer durations, which is physiologically consistent with the energy system crossover."
        )

    with tab_v3:
        st.markdown("#### Feature Distributions by FTP Quartile")
        col_sel = st.selectbox(
            "Select feature",
            options=SPRINT_COLS + ["weightkg", "age", "ftp_wpk", "activities"],
            format_func=lambda c: c.replace("_critical_power", "").replace("_", " ").title(),
            index=5,
        )
        fig_vio, ax_v = plt.subplots(figsize=(10, 5))
        fig_vio.patch.set_facecolor(DARK_BG)
        ax_v.set_facecolor(DARK_AX)
        quartile_order = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
        palette_v = {"Q1 (lowest)": "#e15759", "Q2": "#f28e2b",
                     "Q3": "#4e79a7", "Q4 (highest)": "#59a14f"}

        data_by_q = [df[df["ftp_quartile"] == q][col_sel].dropna() for q in quartile_order]
        for pos, (data, q) in enumerate(zip(data_by_q, quartile_order), start=1):
            parts = ax_v.violinplot(data, positions=[pos], showmedians=False, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(palette_v[q]); pc.set_alpha(0.5)
            ax_v.boxplot(data, positions=[pos], widths=0.12, patch_artist=True,
                         boxprops=dict(facecolor=palette_v[q], alpha=0.8),
                         medianprops=dict(color="white", lw=2),
                         whiskerprops=dict(color="white"), capprops=dict(color="white"),
                         flierprops=dict(marker=".", color=palette_v[q], alpha=0.3))
        ax_v.set_xticks([1, 2, 3, 4])
        ax_v.set_xticklabels(quartile_order, color="white")
        ax_v.set_ylabel(col_sel.replace("_critical_power", " power (W)").replace("_", " ").title(),
                        color="white", fontsize=11)
        ax_v.set_title(f"{col_sel} Distribution by FTP Quartile", fontsize=12,
                       fontweight="bold", color="white")
        ax_v.tick_params(colors="white")
        for sp in ax_v.spines.values(): sp.set_edgecolor("#444")
        plt.tight_layout()
        st.pyplot(fig_vio, use_container_width=True)
        plt.close(fig_vio)

        med_tbl = df.groupby("ftp_quartile", observed=True)[col_sel].agg(
            ["median", "mean", "std"]
        ).rename(columns={"median": "Median", "mean": "Mean", "std": "Std Dev"})
        st.dataframe(med_tbl.style.format("{:.1f}"), use_container_width=True)
        st.caption(
            f"**Violin + boxplot of `{col_sel}` by FTP quartile.** "
            "Higher FTP groups consistently show higher sprint power values; "
            "the distribution shifts and spreads upward monotonically across quartiles. "
            "The violin widths reveal that elite (Q4) athletes are more physiologically "
            "diverse than beginners (Q1) — a common pattern in sports performance data "
            "where early gains narrow the field while advanced performers diverge."
        )

    with tab_v4:
        st.markdown("#### Biometric Relationships with FTP")
        fig_bio, axes_bio = plt.subplots(1, 2, figsize=(13, 5))
        fig_bio.patch.set_facecolor(DARK_BG)

        ax_a = axes_bio[0]
        ax_a.set_facecolor(DARK_AX)
        colors_g = {"M": "steelblue", "F": "orangered"}
        for gender_val, color in colors_g.items():
            sub = df[df["gender"] == gender_val].sample(
                n=min(500, (df["gender"] == gender_val).sum()), random_state=1)
            ax_a.scatter(sub["age"], sub["20m_critical_power"],
                         alpha=0.3, s=12, color=color, label=gender_val)
        valid = df[["age", "20m_critical_power"]].dropna()
        m_a, b_a = np.polyfit(valid["age"], valid["20m_critical_power"], 1)
        x_age = np.linspace(valid["age"].min(), valid["age"].max(), 100)
        ax_a.plot(x_age, m_a * x_age + b_a, color="gold", lw=2, ls="--",
                  label=f"OLS (r={valid['age'].corr(valid['20m_critical_power']):.2f})")
        ax_a.set_xlabel("Age (years)", fontsize=10, color="white")
        ax_a.set_ylabel("20-min FTP (W)", fontsize=10, color="white")
        ax_a.set_title("Age vs. FTP (by gender)", fontsize=11, fontweight="bold", color="white")
        ax_a.legend(fontsize=9, facecolor=DARK_AX, labelcolor="white")
        ax_a.tick_params(colors="white")
        for sp in ax_a.spines.values(): sp.set_edgecolor("#444")

        ax_b = axes_bio[1]
        ax_b.set_facecolor(DARK_AX)
        for i, (gender_val, color) in enumerate([("M", "steelblue"), ("F", "orangered")]):
            sub_wpk = df[df["gender"] == gender_val]["ftp_wpk"].dropna()
            parts = ax_b.violinplot(sub_wpk, positions=[i], showmedians=False, showextrema=False)
            for pc in parts["bodies"]: pc.set_facecolor(color); pc.set_alpha(0.55)
            ax_b.boxplot(sub_wpk, positions=[i], widths=0.12, patch_artist=True,
                         boxprops=dict(facecolor=color, alpha=0.8),
                         medianprops=dict(color="white", lw=2),
                         whiskerprops=dict(color="white"), capprops=dict(color="white"),
                         flierprops=dict(marker=".", color=color, alpha=0.3))
        ax_b.set_xticks([0, 1])
        ax_b.set_xticklabels([f"Male (n={int((df['gender']=='M').sum()):,})",
                               f"Female (n={int((df['gender']=='F').sum())})"], color="white")
        ax_b.set_ylabel("FTP W/kg", fontsize=10, color="white")
        ax_b.set_title("FTP Power-to-Weight by Gender", fontsize=11, fontweight="bold", color="white")
        ax_b.tick_params(colors="white")
        for sp in ax_b.spines.values(): sp.set_edgecolor("#444")

        plt.tight_layout(pad=2)
        st.pyplot(fig_bio, use_container_width=True)
        plt.close(fig_bio)

        age_r = df["age"].corr(df["20m_critical_power"])
        m_wpk = df[df["gender"] == "M"]["ftp_wpk"].median()
        f_wpk = df[df["gender"] == "F"]["ftp_wpk"].median()
        st.caption(
            f"**Left**: Age has a weak negative correlation with FTP (r = {age_r:.2f}), "
            f"reflecting physiological decline across the lifespan, though the relationship "
            f"is noisy — elite masters athletes can outperform younger amateurs in absolute watts. "
            f"**Right**: Males and females have similar *relative* FTP (median {m_wpk:.2f} vs. "
            f"{f_wpk:.2f} W/kg), despite large differences in absolute watts (~50W gap). "
            f"This motivates including W/kg engineered features in the model to normalize "
            f"body composition effects across athletes."
        )

    st.markdown("---")

    # ── 1.4 Correlation Heatmap ───────────────────────────────────────────────
    st.subheader("1.4 Correlation Heatmap")

    heat_cols = (
        SPRINT_COLS +
        ["2m_critical_power", "3m_critical_power", "5m_critical_power",
         "8m_critical_power", "10m_critical_power", "30m_critical_power"] +
        ["weightkg", "age", "20m_critical_power"]
    )
    heat_labels = (
        SPRINT_LABELS +
        ["2m", "3m", "5m", "8m", "10m", "30m"] +
        ["weight (kg)", "age", "20m FTP"]
    )
    heat_cols = [c for c in heat_cols if c in df.columns]
    heat_labels = heat_labels[:len(heat_cols)]

    corr_df   = df[heat_cols].corr()
    corr_disp = corr_df.copy()
    corr_disp.index   = heat_labels
    corr_disp.columns = heat_labels

    fig_heat, ax_h = plt.subplots(figsize=(12, 9))
    fig_heat.patch.set_facecolor(DARK_BG)
    mask = np.zeros_like(corr_disp, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    sns.heatmap(corr_disp, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=-1, vmax=1, center=0, ax=ax_h, linewidths=0.4, linecolor="#333",
                annot_kws={"size": 7}, cbar_kws={"shrink": 0.7, "label": "Pearson r"})
    ax_h.set_title("Correlation Heatmap — Sprint, Sub-Max, and Biometric Features vs. 20-min FTP",
                   fontsize=12, fontweight="bold", color="white", pad=14)
    ax_h.tick_params(axis="x", rotation=45, labelsize=9, colors="white")
    ax_h.tick_params(axis="y", rotation=0,  labelsize=9, colors="white")
    ax_h.set_facecolor(DARK_AX)
    cbar = ax_h.collections[0].colorbar
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white")
    plt.tight_layout()
    st.pyplot(fig_heat, use_container_width=True)
    plt.close(fig_heat)

    target_corrs = corr_df["20m_critical_power"].drop("20m_critical_power").sort_values(ascending=False)
    top5 = target_corrs.head(5)
    st.markdown("**Top 5 correlations with 20m FTP:**")
    st.dataframe(
        pd.DataFrame({"Feature": top5.index, "Pearson r": top5.values})
        .style.format({"Pearson r": "{:.3f}"}),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "**Correlation heatmap** (lower-triangle only; upper masked for readability). "
        "Sub-maximal features (2m, 3m, 5m, 8m) show the strongest correlations with FTP "
        "(r > 0.85), confirming they share physiological space with 20-minute sustained power. "
        "Sprint features show moderate positive correlations (r = 0.44–0.68), with 30s being "
        "the strongest sprint predictor. A key multicollinearity concern is visible: sprint features "
        "are highly inter-correlated (r > 0.90 between adjacent durations), which motivates using "
        "engineered decay ratio features that capture the *shape* of the sprint curve rather than "
        "just individual power readings — reducing redundancy and adding unique signal."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.title("Model Performance")
    st.caption(
        "Feature set: **sprint_bio** (6 sprint + 9 engineered + 3 biometric = 18 features)  ·  "
        "Target: **log(FTP) → exp back-transform**  ·  "
        "Train n=3,338 / Test n=1,430  ·  5-fold CV  ·  random_state=42"
    )
    st.markdown("---")

    all_metrics = load_metrics()
    sb = all_metrics[all_metrics["feature_set"] == "sprint_bio"].copy() \
         if not all_metrics.empty else pd.DataFrame()
    pred_df = load_sprint_bio_predictions()
    ml      = load_all_sklearn_models()

    SPRINT_BIO_FEATS = [
        "1s_critical_power", "5s_critical_power", "10s_critical_power",
        "15s_critical_power", "20s_critical_power", "30s_critical_power",
        "fatigue_index", "early_decay", "mid_decay", "late_decay",
        "anaerobic_reserve", "decay_curvature",
        "sprint_wpk_1s", "sprint_wpk_15s", "sprint_wpk_30s",
        "weightkg", "age", "gender_encoded",
    ]
    FEAT_LABELS = [
        "1s power", "5s power", "10s power", "15s power", "20s power", "30s power",
        "Fatigue Index", "Early Decay", "Mid Decay", "Late Decay",
        "Anaerobic Reserve", "Decay Curvature",
        "W/kg (1s)", "W/kg (15s)", "W/kg (30s)",
        "Weight (kg)", "Age", "Gender",
    ]

    def _m(name):
        if sb.empty: return {}
        r = sb[sb["model"] == name]
        return r.iloc[0].to_dict() if len(r) else {}

    def _scatter_fig(y_true, y_pred, title, r2, mae):
        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(DARK_AX)
        ax.scatter(y_true, y_pred, alpha=0.25, s=10, color="steelblue")
        lo = min(float(y_true.min()), float(y_pred.min()))
        hi = max(float(y_true.max()), float(y_pred.max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y = x (perfect)")
        ax.set_xlabel("Actual FTP (W)", color="white", fontsize=9)
        ax.set_ylabel("Predicted FTP (W)", color="white", fontsize=9)
        ax.set_title(f"{title}\nR²={r2:.3f}  |  MAE={mae:.1f}W",
                     color="white", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, facecolor=DARK_AX, labelcolor="white")
        ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_edgecolor("#444")
        plt.tight_layout()
        return fig

    def _importance_fig(importances, labels, title, color="steelblue", top_n=12):
        idx = np.argsort(importances)[-top_n:]
        fig, ax = plt.subplots(figsize=(6.5, 4))
        fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(DARK_AX)
        ax.barh([labels[i] for i in idx], importances[idx], color=color, alpha=0.85)
        ax.set_xlabel("Feature Importance", color="white", fontsize=9)
        ax.set_title(title, color="white", fontsize=10, fontweight="bold")
        ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_edgecolor("#444")
        plt.tight_layout()
        return fig

    def _metrics_card(name):
        m = _m(name)
        if not m:
            st.caption("Metrics not available.")
            return
        st.dataframe(pd.DataFrame({
            "Metric": ["Test R²", "CV R²", "Test MAE (W)", "Test RMSE (W)", "MAPE (%)"],
            "Value": [f"{m.get('test_R2',0):.3f}", f"{m.get('cv_R2',0):.3f}",
                      f"{m.get('test_MAE',0):.1f}", f"{m.get('test_RMSE',0):.1f}",
                      f"{m.get('test_MAPE',0):.1f}"],
        }), use_container_width=True, hide_index=True)

    t31, t32, t33, t34, t35, t36, t37 = st.tabs([
        "2.1 Feature Set",
        "2.2 Linear Baseline",
        "2.3 Decision Tree",
        "2.4 Random Forest",
        "2.5 XGBoost",
        "2.6 Neural Network",
        "2.7 Comparison & Ablation",
    ])

    # ── 2.1 Feature Set ────────────────────────────────────────────────────────
    with t31:
        st.subheader("2.1 Data Preparation & Feature Set")
        col_l, col_r = st.columns([1.3, 1])
        with col_l:
            st.markdown("**sprint_bio feature set — 18 features**")
            feat_tbl = pd.DataFrame({
                "Group": ["Sprint (raw)"] * 6 + ["Engineered — decay"] * 6 +
                          ["Engineered — W/kg"] * 3 + ["Biometric"] * 3,
                "Feature": SPRINT_BIO_FEATS,
                "Label": FEAT_LABELS,
                "Description": [
                    "Peak 1-s power (W)", "Best 5-s avg power (W)",
                    "Best 10-s avg power (W)", "Best 15-s avg power (W)",
                    "Best 20-s avg power (W)", "Best 30-s avg power (W)",
                    "30s/1s — fraction sustained", "5s/1s — first drop-off",
                    "15s/5s — mid-sprint decay", "30s/15s — late decay",
                    "1s−30s absolute loss (W)", "log curvature of PD curve",
                    "1s power / body weight", "15s power / body weight",
                    "30s power / body weight",
                    "Body weight (kg)", "Age (years)", "Gender (0=M, 1=F)",
                ],
            })
            st.dataframe(feat_tbl, use_container_width=True, hide_index=True)
        with col_r:
            st.markdown("**Pipeline steps**")
            st.markdown("""
**1. Source**: `athletes_clean.csv` — 4,768 athletes after quality filtering

**2. Feature engineering**: `SprintFeatureEngineer` computes 9 derived features (decay ratios, W/kg, curvature) and encodes gender (M=0, F=1)

**3. Train / test split**: Stratified 70/30 on gender label
- Train: **n = 3,338**
- Test: **n = 1,430** (held-out, never seen during tuning)

**4. Target transformation**: `y = log(FTP)`
- Back-transform: `ŷ_W = exp(ŷ_log)` for all reported metrics
- Rationale: FTP is right-skewed (skew=0.64); log normalizes and prevents negative predictions

**5. Scaling**: `RobustScaler` (fit on train, applied to test)
- Uses median/IQR — robust to sprint outliers

**6. CV strategy**: 5-fold cross-validation on train set only

**7. Metrics**: R², MAE (W), RMSE (W), MAPE (%) — all on test set in watts
""")
            st.info(
                "**Gender stratification**: 97% male dataset (n=4,622 M, n=146 F). "
                "Stratified splitting ensures ~3% female in both train and test."
            )

    # ── 2.2 Linear Baseline ────────────────────────────────────────────────────
    with t32:
        st.subheader("2.2 Linear Regression Baseline")
        st.markdown(
            "Three linear variants: **OLS** (no regularization), "
            "**Ridge** (RidgeCV, α ∈ {0.01, 0.1, 1, 10, 100, 1000}), "
            "**Lasso** (LassoCV, α ∈ {0.001, 0.01, 0.1, 1, 10, 100}). "
            "All preceded by `RobustScaler`."
        )
        lin_names = ["Linear", "Ridge", "Lasso"]
        lin_rows  = sb[sb["model"].isin(lin_names)].copy() if not sb.empty else pd.DataFrame()
        if not lin_rows.empty:
            lin_rows["_ord"] = lin_rows["model"].apply(lambda x: lin_names.index(x))
            lin_rows = lin_rows.sort_values("_ord")
            show = lin_rows[["model", "test_R2", "test_MAE", "test_RMSE", "test_MAPE", "cv_R2"]].rename(
                columns={"model": "Model", "test_R2": "Test R²", "test_MAE": "Test MAE (W)",
                         "test_RMSE": "Test RMSE (W)", "test_MAPE": "MAPE (%)", "cv_R2": "CV R²"}
            )
            st.dataframe(show.style.format({
                "Test R²": "{:.3f}", "CV R²": "{:.3f}",
                "Test MAE (W)": "{:.1f}", "Test RMSE (W)": "{:.1f}", "MAPE (%)": "{:.1f}",
            }), use_container_width=True, hide_index=True)

        # Best hyperparameters
        st.markdown("**Best Hyperparameters**")
        hp_rows = []
        if "Ridge" in ml:
            try:
                alpha = ml["Ridge"].named_steps["model"].alpha_
                hp_rows.append({"Model": "Ridge", "Parameter": "best α (RidgeCV)", "Value": f"{alpha:.4f}"})
            except Exception:
                pass
        if "Lasso" in ml:
            try:
                la = ml["Lasso"].named_steps["model"]
                n_nz = int((la.coef_ != 0).sum())
                hp_rows.append({"Model": "Lasso", "Parameter": "best α (LassoCV)", "Value": f"{la.alpha_:.4f}"})
                hp_rows.append({"Model": "Lasso", "Parameter": "non-zero coefficients", "Value": f"{n_nz} / {len(SPRINT_BIO_FEATS)}"})
            except Exception:
                pass
        if not hp_rows:
            hp_rows = [{"Model": "Linear/Ridge/Lasso", "Parameter": "—", "Value": "See metrics above"}]
        st.dataframe(pd.DataFrame(hp_rows), use_container_width=True, hide_index=True)

        col_res, col_sc = st.columns(2)
        with col_res:
            if not pred_df.empty and "y_pred_Linear" in pred_df.columns:
                resid = pred_df["y_pred_Linear"] - pred_df["y_true"]
                fig_r, ax_r = plt.subplots(figsize=(5.5, 4))
                fig_r.patch.set_facecolor(DARK_BG); ax_r.set_facecolor(DARK_AX)
                ax_r.hist(resid, bins=55, color="steelblue", alpha=0.75, density=True, edgecolor="white")
                ax_r.axvline(0, color="orangered", lw=1.8, ls="--", label="Zero error")
                ax_r.axvline(resid.mean(), color="gold", lw=1.5, ls=":", label=f"Mean={resid.mean():.1f}W")
                ax_r.set_xlabel("Residual (Pred − Actual) W", color="white", fontsize=9)
                ax_r.set_ylabel("Density", color="white", fontsize=9)
                ax_r.set_title("Linear — Residual Distribution", color="white", fontsize=10, fontweight="bold")
                ax_r.legend(fontsize=8, facecolor=DARK_AX, labelcolor="white")
                ax_r.tick_params(colors="white")
                for sp in ax_r.spines.values(): sp.set_edgecolor("#444")
                plt.tight_layout()
                st.pyplot(fig_r, use_container_width=True)
                plt.close(fig_r)
        with col_sc:
            if not pred_df.empty and "y_pred_Linear" in pred_df.columns:
                m = _m("Linear")
                fig = _scatter_fig(pred_df["y_true"].values, pred_df["y_pred_Linear"].values,
                                   "Linear Regression", m.get("test_R2", 0), m.get("test_MAE", 0))
                st.pyplot(fig, use_container_width=True); plt.close(fig)

        st.info(
            "**Interpretation**: All three linear models converge to nearly identical performance "
            "(R² ≈ 0.45, MAE ≈ 35W) — this is our **baseline**. The near-parity across Linear, "
            "Ridge, and Lasso was unexpected given high multicollinearity (r > 0.90 between adjacent "
            "sprint durations). The RobustScaler + log(FTP) transformation conditions the problem "
            "well enough that additional regularization adds marginal value. Residuals are roughly "
            "centered on zero with mild right skew — the model undershoots elite athletes."
        )

    # ── 2.3 Decision Tree ──────────────────────────────────────────────────────
    with t33:
        st.subheader("2.3 Decision Tree / CART")
        st.markdown(
            "**GridSearchCV** (5-fold CV, scoring = R²) over: "
            "`max_depth` ∈ {3, 5, 7, 10, None}, "
            "`min_samples_leaf` ∈ {5, 10, 20}, "
            "`min_samples_split` ∈ {10, 20}."
        )
        col_p, col_m2 = st.columns(2)
        with col_p:
            st.markdown("**Best Hyperparameters**")
            if "DecisionTree" in ml:
                try:
                    dt_est = ml["DecisionTree"].named_steps["model"]
                    keys   = ["max_depth", "min_samples_leaf", "min_samples_split"]
                    params = {k: dt_est.get_params()[k] for k in keys}
                    st.dataframe(pd.DataFrame({
                        "Parameter": list(params.keys()),
                        "Best Value": [str(v) for v in params.values()]
                    }), use_container_width=True, hide_index=True)
                    try:
                        st.caption(f"Fitted tree: depth = {dt_est.get_depth()}, leaves = {dt_est.get_n_leaves()}")
                    except Exception: pass
                except Exception as e:
                    st.warning(f"Could not read DT params: {e}")
            else:
                st.caption("DecisionTree model not found.")
        with col_m2:
            st.markdown("**Test-Set Metrics**")
            _metrics_card("DecisionTree")

        col_tree, col_sc2 = st.columns([1.1, 1])
        with col_tree:
            if "DecisionTree" in ml:
                from sklearn.tree import export_text
                try:
                    dt_est    = ml["DecisionTree"].named_steps["model"]
                    tree_text = export_text(dt_est, feature_names=FEAT_LABELS, max_depth=3)
                    st.markdown("**Top 3 levels of best tree**")
                    st.code(tree_text, language="text")
                    max_d = dt_est.get_params().get("max_depth")
                    if max_d is None or (isinstance(max_d, int) and max_d > 3):
                        st.caption("Full tree truncated at depth 3 for readability.")
                except Exception as e:
                    st.caption(f"Tree display unavailable: {e}")
        with col_sc2:
            if not pred_df.empty and "y_pred_DecisionTree" in pred_df.columns:
                m = _m("DecisionTree")
                fig = _scatter_fig(pred_df["y_true"].values, pred_df["y_pred_DecisionTree"].values,
                                   "Decision Tree", m.get("test_R2", 0), m.get("test_MAE", 0))
                st.pyplot(fig, use_container_width=True); plt.close(fig)

        if "DecisionTree" in ml:
            try:
                dt_est = ml["DecisionTree"].named_steps["model"]
                fig_imp = _importance_fig(dt_est.feature_importances_, FEAT_LABELS,
                                          "Decision Tree — Feature Importances", color="#e15759")
                st.pyplot(fig_imp, use_container_width=True); plt.close(fig_imp)
            except Exception: pass

        st.info(
            "**Interpretation**: The Decision Tree improves on the linear baseline by ~3% R² "
            "(0.45 → 0.48). GridSearchCV constrains depth to prevent extreme overfitting. "
            "Top splits are typically on 30s or 20s power — consistent with highest Pearson r. "
            "The gap between CV R² (~0.52) and test R² (~0.48) reflects modest overfitting: "
            "decision trees produce a piecewise-constant surface that retains high variance "
            "even after pruning."
        )

    # ── 2.4 Random Forest ─────────────────────────────────────────────────────
    with t34:
        st.subheader("2.4 Random Forest")
        st.markdown(
            "**RandomizedSearchCV** (n_iter=30, 5-fold CV) over: "
            "`n_estimators` ∈ {100, 200, 300}, `max_depth` ∈ {5, 10, 20, None}, "
            "`max_features` ∈ {'sqrt', 'log2', 0.5}, `min_samples_leaf` ∈ {2, 5, 10}."
        )
        col_p2, col_m3 = st.columns(2)
        with col_p2:
            st.markdown("**Best Hyperparameters**")
            if "RandomForest" in ml:
                try:
                    rf_est = ml["RandomForest"].named_steps["model"]
                    keys   = ["n_estimators", "max_depth", "max_features", "min_samples_leaf"]
                    params = {k: rf_est.get_params()[k] for k in keys}
                    st.dataframe(pd.DataFrame({
                        "Parameter": list(params.keys()),
                        "Best Value": [str(v) for v in params.values()]
                    }), use_container_width=True, hide_index=True)
                except Exception as e:
                    st.warning(f"Could not read RF params: {e}")
            else:
                st.caption("RandomForest model not found.")
        with col_m3:
            st.markdown("**Test-Set Metrics**")
            _metrics_card("RandomForest")

        col_sc3, col_imp3 = st.columns(2)
        with col_sc3:
            if not pred_df.empty and "y_pred_RandomForest" in pred_df.columns:
                m = _m("RandomForest")
                fig = _scatter_fig(pred_df["y_true"].values, pred_df["y_pred_RandomForest"].values,
                                   "Random Forest", m.get("test_R2", 0), m.get("test_MAE", 0))
                st.pyplot(fig, use_container_width=True); plt.close(fig)
        with col_imp3:
            if "RandomForest" in ml:
                try:
                    rf_est = ml["RandomForest"].named_steps["model"]
                    fig_imp = _importance_fig(rf_est.feature_importances_, FEAT_LABELS,
                                              "Random Forest — Feature Importances", color="#76b7b2")
                    st.pyplot(fig_imp, use_container_width=True); plt.close(fig_imp)
                except Exception: pass

        st.info(
            "**Interpretation**: Random Forest improves on Decision Tree by ~4% R² (0.48→0.52). "
            "Bagging + feature subsampling reduces variance substantially: each tree sees a "
            "different bootstrap sample and random feature subset, averaging out individual noise. "
            "Feature importances are more distributed than the single DT — longest sprint durations "
            "dominate, but W/kg and biometrics contribute meaningfully. Predicted-vs-actual shows "
            "tighter clustering around y=x, especially for mid-range FTP (200–350W)."
        )

    # ── 2.5 XGBoost ───────────────────────────────────────────────────────────
    with t35:
        st.subheader("2.5 Boosted Trees — XGBoost")
        st.markdown(
            "**RandomizedSearchCV** (n_iter=50, 5-fold CV) over 7 params: "
            "`n_estimators` ∈ {100, 200, 400}, `max_depth` ∈ {3, 5, 7}, "
            "`learning_rate` ∈ {0.01, 0.05, 0.1, 0.2}, `subsample/colsample_bytree` ∈ {0.7, 0.85, 1.0}, "
            "`reg_alpha` ∈ {0, 0.1, 1.0}, `reg_lambda` ∈ {1, 5, 10}. "
            "Best config **refitted with early stopping** (patience=20) on a 15% holdout."
        )
        col_p3, col_m4 = st.columns(2)
        with col_p3:
            st.markdown("**Best Hyperparameters**")
            if "XGBoost" in ml:
                try:
                    xgb_m = ml["XGBoost"]
                    keys  = ["n_estimators", "max_depth", "learning_rate",
                             "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"]
                    params = {k: xgb_m.get_params().get(k, "—") for k in keys}
                    st.dataframe(pd.DataFrame({
                        "Parameter": list(params.keys()),
                        "Best Value": [str(v) for v in params.values()]
                    }), use_container_width=True, hide_index=True)
                    st.caption(f"Early stopping best iteration: {getattr(xgb_m, 'best_iteration', '—')}")
                except Exception as e:
                    st.warning(f"Could not read XGBoost params: {e}")
            else:
                st.caption("XGBoost model not found.")
        with col_m4:
            st.markdown("**Test-Set Metrics**")
            _metrics_card("XGBoost")

        col_sc4, col_imp4 = st.columns(2)
        with col_sc4:
            if not pred_df.empty and "y_pred_XGBoost" in pred_df.columns:
                m = _m("XGBoost")
                fig = _scatter_fig(pred_df["y_true"].values, pred_df["y_pred_XGBoost"].values,
                                   "XGBoost", m.get("test_R2", 0), m.get("test_MAE", 0))
                st.pyplot(fig, use_container_width=True); plt.close(fig)
        with col_imp4:
            if "XGBoost" in ml:
                try:
                    xgb_m = ml["XGBoost"]
                    fig_imp = _importance_fig(xgb_m.feature_importances_, FEAT_LABELS,
                                              "XGBoost — Feature Importances (gain)", color="#edc948")
                    st.pyplot(fig_imp, use_container_width=True); plt.close(fig_imp)
                except Exception: pass

        shap_img = FIGURES_DIR / "30_shap_summary_beeswarm.png"
        if shap_img.exists():
            st.markdown("**SHAP Beeswarm** — feature-level impact on predicted log(FTP)")
            st.image(str(shap_img), use_container_width=True)

        st.info(
            "**Interpretation**: XGBoost is competitive with Random Forest (R² ≈ 0.51–0.52). "
            "Gradient boosting sequentially corrects residuals from each tree, effective at "
            "capturing non-linear sprint-decay interactions. Early stopping prevents overfitting "
            "beyond the optimal iteration. SHAP confirms **30s power, W/kg features, and "
            "late decay** drive the strongest positive FTP predictions."
        )

    # ── 2.6 Neural Network ────────────────────────────────────────────────────
    with t36:
        st.subheader("2.6 Neural Network — MLP (Keras 3 / PyTorch backend)")
        st.markdown("Target: log(FTP). Output: linear (regression).")
        col_arch, col_hist = st.columns([1, 1.4])
        with col_arch:
            st.markdown("**Architecture**")
            st.dataframe(pd.DataFrame({
                "Layer": ["Input", "Dense 1", "BatchNorm 1", "Dropout 1",
                          "Dense 2", "BatchNorm 2", "Dropout 2", "Dense 3", "Output"],
                "Config": ["18 features", "128 units, ReLU", "—", "rate = 0.30",
                           "64 units, ReLU", "—", "rate = 0.20", "32 units, ReLU", "1 unit, linear"],
            }), use_container_width=True, hide_index=True)
            st.markdown("**Training config**")
            st.dataframe(pd.DataFrame({
                "Setting": ["Loss", "Optimizer", "Learning rate", "Max epochs", "Batch size",
                            "Early stopping", "LR decay", "Val split"],
                "Value":   ["MSE", "Adam", "0.001", "200", "64",
                            "Patience 20 · restore best weights",
                            "ReduceLROnPlateau ×0.5 (patience 10)", "15% of train set"],
            }), use_container_width=True, hide_index=True)
        with col_hist:
            nn_img = FIGURES_DIR / "25_nn_training_history.png"
            if nn_img.exists():
                st.markdown("**Training history** (loss + val_loss)")
                st.image(str(nn_img), use_container_width=True)
            else:
                st.warning("Training history figure not found.")

        st.markdown("**Test-Set Metrics**")
        nn_row = sb[sb["model"] == "NeuralNet"] if not sb.empty else pd.DataFrame()
        if len(nn_row) > 0:
            _metrics_card("NeuralNet")
        else:
            st.warning(
                "The NeuralNet was trained and weights are saved at "
                "`outputs/models/NeuralNet_sprint_bio.weights.h5`, but test-set metrics "
                "were not persisted in the metrics CSV during the pipeline run. "
                "The NeuralNet is excluded from the comparison table below."
            )
        st.info(
            "**Interpretation**: For tabular data at this scale (n≈3,300 train), MLPs typically "
            "underperform gradient boosting. Tree-based methods partition the monotone "
            "sprint-power feature space naturally, while MLPs require careful architecture tuning. "
            "Training history shows rapid loss reduction in the first 20–50 epochs, followed by "
            "a plateau with early stopping typically triggering around epoch 50–100."
        )

    # ── 2.7 Comparison & Ablation ─────────────────────────────────────────────
    with t37:
        st.subheader("2.7 Model Comparison Summary")

        if not sb.empty:
            sb_ord = sb[sb["model"].isin(MODEL_ORDER)].copy()
            sb_ord["_s"] = sb_ord["model"].apply(
                lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 99
            )
            sb_ord = sb_ord.sort_values("_s").drop("_s", axis=1)

            show = sb_ord[["model", "test_R2", "test_MAE", "test_RMSE", "test_MAPE", "cv_R2"]].rename(
                columns={"model": "Model", "test_R2": "Test R²",
                         "test_MAE": "Test MAE (W)", "test_RMSE": "Test RMSE (W)",
                         "test_MAPE": "MAPE (%)", "cv_R2": "CV R²"})

            def _hl(s):
                if s.name == "Test R²":
                    b = s.max()
                    return ["background-color: #1a6b32" if v == b else "" for v in s]
                if s.name in ("Test MAE (W)", "Test RMSE (W)"):
                    b = s.min()
                    return ["background-color: #1a6b32" if v == b else "" for v in s]
                return [""] * len(s)

            st.dataframe(show.style.apply(_hl).format({
                "Test R²": "{:.3f}", "CV R²": "{:.3f}",
                "Test MAE (W)": "{:.1f}", "Test RMSE (W)": "{:.1f}", "MAPE (%)": "{:.1f}",
            }), use_container_width=True, hide_index=True)
            st.caption("Green = best value in column.  NeuralNet excluded (metrics not persisted).")

            # Bar chart
            models_list = sb_ord["model"].tolist()
            bar_colors  = [MODEL_CLR.get(m, "gray") for m in models_list]
            fig_cmp, axes_cmp = plt.subplots(1, 3, figsize=(13, 4.2))
            fig_cmp.patch.set_facecolor(DARK_BG)
            for ax_c, vals, ylabel, title_c in [
                (axes_cmp[0], sb_ord["test_R2"].values,   "Test R²",       "R²  (↑ better)"),
                (axes_cmp[1], sb_ord["test_MAE"].values,  "Test MAE (W)",  "MAE  (↓ better)"),
                (axes_cmp[2], sb_ord["test_RMSE"].values, "Test RMSE (W)", "RMSE  (↓ better)"),
            ]:
                ax_c.set_facecolor(DARK_AX)
                bars = ax_c.bar(models_list, vals, color=bar_colors, alpha=0.85,
                                edgecolor="white", linewidth=0.5)
                ax_c.set_title(title_c, color="white", fontsize=10, fontweight="bold")
                ax_c.set_ylabel(ylabel, color="white", fontsize=9)
                ax_c.tick_params(axis="x", rotation=30, colors="white", labelsize=8)
                ax_c.tick_params(axis="y", colors="white")
                for sp in ax_c.spines.values(): sp.set_edgecolor("#444")
                for bar, v in zip(bars, vals):
                    ax_c.text(bar.get_x() + bar.get_width() / 2,
                              bar.get_height() + (0.003 if ax_c == axes_cmp[0] else 0.3),
                              f"{v:.2f}" if ax_c == axes_cmp[0] else f"{v:.0f}",
                              ha="center", va="bottom", fontsize=7, color="white")
            plt.suptitle("Model Comparison — sprint_bio Feature Set · Test Set (n=1,430)",
                         fontsize=11, fontweight="bold", color="white", y=1.02)
            plt.tight_layout()
            st.pyplot(fig_cmp, use_container_width=True); plt.close(fig_cmp)

        img_test = FIGURES_DIR / "21_test_performance_comparison.png"
        if img_test.exists():
            st.image(str(img_test), caption="Full test performance comparison (pipeline output)",
                     use_container_width=True)

        st.markdown("---")
        st.subheader("Ablation Study — XGBoost Across Feature Sets")
        st.caption(
            "XGBoost with fixed good hyperparameters, tested on 5 progressively richer feature sets. "
            "sprint_only → sprint_eng → sprint_bio → sprint_bio_v2 → full_submax (ceiling)."
        )
        if not all_metrics.empty:
            xgb_rows = all_metrics[all_metrics["model"] == "XGBoost"].copy()
            xgb_rows = xgb_rows[["feature_set", "test_R2", "test_MAE", "test_RMSE"]].rename(
                columns={"feature_set": "Feature Set", "test_R2": "Test R²",
                         "test_MAE": "Test MAE (W)", "test_RMSE": "Test RMSE (W)"})
            order = ["sprint_only", "sprint_eng", "sprint_bio", "sprint_bio_v2", "full_submax"]
            xgb_rows["_sort"] = xgb_rows["Feature Set"].apply(
                lambda x: order.index(x) if x in order else 99)
            xgb_rows = xgb_rows.sort_values("_sort").drop("_sort", axis=1).reset_index(drop=True)
            st.dataframe(xgb_rows.style.format({
                "Test R²": "{:.3f}", "Test MAE (W)": "{:.1f}", "Test RMSE (W)": "{:.1f}"
            }), use_container_width=True, hide_index=True)

        for img_path, caption in [
            (FIGURES_DIR / "40_ablation_r2_heatmap.png", "R² Heatmap: All Models × Feature Sets"),
            (FIGURES_DIR / "41_ablation_mae_comparison.png", "MAE by Feature Set"),
        ]:
            if img_path.exists():
                st.image(str(img_path), caption=caption, use_container_width=True)

        st.markdown("---")
        st.markdown("### Model Selection Analysis")
        st.markdown("""
**Which model performed best?**

Random Forest and XGBoost tie at **R² ≈ 0.52 and MAE ≈ 33W**, representing a **+15% gain**
over the linear baseline (R² ≈ 0.45). This near-parity between the two ensemble methods is
common for tabular regression at this scale (n ≈ 3,300 train).

**Model trade-offs:**

| Model | Test R² | Interpretability | Best For |
|-------|---------|-----------------|---------|
| Linear / Ridge / Lasso | ~0.45 | Coefficients inspectable | Explaining feature effects |
| Decision Tree | ~0.48 | Tree rules visible | Teaching, rule extraction |
| **Random Forest** | **~0.52** | Aggregate importances | Best accuracy/speed trade-off |
| **XGBoost** | **~0.52** | SHAP (model-agnostic) | Complex non-linearities |
| Neural Network | — | Black box | Needs more data / tuning |

For a **deployed diagnostic tool**, Random Forest is preferred: it maximizes predictive accuracy
and its feature importances provide physiologically meaningful explanations. The +7% R² over the
linear baseline reduces average prediction error from ~35W to ~33W — a clinically relevant
improvement. The SHAP analysis in Tab 4 uses Random Forest as the explainability model.
""")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EXPLAINABILITY & INTERACTIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.title("Explainability & Interactive Prediction")
    st.markdown(
        "**SHAP Analysis**: Random Forest (best R²=0.519) · sprint_bio feature set  \n"
        "**Interactive Prediction**: Select any model, set feature values, get predicted FTP + SHAP waterfall."
    )
    st.markdown("---")

    # ── Load SHAP data ─────────────────────────────────────────────────────────
    EX_FEAT_NAMES = [
        "1s_critical_power", "5s_critical_power", "10s_critical_power",
        "15s_critical_power", "20s_critical_power", "30s_critical_power",
        "fatigue_index", "early_decay", "mid_decay", "late_decay",
        "anaerobic_reserve", "decay_curvature",
        "sprint_wpk_1s", "sprint_wpk_15s", "sprint_wpk_30s",
        "weightkg", "age", "gender_encoded",
    ]
    EX_FEAT_LABELS = [
        "1s power (W)", "5s power (W)", "10s power (W)",
        "15s power (W)", "20s power (W)", "30s power (W)",
        "Fatigue Index (30s/1s)", "Early Decay (5s/1s)", "Mid Decay (15s/5s)",
        "Late Decay (30s/15s)", "Anaerobic Reserve (W)", "Decay Curvature",
        "W/kg (1s)", "W/kg (15s)", "W/kg (30s)",
        "Weight (kg)", "Age", "Gender (0=M)",
    ]

    shap_section, pred_section = st.tabs(["SHAP Explainability", "Interactive Prediction"])

    # ── SHAP SECTION ───────────────────────────────────────────────────────────
    with shap_section:
        with st.spinner("Loading SHAP values (first visit only -- may take a moment)..."):
            shap_vals, feat_names, y_test_w, y_pred_w = load_shap_data()

        if shap_vals is None:
            st.error(
                "Could not load SHAP data. Ensure pipeline has been run and models are saved.\n\n"
                + (f"**Error**: `{feat_names}`" if isinstance(feat_names, str) else "")
            )
        else:
            import shap as shap_lib

            # Key stat tiles
            top_feat_idx  = int(np.abs(shap_vals.values).mean(axis=0).argmax())
            top_feat_name = feat_names[top_feat_idx] if feat_names else "30s_critical_power"
            top_feat_shap = float(np.abs(shap_vals.values).mean(axis=0)[top_feat_idx])
            frac_dom = float(
                np.abs(shap_vals.values[:, top_feat_idx]).sum() /
                np.abs(shap_vals.values).sum() * 100
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Top SHAP feature", top_feat_name.replace("_critical_power", ""))
            c2.metric("Mean |SHAP|", f"{top_feat_shap:.3f} log-W")
            c3.metric("% total SHAP mass", f"{frac_dom:.0f}%")
            c4.metric("Test athletes explained", f"{len(shap_vals):,}")
            st.markdown("---")

            # 3.1.1 BEESWARM
            st.subheader("3.1.1 Summary Plot — Beeswarm")
            st.caption(
                "Each dot = one test-set athlete. "
                "X-axis = SHAP value (impact on log(FTP)). "
                "Color = feature value (red = high, blue = low). "
                "Features are ranked by mean absolute SHAP — top = most important overall."
            )
            fig_bee, ax_bee = plt.subplots(figsize=(10, 7))
            plt.sca(ax_bee)
            shap_lib.plots.beeswarm(shap_vals, max_display=18, show=False)
            fig_bee = plt.gcf()
            fig_bee.suptitle("SHAP Beeswarm — Random Forest · sprint_bio",
                             fontsize=12, fontweight="bold", y=1.01)
            plt.tight_layout()
            st.pyplot(fig_bee, use_container_width=True); plt.close(fig_bee)

            st.info(
                "**Reading the beeswarm**: Each horizontal strip is one feature, ranked by mean |SHAP|. "
                "A dot right of zero means that athlete's value of that feature *increased* the "
                "predicted FTP; left means it *decreased* it. Red = high feature value, blue = low. "
                f"\n\n**30s_critical_power dominates**, accounting for "
                f"**{frac_dom:.0f}% of total absolute SHAP mass** across the test set. "
                "High 30s power (red dots) pushes predictions strongly right (higher FTP); this is "
                "the central thesis finding — 30-second power is the best sprint proxy for FTP. "
                "**Late Decay (30s/15s)** ranks top-5 among all 18 features — the model learns "
                "that athletes whose power holds up in the 15→30s aerobic phase have higher FTP. "
                "**W/kg features** add complementary signal beyond absolute watts, confirming "
                "body-composition normalization matters for predicting aerobic performance."
            )
            st.markdown("---")

            # 3.1.2 BAR CHART
            st.subheader("3.1.2 Feature Importance — Mean |SHAP| Values")
            st.caption(
                "Bar length = mean absolute SHAP value across all 1,430 test athletes. "
                "A longer bar means that feature moves predictions further from the baseline "
                "on average — a model-consistent measure of global importance."
            )
            fig_bar, ax_bar = plt.subplots(figsize=(9, 6))
            plt.sca(ax_bar)
            shap_lib.plots.bar(shap_vals, max_display=18, show=False)
            fig_bar = plt.gcf()
            fig_bar.suptitle("SHAP Feature Importance — Random Forest · sprint_bio",
                             fontsize=12, fontweight="bold", y=1.01)
            plt.tight_layout()
            st.pyplot(fig_bar, use_container_width=True); plt.close(fig_bar)

            mean_abs_shap = np.abs(shap_vals.values).mean(axis=0)
            rank_idx      = np.argsort(mean_abs_shap)[::-1]
            imp_tbl = pd.DataFrame({
                "Rank": range(1, len(feat_names) + 1),
                "Feature": [feat_names[i] for i in rank_idx],
                "Label": [EX_FEAT_LABELS[EX_FEAT_NAMES.index(feat_names[i])]
                          if feat_names[i] in EX_FEAT_NAMES else feat_names[i]
                          for i in rank_idx],
                "Mean |SHAP|": [f"{mean_abs_shap[i]:.4f}" for i in rank_idx],
                "Group": [
                    "Sprint (raw)" if "critical_power" in feat_names[i] else
                    "W/kg" if "wpk" in feat_names[i] else
                    "Biometric" if feat_names[i] in ("weightkg", "age", "gender_encoded") else
                    "Engineered"
                    for i in rank_idx],
            })
            st.dataframe(imp_tbl, use_container_width=True, hide_index=True)
            st.info(
                "**Key observations**: 30s_critical_power has mean |SHAP| ~6× larger than any "
                "other feature — physiologically meaningful because the 30-second effort engages "
                "~40% aerobic metabolism (Gastin 2001). Sprint powers rank 1–3 by importance "
                "(30s > 20s > 15s), naturally recovering the monotonic correlation pattern. "
                "**Late Decay** (30s/15s ratio) is the only engineered feature in the top tier — "
                "it captures whether power holds up in the aerobic phase of the sprint. "
                "W/kg features rank #5–7, confirming they add independent signal beyond absolute watts."
            )
            st.markdown("---")

            # 3.1.3 WATERFALL CASES
            st.subheader("3.1.3 Waterfall Plots — Individual Prediction Decomposition")
            st.markdown(
                "Waterfall plots decompose a **single athlete's prediction** into feature-level "
                "contributions, stepping from the population baseline E[f(x)] to the final "
                "prediction f(x). All values are in **log(FTP)** units."
            )

            residuals = y_pred_w - y_test_w
            under_idx = int(np.argmin(residuals))
            over_idx  = int(np.argmax(residuals))
            exact_idx = int(np.argmin(np.abs(residuals)))

            def _waterfall_fig(idx, title):
                fig_wf, ax_wf = plt.subplots(figsize=(9, 5.5))
                plt.sca(ax_wf)
                shap_lib.plots.waterfall(shap_vals[idx], max_display=10, show=False)
                fig_wf = plt.gcf()
                fig_wf.suptitle(title, fontsize=11, fontweight="bold", y=1.01)
                plt.tight_layout()
                return fig_wf

            wf_t1, wf_t2, wf_t3 = st.tabs([
                "Over-Predicted", "Accurately Predicted", "Under-Predicted"
            ])
            with wf_t1:
                st.markdown(
                    f"**Actual FTP**: {y_test_w[over_idx]:.0f}W  ·  "
                    f"**Predicted FTP**: {y_pred_w[over_idx]:.0f}W  ·  "
                    f"**Error**: +{residuals[over_idx]:.0f}W (model overshot)"
                )
                fig_wf = _waterfall_fig(over_idx, "SHAP Waterfall — Most Over-Predicted Athlete")
                st.pyplot(fig_wf, use_container_width=True); plt.close(fig_wf)
                st.warning(
                    "**Sprint-Dominant / Aerobically Limited**: High 30s power drives the model "
                    "to a high FTP prediction, but actual FTP falls short. This athlete has "
                    "exceptional neuromuscular power that the model interprets as aerobic capacity "
                    "— but their aerobic engine doesn't deliver. **Coaching implication**: prioritize "
                    "aerobic base development — VO₂ kinetics work and sustained threshold volume."
                )
            with wf_t2:
                st.markdown(
                    f"**Actual FTP**: {y_test_w[exact_idx]:.0f}W  ·  "
                    f"**Predicted FTP**: {y_pred_w[exact_idx]:.0f}W  ·  "
                    f"**Error**: {residuals[exact_idx]:+.0f}W (near-perfect)"
                )
                fig_wf = _waterfall_fig(exact_idx, "SHAP Waterfall — Most Accurately Predicted Athlete")
                st.pyplot(fig_wf, use_container_width=True); plt.close(fig_wf)
                st.success(
                    "**Balanced / Aerobically Integrated**: Sprint profile and actual FTP are in "
                    "near-perfect alignment. SHAP contributions are moderate and well-distributed — "
                    "no single feature dominates. This athlete's neuromuscular and aerobic systems "
                    "are integrated proportionally, making sprint power a reliable FTP signal. "
                    "**Coaching implication**: balanced training; race-specific preparation will be most effective."
                )
            with wf_t3:
                st.markdown(
                    f"**Actual FTP**: {y_test_w[under_idx]:.0f}W  ·  "
                    f"**Predicted FTP**: {y_pred_w[under_idx]:.0f}W  ·  "
                    f"**Error**: {residuals[under_idx]:+.0f}W (model undershot)"
                )
                fig_wf = _waterfall_fig(under_idx, "SHAP Waterfall — Most Under-Predicted Athlete")
                st.pyplot(fig_wf, use_container_width=True); plt.close(fig_wf)
                st.info(
                    "**Aerobic Over-Performer**: Actual FTP exceeds what the sprint profile implies. "
                    "This athlete has exceptional VO₂ kinetics and lactate tolerance not captured "
                    "by short-sprint metrics. **Coaching implication**: sprint-specific training "
                    "(short power intervals, neuromuscular activation work) would provide the "
                    "greatest marginal returns."
                )

    # ── INTERACTIVE PREDICTION SECTION ─────────────────────────────────────────
    with pred_section:
        st.subheader("Interactive FTP Prediction")
        st.markdown(
            "Set key feature values below and get a real-time FTP prediction from your chosen model. "
            "Features not shown are set to population median values. "
            "A SHAP waterfall plot is computed for tree-based models."
        )

        ml_models = load_all_sklearn_models()
        pop_medians, feat_names_list = get_pop_medians()

        if not ml_models:
            st.error("No models found in `outputs/models/`. Ensure the pipeline has been run.")
        else:
            model_descriptions = {
                "Linear":       "OLS Linear (R²≈0.45 · MAE≈35W)",
                "Ridge":        "Ridge Regression (R²≈0.45 · MAE≈35W)",
                "Lasso":        "Lasso Regression (R²≈0.45 · MAE≈35W)",
                "DecisionTree": "Decision Tree (R²≈0.48 · MAE≈34W)",
                "RandomForest": "Random Forest (R²≈0.52 · MAE≈33W)",
                "XGBoost":      "XGBoost (R2~0.52 / MAE~33W)",
            }
            available_models = [m for m in MODEL_ORDER if m in ml_models]

            col_in, col_out = st.columns([1, 1], gap="large")

            with col_in:
                st.markdown("**Select Model**")
                model_sel = st.selectbox(
                    "Model",
                    options=available_models,
                    format_func=lambda m: model_descriptions.get(m, m),
                    index=len(available_models) - 1 if available_models else 0,
                    label_visibility="collapsed",
                )

                st.markdown("**Key Sprint Inputs** *(most important features per SHAP)*")
                st.caption("Unset features default to the population median.")

                # Use population medians as slider defaults
                def_30s = int(pop_medians.get("30s_critical_power", 350))
                def_15s = int(pop_medians.get("15s_critical_power", 400))
                def_5s  = int(pop_medians.get("5s_critical_power",  550))
                def_1s  = int(pop_medians.get("1s_critical_power",  750))
                def_10s = int(pop_medians.get("10s_critical_power", 470))
                def_20s = int(pop_medians.get("20s_critical_power", 370))

                p30s = st.slider("30s power (W) — #1 SHAP feature",
                                 min_value=80, max_value=1000, value=def_30s, step=5)
                p15s = st.slider("15s power (W) — #2 SHAP feature",
                                 min_value=80, max_value=1200, value=def_15s, step=5)
                p5s  = st.slider("5s power (W)",
                                 min_value=80, max_value=1800, value=def_5s,  step=10)

                st.markdown("**Biometrics**")
                weight = st.slider("Body weight (kg)", min_value=40, max_value=130,
                                   value=int(pop_medians.get("weightkg", 72)), step=1)
                age    = st.slider("Age (years)", min_value=14, max_value=75,
                                   value=int(pop_medians.get("age", 38)), step=1)
                gender = st.radio("Gender", options=["M", "F"], horizontal=True)

                predict_btn = st.button("Predict FTP", type="primary", use_container_width=True)

            with col_out:
                st.subheader("Prediction Results")

                if not predict_btn:
                    st.info("Set feature values on the left and click **Predict FTP**.")
                else:
                    with st.spinner("Computing..."):
                        # Build raw_inputs: start from pop_medians, override with user values
                        # Pop medians are for engineered features (in feat_names_list order)
                        # We need raw sprint power values to apply SprintFeatureEngineer
                        raw_inputs = {
                            "1s_critical_power":  def_1s,
                            "5s_critical_power":  p5s,
                            "10s_critical_power": def_10s,
                            "15s_critical_power": p15s,
                            "20s_critical_power": def_20s,
                            "30s_critical_power": p30s,
                            "weightkg": weight,
                            "age":      age,
                            "gender":   gender,
                        }

                        try:
                            if feat_names_list:
                                ftp_pred, X_aligned = predict_ftp_any_model(
                                    model_sel, raw_inputs, ml_models, feat_names_list
                                )

                                # Show prediction
                                m_info = _m(model_sel)
                                mae_approx = m_info.get("test_MAE", 33)
                                r2_approx  = m_info.get("test_R2", 0.52)

                                st.metric(
                                    f"Predicted FTP ({model_sel})",
                                    f"{ftp_pred:.0f} W",
                                    help=f"Model: {model_sel} · R²≈{r2_approx:.2f} · MAE≈{mae_approx:.0f}W"
                                )
                                st.caption(
                                    f"**Confidence interval** (±1 MAE): "
                                    f"{ftp_pred - mae_approx:.0f}W – {ftp_pred + mae_approx:.0f}W  |  "
                                    f"Model R²: {r2_approx:.3f}"
                                )

                                # Population context
                                df_cln = load_clean_data()
                                if not df_cln.empty:
                                    pct = float((df_cln["20m_critical_power"] < ftp_pred).mean() * 100)
                                    st.info(
                                        f"A predicted FTP of **{ftp_pred:.0f}W** would rank at the "
                                        f"**{pct:.0f}th percentile** of the 4,768 athletes in this dataset."
                                    )

                                # SHAP waterfall
                                st.markdown("---")
                                st.markdown("#### What Drove This Prediction? (SHAP Waterfall)")
                                is_tree = model_sel in ("DecisionTree", "RandomForest", "XGBoost")
                                if is_tree:
                                    wf_fig = shap_waterfall_for_input(
                                        model_sel, X_aligned, feat_names_list, ml_models
                                    )
                                    if wf_fig:
                                        st.pyplot(wf_fig, use_container_width=True)
                                        plt.close(wf_fig)
                                        st.caption(
                                            "Waterfall plot shows each feature's SHAP contribution "
                                            "to this specific prediction, starting from the average "
                                            "model output (E[f(x)]) and adding/subtracting each "
                                            "feature's effect to reach the final prediction f(x). "
                                            "Values are in log(FTP) units."
                                        )
                                    else:
                                        st.warning("SHAP waterfall unavailable. Ensure `shap` is installed.")
                                else:
                                    # Linear model — show feature contribution via coefficients
                                    st.info(
                                        f"SHAP TreeExplainer is not applicable to linear models. "
                                        f"For {model_sel}, the prediction is a linear combination "
                                        f"of scaled features. Use Random Forest or XGBoost "
                                        f"for SHAP waterfall visualizations."
                                    )
                                    # Show approximate feature contributions via coefficients
                                    try:
                                        model_obj = ml_models[model_sel]
                                        if hasattr(model_obj, "named_steps"):
                                            raw_model = model_obj.named_steps["model"]
                                            scaler    = model_obj.named_steps["scaler"]
                                            X_sc      = scaler.transform(X_aligned)
                                            coefs     = getattr(raw_model, "coef_", None)
                                            if coefs is not None and feat_names_list:
                                                contribs = coefs * X_sc[0]
                                                contrib_df = pd.DataFrame({
                                                    "Feature": feat_names_list,
                                                    "Contribution (log-W)": contribs,
                                                }).sort_values("Contribution (log-W)", ascending=False)
                                                st.markdown("**Feature contributions (coeff × scaled value)**")
                                                st.dataframe(
                                                    contrib_df.head(10).style.format(
                                                        {"Contribution (log-W)": "{:.4f}"}
                                                    ),
                                                    use_container_width=True, hide_index=True
                                                )
                                    except Exception:
                                        pass
                            else:
                                st.error(
                                    "Feature name list not found. "
                                    "Please run `python setup_predictor.py` first."
                                )
                        except FileNotFoundError as e:
                            st.error(f"Model file not found: {e}")
                        except Exception as e:
                            st.error(f"Prediction error: {e}")

            # Feature reference
            with st.expander("Feature index reference", expanded=False):
                ref_df = pd.DataFrame({
                    "Internal name": EX_FEAT_NAMES,
                    "Human label": EX_FEAT_LABELS,
                })
                st.dataframe(ref_df, use_container_width=True, hide_index=True)
