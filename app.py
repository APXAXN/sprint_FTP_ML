"""
app.py
Sprint FTP ML — Streamlit Application
MSIS 522 | UW Foster School of Business

Run:
    streamlit run app.py

Pages (sidebar nav):
    1. 🏆 The Thesis         — physiological switch hero chart + finding
    2. 📊 Model Results      — model comparison + ablation study
    3. 🔬 Athlete Profiler   — interactive FTP predictor + shift score profile
    4. 🧬 Methods & Physiology — methods + physiological narrative

Requires setup_predictor.py to have been run first.
"""

import os, warnings
os.environ["KERAS_BACKEND"] = "torch"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
MODELS_DIR   = BASE_DIR / "outputs" / "models"
RESULTS_DIR  = BASE_DIR / "outputs" / "results"
FIGURES_DIR  = BASE_DIR / "outputs" / "figures"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sprint FTP ML",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar nav ───────────────────────────────────────────────────────────────
st.sidebar.title("⚡ Sprint FTP ML")
st.sidebar.caption("MSIS 522 · UW Foster School of Business")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏆 The Thesis", "📊 Model Results", "🔬 Athlete Profiler", "🧬 Methods & Physiology"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "**Data**: GoldenCheetah OpenData (n = 4,768)  \n"
    "**Best model**: XGBoost  \n"
    "**GitHub**: [sprint_FTP_ML](https://github.com/APXAXN/sprint_FTP_ML)"
)


# ══════════════════════════════════════════════════════════════════════════════
# Cached resource loaders
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_sprint_model():
    model  = joblib.load(MODELS_DIR / "XGBoost_sprint_bio.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler_sprint_bio.joblib")
    feats  = joblib.load(MODELS_DIR / "feature_names_sprint_bio.joblib")
    return model, scaler, feats


@st.cache_resource
def load_2m_model():
    model  = joblib.load(MODELS_DIR / "XGBoost_sprint_bio_plus_2m.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler_sprint_bio_plus_2m.joblib")
    feats  = joblib.load(MODELS_DIR / "feature_names_sprint_bio_plus_2m.joblib")
    return model, scaler, feats


@st.cache_data
def load_metrics():
    dfs = []
    for fs in ["sprint_only", "sprint_eng", "sprint_bio", "sprint_bio_v2", "full_submax"]:
        p = RESULTS_DIR / f"metrics_{fs}.csv"
        if p.exists():
            df = pd.read_csv(p)
            df.insert(0, "feature_set", fs)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


@st.cache_data
def load_submax_threshold():
    p = RESULTS_DIR / "submax_threshold.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def predict_ftp(raw_inputs: dict, model, scaler, feat_names):
    """
    Apply SprintFeatureEngineer to a 1-row dict, select+scale features,
    predict log(FTP), and return watts via exp().
    """
    import sys; sys.path.insert(0, str(BASE_DIR))
    from pipeline.features import SprintFeatureEngineer

    df_in = pd.DataFrame([raw_inputs])
    eng   = SprintFeatureEngineer()
    df_eng = eng.transform(df_in)

    # Align to training feature order; fill missing with median-ish defaults
    X = np.zeros((1, len(feat_names)))
    for i, col in enumerate(feat_names):
        if col in df_eng.columns:
            X[0, i] = float(df_eng[col].iloc[0])

    X_sc   = scaler.transform(X)
    log_pred = model.predict(X_sc)[0]
    return float(np.exp(log_pred))


def shap_waterfall(model, scaler, feat_names, raw_inputs: dict):
    """Return a matplotlib Figure of the SHAP waterfall for one prediction."""
    try:
        import shap
        from pipeline.features import SprintFeatureEngineer

        df_in  = pd.DataFrame([raw_inputs])
        eng    = SprintFeatureEngineer()
        df_eng = eng.transform(df_in)

        X = np.zeros((1, len(feat_names)))
        for i, col in enumerate(feat_names):
            if col in df_eng.columns:
                X[0, i] = float(df_eng[col].iloc[0])
        X_sc = scaler.transform(X)

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer(X_sc)

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(shap_values[0], max_display=12, show=False)
        fig = plt.gcf()
        fig.suptitle("SHAP Feature Contributions", fontsize=11, fontweight="bold")
        plt.tight_layout()
        return fig
    except Exception as e:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — THE THESIS
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏆 The Thesis":
    st.title("Can a 30-Second Sprint Predict Your 20-Minute FTP?")
    st.markdown(
        "##### A machine learning study of 4,768 cyclists · MSIS 522 · UW Foster School of Business"
    )
    st.markdown("---")

    # Key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="30s Sprint (sprint_bio model)",
            value="R² = 0.516",
            delta="MAE ≈ ±33.7W",
            delta_color="off",
            help="XGBoost trained on 6 sprint features + 9 engineered ratios + biometrics"
        )
    with col2:
        st.metric(
            label="⭐ 2-Minute Sub-Max Effort",
            value="R² = 0.704",
            delta="+18.8% vs sprint alone",
            delta_color="normal",
            help="Same XGBoost + sprint features, augmented with 2-min critical power"
        )
    with col3:
        st.metric(
            label="30-Minute Effort (ceiling reference)",
            value="R² = 0.943",
            delta="MAE ≈ ±10.6W",
            delta_color="off",
            help="Full sub-max feature set — approaching the theoretical ceiling"
        )

    st.markdown("---")

    # Hero chart
    switch_img = FIGURES_DIR / "60_physiological_switch.png"
    if switch_img.exists():
        st.image(str(switch_img), use_container_width=True)
    else:
        st.warning("Physiological switch chart not found. Run pipeline/plots.py first.")

    st.markdown("---")

    # Thesis narrative
    st.markdown("## The Finding")
    st.info(
        "**A 2-minute sub-maximal effort is where the physiological switch occurs.** "
        "R² crosses the 0.70 threshold — the point at which aerobic metabolism becomes "
        "the dominant predictor of 20-minute FTP. Sprint data alone captures neuromuscular "
        "power output, but not the aerobic engine that sustains near-threshold power."
    )

    st.markdown(
        """
### The Thesis

This study investigates whether the substantial increase in FTP prediction from a
30-second sprint effort to a 2-minute submaximal effort represents a **quantifiable
physiological transition zone** in cycling performance.

The near **20% improvement in predictive accuracy** signals a shift away from performance
dominated by anaerobic explosiveness and neuromuscular output — toward traits more closely
aligned with threshold performance: **aerobic contribution, metabolic stability, and
fatigue resistance**.

By measuring the magnitude of this predictive shift for each athlete, this research
proposes a novel way to **profile individuals along a sprint-to-endurance continuum**.
This quantified shift can inform training design — helping coaches and athletes target
whether the priority is aerobic durability, sustained power production, or integration
between the two systems.

---

### What Changes at 2 Minutes?

The science is clear: around **75 seconds**, aerobic and anaerobic ATP contributions are
roughly equal (Gastin 2001). By **2 minutes**, aerobic metabolism supplies approximately
65% of total energy (Medbø & Tabata). VO₂ kinetics — how quickly the aerobic system
"engages" — becomes the dominant constraint, not peak neuromuscular power.

A 30-second sprint is primarily shaped by **PCr stores, rapid glycolysis, and
force-velocity optimization** at high cadences. These traits are only partially
correlated with the sustained oxidative capacity that governs 20-minute FTP power.
The 2-minute effort sits within the same **critical power / W′ neighborhood** as
a 20-minute TT — making it a much closer physiological proxy.
        """
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Model Results":
    st.title("Model Results")
    st.markdown("All metrics computed on the held-out test set (n = 1,430; stratified 70/30 split by gender).")
    st.markdown("---")

    tab1, tab2 = st.tabs(["Primary Models — sprint_bio", "Ablation Study"])

    with tab1:
        st.subheader("All 7 Models · sprint_bio Feature Set · log(FTP) target")
        st.caption(
            "sprint_bio = 6 raw sprint powers (1s–30s) + 9 engineered decay/ratio features "
            "+ 3 biometric features (weight, age, gender). All metrics back-transformed to watts."
        )
        primary_csv = RESULTS_DIR / "metrics_sprint_bio.csv"
        if primary_csv.exists():
            df_pri = pd.read_csv(primary_csv)
            # Select + rename columns for display
            show_cols = {
                "model": "Model",
                "test_R2": "Test R²",
                "test_MAE": "Test MAE (W)",
                "test_RMSE": "Test RMSE (W)",
                "test_MAPE": "Test MAPE (%)",
                "cv_R2": "CV R²",
                "cv_MAE": "CV MAE (log-W)*",
            }
            disp = df_pri[[c for c in show_cols if c in df_pri.columns]].rename(columns=show_cols)
            disp = disp.sort_values("Test R²", ascending=False).reset_index(drop=True)

            def highlight_best(s):
                is_r2  = s.name == "Test R²"
                is_mae = s.name in ("Test MAE (W)", "Test RMSE (W)")
                if is_r2:
                    best = s.max()
                    return ["background-color: #d4edda" if v == best else "" for v in s]
                if is_mae:
                    best = s.min()
                    return ["background-color: #d4edda" if v == best else "" for v in s]
                return [""] * len(s)

            st.dataframe(
                disp.style
                    .apply(highlight_best)
                    .format({
                        "Test R²": "{:.3f}",
                        "CV R²":   "{:.3f}",
                        "Test MAE (W)": "{:.1f}",
                        "Test RMSE (W)": "{:.1f}",
                        "Test MAPE (%)": "{:.1f}",
                        "CV MAE (log-W)*": "{:.4f}",
                    }),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "\\* CV MAE is in log-W units (internal training signal). "
                "Green = best value in column."
            )

        st.markdown("---")
        img_test = FIGURES_DIR / "21_test_performance_comparison.png"
        if img_test.exists():
            st.image(str(img_test), caption="Test Set Performance: MAE, RMSE, R²", use_container_width=True)

        img_pva = FIGURES_DIR / "22_predicted_vs_actual.png"
        if img_pva.exists():
            st.image(str(img_pva), caption="Predicted vs. Actual FTP (watts)", use_container_width=True)

    with tab2:
        st.subheader("XGBoost · All Feature Sets · log(FTP) target")
        st.caption(
            "Ablation study: how much does each feature group contribute? "
            "Sprint_only → sprint_eng → sprint_bio → sprint_bio_v2 → full_submax (ceiling)."
        )
        all_metrics = load_metrics()
        if not all_metrics.empty:
            xgb_rows = all_metrics[all_metrics["model"] == "XGBoost"].copy()
            xgb_rows = xgb_rows[["feature_set", "test_R2", "test_MAE", "test_RMSE"]].rename(columns={
                "feature_set": "Feature Set",
                "test_R2": "Test R²",
                "test_MAE": "Test MAE (W)",
                "test_RMSE": "Test RMSE (W)",
            })
            order = ["sprint_only", "sprint_eng", "sprint_bio", "sprint_bio_v2", "full_submax"]
            xgb_rows["_sort"] = xgb_rows["Feature Set"].apply(
                lambda x: order.index(x) if x in order else 99
            )
            xgb_rows = xgb_rows.sort_values("_sort").drop("_sort", axis=1).reset_index(drop=True)
            st.dataframe(
                xgb_rows.style.format({
                    "Test R²": "{:.3f}", "Test MAE (W)": "{:.1f}", "Test RMSE (W)": "{:.1f}"
                }),
                use_container_width=True, hide_index=True,
            )

        st.markdown("---")
        img_heat = FIGURES_DIR / "40_ablation_r2_heatmap.png"
        if img_heat.exists():
            st.image(str(img_heat), caption="R² Heatmap: All Models × Feature Sets", use_container_width=True)

        img_mae = FIGURES_DIR / "41_ablation_mae_comparison.png"
        if img_mae.exists():
            st.image(str(img_mae), caption="MAE by Feature Set", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ATHLETE PROFILER
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔬 Athlete Profiler":
    st.title("🔬 Athlete Profiler")
    st.markdown(
        "Enter sprint power outputs and a 2-minute sub-max effort to receive "
        "a predicted FTP and a physiological profile along the "
        "**sprint-to-endurance continuum**."
    )
    st.markdown("---")

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.subheader("Sprint Inputs")
        st.caption("Enter your best power for each effort duration (watts)")

        p1s  = st.slider("1s peak power",   min_value=200, max_value=2000, value=800, step=10, key="p1")
        p5s  = st.slider("5s power",        min_value=100, max_value=1800, value=650, step=10, key="p5")
        p10s = st.slider("10s power",       min_value=100, max_value=1500, value=550, step=10, key="p10")
        p15s = st.slider("15s power",       min_value=100, max_value=1200, value=470, step=10, key="p15")
        p20s = st.slider("20s power",       min_value=100, max_value=1100, value=420, step=10, key="p20")
        p30s = st.slider("30s power",       min_value=100, max_value=1000, value=370, step=10, key="p30")

        st.markdown("---")
        st.subheader("Biometrics")
        weight = st.slider("Body weight (kg)", min_value=40, max_value=130, value=72, step=1)
        age    = st.slider("Age",              min_value=14, max_value=75,  value=35, step=1)
        gender = st.radio("Gender", options=["M", "F"], horizontal=True)

        st.markdown("---")
        st.subheader("2-Minute Sub-Max Effort")
        st.caption(
            "Your best sustainable power for 2 minutes (not an all-out sprint — "
            "a hard but controlled effort at roughly your 2-min max)"
        )
        p2m = st.slider("2-min power (W)", min_value=80, max_value=600, value=270, step=5)

        st.markdown("")
        run_btn = st.button("🔬 Profile Me", type="primary", use_container_width=True)

    with col_out:
        st.subheader("Your Physiological Profile")
        if not run_btn:
            st.info("Enter your data on the left and click **Profile Me** to see your results.")
        else:
            with st.spinner("Computing predictions..."):
                raw_inputs = {
                    "1s_critical_power":  p1s,
                    "5s_critical_power":  p5s,
                    "10s_critical_power": p10s,
                    "15s_critical_power": p15s,
                    "20s_critical_power": p20s,
                    "30s_critical_power": p30s,
                    "weightkg":           weight,
                    "age":                age,
                    "gender":             gender,
                    # 2m not included here — for sprint model only
                }
                raw_inputs_2m = dict(raw_inputs)
                raw_inputs_2m["2m_critical_power"] = p2m

                try:
                    model_s, scaler_s, feats_s = load_sprint_model()
                    model_2, scaler_2, feats_2  = load_2m_model()

                    ftp_sprint = predict_ftp(raw_inputs,    model_s, scaler_s, feats_s)
                    ftp_2min   = predict_ftp(raw_inputs_2m, model_2, scaler_2, feats_2)
                    shift_w    = ftp_2min - ftp_sprint

                    # ── Side-by-side predictions ─────────────────────────────
                    mc1, mc2 = st.columns(2)
                    with mc1:
                        st.metric(
                            "Sprint-Only FTP Estimate",
                            f"{ftp_sprint:.0f} W",
                            help="XGBoost · sprint_bio features · R²=0.52 · ±33.5W",
                        )
                        st.caption("Model: sprint_bio (R² = 0.52, ±33.5W)")
                    with mc2:
                        st.metric(
                            "+ 2-Min FTP Estimate",
                            f"{ftp_2min:.0f} W",
                            delta=f"{shift_w:+.0f} W vs sprint-only",
                            delta_color="normal" if shift_w >= 0 else "inverse",
                            help="XGBoost · sprint_bio + 2m features · R²=0.74 · ±23.9W",
                        )
                        st.caption("Model: sprint_bio + 2m (R² = 0.74, ±23.9W)")

                    st.markdown("---")

                    # ── Physiological Profile ────────────────────────────────
                    st.markdown("#### 🧭 Your Physiological Profile")

                    SHIFT_THR = 20   # ±20W threshold for profile assignment

                    if shift_w > SHIFT_THR:
                        profile_label = "Aerobic Underperformer (Sprint-Dominant)"
                        profile_color = "orange"
                        profile_icon  = "🟠"
                        interpretation = (
                            "**Your sprint predicts less FTP than your 2-minute effort reveals.** "
                            "Your aerobic engine is stronger than your sprint power alone suggests — "
                            "your neuromuscular output may be disproportionately high relative to "
                            "your aerobic contribution.\n\n"
                            "**Training priority:** You have aerobic capacity to leverage. Focus on "
                            "integrating sprint power into sustained threshold efforts. Sub-threshold "
                            "volume and tempo work will help your aerobic system express itself during "
                            "longer race efforts."
                        )
                    elif shift_w < -SHIFT_THR:
                        profile_label = "Sprint-Dominant / Aerobically Limited"
                        profile_color = "red"
                        profile_icon  = "🔴"
                        interpretation = (
                            "**Your sprint projects a higher FTP than your 2-minute effort delivers.** "
                            "Your neuromuscular and anaerobic power is high, but your aerobic system "
                            "may not be engaging at the same level — possibly limited by VO₂ kinetics, "
                            "lactate clearance, or aerobic base.\n\n"
                            "**Training priority:** Aerobic durability. Focus on VO₂ kinetics work "
                            "(4–8 min intervals at ~105% FTP), sustained threshold volume, and "
                            "high-aerobic-flux sessions. Your ceiling is limited by aerobic engagement, "
                            "not by your peak power."
                        )
                    else:
                        profile_label = "Balanced / Aerobically Integrated"
                        profile_color = "green"
                        profile_icon  = "🟢"
                        interpretation = (
                            "**Your sprint and sub-max power are well-aligned.** "
                            "Both neuromuscular and aerobic systems contribute proportionally — "
                            "your sprint power is a reliable signal of your aerobic capacity.\n\n"
                            "**Training priority:** Maintain integration. Your systems are balanced; "
                            "targeted specificity work (race simulations, pacing practice near FTP, "
                            "peak-power maintenance) will be most effective for continued gains."
                        )

                    # Profile badge
                    st.markdown(f"**{profile_icon} {profile_label}**")
                    st.markdown(
                        f"*Sprint→2min shift: {shift_w:+.0f} W*  "
                        f"({'sprint dominant' if shift_w < -SHIFT_THR else 'aerobic dominant' if shift_w > SHIFT_THR else 'balanced'})"
                    )
                    st.info(interpretation)

                    # ── Shift gauge ──────────────────────────────────────────
                    st.markdown("---")
                    st.markdown("#### Sprint→Aerobic Shift Score")
                    fig_gauge, ax_g = plt.subplots(figsize=(7, 1.5))
                    cmap = plt.cm.RdYlGn
                    clipped = max(-80, min(80, shift_w))
                    norm_val = (clipped + 80) / 160   # 0→1 scale

                    ax_g.barh([0], [160], color="lightgray", height=0.6, left=-80)
                    ax_g.barh([0], [clipped], color=cmap(norm_val), height=0.6, alpha=0.85)
                    ax_g.axvline(0, color="black", linewidth=1.5)
                    ax_g.axvline(-SHIFT_THR, color="gray", linewidth=1, linestyle="--", alpha=0.6)
                    ax_g.axvline(+SHIFT_THR, color="gray", linewidth=1, linestyle="--", alpha=0.6)
                    ax_g.scatter([clipped], [0], color="black", s=100, zorder=5)
                    ax_g.text(-80, -0.55, "Sprint-\nDominant", ha="left",   fontsize=7, color="tomato")
                    ax_g.text(  0, -0.55, "Balanced",          ha="center", fontsize=7, color="gray")
                    ax_g.text( 80, -0.55, "Aerobic-\nDominant", ha="right", fontsize=7, color="mediumseagreen")
                    ax_g.set_xlim(-85, 85)
                    ax_g.set_ylim(-0.9, 0.7)
                    ax_g.axis("off")
                    ax_g.set_title(f"Shift = {shift_w:+.0f} W  (2-min estimate − sprint estimate)",
                                   fontsize=9, pad=4)
                    st.pyplot(fig_gauge, use_container_width=True)
                    plt.close(fig_gauge)

                    # ── SHAP waterfall ────────────────────────────────────────
                    st.markdown("---")
                    st.markdown("#### What Drove This Prediction? (SHAP — 2-min model)")
                    shap_fig = shap_waterfall(model_2, scaler_2, feats_2, raw_inputs_2m)
                    if shap_fig:
                        st.pyplot(shap_fig, use_container_width=True)
                        plt.close(shap_fig)
                    else:
                        st.caption("SHAP waterfall unavailable (install the `shap` package).")

                except FileNotFoundError as e:
                    st.error(
                        f"Model file not found: {e}\n\n"
                        "Please run `python setup_predictor.py` first."
                    )
                except Exception as e:
                    st.error(f"Prediction error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — METHODS & PHYSIOLOGY
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🧬 Methods & Physiology":
    st.title("Methods & Physiology")
    st.markdown("---")

    tab_meth, tab_phys = st.tabs(["📐 Methods", "🔬 The Physiology"])

    with tab_meth:
        st.markdown("""
## Data

**Source**: [GoldenCheetah OpenData](https://github.com/GoldenCheetah/OpenData) via Kaggle
(`markliversedge/goldencheetah-opendata-athlete-activity-and-mmp`)

**Initial dataset**: 6,043 athletes
**After quality filters** (power plausibility, weight, age): **4,768 clean athletes**

Quality filters applied:
| Column | Filter |
|--------|--------|
| 1s_critical_power | 200–2500 W |
| 20m_critical_power (target) | 50–600 W |
| weightkg | 40–150 kg |
| age | 14–80 years |

**Gender**: 97% male · Stratified split ensures ~3% female in both train and test sets.

---

## Features

### Sprint Power (raw)
Six critical power columns from the GoldenCheetah MMP (Mean Maximal Power) curves:
`1s`, `5s`, `10s`, `15s`, `20s`, `30s` critical power (watts)

### Engineered Features (9 derived)
| Feature | Formula | Physiological meaning |
|---------|---------|----------------------|
| `fatigue_index` | 30s / 1s | How much sprint power survives 30s |
| `early_decay` | 5s / 1s | First 5-second drop-off |
| `mid_decay` | 15s / 5s | Mid-sprint decay |
| `late_decay` | 30s / 15s | Final-phase decay (most aerobic-influenced) |
| `anaerobic_reserve` | 1s − 30s | Absolute watts lost across sprint |
| `decay_curvature` | log(1s) − 2·log(15s) + log(30s) | Shape of power-duration curve |
| `sprint_wpk_1s` | 1s / weightkg | Peak sprint power-to-weight |
| `sprint_wpk_15s` | 15s / weightkg | Mid-sprint W/kg |
| `sprint_wpk_30s` | 30s / weightkg | Sustained-sprint W/kg |

### Biometrics
`weightkg`, `age`, `gender_encoded` (M=0, F=1)

---

## Models

Seven algorithms tested on the primary feature set (`sprint_bio`):

| Model | Tuning |
|-------|--------|
| OLS Linear | None |
| Ridge | RidgeCV (built-in, 6 alphas) |
| Lasso | LassoCV (built-in) |
| Decision Tree | GridSearchCV |
| **Random Forest** | RandomizedSearchCV (n_iter=30) |
| **XGBoost** | RandomizedSearchCV (n_iter=50) ← best |
| Keras NN | EarlyStopping (patience=20), PyTorch backend |

**Target transformation**: `log(FTP)` during training → `exp()` back-transform for all reported metrics.
All metrics are therefore in watts and directly interpretable.

---

## Validation

- **Split**: Stratified 70/30 (train/test) on gender label
- **Train set**: n = 3,335 · **Test set**: n = 1,430
- **Cross-validation**: 5-fold CV on train set only
- **Scaler**: RobustScaler (fit on train, applied to both) — robust to power outliers

---

## Sub-Max Threshold Experiment

Separate analysis (`submax_threshold.py`) testing each sub-max duration
(2m, 3m, 5m, 8m, 10m, 30m) alone and in combination with sprint features.
XGBoost with fixed good hyperparameters (no re-search) for fair duration comparison.
""")

    with tab_phys:
        st.markdown("""
## Why Does 2 Minutes Change Everything?

The 20% jump in prediction accuracy at 2 minutes is not a statistical artifact —
it reflects a real and well-documented **physiological transition** in how the body
produces power.

---

### The Energy System Crossover

Exercise physiology has long recognized that aerobic and anaerobic energy systems
are not sequential — they operate simultaneously from the first pedal stroke. What
changes with duration is **which system is rate-limiting** and **which fatigue
mechanisms dominate**.

| Duration | Dominant System | Relevance to 20-min FTP |
|----------|----------------|------------------------|
| 0–30s | PCr breakdown + rapid glycolysis · aerobic rising but secondary | Sprint-specific; neuromuscular and force-velocity traits dominate |
| 30–120s | Aerobic fraction rising rapidly · ~75s crossover (aerobic ≈ anaerobic) | VO₂ kinetics and oxidative engagement become the key constraints |
| 2–20min | Oxidative phosphorylation dominates · CP/W′ framework governs tolerance | Shared physiological space with 20-min FTP |

**Quantitative anchors** (Medbø & Tabata; Gastin 2001):
- Aerobic contribution at **30 seconds** ≈ 40%
- Aerobic contribution at **1 minute** ≈ 50%
- Aerobic contribution at **2 minutes** ≈ **65%** ← the switch

---

### VO₂ Kinetics: The Physiological Bridge

When you start a hard effort, VO₂ doesn't instantly reach its steady-state value —
it follows exponential kinetics (Poole & Jones 2012). An **oxygen deficit** accumulates
while the aerobic system "catches up," filled in the short term by PCr and glycolysis.

**Faster VO₂ kinetics → smaller oxygen deficit → less reliance on substrate-level phosphorylation → better endurance tolerance.**

In a 30-second sprint, VO₂ never fully "takes over" — the effort ends before the
aerobic system is fully expressed. In a 2-minute effort, VO₂ kinetics speed is now
a *major performance constraint* — and it's directly related to what limits 20-minute
FTP power.

---

### Critical Power / W′: Why 2-min and 20-min Are Neighbors

The **Critical Power (CP)** framework (Poole et al 2016) describes a hyperbolic
power-duration relationship where CP separates intensities that can be physiologically
sustained from those that cannot. Above CP, VO₂ drifts toward VO₂max and a finite
work capacity (W′) is progressively depleted.

Importantly, time-to-exhaustion *at CP* is approximately 20–30 minutes — placing the
20-minute FTP test squarely in the CP neighborhood. A 2-minute effort at near-maximal
intensity also depends heavily on how much CP and W′ the athlete possesses.

**By contrast**, a 30-second sprint sits at the extreme left of the power-duration
curve — dominated by sprint-specific neuromuscular traits and rapid metabolite
disruption (Pi accumulation, Ca²⁺ handling impairment, pH disturbance) that are
largely irrelevant to 20-minute steady-state power.

---

### The Rigorous Phrasing

> "The improved prediction of 20-min FTP from a 2-min sub-max effort (vs. a 30-s sprint)
> is consistent with a shift in **dominant performance determinants**: from outcomes
> primarily governed by neuromuscular power and rapid substrate-level phosphorylation
> (PCr + glycolysis) to outcomes that increasingly depend on **oxidative metabolism
> engagement, VO₂ kinetics/oxygen deficit, and fatigue resistance near the critical
> power intensity domain**."

This is not a claim that the 30-second sprint is "useless" — it still explains
~52% of the variance in FTP. It is that the sprint embeds *more sprint-specific
variance* (force-velocity optimization, neuromuscular coordination, acute metabolite
tolerance) that is less relevant to sustained 20-minute power than what a 2-minute
effort measures.

---

### Key References

- Medbø & Tabata (1989). *Med Sci Sports Exerc.* [PubMed](https://pubmed.ncbi.nlm.nih.gov/2600022/)
- Gastin (2001). *Sports Med.* [PubMed](https://pubmed.ncbi.nlm.nih.gov/11547894/)
- Poole & Jones (2012). *Compr Physiol.* [PubMed](https://pubmed.ncbi.nlm.nih.gov/23798293/)
- Poole et al (2016). *J Physiol.* [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5070974/)
- Bogdanis et al (1995). *J Physiol.* [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC1157744/)
- Karsten et al (2021). *Front Physiol.* [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7862708/)
- Borszcz et al (2020). *Int J Sports Med.* [Thieme](https://www.thieme-connect.com/products/ejournals/pdf/10.1055/a-1018-1965.pdf)
""")
