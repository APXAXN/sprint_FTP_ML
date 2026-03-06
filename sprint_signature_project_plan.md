# Sprint Signature: Predicting Endurance Power from Sprint Characteristics

## MSIS 522 — Analytics and Machine Learning | HW1: Complete Data Science Workflow

**Author:** Nathan | University of Washington, Foster School of Business
**Date:** February 2026

---

## Project Thesis

**Can a 30-second sprint tell you how strong you are for a 20-minute climb?**

Two cyclists can both produce 1,200 watts in a 1-second maximal effort. But one fades to 400 watts by 30 seconds while the other holds 700 watts. These two athletes have fundamentally different physiological profiles — and that difference in *how they sprint*, not *how hard they sprint*, encodes critical information about their aerobic endurance capacity.

This project uses the GoldenCheetah OpenData dataset — an open-access collection of 6,500+ anonymized endurance athletes and 2.5 million workouts — to build a regression model that predicts 20-minute peak power (a proxy for Functional Threshold Power / FTP) using only sprint-duration power outputs (≤30 seconds) and basic athlete biometrics. By deliberately excluding medium-duration efforts (5-min, 10-min), the model is forced to learn from the *shape* of the neuromuscular power curve rather than from near-correlated endurance metrics.

The key question is not whether sprint power predicts endurance power (it does, loosely), but **which features of the sprint power curve carry the most endurance signal** — and SHAP analysis reveals that answer at the individual athlete level.

---

## Why This Matters

In practice, a 20-minute all-out test is brutal, risky, and requires significant motivation and pacing skill. Many athletes avoid it. A 30-second maximal sprint, by contrast, is simple, short, and can be performed frequently with minimal fatigue or psychological burden.

If sprint-duration characteristics can reliably estimate endurance capacity, this has applications in:

- **Coaching and talent identification** — screening athletes for endurance potential from short-format testing
- **Training monitoring** — tracking aerobic fitness changes without repeated threshold tests
- **Clinical and rehabilitation settings** — estimating aerobic capacity in populations that can't sustain long efforts

The interactive Streamlit app allows any cyclist to input their sprint test numbers and receive a predicted FTP with a SHAP-driven explanation of which aspects of their power curve shape drove the estimate.

---

## Dataset

**Source:** GoldenCheetah OpenData Project
- GitHub: https://github.com/GoldenCheetah/OpenData
- Kaggle mirror: https://www.kaggle.com/markliversedge/goldencheetah-opendata-athlete-activity-and-mmp
- OSF: https://osf.io/6hfpz/ (DOI 10.17605/OSF.IO/6HFPZ)

**Primary files:**

| File | Description | Granularity |
|------|-------------|-------------|
| `athletes.csv` | One row per athlete — bio info (age, gender, weight) + career peak power at various durations | ~6,500 rows |
| `activities.csv` | One row per activity — per-workout metrics | ~2.5M rows |
| `activities_mmp.csv` | One row per activity — peak power bests from 1s to 36,000s | ~2.5M rows |

**Working dataset strategy:** Use `athletes.csv` as the primary dataset for the career-best model. This gives a clean, one-row-per-athlete structure with peak power values already computed across durations. If the athlete-level dataset is too small for robust model training, supplement with aggregated features from `activities_mmp.csv` (e.g., season bests, percentile ranks).

---

## Prediction Task

**Type:** Regression

**Target variable (y):** 20-minute peak power (watts) — career best per athlete

**Why 20 minutes:** The 20-minute peak power test is the most widely used field estimate of FTP (typically FTP ≈ 95% of 20-min power). It represents sustained aerobic output and is the gold standard for endurance performance benchmarking.

---

## Feature Engineering

### Raw Sprint Features (from athletes.csv peak power columns)

| Feature | Duration | Physiological System |
|---------|----------|---------------------|
| `peak_1s` | 1 second | Pure neuromuscular / peak force |
| `peak_5s` | 5 seconds | Neuromuscular + phosphocreatine |
| `peak_10s` | 10 seconds | Phosphocreatine dominant |
| `peak_15s` | 15 seconds | PCr → glycolytic transition |
| `peak_20s` | 20 seconds | Glycolytic onset |
| `peak_30s` | 30 seconds | Anaerobic capacity (Wingate equivalent) |

### Athlete Biometrics

| Feature | Description |
|---------|-------------|
| `weight` | Body mass (kg) |
| `age` | Athlete age |
| `gender` | Male / Female |

### Engineered Features (the "Sprint Signature")

These features capture the *shape* of the power decay curve, not just absolute values:

| Feature | Formula | What It Captures |
|---------|---------|-----------------|
| `fatigue_index` | peak_30s / peak_1s | Overall power retention — high = "diesel", low = "sprinter" |
| `early_decay` | peak_5s / peak_1s | Immediate power dropoff from peak |
| `mid_decay` | peak_15s / peak_5s | Sustained power through glycolytic transition |
| `late_decay` | peak_30s / peak_15s | Power retention in the anaerobic endurance window |
| `anaerobic_reserve` | peak_1s − peak_30s | Absolute gap between sprint and sustained power |
| `sprint_pwr_to_wt` | peak_5s / weight | Neuromuscular power relative to body size |
| `sustained_pwr_to_wt` | peak_30s / weight | Anaerobic endurance relative to body size |
| `decay_curvature` | (peak_15s − peak_30s) / (peak_1s − peak_15s) | Shape of the decay — linear vs. exponential vs. plateau |

### Explicitly Excluded Features

| Feature | Reason for Exclusion |
|---------|---------------------|
| `peak_1min` | Too close to endurance — trivializes the prediction |
| `peak_2min` | Same — aerobic contribution is substantial at 2 min |
| `peak_5min` | Near-perfect correlate of 20-min power — makes model uninteresting |
| `peak_10min` | Same — model would just learn "10-min ≈ 20-min" |
| `peak_20min` | This IS the target variable |

---

## Modeling Plan

All models use `random_state=42`. Train/test split: 70/30.

### 2.2 — Baseline: Linear Regression
- Standard OLS with all features
- Also run Ridge and Lasso variants to assess multicollinearity among the sprint features
- **Metrics:** MAE, RMSE, R²

### 2.3 — Decision Tree (CART)
- GridSearchCV (5-fold) over:
  - `max_depth`: [3, 5, 7, 10, 15]
  - `min_samples_leaf`: [5, 10, 20, 50]
- Visualize best tree to show decision logic

### 2.4 — Random Forest
- GridSearchCV (5-fold) over:
  - `n_estimators`: [50, 100, 200, 300]
  - `max_depth`: [3, 5, 8, 12]
- Predicted-vs-actual scatter plot

### 2.5 — XGBoost
- GridSearchCV (5-fold) over:
  - `n_estimators`: [50, 100, 200]
  - `max_depth`: [3, 4, 5, 6]
  - `learning_rate`: [0.01, 0.05, 0.1]
- Predicted-vs-actual scatter plot

### 2.6 — Neural Network (MLP)
- Keras Sequential model
- Architecture: Input → 128 (ReLU) → 64 (ReLU) → 32 (ReLU) → 1 (Linear)
- Adam optimizer, MSE loss
- Training history plot (loss curve)
- **Bonus:** Grid search over hidden layer sizes, learning rates, dropout rates

### 2.7 — Model Comparison
- Summary table of all models with MAE, RMSE, R²
- Bar chart comparing RMSE across models

---

## SHAP Analysis Plan (Part 3)

Use the best-performing tree-based model (likely XGBoost or Random Forest).

### Required Plots

1. **Beeswarm summary plot** — Shows which features have the most impact and in which direction. Hypothesis: `fatigue_index` and `sustained_pwr_to_wt` will dominate. The direction of impact tells us whether "diesel" athletes (high fatigue index) systematically have higher 20-min power.

2. **Bar plot of mean |SHAP values|** — Clean feature importance ranking. This answers the central question: which sprint characteristic is the best predictor of endurance?

3. **Waterfall plot for a specific athlete** — Pick an interesting edge case:
   - A "sprinter" with high 1s power but lower-than-expected 20-min power
   - A "diesel" with modest sprint numbers but strong endurance prediction
   - Show how the model pieces together the sprint signature for that individual

### Key Interpretive Questions
- Is fatigue_index (30s/1s) really the strongest predictor, or does a specific timepoint dominate?
- Does body weight interact with sprint shape? (Heavy diesel vs. light sprinter)
- At what point in the 1s→30s curve does endurance information emerge?
- Are there non-linear effects? (e.g., fatigue_index matters more above a certain threshold)

---

## Streamlit App Architecture (Part 4)

### Tab 1 — Executive Summary
- Project thesis: "Can a 30-second sprint predict a 20-minute climb?"
- Plain-language explanation of why this matters for athletes and coaches
- Key finding: which sprint characteristics best predict endurance power
- Dataset overview and approach summary

### Tab 2 — Descriptive Analytics
- Distribution of 20-min peak power across athletes (target variable)
- Sprint power curve visualization: overlay of athlete power curves showing the sprinter-diesel spectrum
- Feature distributions broken down by performance tiers
- Correlation heatmap of all features

### Tab 3 — Model Performance
- Model comparison table (all 5 models, MAE/RMSE/R²)
- Bar chart of RMSE comparison
- Predicted-vs-actual scatter for RF and XGBoost
- Best hyperparameters for each tuned model
- Training history for MLP

### Tab 4 — Explainability & Interactive Prediction
- SHAP beeswarm and bar plots
- **Interactive Sprint Test Predictor:**
  - Sliders for: 1s, 5s, 10s, 15s, 20s, 30s peak power
  - Inputs for: weight, age, gender
  - Engineered features computed automatically from slider values
  - Model selector dropdown
  - Output: Predicted 20-min power (watts) + predicted FTP (95% of 20-min)
  - SHAP waterfall plot for the custom input
  - Text interpretation: "Your sprint signature suggests a [diesel/balanced/sprinter] profile"

---

## Data Quality Considerations

The GoldenCheetah data has known quality issues that must be addressed:

- **Erroneous power values** — Some athletes have clearly impossible peak power numbers (e.g., >3000W at 20 min). Apply physiological plausibility filters.
- **Missing biometric data** — Some athletes lack weight or age. Decide: impute or drop.
- **Non-cycling data** — The dataset includes runners and other endurance athletes. Filter to cycling activities only if possible, or document the mixed-sport nature.
- **Self-selected population** — GoldenCheetah users skew toward serious/competitive cyclists. Acknowledge this in the Executive Summary.
- **Power meter quality** — Different power meters have different accuracy. No way to control for this, but worth noting.

### Cleaning Thresholds (starting points, refine during EDA)

| Metric | Min | Max | Rationale |
|--------|-----|-----|-----------|
| peak_1s (watts) | 200 | 2500 | Below 200 = bad data; above 2500 = world-class track sprinter territory |
| peak_30s (watts) | 100 | 1500 | Physiological plausibility |
| peak_20min (watts) | 50 | 600 | Below 50 = clearly erroneous; above 600 = near pro-tour |
| weight (kg) | 40 | 150 | Plausible athlete range |
| age | 14 | 80 | Reasonable range for active cyclists |

---

## Deliverables Checklist

```
project-root/
├── README.md                    # How to run everything
├── requirements.txt             # All dependencies
├── data/
│   └── athletes_clean.csv       # Cleaned dataset (committed or downloaded via script)
├── notebooks/
│   └── eda_and_modeling.ipynb   # Optional: Jupyter exploration
├── src/
│   ├── data_prep.py             # Download, clean, feature engineering
│   ├── train_models.py          # Train all 5 models, save with joblib
│   ├── shap_analysis.py         # Generate SHAP plots
│   └── utils.py                 # Shared functions
├── models/
│   ├── linear_regression.joblib
│   ├── decision_tree.joblib
│   ├── random_forest.joblib
│   ├── xgboost_model.joblib
│   └── mlp_model.keras
├── plots/
│   ├── target_distribution.png
│   ├── power_curve_profiles.png
│   ├── feature_distributions.png
│   ├── correlation_heatmap.png
│   ├── shap_beeswarm.png
│   ├── shap_bar.png
│   └── shap_waterfall.png
├── streamlit_app.py             # Main Streamlit app with 4 tabs
└── .streamlit/
    └── config.toml              # Streamlit theme config
```

---

## Claude Code Workflow

### Session 1: Data Acquisition & EDA
```
"Download the GoldenCheetah athletes.csv from Kaggle/S3. 
 Clean it using the thresholds in the plan doc. 
 Engineer the sprint signature features. 
 Generate all Part 1 visualizations and save as PNGs."
```

### Session 2: Model Training Pipeline
```
"Train all 5 models per the plan doc specs. 
 Use GridSearchCV with the specified parameter grids. 
 Save all models with joblib. 
 Generate predicted-vs-actual plots and the comparison table."
```

### Session 3: SHAP & Streamlit
```
"Run SHAP analysis on the best tree model. 
 Generate all 3 required SHAP plots. 
 Build the Streamlit app with all 4 tabs per the plan doc. 
 Include the interactive sprint test predictor with SHAP waterfall."
```

### Session 4: Polish & Deploy
```
"Review the Streamlit app for polish. 
 Write the README. Generate requirements.txt. 
 Push to GitHub and deploy to Streamlit Community Cloud."
```

---

## Working Title Options

- **Sprint Signature** — Predicting Endurance Power from Sprint Characteristics
- **The 30-Second FTP Test** — What Your Sprint Says About Your Climb
- **Diesel or Dynamite?** — Classifying Endurance Potential from Neuromuscular Power Curves

---

*Ready to build.*
