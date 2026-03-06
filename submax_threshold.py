"""
submax_threshold.py
At what sub-max test duration does R² first exceed 0.7 when predicting 20-min FTP?
Three model configurations tested per duration:
  A: sub-max only
  B: sub-max + biometrics
  C: sub-max + full sprint_bio features
"""

import os, warnings
os.environ["KERAS_BACKEND"] = "torch"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler

import sys
sys.path.insert(0, "/Users/nathanfitzgerald/Sprint_FTP_ML")

from pipeline.features import SprintFeatureEngineer
from pipeline.config import DATA_PATH, RANDOM_STATE, TEST_SIZE, FEATURE_SETS
from pipeline.evaluate import compute_metrics

# ── Load & engineer ──────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
eng = SprintFeatureEngineer()
df = eng.transform(df)

TARGET = "20m_critical_power"

SUBMAX_COLS = [
    "2m_critical_power",
    "3m_critical_power",
    "5m_critical_power",
    "8m_critical_power",
    "10m_critical_power",
    "30m_critical_power",
]

BIOMETRIC_COLS = ["age", "weightkg", "gender_encoded"]
SPRINT_BIO_COLS = FEATURE_SETS["sprint_bio"]

# ── Stratified split (same as pipeline) ──────────────────────────────────────
df_model = df[[TARGET] + SUBMAX_COLS + SPRINT_BIO_COLS + BIOMETRIC_COLS].dropna()
strat_col = df_model["gender_encoded"].values
y_raw = df_model[TARGET].values

sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
train_idx, test_idx = next(sss.split(df_model.values, strat_col))

y_train_log = np.log(y_raw[train_idx])
y_test_w    = y_raw[test_idx]

# ── XGBoost factory ───────────────────────────────────────────────────────────
def fit_xgb(X_train, y_train, X_test, y_test_w):
    scaler = RobustScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        objective="reg:squarederror", tree_method="hist",
        random_state=RANDOM_STATE, verbosity=0,
    )
    model.fit(Xtr, y_train)
    y_pred_w = np.exp(model.predict(Xte))
    m = compute_metrics(y_test_w, y_pred_w)
    return m["R2"], m["MAE"]

# ── Run ───────────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("  SUB-MAX THRESHOLD ANALYSIS")
print("  Q: What is the earliest sub-max duration where R² > 0.70?")
print("="*72)
print(f"\n  {'Duration':<8} {'A: alone':>10} {'B: +bio':>10} {'C: +sprint_bio':>16}  {'First >0.70?':>12}")
print("  " + "─"*62)

results = []
for col in SUBMAX_COLS:
    dur = col.replace("_critical_power", "").replace("m", " min")

    # Feature matrices
    feat_A = df_model[[col]].values
    feat_B = df_model[[col] + BIOMETRIC_COLS].values
    feat_C = df_model[[col] + SPRINT_BIO_COLS].values

    r2_A, mae_A = fit_xgb(feat_A[train_idx], y_train_log, feat_A[test_idx], y_test_w)
    r2_B, mae_B = fit_xgb(feat_B[train_idx], y_train_log, feat_B[test_idx], y_test_w)
    r2_C, mae_C = fit_xgb(feat_C[train_idx], y_train_log, feat_C[test_idx], y_test_w)

    first_above = None
    if r2_A >= 0.70: first_above = "A (alone)"
    elif r2_B >= 0.70: first_above = "B (+bio)"
    elif r2_C >= 0.70: first_above = "C (+sprint)"
    flag = f"✓ {first_above}" if first_above else "─"

    print(f"  {dur:<8} {r2_A:>10.3f} {r2_B:>10.3f} {r2_C:>16.3f}  {flag:>12}")
    results.append({
        "duration": col, "R2_alone": r2_A, "MAE_alone": mae_A,
        "R2_bio": r2_B, "MAE_bio": mae_B,
        "R2_sprint_bio": r2_C, "MAE_sprint_bio": mae_C,
    })

print("  " + "─"*62)

# ── MAE detail rows ───────────────────────────────────────────────────────────
print(f"\n  MAE (watts) detail:")
print(f"  {'Duration':<8} {'A: alone':>10} {'B: +bio':>10} {'C: +sprint_bio':>16}")
print("  " + "─"*46)
for r in results:
    dur = r["duration"].replace("_critical_power","").replace("m"," min")
    print(f"  {dur:<8} {r['MAE_alone']:>10.1f} {r['MAE_bio']:>10.1f} {r['MAE_sprint_bio']:>16.1f}")

# ── Sprint-only baseline for context ─────────────────────────────────────────
print(f"\n  ── Baseline for context ──")
feat_sprint = df_model[SPRINT_BIO_COLS].values
r2_sprint, mae_sprint = fit_xgb(feat_sprint[train_idx], y_train_log, feat_sprint[test_idx], y_test_w)
print(f"  sprint_bio only (no sub-max): R²={r2_sprint:.3f}  MAE={mae_sprint:.1f}W")

print("\n" + "="*72 + "\n")

# Save
out_df = pd.DataFrame(results)
out_df.to_csv("/Users/nathanfitzgerald/Sprint_FTP_ML/outputs/results/submax_threshold.csv", index=False)
print("  Saved → outputs/results/submax_threshold.csv")
