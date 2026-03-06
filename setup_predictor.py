"""
setup_predictor.py
One-time setup: trains and persists the models + scalers the Streamlit
athlete profiler needs. Run once after run_pipeline.py completes.

Saves to outputs/models/:
  scaler_sprint_bio.joblib              — RobustScaler for sprint_bio features
  feature_names_sprint_bio.joblib       — ordered feature name list
  XGBoost_sprint_bio_plus_2m.joblib     — XGBoost on sprint_bio + 2m_critical_power
  scaler_sprint_bio_plus_2m.joblib      — RobustScaler for sprint_bio + 2m features
  feature_names_sprint_bio_plus_2m.joblib
"""

import os, warnings
os.environ["KERAS_BACKEND"] = "torch"
warnings.filterwarnings("ignore")

import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler

from pipeline.config import (
    DATA_PATH, FEATURE_SETS, TARGET,
    OUTPUT_MODELS, TEST_SIZE, RANDOM_STATE,
)
from pipeline.data_loader import load_and_prepare
from pipeline.features import SprintFeatureEngineer
from pipeline.evaluate import compute_metrics

OUTPUT_MODELS.mkdir(parents=True, exist_ok=True)

print("\n" + "="*60)
print("  SETUP PREDICTOR — saving models + scalers for Streamlit")
print("="*60)

# ── XGBoost params (fixed good values — no re-search needed) ──────────────
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=5.0,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=RANDOM_STATE,
    verbosity=0,
)


# ── 1. sprint_bio scaler + feature names ────────────────────────────────────
print("\n[1/2] sprint_bio — extracting scaler + feature names ...")
X_train, X_test, y_train, y_test, feat_names, scaler, df_test_raw, back_transform = \
    load_and_prepare("sprint_bio", log_target=True)

joblib.dump(scaler,     OUTPUT_MODELS / "scaler_sprint_bio.joblib")
joblib.dump(feat_names, OUTPUT_MODELS / "feature_names_sprint_bio.joblib")
print(f"  Saved scaler_sprint_bio.joblib  ({len(feat_names)} features)")
print(f"  Saved feature_names_sprint_bio.joblib")

# Quick sanity: train sprint_bio XGBoost using this scaler (verify pipeline parity)
_model_sb = xgb.XGBRegressor(**XGB_PARAMS)
_model_sb.fit(X_train, y_train)
_pred_w   = np.exp(_model_sb.predict(X_test))
_y_test_w = np.exp(y_test)
_m = compute_metrics(_y_test_w, _pred_w)
print(f"  sprint_bio verification — R²={_m['R2']:.3f}  MAE={_m['MAE']:.1f}W")


# ── 2. sprint_bio + 2m model ────────────────────────────────────────────────
print("\n[2/2] sprint_bio + 2m_critical_power — training XGBoost ...")

import pandas as pd
from pipeline.features import SprintFeatureEngineer

FEAT_COLS_PLUS2M = FEATURE_SETS["sprint_bio"] + ["2m_critical_power"]

df = pd.read_csv(DATA_PATH)
eng = SprintFeatureEngineer()
df  = eng.transform(df)

required = FEAT_COLS_PLUS2M + [TARGET]
df_model = df[[c for c in required if c in df.columns]].dropna()

# Only keep cols that survived
feat_cols_2m = [c for c in FEAT_COLS_PLUS2M if c in df_model.columns]

X_raw = df_model[feat_cols_2m].values
y_raw = df_model[TARGET].values

# Stratify by gender_encoded (same as pipeline)
strat_col = df_model["gender_encoded"].values
sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
train_idx, test_idx = next(sss.split(X_raw, strat_col))

X_train_raw = X_raw[train_idx];  X_test_raw  = X_raw[test_idx]
y_train_log = np.log(y_raw[train_idx])
y_test_w    = y_raw[test_idx]

scaler_2m   = RobustScaler()
X_train_sc  = scaler_2m.fit_transform(X_train_raw)
X_test_sc   = scaler_2m.transform(X_test_raw)

model_2m = xgb.XGBRegressor(**XGB_PARAMS)
model_2m.fit(X_train_sc, y_train_log)

y_pred_w = np.exp(model_2m.predict(X_test_sc))
m2 = compute_metrics(y_test_w, y_pred_w)
print(f"  sprint_bio+2m — R²={m2['R2']:.3f}  MAE={m2['MAE']:.1f}W  "
      f"RMSE={m2['RMSE']:.1f}W")

joblib.dump(model_2m,     OUTPUT_MODELS / "XGBoost_sprint_bio_plus_2m.joblib")
joblib.dump(scaler_2m,    OUTPUT_MODELS / "scaler_sprint_bio_plus_2m.joblib")
joblib.dump(feat_cols_2m, OUTPUT_MODELS / "feature_names_sprint_bio_plus_2m.joblib")
print("  Saved XGBoost_sprint_bio_plus_2m.joblib")
print("  Saved scaler_sprint_bio_plus_2m.joblib")
print(f"  Saved feature_names_sprint_bio_plus_2m.joblib  ({len(feat_cols_2m)} features)")

print(f"\n{'='*60}")
print("  DONE — 5 files written to outputs/models/")
print(f"  sprint_bio:      R²={_m['R2']:.3f}  MAE={_m['MAE']:.1f}W")
print(f"  sprint_bio+2m:   R²={m2['R2']:.3f}  MAE={m2['MAE']:.1f}W")
print(f"{'='*60}\n")
