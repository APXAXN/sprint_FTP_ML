"""
pipeline/data_loader.py
Loads athletes_clean.csv, runs SprintFeatureEngineer, selects the
requested feature set, and returns stratified train/test splits.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler

from pipeline.config import (
    DATA_PATH, FEATURE_SETS, TARGET,
    TEST_SIZE, RANDOM_STATE,
)
from pipeline.features import SprintFeatureEngineer


def load_and_prepare(feature_set_name: str = "sprint_bio",
                     log_target: bool = False):
    """
    Load the clean dataset, engineer features, and return a stratified
    70/30 train/test split for the requested feature set.

    Parameters
    ----------
    feature_set_name : one of 'sprint_only', 'sprint_eng',
                       'sprint_bio', 'sprint_bio_v2', 'full_submax'
    log_target       : if True, y_train / y_test are log-transformed.
                       back_transform (np.exp) is returned as the last
                       element so callers can recover watts-scale metrics.

    Returns
    -------
    X_train, X_test : np.ndarray (RobustScaler applied)
    y_train, y_test : np.ndarray  (log-scale if log_target=True)
    feature_names   : list[str]
    scaler          : fitted RobustScaler (needed for SHAP / Streamlit)
    df_test_raw     : unscaled test-set DataFrame (for residual plots)
    back_transform  : callable (np.exp) or None
    """
    if feature_set_name not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set '{feature_set_name}'. "
            f"Choose from: {list(FEATURE_SETS.keys())}"
        )

    # ── Load & engineer ───────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    engineer = SprintFeatureEngineer()
    df = engineer.transform(df)

    # ── Select columns ────────────────────────────────────────────────────
    feat_cols = FEATURE_SETS[feature_set_name]
    # Guard: only keep columns that actually exist after engineering
    feat_cols = [c for c in feat_cols if c in df.columns]
    required  = feat_cols + [TARGET]
    df_model  = df[required].dropna()

    X_raw = df_model[feat_cols].values
    y     = df_model[TARGET].values

    # ── Stratified split on gender (3% female — must be in both splits) ───
    if "gender_encoded" in feat_cols:
        strat_col = df_model["gender_encoded"].values
    else:
        # Build gender_encoded even if not in feature set for stratification
        strat_col = (df_model["gender"] == "F").astype(int).values \
                    if "gender" in df_model.columns \
                    else np.zeros(len(df_model), dtype=int)

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_idx, test_idx = next(sss.split(X_raw, strat_col))

    X_train_raw = X_raw[train_idx]
    X_test_raw  = X_raw[test_idx]
    y_train_raw = y[train_idx]
    y_test_raw  = y[test_idx]

    # ── Optional log-transform of target ──────────────────────────────────
    if log_target:
        y_train         = np.log(y_train_raw)
        y_test          = np.log(y_test_raw)
        back_transform  = np.exp
    else:
        y_train         = y_train_raw
        y_test          = y_test_raw
        back_transform  = None

    # ── Scale (fit on train only) ─────────────────────────────────────────
    scaler      = RobustScaler()
    X_train     = scaler.fit_transform(X_train_raw)
    X_test      = scaler.transform(X_test_raw)

    # ── Raw test DataFrame for diagnostic plots ───────────────────────────
    df_test_raw = df_model.iloc[test_idx].reset_index(drop=True)

    log_tag = "  log(FTP) target" if log_target else ""
    print(
        f"  [{feature_set_name}] "
        f"train={len(y_train):,}  test={len(y_test):,}  "
        f"features={len(feat_cols)}  "
        f"female%_train={100*strat_col[train_idx].mean():.1f}  "
        f"female%_test={100*strat_col[test_idx].mean():.1f}"
        f"{log_tag}"
    )

    return X_train, X_test, y_train, y_test, feat_cols, scaler, df_test_raw, back_transform
