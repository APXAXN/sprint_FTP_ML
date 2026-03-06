"""
pipeline/config.py
Single source of truth for all constants, paths, column groups, and
hyperparameter grids. Every other module imports from here — nothing
is hardcoded elsewhere.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
DATA_PATH       = BASE_DIR / "data_inspection" / "athletes_clean.csv"
OUTPUT_MODELS   = BASE_DIR / "outputs" / "models"
OUTPUT_RESULTS  = BASE_DIR / "outputs" / "results"
OUTPUT_FIGURES  = BASE_DIR / "outputs" / "figures"

# ── Target ─────────────────────────────────────────────────────────────────
TARGET = "20m_critical_power"

# ── Raw feature groups ──────────────────────────────────────────────────────
SPRINT_COLS = [
    "1s_critical_power",
    "5s_critical_power",
    "10s_critical_power",
    "15s_critical_power",
    "20s_critical_power",
    "30s_critical_power",
]

SUBMAX_COLS = [
    "2m_critical_power",
    "3m_critical_power",
    "5m_critical_power",
    "8m_critical_power",
    "10m_critical_power",
    "30m_critical_power",
]

BIOMETRIC_COLS = ["weightkg", "age", "gender_encoded"]

ACTIVITY_COLS = ["activities", "bike"]   # 'run','swim','other' have low cycling relevance

# ── Engineered feature names (computed in features.py) ────────────────────
ENGINEERED_COLS = [
    "fatigue_index",       # 30s / 1s
    "early_decay",         # 5s / 1s
    "mid_decay",           # 15s / 5s
    "late_decay",          # 30s / 15s
    "anaerobic_reserve",   # 1s - 30s  (watts)
    "decay_curvature",     # log(1s) - 2*log(15s) + log(30s)
    "sprint_wpk_1s",       # 1s / weightkg
    "sprint_wpk_15s",      # 15s / weightkg
    "sprint_wpk_30s",      # 30s / weightkg
]

# ── Feature sets (ablation study) ─────────────────────────────────────────
FEATURE_SETS = {
    # Baseline: raw sprint power only
    "sprint_only": SPRINT_COLS,

    # + decay / W/kg derived features
    "sprint_eng": SPRINT_COLS + ENGINEERED_COLS,

    # + athlete biometrics  ← PRIMARY MODEL SET
    "sprint_bio": SPRINT_COLS + ENGINEERED_COLS + BIOMETRIC_COLS,

    # Ceiling: what sub-max features give us (not the thesis claim, just reference)
    "full_submax": SPRINT_COLS + ENGINEERED_COLS + BIOMETRIC_COLS + SUBMAX_COLS,
}

# ── Train / test split ─────────────────────────────────────────────────────
TEST_SIZE    = 0.30
RANDOM_STATE = 42
CV_FOLDS     = 5

# ── Plot style (matches existing EDA palette) ──────────────────────────────
CLR_PRIMARY  = "steelblue"
CLR_ACCENT   = "orangered"
CLR_GOLD     = "gold"
CLR_NEUTRAL  = "#888888"
CLR_GREEN    = "mediumseagreen"

MODEL_COLORS = {
    "Linear":       "#4e79a7",
    "Ridge":        "#59a14f",
    "Lasso":        "#f28e2b",
    "DecisionTree": "#e15759",
    "RandomForest": "#76b7b2",
    "XGBoost":      "#edc948",
    "NeuralNet":    "#b07aa1",
}

# ── Hyperparameter grids ───────────────────────────────────────────────────
DT_PARAM_GRID = {
    "model__max_depth":        [3, 5, 7, 10, None],
    "model__min_samples_leaf": [5, 10, 20],
    "model__min_samples_split":[10, 20],
}

RF_PARAM_DIST = {
    "model__n_estimators":     [100, 200, 300],
    "model__max_depth":        [5, 10, 20, None],
    "model__max_features":     ["sqrt", "log2", 0.5],
    "model__min_samples_leaf": [2, 5, 10],
}
RF_N_ITER = 30

XGB_PARAM_DIST = {
    "n_estimators":    [100, 200, 400],
    "max_depth":       [3, 5, 7],
    "learning_rate":   [0.01, 0.05, 0.1, 0.2],
    "subsample":       [0.7, 0.85, 1.0],
    "colsample_bytree":[0.7, 0.85, 1.0],
    "reg_alpha":       [0, 0.1, 1.0],
    "reg_lambda":      [1.0, 5.0, 10.0],
}
XGB_N_ITER = 50

# ── Neural network ─────────────────────────────────────────────────────────
NN_EPOCHS         = 200
NN_BATCH_SIZE     = 64
NN_PATIENCE       = 20
NN_VAL_SPLIT      = 0.15
NN_LEARNING_RATE  = 0.001
NN_HIDDEN_LAYERS  = [128, 64, 32]
NN_DROPOUT_RATES  = [0.3, 0.2, 0.0]
