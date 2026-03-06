"""
pipeline/evaluate.py
Metric computation, SHAP analysis, and residual diagnostics.
No plotting here — all figures are generated in plots.py.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ── Metrics ────────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Returns MAE, RMSE, R², and MAPE for a set of predictions.
    All error metrics in the same units as y (watts).
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    # MAPE: exclude near-zero actuals to avoid division instability
    mask = np.abs(y_true) > 1
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def metrics_to_df(results: dict) -> pd.DataFrame:
    """
    Convert the nested results dict {model_name: {cv_*, test_*}} into
    a clean wide DataFrame for display and CSV export.
    """
    rows = []
    for name, d in results.items():
        row = {"model": name}
        row.update({f"cv_{k}": v for k, v in d.get("cv_metrics", {}).items()})
        row.update({f"test_{k}": v for k, v in d.get("test_metrics", {}).items()})
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


# ── SHAP ───────────────────────────────────────────────────────────────────
def run_shap(fitted_estimator, X_train: np.ndarray, X_test: np.ndarray,
             feature_names: list, model_name: str = ""):
    """
    Compute SHAP values using TreeExplainer for tree-based models.
    For the NN, falls back to a lightweight KernelExplainer on a subsample.

    Parameters
    ----------
    fitted_estimator : the raw sklearn estimator (not the Pipeline wrapper)
    X_train, X_test  : scaled numpy arrays
    feature_names    : list of column name strings
    model_name       : used to decide explainer type

    Returns
    -------
    shap_values : np.ndarray shape (n_test, n_features)
    explainer   : the fitted shap Explainer object (saved for Streamlit)
    """
    import shap

    if model_name in ("RandomForest", "XGBoost", "DecisionTree"):
        explainer   = shap.TreeExplainer(fitted_estimator, data=X_train)
        shap_values = explainer(X_test)
    else:
        # Linear or NN fallback — use LinearExplainer or KernelExplainer
        if model_name in ("Linear", "Ridge", "Lasso"):
            explainer   = shap.LinearExplainer(fitted_estimator, X_train)
            shap_values = explainer(X_test)
        else:
            # NN: KernelExplainer on a 100-sample background
            background  = shap.sample(X_train, 100, random_state=42)
            explainer   = shap.KernelExplainer(fitted_estimator.predict, background)
            shap_values = explainer.shap_values(X_test[:200])  # subsample test

    return shap_values, explainer


# ── Residual analysis ──────────────────────────────────────────────────────
def residual_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute residual diagnostics (for plots.py to visualise).

    Returns
    -------
    dict with keys: residuals, std_residuals, pct_within_10W,
                    pct_within_20W, pct_within_5pct
    """
    residuals     = y_pred - y_true
    std_residuals = (residuals - residuals.mean()) / (residuals.std() + 1e-9)
    return {
        "residuals":       residuals,
        "std_residuals":   std_residuals,
        "pct_within_10W":  np.mean(np.abs(residuals) <= 10) * 100,
        "pct_within_20W":  np.mean(np.abs(residuals) <= 20) * 100,
        "pct_within_5pct": np.mean(np.abs(residuals / (y_true + 1e-9)) <= 0.05) * 100,
    }
