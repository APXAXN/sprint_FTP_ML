"""
run_pipeline.py
Single entry point for the Sprint FTP ML pipeline.

Usage:
    python run_pipeline.py

Execution order:
  1.  Validate data & engineer features
  2.  EDA extension plots
  3.  Train all 7 models on 'sprint_bio' (primary feature set)
  4.  Evaluate + save test predictions
  5.  SHAP analysis on best tree-based model
  6.  Ablation study: XGBoost-only across all 4 feature sets
  7.  All evaluation and ablation plots
  8.  Save metrics CSVs
  9.  Print final summary

Runtime: ~15–25 min on a modern laptop (dominated by XGBoost search + NN).
"""

import time
import warnings
import numpy as np
import pandas as pd
import joblib
import os

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # suppress TF startup noise
os.environ["KERAS_BACKEND"] = "torch"       # use PyTorch backend (TF 2.20 segfaults on Py3.13)

from pathlib import Path
from pipeline.config import (
    DATA_PATH, OUTPUT_MODELS, OUTPUT_RESULTS, OUTPUT_FIGURES,
    FEATURE_SETS, SPRINT_COLS, ENGINEERED_COLS,
)
from pipeline.features import SprintFeatureEngineer
from pipeline.train    import run_all_models
from pipeline.evaluate import run_shap, residual_stats
import pipeline.plots   as P


def setup():
    """Create all output directories."""
    for d in (OUTPUT_MODELS, OUTPUT_RESULTS, OUTPUT_FIGURES):
        d.mkdir(parents=True, exist_ok=True)
    print("Output directories ready.")


def validate_data():
    """Quick sanity check on the clean dataset."""
    df = pd.read_csv(DATA_PATH)
    print(f"\nData check: {len(df):,} rows × {df.shape[1]} columns")
    assert len(df) >= 4000, "Unexpectedly few rows — check athletes_clean.csv"
    assert "20m_critical_power" in df.columns, "Target column missing"
    assert "30s_critical_power" in df.columns, "Sprint feature missing"
    print("  ✓ athletes_clean.csv validated")
    return df


def eda_plots(df_raw: pd.DataFrame):
    """Generate EDA extension plots from the engineered dataframe."""
    print("\n── EDA extension plots ──────────────────────────────────────────")
    engineer = SprintFeatureEngineer()
    df       = engineer.transform(df_raw)

    # Correlation: sprint + engineered features
    corr_cols = [c for c in SPRINT_COLS + ENGINEERED_COLS if c in df.columns]
    P.plot_feature_correlations(df, corr_cols)
    P.plot_decay_distributions(df)
    P.plot_target_by_gender(df)
    P.plot_sprint_vs_target(df)

    # Save engineered dataset
    out = OUTPUT_RESULTS / "athletes_engineered.csv"
    df.to_csv(out, index=False)
    print(f"  Saved engineered dataset → {out.name}")

    return df


def train_primary(df_engineered: pd.DataFrame):
    """Train all 7 models on sprint_bio (primary feature set)."""
    print("\n── Primary training: sprint_bio — all 7 models ─────────────────")
    print("   Target: log(FTP) — metrics back-transformed to watts")
    results = run_all_models("sprint_bio", skip_nn=False, verbose=True,
                             log_target=True)
    return results


def save_results(results: dict, feature_set: str):
    """Save metrics and predictions to CSVs."""
    rows = []
    for name, d in results.items():
        if name == "__meta__": continue
        row = {"model": name, "feature_set": feature_set}
        row.update({f"cv_{k}": v for k, v in d.get("cv_metrics", {}).items()})
        row.update({f"test_{k}": v for k, v in d.get("test_metrics", {}).items()})
        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    metrics_path = OUTPUT_RESULTS / f"metrics_{feature_set}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  Saved metrics → {metrics_path.name}")

    # Predictions CSV
    meta    = results["__meta__"]
    y_test  = meta["y_test"]
    pred_df = pd.DataFrame({"y_true": y_test})
    for name, d in results.items():
        if name == "__meta__": continue
        y_pred = d.get("test_predictions")
        if y_pred is not None:
            pred_df[f"y_pred_{name}"] = y_pred

    pred_path = OUTPUT_RESULTS / f"predictions_{feature_set}.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"  Saved predictions → {pred_path.name}")

    return metrics_df


def run_shap_analysis(results: dict):
    """SHAP analysis on the best tree-based model."""
    print("\n── SHAP analysis ────────────────────────────────────────────────")
    meta          = results["__meta__"]
    X_train       = meta["X_train"]
    X_test        = meta["X_test"]
    y_test        = meta["y_test"]
    feature_names = meta["feature_names"]

    # Pick best tree-based model by test R²
    tree_models = ["XGBoost", "RandomForest", "DecisionTree"]
    available   = {n: results[n]["test_R2"] for n in tree_models if n in results}
    if not available:
        print("  No tree models found — skipping SHAP.")
        return None, None

    best_name  = max(available, key=available.get)
    best_model = results[best_name]["fitted_model"]
    print(f"  Best tree model: {best_name}  (test R²={available[best_name]:.3f})")

    # For XGBoost (not in a Pipeline), pass X directly; for sklearn Pipelines, unwrap
    if best_name in ("RandomForest", "DecisionTree"):
        # Unwrap Pipeline → get the raw estimator
        raw_model = best_model.named_steps["model"] \
                    if hasattr(best_model, "named_steps") else best_model
        X_train_shap = best_model.named_steps["scaler"].transform(X_train) \
                       if hasattr(best_model, "named_steps") else X_train
        X_test_shap  = best_model.named_steps["scaler"].transform(X_test) \
                       if hasattr(best_model, "named_steps") else X_test
    else:
        raw_model    = best_model
        X_train_shap = X_train
        X_test_shap  = X_test

    shap_values, explainer = run_shap(
        raw_model, X_train_shap, X_test_shap, feature_names, best_name
    )

    # Save explainer for Streamlit
    joblib.dump(explainer, OUTPUT_MODELS / "shap_explainer.joblib")

    # Plots
    try:
        P.plot_shap_summary(shap_values, feature_names, best_name)
        P.plot_shap_importance(shap_values, feature_names, best_name)

        # Waterfall: under / accurate / over-predicted examples
        y_pred    = results[best_name]["test_predictions"]
        errors    = y_pred - y_test
        under_i   = int(np.argmin(errors))        # most under-predicted
        over_i    = int(np.argmax(errors))        # most over-predicted
        accurate_i = int(np.argmin(np.abs(errors)))   # most accurate

        for idx, label in [(under_i, "Under"), (accurate_i, "Accurate"), (over_i, "Over")]:
            try:
                P.plot_shap_waterfall(shap_values, idx, label)
            except Exception as e:
                print(f"  Waterfall {label} failed: {e}")

        P.plot_shap_dependence(shap_values, X_test_shap, feature_names)

    except Exception as e:
        print(f"  SHAP plot error: {e}")

    return shap_values, best_name


def run_ablation(primary_results: dict):
    """
    Train XGBoost-only on all 4 feature sets to show the ablation story.
    Returns nested dict: {feature_set: {XGBoost: metrics_dict}}.
    """
    print("\n── Ablation study: XGBoost across all feature sets ──────────────")
    ablation = {}

    for fs in FEATURE_SETS.keys():
        print(f"\n  Feature set: {fs}")
        res = run_all_models(fs, skip_nn=True, verbose=True, log_target=True)
        ablation[fs] = {
            name: d for name, d in res.items()
            if name != "__meta__"
        }

        # Save per-feature-set metrics
        save_results(res, fs)

    return ablation


def evaluation_plots(results: dict, shap_values, best_shap_model: str):
    """Generate all model evaluation plots for the primary sprint_bio set."""
    print("\n── Evaluation plots ─────────────────────────────────────────────")
    meta    = results["__meta__"]
    y_test  = meta["y_test"]
    X_test  = meta["X_test"]
    feat_names = meta["feature_names"]

    # Reformat results for plotting helpers (they expect flat metric keys)
    plot_results = {
        name: d for name, d in results.items() if name != "__meta__"
    }

    P.plot_cv_performance(plot_results)
    P.plot_test_performance(plot_results)
    P.plot_predicted_vs_actual(plot_results, y_test)
    P.plot_residual_distributions(plot_results, y_test)

    # NN history
    if "NeuralNet" in results and results["NeuralNet"].get("nn_history"):
        P.plot_nn_history(results["NeuralNet"]["nn_history"])

    # Feature importance comparison
    P.plot_importance_comparison(plot_results, feat_names, shap_values)


def ablation_plots(ablation_results: dict):
    """Generate ablation summary plots."""
    print("\n── Ablation plots ───────────────────────────────────────────────")
    P.plot_ablation_heatmap(ablation_results)
    P.plot_ablation_mae(ablation_results)


def print_final_summary(primary_results: dict, ablation_results: dict):
    """Print a clean final summary table."""
    print(f"\n{'='*65}")
    print("  FINAL SUMMARY — Sprint FTP ML Pipeline")
    print(f"{'='*65}")

    print("\n  PRIMARY MODEL SET: sprint_bio")
    print(f"  {'Model':<16} {'Test R²':>9} {'Test MAE':>10} {'Test RMSE':>11}")
    print("  " + "─" * 50)
    for name, d in primary_results.items():
        if name == "__meta__": continue
        test = d.get("test_metrics", {})
        print(
            f"  {name:<16} "
            f"{test.get('R2', float('nan')):>9.3f} "
            f"{test.get('MAE', float('nan')):>10.1f}W "
            f"{test.get('RMSE', float('nan')):>10.1f}W"
        )

    print("\n  ABLATION STUDY — Best model per feature set (XGBoost):")
    print(f"  {'Feature Set':<20} {'Test R²':>9} {'Test MAE':>10}")
    print("  " + "─" * 44)
    for fs, model_results in ablation_results.items():
        xgb = model_results.get("XGBoost", {})
        test = xgb.get("test_metrics", {})
        print(
            f"  {fs:<20} "
            f"{test.get('R2', float('nan')):>9.3f} "
            f"{test.get('MAE', float('nan')):>10.1f}W"
        )

    print(f"\n  Output files written to:")
    print(f"    {OUTPUT_MODELS}")
    print(f"    {OUTPUT_RESULTS}")
    print(f"    {OUTPUT_FIGURES}")
    print(f"{'='*65}\n")


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t_start = time.time()

    print("=" * 65)
    print("  SPRINT FTP ML PIPELINE")
    print("  MSIS 522 | UW Foster School of Business")
    print("=" * 65)

    setup()

    # 1. Validate + EDA
    df_raw       = validate_data()
    df_engineered = eda_plots(df_raw)

    # 2. Primary training (all 7 models, sprint_bio)
    primary_results = train_primary(df_engineered)
    save_results(primary_results, "sprint_bio")

    # 3. SHAP
    shap_values, best_shap_model = run_shap_analysis(primary_results)

    # 4. Ablation (XGBoost only, all 4 feature sets)
    ablation_results = run_ablation(primary_results)

    # 5. All plots
    evaluation_plots(primary_results, shap_values, best_shap_model)
    ablation_plots(ablation_results)

    # 6. Final summary
    print_final_summary(primary_results, ablation_results)

    elapsed = time.time() - t_start
    print(f"Pipeline complete in {elapsed/60:.1f} min.\n")
