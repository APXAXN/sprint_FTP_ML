"""
pipeline/plots.py
All visualization logic — EDA extensions, model evaluation, SHAP, ablation.
No model training or data loading here; receives pre-computed artefacts.

Naming convention: outputs/figures/{prefix}_{description}.png
  10–13 : EDA extensions
  20–25 : Model evaluation
  30–33 : SHAP
  40–41 : Ablation study
  50    : Feature importance comparison
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

from pipeline.config import (
    OUTPUT_FIGURES, TARGET,
    CLR_PRIMARY, CLR_ACCENT, CLR_GOLD, CLR_NEUTRAL, CLR_GREEN,
    MODEL_COLORS,
)

warnings.filterwarnings("ignore")
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

_SAVE_KW = dict(dpi=150, bbox_inches="tight")


def _save(fig, name: str):
    path = OUTPUT_FIGURES / name
    fig.savefig(path, **_SAVE_KW)
    plt.close(fig)
    print(f"  Saved: {path.name}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 10–13  EDA EXTENSIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_correlations(df: pd.DataFrame, feature_cols: list):
    """Horizontal bar chart: each feature's Pearson r with 20m target."""
    corrs = df[feature_cols].corrwith(df[TARGET]).sort_values()

    fig, ax = plt.subplots(figsize=(10, max(5, len(corrs) * 0.35)))
    colors  = [CLR_ACCENT if v < 0 else CLR_PRIMARY for v in corrs]
    bars    = ax.barh(corrs.index, corrs.values, color=colors, edgecolor="white", alpha=0.88)

    for bar, val in zip(bars, corrs.values):
        ax.text(val + (0.005 if val >= 0 else -0.005),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center",
                ha="left" if val >= 0 else "right", fontsize=8)

    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Pearson r with 20-min peak power (target)", fontsize=11)
    ax.set_title("Feature Correlations with Target\n(20-min critical power)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return _save(fig, "10_feature_correlations.png")


def plot_decay_distributions(df: pd.DataFrame):
    """2×3 histogram grid for the engineered decay / ratio features."""
    from pipeline.config import ENGINEERED_COLS
    eng = [c for c in ENGINEERED_COLS if c in df.columns][:6]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Engineered Sprint-Signature Feature Distributions",
                 fontsize=13, fontweight="bold")

    for ax, col in zip(axes.flatten(), eng):
        data = df[col].dropna()
        ax.hist(data, bins=50, color=CLR_GREEN, edgecolor="white", alpha=0.85)
        ax.axvline(data.median(), color=CLR_ACCENT, lw=2, ls="--",
                   label=f"med={data.median():.3f}")
        ax.set_title(col, fontweight="bold", fontsize=9)
        ax.set_xlabel("Value"); ax.set_ylabel("Athletes")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.tight_layout()
    return _save(fig, "11_decay_index_distributions.png")


def plot_target_by_gender(df: pd.DataFrame):
    """Overlapping KDE of 20m target for M vs F athletes."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for gender, color, label in [("M", CLR_PRIMARY, "Male (n={})"),
                                   ("F", CLR_ACCENT,  "Female (n={})")]:
        subset = df[df["gender"] == gender][TARGET].dropna()
        label  = label.format(len(subset))
        subset.plot.kde(ax=ax, color=color, linewidth=2.5, label=label)
        ax.axvline(subset.median(), color=color, ls="--", lw=1.5, alpha=0.7)

    ax.set_xlabel("20-min Peak Power (W)", fontsize=11)
    ax.set_title("20-min Power Distribution by Gender",
                 fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    return _save(fig, "12_target_by_gender.png")


def plot_sprint_vs_target(df: pd.DataFrame):
    """3×2 scatter: each sprint feature vs. target, coloured by gender."""
    from pipeline.config import SPRINT_COLS
    cols = [c for c in SPRINT_COLS if c in df.columns][:6]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Sprint Power vs 20-min Target (by gender)",
                 fontsize=13, fontweight="bold")

    gender_colors = {"M": CLR_PRIMARY, "F": CLR_ACCENT}

    for ax, col in zip(axes.flatten(), cols):
        for g, c in gender_colors.items():
            sub = df[df["gender"] == g][[col, TARGET]].dropna()
            ax.scatter(sub[col], sub[TARGET], c=c, alpha=0.15,
                       s=8, label=g, rasterized=True)

        r = df[[col, TARGET]].dropna().corr().iloc[0, 1]
        dur = col.replace("_critical_power", "").replace("s", "s ").strip()
        ax.set_title(f"{dur} — r={r:+.3f}", fontweight="bold")
        ax.set_xlabel("Sprint Power (W)")
        ax.set_ylabel("20m Power (W)")
        ax.grid(alpha=0.3)
        if col == cols[0]:
            ax.legend(markerscale=3)

    plt.tight_layout()
    return _save(fig, "13_sprint_vs_target_scatter.png")


# ══════════════════════════════════════════════════════════════════════════════
# 20–25  MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def _metric_comparison(results: dict, split: str, filename: str, title: str):
    """Shared helper: grouped bar chart of MAE / RMSE / R² across models."""
    names   = list(results.keys())
    metrics = ["MAE", "RMSE", "R2"]
    prefix  = f"{split}_"
    x       = np.arange(len(names))
    width   = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        vals   = [results[n].get(prefix + metric, np.nan) for n in names]
        colors = [MODEL_COLORS.get(n, CLR_PRIMARY) for n in names]
        bars   = ax.bar(x, vals, color=colors, edgecolor="white", alpha=0.88)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_title(metric, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        if metric == "R2":
            ax.set_ylim(0, 1)
            ax.set_ylabel("R² (higher is better)")
        else:
            ax.set_ylabel("Watts (lower is better)")

        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.002 if metric == "R2" else 0.3),
                        f"{val:.3f}" if metric == "R2" else f"{val:.1f}",
                        ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    return _save(fig, filename)


def plot_cv_performance(results: dict):
    return _metric_comparison(
        results, "cv",
        "20_cv_performance_comparison.png",
        "5-Fold Cross-Validation Performance (sprint_bio feature set)",
    )


def plot_test_performance(results: dict):
    return _metric_comparison(
        results, "test",
        "21_test_performance_comparison.png",
        "Hold-Out Test Set Performance (sprint_bio feature set, 30%)",
    )


def plot_predicted_vs_actual(results: dict, y_test: np.ndarray):
    """One scatter panel per model; identity line in orangered."""
    names = list(results.keys())
    n     = len(names)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    fig.suptitle("Predicted vs Actual — 20-min Peak Power (W)",
                 fontsize=13, fontweight="bold")
    axes_flat = np.array(axes).flatten()

    lo = y_test.min() * 0.95
    hi = y_test.max() * 1.05

    for ax, name in zip(axes_flat, names):
        y_pred = results[name].get("test_predictions", np.full_like(y_test, np.nan))
        r2     = results[name].get("test_R2", np.nan)
        ax.scatter(y_test, y_pred, alpha=0.25, s=10,
                   color=MODEL_COLORS.get(name, CLR_PRIMARY), rasterized=True)
        ax.plot([lo, hi], [lo, hi], color=CLR_ACCENT, lw=1.8, ls="--", label="Perfect")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Actual (W)"); ax.set_ylabel("Predicted (W)")
        ax.set_title(f"{name}  R²={r2:.3f}", fontweight="bold")
        ax.grid(alpha=0.3)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    return _save(fig, "22_predicted_vs_actual.png")


def plot_residual_distributions(results: dict, y_test: np.ndarray):
    """Overlaid KDE of residuals for each model."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for name, d in results.items():
        y_pred = d.get("test_predictions")
        if y_pred is None: continue
        residuals = y_pred - y_test
        pd.Series(residuals).plot.kde(
            ax=ax, label=f"{name} (MAE={d.get('test_MAE', 0):.1f}W)",
            color=MODEL_COLORS.get(name, CLR_PRIMARY), linewidth=2, alpha=0.8,
        )

    ax.axvline(0, color="black", lw=1.2)
    ax.set_xlabel("Residual (Predicted − Actual) [Watts]", fontsize=11)
    ax.set_title("Residual Distributions — All Models", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    return _save(fig, "23_residual_distributions.png")


def plot_nn_history(history):
    """Training / validation loss curve for the Keras NN."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Neural Network Training History", fontsize=12, fontweight="bold")

    for ax, metric, title in [
        (axes[0], "loss", "MSE Loss"),
        (axes[1], "mae",  "MAE (Watts)"),
    ]:
        ax.plot(history.history[metric],       color=CLR_PRIMARY,  label="Train")
        ax.plot(history.history[f"val_{metric}"], color=CLR_ACCENT, label="Validation")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel(metric.upper())
        ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    return _save(fig, "25_nn_training_history.png")


# ══════════════════════════════════════════════════════════════════════════════
# 30–33  SHAP
# ══════════════════════════════════════════════════════════════════════════════

def plot_shap_summary(shap_values, feature_names: list, model_name: str):
    """Beeswarm summary plot."""
    import shap as shap_lib
    fig, ax = plt.subplots(figsize=(10, max(5, len(feature_names) * 0.45)))
    shap_lib.plots.beeswarm(shap_values, max_display=20, show=False)
    fig = plt.gcf()
    fig.suptitle(f"SHAP Summary — {model_name}", fontsize=12, fontweight="bold", y=1.01)
    return _save(fig, "30_shap_summary_beeswarm.png")


def plot_shap_importance(shap_values, feature_names: list, model_name: str):
    """Bar chart of mean |SHAP| values."""
    import shap as shap_lib
    shap_lib.plots.bar(shap_values, max_display=20, show=False)
    fig = plt.gcf()
    fig.suptitle(f"SHAP Feature Importance — {model_name}",
                 fontsize=12, fontweight="bold", y=1.01)
    return _save(fig, "31_shap_feature_importance.png")


def plot_shap_waterfall(shap_values, idx: int, label: str):
    """Single-prediction waterfall plot."""
    import shap as shap_lib
    shap_lib.plots.waterfall(shap_values[idx], show=False)
    fig = plt.gcf()
    fig.suptitle(f"SHAP Waterfall — {label}", fontsize=11, fontweight="bold", y=1.01)
    fname = f"32_shap_waterfall_{label.lower().replace(' ', '_')}.png"
    return _save(fig, fname)


def plot_shap_dependence(shap_values, X_test: np.ndarray,
                         feature_names: list, top_n: int = 3):
    """Dependence scatter for top N features by mean |SHAP|."""
    import shap as shap_lib

    vals   = shap_values.values
    means  = np.abs(vals).mean(axis=0)
    top_i  = np.argsort(means)[-top_n:][::-1]

    fig, axes = plt.subplots(1, top_n, figsize=(6 * top_n, 5))
    if top_n == 1:
        axes = [axes]

    for ax, fi in zip(axes, top_i):
        feat  = feature_names[fi]
        shap_lib.dependence_plot(
            fi, vals, X_test,
            feature_names=feature_names,
            ax=ax, show=False, alpha=0.4,
        )
        ax.set_title(f"SHAP Dependence: {feat}", fontweight="bold")

    fig.suptitle("SHAP Dependence Plots — Top 3 Features",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "33_shap_dependence_top3.png")


# ══════════════════════════════════════════════════════════════════════════════
# 40–41  ABLATION STUDY
# ══════════════════════════════════════════════════════════════════════════════

def plot_ablation_heatmap(ablation_results: dict):
    """
    R² heatmap: rows = models, cols = feature sets.
    ablation_results: {feature_set: {model_name: {test_R2: float}}}
    """
    feat_sets = list(ablation_results.keys())
    models    = list(next(iter(ablation_results.values())).keys())

    data = pd.DataFrame(
        {fs: {m: ablation_results[fs][m].get("test_R2", np.nan) for m in models}
         for fs in feat_sets}
    )

    fig, ax = plt.subplots(figsize=(len(feat_sets) * 2.5 + 1, len(models) * 0.8 + 1))
    sns.heatmap(
        data, annot=True, fmt=".3f", cmap="YlGn",
        linewidths=0.5, linecolor="white",
        vmin=0, vmax=1, ax=ax,
        cbar_kws={"label": "Test R²"},
    )
    ax.set_title("Ablation Study — Test R² by Model & Feature Set",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Feature Set"); ax.set_ylabel("Model")
    plt.tight_layout()
    return _save(fig, "40_ablation_r2_heatmap.png")


def plot_ablation_mae(ablation_results: dict):
    """MAE bar chart per feature set, relative to sprint_only baseline."""
    feat_sets = list(ablation_results.keys())
    models    = list(next(iter(ablation_results.values())).keys())
    x         = np.arange(len(models))
    width     = 0.8 / len(feat_sets)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Ablation Study — MAE by Feature Set\n(lower = better)",
                 fontsize=12, fontweight="bold")

    colors = [CLR_NEUTRAL, CLR_GREEN, CLR_PRIMARY, CLR_ACCENT]
    for i, (fs, color) in enumerate(zip(feat_sets, colors)):
        maes = [ablation_results[fs][m].get("test_MAE", np.nan) for m in models]
        offset = (i - len(feat_sets) / 2 + 0.5) * width
        bars = ax.bar(x + offset, maes, width * 0.9, label=fs,
                      color=color, alpha=0.85, edgecolor="white")

    ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("MAE (Watts)")
    ax.legend(title="Feature Set", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return _save(fig, "41_ablation_mae_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 50  FEATURE IMPORTANCE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def plot_importance_comparison(results: dict, feature_names: list, shap_values=None):
    """
    4-panel figure comparing feature importance across methods:
    Ridge coefficients, RF importance, XGB importance, SHAP values.
    """
    panels = []

    # Ridge: |coefficient| (sorted)
    if "Ridge" in results:
        ridge_model = results["Ridge"].get("fitted_model")
        if ridge_model is not None:
            try:
                coef = ridge_model.named_steps["model"].coef_
                panels.append(("Ridge |Coefficient|", np.abs(coef), CLR_GREEN))
            except Exception:
                pass

    # Random Forest: feature_importances_
    if "RandomForest" in results:
        rf = results["RandomForest"].get("fitted_model")
        if rf is not None:
            try:
                best = rf.best_estimator_ if hasattr(rf, "best_estimator_") else rf
                imp  = best.named_steps["model"].feature_importances_
                panels.append(("Random Forest Importance", imp, CLR_PRIMARY))
            except Exception:
                pass

    # XGBoost: feature_importances_
    if "XGBoost" in results:
        xgb_m = results["XGBoost"].get("fitted_model")
        if xgb_m is not None:
            try:
                best = xgb_m.best_estimator_ if hasattr(xgb_m, "best_estimator_") else xgb_m
                imp  = best.feature_importances_
                panels.append(("XGBoost Importance", imp, CLR_GOLD))
            except Exception:
                pass

    # SHAP: mean |SHAP|
    if shap_values is not None:
        vals = shap_values.values if hasattr(shap_values, "values") else shap_values
        panels.append(("Mean |SHAP|", np.abs(vals).mean(axis=0), CLR_ACCENT))

    if not panels:
        print("  [plots] No importance data available for comparison plot.")
        return None

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, max(5, len(feature_names) * 0.35)))
    fig.suptitle("Feature Importance Comparison", fontsize=13, fontweight="bold")
    if n == 1:
        axes = [axes]

    for ax, (title, importances, color) in zip(axes, panels):
        if len(importances) != len(feature_names):
            ax.set_visible(False)
            continue
        order = np.argsort(importances)
        ax.barh(np.array(feature_names)[order], importances[order],
                color=color, edgecolor="white", alpha=0.88)
        ax.set_title(title, fontweight="bold", fontsize=9)
        ax.set_xlabel("Importance"); ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    return _save(fig, "50_importance_comparison.png")
