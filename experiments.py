"""
experiments.py
Three target / feature improvements tested against the sprint_bio baseline.

Experiments
-----------
  Baseline  : sprint_bio features, raw FTP (watts) — XGBoost + RandomForest
  Exp 2     : sprint_bio + log(FTP) target → back-transform via exp()
  Exp 3     : sprint_bio_v2 (+ 4 ramp/power-law features), raw FTP
  Exp 4     : sprint_bio + FTP/kg target → back-transform via × weightkg
  Best combo: sprint_bio_v2 + log(FTP/kg) target

All experiments use the same 70/30 stratified split and identical XGBoost
hyperparameter search (n_iter=25) for a fair comparison.  RandomForest is
also tested on the winning configuration.

Runtime: ~5 min.
"""

import os
import warnings
os.environ["KERAS_BACKEND"] = "torch"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from pipeline.config import (
    XGB_PARAM_DIST, RF_PARAM_DIST,
    RANDOM_STATE, CV_FOLDS,
)
from pipeline.data_loader import load_and_prepare
from pipeline.evaluate import compute_metrics


# ── Config ─────────────────────────────────────────────────────────────────
N_ITER = 25      # fewer than production (50) for speed
VERBOSE = False


# ── Model factories ────────────────────────────────────────────────────────
def _xgb():
    return RandomizedSearchCV(
        xgb.XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
        param_distributions=XGB_PARAM_DIST,
        n_iter=N_ITER,
        cv=CV_FOLDS,
        scoring="r2",
        n_jobs=-1,
        refit=True,
        random_state=RANDOM_STATE,
        verbose=0,
    )


def _rf():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler
    return RandomizedSearchCV(
        Pipeline([
            ("scaler", RobustScaler()),
            ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        param_distributions={f"model__{k}": v for k, v in {
            "n_estimators":     [200, 400],
            "max_depth":        [10, 20, None],
            "max_features":     ["sqrt", 0.5],
            "min_samples_leaf": [2, 5],
        }.items()},
        n_iter=15,
        cv=CV_FOLDS,
        scoring="r2",
        n_jobs=-1,
        refit=True,
        random_state=RANDOM_STATE,
        verbose=0,
    )


# ── Core runner ────────────────────────────────────────────────────────────
def run(name, model_fn, X_tr, X_te, y_tr_transformed, y_te_raw,
        back_fn=None):
    """
    Fit model on (X_tr, y_tr_transformed).
    Predict on X_te, optionally back-transform, then score against y_te_raw.
    Returns metrics dict.
    """
    m = model_fn()
    m.fit(X_tr, y_tr_transformed)
    best = m.best_estimator_
    y_pred = best.predict(X_te)
    if back_fn is not None:
        y_pred = back_fn(y_pred)
    metrics = compute_metrics(y_te_raw, y_pred)
    bp = getattr(m, "best_params_", None)
    return metrics, bp


# ── Pretty printer ─────────────────────────────────────────────────────────
results_table = []

def record(label, metrics):
    results_table.append({
        "Experiment": label,
        "R²":   metrics["R2"],
        "MAE":  metrics["MAE"],
        "RMSE": metrics["RMSE"],
    })
    delta_r2 = metrics["R2"] - baseline_r2 if baseline_r2 is not None else 0.0
    sign = "▲" if delta_r2 > 0.001 else ("▼" if delta_r2 < -0.001 else "─")
    print(
        f"  {label:<38} "
        f"R²={metrics['R2']:.4f} ({sign}{abs(delta_r2):.4f})  "
        f"MAE={metrics['MAE']:.1f}W"
    )


# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  EXPERIMENTS: Log-target / Ramp features / W·kg⁻¹ target")
print("=" * 65)

# ── Load both feature sets (done once) ────────────────────────────────────
print("\nLoading sprint_bio ...")
X_tr,  X_te,  y_tr,  y_te,  fcols,  scaler,  df_te_raw,  _ = load_and_prepare("sprint_bio")

print("Loading sprint_bio_v2 ...")
X_tr2, X_te2, y_tr2, y_te2, fcols2, scaler2, df_te2_raw, _ = load_and_prepare("sprint_bio_v2")

# Raw train weights (needed for W/kg target)
wt_col_idx  = fcols.index("weightkg")
wt_tr_raw   = scaler.inverse_transform(X_tr)[:, wt_col_idx]
wt_te_raw   = df_te_raw["weightkg"].values

wt_col_idx2 = fcols2.index("weightkg")
wt_tr2_raw  = scaler2.inverse_transform(X_tr2)[:, wt_col_idx2]
wt_te2_raw  = df_te2_raw["weightkg"].values

# ── Baseline R² for delta reporting ───────────────────────────────────────
baseline_r2 = None   # filled after first result

print("\n── Results (XGBoost, n_iter=25) " + "─" * 32)
print(f"  {'Experiment':<38} {'R² (Δ vs baseline)':>20}  MAE")
print("  " + "─" * 63)

# ── 0. Baseline ───────────────────────────────────────────────────────────
print("\n  Running baseline ...", end=" ", flush=True)
m0, _ = run("baseline", _xgb, X_tr, X_te, y_tr, y_te)
baseline_r2 = m0["R2"]
record("0. Baseline (sprint_bio, raw)", m0)

# ── Exp 2: Log-transform target ───────────────────────────────────────────
print("  Exp 2: log-target ...", end=" ", flush=True)
m2, _ = run("log", _xgb, X_tr, X_te,
            np.log(y_tr), y_te,
            back_fn=np.exp)
record("2. Log(FTP) target → exp(pred)", m2)

# ── Exp 3: Ramp-rate features ─────────────────────────────────────────────
print("  Exp 3: ramp features ...", end=" ", flush=True)
m3, _ = run("ramps", _xgb, X_tr2, X_te2, y_tr2, y_te2)
record("3. Ramp + power-law features", m3)

# ── Exp 4: W/kg target ───────────────────────────────────────────────────
print("  Exp 4: W/kg target ...", end=" ", flush=True)
y_tr_wpk = y_tr / wt_tr_raw
m4, _ = run("wpk", _xgb, X_tr, X_te,
            y_tr_wpk, y_te,
            back_fn=lambda pred: pred * wt_te_raw)
record("4. FTP/kg target → pred × kg", m4)

# ── Best combo: ramps + log ───────────────────────────────────────────────
print("  Combo: ramps + log ...", end=" ", flush=True)
m_combo, _ = run("combo", _xgb, X_tr2, X_te2,
                 np.log(y_tr2), y_te2,
                 back_fn=np.exp)
record("5. Ramps + log(FTP) [combo]", m_combo)

# ── Best combo: ramps + W/kg ─────────────────────────────────────────────
print("  Combo: ramps + W/kg ...", end=" ", flush=True)
y_tr2_wpk = y_tr2 / wt_tr2_raw
m_combo2, _ = run("combo2", _xgb, X_tr2, X_te2,
                  y_tr2_wpk, y_te2,
                  back_fn=lambda pred: pred * wt_te2_raw)
record("6. Ramps + FTP/kg [combo]", m_combo2)

# ── Best combo: ramps + log(FTP/kg) ─────────────────────────────────────
print("  Combo: ramps + log(W/kg) ...", end=" ", flush=True)
y_tr2_log_wpk = np.log(y_tr2 / wt_tr2_raw)
m_combo3, _ = run("combo3", _xgb, X_tr2, X_te2,
                  y_tr2_log_wpk, y_te2,
                  back_fn=lambda pred: np.exp(pred) * wt_te2_raw)
record("7. Ramps + log(FTP/kg) [best?]", m_combo3)


# ── Run RF on the overall winner ──────────────────────────────────────────
best_exp = max(results_table, key=lambda r: r["R²"])
print(f"\n\n  Best config so far: '{best_exp['Experiment']}' (R²={best_exp['R²']:.4f})")
print(f"  Running RandomForest on the same config ...", end=" ", flush=True)

# Determine which data / target to use for the winner
label = best_exp["Experiment"]
if "Ramps" in label and "log(FTP/kg)" in label:
    Xr, Xe, yr, ye_raw, wt_e = X_tr2, X_te2, y_tr2_log_wpk, y_te2, wt_te2_raw
    back = lambda p: np.exp(p) * wt_e
elif "Ramps" in label and "FTP/kg" in label:
    Xr, Xe, yr, ye_raw, wt_e = X_tr2, X_te2, y_tr2_wpk, y_te2, wt_te2_raw
    back = lambda p: p * wt_e
elif "Ramps" in label and "log" in label:
    Xr, Xe, yr, ye_raw = X_tr2, X_te2, np.log(y_tr2), y_te2
    back = np.exp
elif "Ramps" in label:
    Xr, Xe, yr, ye_raw = X_tr2, X_te2, y_tr2, y_te2; back = None
elif "log(FTP/kg)" in label:
    Xr, Xe, yr, ye_raw, wt_e = X_tr, X_te, np.log(y_tr/wt_tr_raw), y_te, wt_te_raw
    back = lambda p: np.exp(p) * wt_e
elif "FTP/kg" in label:
    Xr, Xe, yr, ye_raw, wt_e = X_tr, X_te, y_tr_wpk, y_te, wt_te_raw
    back = lambda p: p * wt_e
elif "Log" in label:
    Xr, Xe, yr, ye_raw = X_tr, X_te, np.log(y_tr), y_te; back = np.exp
else:
    Xr, Xe, yr, ye_raw = X_tr, X_te, y_tr, y_te; back = None

mrf, _ = run("rf_best", _rf, Xr, Xe, yr, ye_raw, back_fn=back)
record(f"RF on best config", mrf)


# ── Final table ───────────────────────────────────────────────────────────
print(f"\n\n{'='*65}")
print("  SUMMARY TABLE")
print(f"{'='*65}")
df_res = pd.DataFrame(results_table)
df_res["ΔR²"] = df_res["R²"] - df_res.loc[0, "R²"]
df_res["ΔMAE"] = df_res["MAE"] - df_res.loc[0, "MAE"]
print(df_res.to_string(index=False, float_format=lambda x: f"{x:+.4f}" if abs(x) < 1 else f"{x:.1f}"))
print(f"{'='*65}\n")

# Save
out = "outputs/results/experiment_results.csv"
df_res.to_csv(out, index=False)
print(f"  Saved → {out}")
