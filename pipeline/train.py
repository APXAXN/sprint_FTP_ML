"""
pipeline/train.py
Training loop for all 7 models on a given feature set.
Handles grid search, CV scoring, test-set evaluation, and model persistence.
"""

import time
import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_validate

from pipeline.config import (
    OUTPUT_MODELS, CV_FOLDS, RANDOM_STATE,
    NN_EPOCHS, NN_BATCH_SIZE, NN_PATIENCE, NN_VAL_SPLIT,
)
from pipeline.evaluate import compute_metrics


def _unwrap(estimator):
    """
    Return the underlying estimator from a GridSearchCV/RandomizedSearchCV
    wrapper, or the object itself if it's already a plain estimator.
    """
    if hasattr(estimator, "best_estimator_"):
        return estimator.best_estimator_
    return estimator


def _cv_score(estimator, X_train, y_train) -> dict:
    """Run 5-fold CV on train set and return mean MAE, RMSE, R²."""
    cv_results = cross_validate(
        estimator, X_train, y_train,
        cv=CV_FOLDS,
        scoring={
            "MAE":  "neg_mean_absolute_error",
            "RMSE": "neg_root_mean_squared_error",
            "R2":   "r2",
        },
        return_train_score=False,
        n_jobs=-1,
    )
    return {
        "MAE":  -cv_results["test_MAE"].mean(),
        "RMSE": -cv_results["test_RMSE"].mean(),
        "R2":    cv_results["test_R2"].mean(),
    }


def run_all_models(feature_set_name: str,
                   skip_nn: bool = False,
                   verbose: bool = True) -> dict:
    """
    Train all 7 models on the given feature set.

    Parameters
    ----------
    feature_set_name : one of 'sprint_only', 'sprint_eng',
                       'sprint_bio', 'full_submax'
    skip_nn          : if True, skip the Keras NN (useful for ablation runs)
    verbose          : print progress

    Returns
    -------
    results : dict
        {model_name: {
            cv_metrics:        {MAE, RMSE, R2},
            test_metrics:      {MAE, RMSE, R2, MAPE},
            test_predictions:  np.ndarray,
            best_params:       dict or None,
            fitted_model:      fitted estimator object,
            nn_history:        keras History object (NN only),
        }}
    Also exposes module-level  X_test, y_test, feature_names, scaler
    (set after the first call) so plots.py can access them.
    """
    from pipeline.data_loader import load_and_prepare
    from pipeline.models import get_sklearn_models, build_nn

    if verbose:
        print(f"\n{'='*65}")
        print(f"  TRAINING — feature set: {feature_set_name}")
        print(f"{'='*65}")

    X_train, X_test, y_train, y_test, feat_names, scaler, df_test_raw = \
        load_and_prepare(feature_set_name)

    OUTPUT_MODELS.mkdir(parents=True, exist_ok=True)
    results  = {}
    sk_models = get_sklearn_models(n_features=X_train.shape[1])

    # ── sklearn models ─────────────────────────────────────────────────────
    for name, estimator in sk_models.items():
        t0 = time.time()
        if verbose:
            print(f"\n  [{name}] fitting...", end=" ", flush=True)

        estimator.fit(X_train, y_train)
        best_est = _unwrap(estimator)

        # Best params (for grid-searched models)
        best_params = getattr(estimator, "best_params_", None)

        # CV score on the fitted best estimator
        try:
            cv_m = _cv_score(best_est, X_train, y_train)
        except Exception as e:
            cv_m = {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}
            print(f"(CV failed: {e})", end=" ")

        # XGBoost: re-fit best params with early stopping on a hold-out val set
        # (this gives a cleaner final model than the CV-fitted one)
        if name == "XGBoost" and best_params is not None:
            import xgboost as xgb
            n_val   = int(len(X_train) * 0.15)
            X_val   = X_train[-n_val:];   y_val   = y_train[-n_val:]
            X_tr2   = X_train[:-n_val];   y_tr2   = y_train[:-n_val]
            xgb_final = xgb.XGBRegressor(
                **best_params,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=RANDOM_STATE,
                verbosity=0,
                early_stopping_rounds=NN_PATIENCE,
            )
            xgb_final.fit(
                X_tr2, y_tr2,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            best_est = xgb_final

        # Test-set evaluation
        y_pred    = best_est.predict(X_test)
        test_m    = compute_metrics(y_test, y_pred)

        # Persist model
        model_path = OUTPUT_MODELS / f"{name}_{feature_set_name}.joblib"
        joblib.dump(best_est, model_path)

        elapsed = time.time() - t0
        if verbose:
            print(
                f"done ({elapsed:.1f}s)  "
                f"CV R²={cv_m['R2']:.3f}  "
                f"test R²={test_m['R2']:.3f}  "
                f"test MAE={test_m['MAE']:.1f}W"
            )
            if best_params:
                print(f"    best params: {best_params}")

        results[name] = {
            "cv_metrics":       cv_m,
            "test_metrics":     test_m,
            "test_predictions": y_pred,
            "best_params":      best_params,
            "fitted_model":     best_est,
            # Flatten for easy DataFrame access
            **{f"cv_{k}":   v for k, v in cv_m.items()},
            **{f"test_{k}": v for k, v in test_m.items()},
        }

    # ── Keras Neural Network (Keras 3 / PyTorch backend) ──────────────────
    if not skip_nn:
        import keras
        from pipeline.config import NN_LEARNING_RATE

        name = "NeuralNet"
        t0   = time.time()
        if verbose:
            print(f"\n  [{name}] fitting...", end=" ", flush=True)

        import torch as _torch
        _torch.manual_seed(RANDOM_STATE)   # keras.utils.set_random_seed segfaults on Py3.13
        nn = build_nn(n_features=X_train.shape[1])

        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=NN_PATIENCE,
                restore_best_weights=True,
                monitor="val_loss",
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=10, min_lr=1e-6, monitor="val_loss",
            ),
        ]

        history = nn.fit(
            X_train, y_train,
            epochs=NN_EPOCHS,
            batch_size=NN_BATCH_SIZE,
            validation_split=NN_VAL_SPLIT,
            callbacks=callbacks,
            verbose=0,
        )

        y_pred  = nn.predict(X_test, verbose=0).flatten()
        test_m  = compute_metrics(y_test, y_pred)

        # Try CV-style score (manual 5-fold) for consistency
        try:
            cv_m = _nn_cv_score(X_train, y_train, X_train.shape[1])
        except Exception:
            # Fallback: use val loss from training as proxy
            cv_m = {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}

        # Persist model — use save_weights (nn.save() segfaults on Py3.13+torch)
        nn_weights_path = OUTPUT_MODELS / f"NeuralNet_{feature_set_name}.weights.h5"
        nn.save_weights(str(nn_weights_path))

        elapsed = time.time() - t0
        if verbose:
            print(
                f"done ({elapsed:.1f}s)  "
                f"epochs={len(history.history['loss'])}  "
                f"test R²={test_m['R2']:.3f}  "
                f"test MAE={test_m['MAE']:.1f}W"
            )

        results[name] = {
            "cv_metrics":       cv_m,
            "test_metrics":     test_m,
            "test_predictions": y_pred,
            "best_params":      None,
            "fitted_model":     nn,
            "nn_history":       history,
            **{f"cv_{k}":   v for k, v in cv_m.items()},
            **{f"test_{k}": v for k, v in test_m.items()},
        }

    # Attach split data for downstream use
    results["__meta__"] = {
        "X_test":       X_test,
        "y_test":       y_test,
        "X_train":      X_train,
        "y_train":      y_train,
        "feature_names": feat_names,
        "scaler":        scaler,
        "df_test_raw":   df_test_raw,
    }

    if verbose:
        _print_summary(results, feature_set_name)

    return results


def _nn_cv_score(X_train, y_train, n_features, folds=5):
    """Manual 5-fold CV for NN — lightweight (fewer epochs, no early stopping)."""
    from pipeline.models import build_nn
    import keras

    fold_size = len(X_train) // folds
    maes, r2s = [], []

    for i in range(folds):
        val_idx   = slice(i * fold_size, (i + 1) * fold_size)
        train_idx = list(range(0, i * fold_size)) + \
                    list(range((i + 1) * fold_size, len(X_train)))

        Xtr, ytr = X_train[train_idx], y_train[train_idx]
        Xva, yva = X_train[val_idx],   y_train[val_idx]

        nn = build_nn(n_features)
        nn.fit(Xtr, ytr, epochs=50, batch_size=64, verbose=0)
        y_pred = nn.predict(Xva, verbose=0).flatten()
        metrics = compute_metrics(yva, y_pred)
        maes.append(metrics["MAE"])
        r2s.append(metrics["R2"])
        keras.backend.clear_session()

    return {"MAE": np.mean(maes), "RMSE": np.nan, "R2": np.mean(r2s)}


def _print_summary(results: dict, fs_name: str):
    """Print a final summary table to stdout."""
    print(f"\n{'─'*65}")
    print(f"  RESULTS SUMMARY — {fs_name}")
    print(f"  {'Model':<16} {'CV R²':>8} {'CV MAE':>8} {'Test R²':>9} {'Test MAE':>10}")
    print(f"{'─'*65}")
    for name, d in results.items():
        if name == "__meta__": continue
        cv   = d.get("cv_metrics",   {})
        test = d.get("test_metrics", {})
        print(
            f"  {name:<16} "
            f"{cv.get('R2', float('nan')):>8.3f} "
            f"{cv.get('MAE', float('nan')):>8.1f} "
            f"{test.get('R2', float('nan')):>9.3f} "
            f"{test.get('MAE', float('nan')):>10.1f}"
        )
    print(f"{'─'*65}")
