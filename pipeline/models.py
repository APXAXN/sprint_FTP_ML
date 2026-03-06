"""
pipeline/models.py
Model factory — returns a dict of all 7 estimators.

sklearn models are wrapped in Pipeline([('scaler', RobustScaler()), ('model', ...)])
so that GridSearchCV / RandomizedSearchCV see prefixed param names (model__*).

The Keras NN is returned as a build function + compile config dict because
it needs the input dimension at instantiation time and doesn't fit cleanly
inside an sklearn Pipeline at this architecture complexity.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import xgboost as xgb

from pipeline.config import (
    DT_PARAM_GRID, RF_PARAM_DIST, RF_N_ITER,
    XGB_PARAM_DIST, XGB_N_ITER,
    NN_HIDDEN_LAYERS, NN_DROPOUT_RATES, NN_LEARNING_RATE,
    RANDOM_STATE, CV_FOLDS,
)


# ── Helpers ────────────────────────────────────────────────────────────────
def _pipe(estimator):
    """Wrap estimator in a RobustScaler pipeline."""
    return Pipeline([("scaler", RobustScaler()), ("model", estimator)])


# ── sklearn model factory ──────────────────────────────────────────────────
def get_sklearn_models(n_features: int = None):
    """
    Returns a dict of {name: estimator_or_search_object}.

    For models that require grid search the returned object IS the
    GridSearchCV / RandomizedSearchCV wrapper — call .fit() once and
    the best estimator is accessible via .best_estimator_.

    Parameters
    ----------
    n_features : unused here but kept for API symmetry with get_nn_model
    """
    models = {}

    # ── 1. OLS Linear Regression ───────────────────────────────────────────
    models["Linear"] = _pipe(LinearRegression())

    # ── 2. Ridge (built-in efficient CV) ──────────────────────────────────
    models["Ridge"] = _pipe(
        RidgeCV(
            alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            cv=CV_FOLDS,
            scoring="r2",
        )
    )

    # ── 3. Lasso (built-in efficient CV) ──────────────────────────────────
    models["Lasso"] = _pipe(
        LassoCV(
            alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            cv=CV_FOLDS,
            max_iter=10_000,
            random_state=RANDOM_STATE,
        )
    )

    # ── 4. Decision Tree with GridSearchCV ────────────────────────────────
    dt_pipe = _pipe(DecisionTreeRegressor(random_state=RANDOM_STATE))
    models["DecisionTree"] = GridSearchCV(
        dt_pipe,
        param_grid=DT_PARAM_GRID,
        cv=CV_FOLDS,
        scoring="r2",
        n_jobs=-1,
        refit=True,
        verbose=0,
    )

    # ── 5. Random Forest with RandomizedSearchCV ──────────────────────────
    rf_pipe = _pipe(RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))
    models["RandomForest"] = RandomizedSearchCV(
        rf_pipe,
        param_distributions=RF_PARAM_DIST,
        n_iter=RF_N_ITER,
        cv=CV_FOLDS,
        scoring="r2",
        n_jobs=-1,
        refit=True,
        random_state=RANDOM_STATE,
        verbose=0,
    )

    # ── 6. XGBoost with RandomizedSearchCV ───────────────────────────────
    # XGBoost is NOT wrapped in a Pipeline because we use eval_set for
    # early stopping during the best-estimator refit step in train.py.
    # RandomizedSearchCV handles the scaler-less feature matrix fine since
    # tree models are scale-invariant.
    xgb_estimator = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    models["XGBoost"] = RandomizedSearchCV(
        xgb_estimator,
        param_distributions=XGB_PARAM_DIST,
        n_iter=XGB_N_ITER,
        cv=CV_FOLDS,
        scoring="r2",
        n_jobs=-1,
        refit=True,
        random_state=RANDOM_STATE,
        verbose=0,
    )

    return models


# ── Keras Neural Network ───────────────────────────────────────────────────
def build_nn(n_features: int):
    """
    Build and compile the Keras MLP.  Called inside train.py after the
    input dimension is known.  Uses Keras 3 with PyTorch backend.

    Architecture: Dense(128) → BN → Dropout(0.3)
                  Dense(64)  → BN → Dropout(0.2)
                  Dense(32)  → ReLU
                  Dense(1)   → linear

    Returns
    -------
    model : compiled keras.Model
    """
    import keras
    import torch, numpy as np_

    # keras.utils.set_random_seed segfaults on Py3.13+torch; seed manually
    torch.manual_seed(RANDOM_STATE)
    np_.random.seed(RANDOM_STATE)

    inputs = keras.Input(shape=(n_features,))
    x = inputs

    for units, dropout in zip(NN_HIDDEN_LAYERS, NN_DROPOUT_RATES):
        x = keras.layers.Dense(units, activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        if dropout > 0:
            x = keras.layers.Dropout(dropout)(x)

    outputs = keras.layers.Dense(1, activation="linear")(x)
    model   = keras.Model(inputs, outputs, name="sprint_ftp_nn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=NN_LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model
