"""
pipeline/features.py
SprintFeatureEngineer — transforms the clean athlete dataframe by adding
engineered sprint-signature features and dropping redundant columns.

All formulas are documented in config.py.  Division-by-zero is protected
by clipping every denominator to a minimum of 1W before dividing.
"""

import numpy as np
import pandas as pd


class SprintFeatureEngineer:
    """
    Stateless transformer: call .transform(df) on the raw clean dataframe
    and receive a new dataframe with additional engineered columns.

    Does NOT scale, split, or drop the target column.
    """

    # Columns already in the CSV that are redundant / out-of-scope
    _COLS_TO_DROP = [
        "1m_peak_wpk", "5m_peak_wpk", "10m_peak_wpk",
        "20m_peak_wpk", "30m_peak_wpk",
        "run", "swim", "other",          # low relevance for cycling prediction
    ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : raw athletes_clean.csv as a DataFrame

        Returns
        -------
        df_out : copy of df with engineered columns added and
                 redundant columns removed
        """
        df_out = df.copy()

        # ── 1. Gender encoding ───────────────────────────────────────────────
        df_out["gender_encoded"] = (df_out["gender"] == "F").astype(int)

        # ── 2. Shorthand references (clipped to ≥1 for safe division) ────────
        p1s  = df_out["1s_critical_power"].clip(lower=1)
        p5s  = df_out["5s_critical_power"].clip(lower=1)
        p15s = df_out["15s_critical_power"].clip(lower=1)
        p30s = df_out["30s_critical_power"].clip(lower=1)
        wt   = df_out["weightkg"].clip(lower=1)

        # ── 3. Decay ratios ───────────────────────────────────────────────────
        # How much power is retained as effort duration increases.
        # A pure sprinter retains more; an FTP specialist loses power fast early.
        df_out["fatigue_index"]    = p30s / p1s      # full-sprint power retention
        df_out["early_decay"]      = p5s  / p1s      # first 5s drop-off
        df_out["mid_decay"]        = p15s / p5s      # 5s → 15s drop-off
        df_out["late_decay"]       = p30s / p15s     # 15s → 30s (most aerobic phase)

        # ── 4. Anaerobic reserve (watts) ──────────────────────────────────────
        # Raw watt difference between 1s peak and 30s sustained.
        # High value = large anaerobic capacity relative to sustained power.
        df_out["anaerobic_reserve"] = (
            df_out["1s_critical_power"] - df_out["30s_critical_power"]
        )

        # ── 5. Decay curvature ────────────────────────────────────────────────
        # Second finite difference on log-power curve at three points.
        # Positive → concave (fast early drop, slow later).
        # Negative → convex (slower early, steeper late).
        # Zero     → log-linear decay.
        log1  = np.log(p1s)
        log15 = np.log(p15s)
        log30 = np.log(p30s)
        df_out["decay_curvature"] = log1 - 2 * log15 + log30

        # ── 6. W/kg sprint features ───────────────────────────────────────────
        # Normalise by body weight — allows fairer comparison across athlete sizes.
        df_out["sprint_wpk_1s"]  = df_out["1s_critical_power"]  / wt
        df_out["sprint_wpk_15s"] = df_out["15s_critical_power"] / wt
        df_out["sprint_wpk_30s"] = df_out["30s_critical_power"] / wt

        # ── 7. Drop redundant / out-of-scope columns ──────────────────────────
        drop = [c for c in self._COLS_TO_DROP if c in df_out.columns]
        df_out = df_out.drop(columns=drop)

        return df_out

    def feature_summary(self, df_transformed: pd.DataFrame) -> pd.DataFrame:
        """Return a descriptive stats table for just the engineered columns."""
        from pipeline.config import ENGINEERED_COLS
        eng_cols = [c for c in ENGINEERED_COLS if c in df_transformed.columns]
        return df_transformed[eng_cols].describe().T
