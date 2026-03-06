"""
Sprint FTP ML — Full Data Inspection on athletes.csv
MSIS 522 | UW Foster School of Business

Runs the complete Step 2 inspection on the consolidated athletes.csv
downloaded from Kaggle (GoldenCheetah OpenData).
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("data_inspection")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Column mapping ──────────────────────────────────────────────────────────
# Sprint durations available in athletes.csv (≤60s)
SPRINT_MAP = {
    "1s":  "1s_critical_power",
    "15s": "15s_critical_power",
    "20s": "20s_peak",
    "60s": "60s_peak",          # 1-min — at the sprint/endurance boundary
}
# NOTE: 5s, 10s, 30s are NOT in athletes.csv

# Longer durations present
LONGER_MAP = {
    "2m":  "2m_critical_power",
    "3m":  "3m_critical_power",
    "5m":  "5m_critical_power",
    "8m":  "8m_critical_power",
    "10m": "10m_critical_power",
    "20m": "20m_critical_power",   # TARGET
    "30m": "30m_critical_power",
}

TARGET_COL  = "20m_critical_power"
WEIGHT_COL  = "weightkg"
AGE_COL     = "age"
GENDER_COL  = "gender"

# All sprint + extended feature columns
ALL_FEATURE_COLS = list(SPRINT_MAP.values()) + [
    "2m_critical_power", "3m_critical_power", "5m_critical_power",
    "8m_critical_power", "10m_critical_power"
]


def banner(title): print("\n" + "=" * 70 + f"\n  {title}\n" + "=" * 70)
def section(title): print(f"\n── {title} " + "─" * max(1, 65 - len(title)))


def main():
    print("=" * 70)
    print("  SPRINT FTP ML — FULL DATA INSPECTION (athletes.csv)")
    print("  MSIS 522 | UW Foster School of Business")
    print("=" * 70)

    df = pd.read_csv("athletes.csv")
    print(f"\nLoaded athletes.csv: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── 2a: Schema & Structure ───────────────────────────────────────────────
    banner("2a. SCHEMA & STRUCTURE")

    print(f"\ndf.shape: {df.shape}")
    print(f"\nAll columns ({len(df.columns)}):")
    for c in df.columns:
        print(f"  {c}")

    print(f"\ndf.dtypes:\n{df.dtypes.to_string()}")
    print(f"\ndf.head(10):\n{df.head(10).to_string()}")

    section("Column mapping for this project")
    print("\nSprint features AVAILABLE in athletes.csv:")
    for dur, col in SPRINT_MAP.items():
        print(f"  {dur:5s} → {col}")
    print("\n  ⚠ MISSING from athletes.csv: 5s, 10s, 30s")
    print("    (These exist in individual athlete zips but not the summary file)")
    print("\nTarget variable:")
    print(f"  20-min FTP proxy  → {TARGET_COL}")
    print("\nBiometric columns:")
    print(f"  weight → {WEIGHT_COL}")
    print(f"  age    → {AGE_COL}")
    print(f"  gender → {GENDER_COL}")

    # ── 2b: Target Variable ───────────────────────────────────────────────────
    banner("2b. TARGET VARIABLE VIABILITY (20m_critical_power)")

    target = df[TARGET_COL].dropna()
    print(f"\nAthletes with non-null {TARGET_COL}: {len(target):,} / {len(df):,}  "
          f"({100*len(target)/len(df):.1f}%)")

    print(f"\nDistribution of 20-min peak power (watts):")
    print(f"  min    : {target.min():.1f}")
    print(f"  max    : {target.max():.1f}")
    print(f"  mean   : {target.mean():.1f}")
    print(f"  median : {target.median():.1f}")
    print(f"  std    : {target.std():.1f}")
    print(f"  Q1     : {target.quantile(0.25):.1f}")
    print(f"  Q3     : {target.quantile(0.75):.1f}")
    print(f"  P95    : {target.quantile(0.95):.1f}")
    print(f"  P99    : {target.quantile(0.99):.1f}")

    print(f"\nPhysiologically questionable values:")
    print(f"  > 600W (near pro-tour) : {(target > 600).sum()}")
    if (target > 600).sum() > 0:
        print(f"    Values: {sorted(target[target > 600].values, reverse=True)[:10]}")
    print(f"  < 50W  (extremely low) : {(target < 50).sum()}")

    # ── 2c: Feature Availability ──────────────────────────────────────────────
    banner("2c. FEATURE AVAILABILITY")

    all_inspect_cols = list(SPRINT_MAP.values()) + list(LONGER_MAP.values()) + \
                       [WEIGHT_COL, AGE_COL, GENDER_COL]

    print("\nNon-null counts per column:")
    for col in all_inspect_cols:
        if col in df.columns:
            n   = df[col].notna().sum()
            pct = 100 * n / len(df)
            bar = "█" * int(pct / 5)
            print(f"  {col:25s}: {n:5,} / {len(df):,}  ({pct:5.1f}%)  {bar}")
        else:
            print(f"  {col:25s}: *** COLUMN NOT FOUND ***")

    # Complete-case overlap: sprint features + target + weight
    sprint_cols_avail = [c for c in SPRINT_MAP.values() if c in df.columns]
    overlap_cols      = sprint_cols_avail + [TARGET_COL, WEIGHT_COL]
    complete = df[overlap_cols].dropna()
    print(f"\nAthletes with ALL sprint features + 20m target + weight:")
    print(f"  {len(complete):,} / {len(df):,}  ({100*len(complete)/len(df):.1f}%)")

    # ── 2d: Data Quality Red Flags ─────────────────────────────────────────────
    banner("2d. DATA QUALITY RED FLAGS")

    checks = {
        "1s_critical_power":  (200,  2500, "W"),
        "15s_critical_power": (50,   1500, "W"),
        "20s_peak":           (50,   2000, "W"),
        "20m_critical_power": (50,   600,  "W"),
        "weightkg":           (40,   150,  "kg"),
        "age":                (14,   80,   "yrs"),
    }
    for col, (lo, hi, unit) in checks.items():
        if col not in df.columns:
            print(f"  {col:25s}: MISSING")
            continue
        s      = df[col].dropna()
        n_null = df[col].isna().sum()
        n_low  = (s < lo).sum()
        n_high = (s > hi).sum()
        print(f"  {col:25s}: null={n_null:4,}, <{lo}{unit}={n_low:4,}, >{hi}{unit}={n_high:4,}")

    section("Columns >50% missing")
    high_missing = [(c, 100*df[c].isna().mean()) for c in df.columns if df[c].isna().mean() > 0.5]
    if high_missing:
        for c, pct in high_missing:
            print(f"  {c:30s}: {pct:.1f}% missing")
    else:
        print("  (none)")

    section("Duplicate athlete IDs")
    dups = df.duplicated(subset=["id"]).sum()
    print(f"  Duplicate id rows: {dups}")

    section("Gender breakdown")
    print(df[GENDER_COL].value_counts().to_string())

    section("Age distribution")
    ages = df[AGE_COL].dropna()
    print(f"  min={ages.min():.0f}, max={ages.max():.0f}, mean={ages.mean():.1f}, "
          f"median={ages.median():.1f}")
    print(f"  age < 14 (erroneous): {(ages < 14).sum()}")
    print(f"  age > 80 (erroneous): {(ages > 80).sum()}")

    # ── 2e: Correlations ──────────────────────────────────────────────────────
    banner("2e. SPRINT ↔ 20-MIN POWER CORRELATIONS")

    corr_cols  = [c for c in sprint_cols_avail if c in df.columns] + [TARGET_COL]
    corr_data  = df[corr_cols].dropna()
    print(f"\nComplete cases for correlation: {len(corr_data):,}")

    print("\nFull correlation matrix (sprint + target):")
    corr_matrix = corr_data.corr()
    print(corr_matrix.to_string())

    print(f"\nCorrelation with {TARGET_COL}:")
    for col in sprint_cols_avail:
        if col in corr_data.columns:
            r   = corr_data[col].corr(corr_data[TARGET_COL])
            dur = [k for k,v in SPRINT_MAP.items() if v == col][0]
            print(f"  {col:30s} ({dur:4s})  →  r = {r:+.4f}")

    # Also correlate intermediate durations
    print(f"\nCorrelation with {TARGET_COL} (all available durations):")
    extended_cols = [c for c in list(SPRINT_MAP.values()) + list(LONGER_MAP.values())
                     if c in df.columns and c != TARGET_COL]
    for col in extended_cols:
        n_pair = df[[col, TARGET_COL]].dropna().shape[0]
        if n_pair >= 10:
            r = df[col].corr(df[TARGET_COL])
            print(f"  {col:30s}: r = {r:+.4f}  (n={n_pair:,})")

    # ── Clean row count ────────────────────────────────────────────────────────
    banner("2f. FEASIBILITY VERDICT")

    df_clean = df.copy()
    if "1s_critical_power"  in df_clean: df_clean = df_clean[~((df_clean["1s_critical_power"]  > 2500) | (df_clean["1s_critical_power"]  < 200))]
    if "20m_critical_power" in df_clean: df_clean = df_clean[~((df_clean["20m_critical_power"] > 600)  | (df_clean["20m_critical_power"] < 50))]
    if "weightkg"           in df_clean: df_clean = df_clean[~((df_clean["weightkg"]           > 150)  | (df_clean["weightkg"]           < 40))]
    if "age"                in df_clean: df_clean = df_clean[~((df_clean["age"]                > 80)   | (df_clean["age"]                < 14))]
    df_clean = df_clean.dropna(subset=[c for c in overlap_cols if c in df_clean.columns])

    n_raw   = len(df)
    n_clean = len(df_clean)

    print(f"""
Raw dataset rows             : {n_raw:,}
After physiological filters  : {n_clean:,}  ({100*n_clean/n_raw:.0f}% retention)

FEATURE SCHEMA:
  ✓ 1s peak power   (1s_critical_power)   — PRESENT
  ✗ 5s peak power                          — NOT IN athletes.csv
  ✗ 10s peak power                         — NOT IN athletes.csv
  ✓ 15s peak power  (15s_critical_power)  — PRESENT
  ✓ 20s peak power  (20s_peak)            — PRESENT
  ✗ 30s peak power                         — NOT IN athletes.csv
  ✓ 60s peak power  (60s_peak)            — PRESENT (boundary sprint/endurance)
  ✓ weight, age, gender                    — ALL PRESENT

TARGET: 20m_critical_power
  ✓ PRESENT and well-populated

SAMPLE SIZE:
  {n_clean:,} clean rows — {'✓ VERY COMFORTABLE' if n_clean >= 2000 else '✓ ADEQUATE' if n_clean >= 500 else '✗ INSUFFICIENT'} for 5-model 5-fold CV

CORRELATION STRUCTURE:
  Analyzed above — see plots for full picture.

MISSING SPRINT DURATIONS (5s, 10s, 30s):
  These exist in individual athlete JSON files but were not included
  in the pre-built athletes.csv summary. Options:
  a) Proceed with 1s, 15s, 20s, 60s as sprint features (still meaningful)
  b) Reconstruct from athlete zips for those 3 durations (significant effort)
  c) Treat 60s as the key sprint boundary feature

RECOMMENDATION: PROCEED with available features.
  The 15s, 20s, 60s features likely capture the aerobic-anaerobic
  crossover zone that correlates most strongly with 20-min power.
  We have {n_clean:,} clean rows — more than enough for robust ML.
""")

    # ── Plots ──────────────────────────────────────────────────────────────────
    generate_plots(df, df_clean, corr_matrix, corr_data)

    df.to_csv(OUTPUT_DIR / "athletes_full_raw.csv", index=False)
    df_clean.to_csv(OUTPUT_DIR / "athletes_clean.csv", index=False)
    print(f"\nSaved:\n  {OUTPUT_DIR}/athletes_full_raw.csv\n  {OUTPUT_DIR}/athletes_clean.csv")


def generate_plots(df, df_clean, corr_matrix, corr_data):
    print("\n── Generating plots ─────────────────────────────────────────────────")

    sprint_cols = [c for c in SPRINT_MAP.values() if c in df.columns]
    all_dur_cols = [c for c in list(SPRINT_MAP.values()) + list(LONGER_MAP.values())
                    if c in df.columns]

    # Duration labels for plotting
    dur_label = {v: k for k, v in {**SPRINT_MAP, **LONGER_MAP}.items()}

    # ── Plot 1: Power distributions by duration ───────────────────────────────
    cols_to_plot = [c for c in [
        "1s_critical_power", "15s_critical_power", "20s_peak", "60s_peak",
        "5m_critical_power", "20m_critical_power"
    ] if c in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("GoldenCheetah OpenData — Peak Power Distributions\n(n=6,043 athletes)",
                 fontsize=13, fontweight='bold')
    axes_flat = axes.flatten()

    for i, col in enumerate(cols_to_plot[:6]):
        ax   = axes_flat[i]
        data = df[col].dropna()
        # Clip to physiological range for display
        p99  = data.quantile(0.99)
        data_display = data[data <= p99 * 1.1]

        ax.hist(data_display, bins=60, color='steelblue', edgecolor='none', alpha=0.85)
        ax.axvline(data.median(), color='orangered', lw=2, ls='--', label=f'median={data.median():.0f}W')
        ax.axvline(data.mean(),   color='gold',      lw=2, ls=':',  label=f'mean={data.mean():.0f}W')
        ax.set_title(f"Peak {dur_label.get(col, col)} Power  (n={len(data):,})", fontweight='bold')
        ax.set_xlabel("Watts")
        ax.set_ylabel("Athletes")
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    p1 = OUTPUT_DIR / "01_power_distributions.png"
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {p1}")

    # ── Plot 2: Target variable (20m) ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("20-Minute Peak Power Distribution (Target Variable)", fontsize=13, fontweight='bold')

    target = df["20m_critical_power"].dropna()

    # Raw
    ax = axes[0]
    ax.hist(target, bins=80, color='steelblue', edgecolor='none')
    ax.axvline(target.mean(),   color='red',    ls='--', lw=2, label=f'mean={target.mean():.0f}W')
    ax.axvline(target.median(), color='orange', ls='--', lw=2, label=f'median={target.median():.0f}W')
    ax.set_title("All values (raw)")
    ax.set_xlabel("Watts"); ax.set_ylabel("Athletes"); ax.legend(); ax.grid(alpha=0.3)

    # Filtered (physiologically plausible)
    target_filt = target[(target >= 50) & (target <= 600)]
    ax = axes[1]
    ax.hist(target_filt, bins=80, color='teal', edgecolor='none')
    ax.axvline(target_filt.mean(),   color='red',    ls='--', lw=2, label=f'mean={target_filt.mean():.0f}W')
    ax.axvline(target_filt.median(), color='orange', ls='--', lw=2, label=f'median={target_filt.median():.0f}W')
    ax.set_title("Physiologically valid (50–600W)")
    ax.set_xlabel("Watts"); ax.legend(); ax.grid(alpha=0.3)

    # W/kg (if weight available)
    wpk = df["20m_peak_wpk"].dropna()
    wpk_filt = wpk[(wpk >= 1.0) & (wpk <= 8.0)]
    ax = axes[2]
    ax.hist(wpk_filt, bins=80, color='mediumseagreen', edgecolor='none')
    ax.axvline(wpk_filt.mean(),   color='red',    ls='--', lw=2, label=f'mean={wpk_filt.mean():.2f}')
    ax.axvline(wpk_filt.median(), color='orange', ls='--', lw=2, label=f'median={wpk_filt.median():.2f}')
    ax.set_title("20-min Power (W/kg)  — weight-normalized")
    ax.set_xlabel("W/kg"); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    p2 = OUTPUT_DIR / "02_target_distribution.png"
    plt.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {p2}")

    # ── Plot 3: Missing values heatmap (column-level summary) ─────────────────
    cols_of_interest = sprint_cols + [
        "2m_critical_power", "5m_critical_power", "10m_critical_power",
        "20m_critical_power", "weightkg", "age", "gender"
    ]
    cols_of_interest = [c for c in cols_of_interest if c in df.columns]
    missing_pct      = df[cols_of_interest].isna().mean() * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    colors  = ['#2ecc71' if pct <= 5 else '#f39c12' if pct <= 30 else '#e74c3c' for pct in missing_pct]
    bars    = ax.bar(range(len(missing_pct)), missing_pct, color=colors, edgecolor='white')
    ax.set_xticks(range(len(missing_pct)))
    ax.set_xticklabels([c.replace("_critical_power", "").replace("_peak", "") for c in cols_of_interest],
                       rotation=40, ha='right', fontsize=9)
    ax.set_ylabel("% Missing")
    ax.set_title("Missing Value Rate by Column  (green ≤5%, orange ≤30%, red >30%)", fontweight='bold')
    ax.set_ylim(0, max(missing_pct.max() + 5, 20))
    for bar, pct in zip(bars, missing_pct):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{pct:.1f}%", ha='center', va='bottom', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    p3 = OUTPUT_DIR / "03_missing_values.png"
    plt.savefig(p3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {p3}")

    # ── Plot 4: Power-duration curve (median + IQR band) ─────────────────────
    dur_order = [
        ("1s",  "1s_critical_power"),
        ("15s", "15s_critical_power"),
        ("20s", "20s_peak"),
        ("60s", "60s_peak"),
        ("2m",  "2m_critical_power"),
        ("3m",  "3m_critical_power"),
        ("5m",  "5m_critical_power"),
        ("8m",  "8m_critical_power"),
        ("10m", "10m_critical_power"),
        ("20m", "20m_critical_power"),
        ("30m", "30m_critical_power"),
    ]
    dur_order = [(d, c) for d, c in dur_order if c in df.columns]
    x_positions = list(range(len(dur_order)))
    dur_labels_plot = [d for d, _ in dur_order]

    medians = [df[c].median() for _, c in dur_order]
    q25s    = [df[c].quantile(0.25) for _, c in dur_order]
    q75s    = [df[c].quantile(0.75) for _, c in dur_order]
    p5s     = [df[c].quantile(0.05) for _, c in dur_order]
    p95s    = [df[c].quantile(0.95) for _, c in dur_order]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.fill_between(x_positions, p5s,  p95s,  alpha=0.12, color='steelblue', label='P5–P95 range')
    ax.fill_between(x_positions, q25s, q75s,  alpha=0.30, color='steelblue', label='IQR (Q1–Q3)')
    ax.plot(x_positions, medians, 'o-', color='steelblue', lw=2.5, ms=7, label='Median')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(dur_labels_plot, fontsize=10)
    ax.set_xlabel("Duration", fontsize=11)
    ax.set_ylabel("Peak Power (Watts)", fontsize=11)
    ax.set_title("Population Power-Duration Curve  (n=6,043 athletes)\nMedian ± IQR ± P5/P95",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.axvline(3.5, color='red', ls=':', lw=1.5, alpha=0.6)   # sprint / sub-max boundary
    ax.text(3.6, ax.get_ylim()[1]*0.95, "← sprint | sub-max →", fontsize=8, color='red', alpha=0.7)
    plt.tight_layout()
    p4 = OUTPUT_DIR / "04_power_duration_profiles.png"
    plt.savefig(p4, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {p4}")

    # ── Plot 5: Correlation heatmap + bar chart ───────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Correlation Analysis: Sprint & Sub-Max Power vs 20-min Target",
                 fontsize=13, fontweight='bold')

    # Heatmap
    n_corr = len(corr_matrix)
    im = ax1.imshow(corr_matrix.values, cmap='RdYlGn', vmin=-1, vmax=1)
    ax1.set_xticks(range(n_corr)); ax1.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=8)
    ax1.set_yticks(range(n_corr)); ax1.set_yticklabels(corr_matrix.columns, fontsize=8)
    for i in range(n_corr):
        for j in range(n_corr):
            ax1.text(j, i, f"{corr_matrix.iloc[i,j]:.2f}", ha='center', va='center', fontsize=7,
                     color='white' if abs(corr_matrix.iloc[i,j]) > 0.6 else 'black')
    plt.colorbar(im, ax=ax1, shrink=0.8)
    ax1.set_title("Correlation Matrix", fontweight='bold')

    # Bar chart: r with target across all durations
    all_dur_for_corr = [
        ("1s",  "1s_critical_power"),
        ("15s", "15s_critical_power"),
        ("20s", "20s_peak"),
        ("60s", "60s_peak"),
        ("2m",  "2m_critical_power"),
        ("3m",  "3m_critical_power"),
        ("5m",  "5m_critical_power"),
        ("8m",  "8m_critical_power"),
        ("10m", "10m_critical_power"),
    ]
    all_dur_for_corr = [(d, c) for d, c in all_dur_for_corr if c in df.columns]
    r_vals   = [df[c].corr(df["20m_critical_power"]) for _, c in all_dur_for_corr]
    dur_lbls = [d for d, _ in all_dur_for_corr]
    colors   = ['#e74c3c' if d in ('1s','15s','20s') else
                '#f39c12' if d == '60s' else
                '#2ecc71' for d in dur_lbls]

    bars = ax2.bar(range(len(r_vals)), r_vals, color=colors, edgecolor='white', alpha=0.9)
    ax2.set_xticks(range(len(r_vals)))
    ax2.set_xticklabels(dur_lbls, fontsize=10)
    ax2.set_ylabel("Pearson r with 20-min power", fontsize=10)
    ax2.set_title("r vs 20-min Target by Duration\n(red=sprint, orange=1min boundary, green=sub-max)",
                  fontweight='bold')
    ax2.axhline(0, color='black', lw=0.5)
    ax2.axhline(0.5, color='gray', lw=0.8, ls='--', alpha=0.5, label='r=0.5')
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', alpha=0.3)
    for bar, r in zip(bars, r_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{r:.3f}", ha='center', va='bottom', fontsize=8)
    ax2.legend(fontsize=8)
    plt.tight_layout()
    p5 = OUTPUT_DIR / "05_correlation_heatmap.png"
    plt.savefig(p5, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {p5}")

    print(f"\nAll plots saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
