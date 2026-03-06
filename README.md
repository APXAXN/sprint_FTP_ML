# Sprint FTP ML

**MSIS 522 — Machine Learning | UW Foster School of Business**

> *Can a 30-second sprint tell you how strong you are for a 20-minute climb?*

Predicting 20-minute peak cycling power (a proxy for FTP) from sprint-duration power outputs (≤30s) and basic athlete biometrics using the GoldenCheetah OpenData dataset.

---

## Project Overview

Functional Threshold Power (FTP) is the gold standard for measuring cycling fitness, but testing it requires a hard 20-minute all-out effort. This project asks: **can we predict FTP from a short sprint instead?**

Using mean-maximal power data from 6,043 cyclists (cleaned to 4,768), we train regression models to predict 20-minute peak power from:
- Sprint power at 1s, 5s, 10s, 15s, 20s, and 30s
- Biometrics: weight, age, gender

## Dataset

**Source:** [GoldenCheetah OpenData](https://www.kaggle.com/datasets/markliversedge/goldencheetah-opendata-athlete-activity-and-mmp) (Kaggle)

| File | Size | Description |
|------|------|-------------|
| `athletes.csv` | 1.8 MB | Athlete-level summary — sprint features + target |
| `activities_mmp.csv` | 556 MB | Per-activity MMP curves — used to recover 5s/10s/20s/30s |

> `activities_mmp.csv` is excluded from git (556 MB). Run `scripts/download_data.sh` or see **Setup** below.

### Key Data Findings

| Sprint Duration | r with 20-min power |
|----------------|---------------------|
| 1s  | +0.44 |
| 5s  | +0.53 |
| 10s | +0.59 |
| 15s | +0.60 |
| 20s | +0.64 |
| **30s** | **+0.68** |

Correlation increases monotonically with sprint duration — 30s power is the strongest sprint predictor of aerobic capacity.

## Repository Structure

```
Sprint_FTP_ML/
├── athletes.csv                      # Raw athlete summary (1.8 MB)
├── inspect_data.py                   # Extraction pipeline from raw athlete zips
├── inspect_athletes_csv.py           # Full inspection of athletes.csv
├── data_inspection/
│   ├── athletes_clean.csv            # Quality-filtered dataset (4,768 rows)
│   ├── 01_power_distributions.png
│   ├── 02_target_distribution.png
│   ├── 03_missing_values.png
│   ├── 04_power_duration_profiles.png
│   └── 05_correlation_heatmap.png
└── sprint_signature_project_plan.md  # Project plan
```

## Setup

### 1. Install dependencies
```bash
pip install pandas numpy matplotlib scikit-learn kaggle
```

### 2. Get the data
```bash
# Place your kaggle.json at ~/.kaggle/kaggle.json, then:
kaggle datasets download -d markliversedge/goldencheetah-opendata-athlete-activity-and-mmp \
  --file "athletes.csv/athletes.csv" --path .

kaggle datasets download -d markliversedge/goldencheetah-opendata-athlete-activity-and-mmp \
  --file "activities_mmp.csv/activities_mmp.csv" --path .

unzip athletes.csv.zip && unzip activities_mmp.csv.zip
```

### 3. Run inspection
```bash
python inspect_athletes_csv.py    # Full data inspection + plots
```

## Data Quality Notes

- **Age corruption**: 549 athletes dropped (values like -965, 2020 from bad `yob` data — irrecoverable)
- **5s/10s/20s/30s features**: Not in `athletes.csv`; recovered from `activities_mmp.csv` by filtering to cycling activities (`avg_power` 20–800W, `duration` >5 min)
- **Cross-sport contamination**: `20s_peak` and `60s_peak` columns in `athletes.csv` include all sports (r ≈ 0.04 with target) — dropped in favor of cycling-specific values from `activities_mmp.csv`
- **Quality filters**: `20m_critical_power` 50–600W, `weightkg` 40–150kg, physiological bounds per sprint duration

## Models (Planned)

- Linear Regression (baseline)
- Ridge / Lasso
- Random Forest
- Gradient Boosting (XGBoost / LightGBM)
- 5-fold cross-validation, 70/30 train-test split

## Author

Nathan Fitzgerald — UW Foster School of Business, MSIS 522
