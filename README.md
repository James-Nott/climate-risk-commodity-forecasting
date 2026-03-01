# Soil Moisture Drought Index (SMDI) Modelling Toolkit
_Compute SMDI from ESA soil-moisture time series, combine with other features, train ML models, and write out results._

**Author:** James Nott  
**Date:** 2025-08-11

---

## Overview
This project builds an explanatory variable—**SMDI (Soil Moisture Deficit Index)**—from ESA daily soil-moisture data and develops ML models utilising this index.
Project:
1) Masks ESA time series to national boundaries  
2) Aggregates to weekly statistics (min/median/mean/max)  
3) Computes SMDI and an extremes summary  
4) Merges SMDI with other features into a modelling dataset  
5) Trains machine-learning models (LSTM & XGBoost) and visualises errors.

---

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Data Pipeline](#data-pipeline)
  - [A) Mask ESA data to national boundaries](#a-mask-esa-data-to-national-boundaries)
  - [B) Weekly statistics (min/median/mean/max)](#b-weekly-statistics-minmedianmeanmax)
  - [C) Compute SMDI and extremes summary](#c-compute-smdi-and-extremes-summary)
  - [D) Build the modelling dataset](#d-build-the-modelling-dataset)
- [Modelling](#modelling)
  - [LSTM (time-series)](#lstm-time-series)
  - [XGBoost baseline](#xgboost-baseline)
- [Visualisation](#visualisation)
- [Requirements](#requirements)
- [Contact](#contact)

---

## Repository Structure

```
.
├── data/
│   └── dataset_complete.csv                # Combined feature set for ML
├── james_nott_csc8099/
│   ├── __init__.py
│   ├── config.py                           # Project paths
│   ├── data_visualisation/
│   │   └── create_error_vis.py             # Plots regime-based errors
│   ├── lstm_scripts/
│   │   ├── __init__.py
│   │   ├── train_test_split.py             # LSTM with train/test split
│   │   └── walk_forward.py                 # LSTM with walk-forward validation
│   ├── smdi_creation/
│   │   ├── __init__.py
│   │   ├── calculate_final_smdi.py         # Combines weekly stats → SMDI & extremes
│   │   ├── calculate_week_maximum.py       # Weekly max by location
│   │   ├── calculate_week_mean.py          # Weekly mean by location *year*
│   │   ├── calculate_week_median.py        # Weekly median by location
│   │   ├── calculate_week_minimum.py       # Weekly min by location
│   │   └── mask_esa_data_to_nations.py     # Filter ESA TS by country polygons
│   └── xgboost_scripts/
│       ├── __init__.py
│       └── train_test.py                   # XGBoost train/test
├── shapefiles/                             # Natural Earth admin 0 boundaries
├── README.md
└── requirements.txt
```

---

## Quick Start

### 1) Environment
- **Python:** 3.10 recommended  
- **Install dependencies**
  ```bash
  pip install -r requirements.txt
  ```

### 2) Configure paths
Edit `james_nott_csc8099/config.py` and set:
```python
from pathlib import Path

ESA_TS_DIR    = Path("path/to/esa_timeseries_nc/")   # directory of *.nc tiles
SHAPEFILES_DIR= Path("shapefiles/")                  # country boundaries
OUT_DIR       = Path("outputs/")                     # where scripts will write
DATASET_PATH  = Path("data/dataset_complete.csv")    # modelling dataset
```

---

## Data Pipeline

### A) Mask ESA data to national boundaries
Create per-country NetCDFs from a directory of ESA time-series tiles.
```bash
python smdi_creation/mask_esa_data_to_nations.py
```
- Reads `ESA_TS_DIR` and `SHAPEFILES_DIR` from `config.py`
- Writes per-country `.nc` files to `OUT_DIR`

### B) Weekly statistics (min/median/mean/max)
Run on each per-country `.nc` file:

```bash
python smdi_creation/calculate_week_minimum.py  OUT_DIR/Country.nc  OUT_DIR/Country_min.csv
python smdi_creation/calculate_week_median.py   OUT_DIR/Country.nc  OUT_DIR/Country_median.csv
python smdi_creation/calculate_week_maximum.py  OUT_DIR/Country.nc  OUT_DIR/Country_max.csv
python smdi_creation/calculate_week_mean.py     OUT_DIR/Country.nc  OUT_DIR/Country_mean.csv
```

**Notes**
- ISO weeks are used; any week 53 is folded into week 52.
- Median/Min/Max output: rows = locations, columns = `week_01 … week_52`.
- Mean output is per **(location, year)** → `week_01 … week_52`.

### C) Compute SMDI and extremes summary
Combine the weekly tables to produce long-format SMDI and an extremes count (|SMDI| ≥ 3) per `(year, week)`.

```bash
python smdi_creation/calculate_final_smdi.py   OUT_DIR/Country_median.csv   OUT_DIR/Country_mean.csv     OUT_DIR/Country_max.csv      OUT_DIR/Country_min.csv      OUT_DIR/Country_smdi.csv     OUT_DIR/Country_extremes.csv
```

### D) Build the modelling dataset
Join `*_smdi.csv` (and any `*_extremes.csv`) with other features to produce
`data/dataset_complete.csv`. (`DATASET_PATH` points to the result.)

---

## Modelling

### LSTM (time-series)
**Simple train/test split**
```bash
python lstm_scripts/train_test_split.py
```
**Walk-forward validation**
```bash
python lstm_scripts/walk_forward.py
```
Both scripts load the dataset, add seasonality/lags, scale features, train an LSTM,
and report metrics (per fold for walk-forward; overall for split).  
Tip: for deterministic TensorFlow runs, set `TF_DETERMINISTIC_OPS=1`.

### XGBoost baseline
```bash
python xgboost_scripts/train_test.py
```
Loads the dataset, performs time-based split and feature selection, fits XGBoost,
and prints/saves evaluation metrics.

---

## Visualisation

Create regime-based error plots (LSTM vs XGBoost):
```bash
python data_visualisation/create_error_vis.py   --csv /path/to/rolling_std_and_abs_errors.csv   --outdir figures/
```
Outputs PNGs like `error_by_regime_xgboost.png` and `error_by_regime_lstm.png`.

---

## Requirements
See [`requirements.txt`](requirements.txt). Typical stack:  
`pandas`, `numpy`, `xarray`, `h5netcdf`, `geopandas`, `regionmask`, `shapely`, `scikit-learn`, `xgboost`, `tensorflow`, `keras`, `matplotlib`.

---


## Contact
**Author:** James Nott  
**Date:** 2025-08-11
