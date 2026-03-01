# james_nott_csc8099/config.py
from pathlib import Path

# Core locations (all resolved relative to this file)
PKG_DIR = Path(__file__).resolve().parent           # .../james_nott_csc8099
PROJECT_ROOT = PKG_DIR.parent                       # repo root

# Data folders 
DATA_DIR = PROJECT_ROOT / "data"
ESA_TS_DIR = PROJECT_ROOT / "esa_time_series_data.nosync"
SHAPEFILES_DIR = PROJECT_ROOT / "shapefiles"

# Existing dataset path
DATASET_PATH = DATA_DIR / "dataset_complete.csv"

# Where scripts can write results
OUT_DIR = DATA_DIR / "esa_ts_filtered_countries"
OUT_DIR.mkdir(parents=True, exist_ok=True)