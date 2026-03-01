"""
create_error_vis.py

Author: James Nott
Date: 2025-08-11

Make regime-based error charts for XGBoost and LSTM.

Input:
  /mnt/data/vol_error_vis_pt2.csv  (or pass a different path via --csv)

Output:
  error_by_regime_xgboost.png
  error_by_regime_lstm.png
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def compute_regimes(df, commodity):
    """
    Given datafame with columns:
      rolling_stdev_{commodity}
    returns boolean masks for low/high volatility for that commodity based on
    the 20th and 80th percentiles of its rolling stdev.
    """
    rs_col = f"rolling_stdev_{commodity}"
    if rs_col not in df.columns:
        raise ValueError(f"Missing column: {rs_col}") # Errors if cannot find column

    # Quantile thresholds per commodity
    low_thr = df[rs_col].quantile(0.20)
    high_thr = df[rs_col].quantile(0.80)

    low_mask = df[rs_col] <= low_thr
    high_mask = df[rs_col] >= high_thr
    return low_mask, high_mask, low_thr, high_thr

def summarize_errors(df, commodity, model):
    """
    Compute mean absolute error for Low and High volatility regimes for a
    given commodity and model.
    Columns expected for errors:
      abs_error_{model}_{commodity} 
    """
    err_col = f"abs_error_{model}_{commodity}" # Locates absolute error column
    if err_col not in df.columns:
        raise ValueError(f"Missing column: {err_col}")

    low_mask, high_mask, low_thr, high_thr = compute_regimes(df, commodity)
    low_vals = df.loc[low_mask, err_col].dropna()
    high_vals = df.loc[high_mask, err_col].dropna() # Gathers values in percentiles

    return {
        "low_mean": float(low_vals.mean()) if len(low_vals) else np.nan,
        "high_mean": float(high_vals.mean()) if len(high_vals) else np.nan,
        "low_n": int(low_vals.shape[0]),
        "high_n": int(high_vals.shape[0]),
        "low_thr": float(low_thr),
        "high_thr": float(high_thr),
    }

def plot_model_figure(stats_by_commodity, model, outpath):
    """
    Create a bar chart for one model with both commodities, comparing
    Low vs High volatility mean absolute errors.
    """
    commodities = list(stats_by_commodity.keys())  # corn and soy for example 
    # Prepare data
    low_means = [stats_by_commodity[c]["low_mean"] for c in commodities]
    high_means = [stats_by_commodity[c]["high_mean"] for c in commodities]
    low_ns = [stats_by_commodity[c]["low_n"] for c in commodities]
    high_ns = [stats_by_commodity[c]["high_n"] for c in commodities]

    x = np.arange(len(commodities))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, low_means, width, label="Low vol (≤ 20th pct.)")
    bars2 = ax.bar(x + width/2, high_means, width, label="High vol (≥ 80th pct.)")

    ax.set_title(f"Absolute Error by Volatility Regime — {model.upper()}")
    ax.set_xticks(x, [c.capitalize() for c in commodities])
    ax.set_ylabel("Mean absolute error")
    ax.legend()

    # Add counts on/above bars
    def annotate(bars, counts):
        for bar, n in zip(bars, counts):
            height = bar.get_height()
            ax.annotate(f"n={n}",
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    annotate(bars1, low_ns)
    annotate(bars2, high_ns)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="/mnt/data/vol_error_vis_pt2.csv",
                        help="Path to CSV containing rolling stdev and abs errors.")
    parser.add_argument("--outdir", default=".",
                        help="Directory to save figures.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="ignore", dayfirst=True)

    commodities = ["corn", "soy"]
    models = ["xgboost", "lstm"]

    for model in models:
        stats_by_commodity = {}
        for com in commodities:
            stats = summarize_errors(df, com, model)
            stats_by_commodity[com] = stats
        outpath = outdir / f"error_by_regime_{model}.png"
        plot_model_figure(stats_by_commodity, model, outpath)
        print(f"Saved: {outpath}")

if __name__ == "__main__":
    main()
