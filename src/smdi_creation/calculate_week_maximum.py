"""
Weekly Maximum Soil Moisture → CSV
File: calculate_week_maximum.py
Author: James Nott
Date: 2025-08-11

Description:
Computes the per-ISO-week (1–52) maximum soil moisture for every location in a
large daily NetCDF cube. Handles missing values and filters locations with
insufficient data. Designed to be memory-safe.

Inputs:
- A daily soil moisture NetCDF file with a 'time' dimension and 'locations'.

Outputs:
- CSV file where rows are locations and columns are week_01 … week_52 containing
  weekly maximum soil moisture values.

Usage:
python calculate_week_maximum.py <input.nc> <output.csv>

"""

import xarray as xr
import pandas as pd
from pathlib import Path



def write_weekly_maximums(nc_path: Path,
                         out_csv: Path,
                         loc_chunk: int = 500,
                         time_chunk: int = 365,
                         var_name: str = "sm") -> None:
    """Compute 52-week climatological maximums and save them to *out_csv*."""

    # 1. Chunked open 

    ds = xr.open_dataset(
        nc_path,
        chunks={"locations": loc_chunk, "time": time_chunk},
        decode_times=True
    )

    # 2. ISO-calendar coordinates

    iso = ds.time.dt.isocalendar()         

    # 3. Merges any data from the 53rd week into the adjacent week

    month = ds['time'].dt.month

    week_adj = iso['week'].copy()
    week_adj = week_adj.where(~((iso['week'] == 53) & (month == 12)), 52)  # late-Dec → 52
    week_adj = week_adj.where(~((iso['week'] == 53) & (month == 1)),   1)  # early-Jan → 1

    year_adj = iso['year'].where(~((iso['week'] == 53) & (month == 1)),
                                iso['year'] + 1)

    ds = ds.assign_coords(
        iso_year=("time", year_adj.values),         
        iso_week=("time", week_adj.values)
    )

    # 4. Weekly mean and Interpolation, then maximum across years (all locations at once). 
    # Returns NaN if there wasn't 5 data points to develop a maximum

    weekly_mean = (
        ds[var_name]                              # daily data
        .groupby(["iso_year", "iso_week"])
        .mean("time")                             # daily → weekly
    )

    # Adapt chunking for interpolation
    weekly_mean = weekly_mean.chunk({"iso_week": -1})
    weekly_mean = (
        weekly_mean
        .sortby("iso_week")                 # make sure weeks are 1-52
        .interpolate_na(
            dim="iso_week",
            method="linear",                # straight-line between neighbours
            limit=2,                        # only up to 2 missing weeks
            use_coordinate=False,           # treat week numbers as equally spaced
        )
    )

    valid_n = weekly_mean.count("iso_year")          # number of yearly means per week
    maximum_52wk = (
        weekly_mean.max("iso_year")               
                .where(valid_n >= 5)              # Maximum but drops if < 5 values
    ) 

    # 5. Drops locations where there is less than 49/52 weeks

    # counts across the 52 weeks
    valid_per_loc = maximum_52wk.count("iso_week") 
     # keep only locations with at least 52 – 3 = 49 valid weeks

    needed = maximum_52wk.sizes["iso_week"] - 3             # 49
    loc_mask = valid_per_loc >= needed
    loc_mask = loc_mask.compute()
    maximum_52wk = maximum_52wk.where(loc_mask, drop=True)

    # num dropped
    dropped = int(ds.sizes["locations"] - maximum_52wk.sizes["locations"])
    print(f"Dropped {dropped} locations that had >3 NaNs in the 52-week record.")

    # 6. Bring the small result to memory and convert to Pandas table

    table = maximum_52wk.compute().to_pandas()     # shape: (n_locations, 52)
    table.columns = [f"wk{wk:02d}" for wk in table.columns]

    # 7. linearly interpolate the at-most-two missing weeks per location

    table = (
        table
        .interpolate(method="linear", axis=1, limit_direction="both")  # fill gaps
        .fillna(method="bfill", axis=1)                                # safety net
        .fillna(method="ffill", axis=1)
    )
    assert table.isna().sum().sum() == 0     # check

    # 8. Writes to CSV

    table.to_csv(out_csv)
    print(f"Saved climatological maximums to: {out_csv}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Compute 52-week maximum soil-moisture for a NetCDF cube")
    p.add_argument("nc_path",  type=Path, help="Input .nc file")
    p.add_argument("out_csv",  type=Path, help="Output CSV file")
    
    args = p.parse_args()

    write_weekly_maximums(args.nc_path,
                         args.out_csv)