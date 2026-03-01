"""
Weekly Mean Soil Moisture (by Year) → CSV
File: calculate_week_mean.py
Author: James Nott
Date: 2025-08-11

Description:
Computes ISO-weekly mean soil moisture for every location and year in a large
daily NetCDF cube. Applies the same missing-data handling as the median/min/max
scripts and is designed to run memory-safely.

Inputs:
- A daily soil moisture NetCDF file with a 'time' dimension and 'locations'.

Outputs:
- CSV file where the index/rows represent (location, year) and columns are
  week_01 … week_52 containing weekly mean soil moisture values.

Usage:
python calculate_week_mean.py <input.nc> <output.csv>

"""

import xarray as xr
import pandas as pd
from pathlib import Path
import numpy as np



def write_weekly_means_all_years(nc_path: Path,
                         out_csv: Path,
                         loc_chunk: int = 50,
                         time_chunk: int = 365,
                         var_name: str = "sm") -> None:
    """Compute weekly means and save them to *out_csv*."""

    # 1. Chunked open and ID creation

    ds = xr.open_dataset(
        nc_path,
        chunks={"locations": loc_chunk, "time": time_chunk},
        decode_times=True
    )

    #Creates ID's for all locations 
    loc_id_array = np.arange(ds.dims["locations"])
    ds = ds.assign_coords(
        loc_id=("locations", loc_id_array)
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

    # 4. Weekly mean 
    # Returns NaN if there wasn't 5 data points to develop a maximum/minimum/median in 

    weekly_mean = (
        ds[var_name]                              # daily data
        .groupby(["iso_year", "iso_week"])
        .mean("time")                             # daily → weekly
    )

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

    ### 5. Section from Median calculater to work out what locations to drop
    valid_n = weekly_mean.count("iso_year")          
    median_52wk = (
        weekly_mean.median("iso_year")               
                .where(valid_n >= 5)             
    )
    valid_per_loc = median_52wk.count("iso_week")          
    needed = median_52wk.sizes["iso_week"] - 3            
    loc_mask = valid_per_loc >= needed
    loc_mask = loc_mask.compute()
    median_52wk = median_52wk.where(loc_mask, drop=True)
    # Works out what locations have been dropped
    kept   = median_52wk["loc_id"].values 
    original = ds["loc_id"].values 
    removed_ids = (np.setdiff1d(original, kept)).tolist()

    # num dropped
    dropped = int(ds.sizes["locations"] - median_52wk.sizes["locations"])
    print(f"Dropped {dropped} locations that had >3 NaNs in the 52-week record.")

    # 6. Strip locations excluded from Median data
    keep_mask = ~np.isin(weekly_mean.loc_id.values, removed_ids)
    weekly_mean_stripped = weekly_mean.isel(locations=keep_mask)

    print("Locations after drops:", weekly_mean_stripped.sizes["locations"])


    # 7. Bring the result to memory and pivot to table
    table = (
        weekly_mean_stripped
      .to_series()            # Series with MultiIndex (location, iso_year, iso_week)
      .unstack(level="iso_week")       
    )

    table.columns = [f"wk{wk:02d}" for wk in table.columns]   # wk01 - wk52
    table = table.sort_index(level=["locations", "iso_year"]) # order by location first
    table.index = table.index.set_names(["location", "year"])

    # 8. Writes to CSV
    
    table.to_csv(out_csv)
    print(f"Saved climatological medians to: {out_csv}")
    

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Compute 52-week median soil-moisture for a NetCDF cube")
    p.add_argument("nc_path",  type=Path, help="Input .nc file")
    p.add_argument("out_csv",  type=Path, help="Output CSV file")
    
    args = p.parse_args()

    write_weekly_means_all_years(args.nc_path,
                         args.out_csv)
