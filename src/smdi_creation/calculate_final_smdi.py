"""
Compute Final SMDI and Extremes Summary
File: calculate_final_smdi.py
Author: James Nott
Date: 2025-08-11

Description:
Combines weekly median/mean/min/max CSV tables to compute the Soil Moisture
Deficit Index (SMDI) per location, year, and ISO week. Produces:
  (1) a long-format SMDI dataset, and
  (2) an extremes summary (counts per year–week where |SMDI| ≥ 3).

Inputs:
- median_path: CSV of weekly medians by location (columns week_01 … week_52)
- mean_path:   CSV of weekly means   by location (columns week_01 … week_52)
- max_path:    CSV of weekly maxima  by location (columns week_01 … week_52)
- min_path:    CSV of weekly minima  by location (columns week_01 … week_52)

Outputs:
- out_path1: CSV with columns [location, year, wk, SD, SMDI]
- out_path2: CSV with extremes counts per (year, wk)

Usage:
python calculate_final_smdi.py <median.csv> <mean.csv> <max.csv> <min.csv> <smdi_out.csv> <extremes_out.csv>
"""

import pandas as pd
import numpy as np
from pathlib import Path

def compute_smdi (median_path = Path,
               mean_path = Path,
               max_path = Path,
               min_path = Path,
               out_path1 = Path,
               out_path2 = Path):

    # Read input files
    median_long = pd.read_csv(median_path).melt(id_vars=['locations'], var_name='wk', value_name= 'MSW') 
    min_long = pd.read_csv(min_path).melt(id_vars=['locations'], var_name='wk', value_name= 'minSW')
    max_long = pd.read_csv(max_path).melt(id_vars=['locations'], var_name='wk', value_name= 'maxSW')

    # Merge required data
    climate = (
    median_long
    .merge(min_long, on=['locations', 'wk'])
    .merge(max_long, on=['locations', 'wk'])   
    )                    

    climate = climate.astype({
    'MSW':  'float32',
    'minSW':'float32',
    'maxSW':'float32'
    })

    climate = (climate
        .set_index(['locations'])
        .rename_axis(index={'locations': 'location'})
        .assign(wk=lambda d: d['wk'].str.extract(r'(\d+)').astype(int)))

    climate = (
        climate                         
        .reset_index()                
        .sort_values(['location', 'wk'])   # 52 weeks per location
        .set_index('location')        # put location back as the index
    )

    years = pd.DataFrame({'year': range(1999, 2024)})        # 1999-2023 inclusive

    climate = (
        climate
        .reset_index()                  
        .assign(key=1)                  # helper column for merge
        .merge(years.assign(key=1), on='key', how='outer')  # replicate each row for every year
        .drop(columns='key')            # cleans
        .sort_values(['location', 'year', 'wk'])            # desired order
        .set_index(['location', 'year', 'wk'])              
    )

    idx = ['location', 'year', 'wk']
    def _clean_keys(df):
            df = df.copy()
            for c in idx:
                df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')
            return df
    climate   = _clean_keys(climate.reset_index())

    # To handle large dataset
    CHUNK = 50000
    out_file = out_path1
    header_written = False

    week_cols  = [f"{w:02d}" for w in range(1, 53)]
    dtype_map  = {c: 'float32' for c in week_cols}
    dtype_map.update({'year': 'int16', 'location': 'category'})

    # Extremes counter list
    all_extreme_counts = []

    for chunk in pd.read_csv(mean_path, chunksize=CHUNK, dtype=dtype_map, low_memory=False):
        mean_long = (
        chunk
        .melt(id_vars=['location', 'year'], var_name='wk', value_name='SW')
        .assign(wk=lambda d: d['wk'].str[2:].astype('int8'))
        .astype({'SW': 'float32', 'wk': 'int16'})
        .sort_values(['location', 'year', 'wk'])          
        )

        #Computes basic SMDI, before recursive feature added 

        mean_long = _clean_keys(mean_long.reset_index())

        df = (
            climate.merge(mean_long[['location', 'year', 'wk', 'SW']], 
            on=['location', 'year', 'wk'], 
            how='left')
            .reindex(columns=['location', 'year', 'wk', 'SW', 'MSW', 'minSW', 'maxSW'])
            .sort_values(['location', 'year', 'wk'])
)       
        # As per the SMDI formulas 
        cond = df['SW'] <= df['MSW']
        num   = df['SW'] - df['MSW']
        denom = np.where(cond, df['MSW'] - df['minSW'], df['maxSW'] - df['MSW'])
        df['SD'] = np.where(denom != 0, (num / denom) * 100, np.nan).astype('float32')

        # Uneeded deleted for memory
        del mean_long, cond, num, denom

        #Recurvsive calculation 
        df.sort_index(inplace=True)

        def smdi_series(sub: pd.DataFrame) -> pd.DataFrame:
            #If SD is NaN, SMDI is NaN.
            #As soon hit a NaN, the running memory is broken.
            #The next non-NaN point restarts with SD/50.
    
            sd   = sub['SD'].values.astype('float32')
            smdi = np.full_like(sd, np.nan, dtype='float32')          # preset everything to NaN
            
            for k in range(len(sd)):
                if np.isnan(sd[k]):                  # missing week ➜ leave NaN
                    continue
                if k == 0 or np.isnan(smdi[k-1]):    # first valid point OR after a gap
                    smdi[k] = sd[k] / 50
                else:                                # normal 
                    smdi[k] = 0.5 * smdi[k-1] + sd[k] / 50

            sub = sub.copy()
            sub['SMDI'] = smdi
            return sub

        
        df = smdi_series(df)

        # Count extremes for each year-week combination
        extremes = df[df['SMDI'].notna()].copy()
        extremes['is_extreme'] = (extremes['SMDI'] <= -3) | (extremes['SMDI'] >= 3) # Definition chosen for extremes

        chunk_extreme_counts = (extremes
                       .groupby(['year', 'wk'])
                       .agg({
                           'is_extreme': 'sum',      # count of extremes
                           'SMDI': 'count'           # count of non-NaN values
                       })
                       .reset_index()
                       .rename(columns={
                           'is_extreme': 'extreme_count',
                           'SMDI': 'total_count'
                       }))

        all_extreme_counts.append(chunk_extreme_counts)

        # Writes out
        df[['location', 'year', 'wk', 'SD', 'SMDI']].to_csv(out_file, mode='a', index=False, header=not header_written, float_format='%.4f')
        header_written = True
        
    print(f"SMDI data written to: {out_file}")    

    # Combine all extreme counts and aggregate
    if all_extreme_counts:
        final_extreme_counts = (pd.concat(all_extreme_counts, ignore_index=True)
                           .groupby(['year', 'wk'])
                           .agg({
                               'extreme_count': 'sum',
                               'total_count': 'sum'
                           })
                           .reset_index()
                           .assign(
                               extreme_decimal = lambda x: x['extreme_count'] / x['total_count']
                           )
                           .sort_values(['year', 'wk']))
        # Writes out to file
        extreme_file = out_path2
        final_extreme_counts.to_csv(extreme_file, index=False, float_format='%.6f')
        print(f"Extreme counts written to: {extreme_file}")

'''Main'''

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Compute SMDI for each nation, every year and week")
    p.add_argument("median_path",  type=Path, help="Median path for nation")
    p.add_argument("mean_path",  type=Path, help="Mean path for nation")
    p.add_argument("min_path",  type=Path, help="Min path for nation")
    p.add_argument("max_path",  type=Path, help="Max path for nation")
    p.add_argument("out_path1",  type=Path, help="Out path for SMDI data")
    p.add_argument("out_path2",  type=Path, help="Out path for SMDI analysis")
    
    args = p.parse_args()

    compute_smdi(args.median_path, args.mean_path, args.max_path, args.min_path, args.out_path1, args.out_path2)