"""
Mask ESA Time-Series to National Boundaries
File: mask_esa_data_to_nations.py
Author: James Nott
Date: 2025-08-11

Description:
Loads ESA time-series NetCDF tiles in manageable chunks, intersects their
'locations' with national polygons, and writes a per-country NetCDF containing
only the locations within that nation. Intended for memory-safe processing.

Inputs:
- ESA_TS_DIR: directory of ESA NetCDF files (*.nc) (from project config)
- SHAPEFILES_DIR: directory with country boundary shapefiles (from config)

Outputs:
- One NetCDF per country written to OUT_DIR (from project config).

Usage:
python mask_esa_data_to_nations.py
  (relies on project configuration in `james_nott_csc8099.config`)
"""

import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
from james_nott_csc8099.config import ESA_TS_DIR, SHAPEFILES_DIR, OUT_DIR, DATASET_PATH


files = sorted((ESA_TS_DIR).glob("*.nc"))
grouped_files = [files[i:i+20] for i in range(0, len(files), 20)]

datasets = []
for chunk in grouped_files:
    ds_chunk = xr.open_mfdataset(
        chunk,
        combine="nested",
        concat_dim="locations",
        parallel=True,
        engine = "h5netcdf"
    )
    datasets.append(ds_chunk)


# Combine across all chunks
ds = xr.concat(datasets, dim="locations")



# Reading shapefiles
countries_path = SHAPEFILES_DIR / "ne_10m_admin_0_countries.shp"
shp = gpd.read_file(countries_path)


# Picking countries for study
countries = shp[shp["NAME_LONG"].isin([
    "United States",
    "China",
    "Brazil",
    "Argentina"
])]

# Extract unique locations and their coordinates
unique_locations = ds.groupby('locations').first()  # Get one record per location
location_coords = list(zip(unique_locations['lon'].values, unique_locations['lat'].values))

print("Locations succesfully extracted")

# Create GeoDataFrame of points
points_gdf = gpd.GeoDataFrame(
    geometry=[Point(lon, lat) for lon, lat in location_coords],
    index=unique_locations.locations.values,
    crs='EPSG:4326'  # Set CRS to match the countries shapefile
)

print(f"Created {len(points_gdf)} location points")

# Ensure both GeoDataFrames have the same CRS
print(f"Points CRS: {points_gdf.crs}")
print(f"Countries CRS: {countries.crs}")

# Check country boundaries
print("\nCountry boundaries:")
for idx, row in countries.iterrows():
    bounds = row.geometry.bounds
    print(f"{row['NAME_LONG']}: lon({bounds[0]:.2f} to {bounds[2]:.2f}), lat({bounds[1]:.2f} to {bounds[3]:.2f})")

# Spatial join to find which country each location belongs to
points_with_countries = gpd.sjoin(points_gdf, countries, how='left', predicate='within')
print(f"Points matched to countries: {len(points_with_countries)}")

matches1 = points_with_countries.dropna(subset=['index_right'])
print(f"Matched points (within): {len(matches1)}")



# Filter and save data by country
for idx, country_name in enumerate(countries["NAME_LONG"]):
    # Find locations that belong to this country
    country_locations = matches1[matches1['NAME_LONG'] == country_name].index
    
    print(f"Found {len(country_locations)} locations in {country_name}")
    
    # Filter dataset to only include these locations
    ds_country = ds.sel(locations=ds.locations.isin(country_locations))
        
    print(f"Filtered dataset shape for {country_name}: {ds_country.dims}")
    
    # Save to file
    out_path = OUT_DIR / f"{country_name.replace(' ', '_')}.nc"
    ds_country.to_netcdf(out_path)
    print(f"Wrote: {out_path}")
     

    