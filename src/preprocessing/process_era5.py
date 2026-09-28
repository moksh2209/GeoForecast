import xarray as xr
import pandas as pd
import numpy as np
import os

input_file = 'data/raw/ab52b4c7de0e419f7b9b87515ec22f03.nc'
output_file = 'data/processed/era5_monthly.csv'

# Study area approx bounding box for Maharashtra
lat_min, lat_max = 13, 22
lon_min, lon_max = 72, 81

def main():
    ds = xr.open_dataset(input_file)
    # Subset study area (note ERA5 might have decreasing lat, sel(lat=slice(lat_max, lat_min)) if needed)
    # Let's handle generic lat slice
    if ds['latitude'][0] > ds['latitude'][-1]:
        ds_sub = ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
    else:
        ds_sub = ds.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
    
    # variables: t2m (Kelvin), tp (meters)
    # convert temperature to Celsius
    t2m_celsius = ds_sub['t2m'] - 273.15
    # convert precipitation to mm (1m = 1000mm)
    tp_mm = ds_sub['tp'] * 1000
    
    # calculate spatial mean for each month (valid_time)
    mean_temp = t2m_celsius.mean(dim=['latitude', 'longitude'], skipna=True)
    mean_precip = tp_mm.mean(dim=['latitude', 'longitude'], skipna=True)
    
    df = pd.DataFrame({
        'Date': mean_temp['valid_time'].values,
        'Temperature': mean_temp.values,
        'Rainfall': mean_precip.values
    })
    
    # Format Date to Month start (YYYY-MM-01)
    df['Date'] = pd.to_datetime(df['Date']).dt.to_period('M').dt.to_timestamp()
    
    df = df.groupby('Date', as_index=False).mean()
    
    df.to_csv(output_file, index=False)
    print(f"Processed ERA5 data. Saved to {output_file}")
    
if __name__ == "__main__":
    main()
