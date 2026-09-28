import xarray as xr
import pandas as pd
import numpy as np
import os

input_file = 'data/raw/GRCTellus.JPL.200204_202607.GLO.RL06.3M.MSCNv04CRI.nc'
output_file = 'data/processed/grace_monthly.csv'

# Study area approx bounding box for Maharashtra
lat_min, lat_max = 13, 22
lon_min, lon_max = 72, 81

def main():
    ds = xr.open_dataset(input_file)
    # The variable is lwe_thickness
    # Subset study area
    ds_sub = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    
    # Calculate spatial mean for each month (time)
    mean_ts = ds_sub['lwe_thickness'].mean(dim=['lat', 'lon'], skipna=True)
    std_ts = ds_sub['lwe_thickness'].std(dim=['lat', 'lon'], skipna=True)
    count_ts = ds_sub['lwe_thickness'].count(dim=['lat', 'lon'])
    
    df = pd.DataFrame({
        'Date': mean_ts['time'].values,
        'GRACE_TWS': mean_ts.values,
        'GRACE_TWS_STD': std_ts.values,
        'GRACE_PIXEL_COUNT': count_ts.values
    })
    
    # Format Date to Month start (YYYY-MM-01) for consistency
    df['Date'] = pd.to_datetime(df['Date']).dt.to_period('M').dt.to_timestamp()
    
    # Drop rows where GRACE_TWS is NaN (missing months)
    df = df.dropna(subset=['GRACE_TWS'])
    
    # Keep only unique dates if multiple observations per month exist
    df = df.groupby('Date', as_index=False).mean()
    
    df.to_csv(output_file, index=False)
    print(f"Processed GRACE data. Saved to {output_file}")
    
if __name__ == "__main__":
    main()
