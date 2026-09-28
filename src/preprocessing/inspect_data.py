import os
import pandas as pd
import xarray as xr

data_dir = 'data/raw/'

files = [
    'GRCTellus.JPL.200204_202607.GLO.RL06.3M.MSCNv04CRI.nc',
    'ab52b4c7de0e419f7b9b87515ec22f03.nc',
    'oni.csv',
    'meiv2.csv',
    'ground_water_level_manual_quarterly_maharashtra_1970_2025.csv'
]

for f in files:
    path = os.path.join(data_dir, f)
    print("==================================================")
    print(f"Inspecting: {f}")
    size = os.path.getsize(path)
    print(f"File size: {size / 1024 / 1024:.2f} MB")
    
    if f.endswith('.nc'):
        try:
            ds = xr.open_dataset(path)
            print("\nDimensions:", ds.dims)
            print("Coordinates:", list(ds.coords))
            print("Variables:")
            for var in ds.data_vars:
                print(f"  {var}:")
                print(f"    Dimensions: {ds[var].dims}")
                print(f"    Units: {ds[var].attrs.get('units', 'N/A')}")
                print(f"    Long name: {ds[var].attrs.get('long_name', 'N/A')}")
            
            if 'time' in ds.coords:
                times = ds['time'].values
                print(f"Time range: {times.min()} to {times.max()}")
            
            if 'lat' in ds.coords and 'lon' in ds.coords:
                print(f"Lat range: {ds['lat'].min().item()} to {ds['lat'].max().item()}")
                print(f"Lon range: {ds['lon'].min().item()} to {ds['lon'].max().item()}")
            elif 'latitude' in ds.coords and 'longitude' in ds.coords:
                print(f"Lat range: {ds['latitude'].min().item()} to {ds['latitude'].max().item()}")
                print(f"Lon range: {ds['longitude'].min().item()} to {ds['longitude'].max().item()}")

            ds.close()
        except Exception as e:
            print(f"Error reading NC file: {e}")
            
    elif f.endswith('.csv'):
        try:
            df = pd.read_csv(path, low_memory=False, encoding='iso-8859-1')
            print(f"Shape: {df.shape}")
            print("Columns and Data Types:")
            print(df.dtypes)
            print("Missing Values:")
            print(df.isnull().sum())
            print(f"Duplicate rows: {df.duplicated().sum()}")
            print("\nFirst 3 rows:")
            print(df.head(3))
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            
print("==================================================")
