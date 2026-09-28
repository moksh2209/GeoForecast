import pandas as pd
import numpy as np

input_file = 'data/raw/ground_water_level_manual_quarterly_maharashtra_1970_2025.csv'
output_file = 'data/processed/groundwater_monthly.csv'

def main():
    df = pd.read_csv(input_file, low_memory=False, encoding='iso-8859-1')
    
    # Parse Date of Monitoring (it's DD-MM-YYYY)
    df['Date of Monitoring'] = pd.to_datetime(df['Date of Monitoring'], format='%d-%m-%Y', errors='coerce')
    
    # Drop rows without a valid date
    df = df.dropna(subset=['Date of Monitoring'])
    
    # Get month start
    df['Date'] = df['Date of Monitoring'].dt.to_period('M').dt.to_timestamp()
    
    # Convert 'Water Level, m bgl' to numeric just in case
    df['Water Level, m bgl'] = pd.to_numeric(df['Water Level, m bgl'], errors='coerce')
    
    # Group by Date and calculate mean, median, std, and count
    agg_df = df.groupby('Date').agg(
        Groundwater_Mean=('Water Level, m bgl', 'mean'),
        Groundwater_Median=('Water Level, m bgl', 'median'),
        Groundwater_Std=('Water Level, m bgl', 'std'),
        Groundwater_Well_Count=('Well ID', 'nunique')
    ).reset_index()
    
    agg_df.to_csv(output_file, index=False)
    print(f"Processed Groundwater data. Saved to {output_file}")
    
if __name__ == "__main__":
    main()
