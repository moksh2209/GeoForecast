import pandas as pd
import numpy as np

input_file = 'data/raw/oni.csv'
output_file = 'data/processed/oni_clean.csv'

def main():
    df = pd.read_csv(input_file)
    # The column name is '  ONI from CPC  missing value -99.9 https://psl.noaa.gov/data/timeseries/month/'
    # Rename it to 'ONI'
    col_name = [c for c in df.columns if 'ONI' in c][0]
    df.rename(columns={col_name: 'ONI'}, inplace=True)
    
    # Replace values < -10 with NaN
    df['ONI'] = df['ONI'].apply(lambda x: np.nan if x < -10 else x)
    
    # Ensure Date is YYYY-MM-01
    df['Date'] = pd.to_datetime(df['Date']).dt.to_period('M').dt.to_timestamp()
    
    df = df[['Date', 'ONI']]
    df.to_csv(output_file, index=False)
    print(f"Processed ONI data. Saved to {output_file}")
    
if __name__ == "__main__":
    main()
