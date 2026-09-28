import pandas as pd
import numpy as np

input_file = 'data/raw/meiv2.csv'
output_file = 'data/processed/mei_clean.csv'

def main():
    df = pd.read_csv(input_file)
    # The column name has MEI in it
    col_name = [c for c in df.columns if 'MEI' in c][0]
    df.rename(columns={col_name: 'MEI'}, inplace=True)
    
    # Replace values < -10 with NaN
    df['MEI'] = df['MEI'].apply(lambda x: np.nan if x < -10 else x)
    
    # Ensure Date is YYYY-MM-01
    df['Date'] = pd.to_datetime(df['Date']).dt.to_period('M').dt.to_timestamp()
    
    df = df[['Date', 'MEI']]
    df.to_csv(output_file, index=False)
    print(f"Processed MEI data. Saved to {output_file}")
    
if __name__ == "__main__":
    main()
