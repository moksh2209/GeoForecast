import pandas as pd

def main():
    df = pd.read_csv('data/processed/master_monthly.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Create forecasting targets
    # Shift operations move data forward/backward. 
    # df['Groundwater_Mean'].shift(-1) gets the value from the next row (month t+1) 
    # IF the dataframe is purely monthly with no missing months in between.
    # To be safe against missing months, we should use a date-based merge.
    
    # Let's create shifted dataframes
    df_t1 = df[['Date', 'Groundwater_Mean']].copy()
    df_t1['Date'] = df_t1['Date'] - pd.DateOffset(months=1)
    df_t1 = df_t1.rename(columns={'Groundwater_Mean': 'Groundwater_t+1'})
    
    df_t3 = df[['Date', 'Groundwater_Mean']].copy()
    df_t3['Date'] = df_t3['Date'] - pd.DateOffset(months=3)
    df_t3 = df_t3.rename(columns={'Groundwater_Mean': 'Groundwater_t+3'})
    
    df_t6 = df[['Date', 'Groundwater_Mean']].copy()
    df_t6['Date'] = df_t6['Date'] - pd.DateOffset(months=6)
    df_t6 = df_t6.rename(columns={'Groundwater_Mean': 'Groundwater_t+6'})
    
    master_targets = df.merge(df_t1, on='Date', how='left')
    master_targets = master_targets.merge(df_t3, on='Date', how='left')
    master_targets = master_targets.merge(df_t6, on='Date', how='left')
    
    master_targets.to_csv('data/processed/master_with_targets.csv', index=False)
    print("Created data/processed/master_with_targets.csv")
    
if __name__ == "__main__":
    main()
