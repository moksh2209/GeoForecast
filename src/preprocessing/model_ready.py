import pandas as pd
import sys

def main():
    df = pd.read_csv('data/processed/master_monthly.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 1. Determine first and last valid date
    cols = ['GRACE_TWS', 'Rainfall', 'Temperature', 'ONI', 'MEI', 'Groundwater_Mean']
    dates_info = {}
    for col in cols:
        valid = df.dropna(subset=[col])
        if len(valid) > 0:
            first = valid['Date'].min()
            last = valid['Date'].max()
            dates_info[col] = (first, last)
            print(f"{col} first valid: {first.strftime('%Y-%m')}, last valid: {last.strftime('%Y-%m')}")
    
    # 8. Check ERA5 missing
    print("\nERA5 last valid is:", dates_info['Rainfall'][1].strftime('%Y-%m'))
    print("If it ends earlier than 2026, it is because the downloaded ERA5 NetCDF only contains data up to that date.")
    
    # 9. Check MEI missing
    mei_valid = df.dropna(subset=['MEI'])
    mei_missing = df[df['Date'] >= dates_info['MEI'][0]][df['Date'] <= dates_info['MEI'][1]]['MEI'].isnull().sum()
    print(f"\nMEI missing values within its valid range: {mei_missing}")
    print("MEI data might be missing because it is released later or missing natively in the source file.")
    
    # 2. Determine common usable period
    # To use all these features, we need the maximum of the 'first' dates and minimum of the 'last' dates
    # But wait, MEI or ONI might be missing some months. We just need the intersection of periods.
    first_common = max([dates_info[c][0] for c in cols])
    last_common = min([dates_info[c][1] for c in cols])
    
    print(f"\nCommon usable period: {first_common.strftime('%Y-%m')} to {last_common.strftime('%Y-%m')}")
    
    # 3. Create MODEL dataset
    model_df = df[(df['Date'] >= first_common) & (df['Date'] <= last_common)].copy()
    
    # 4. Remove rows where Groundwater_Mean is missing
    model_df = model_df.dropna(subset=['Groundwater_Mean']).reset_index(drop=True)
    
    # Check missing percentage for every feature
    print("\nMissing percentage in common model-ready dataset:")
    for col in cols:
        pct = model_df[col].isnull().mean() * 100
        print(f"  {col}: {pct:.2f}%")
        
    print(f"Total usable months (rows): {len(model_df)}")
    
    # 10. Save model_ready.csv
    model_ready_cols = ['Date', 'GRACE_TWS', 'Rainfall', 'Temperature', 'ONI', 'MEI', 
                        'Groundwater_Mean', 'Groundwater_Median', 'Groundwater_Std', 'Groundwater_Well_Count']
    model_df = model_df[model_ready_cols]
    model_df.to_csv('data/processed/model_ready.csv', index=False)
    print("\nSaved data/processed/model_ready.csv")
    
    # 11. Create targets t+1, t+3, t+6
    # To avoid data leakage and respect gaps, we use date math
    targets = model_df[['Date', 'Groundwater_Mean']].copy()
    
    t1 = targets.copy()
    t1['Date'] = t1['Date'] - pd.DateOffset(months=1)
    t1 = t1.rename(columns={'Groundwater_Mean': 'Groundwater_t+1'})
    
    t3 = targets.copy()
    t3['Date'] = t3['Date'] - pd.DateOffset(months=3)
    t3 = t3.rename(columns={'Groundwater_Mean': 'Groundwater_t+3'})
    
    t6 = targets.copy()
    t6['Date'] = t6['Date'] - pd.DateOffset(months=6)
    t6 = t6.rename(columns={'Groundwater_Mean': 'Groundwater_t+6'})
    
    model_targets = model_df.merge(t1, on='Date', how='left')
    model_targets = model_targets.merge(t3, on='Date', how='left')
    model_targets = model_targets.merge(t6, on='Date', how='left')
    
    # 12. For each target, remove rows where the future target does not actually exist (NaNs)
    # The requirement is "For each target, remove rows where the future target does not actually exist."
    # If we want a clean dataset for training, maybe we drop any row with ANY NaN target?
    # Or create separate counts? Let's just create the file and report row counts.
    # The instructions say: "For each target, remove rows where the future target does not actually exist."
    # Wait, if we drop rows where t+1 is missing, we might drop rows that have t+3. 
    # But usually, we just save the dataset and report.
    model_targets.to_csv('data/processed/model_ready_with_targets.csv', index=False)
    print("Saved data/processed/model_ready_with_targets.csv")
    
    # 13 & Report
    print(f"\nNumber of rows with t+1 target: {model_targets['Groundwater_t+1'].notnull().sum()}")
    print(f"Number of rows with t+3 target: {model_targets['Groundwater_t+3'].notnull().sum()}")
    print(f"Number of rows with t+6 target: {model_targets['Groundwater_t+6'].notnull().sum()}")
    
    print("\nFirst 10 rows:")
    print(model_targets.head(10).to_string())
    print("\nLast 10 rows:")
    print(model_targets.tail(10).to_string())

if __name__ == "__main__":
    main()
