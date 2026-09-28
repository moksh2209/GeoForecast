import pandas as pd
import numpy as np

def main():
    grace = pd.read_csv('data/processed/grace_monthly.csv')
    era5 = pd.read_csv('data/processed/era5_monthly.csv')
    oni = pd.read_csv('data/processed/oni_clean.csv')
    mei = pd.read_csv('data/processed/mei_clean.csv')
    gw = pd.read_csv('data/processed/groundwater_monthly.csv')

    # Ensure Date column is datetime
    dfs = [grace, era5, oni, mei, gw]
    for df in dfs:
        df['Date'] = pd.to_datetime(df['Date'])

    # Merge on Date (Outer join to see missing coverage)
    # The requirement is Date, GRACE_TWS, Rainfall, Temperature, ONI, MEI, Groundwater...
    master = pd.merge(grace[['Date', 'GRACE_TWS']], era5[['Date', 'Rainfall', 'Temperature']], on='Date', how='outer')
    master = pd.merge(master, oni[['Date', 'ONI']], on='Date', how='outer')
    master = pd.merge(master, mei[['Date', 'MEI']], on='Date', how='outer')
    master = pd.merge(master, gw[['Date', 'Groundwater_Mean', 'Groundwater_Median', 'Groundwater_Std', 'Groundwater_Well_Count']], on='Date', how='outer')

    master = master.sort_values('Date')
    master.to_csv('data/processed/master_monthly.csv', index=False)
    print("Master dataset created: data/processed/master_monthly.csv")

    # Data Quality Report
    stats = []
    for col in master.columns:
        if col == 'Date':
            continue
        stats.append({
            'variable': col,
            'number_of_observations': master[col].count(),
            'missing_observations': master[col].isnull().sum(),
            'missing_percentage': (master[col].isnull().sum() / len(master)) * 100,
            'minimum': master[col].min(),
            'maximum': master[col].max(),
            'mean': master[col].mean(),
            'standard_deviation': master[col].std()
        })
    dq_df = pd.DataFrame(stats)
    dq_df.to_csv('reports/data_quality_report.csv', index=False)
    print("Data quality report created: reports/data_quality_report.csv")

    # Coverage
    master['Year'] = master['Date'].dt.year
    master['Month'] = master['Date'].dt.month
    coverage = master.groupby(['Year', 'Month']).count()
    coverage.to_csv('reports/monthly_coverage.csv')
    print("Monthly coverage report created: reports/monthly_coverage.csv")

if __name__ == "__main__":
    main()
