import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def calculate_metrics(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(mask):
        return np.nan, np.nan, np.nan, 0
    y_t, y_p = y_true[mask], y_pred[mask]
    return mean_absolute_error(y_t, y_p), np.sqrt(mean_squared_error(y_t, y_p)), r2_score(y_t, y_p), len(y_t)

def main():
    print("==================================================")
    print("RECOVERING EXPERIMENT 6 RANDOM FOREST T+6 MODEL")
    print("==================================================")
    
    # Load dataset exactly like train_experiment6.py
    df = pd.read_csv('data/processed/district_spatiotemporal_monthly.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Common modeling period filtering
    df_modeling = df.dropna(subset=['GRACE_TWS', 'Rainfall', 'Temperature', 'ONI', 'MEI']).copy()
    df_modeling = df_modeling.sort_values(['District Name', 'Date']).reset_index(drop=True)
    
    # Target construction for t+6
    t = 't+6'
    shift_val = 6
    df_modeling[f'Target_Date_{t}'] = df_modeling['Date'] + pd.DateOffset(months=shift_val)
    target_mapping = df_modeling[['District Name', 'Date', 'Groundwater_Mean']].rename(
        columns={'Date': f'Target_Date_{t}', 'Groundwater_Mean': f'Groundwater_{t}'}
    )
    df_modeling = pd.merge(df_modeling, target_mapping, on=['District Name', f'Target_Date_{t}'], how='left')
    
    # Chronological Split
    unique_dates = sorted(df_modeling['Date'].unique())
    n_dates = len(unique_dates)
    train_end_idx = int(n_dates * 0.7)
    val_end_idx = int(n_dates * 0.85)
    
    train_dates = unique_dates[:train_end_idx]
    val_dates = unique_dates[train_end_idx:val_end_idx]
    test_dates = unique_dates[val_end_idx:]
    
    train_df = df_modeling[df_modeling['Date'].isin(train_dates)].copy()
    val_df = df_modeling[df_modeling['Date'].isin(val_dates)].copy()
    test_df = df_modeling[df_modeling['Date'].isin(test_dates)].copy()
    
    # District encoding (One-hot)
    districts = df_modeling['District Name'].unique()
    for d in districts:
        train_df[f'Dist_{d}'] = (train_df['District Name'] == d).astype(int)
        val_df[f'Dist_{d}'] = (val_df['District Name'] == d).astype(int)
        test_df[f'Dist_{d}'] = (test_df['District Name'] == d).astype(int)
        
    dist_cols = [f'Dist_{d}' for d in districts]
    base_features = ['GRACE_TWS', 'Rainfall', 'Temperature', 'ONI', 'MEI', 'Groundwater_Mean', 'Groundwater_Well_Count']
    feats_no_coord = base_features + dist_cols
    
    # Data Prep
    t_col = f'Groundwater_{t}'
    y_train = train_df[t_col].values
    y_test = test_df[t_col].values
    
    mask_tr = ~np.isnan(y_train)
    X_tr_no = train_df[feats_no_coord].values[mask_tr]
    y_tr = y_train[mask_tr]
    X_t_no = test_df[feats_no_coord].values
    
    # Model Setup
    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_tr_no, y_tr)
    
    pr_rf_t = rf.predict(X_t_no)
    
    mae_t, rmse_t, r2_t, n_t = calculate_metrics(y_test, pr_rf_t)
    
    print(f"Reproduced Metrics:")
    print(f"RMSE: {rmse_t:.5f}")
    print(f"R2: {r2_t:.5f}")
    print(f"Test Samples: {n_t}")
    
    # Check if consistent
    expected_rmse = 1.70279
    expected_r2 = 0.59789
    
    if abs(rmse_t - expected_rmse) < 0.01 and abs(r2_t - expected_r2) < 0.01:
        print("Metrics match expected results. Saving model and metadata...")
        
        # Save model
        model_path = 'models/experiment6/random_forest_nocoord_t+6.joblib'
        joblib.dump(rf, model_path)
        
        # Save metadata
        metadata = {
            "experiment": 6,
            "level": "district",
            "horizon": "6 months",
            "model": "Random Forest",
            "coordinates": False,
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "feature_list": feats_no_coord,
            "reproduced_RMSE": rmse_t,
            "reproduced_R2": r2_t,
            "test_sample_count": n_t,
            "recovery_status": "Success",
            "recovery_script_path": "src/models/recover_experiment6_rf_tplus6.py"
        }
        
        metadata_path = 'models/experiment6/random_forest_nocoord_t+6_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        print("Model recovered and saved.")
    else:
        print("ERROR: Reproduced metrics do not match expected results. Model not saved.")

if __name__ == "__main__":
    main()
