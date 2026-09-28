import os
import shutil
import json
import pandas as pd

def setup_production():
    # 1. Directories
    dirs = [
        'models/production/maharashtra',
        'models/production/district',
        'src/prediction',
        'reports/model_registry'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # 2. Copy models
    maha_src = {
        'h1': 'models/final/tplus1/xgboost_Groundwater_t+1.joblib',
        'h3': 'models/final/tplus3/xgboost_Groundwater_t+3.joblib',
        'h6': 'models/final/tplus6/xgboost_Groundwater_t+6.joblib'
    }
    dist_src = {
        'h1': 'models/experiment6/xgboost_nocoord_t+1.joblib',
        'h3': 'models/experiment6/xgboost_nocoord_t+3.joblib',
        'h6': 'models/experiment6/random_forest_nocoord_t+6.joblib'
    }

    maha_dest = 'models/production/maharashtra'
    dist_dest = 'models/production/district'

    for h, src in maha_src.items():
        shutil.copy(src, f"{maha_dest}/model_{h}.joblib")
    
    for h, src in dist_src.items():
        shutil.copy(src, f"{dist_dest}/model_{h}.joblib")

    # 3. Metadata
    
    # Feature lists
    f_maha_13 = ['GRACE_TWS', 'Rainfall', 'Temperature', 'ONI', 'MEI', 'Groundwater_Mean', 'Groundwater_Well_Count', 'GRACE_Missing']
    f_maha_6 = ['GRACE_TWS', 'Rainfall', 'Temperature', 'ONI', 'MEI', 'Groundwater_Mean', 'Groundwater_Well_Count', 'GRACE_Missing', 'ONI_Lag_1', 'Groundwater_Rolling_3M', 'Rainfall_Rolling_3M', 'Temperature_Lag_1', 'Sin_Month']
    
    # Need district list from data to build exact features
    df = pd.read_csv('data/processed/district_spatiotemporal_monthly.csv')
    districts = sorted(df['District Name'].dropna().unique())
    f_dist = ['GRACE_TWS', 'Rainfall', 'Temperature', 'ONI', 'MEI', 'Groundwater_Mean', 'Groundwater_Well_Count'] + [f"Dist_{d}" for d in districts]

    maha_meta = {
        "h1": {
            "experiment": 5, "geographic_level": "maharashtra", "forecast_horizon": "1 month",
            "model_type": "XGBoost", "feature_list": f_maha_13,
            "test_R2": 0.61, "test_RMSE": 1.14,
            "test_sample_count": 30,
            "model_version": "experiment5", "artifact_source": maha_src['h1']
        },
        "h3": {
            "experiment": 5, "geographic_level": "maharashtra", "forecast_horizon": "3 months",
            "model_type": "XGBoost", "feature_list": f_maha_13,
            "test_R2": 0.55, "test_RMSE": 1.31,
            "test_sample_count": 27,
            "model_version": "experiment5", "artifact_source": maha_src['h3']
        },
        "h6": {
            "experiment": 5, "geographic_level": "maharashtra", "forecast_horizon": "6 months",
            "model_type": "Regularized XGBoost", "feature_list": f_maha_6,
            "test_R2": 0.51, "test_RMSE": 1.51,
            "test_sample_count": 23,
            "model_version": "experiment5", "artifact_source": maha_src['h6']
        }
    }

    dist_meta = {
        "h1": {
            "experiment": 6, "geographic_level": "district", "forecast_horizon": "1 month",
            "model_type": "XGBoost", "feature_list": f_dist,
            "test_R2": 0.43, "test_RMSE": 2.22,
            "test_sample_count": 215,
            "model_version": "experiment6", "artifact_source": dist_src['h1']
        },
        "h3": {
            "experiment": 6, "geographic_level": "district", "forecast_horizon": "3 months",
            "model_type": "XGBoost", "feature_list": f_dist,
            "test_R2": 0.35, "test_RMSE": 2.34,
            "test_sample_count": 203,
            "model_version": "experiment6", "artifact_source": dist_src['h3']
        },
        "h6": {
            "experiment": 6, "geographic_level": "district", "forecast_horizon": "6 months",
            "model_type": "Random Forest", "feature_list": f_dist,
            "test_R2": 0.59789, "test_RMSE": 1.70280,
            "test_sample_count": 233, "recovery_status": "Recovered successfully",
            "model_version": "experiment6", "artifact_source": dist_src['h6']
        }
    }

    with open(f"{maha_dest}/metadata.json", "w") as f:
        json.dump(maha_meta, f, indent=4)
        
    with open(f"{dist_dest}/metadata.json", "w") as f:
        json.dump(dist_meta, f, indent=4)

    # 4. Final Model Summary CSV & MD
    summary_data = []
    for k, v in maha_meta.items():
        summary_data.append([v['geographic_level'], v['forecast_horizon'], v['model_type'], v['test_R2'], v['test_RMSE'], v['experiment']])
    for k, v in dist_meta.items():
        summary_data.append([v['geographic_level'], v['forecast_horizon'], v['model_type'], v['test_R2'], v['test_RMSE'], v['experiment']])
        
    df_sum = pd.DataFrame(summary_data, columns=['Level', 'Horizon', 'Model', 'R2', 'RMSE', 'Experiment'])
    df_sum.to_csv('reports/model_registry/final_model_summary.csv', index=False)
    
    with open('reports/model_registry/final_model_summary.md', 'w') as f:
        f.write("# Final Model Summary\n\n")
        f.write("This registry contains the verified production models.\n\n")
        f.write("## Models\n\n")
        f.write("| Level | Horizon | Model | R2 | RMSE | Experiment |\n")
        f.write("|---|---|---|---|---|---|\n")
        for row in summary_data:
            f.write(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} |\n")
        f.write("\n")
        f.write("### Note on Comparison\n")
        f.write("**Experiment 5** = Maharashtra-wide temporal forecasting.\n\n")
        f.write("**Experiment 6** = district-level spatiotemporal panel forecasting.\n\n")
        f.write("Their R² values are not directly comparable because they evaluate different prediction populations.\n")

if __name__ == "__main__":
    setup_production()
