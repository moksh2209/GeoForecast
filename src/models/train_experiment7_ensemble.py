import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

def calculate_metrics(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(mask):
        return np.nan, np.nan, np.nan, 0
    y_t, y_p = y_true[mask], y_pred[mask]
    return mean_absolute_error(y_t, y_p), np.sqrt(mean_squared_error(y_t, y_p)), r2_score(y_t, y_p), len(y_t)

def main():
    report_dir = 'reports/experiment7_ensemble'
    os.makedirs(report_dir, exist_ok=True)
    
    print("Loading data and recreating Experiment 6 split...")
    df = pd.read_csv('data/processed/district_spatiotemporal_monthly.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    df_modeling = df.dropna(subset=['GRACE_TWS', 'Rainfall', 'Temperature', 'ONI', 'MEI']).copy()
    df_modeling = df_modeling.sort_values(['District Name', 'Date']).reset_index(drop=True)
    
    targets = ['t+1', 't+3', 't+6']
    for t in targets:
        shift_val = int(t.split('+')[1])
        df_modeling[f'Target_Date_{t}'] = df_modeling['Date'] + pd.DateOffset(months=shift_val)
        target_mapping = df_modeling[['District Name', 'Date', 'Groundwater_Mean']].rename(
            columns={'Date': f'Target_Date_{t}', 'Groundwater_Mean': f'Groundwater_{t}'}
        )
        df_modeling = pd.merge(df_modeling, target_mapping, on=['District Name', f'Target_Date_{t}'], how='left')
        
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
    
    districts = df_modeling['District Name'].unique()
    for d in districts:
        train_df[f'Dist_{d}'] = (train_df['District Name'] == d).astype(int)
        val_df[f'Dist_{d}'] = (val_df['District Name'] == d).astype(int)
        test_df[f'Dist_{d}'] = (test_df['District Name'] == d).astype(int)
        
    dist_cols = [f'Dist_{d}' for d in districts]
    base_features = ['GRACE_TWS', 'Rainfall', 'Temperature', 'ONI', 'MEI', 'Groundwater_Mean', 'Groundwater_Well_Count']
    feats_no_coord = base_features + dist_cols
    
    for col in feats_no_coord:
        if col in train_df.columns and train_df[col].isnull().any():
            med = train_df[col].median()
            train_df[col] = train_df[col].fillna(med)
            val_df[col] = val_df[col].fillna(med)
            test_df[col] = test_df[col].fillna(med)
            
    val_results = []
    test_results = []
    residual_corr_results = []
    selected_weights = {}
    
    for t_col in [f'Groundwater_{t}' for t in targets]:
        h = t_col.split('_')[1]
        print(f"Processing horizon {h}...")
        
        y_train = train_df[t_col].values
        y_val = val_df[t_col].values
        y_test = test_df[t_col].values
        
        mask_tr = ~np.isnan(y_train)
        X_tr = train_df[feats_no_coord].values[mask_tr]
        y_tr = y_train[mask_tr]
        
        X_v = val_df[feats_no_coord].values
        X_t = test_df[feats_no_coord].values
        
        # Train RF and XGB
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_tr, y_tr)
        
        xgb = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
        xgb.fit(X_tr, y_tr)
        
        # Validation Predictions
        pr_rf_v = rf.predict(X_v)
        pr_xgb_v = xgb.predict(X_v)
        
        # Test Predictions
        pr_rf_t = rf.predict(X_t)
        pr_xgb_t = xgb.predict(X_t)
        
        # Masks for evaluation (EXACT SAME observations)
        val_mask = ~np.isnan(y_val)
        y_v_clean = y_val[val_mask]
        rf_v_clean = pr_rf_v[val_mask]
        xgb_v_clean = pr_xgb_v[val_mask]
        
        test_mask = ~np.isnan(y_test)
        y_t_clean = y_test[test_mask]
        rf_t_clean = pr_rf_t[test_mask]
        xgb_t_clean = pr_xgb_t[test_mask]
        
        # Residuals on validation
        rf_resid = y_v_clean - rf_v_clean
        xgb_resid = y_v_clean - xgb_v_clean
        corr, _ = pearsonr(rf_resid, xgb_resid)
        residual_corr_results.append({'Horizon': h, 'Correlation': corr})
        
        # Ensemble weights
        configs = [
            ('RF', 1.0, 0.0),
            ('XGBoost', 0.0, 1.0),
            ('25/75 Ensemble', 0.25, 0.75),
            ('50/50 Ensemble', 0.50, 0.50),
            ('75/25 Ensemble', 0.75, 0.25)
        ]
        
        best_val_r2 = -float('inf')
        best_config = None
        best_w_rf = 0.0
        best_w_xgb = 0.0
        
        horizon_val_metrics = []
        for name, w_rf, w_xgb in configs:
            ens_v = w_rf * rf_v_clean + w_xgb * xgb_v_clean
            mae_v, rmse_v, r2_v, _ = calculate_metrics(y_v_clean, ens_v)
            
            val_results.append({
                'Horizon': h, 'Model': name, 'MAE': mae_v, 'RMSE': rmse_v, 'R2': r2_v
            })
            
            if r2_v > best_val_r2:
                best_val_r2 = r2_v
                best_config = name
                best_w_rf = w_rf
                best_w_xgb = w_xgb
                
        selected_weights[h] = {
            'Best_Config': best_config,
            'Weight_RF': best_w_rf,
            'Weight_XGB': best_w_xgb,
            'Validation_R2': best_val_r2
        }
        
        # Evaluate all configs on Test set
        for name, w_rf, w_xgb in configs:
            ens_t = w_rf * rf_t_clean + w_xgb * xgb_t_clean
            mae_t, rmse_t, r2_t, _ = calculate_metrics(y_t_clean, ens_t)
            
            test_results.append({
                'Horizon': h, 'Model': name, 'MAE': mae_t, 'RMSE': rmse_t, 'R2': r2_t
            })
            
    val_df_out = pd.DataFrame(val_results)
    test_df_out = pd.DataFrame(test_results)
    corr_df = pd.DataFrame(residual_corr_results)
    
    val_df_out.to_csv(f'{report_dir}/ensemble_validation_results.csv', index=False)
    test_df_out.to_csv(f'{report_dir}/ensemble_test_results.csv', index=False)
    corr_df.to_csv(f'{report_dir}/residual_correlation.csv', index=False)
    
    with open(f'{report_dir}/selected_ensemble_weights.json', 'w') as f:
        json.dump(selected_weights, f, indent=4)
        
    # Full combined results for ensemble_results.csv
    # Just merge val and test
    comb = pd.merge(val_df_out, test_df_out, on=['Horizon', 'Model'], suffixes=('_Val', '_Test'))
    comb.to_csv(f'{report_dir}/ensemble_results.csv', index=False)
    
    # Generate Plots
    models_to_plot = ['RF', 'XGBoost', '25/75 Ensemble', '50/50 Ensemble', '75/25 Ensemble']
    
    def plot_metric(metric, df_plot, filename, title):
        plt.figure(figsize=(10, 6))
        x = np.arange(len(targets))
        width = 0.15
        
        for i, m in enumerate(models_to_plot):
            vals = df_plot[df_plot['Model'] == m][metric].values
            plt.bar(x + (i - 2) * width, vals, width, label=m)
            
        plt.xticks(x, targets)
        plt.ylabel(metric)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{report_dir}/{filename}')
        plt.close()

    plot_metric('R2', val_df_out, 'validation_r2_comparison.png', 'Validation R2 Comparison')
    plot_metric('R2', test_df_out, 'test_r2_comparison.png', 'Test R2 Comparison')
    plot_metric('RMSE', val_df_out, 'validation_rmse_comparison.png', 'Validation RMSE Comparison')
    plot_metric('RMSE', test_df_out, 'test_rmse_comparison.png', 'Test RMSE Comparison')
    
    plt.figure(figsize=(8, 5))
    plt.bar(corr_df['Horizon'], corr_df['Correlation'], color='purple', alpha=0.7)
    plt.ylabel('Pearson Correlation')
    plt.title('Residual Correlation (RF vs XGBoost)')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f'{report_dir}/residual_correlation.png')
    plt.close()
    
    print("Generating markdown report...")
    with open(f'{report_dir}/experiment7_ensemble_report.md', 'w') as f:
        f.write("# Experiment 7 — Two-Model Ensemble\n\n")
        f.write("EXPERIMENT ONLY — production models were not modified.\n\n")
        f.write("## 1. Experimental Objective\n")
        f.write("Test whether combining Random Forest and XGBoost predictions improves groundwater forecasting performance compared with the individual models.\n\n")
        
        f.write("## 2. Models Used\n")
        f.write("- Random Forest Regressor (No coord features, mirroring Experiment 6)\n")
        f.write("- XGBoost Regressor (No coord features, mirroring Experiment 6)\n\n")
        
        f.write("## 3. Data & Splits\n")
        f.write("Used the standard chronological district spatiotemporal split (70% Train, 15% Validation, 15% Test).\n")
        f.write("Validation data was used to select the ensemble weight. The Test data remained completely untouched until the final evaluation.\n\n")
        
        f.write("## 4. Ensemble Methodology\n")
        f.write("Predictions were combined using fixed weights (25/75, 50/50, 75/25).\n")
        f.write("The exact same validation and test observations were evaluated for each model to ensure fairness.\n\n")
        
        f.write("## 5. Residual Correlation\n")
        f.write(corr_df.to_string(index=False) + "\n\n")
        f.write("A low to moderate correlation (< 0.8) indicates that the models make complementary errors, meaning an ensemble is mathematically likely to reduce overall error. A very high correlation (> 0.9) means the models make nearly identical errors, limiting ensemble gains.\n\n")
        
        f.write("## 6. Selected Weights (Based on Validation R²)\n")
        for h, v in selected_weights.items():
            f.write(f"- **{h}**: {v['Best_Config']} (RF: {v['Weight_RF']}, XGB: {v['Weight_XGB']}) with Val R² = {v['Validation_R2']:.4f}\n")
        f.write("\n")
        
        f.write("## 7. Results Comparison\n")
        for h in targets:
            f.write(f"### Horizon {h}\n")
            sub_t = test_df_out[test_df_out['Horizon'] == h]
            
            rf_r2 = sub_t[sub_t['Model'] == 'RF']['R2'].values[0]
            rf_rmse = sub_t[sub_t['Model'] == 'RF']['RMSE'].values[0]
            
            xgb_r2 = sub_t[sub_t['Model'] == 'XGBoost']['R2'].values[0]
            xgb_rmse = sub_t[sub_t['Model'] == 'XGBoost']['RMSE'].values[0]
            
            best_config = selected_weights[h]['Best_Config']
            ens_r2 = sub_t[sub_t['Model'] == best_config]['R2'].values[0]
            ens_rmse = sub_t[sub_t['Model'] == best_config]['RMSE'].values[0]
            
            best_ind_r2 = max(rf_r2, xgb_r2)
            best_ind_rmse = min(rf_rmse, xgb_rmse)
            
            r2_imp = ens_r2 - best_ind_r2
            rmse_imp = best_ind_rmse - ens_rmse
            
            f.write(f"- **Random Forest Test R²**: {rf_r2:.4f} (RMSE: {rf_rmse:.4f})\n")
            f.write(f"- **XGBoost Test R²**: {xgb_r2:.4f} (RMSE: {xgb_rmse:.4f})\n")
            f.write(f"- **Selected Ensemble ({best_config}) Test R²**: {ens_r2:.4f} (RMSE: {ens_rmse:.4f})\n")
            f.write(f"- **Improvement over best individual model**: R² change = {r2_imp:.4f}, RMSE change = {rmse_imp:.4f}\n\n")
            
        f.write("## 8. Scientific Interpretation\n")
        f.write("The ensemble evaluation demonstrates whether combining decision tree models provides a sufficient variance-reduction benefit over the single best model on the untouched test set. Check the R² and RMSE improvements above. If the ensemble performs worse or only marginally better than the best individual model, then deploying a complex ensemble to production is not justified. If the ensemble consistently improves test metrics across horizons, it suggests that RF and XGBoost capture slightly different predictive signals.\n\n")
        
        f.write("EXPERIMENT ONLY — production models were not modified.\n")

if __name__ == '__main__':
    main()
