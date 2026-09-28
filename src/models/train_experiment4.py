import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterSampler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Ensure directories exist
for path in ['reports/experiment4/predictions', 'reports/experiment4/plots', 'models/experiment4']:
    os.makedirs(path, exist_ok=True)

# Set random seed
np.random.seed(42)

def calculate_metrics(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(mask):
        return np.nan, np.nan, np.nan
    y_true, y_pred = y_true[mask], y_pred[mask]
    return mean_absolute_error(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred)), r2_score(y_true, y_pred)

def train_and_eval(model_class, X_train, y_train, X_val, y_val, X_test, y_test, **kwargs):
    train_mask = (~np.isnan(y_train)) & (~np.isnan(X_train).any(axis=1))
    X_train_clean, y_train_clean = X_train[train_mask], y_train[train_mask]
    
    val_mask = (~np.isnan(y_val)) & (~np.isnan(X_val).any(axis=1))
    X_val_clean, y_val_clean = X_val[val_mask], y_val[val_mask]
    
    test_mask = (~np.isnan(y_test)) & (~np.isnan(X_test).any(axis=1))
    X_test_clean, y_test_clean = X_test[test_mask], y_test[test_mask]
    
    if len(y_train_clean) == 0:
        return None, None
        
    model = model_class(random_state=42, **kwargs)
    model.fit(X_train_clean, y_train_clean)
    
    val_preds = model.predict(X_val_clean)
    test_preds = model.predict(X_test_clean)
    
    val_metrics = calculate_metrics(y_val_clean, val_preds)
    test_metrics = calculate_metrics(y_test_clean, test_preds)
    
    return val_metrics, test_metrics, test_mask.sum(), model

def main():
    print("Loading data...")
    df = pd.read_csv('data/processed/model_ready_with_targets.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    # Preprocessing (Experiment 2B Treatment)
    df['GRACE_Missing'] = df['GRACE_TWS'].isnull().astype(int)
    train_grace_median = df.loc[:train_end-1, 'GRACE_TWS'].median()
    df['GRACE_TWS'] = df['GRACE_TWS'].fillna(train_grace_median)
    
    # Seasonal Features
    df['Month'] = df['Date'].dt.month
    df['Sin_Month'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Cos_Month'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    # Lag Features
    df['Groundwater_Lag_1'] = df['Groundwater_Mean'].shift(1)
    df['Groundwater_Lag_3'] = df['Groundwater_Mean'].shift(3)
    df['Groundwater_Lag_6'] = df['Groundwater_Mean'].shift(6)
    df['Groundwater_Lag_12'] = df['Groundwater_Mean'].shift(12)
    
    df['Rainfall_Lag_1'] = df['Rainfall'].shift(1)
    df['Rainfall_Lag_3'] = df['Rainfall'].shift(3)
    df['Rainfall_Lag_6'] = df['Rainfall'].shift(6)
    
    df['Temperature_Lag_1'] = df['Temperature'].shift(1)
    df['Temperature_Lag_3'] = df['Temperature'].shift(3)
    
    df['GRACE_Lag_1'] = df['GRACE_TWS'].shift(1)
    df['GRACE_Lag_3'] = df['GRACE_TWS'].shift(3)
    df['GRACE_Lag_6'] = df['GRACE_TWS'].shift(6)
    
    df['ONI_Lag_1'] = df['ONI'].shift(1)
    df['ONI_Lag_3'] = df['ONI'].shift(3)
    
    df['MEI_Lag_1'] = df['MEI'].shift(1)
    df['MEI_Lag_3'] = df['MEI'].shift(3)
    
    # Rolling Features
    df['Rainfall_Rolling_3M'] = df['Rainfall'].rolling(3).mean()
    df['Rainfall_Rolling_6M'] = df['Rainfall'].rolling(6).mean()
    df['Groundwater_Rolling_3M'] = df['Groundwater_Mean'].rolling(3).mean()
    df['Groundwater_Rolling_6M'] = df['Groundwater_Mean'].rolling(6).mean()
    df['GRACE_Rolling_3M'] = df['GRACE_TWS'].rolling(3).mean()
    df['GRACE_Rolling_6M'] = df['GRACE_TWS'].rolling(6).mean()
    
    train_base = df.iloc[:train_end].copy()
    val_base = df.iloc[train_end:val_end].copy()
    test_base = df.iloc[val_end:].copy()
    
    targets = ['Groundwater_t+1', 'Groundwater_t+3', 'Groundwater_t+6']
    base_features = [
        'GRACE_TWS', 'Rainfall', 'Temperature', 'ONI', 'MEI', 
        'Groundwater_Mean', 'Groundwater_Well_Count', 'GRACE_Missing'
    ]
    
    candidate_features = [
        'Groundwater_Lag_1', 'Groundwater_Lag_3', 'Groundwater_Lag_6', 'Groundwater_Lag_12',
        'Rainfall_Lag_1', 'Rainfall_Lag_3', 'Rainfall_Lag_6',
        'GRACE_Lag_1', 'GRACE_Lag_3', 'GRACE_Lag_6',
        'Temperature_Lag_1', 'Temperature_Lag_3',
        'ONI_Lag_1', 'ONI_Lag_3',
        'MEI_Lag_1', 'MEI_Lag_3',
        'Groundwater_Rolling_3M', 'Groundwater_Rolling_6M',
        'Rainfall_Rolling_3M', 'Rainfall_Rolling_6M',
        'GRACE_Rolling_3M', 'GRACE_Rolling_6M',
        'Sin_Month', 'Cos_Month'
    ]
    
    print("==================================================")
    print("STEP 1: INDIVIDUAL FEATURE TESTING")
    print("==================================================")
    screening_results = []
    
    for feature in candidate_features:
        features_to_use = base_features + [feature]
        for target in targets:
            X_train = train_base[features_to_use].values
            X_val = val_base[features_to_use].values
            X_test = test_base[features_to_use].values
            
            y_train = train_base[target].values
            y_val = val_base[target].values
            y_test = test_base[target].values
            
            val_mets, test_mets, ts, _ = train_and_eval(XGBRegressor, X_train, y_train, X_val, y_val, X_test, y_test)
            if val_mets is not None:
                screening_results.append({
                    'Feature': feature,
                    'Horizon': target,
                    'Validation MAE': val_mets[0],
                    'Validation RMSE': val_mets[1],
                    'Validation R2': val_mets[2],
                    'Test MAE': test_mets[0],
                    'Test RMSE': test_mets[1],
                    'Test R2': test_mets[2],
                    'Test Samples': ts
                })
                
    screening_df = pd.DataFrame(screening_results)
    screening_df.to_csv('reports/experiment4/feature_screening.csv', index=False)
    
    # Calculate average validation RMSE rank across horizons
    feature_ranks = []
    for feature in candidate_features:
        avg_rank = 0
        valid = True
        for target in targets:
            sub = screening_df[(screening_df['Feature'] == feature) & (screening_df['Horizon'] == target)]
            if not sub.empty:
                avg_rank += sub['Validation RMSE'].values[0]
            else:
                valid = False
        if valid:
            feature_ranks.append({'Feature': feature, 'Avg_Val_RMSE': avg_rank / 3})
            
    rank_df = pd.DataFrame(feature_ranks).sort_values('Avg_Val_RMSE')
    top_features = rank_df['Feature'].tolist()
    
    # Create Compact Feature Sets
    print("==================================================")
    print("STEP 2: SMALL FEATURE COMBINATIONS")
    print("==================================================")
    feature_sets = {
        'SET_A': top_features[:2],
        'SET_B': top_features[:3],
        'SET_C': top_features[:5],
        'SET_D': [f for f in ['Groundwater_Lag_1', 'Rainfall_Rolling_3M', 'GRACE_Lag_3'] if f in candidate_features][:3]
    }
    
    # Evaluate feature sets on validation data to pick the best set
    set_results = []
    for set_name, fs in feature_sets.items():
        avg_val_rmse = 0
        features_to_use = base_features + fs
        for target in targets:
            X_train = train_base[features_to_use].values
            X_val = val_base[features_to_use].values
            X_test = test_base[features_to_use].values
            y_train = train_base[target].values
            y_val = val_base[target].values
            y_test = test_base[target].values
            
            val_mets, _, _, _ = train_and_eval(XGBRegressor, X_train, y_train, X_val, y_val, X_test, y_test)
            if val_mets is not None:
                avg_val_rmse += val_mets[1]
        set_results.append({'Set': set_name, 'Avg_Val_RMSE': avg_val_rmse / 3})
    
    best_set_name = sorted(set_results, key=lambda x: x['Avg_Val_RMSE'])[0]['Set']
    best_features = feature_sets[best_set_name]
    print(f"Selected Feature Set: {best_set_name} with features {best_features}")
    
    pd.DataFrame({'Feature': best_features}).to_csv('reports/experiment4/selected_features.csv', index=False)
    
    print("==================================================")
    print("STEP 3: XGBOOST REGULARIZATION (VALIDATION ONLY)")
    print("==================================================")
    
    param_grid = {
        'max_depth': [2, 3, 4],
        'learning_rate': [0.03, 0.05, 0.1],
        'n_estimators': [50, 100, 200],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.7, 0.85, 1.0],
        'colsample_bytree': [0.7, 0.85, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 5, 10]
    }
    
    param_list = list(ParameterSampler(param_grid, n_iter=25, random_state=42))
    
    features_to_use = base_features + best_features
    
    tuning_results = []
    best_params_overall = None
    best_val_rmse = float('inf')
    
    for i, params in enumerate(param_list):
        avg_val_rmse = 0
        avg_val_r2 = 0
        valid = True
        
        for target in targets:
            X_train = train_base[features_to_use].values
            X_val = val_base[features_to_use].values
            X_test = test_base[features_to_use].values
            y_train = train_base[target].values
            y_val = val_base[target].values
            y_test = test_base[target].values
            
            val_mets, _, _, _ = train_and_eval(XGBRegressor, X_train, y_train, X_val, y_val, X_test, y_test, **params)
            
            if val_mets is not None:
                avg_val_rmse += val_mets[1]
                avg_val_r2 += val_mets[2]
            else:
                valid = False
                
        if valid:
            avg_val_rmse /= 3
            avg_val_r2 /= 3
            row = params.copy()
            row['Feature Set'] = best_set_name
            row['Validation RMSE'] = avg_val_rmse
            row['Validation R2'] = avg_val_r2
            tuning_results.append(row)
            
            if avg_val_rmse < best_val_rmse:
                best_val_rmse = avg_val_rmse
                best_params_overall = params
                
    pd.DataFrame(tuning_results).to_csv('reports/experiment4/xgboost_tuning_results.csv', index=False)
    print(f"Best XGBoost Params: {best_params_overall}")
    
    print("==================================================")
    print("FINAL EVALUATION ON UNTOUCHED TEST SET")
    print("==================================================")
    
    final_results = []
    
    # Read prev baselines for comparison
    prev_results = []
    try:
        exp1_df = pd.read_csv('reports/ml_results.csv')
        for _, row in exp1_df.iterrows():
            prev_results.append({'Experiment': 'Exp 1', 'Model': row['Model'], 'Horizon': row['Horizon'], 'RMSE': row['RMSE'], 'R2': row['R2']})
    except:
        pass
    try:
        exp2_df = pd.read_csv('reports/experiment2/experiment2_results.csv')
        for _, row in exp2_df.iterrows():
            prev_results.append({'Experiment': row['Variant'], 'Model': row['Model'], 'Horizon': row['Horizon'], 'RMSE': row['RMSE'], 'R2': row['R2']})
    except:
        pass
    try:
        exp3_df = pd.read_csv('reports/experiment3/experiment3_results.csv')
        for _, row in exp3_df.iterrows():
            prev_results.append({'Experiment': 'Exp 3', 'Model': row['Model'], 'Horizon': row['Horizon'], 'RMSE': row['RMSE'], 'R2': row['R2']})
    except:
        pass
    prev_df = pd.DataFrame(prev_results)

    final_models_saved = False
    
    for target in targets:
        X_train = train_base[features_to_use].values
        X_val = val_base[features_to_use].values
        X_test = test_base[features_to_use].values
        y_train = train_base[target].values
        y_val = val_base[target].values
        y_test = test_base[target].values
        
        # Persistence Baseline
        y_val_persistence = val_base['Groundwater_Mean'].values
        y_test_persistence = test_base['Groundwater_Mean'].values
        val_mae, val_rmse, val_r2 = calculate_metrics(y_val, y_val_persistence)
        test_mae, test_rmse, test_r2 = calculate_metrics(y_test, y_test_persistence)
        ts = (~np.isnan(y_test)).sum()
        final_results.append({'Experiment': 'Exp 4', 'Model': 'Persistence', 'Horizon': target, 'Validation RMSE': val_rmse, 'Validation R2': val_r2, 'Test RMSE': test_rmse, 'Test R2': test_r2, 'Test Samples': ts})
        
        # Random Forest (with best compact features, default params)
        val_mets, test_mets, ts_tree, rf_model = train_and_eval(RandomForestRegressor, X_train, y_train, X_val, y_val, X_test, y_test, n_estimators=100, max_depth=10)
        if val_mets:
            final_results.append({'Experiment': 'Exp 4', 'Model': 'Random Forest', 'Horizon': target, 'Validation RMSE': val_mets[1], 'Validation R2': val_mets[2], 'Test RMSE': test_mets[1], 'Test R2': test_mets[2], 'Test Samples': ts_tree})
        
        # XGBoost (with best compact features AND tuned params)
        val_mets, test_mets, ts_tree, xgb_model = train_and_eval(XGBRegressor, X_train, y_train, X_val, y_val, X_test, y_test, **best_params_overall)
        if val_mets:
            final_results.append({'Experiment': 'Exp 4', 'Model': 'XGBoost', 'Horizon': target, 'Validation RMSE': val_mets[1], 'Validation R2': val_mets[2], 'Test RMSE': test_mets[1], 'Test R2': test_mets[2], 'Test Samples': ts_tree})
            joblib.dump(xgb_model, f'models/experiment4/xgboost_{target}.joblib')
            final_models_saved = True
            
            # Save XGB plot Actual vs Pred
            test_mask = (~np.isnan(y_test)) & (~np.isnan(X_test).any(axis=1))
            X_test_clean = X_test[test_mask]
            y_test_clean = y_test[test_mask]
            preds_xgb = xgb_model.predict(X_test_clean)
            
            plt.figure(figsize=(10, 5))
            plt.scatter(y_test_clean, preds_xgb, alpha=0.5)
            plt.plot([y_test_clean.min(), y_test_clean.max()], [y_test_clean.min(), y_test_clean.max()], 'r--')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title(f'XGBoost {target} - Actual vs Predicted (Exp 4)')
            plt.tight_layout()
            plt.savefig(f'reports/experiment4/plots/XGB_Actual_vs_Pred_{target}.png')
            plt.close()
            
            # Feature Importance for Final Model
            fi_xgb = pd.DataFrame({'Feature': features_to_use, 'Importance': xgb_model.feature_importances_})
            plt.figure(figsize=(10, 8))
            fi_xgb.sort_values('Importance', ascending=True).plot.barh(x='Feature', y='Importance', legend=False)
            plt.title(f'Feature Importance XGBoost ({target})')
            plt.tight_layout()
            plt.savefig(f'reports/experiment4/plots/XGB_Importance_{target}.png')
            plt.close()

    # Create Comparison Table
    combined_results = []
    exp4_df = pd.DataFrame(final_results)
    
    for target in targets:
        for model in ['Persistence', 'Random Forest', 'XGBoost']:
            row = {'Model': model, 'Feature Set': best_set_name if model != 'Persistence' else 'Baseline', 'Horizon': target}
            
            # Exp 2B (Baseline)
            exp2b_row = prev_df[(prev_df['Horizon'] == target) & (prev_df['Model'] == model) & (prev_df['Experiment'] == '2B - Impute')] if not prev_df.empty else pd.DataFrame()
            if not exp2b_row.empty:
                row['Exp 2B Test RMSE'] = exp2b_row['RMSE'].values[0]
                row['Exp 2B Test R2'] = exp2b_row['R2'].values[0]
            else:
                row['Exp 2B Test RMSE'] = np.nan
                row['Exp 2B Test R2'] = np.nan
                
            # Exp 4
            exp4_row = exp4_df[(exp4_df['Horizon'] == target) & (exp4_df['Model'] == model)]
            if not exp4_row.empty:
                row['Validation RMSE'] = exp4_row['Validation RMSE'].values[0]
                row['Validation R2'] = exp4_row['Validation R2'].values[0]
                row['Test RMSE'] = exp4_row['Test RMSE'].values[0]
                row['Test R2'] = exp4_row['Test R2'].values[0]
                row['Test Samples'] = exp4_row['Test Samples'].values[0]
                
                if not np.isnan(row['Exp 2B Test RMSE']):
                    row['ΔRMSE vs Exp 2B'] = row['Test RMSE'] - row['Exp 2B Test RMSE']
                    row['ΔR2 vs Exp 2B'] = row['Test R2'] - row['Exp 2B Test R2']
            
            combined_results.append(row)
            
    combined_df = pd.DataFrame(combined_results)
    combined_df.to_csv('reports/experiment4/experiment4_comparison.csv', index=False)
    
    # Validation / Test Plot Comparison for Exp 4 XGBoost
    xgb_exp4 = exp4_df[exp4_df['Model'] == 'XGBoost']
    
    plt.figure(figsize=(8, 5))
    plt.bar(xgb_exp4['Horizon'], xgb_exp4['Validation R2'], color='blue', alpha=0.6, label='Validation R2')
    plt.bar(xgb_exp4['Horizon'], xgb_exp4['Test R2'], color='green', alpha=0.6, label='Test R2')
    plt.title('XGBoost R2: Validation vs Test (Exp 4)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('reports/experiment4/plots/Validation_vs_Test_R2.png')
    plt.close()

if __name__ == "__main__":
    main()
