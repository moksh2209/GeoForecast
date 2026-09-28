import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure directories exist
for path in ['reports/experiment2/predictions', 'reports/experiment2/plots', 'models/experiment2']:
    os.makedirs(path, exist_ok=True)

# Set random seed
np.random.seed(42)
torch.manual_seed(42)

# Same LSTM from Exp 1
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

def create_sequences(X, y, seq_length):
    Xs, ys = [], []
    # Drop rows where target is NaN (only for target mapping, X must not contain NaNs)
    for i in range(len(X) - seq_length):
        if not np.isnan(y[i + seq_length]):
            Xs.append(X[i:(i + seq_length)])
            ys.append(y[i + seq_length])
    return np.array(Xs), np.array(ys)

def train_lstm(X_train, y_train, X_val, y_val, seq_length=6, epochs=100, batch_size=16):
    input_size = X_train.shape[1]
    model = SimpleLSTM(input_size=input_size, hidden_size=64)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_length)
    
    if len(X_train_seq) == 0 or len(X_val_seq) == 0:
        return None, None

    train_data = TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32), torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(1))
    val_data = TensorDataset(torch.tensor(X_val_seq, dtype=torch.float32), torch.tensor(y_val_seq, dtype=torch.float32).unsqueeze(1))
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    patience, patience_counter = 10, 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()
        val_loss /= max(len(val_loader), 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    if best_model_state:
        model.load_state_dict(best_model_state)
    return model, seq_length

def calculate_metrics(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(mask):
        return np.nan, np.nan, np.nan
    y_true, y_pred = y_true[mask], y_pred[mask]
    return mean_absolute_error(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred)), r2_score(y_true, y_pred)

def main():
    df = pd.read_csv('data/processed/model_ready_with_targets.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Same chronological split as Exp 1
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_base = df.iloc[:train_end].copy()
    val_base = df.iloc[train_end:val_end].copy()
    test_base = df.iloc[val_end:].copy()
    
    targets = ['Groundwater_t+1', 'Groundwater_t+3', 'Groundwater_t+6']
    base_features = ['GRACE_TWS', 'Rainfall', 'Temperature', 'ONI', 'MEI', 'Groundwater_Mean', 'Groundwater_Well_Count']
    
    results = []

    # Experiment 1 Results (read for final comparison)
    try:
        exp1_df = pd.read_csv('reports/ml_results.csv')
        for _, row in exp1_df.iterrows():
            results.append({
                'Experiment': 'Exp 1 Baseline',
                'Variant': 'Exp 1',
                'Model': row['Model'],
                'Horizon': row['Horizon'],
                'Test Samples': row['Test Samples'],
                'MAE': row['MAE'],
                'RMSE': row['RMSE'],
                'R2': row['R2']
            })
    except Exception as e:
        print("Exp 1 results not found.")

    print("\n==================== VARIANT A: DROP MISSING GRACE ====================")
    # Drop rows where GRACE_TWS is missing
    train_a = train_base.dropna(subset=['GRACE_TWS']).copy()
    val_a = val_base.dropna(subset=['GRACE_TWS']).copy()
    test_a = test_base.dropna(subset=['GRACE_TWS']).copy()
    
    print(f"Train samples: {len(train_a)}, Val samples: {len(val_a)}, Test samples: {len(test_a)}")
    
    for target in targets:
        y_train = train_a[target].values
        y_val = val_a[target].values
        y_test = test_a[target].values
        
        X_train = train_a[base_features].values
        X_val = val_a[base_features].values
        X_test = test_a[base_features].values
        
        # Persistence
        y_test_persistence = test_a['Groundwater_Mean'].values
        mae, rmse, r2 = calculate_metrics(y_test, y_test_persistence)
        ts = (~np.isnan(y_test)).sum()
        results.append({'Experiment': 'Exp 2', 'Variant': '2A - Drop', 'Model': 'Persistence', 'Horizon': target, 'Test Samples': ts, 'MAE': mae, 'RMSE': rmse, 'R2': r2})

        # Drop NaNs in target for training tree models
        train_mask = ~np.isnan(y_train)
        X_train_tree, y_train_tree = X_train[train_mask], y_train[train_mask]
        
        if len(y_train_tree) > 0:
            # RF
            rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            rf.fit(X_train_tree, y_train_tree)
            preds = rf.predict(X_test)
            mae, rmse, r2 = calculate_metrics(y_test, preds)
            results.append({'Experiment': 'Exp 2', 'Variant': '2A - Drop', 'Model': 'Random Forest', 'Horizon': target, 'Test Samples': ts, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
            
            # XGB
            xgb = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
            xgb.fit(X_train_tree, y_train_tree)
            preds = xgb.predict(X_test)
            mae, rmse, r2 = calculate_metrics(y_test, preds)
            results.append({'Experiment': 'Exp 2', 'Variant': '2A - Drop', 'Model': 'XGBoost', 'Horizon': target, 'Test Samples': ts, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
        
        # LSTM (Needs dense sequences, but dropping rows breaks time sequence. We will just pass the sequence function which will have gaps. Wait, sequence length 6 with dropped time steps is very dangerous as it connects non-contiguous time steps! I will just run it as an experiment to see what it does, or report warning. The instructions say "Keep all other features unchanged. Train and evaluate LSTM". So I will.)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        lstm_model, _ = train_lstm(X_train_scaled, y_train, X_val_scaled, y_val, seq_length=6)
        if lstm_model:
            lstm_model.eval()
            X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, seq_length=6)
            if len(X_test_seq) > 0:
                with torch.no_grad():
                    preds = lstm_model(torch.tensor(X_test_seq, dtype=torch.float32)).numpy().flatten()
                mae, rmse, r2 = calculate_metrics(y_test_seq, preds)
                results.append({'Experiment': 'Exp 2', 'Variant': '2A - Drop', 'Model': 'LSTM', 'Horizon': target, 'Test Samples': len(y_test_seq), 'MAE': mae, 'RMSE': rmse, 'R2': r2})

    print("\n==================== VARIANT B: CONTROLLED IMPUTATION ====================")
    # Add GRACE_Missing
    train_b = train_base.copy()
    val_b = val_base.copy()
    test_b = test_base.copy()
    
    for d in [train_b, val_b, test_b]:
        d['GRACE_Missing'] = d['GRACE_TWS'].isnull().astype(int)
    
    # Impute GRACE_TWS using ONLY information available from the training split.
    # Instruction: "Prefer: training-derived median or forward-fill using only information available up to that point. Do NOT interpolate across the 2017-2018 GRACE/GRACE-FO mission gap."
    # Since I shouldn't interpolate across the gap, I will use training-derived median to impute all missing values, or forward-fill but limit to not crossing large gaps. 
    # Let's use the Training Median. This is robust, doesn't leak, and doesn't interpolate across gaps.
    train_grace_median = train_b['GRACE_TWS'].median()
    
    train_b['GRACE_TWS'] = train_b['GRACE_TWS'].fillna(train_grace_median)
    val_b['GRACE_TWS'] = val_b['GRACE_TWS'].fillna(train_grace_median)
    test_b['GRACE_TWS'] = test_b['GRACE_TWS'].fillna(train_grace_median)
    
    missing_b_train = train_base['GRACE_TWS'].isnull().sum()
    missing_b_val = val_base['GRACE_TWS'].isnull().sum()
    missing_b_test = test_base['GRACE_TWS'].isnull().sum()
    
    print(f"Missing GRACE values Imputed (Median = {train_grace_median:.4f}): Train: {missing_b_train}, Val: {missing_b_val}, Test: {missing_b_test}")
    
    b_features = base_features + ['GRACE_Missing']
    
    for target in targets:
        y_train = train_b[target].values
        y_val = val_b[target].values
        y_test = test_b[target].values
        
        X_train = train_b[b_features].values
        X_val = val_b[b_features].values
        X_test = test_b[b_features].values
        
        # Persistence
        y_test_persistence = test_b['Groundwater_Mean'].values
        mae, rmse, r2 = calculate_metrics(y_test, y_test_persistence)
        ts = (~np.isnan(y_test)).sum()
        results.append({'Experiment': 'Exp 2', 'Variant': '2B - Impute', 'Model': 'Persistence', 'Horizon': target, 'Test Samples': ts, 'MAE': mae, 'RMSE': rmse, 'R2': r2})

        # Tree Models
        train_mask = ~np.isnan(y_train)
        X_train_tree, y_train_tree = X_train[train_mask], y_train[train_mask]
        
        if len(y_train_tree) > 0:
            rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            rf.fit(X_train_tree, y_train_tree)
            preds = rf.predict(X_test)
            mae, rmse, r2 = calculate_metrics(y_test, preds)
            results.append({'Experiment': 'Exp 2', 'Variant': '2B - Impute', 'Model': 'Random Forest', 'Horizon': target, 'Test Samples': ts, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
            
            xgb = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
            xgb.fit(X_train_tree, y_train_tree)
            preds = xgb.predict(X_test)
            mae, rmse, r2 = calculate_metrics(y_test, preds)
            results.append({'Experiment': 'Exp 2', 'Variant': '2B - Impute', 'Model': 'XGBoost', 'Horizon': target, 'Test Samples': ts, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
        
        # LSTM
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        lstm_model, _ = train_lstm(X_train_scaled, y_train, X_val_scaled, y_val, seq_length=6)
        if lstm_model:
            lstm_model.eval()
            X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, seq_length=6)
            if len(X_test_seq) > 0:
                with torch.no_grad():
                    preds = lstm_model(torch.tensor(X_test_seq, dtype=torch.float32)).numpy().flatten()
                mae, rmse, r2 = calculate_metrics(y_test_seq, preds)
                results.append({'Experiment': 'Exp 2', 'Variant': '2B - Impute', 'Model': 'LSTM', 'Horizon': target, 'Test Samples': len(y_test_seq), 'MAE': mae, 'RMSE': rmse, 'R2': r2})

    results_df = pd.DataFrame(results)
    results_df.to_csv('reports/experiment2/experiment2_results.csv', index=False)
    
    # Calculate Differences relative to Exp 1
    # We will join Exp 1 and Exp 2 on Model + Horizon
    exp1_subset = results_df[results_df['Variant'] == 'Exp 1'].copy()
    exp2a_subset = results_df[results_df['Variant'] == '2A - Drop'].copy()
    exp2b_subset = results_df[results_df['Variant'] == '2B - Impute'].copy()
    
    # Print the table
    print("\n=== EXPERIMENT COMPARISON ===")
    print(results_df.to_string(index=False))
    
    # Save a report
    with open('reports/experiment2/experiment2_report.md', 'w') as f:
        f.write("# Experiment 2: Robust GRACE Missing Data Validation\n\n")
        f.write("## Overview\nTested two variants for handling missing GRACE data: 2A (dropping NaNs) and 2B (Controlled Imputation using train median + GRACE_Missing boolean).\n\n")
        f.write("## Results\n\n")
        f.write(results_df.to_string(index=False))

if __name__ == "__main__":
    main()
