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

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Ensure directories exist
os.makedirs('reports/predictions', exist_ok=True)
os.makedirs('reports/plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Define LSTM Model
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

def create_sequences(X, y, seq_length):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        if not np.isnan(y[i + seq_length]):
            Xs.append(X[i:(i + seq_length)])
            ys.append(y[i + seq_length])
    return np.array(Xs), np.array(ys)

def train_lstm(X_train, y_train, X_val, y_val, seq_length=12, epochs=100, batch_size=16):
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
        val_loss /= len(val_loader)

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

def impute_grace(df):
    # Impute GRACE_TWS: forward fill, then backward fill (if starting with NaN)
    # This must be done strictly within the split passed to avoid leakage
    imputed_count = df['GRACE_TWS'].isnull().sum()
    df_imputed = df.copy()
    df_imputed['GRACE_TWS'] = df_imputed['GRACE_TWS'].ffill().bfill() # bfill only for edge case at start
    return df_imputed, imputed_count

def calculate_metrics(y_true, y_pred):
    # drop NaNs if any
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(mask):
        return np.nan, np.nan, np.nan
    y_true, y_pred = y_true[mask], y_pred[mask]
    return mean_absolute_error(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred)), r2_score(y_true, y_pred)

def main():
    df = pd.read_csv('data/processed/model_ready_with_targets.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # 70/15/15 chronological split
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print(f"Train dates: {train_df['Date'].min().strftime('%Y-%m')} to {train_df['Date'].max().strftime('%Y-%m')} (Samples: {len(train_df)})")
    print(f"Val dates:   {val_df['Date'].min().strftime('%Y-%m')} to {val_df['Date'].max().strftime('%Y-%m')} (Samples: {len(val_df)})")
    print(f"Test dates:  {test_df['Date'].min().strftime('%Y-%m')} to {test_df['Date'].max().strftime('%Y-%m')} (Samples: {len(test_df)})")
    
    # Impute GRACE separately
    train_df, train_imputed = impute_grace(train_df)
    val_df, val_imputed = impute_grace(val_df)
    test_df, test_imputed = impute_grace(test_df)
    
    print(f"\nGRACE_TWS Imputations (Forward Fill): Train: {train_imputed}, Val: {val_imputed}, Test: {test_imputed}")
    
    features = ['GRACE_TWS', 'Rainfall', 'Temperature', 'ONI', 'MEI', 'Groundwater_Mean', 'Groundwater_Well_Count']
    targets = ['Groundwater_t+1', 'Groundwater_t+3', 'Groundwater_t+6']
    
    X_train = train_df[features].values
    X_val = val_df[features].values
    X_test = test_df[features].values
    
    # Scaling for LSTM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    results = []

    for target in targets:
        print(f"\n--- Training models for {target} ---")
        y_train = train_df[target].values
        y_val = val_df[target].values
        y_test = test_df[target].values
        
        # Valid mask for tree models (they can't train on NaN targets)
        train_mask = ~np.isnan(y_train)
        X_train_tree, y_train_tree = X_train[train_mask], y_train[train_mask]
        
        # 1. Persistence Baseline
        # Prediction is current Groundwater_Mean
        y_test_persistence = test_df['Groundwater_Mean'].values
        mae_p, rmse_p, r2_p = calculate_metrics(y_test, y_test_persistence)
        test_samples = (~np.isnan(y_test)).sum()
        results.append({'Model': 'Persistence', 'Horizon': target, 'Test Samples': test_samples, 'MAE': mae_p, 'RMSE': rmse_p, 'R2': r2_p})
        
        # Persistence Plots
        plt.figure(figsize=(10, 5))
        valid_idx = ~np.isnan(y_test)
        plt.plot(test_df['Date'][valid_idx], y_test[valid_idx], label='Actual', marker='.')
        plt.plot(test_df['Date'][valid_idx], y_test_persistence[valid_idx], label='Persistence', linestyle='--')
        plt.legend()
        plt.title(f'Persistence - {target}')
        plt.savefig(f'reports/plots/Persistence_{target}.png')
        plt.close()

        # 2. Random Forest
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train_tree, y_train_tree)
        y_pred_rf = rf.predict(X_test)
        mae_rf, rmse_rf, r2_rf = calculate_metrics(y_test, y_pred_rf)
        results.append({'Model': 'Random Forest', 'Horizon': target, 'Test Samples': test_samples, 'MAE': mae_rf, 'RMSE': rmse_rf, 'R2': r2_rf})
        
        # RF Plots
        plt.figure(figsize=(10, 5))
        plt.plot(test_df['Date'][valid_idx], y_test[valid_idx], label='Actual', marker='.')
        plt.plot(test_df['Date'][valid_idx], y_pred_rf[valid_idx], label='Random Forest', linestyle='--')
        plt.legend()
        plt.title(f'Random Forest - {target}')
        plt.savefig(f'reports/plots/Random_Forest_{target}.png')
        plt.close()

        # 3. XGBoost
        xgb = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
        xgb.fit(X_train_tree, y_train_tree)
        y_pred_xgb = xgb.predict(X_test)
        mae_xgb, rmse_xgb, r2_xgb = calculate_metrics(y_test, y_pred_xgb)
        results.append({'Model': 'XGBoost', 'Horizon': target, 'Test Samples': test_samples, 'MAE': mae_xgb, 'RMSE': rmse_xgb, 'R2': r2_xgb})
        
        # XGB Plots
        plt.figure(figsize=(10, 5))
        plt.plot(test_df['Date'][valid_idx], y_test[valid_idx], label='Actual', marker='.')
        plt.plot(test_df['Date'][valid_idx], y_pred_xgb[valid_idx], label='XGBoost', linestyle='--')
        plt.legend()
        plt.title(f'XGBoost - {target}')
        plt.savefig(f'reports/plots/XGBoost_{target}.png')
        plt.close()

        # 4. LSTM
        seq_length = 6
        lstm_model, _ = train_lstm(X_train_scaled, y_train, X_val_scaled, y_val, seq_length=seq_length)
        if lstm_model:
            lstm_model.eval()
            X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, seq_length)
            if len(X_test_seq) > 0:
                with torch.no_grad():
                    y_pred_lstm = lstm_model(torch.tensor(X_test_seq, dtype=torch.float32)).numpy().flatten()
                mae_lstm, rmse_lstm, r2_lstm = calculate_metrics(y_test_seq, y_pred_lstm)
                lstm_test_samples = len(y_test_seq)
                results.append({'Model': 'LSTM', 'Horizon': target, 'Test Samples': lstm_test_samples, 'MAE': mae_lstm, 'RMSE': rmse_lstm, 'R2': r2_lstm})
                
                # LSTM Plots (Note: Dates are offset by seq_length)
                plt.figure(figsize=(10, 5))
                test_dates_seq = test_df['Date'].values[seq_length:seq_length+len(y_test_seq)]
                plt.plot(test_dates_seq, y_test_seq, label='Actual', marker='.')
                plt.plot(test_dates_seq, y_pred_lstm, label='LSTM', linestyle='--')
                plt.legend()
                plt.title(f'LSTM - {target}')
                plt.savefig(f'reports/plots/LSTM_{target}.png')
                plt.close()
            else:
                print(f"Not enough test data for LSTM {target}")
        else:
            print(f"Not enough training data for LSTM {target}")

    results_df = pd.DataFrame(results)
    results_df.to_csv('reports/ml_results.csv', index=False)
    print("\n=== FINAL RESULTS ===")
    print(results_df.to_string(index=False))
    
    print("\nSaved predictions and plots to reports/")

if __name__ == "__main__":
    main()
