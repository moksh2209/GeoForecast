import os
import pandas as pd
from datetime import datetime
from src.prediction.predictor import predict_maharashtra, predict_district
from src.prediction.model_registry import ModelRegistry
from .database_service import database_service
from dateutil.relativedelta import relativedelta

class PredictionService:
    def __init__(self):
        # We need a sample historical data point for Maharashtra since it's not in the DB
        # Loading real processed data for Maharashtra-wide prediction fallback
        csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/processed/district_spatiotemporal_monthly.csv'))
        self.df = pd.read_csv(csv_path)
        self.registry = ModelRegistry()
            
    def get_supported_districts(self):
        return database_service.get_supported_districts()
        
    def _get_historical_sample_maharashtra(self):
        """Returns a real historical feature dictionary to serve as demo input."""
        base_features = ['GRACE_TWS', 'Rainfall', 'Temperature', 'ONI', 'MEI', 'Groundwater_Mean', 'Groundwater_Well_Count']
        df_valid = self.df.dropna(subset=base_features)
        
        row = df_valid.iloc[-1]
            
        ext_features = ['ONI_Lag_1', 'Groundwater_Rolling_3M', 'Rainfall_Rolling_3M', 'Temperature_Lag_1', 'Sin_Month']
        features = row.to_dict()
        
        for col in base_features + ext_features:
            if col not in features or pd.isna(features[col]):
                features[col] = 0.0
                
        forecast_date = str(row['Date']) if 'Date' in row else "2024-01-01"
        return features, forecast_date

    def forecast_maharashtra(self, horizon: int):
        if horizon not in [1, 3, 6]:
            raise ValueError("Invalid horizon")
            
        features, forecast_date = self._get_historical_sample_maharashtra()
        return predict_maharashtra(horizon, features, forecast_date=forecast_date)
        
    def forecast_district(self, district: str, horizon: int):
        if horizon not in [1, 3, 6]:
            raise ValueError("Invalid horizon")
            
        dist_record = database_service.get_district(district)
        if not dist_record or not dist_record.is_supported:
            raise ValueError(f"District {district} not supported")
            
        obs = database_service.get_latest_district_observation(district)
        if not obs:
            raise ValueError(f"No historical observations found for district {district}")
            
        df_dist = self.df[self.df['District Name'] == district].copy()
        if df_dist.empty:
            raise ValueError(f"No historical observations found in dataset for district {district}")
            
        df_dist['Date'] = pd.to_datetime(df_dist['Date'])
        df_dist = df_dist.sort_values('Date').reset_index(drop=True)
        
        # Calculate all temporal features needed by Experiment 8
        df_dist['Month'] = df_dist['Date'].dt.month
        import numpy as np
        df_dist['Month_Sin'] = np.sin(2 * np.pi * df_dist['Month'] / 12)
        df_dist['Month_Cos'] = np.cos(2 * np.pi * df_dist['Month'] / 12)
        
        for lag in [1, 2, 3, 6, 12]:
            df_dist[f'Groundwater_Mean_Lag_{lag}'] = df_dist['Groundwater_Mean'].shift(lag)
            df_dist[f'Rainfall_Lag_{lag}'] = df_dist['Rainfall'].shift(lag)
            df_dist[f'Temperature_Lag_{lag}'] = df_dist['Temperature'].shift(lag)
            
        for lag in [1, 3, 6]:
            df_dist[f'ONI_Lag_{lag}'] = df_dist['ONI'].shift(lag)
            df_dist[f'MEI_Lag_{lag}'] = df_dist['MEI'].shift(lag)
            df_dist[f'GRACE_TWS_Lag_{lag}'] = df_dist['GRACE_TWS'].shift(lag)
            
        for w in [3, 6]:
            df_dist[f'Groundwater_Mean_Rolling_{w}M'] = df_dist['Groundwater_Mean'].rolling(w, min_periods=1).mean()
            df_dist[f'Rainfall_Rolling_{w}M'] = df_dist['Rainfall'].rolling(w, min_periods=1).mean()
            
        df_dist['GRACE_TWS_Rolling_3M'] = df_dist['GRACE_TWS'].rolling(3, min_periods=1).mean()
        
        for p in [1, 3, 6]:
            df_dist[f'Groundwater_Mean_Change_{p}M'] = df_dist['Groundwater_Mean'] - df_dist['Groundwater_Mean'].shift(p)
            df_dist[f'GRACE_TWS_Change_{p}M'] = df_dist['GRACE_TWS'] - df_dist['GRACE_TWS'].shift(p)
            
        last_row = df_dist.iloc[-1].to_dict()
        features = {}
        for k, v in last_row.items():
            features[k] = 0.0 if pd.isna(v) else v
        
        # Calculate forecast date based on horizon
        obs_date = obs.date
        forecast_dt = obs_date + relativedelta(months=horizon)
        forecast_date_str = forecast_dt.strftime("%Y-%m-%d")
        
        res = predict_district(district, horizon, features, forecast_date=forecast_date_str)
        
        # Save to DB
        metadata = database_service.get_model_metadata_by_params('district', horizon)
        if metadata:
            database_service.save_forecast(
                district_id=dist_record.id,
                geographic_level="district",
                forecast_date=forecast_dt,
                horizon_months=horizon,
                predicted=res['predicted_groundwater_level_m_bgl'],
                current=res['current_groundwater_level_m_bgl'],
                change=res['predicted_change_m_bgl'],
                metadata_id=metadata.id
            )
            
        return res

prediction_service = PredictionService()
