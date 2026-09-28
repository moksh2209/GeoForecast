from datetime import datetime
from .model_registry import ModelRegistry
from .feature_builder import build_maharashtra_features, build_district_features
from .schemas import PredictionResponse

registry = ModelRegistry()

def predict_maharashtra(horizon, features, forecast_date=None):
    if forecast_date is None:
        forecast_date = datetime.now().strftime("%Y-%m-%d")
        
    model = registry.get_model('maharashtra', horizon)
    meta = registry.get_metadata('maharashtra', horizon)
    
    X = build_maharashtra_features(horizon, features)
    pred_val = model.predict(X)[0]
    
    current_val = features.get('Groundwater_Mean', 0)
    
    response = PredictionResponse(
        geographic_level="maharashtra",
        forecast_date=forecast_date,
        horizon_months=horizon,
        predicted_groundwater_level_m_bgl=float(pred_val),
        current_groundwater_level_m_bgl=float(current_val),
        predicted_change_m_bgl=float(pred_val - current_val),
        model=meta.get("model_type", "Unknown"),
        model_version=meta.get("model_version", "Unknown")
    )
    return response.model_dump()

def predict_district(district, horizon, features, forecast_date=None):
    if forecast_date is None:
        forecast_date = datetime.now().strftime("%Y-%m-%d")
        
    model = registry.get_model('district', horizon)
    meta = registry.get_metadata('district', horizon)
    
    X = build_district_features(district, meta['feature_list'], features)
    pred_val = model.predict(X)[0]
    
    current_val = features.get('Groundwater_Mean', 0)
    
    response = PredictionResponse(
        geographic_level="district",
        district=district,
        forecast_date=forecast_date,
        horizon_months=horizon,
        predicted_groundwater_level_m_bgl=float(pred_val),
        current_groundwater_level_m_bgl=float(current_val),
        predicted_change_m_bgl=float(pred_val - current_val),
        model=meta.get("model_type", "Unknown"),
        model_version=meta.get("model_version", "Unknown")
    )
    return response.model_dump()
