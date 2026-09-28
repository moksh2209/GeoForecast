from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from ..services.database_service import database_service

router = APIRouter()

@router.get("/{district}")
def get_forecast_history(district: str, horizon: Optional[int] = Query(None, description="Forecast horizon in months")):
    forecasts = database_service.get_forecasts(district, horizon)
    if forecasts is None:
        raise HTTPException(status_code=404, detail="District not found in database")
        
    return [
        {
            "forecast_generated_at": f.forecast_generated_at,
            "forecast_date": f.forecast_date,
            "horizon_months": f.horizon_months,
            "predicted_groundwater_level_m_bgl": f.predicted_groundwater_level_m_bgl,
            "current_groundwater_level_m_bgl": f.current_groundwater_level_m_bgl,
            "predicted_change_m_bgl": f.predicted_change_m_bgl,
            "model_metadata_id": f.model_metadata_id
        } for f in forecasts
    ]
