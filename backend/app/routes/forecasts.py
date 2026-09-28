from fastapi import APIRouter, HTTPException, Query
from ..services.prediction_service import prediction_service

router = APIRouter()

@router.get("/maharashtra")
def get_forecast_maharashtra(horizon: int = Query(..., description="Forecast horizon in months (1, 3, 6)")):
    try:
        res = prediction_service.forecast_maharashtra(horizon)
        # Adding note about demo data
        res['note'] = "Using historical input data for demonstration"
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal prediction failure")

@router.get("/{district}")
def get_forecast_district(district: str, horizon: int = Query(..., description="Forecast horizon in months (1, 3, 6)")):
    try:
        res = prediction_service.forecast_district(district, horizon)
        res['note'] = "Using historical input data for demonstration"
        return res
    except ValueError as e:
        # Invalid horizon or district
        if "supported" in str(e):
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal prediction failure")
