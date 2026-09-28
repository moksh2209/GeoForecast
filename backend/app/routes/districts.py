from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from datetime import date
from ..services.prediction_service import prediction_service
from ..services.database_service import database_service

router = APIRouter()

@router.get("")
def list_districts():
    return prediction_service.get_supported_districts()

@router.get("/{district}")
def get_district(district: str):
    supported = prediction_service.get_supported_districts()
    if district not in supported:
        raise HTTPException(status_code=404, detail="District not supported")
        
    return {
        "district": district,
        "supported": True
    }

@router.get("/{district}/history")
def get_district_history(
    district: str, 
    start_date: Optional[date] = None, 
    end_date: Optional[date] = None
):
    supported = prediction_service.get_supported_districts()
    if district not in supported:
        raise HTTPException(status_code=404, detail="District not supported")
        
    records = database_service.get_district_history(district, start_date, end_date)
    if records is None:
        raise HTTPException(status_code=404, detail="District not found in database")
        
    return {
        "district": district,
        "records": records
    }
