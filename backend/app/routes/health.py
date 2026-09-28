from fastapi import APIRouter
from sqlalchemy.orm import Session
from ..database import engine

router = APIRouter()

@router.get("/health")
def get_health():
    db_status = "disconnected"
    try:
        with engine.connect() as conn:
            db_status = "connected"
    except Exception:
        pass
        
    return {
        "status": "ok",
        "service": "GeoForecast API",
        "database": db_status
    }
