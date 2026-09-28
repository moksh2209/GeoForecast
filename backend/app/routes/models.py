from fastapi import APIRouter
from ..services.database_service import database_service

router = APIRouter()

@router.get("")
def get_models():
    models = database_service.get_model_metadata()
    return [
        {
            "geographic_level": m.geographic_level,
            "horizon_months": m.horizon_months,
            "model_type": m.model_type,
            "experiment": m.experiment,
            "model_version": m.model_version,
            "test_r2": m.test_r2,
            "test_rmse": m.test_rmse,
            "test_samples": m.test_samples
        } for m in models
    ]
