from fastapi import APIRouter
import json
import os

router = APIRouter()

@router.get("")
def get_metrics():
    # Load metrics from metadata.json dynamically
    maha_meta_path = 'models/production/maharashtra/metadata.json'
    dist_meta_path = 'models/production/district/metadata.json'
    
    metrics = {
        "Maharashtra": {},
        "District": {}
    }
    
    if os.path.exists(maha_meta_path):
        with open(maha_meta_path, 'r') as f:
            data = json.load(f)
            for h, meta in data.items():
                metrics["Maharashtra"][f"t+{meta['forecast_horizon'].split()[0]}"] = {
                    "model_type": meta["model_type"],
                    "test_R2": meta["test_R2"],
                    "test_RMSE": meta["test_RMSE"],
                    "experiment": meta["experiment"],
                    "geographic_level": meta["geographic_level"]
                }
                
    if os.path.exists(dist_meta_path):
        with open(dist_meta_path, 'r') as f:
            data = json.load(f)
            for h, meta in data.items():
                metrics["District"][f"t+{meta['forecast_horizon'].split()[0]}"] = {
                    "model_type": meta["model_type"],
                    "test_R2": meta["test_R2"],
                    "test_RMSE": meta["test_RMSE"],
                    "experiment": meta["experiment"],
                    "geographic_level": meta["geographic_level"]
                }
                
    return {
        "description": "TEST-SET Metrics from the final verified models",
        "metrics": metrics
    }
