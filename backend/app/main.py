import os
import sys

# Ensure src is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import health, districts, forecasts, metrics, models, forecast_history

app = FastAPI(title="GeoForecast API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(districts.router, prefix="/api/districts")
app.include_router(forecasts.router, prefix="/api/forecast")
app.include_router(forecast_history.router, prefix="/api/forecasts")
app.include_router(metrics.router, prefix="/api/metrics")
app.include_router(models.router, prefix="/api/models")
