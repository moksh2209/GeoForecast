from dataclasses import dataclass
from typing import Optional

@dataclass
class PredictionResponse:
    geographic_level: str
    forecast_date: str
    horizon_months: int
    predicted_groundwater_level_m_bgl: float
    current_groundwater_level_m_bgl: float
    predicted_change_m_bgl: float
    model: str
    model_version: str
    district: Optional[str] = None
    
    def model_dump(self):
        return self.__dict__
