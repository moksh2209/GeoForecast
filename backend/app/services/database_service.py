from sqlalchemy.orm import Session
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from backend.app.database import SessionLocal
from backend.app.models import District, MonthlyObservation, ModelMetadata, Forecast

class DatabaseService:
    def __init__(self):
        pass

    def get_supported_districts(self):
        with SessionLocal() as db:
            districts = db.query(District).filter(District.is_supported == True).all()
            return [d.district_name for d in districts]
            
    def get_district(self, district_name: str):
        with SessionLocal() as db:
            return db.query(District).filter(District.district_name == district_name).first()
            
    def get_district_history(self, district_name: str, start_date=None, end_date=None):
        with SessionLocal() as db:
            district = db.query(District).filter(District.district_name == district_name).first()
            if not district:
                return None
                
            query = db.query(MonthlyObservation).filter(MonthlyObservation.district_id == district.id)
            if start_date:
                query = query.filter(MonthlyObservation.date >= start_date)
            if end_date:
                query = query.filter(MonthlyObservation.date <= end_date)
                
            observations = query.order_by(MonthlyObservation.date.asc()).all()
            
            records = []
            for obs in observations:
                records.append({
                    "date": obs.date.strftime("%Y-%m-%d"),
                    "groundwater_level_m_bgl": obs.groundwater_mean,
                    "rainfall_mm": obs.rainfall,
                    "temperature_c": obs.temperature,
                    "grace_tws": obs.grace_tws,
                    "oni": obs.oni,
                    "mei": obs.mei
                })
            return records
            
    def get_latest_district_observation(self, district_name: str):
        with SessionLocal() as db:
            district = db.query(District).filter(District.district_name == district_name).first()
            if not district:
                return None
            return db.query(MonthlyObservation).filter(MonthlyObservation.district_id == district.id).order_by(MonthlyObservation.date.desc()).first()

    def get_model_metadata(self):
        with SessionLocal() as db:
            return db.query(ModelMetadata).all()
            
    def get_model_metadata_by_params(self, geo_level: str, horizon: int):
        with SessionLocal() as db:
            return db.query(ModelMetadata).filter(
                ModelMetadata.geographic_level == geo_level,
                ModelMetadata.horizon_months == horizon
            ).first()

    def save_forecast(self, district_id, geographic_level, forecast_date, horizon_months, 
                      predicted, current, change, metadata_id):
        with SessionLocal() as db:
            # Upsert logic based on district_id, forecast_date, horizon_months, metadata_id
            query = db.query(Forecast).filter(
                Forecast.forecast_date == forecast_date,
                Forecast.horizon_months == horizon_months,
                Forecast.model_metadata_id == metadata_id
            )
            if district_id:
                query = query.filter(Forecast.district_id == district_id)
            else:
                query = query.filter(Forecast.district_id == None)
                
            existing = query.first()
            
            if existing:
                existing.predicted_groundwater_level_m_bgl = predicted
                existing.current_groundwater_level_m_bgl = current
                existing.predicted_change_m_bgl = change
                db.commit()
                return existing
            else:
                forecast = Forecast(
                    district_id=district_id,
                    geographic_level=geographic_level,
                    forecast_date=forecast_date,
                    horizon_months=horizon_months,
                    predicted_groundwater_level_m_bgl=predicted,
                    current_groundwater_level_m_bgl=current,
                    predicted_change_m_bgl=change,
                    model_metadata_id=metadata_id
                )
                db.add(forecast)
                db.commit()
                db.refresh(forecast)
                return forecast
                
    def get_forecasts(self, district_name: str, horizon: int = None):
        with SessionLocal() as db:
            district = db.query(District).filter(District.district_name == district_name).first()
            if not district:
                return None
            query = db.query(Forecast).filter(Forecast.district_id == district.id)
            if horizon:
                query = query.filter(Forecast.horizon_months == horizon)
            return query.order_by(Forecast.forecast_date.desc()).all()

database_service = DatabaseService()
