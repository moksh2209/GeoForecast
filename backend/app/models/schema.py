from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from datetime import datetime
from ..database import Base

class District(Base):
    __tablename__ = 'districts'

    id = Column(Integer, primary_key=True, index=True)
    district_name = Column(String, unique=True, index=True, nullable=False)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    is_supported = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    observations = relationship("MonthlyObservation", back_populates="district")
    forecasts = relationship("Forecast", back_populates="district")

class MonthlyObservation(Base):
    __tablename__ = 'monthly_observations'

    id = Column(Integer, primary_key=True, index=True)
    district_id = Column(Integer, ForeignKey('districts.id'), nullable=False)
    date = Column(DateTime, nullable=False, index=True)
    grace_tws = Column(Float, nullable=True)
    rainfall = Column(Float, nullable=True)
    temperature = Column(Float, nullable=True)
    oni = Column(Float, nullable=True)
    mei = Column(Float, nullable=True)
    groundwater_mean = Column(Float, nullable=True)
    groundwater_well_count = Column(Float, nullable=True)
    groundwater_median = Column(Float, nullable=True)
    groundwater_std = Column(Float, nullable=True)
    groundwater_min = Column(Float, nullable=True)
    groundwater_max = Column(Float, nullable=True)
    observation_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    district = relationship("District", back_populates="observations")

    __table_args__ = (
        UniqueConstraint('district_id', 'date', name='uq_monthly_observation_district_date'),
    )

class ModelMetadata(Base):
    __tablename__ = 'model_metadata'

    id = Column(Integer, primary_key=True, index=True)
    geographic_level = Column(String, nullable=False)
    horizon_months = Column(Integer, nullable=False)
    model_type = Column(String, nullable=False)
    experiment = Column(Integer, nullable=False)
    model_version = Column(String, nullable=False)
    test_r2 = Column(Float, nullable=True)
    test_rmse = Column(Float, nullable=True)
    test_samples = Column(Integer, nullable=True)
    artifact_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    forecasts = relationship("Forecast", back_populates="model_metadata")

class Forecast(Base):
    __tablename__ = 'forecasts'

    id = Column(Integer, primary_key=True, index=True)
    district_id = Column(Integer, ForeignKey('districts.id'), nullable=True)
    geographic_level = Column(String, nullable=False)
    forecast_generated_at = Column(DateTime, default=datetime.utcnow)
    forecast_date = Column(DateTime, nullable=False)
    horizon_months = Column(Integer, nullable=False)
    predicted_groundwater_level_m_bgl = Column(Float, nullable=False)
    current_groundwater_level_m_bgl = Column(Float, nullable=False)
    predicted_change_m_bgl = Column(Float, nullable=False)
    model_metadata_id = Column(Integer, ForeignKey('model_metadata.id'), nullable=False)

    district = relationship("District", back_populates="forecasts")
    model_metadata = relationship("ModelMetadata", back_populates="forecasts")
