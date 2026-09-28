import os
import sys
import pandas as pd
from datetime import datetime
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from backend.app.database import SessionLocal, engine
from backend.app.models import District, MonthlyObservation, ModelMetadata

def import_data():
    db = SessionLocal()
    
    print("Loading data...")
    # 1. Load the existing district CSV
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/district_spatiotemporal_monthly.csv'))
    df = pd.read_csv(csv_path)
    
    # 2. Insert supported districts
    print("Importing districts...")
    districts_inserted = 0
    unique_districts = df['District Name'].dropna().unique()
    
    district_map = {}
    for d_name in unique_districts:
        d_rec = db.query(District).filter(District.district_name == d_name).first()
        if not d_rec:
            # We don't have lat/lon in this CSV reliably without coords, but we just insert what we have
            # Let's check if the CSV has lat/lon
            lat = None
            lon = None
            if 'District_Centroid_Latitude' in df.columns:
                lat = df[df['District Name'] == d_name]['District_Centroid_Latitude'].iloc[0]
            if 'District_Centroid_Longitude' in df.columns:
                lon = df[df['District Name'] == d_name]['District_Centroid_Longitude'].iloc[0]
                
            d_rec = District(
                district_name=d_name,
                latitude=float(lat) if pd.notnull(lat) else None,
                longitude=float(lon) if pd.notnull(lon) else None,
                is_supported=True
            )
            db.add(d_rec)
            db.commit()
            db.refresh(d_rec)
            districts_inserted += 1
            
        district_map[d_name] = d_rec.id

    # 3. Insert monthly observations
    print("Importing monthly observations...")
    obs_inserted = 0
    skipped = 0
    
    # Avoid duplicate records by checking what's in DB
    existing_obs = set()
    for obs in db.query(MonthlyObservation.district_id, MonthlyObservation.date).all():
        existing_obs.add((obs.district_id, obs.date.strftime("%Y-%m-%d")))
        
    records = []
    
    # Get the date range
    df['Date'] = pd.to_datetime(df['Date'])
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    
    for _, row in df.iterrows():
        d_name = row['District Name']
        if pd.isna(d_name):
            skipped += 1
            continue
            
        d_id = district_map.get(d_name)
        dt = row['Date']
        dt_str = dt.strftime("%Y-%m-%d")
        
        if (d_id, dt_str) in existing_obs:
            skipped += 1
            continue
            
        obs = MonthlyObservation(
            district_id=d_id,
            date=dt,
            grace_tws=float(row['GRACE_TWS']) if pd.notnull(row.get('GRACE_TWS')) else None,
            rainfall=float(row['Rainfall']) if pd.notnull(row.get('Rainfall')) else None,
            temperature=float(row['Temperature']) if pd.notnull(row.get('Temperature')) else None,
            oni=float(row['ONI']) if pd.notnull(row.get('ONI')) else None,
            mei=float(row['MEI']) if pd.notnull(row.get('MEI')) else None,
            groundwater_mean=float(row['Groundwater_Mean']) if pd.notnull(row.get('Groundwater_Mean')) else None,
            groundwater_well_count=float(row['Groundwater_Well_Count']) if pd.notnull(row.get('Groundwater_Well_Count')) else None
        )
        records.append(obs)
        existing_obs.add((d_id, dt_str))
        obs_inserted += 1
        
        # Batch insert
        if len(records) >= 1000:
            db.bulk_save_objects(records)
            db.commit()
            records = []
            
    if records:
        db.bulk_save_objects(records)
        db.commit()

    # 4. Insert Model Metadata
    print("Importing model metadata...")
    meta_inserted = 0
    maha_meta_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/production/maharashtra/metadata.json'))
    dist_meta_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/production/district/metadata.json'))
    
    # helper
    def load_meta(path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                for h, m in data.items():
                    # check exists
                    exists = db.query(ModelMetadata).filter(
                        ModelMetadata.geographic_level == m['geographic_level'],
                        ModelMetadata.horizon_months == int(m['forecast_horizon'].split()[0])
                    ).first()
                    
                    if not exists:
                        meta_rec = ModelMetadata(
                            geographic_level=m['geographic_level'],
                            horizon_months=int(m['forecast_horizon'].split()[0]),
                            model_type=m['model_type'],
                            experiment=m['experiment'],
                            model_version=m['model_version'],
                            test_r2=m.get('test_R2'),
                            test_rmse=m.get('test_RMSE'),
                            test_samples=m.get('test_sample_count'),
                            artifact_path=m['artifact_source']
                        )
                        db.add(meta_rec)
                        db.commit()
                        nonlocal meta_inserted
                        meta_inserted += 1

    load_meta(maha_meta_path)
    load_meta(dist_meta_path)

    print("==================================================")
    print("IMPORT REPORT")
    print("==================================================")
    print(f"Districts inserted: {districts_inserted}")
    print(f"Observations inserted: {obs_inserted}")
    print(f"Date range: {min_date.date()} to {max_date.date()}")
    print(f"Skipped/invalid rows: {skipped}")
    print(f"Model metadata records inserted: {meta_inserted}")
    
    db.close()

if __name__ == "__main__":
    import_data()
