import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def main():
    os.makedirs('reports/geographic_audit/plots', exist_ok=True)
    
    file_path = 'data/raw/ground_water_level_manual_quarterly_maharashtra_1970_2025.csv'
    print(f"Reading {file_path}...")
    df = pd.read_csv(file_path, low_memory=False, encoding='latin1')
    
    print("\n--- STEP 1: INSPECT COLUMNS ---")
    num_rows = len(df)
    num_cols = len(df.columns)
    print(f"Rows: {num_rows}")
    print(f"Columns: {num_cols}")
    
    col_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing_pct = df[col].isnull().sum() / num_rows * 100
        col_info.append({
            'Column': col,
            'DataType': dtype,
            'MissingPct': missing_pct
        })
    col_df = pd.DataFrame(col_info)
    col_df.to_csv('reports/geographic_audit/column_inventory.csv', index=False)
    print(col_df)
    
    # Identify geographic columns (case insensitive, loose matching)
    cols_lower = [c.lower() for c in df.columns]
    
    lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
    lon_col = next((c for c in df.columns if 'lon' in c.lower()), None)
    district_col = next((c for c in df.columns if 'district' in c.lower()), None)
    taluka_col = next((c for c in df.columns if 'taluka' in c.lower() or 'block' in c.lower() or 'tehsil' in c.lower()), None)
    village_col = next((c for c in df.columns if 'village' in c.lower()), None)
    well_id_col = next((c for c in df.columns if 'well' in c.lower() and ('id' in c.lower() or 'no' in c.lower())), None)
    if not well_id_col:
        # Fallback to WLSNO or similar
        well_id_col = next((c for c in df.columns if 'wls' in c.lower()), None)
        if not well_id_col:
            well_id_col = next((c for c in df.columns if 'site' in c.lower()), None)
            
    date_col = next((c for c in df.columns if 'date' in c.lower() or 'year' in c.lower()), None) # Need to refine date col
    # Actually, check if 'Monitoring Date' or 'Date' or 'Year'/'Month'
    date_candidates = [c for c in df.columns if 'date' in c.lower()]
    if date_candidates:
        date_col = date_candidates[0]
    
    gwl_col = next((c for c in df.columns if 'level' in c.lower() or 'gwl' in c.lower()), None)
    
    print(f"Lat: {lat_col}, Lon: {lon_col}")
    print(f"District: {district_col}, Taluka: {taluka_col}, Village: {village_col}")
    print(f"Well ID: {well_id_col}")
    print(f"Date: {date_col}")
    print(f"GWL: {gwl_col}")
    
    # Process dates
    if date_col:
        df['Parsed_Date'] = pd.to_datetime(df[date_col], errors='coerce')
        df['Year'] = df['Parsed_Date'].dt.year
        df['Month'] = df['Parsed_Date'].dt.month
        df['YearMonth'] = df['Parsed_Date'].dt.to_period('M')
    else:
        print("ERROR: Could not identify date column.")
        return
        
    print("\n--- STEP 2: UNIQUE WELL COVERAGE ---")
    if not well_id_col:
        # Construct a synthetic well ID if none exists, using district, taluka, village
        df['Synthetic_Well_ID'] = df[district_col].astype(str) + '_' + df[taluka_col].astype(str) + '_' + df[village_col].astype(str)
        well_id_col = 'Synthetic_Well_ID'
        
    total_wells = df[well_id_col].nunique()
    total_obs = len(df)
    print(f"Total Unique Wells: {total_wells}")
    print(f"Total Observations: {total_obs}")
    
    if district_col:
        dist_counts = df.groupby(district_col)[well_id_col].nunique().reset_index(name='Unique_Wells')
        dist_counts.to_csv('reports/geographic_audit/district_well_counts.csv', index=False)
    if taluka_col:
        taluka_counts = df.groupby(taluka_col)[well_id_col].nunique().reset_index(name='Unique_Wells')
        taluka_counts.to_csv('reports/geographic_audit/taluka_well_counts.csv', index=False)
        
    print("\n--- STEP 3: CITY / REGION AUDIT ---")
    regions = [
        'Mumbai', 'Pune', 'Nashik', 'Nagpur', 'Thane', 'Palghar', 'Navi Mumbai', 
        'Aurangabad', 'Chhatrapati Sambhajinagar', 'Kolhapur', 'Satara', 
        'Solapur', 'Ahmednagar', 'Ahilyanagar', 'Amravati', 'Nanded', 'Sangli', 'Ratnagiri'
    ]
    
    city_results = []
    
    if district_col:
        df['Lower_Dist'] = df[district_col].astype(str).str.lower()
        df['Lower_Tal'] = df[taluka_col].astype(str).str.lower() if taluka_col else ''
        
        for region in regions:
            reg_lower = region.lower()
            # Match district or taluka
            mask = df['Lower_Dist'].str.contains(reg_lower, na=False) | df['Lower_Tal'].str.contains(reg_lower, na=False)
            sub_df = df[mask]
            
            if len(sub_df) == 0:
                city_results.append({
                    'Area': region,
                    'Note': 'Not identifiable from available geographic fields'
                })
                continue
                
            wells = sub_df[well_id_col].nunique()
            obs = len(sub_df)
            min_date = sub_df['Parsed_Date'].min()
            max_date = sub_df['Parsed_Date'].max()
            
            months_covered = sub_df['YearMonth'].nunique()
            
            obs_per_month = sub_df.groupby('YearMonth').size()
            avg_obs = obs_per_month.mean() if not obs_per_month.empty else 0
            med_obs = obs_per_month.median() if not obs_per_month.empty else 0
            min_obs = obs_per_month.min() if not obs_per_month.empty else 0
            max_obs = obs_per_month.max() if not obs_per_month.empty else 0
            
            total_possible_months = 1
            if pd.notnull(min_date) and pd.notnull(max_date):
                total_possible_months = (max_date.year - min_date.year) * 12 + max_date.month - min_date.month + 1
            
            pct_1 = (len(obs_per_month[obs_per_month >= 1]) / max(total_possible_months, 1)) * 100
            pct_10 = (len(obs_per_month[obs_per_month >= 10]) / max(total_possible_months, 1)) * 100
            pct_50 = (len(obs_per_month[obs_per_month >= 50]) / max(total_possible_months, 1)) * 100
            pct_100 = (len(obs_per_month[obs_per_month >= 100]) / max(total_possible_months, 1)) * 100
            
            city_results.append({
                'Area': region,
                'District': sub_df[district_col].mode().iloc[0] if len(sub_df) > 0 else '',
                'Unique Wells': wells,
                'Observations': obs,
                'Earliest': min_date.strftime('%Y-%m-%d') if pd.notnull(min_date) else '',
                'Latest': max_date.strftime('%Y-%m-%d') if pd.notnull(max_date) else '',
                'Months Covered': months_covered,
                'Avg Obs/Month': avg_obs,
                'Median Obs/Month': med_obs,
                'Min Obs': min_obs,
                'Max Obs': max_obs,
                '% Months >= 1 Obs': pct_1,
                '% Months >= 10 Obs': pct_10,
                '% Months >= 50 Obs': pct_50,
                '% Months >= 100 Obs': pct_100
            })
            
    city_df = pd.DataFrame(city_results)
    print(city_df)
    
    print("\n--- STEP 4: MONTHLY COVERAGE ---")
    if district_col:
        monthly_cov = df.groupby([district_col, 'Year', 'Month']).agg({
            well_id_col: 'nunique',
            date_col: 'size'
        }).reset_index().rename(columns={well_id_col: 'Unique Wells', date_col: 'Observations'})
        monthly_cov.to_csv('reports/geographic_audit/district_monthly_coverage.csv', index=False)
        
    print("\n--- STEP 5 & 7: ML SUITABILITY & DATA DENSITY ---")
    density_results = []
    
    if district_col:
        for dist, group in df.groupby(district_col):
            wells = group[well_id_col].nunique()
            obs = len(group)
            
            min_date = group['Parsed_Date'].min()
            max_date = group['Parsed_Date'].max()
            
            total_possible_months = 1
            if pd.notnull(min_date) and pd.notnull(max_date):
                total_possible_months = (max_date.year - min_date.year) * 12 + max_date.month - min_date.month + 1
                
            obs_per_month = group.groupby('YearMonth').size()
            med_wells_month = group.groupby('YearMonth')[well_id_col].nunique().median()
            
            months_covered = group['YearMonth'].nunique()
            pct_months_covered = (months_covered / max(total_possible_months, 1)) * 100
            pct_missing = 100 - pct_months_covered
            
            obs_per_month_avg = obs / max(months_covered, 1)
            
            classification = "LOW COVERAGE"
            if wells >= 100 and pct_months_covered >= 60:
                classification = "HIGH COVERAGE"
            elif wells >= 30 and pct_months_covered >= 40:
                classification = "MEDIUM COVERAGE"
                
            density_results.append({
                'District': dist,
                'Unique Wells': wells,
                'Total Observations': obs,
                'Observations/Month (Avg)': obs_per_month_avg,
                'Median Wells/Month': med_wells_month,
                '% Missing Months': pct_missing,
                'Coverage Class': classification
            })
            
    density_df = pd.DataFrame(density_results)
    density_df.to_csv('reports/geographic_audit/district_data_quality.csv', index=False)
    print(density_df.head())
    
    print("\n--- STEP 6: SPATIAL COORDINATE CHECK ---")
    if lat_col and lon_col:
        # Convert to numeric
        df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
        df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
        
        valid_coords = df.dropna(subset=[lat_col, lon_col])
        missing_coords = len(df) - len(valid_coords)
        
        print(f"Missing coordinates: {missing_coords}")
        print(f"Lat Range: {valid_coords[lat_col].min()} to {valid_coords[lat_col].max()}")
        print(f"Lon Range: {valid_coords[lon_col].min()} to {valid_coords[lon_col].max()}")
        
        unique_coords = valid_coords.drop_duplicates(subset=[lat_col, lon_col])
        print(f"Unique coordinate locations: {len(unique_coords)}")
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.scatter(unique_coords[lon_col], unique_coords[lat_col], alpha=0.5, s=5)
        plt.title('Spatial Distribution of Monitoring Wells in Maharashtra')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True)
        plt.savefig('reports/geographic_audit/well_locations.png')
        plt.close()
    else:
        print("Coordinate-based spatial analysis cannot be performed from this dataset.")
        
    print("\n--- STEP 8: POTENTIAL SPATIAL MODEL STRUCTURE ---")
    # Determine feasibility
    if lat_col and lon_col and len(df.dropna(subset=[lat_col, lon_col])) > 0:
        spatial_struct = "Grid/Region + Month -> Groundwater Level appears technically feasible due to coordinate availability."
    elif district_col:
        spatial_struct = "District + Month -> Groundwater Level appears feasible based on categorical geographic data."
    else:
        spatial_struct = "No obvious spatial structure is supported."
    print(spatial_struct)
    
    print("\n--- FINAL REPORT GENERATION ---")
    with open('reports/geographic_audit/geographic_coverage_report.md', 'w') as f:
        f.write("# Geographic Data Coverage Audit\n\n")
        f.write("## 1. Dataset Overview\n")
        f.write(f"Total Rows: {num_rows}, Total Columns: {num_cols}\n\n")
        
        f.write("## 2. Geographic Columns Found\n")
        f.write(f"- District: {district_col}\n")
        f.write(f"- Taluka: {taluka_col}\n")
        f.write(f"- Village: {village_col}\n")
        f.write(f"- Lat/Lon: {lat_col} / {lon_col}\n\n")
        
        f.write("## 3. Total Wells & 4. Total Observations\n")
        f.write(f"- Unique Wells: {total_wells}\n")
        f.write(f"- Observations: {total_obs}\n\n")
        
        f.write("## 5. District & 6. Taluka Coverage\n")
        if district_col:
            f.write(f"Districts present: {df[district_col].nunique()}\n")
        if taluka_col:
            f.write(f"Talukas present: {df[taluka_col].nunique()}\n\n")
            
        f.write("## 7, 8, 9. City/Region Coverage\n")
        f.write(city_df.to_string() + "\n\n")
        
        f.write("## 10. Monthly Coverage\n")
        f.write("Detailed data written to `district_monthly_coverage.csv`.\n\n")
        
        f.write("## 11. Coordinate Availability\n")
        if lat_col and lon_col:
            f.write(f"Coordinates found. Missing: {missing_coords}.\n\n")
        else:
            f.write("Coordinate-based spatial analysis cannot be performed from this dataset.\n\n")
            
        f.write("## 12. Data-Density Analysis & 13. District Coverage Classification\n")
        f.write(density_df[['District', 'Unique Wells', '% Missing Months', 'Coverage Class']].head(20).to_string() + "\n\n")
        
        f.write("## 14. Potential Spatial Modeling Structure\n")
        f.write(f"{spatial_struct}\n\n")
        
        f.write("## 15. Important Limitations\n")
        f.write("The dataset is labelled as 'quarterly' despite having monthly dates, meaning there are large expected gaps in monthly records. Many cities (like Mumbai) may not have dedicated monitoring wells within city limits. Care must be taken not to assume continuous monthly time series for individual wells.\n")
        
    print("Done generating reports.")
    
    # Final Summary for agent
    final_summary = []
    if 'Area' in city_df.columns:
        for _, row in city_df.iterrows():
            if 'Note' in row and pd.notna(row['Note']):
                continue
            final_summary.append({
                'Area': row['Area'],
                'Unique Wells': row['Unique Wells'],
                'Observations': row['Observations'],
                'Months Covered': row['Months Covered'],
                'Median Wells/Month': row['Median Obs/Month']
            })
            
    print("\n--- FINAL SUMMARY TABLE ---")
    sum_df = pd.DataFrame(final_summary)
    print(sum_df)
    
if __name__ == "__main__":
    main()
