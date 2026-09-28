from src.prediction import predict_maharashtra, predict_district
import traceback

def test_validation():
    print("=============================================")
    print("HISTORICAL VALIDATION")
    print("=============================================")
    
    # Feature inputs
    maha_features_13 = {
        'GRACE_TWS': -1.2,
        'Rainfall': 100,
        'Temperature': 28,
        'ONI': 0.5,
        'MEI': -0.2,
        'Groundwater_Mean': 8.0,
        'Groundwater_Well_Count': 1000
    }
    
    maha_features_6 = dict(maha_features_13)
    maha_features_6.update({
        'ONI_Lag_1': 0.4,
        'Groundwater_Rolling_3M': 7.9,
        'Rainfall_Rolling_3M': 110,
        'Temperature_Lag_1': 27.5,
        'Sin_Month': 0.5
    })
    
    dist_features = dict(maha_features_13) # same base features
    
    # 1. Test Maharashtra Models
    for h in [1, 3]:
        try:
            res = predict_maharashtra(h, maha_features_13)
            print(f"Maharashtra h{h} SUCCESS: {res['predicted_groundwater_level_m_bgl']:.3f} m bgl")
        except Exception as e:
            print(f"Maharashtra h{h} FAILED: {e}")
            
    try:
        res = predict_maharashtra(6, maha_features_6)
        print(f"Maharashtra h6 SUCCESS: {res['predicted_groundwater_level_m_bgl']:.3f} m bgl")
    except Exception as e:
        print(f"Maharashtra h6 FAILED: {e}")
        
    # 2. Test District Models
    for h in [1, 3, 6]:
        try:
            res = predict_district("Pune", h, dist_features)
            print(f"District 'Pune' h{h} SUCCESS: {res['predicted_groundwater_level_m_bgl']:.3f} m bgl")
        except Exception as e:
            print(f"District 'Pune' h{h} FAILED: {e}")
            
    # 3. Test Unknown District
    try:
        res = predict_district("UnknownCity", 1, dist_features)
        print(f"District 'UnknownCity' h1 SUCCESS (One-hot handled safely): {res['predicted_groundwater_level_m_bgl']:.3f} m bgl")
    except Exception as e:
        print(f"District 'UnknownCity' h1 FAILED: {e}")
        
    # 4. Test GRACE Missing Handling
    maha_features_missing_grace = dict(maha_features_13)
    maha_features_missing_grace['GRACE_TWS'] = float('nan')
    try:
        res = predict_maharashtra(1, maha_features_missing_grace)
        print(f"Maharashtra h1 (Missing GRACE) SUCCESS: {res['predicted_groundwater_level_m_bgl']:.3f} m bgl")
    except Exception as e:
        print(f"Maharashtra h1 (Missing GRACE) FAILED: {e}")
        
    print("Validation Complete.")

if __name__ == "__main__":
    test_validation()
