import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    df = pd.read_csv('data/processed/master_monthly.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    os.makedirs('reports/plots', exist_ok=True)
    
    # 1. Groundwater level over time
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['Groundwater_Mean'], label='Groundwater Mean')
    plt.ylabel('Water Level (m bgl)')
    plt.title('Groundwater level over time')
    # m bgl usually means lower is deeper, but we just plot the values. If user wants inverted we can later
    plt.savefig('reports/plots/groundwater_level.png')
    plt.close()
    
    # 2. GRACE TWS over time
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['GRACE_TWS'], color='orange')
    plt.ylabel('LWE Thickness (cm)')
    plt.title('GRACE TWS over time')
    plt.savefig('reports/plots/grace_tws.png')
    plt.close()
    
    # 3. Rainfall over time
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['Rainfall'], color='blue')
    plt.ylabel('Rainfall (mm)')
    plt.title('Rainfall over time')
    plt.savefig('reports/plots/rainfall.png')
    plt.close()
    
    # 4. Temperature over time
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['Temperature'], color='red')
    plt.ylabel('Temperature (Celsius)')
    plt.title('Temperature over time')
    plt.savefig('reports/plots/temperature.png')
    plt.close()
    
    # 5. ONI over time
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['ONI'], color='purple')
    plt.ylabel('ONI')
    plt.title('ONI over time')
    plt.savefig('reports/plots/oni.png')
    plt.close()
    
    # 6. MEI over time
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['MEI'], color='brown')
    plt.ylabel('MEI')
    plt.title('MEI over time')
    plt.savefig('reports/plots/mei.png')
    plt.close()
    
    # 7. Groundwater well-count coverage
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['Groundwater_Well_Count'], color='green')
    plt.ylabel('Well Count')
    plt.title('Groundwater well-count coverage')
    plt.savefig('reports/plots/well_count.png')
    plt.close()
    
    print("Created plots in reports/plots/")

if __name__ == "__main__":
    main()
