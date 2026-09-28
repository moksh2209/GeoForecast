import psycopg2
import pandas as pd

conn = psycopg2.connect(user='postgres', password='1234', host='localhost', port='5432', database='geoforecast')

def query(q):
    return pd.read_sql(q, conn)

print("Districts count:", query("SELECT COUNT(*) FROM districts").iloc[0, 0])
print("Observations count:", query("SELECT COUNT(*) FROM monthly_observations").iloc[0, 0])
print("Metadata count:", query("SELECT COUNT(*) FROM model_metadata").iloc[0, 0])
print("Forecast count:", query("SELECT COUNT(*) FROM forecasts").iloc[0, 0])

obs = query("SELECT MIN(date) as min_date, MAX(date) as max_date FROM monthly_observations")
print("Date range:", obs['min_date'][0], "to", obs['max_date'][0])

for d in ['Pune', 'Nashik', 'Nagpur']:
    res = query(f"SELECT d.district_name, COUNT(o.id) as count, MIN(o.date) as min_date, MAX(o.date) as max_date FROM districts d LEFT JOIN monthly_observations o ON d.id = o.district_id WHERE d.district_name = '{d}' GROUP BY d.district_name")
    if not res.empty:
        print(f"{d}: {res['count'][0]} records, {res['min_date'][0]} to {res['max_date'][0]}")

print("Checking duplicates (district_id, date):", query("SELECT district_id, date, COUNT(*) FROM monthly_observations GROUP BY district_id, date HAVING COUNT(*) > 1").shape[0])
print("Orphan observations:", query("SELECT COUNT(*) FROM monthly_observations WHERE district_id IS NULL OR district_id NOT IN (SELECT id FROM districts)").iloc[0,0])

conn.close()
