import psycopg2

conn = psycopg2.connect(user='postgres', password='1234', host='localhost', port='5432', database='geoforecast')
cur = conn.cursor()

tables = ['districts', 'monthly_observations', 'model_metadata', 'forecasts']
for table in tables:
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    count = cur.fetchone()[0]
    print(f"{table}: {count}")

cur.close()
conn.close()
