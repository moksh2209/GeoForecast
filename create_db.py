import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

conn = psycopg2.connect(user='postgres', password='1234', host='localhost', port='5432', database='postgres')
conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
cur = conn.cursor()
cur.execute("SELECT 1 FROM pg_database WHERE datname = 'geoforecast'")
exists = cur.fetchone()
if not exists:
    cur.execute('CREATE DATABASE geoforecast')
    print('DATABASE_CREATED')
else:
    print('DATABASE_ALREADY_EXISTS')
cur.close()
conn.close()
