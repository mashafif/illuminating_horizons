from sqlalchemy import create_engine
import pandas as pd
from google.cloud import bigquery
from tqdm import tqdm
import numpy as np


# PostgreSQL connection details
pg_host = 'localhost'
pg_database = 'geospatial_db'
pg_user = 'postgres'
pg_password = 'lewagon'
pg_port = '5432'
table = "powerline"

# Create PostgreSQL engine
postgresql_engine = create_engine(f'postgresql+psycopg2://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}')

# Query to get the geometry data as WKT
query = "SELECT ST_AsText(geometry) AS geometry FROM road"

# Load the data into a pandas DataFrame
df = pd.read_sql(query, postgresql_engine)
# Google Cloud BigQuery details
bq_project = 'wagon-bootcamp-429305'
bq_dataset = 'illuminating_horizons'
bq_table = 'powerline'

# Initialize BigQuery client
client = bigquery.Client(project=bq_project)

# Define BigQuery table schema
job_config = bigquery.LoadJobConfig(
    schema=[
        bigquery.SchemaField("geometry", "GEOGRAPHY"),
    ],
    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,  # Overwrite table if exists
)

# Load the data into BigQuery as a GEOGRAPHY column
table_id = f"{bq_project}.{bq_dataset}.{bq_table}"
load_job = client.load_table_from_dataframe(df, table_id, job_config=job_config)

# Wait for the job to complete
load_job.result()

print(f"Loaded {load_job.output_rows} rows into {table_id}.")
