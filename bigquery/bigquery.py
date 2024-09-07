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

# Create PostgreSQL engine
postgresql_engine = create_engine(f'postgresql+psycopg2://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}')

# Query to get the geometry data as WKT
query = "SELECT ST_AsText(geometry) AS geometry FROM road"

# Fetch data from PostgreSQL in chunks and display progress
chunk_size = 10000  # Adjust as needed
chunks = []
with tqdm(total=np.inf, desc="Importing from PostgreSQL") as pbar:
    for chunk in pd.read_sql(query, postgresql_engine, chunksize=chunk_size):
        chunks.append(chunk)
        pbar.update(chunk_size)  # Update progress bar with chunk size
    df = pd.concat(chunks)

# Google Cloud BigQuery details
bq_project = 'wagon-bootcamp-429305'
bq_dataset = 'illuminating_horizons'
bq_table = 'road'

# Initialize BigQuery client
client = bigquery.Client(project=bq_project)

# Define BigQuery table schema
job_config = bigquery.LoadJobConfig(
    schema=[
        bigquery.SchemaField("geometry", "GEOGRAPHY"),
    ],
    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,  # Overwrite table if exists
)

# Load the data into BigQuery in chunks and display progress
def load_dataframe_to_bigquery(df, table_id, job_config):
    num_chunks = (len(df) // chunk_size) + 1
    with tqdm(total=num_chunks, desc="Uploading to BigQuery") as pbar:
        for start in range(0, len(df), chunk_size):
            end = min(start + chunk_size, len(df))
            chunk_df = df.iloc[start:end]
            load_job = client.load_table_from_dataframe(chunk_df, table_id, job_config=job_config)
            load_job.result()  # Wait for the job to complete
            pbar.update(1)  # Update progress bar after each chunk

# Define the BigQuery table ID
table_id = f"{bq_project}.{bq_dataset}.{bq_table}"

# Call the function to load the DataFrame into BigQuery
load_dataframe_to_bigquery(df, table_id, job_config)

print(f"Data successfully loaded into {table_id}.")
