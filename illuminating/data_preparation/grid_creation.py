import geopandas as gpd
import json
import pandas as pd
import os
from shapely.geometry import Polygon,LineString
from math import sqrt,cos
from jaxa.earth import je
import numpy as np
import pdb
import math
from concurrent.futures import ThreadPoolExecutor,as_completed
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from tqdm import tqdm  # For progress bars
import time
from illuminating.data_preparation.geodata_processing import download_gdf


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE = os.path.join(SCRIPT_DIR, 'checkpoint.json')
BATCH_SIZE = 200  # Number of grids to process in a batch
CRS = "EPSG:4326"



def save_checkpoint(i, j):
    checkpoint_data = {'i': i, 'j': j}
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint_data, f)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint_data = json.load(f)
        return checkpoint_data['i'], checkpoint_data['j']
    else:
        return 0, 0

def save_polygons_to_postgis(polygons, table_name, connection_string):
    gdf = gpd.GeoDataFrame(geometry=polygons, crs=CRS)
    engine = create_engine(connection_string)

    try:
        gdf.to_postgis(name=table_name, con=engine, if_exists='append', index=False)
        print(f"Batch of {len(polygons)} polygons saved to PostGIS table {table_name}")
    except SQLAlchemyError as e:
        print(f"Error saving to PostGIS: {e}")
        raise

def create_polygon(i, j, bbox, grid_size_km):
    grid_size_km = float(grid_size_km)

    lat_size = grid_size_km / 111
    lon_size = grid_size_km / (111 * np.cos(np.radians((bbox['min_lat'] + bbox['max_lat']) / 2)))

    min_lat = bbox['min_lat'] + i * lat_size
    max_lat = min_lat + lat_size
    min_lon = bbox['min_lon'] + j * lon_size
    max_lon = min_lon + lon_size

    return Polygon([(min_lon, min_lat), (min_lon, max_lat), (max_lon, max_lat), (max_lon, min_lat)])

def process_grid_batch(start_i, end_i, bbox, grid_size_km, table_name, connection_string, cols):
    polygons = []
    for i in range(start_i, end_i):
        save_checkpoint(i, 0)
        for j in range(cols):
            try:
                polygon = create_polygon(i, j, bbox, grid_size_km)
                polygons.append(polygon)
                #print(f"Processed row {i}, column {j}")
            except Exception as e:
                print(f"Error creating polygon at row {i}, column {j}: {e}")
                continue

    if polygons:
        save_polygons_to_postgis(polygons, table_name, connection_string)

def save_grid_with_threadpool(bbox, grid_size_km=1, table_name="your_table_name", connection_string="postgresql://username:password@localhost:5432/your_database", num_workers=4):
    bbox = {k: float(v) for k, v in bbox.items()}

    lat_diff = bbox['max_lat'] - bbox['min_lat']
    lon_diff = bbox['max_lon'] - bbox['min_lon']

    grid_size_km = float(grid_size_km)

    lat_size = grid_size_km / 111
    lon_size = grid_size_km / (111 * np.cos(np.radians((bbox['min_lat'] + bbox['max_lat']) / 2)))

    rows = int(np.ceil(lat_diff / lat_size))
    cols = int(np.ceil(lon_diff / lon_size))

    start_i, _ = load_checkpoint()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        for i in range(start_i, rows, BATCH_SIZE):
            end_i = min(i + BATCH_SIZE, rows)
            futures.append(
                executor.submit(process_grid_batch, i, end_i, bbox, grid_size_km, table_name, connection_string, cols)
            )
            print(i)


        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error during batch processing: {e}")
                save_checkpoint(i, 0)
                break

    os.remove(CHECKPOINT_FILE)

if __name__ == "__main__":
    file_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.abspath(os.path.join(file_path,"../.."))
    raw_data_path = os.path.join(root_path,"raw_data")
    filepath = os.path.join(raw_data_path,"USA_grid.geojson")
 #   grid_checkpoint = os.path.join(raw_data_path,"checkpoint.json")
    USA_gdf = download_gdf("GHA",1)
    # USA_gdf.drop(USA_gdf[USA_gdf.NAME_1=="Alaska"].index,inplace=True)
    # USA_gdf.drop(USA_gdf[USA_gdf.NAME_1=="Hawaii"].index,inplace=True)
    #USA_gdf = USA_gdf[USA_gdf["NAME_1"]=="Florida"]
    USA_bbox = {'min_lon': USA_gdf.total_bounds[0],
                'min_lat': USA_gdf.total_bounds[1],
                'max_lon': USA_gdf.total_bounds[2],
                'max_lat': USA_gdf.total_bounds[3]}
    print(USA_bbox)

    # USA_bbox = {'min_lat': 34.870060391099656,
    # 'max_lat': 35.05033826852978,
    # 'min_lon': -116.93259018663741,
    # 'max_lon': -116.71360981336258}

    save_grid_with_threadpool(USA_bbox, grid_size_km=5, table_name="Africa_five_km", connection_string="postgresql://postgres:lewagon@localhost:5432/geospatial_db",num_workers=1)

    # print(filepath)
