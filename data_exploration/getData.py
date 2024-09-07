from state_dict import *
import glob
import xarray as xr
import geopandas as gpd
import os
import sys
from shapely.geometry import mapping
from rasterio import features
import numpy as np
from rasterio.transform import from_bounds
import pandas as pd
import rioxarray
import pyproj
import rasterio
from rasterio.warp import reproject, Resampling
from shapely.geometry import LineString
from rasterio.warp import calculate_default_transform, reproject, Resampling

# Local
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)
from illuminating.data_preparation.geodata_processing_kaoru import download_gdf

#Parameters

bearer = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6Im1hc2hoYWZpZiIsImV4cCI6MTcyNjAxNDQwMSwiaWF0IjoxNzIwODMwNDAxLCJpc3MiOiJFYXJ0aGRhdGEgTG9naW4ifQ.MffkJ_59FpDWDnES4xntvyRcSTfeVQDPJUwJnIbnk74zy9vbZA6iFz_GmAI5KdNwoVWKcKH_tfP0Byk63X2RwN89RUwShTTKvL86IH4hcc6ZHI3GpheS9M-Pi7_0BphHyDU3Aay7yIjGsd6LupfcGvMalnlnSU_cXCEOoZ_qfm19AQ0b37PwUQGci7snyz1pnb8NlDx-PZpL2-jLyPrndS9X-XDEEH_j2z5rh2nm-LXrq6IScZuAsn4_eD684CQVprT1VqngZOAAZTwL0yWhKVM7VwTr5wH0MXr2s5YKfJ1PnwqVFek0Vvrka4CjoJPZmKQkqAt-UHJVUgeZJM37pg"
PRODUCT = 'VNP46A4'  # Black Marble Annual Nighttime Lights with Cloud-Free Coverage
YEAR = 2023

def getStateNightData(state, year=2023, force_dl = False):
    from blackmarble.raster import bm_raster
    iso = state_to_iso[state]
    print(f'{state}({iso})')
    if len(state)!=2:
        state=state_to_iso[state]
    state_name = iso_to_state[state]
    #Check if state is in local
    print(f'../raw_data/bm/blackmarble_{year}_US-{iso}.nc')
    F = glob.glob(f'../raw_data/bm/blackmarble_{year}_US-{iso}.nc')
    if len(F) == 1 and not force_dl:
        print(f'Found local data for {state_name}, reading to memory')
        D = xr.open_dataset(F[0])
    else:
        if force_dl:
            print(f'force_dl is True, force downloading data for {state_name}')
        else:
            USA = download_gdf(country='USA',resolution=1)
            shape = USA[USA.NAME_1==state].geometry
            # Get the bounding box of the shapefile
            gdf = gpd.GeoDataFrame({'geometry': shape}, crs="EPSG:4326")
            print(f'No local data for {state_name}, downloading...')
            D = bm_raster(
            gdf,
            product_id="VNP46A4",
            date_range=pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="YS"),
            bearer=bearer
            )
    return D

def getStateNightData_global(country, year=2023, force_dl = False):
    from blackmarble.raster import bm_raster
    from state_dict import alpha3_to_alpha2
    if len(country) == 3: country = alpha3_to_alpha2[country]
    #print(f'{state}({iso})')
    #if len(state)!=2:
    #    state=state_to_iso[state]
    #state_name = iso_to_state[state]
    #Check if state is in local
    print(f'../raw_data/bm/blackmarble_{year}_{country}.nc')
    F = glob.glob(f'../raw_data/bm/blackmarble_{year}_{country}.nc')
    if len(F) == 1 and not force_dl:
        print(f'Found local data for {country}, reading to memory')
        D = xr.open_dataset(F[0])
    else:
        if force_dl:
            print(f'force_dl is True, force downloading data for {country}')
        else:
            COUNTRY = download_gdf(country=country,resolution=1)
            shape = COUNTRY.geometry
            # Get the bounding box of the shapefile
            gdf = gpd.GeoDataFrame({'geometry': shape}, crs="EPSG:4326")
            print(f'No local data for {country}, downloading...')
            D = bm_raster(
            gdf,
            product_id="VNP46A4",
            date_range=pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="YS"),
            bearer=bearer
            )
        D.to_netcdf(f'../raw_data/bm/blackmarble_{year}_{country}.nc')
    return D

def raster_to_xr(file_path):
    # Open the raster file
    with rasterio.open(file_path) as dataset:
        # Read the first band (assuming population data is in the first band)
        population_data = dataset.read(1)
        # Get the affine transformation to convert from pixel coordinates to spatial coordinates
        transform = dataset.transform
        # Get the coordinate arrays (x, y)
        rows, cols = np.indices(population_data.shape)
        xs, ys = rasterio.transform.xy(transform, rows, cols)
        # Convert to 1D arrays
        xs = np.array(xs)[0, :]  # Take one row since xs are the same across columns
        ys = np.array(ys)[:, 0]  # Take one column since ys are the same across rows
    # Create an xarray.Dataset
        XR = xr.Dataset(
        {
            "population": (["y", "x"], population_data)
        },
        coords={
            "x": ("x", xs),
            "y": ("y", ys)
        }
    )
    return XR

def getPopulation(state,year):
    F = glob.glob(f'../raw_data/population/population_{year}_US-{state_to_iso[state]}.tif')
    if len(F) == 1:
        pop_density = raster_to_xr(F[0])
        #print(pop_density['band'].mean())
        #print(pop_density)
    #else:
    #    print(f'Location {state} not in Locale attempting to generate it...')
    #    pop_density = rioxarray.open_rasterio('../raw_data/population/population.tif')
    #    pop_density = pop_density.rio.write_crs("EPSG:4326")
    #    gdf = download_gdf(country='USA',resolution=1)
    #    gdf = gdf[gdf.NAME_1==state]
    #    pop_density = pop_density.rio.clip(gdf.geometry, gdf.crs)
    # Convert it back to a DataFrame for further analysis
    return pop_density

# Reprojecting function using pyproj
def reproject_to_epsg3857(data_array, x, y):
    # Define the source and target CRS
    src_crs = pyproj.CRS("EPSG:4326")
    dst_crs = pyproj.CRS("EPSG:3857")

    # Create the transform for the target CRS
    x_res = (x[-1] - x[0]) / (len(x) - 1)
    y_res = (y[-1] - y[0]) / (len(y) - 1)
    transform = from_bounds(x[0], y[-1], x[-1], y[0], len(x), len(y))

    # Output arrays for the reprojected data
    dst_data = np.empty_like(data_array)
    dst_x = np.linspace(x[0], x[-1], len(x))
    dst_y = np.linspace(y[0], y[-1], len(y))

    # Reproject
    reproject(
        source=data_array,
        destination=dst_data,
        src_transform=transform,
        src_crs=src_crs,
        dst_crs=dst_crs,
        resampling=Resampling.nearest
    )
    return dst_data, dst_x, dst_y

'''def reproject_to_epsg3857(data, transform, width, height):
    """Reproject the data to EPSG:3857."""
    # Calculate the bounds using the transform
    x_min, y_min = transform * (0, 0)
    x_max, y_max = transform * (width, height)

    # Calculate transform and bounds for destination
    dst_transform, dst_width, dst_height = calculate_default_transform(
        'EPSG:4326', 'EPSG:3857', width, height, left=x_min, bottom=y_min, right=x_max, top=y_max
    )

    # Prepare output array
    data_reprojected = np.empty((dst_height, dst_width), dtype=np.float32)

    # Reproject data
    reproject(
        source=data,
        destination=data_reprojected,
        src_transform=transform,
        src_crs='EPSG:4326',
        dst_transform=dst_transform,
        dst_crs='EPSG:3857',
        resampling=Resampling.nearest
    )

    # Generate new x and y coordinates
    xs = np.linspace(dst_transform.c, dst_transform.c + dst_transform.a * dst_width, dst_width)
    ys = np.linspace(dst_transform.f + dst_transform.e * dst_height, dst_transform.f, dst_height)  # Notice the flip of y-axis

    return data_reprojected, xs, ys'''

def geoMasking(data_array, shape):
    # Get the bounding box of the shapefile
    gdf = gpd.GeoDataFrame({'geometry': shape}, crs="EPSG:4326")
    # Convert the GeoDataFrame to a mask that matches the data shape
    shapes = [mapping(geom) for geom in gdf.geometry]
    # Extracting the exact bounds from your data
    min_lon, max_lon = data_array.x.values.min(), data_array.x.values.max()
    min_lat, max_lat = data_array.y.values.min(), data_array.y.values.max()

    # Use these bounds to create a transform
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, data_array.sizes['x'], data_array.sizes['y'])
    mask = False
    try:
        mask = features.geometry_mask(shapes, transform=transform, invert=True, out_shape=(data_array.sizes['y'], data_array.sizes['x']))
    except:
        pass
    if mask is not False:
        #masked data
        data_masked = np.where(mask, data_array, np.nan)
    else:
        data_masked = data_array
    return data_masked

# Function to calculate the endpoint of a line given a start point, bearing, and distance
def calculate_endpoint(lon, lat, bearing, distance=200):  # Shorter distance, e.g., 1 km (1000 meters)
    # Convert distance to degrees
    delta_lat = distance / 111320  # 1 degree latitude = ~111.32 km
    delta_lon = distance / (111320 * np.cos(np.radians(lat)))  # Longitude correction by latitude

    # Calculate the endpoint using bearing
    end_lat = lat + delta_lat * np.sin(np.radians(bearing))
    end_lon = lon + delta_lon * np.cos(np.radians(bearing))

    return end_lon, end_lat

# Function to map slope to a color from red to green
def slope_to_color(slope,gdf_boxes):
    norm_slope = (slope - gdf_boxes['slope_angle'].min()) / (gdf_boxes['slope_angle'].max() - gdf_boxes['slope_angle'].min())
    r = int(255 * (1 - norm_slope))
    g = int(255 * norm_slope)
    return f'#{r:02x}{g:02x}{0:02x}'  # RGB hex string

# Function to calculate the arrowhead points
def calculate_arrowhead(lon, lat, bearing, distance=80):
    # Rotate the bearing 90 degrees to the right
    bearing = (bearing + 90) % 360

    # Left and right bearings for the arrowhead
    left_bearing = (bearing + 150) % 360
    right_bearing = (bearing - 150) % 360

    # Calculate the left and right points of the arrowhead
    left_point = calculate_endpoint(lon, lat, left_bearing, distance)
    right_point = calculate_endpoint(lon, lat, right_bearing, distance)

    return [left_point, (lon, lat), right_point]

# Define the functions used in your light data code
def calculate_endpoint(lon, lat, bearing, distance=1000):
    delta_lat = distance / 111320
    delta_lon = distance / (111320 * np.cos(np.radians(lat)))
    end_lat = lat + delta_lat * np.sin(np.radians(bearing))
    end_lon = lon + delta_lon * np.cos(np.radians(bearing))
    return end_lon, end_lat

def slope_to_color(slope,gdf_boxes):
    norm_slope = (slope - gdf_boxes['slope_angle'].min()) / (gdf_boxes['slope_angle'].max() - gdf_boxes['slope_angle'].min())
    r = int(255 * (1 - norm_slope))
    g = int(255 * norm_slope)
    return f'#{r:02x}{g:02x}{0:02x}'

def calculate_arrowhead(lon, lat, bearing, distance=500):
    bearing = (bearing + 90) % 360
    left_bearing = (bearing + 150) % 360
    right_bearing = (bearing - 150) % 360
    left_point = calculate_endpoint(lon, lat, left_bearing, distance)
    right_point = calculate_endpoint(lon, lat, right_bearing, distance)
    return [left_point, (lon, lat), right_point]
