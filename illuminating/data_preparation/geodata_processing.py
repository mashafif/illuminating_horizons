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


def decimal_range(start, stop, increment):
    while start < stop: # and not math.isclose(start, stop): Py>3.5
        yield start
        start += increment


def download_gdf(
    country:str="USA",
    resolution:int=1,
):

    json_root_path = os.path.dirname(os.path.dirname(__file__))
    json_path = os.path.join(json_root_path,"data_preparation","country_ISO.json")
    with open(json_path,"r") as file:
        country_ISO_df = pd.DataFrame(json.load(file))
    if len(country)!=3:
        country_ISO = country_ISO_df[country_ISO_df["name"]==country.title()].iloc[0]["alpha-3"]
    else:
        country_ISO=country

    file_name=f"{country_ISO}_border.geojson"
    file_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.abspath(os.path.join(file_dir,"../.."))
    border_data_path  = os.path.join(root_path,"raw_data","borders_info",file_name)
    print(border_data_path)


    if os.path.isfile(border_data_path):
        gdf = gpd.read_file(border_data_path)
        print(f"importing from{border_data_path}")
    else:
        base_url = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41"
        download_url = f"{base_url}_{country_ISO}_{resolution}.json.zip"
        gdf = gpd.read_file(download_url)
    return gdf




def km_to_long(km,lat):
    return abs(km/(cos(lat)*111.320))

def km_to_lat(km):
    return abs(km/110.574)



from geopy.distance import geodesic

def get_bounding_box(center_lat, center_lon, distance_km=5):
    """
    Calculate the bounding box of a square area of size 10x10 km centered on a given latitude and longitude.

    Parameters:
    center_lat (float): Latitude of the center point.
    center_lon (float): Longitude of the center point.
    distance_km (float): Half the side length of the square (5 km for a 10x10 km box).

    Returns:
    dict: Bounding box with min_lat, max_lat, min_lon, max_lon.
    """

    # Calculate the coordinates of the bounding box
    north = geodesic(kilometers=distance_km).destination((center_lat, center_lon), 0)
    south = geodesic(kilometers=distance_km).destination((center_lat, center_lon), 180)
    east = geodesic(kilometers=distance_km).destination((center_lat, center_lon), 90)
    west = geodesic(kilometers=distance_km).destination((center_lat, center_lon), 270)

    bounding_box = {
        "min_lat": south.latitude,
        "max_lat": north.latitude,
        "min_lon": west.longitude,
        "max_lon": east.longitude,
    }

    return bounding_box


def create_grid_from_bbox(bbox, grid_size_km=1):
    """
    Create a grid of squares within a bounding box and return a GeoDataFrame containing the geometry of each square.

    Parameters:
    bbox (dict): Bounding box with keys min_lat, max_lat, min_lon, max_lon.
    grid_size_km (float): Size of each grid square in kilometers.

    Returns:
    gpd.GeoDataFrame: GeoDataFrame containing the geometry of each grid square.
    """
    # Calculate the number of rows and columns based on grid size
    lat_diff = bbox['max_lat'] - bbox['min_lat']
    lon_diff = bbox['max_lon'] - bbox['min_lon']

    # Calculate approximate size in degrees for the grid based on the latitude
    lat_size = grid_size_km / 111  # 1 degree latitude is ~111 km
    lon_size = grid_size_km / (111 * np.cos(np.radians((bbox['min_lat'] + bbox['max_lat']) / 2)))  # Adjust for longitude

    rows = int(np.ceil(lat_diff / lat_size))
    cols = int(np.ceil(lon_diff / lon_size))

    polygons = []

    for i in range(rows):
        for j in range(cols):
            min_lat = bbox['min_lat'] + i * lat_size
            max_lat = min_lat + lat_size
            min_lon = bbox['min_lon'] + j * lon_size
            max_lon = min_lon + lon_size

            # Create a polygon for each grid square
            polygons.append(Polygon([(min_lon, min_lat), (min_lon, max_lat), (max_lon, max_lat), (max_lon, min_lat)]))
            #print(f"converting{i}{j} as {polygons}")

    # Create a GeoDataFrame
    grid_gdf = gpd.GeoDataFrame(geometry=polygons)

    return grid_gdf

def bbox_to_geodataframe(bbox):
    """
    Convert a bounding box to a GeoDataFrame containing a single polygon representing the bounding box.

    Parameters:
    bbox (dict): Bounding box with keys min_lat, max_lat, min_lon, max_lon.

    Returns:
    gpd.GeoDataFrame: GeoDataFrame containing the bounding box as a single polygon.
    """
    # Create a polygon representing the bounding box
    bbox_polygon = Polygon([
        (bbox['min_lon'], bbox['min_lat']),  # Bottom-left corner
        (bbox['min_lon'], bbox['max_lat']),  # Top-left corner
        (bbox['max_lon'], bbox['max_lat']),  # Top-right corner
        (bbox['max_lon'], bbox['min_lat']),  # Bottom-right corner
        (bbox['min_lon'], bbox['min_lat'])   # Close the polygon
    ])

    # Create a GeoDataFrame with the polygon
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_polygon])

    return bbox_gdf

def check_pop(pop,first_q,sec_q,third_q):
    if pop>first_q and pop<sec_q:
        return 10
    elif pop>sec_q and pop<third_q:
        return 50
    elif pop>third_q:
        return 100
    return 0

def categorize_pop(pop_gdf):
    first_q =math.floor(pop_gdf[pop_gdf.population > 10]["population"].quantile([0.25]).iloc[0])
    sec_q = math.floor(pop_gdf[pop_gdf.population > 10]["population"].quantile([0.5]).iloc[0])
    third_q = math.floor(pop_gdf[pop_gdf.population > 10]["population"].quantile([0.75]).iloc[0])
    pop_gdf["population_size"] = pop_gdf["population"].apply(check_pop,args=[first_q,sec_q,third_q])
    return pop_gdf




def create_grids(
  gdf,
  km:float #in kilometer
):
    minx, miny, maxx, maxy = gdf.total_bounds

    squares = []
#side_length_x =
    x=minx
    y=miny

    side_length_x = km_to_long(km,y)
    side_length_y = km_to_lat(km)

    for y in decimal_range(miny,maxy,side_length_y):
        side_length_x = km_to_long(km,y)
        for x in decimal_range(minx,maxx,km_to_long(km,y)):
            square = Polygon([(x, y), (x + side_length_x, y), (x + side_length_x, y + side_length_y), (x, y + side_length_y)])
            squares.append(square)

    squares_gdf = gpd.GeoDataFrame(geometry=squares, crs=gdf.crs)
    result_gdf = gpd.overlay(squares_gdf, gdf, how="intersection").iloc[1:]
    return result_gdf[["geometry"]]


def import_road(
    geojson_dir:str,
    crs:str):

    with open(geojson_dir) as file:
        road_json = json.load(file)
    geometry=[]
    for feature in road_json["features"]:
        geom_type = feature["geometry"]["type"]
        geom_coor = feature["geometry"]["coordinates"]
        if geom_type == "LineString":
            geometry.append(LineString(geom_coor))

    #print(geometry)

    road_gdf = gpd.GeoDataFrame({
        'geometry': geometry
    },crs=crs)
    return road_gdf


def jaxaraster_to_pixel(data,x_1,x_2,y_1,y_2):
    x_start = x_1
    x_end = x_2
    y_start = y_2
    y_end = y_1
    x_px = data.shape[1]
    y_px = data.shape[0]
    x_range = x_end-x_start
    y_range = y_start-y_end
    x_res = x_range/x_px
    y_res = y_range/y_px
    # x_start = data.raster.lonlim[0][0]
    # y_start = data.raster.latlim[0][1]

    poly_list=[]
    for y in range(0,y_px):
        for x in range(0,x_px):
            pixel = Polygon([[x_start,y_start],
                        [x_start+x_res,y_start],
                        [x_start+y_res,y_start+y_res],
                        [x_start,y_start+y_res]])
            poly_list.append(pixel)
            x_start+=x_res
        x_start = x_1
        y_start-=y_res
    return poly_list


def get_jaxa_dataset(
    gdf,
    dataset:str="landcover",
    raw_data_dir:str="raw_data",
    start_time:str = "2019-01-01T00:00:00",
    end_time:str = "2019-01-01T00:00:00",
    ppu:int=20,
    crs="epsg:4326"
    ):
    # root_path = os.path.abspath('../')
    # json_path = os.path.join(root_path,raw_data_dir,json_name)

    dlim = [start_time,end_time]
    ppu  = ppu

    # the dataset dictionary
    datasets_dict={
        "landcover":{
            "collection":"Copernicus.C3S_PROBA-V_LCCS_global_yearly",
            "band": "LCCS"
        },
        "sun_radiation":{
            "collection":"JAXA.JASMES_Aqua.MODIS_swr.v811_global_monthly",
            "band": "swr"
        },
        "daytime_temperature":{
            "collection":"JAXA.G-Portal_GCOM-C.SGLI_standard.L3-LST.daytime.v3_global_monthly",
            "band": "LST"
        }

    }

    # Set information of collection,band
    collection = datasets_dict[dataset]["collection"]
    band       = datasets_dict[dataset]["band"]

    # Get bounding box
    # geoj_path = json_path
    # geoj = je.FeatureCollection().read(geoj_path).select()
    bbox = gdf.total_bounds


    # Get an image
    data = je.ImageCollection(collection=collection,ssl_verify=True)\
            .filter_date(dlim=dlim)\
            .filter_resolution(ppu=ppu)\
            .filter_bounds(bbox=bbox)\
            .select(band=band)\
            .get_images()

    # Process and show an image
    img = je.ImageProcess(data)

    # Convert the raster data to geodataframe
    x_start = img.raster.lonlim[0][0]
    x_end = img.raster.lonlim[0][1]
    y_start = img.raster.latlim[0][0]
    y_end = img.raster.latlim[0][1]

    no_of_time = img.raster.img.shape[0]
    total_row = img.raster.img.shape[1]*img.raster.img.shape[2]
    if no_of_time >1:
        lc_coor_res=[]
        for n in range(no_of_time):
            lc_coor = gpd.GeoDataFrame(pd.DataFrame({"geometry":jaxaraster_to_pixel(
                                                img.raster.img[n],
                                                x_start,x_end,
                                                y_start,y_end)}),
                                                crs=crs)
            landcover = np.reshape(img.raster.img[n],(total_row,1))
            lc_coor[dataset]=landcover
            lc_coor_res.append(lc_coor)
    else:
        lc_coor = gpd.GeoDataFrame(pd.DataFrame({"geometry":jaxaraster_to_pixel(
                                                img.raster.img[0],
                                                x_start,x_end,
                                                y_start,y_end)}),
                                                crs=crs)
        landcover = np.reshape(img.raster.img[0],(total_row,1))
        lc_coor["landcover"]=landcover
        lc_coor_res = lc_coor

    return lc_coor_res

def get_jaxa_average(
    gdf_list,
    coor_crs="epsg:4326",
    dist_crs="epsg:32733"):
    feature_name = gdf_list[0].columns[1]
    gdf_res = gdf_list[0].copy()
    gdf_value = gdf_list[0][[feature_name]]

    for index,gdf in enumerate(gdf_list[1:]):
        gdf_res[feature_name] +=gdf[feature_name].copy()
    gdf_res[feature_name] = gdf_res[feature_name]/len(gdf_list)
    return gdf_res


def calculate_pop_z_scores(df, column_name,minimum_val=10):
    """
    Calculate the Z-scores for a specific column in a pandas DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the column for which to calculate the Z-scores.

    Returns:
    pandas.Series: A Series of Z-scores for the specified column.
    """
    df = df[df[column_name]>=minimum_val]
    mean = df[column_name].mean()
    std_deviation = df[column_name].std()

    z_scores = (df[column_name] - mean) / std_deviation
    return z_scores

def calculate_pop_percentile(df,column,minimum_val=10):

    df_res = df[df[column]>=10].copy()
    df_res[f'{column}_percentile'] = df_res[column].rank(pct=True) * 100
    return df_res

def sjoin_nearest_illuminating(gdf1,gdf2,distance_name=None,crs = "epsg:4326"):
    gdf1_cp = gdf1.copy()
    gdf2_cp = gdf2.copy()
    res_gdf = gpd.sjoin_nearest(gdf1_cp.to_crs(epsg=32733),\
        gdf2_cp.to_crs(epsg=32733),\
            distance_col=distance_name).to_crs(crs)
    res_gdf.drop(columns=['index_right'],inplace=True)
    return res_gdf




if __name__ == "__main__":
    pass
