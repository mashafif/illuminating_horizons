import geopandas as gpd
import json
import pandas as pd
import os
<<<<<<< HEAD

def download_gdf(
    country:str="USA",
=======
from shapely.geometry import Polygon
from math import sqrt,cos
from shapely.geometry import LineString


def decimal_range(start, stop, increment):
    while start < stop: # and not math.isclose(start, stop): Py>3.5
        yield start
        start += increment


def download_gdf(
    country:str="GHA",
>>>>>>> 2dbcf7d3c239b35d736bb382c1bfe93161ecd355
    resolution:int=1,
):
    base_url = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41"
    root_path = os.path.dirname(os.path.dirname(__file__))
    json_path = os.path.join(root_path,"data_preparation","country_ISO.json")
    with open(json_path,"r") as file:
        country_ISO_df = pd.DataFrame(json.load(file))
    if len(country)!=3:
        country_ISO = country_ISO_df[country_ISO_df["name"]==country.title()].iloc[0]["alpha-3"]
    else:
        country_ISO=country
    download_url = f"{base_url}_{country_ISO}_{resolution}.json.zip"
    gdf = gpd.read_file(download_url)
    return gdf

<<<<<<< HEAD
=======
def km_to_long(km,lat):
    return abs(km/(cos(lat)*111.320))

def km_to_lat(km):
    return abs(km/110.574)

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
        road_json = json.loads(file.readline())
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


>>>>>>> 2dbcf7d3c239b35d736bb382c1bfe93161ecd355


if __name__ == "__main__":
    print(download_gdf("Afghanistan"))
    root_path = os.path.dirname(os.path.dirname(__file__))
    print(root_path)
