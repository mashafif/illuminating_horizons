import geopandas as gpd
import json
import pandas as pd
import os
from shapely.geometry import Polygon,LineString
from math import sqrt,cos
from jaxa.earth import je
import numpy as np

def decimal_range(start, stop, increment):
    while start < stop: # and not math.isclose(start, stop): Py>3.5
        yield start
        start += increment


def download_gdf(
    country:str="GHA",
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


<<<<<<< HEAD:illuminating/data_preparation/geodata_preprocessing_tmp.py
=======
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

>>>>>>> 251461230b58e1d73ab2e2de084a5486b9acfe35:illuminating/data_preparation/geodata_processing.py
if __name__ == "__main__":
    print(download_gdf("Afghanistan"))
    root_path = os.path.dirname(os.path.dirname(__file__))
    print(root_path)
