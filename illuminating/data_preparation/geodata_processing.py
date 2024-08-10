import geopandas as gpd
import json
import pandas as pd
import os

def download_gdf(
    country:str="USA",
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



if __name__ == "__main__":
    print(download_gdf("Afghanistan"))
    root_path = os.path.dirname(os.path.dirname(__file__))
    print(root_path)
