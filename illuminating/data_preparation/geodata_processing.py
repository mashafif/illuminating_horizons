import geopandas as gpd
import json
import pandas as pd

def download_gdf(
    country:str="GHA",
    resolution:int=1,
):
    base_url = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41"
    with open("illuminating/data_preparation/country_ISO.json","r") as file:
        country_ISO_df = pd.DataFrame(json.load(file))
    if len(country)!=3:
        country_ISO = country_ISO_df[country_ISO_df["name"]==country.title()].iloc[0]["alpha-3"]

    download_url = f"{base_url}_{country_ISO}_{resolution}.json.zip"
    gdf = gpd.read_file(download_url)
    return gdf



if __name__ == "__main__":
    print(download_gdf("Afghanistan"))
