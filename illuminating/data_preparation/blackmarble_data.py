from blackmarble.extract import bm_extract
from blackmarble.raster import bm_raster
import geopandas as gdf
import pandas as pd

def get_blackmable(
    gdf:gdf.geodataframe.GeoDataFrame,
    bearer:str,
    date_start:str = "2022-01-01",
    date_end:str ="2022-01-01",
    freq:str="monthly",
    #return_format:str="raw",
):
    bm_query_dict = {
        "daily":
            {"type":"VNP46A2",
             "freq":"D"},
        "monthly":
            {"type":"VNP46A3",
             "freq":"MS"},
        "yearly":
            {"type":"VNP46A4",
             "freq":"YS"},
    }

    date_range = pd.date_range(date_start,date_end,
                               freq=bm_query_dict[freq.lower()]["freq"])

    bm_data = bm_raster(gdf,
                        product_id=bm_query_dict[freq.lower()]["type"],
                        date_range=date_range,
                        bearer=bearer
                        )
    # if return_format=="raw":
    #     return bm_data
    return bm_data


def blackmarble_to_df(
    input_xarray,
    split_time:bool = True,
):
    bm_df = input_xarray.to_dataframe().reset_index(level=[0,1]).dropna()
    time_indexes = [value for index,value in pd.DataFrame(bm_df.index).drop_duplicates().iterrows()]
    if not split_time or (len(time_indexes)==1):
        return bm_df

    df_list = []
    for time in time_indexes:
        df_list.append(bm_df.loc[time,:].reset_index())
    return df_list
