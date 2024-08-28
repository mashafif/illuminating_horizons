from illuminating.data_preparation.geodata_processing import download_gdf,\
    calculate_pop_percentile, get_jaxa_dataset, get_jaxa_average,\
    sjoin_nearest_illuminating
from sklearn.model_selection import train_test_split
from illuminating.ml.preprocessor import preprocess_features
from illuminating.ml.model import ml_model_selection,dl_model_selection
from sqlalchemy.orm import sessionmaker


import zipfile
import os
import geopandas as gpd
import pandas as pd
import gc
from sqlalchemy import create_engine,text
import geoalchemy2
import time
import joblib
import numpy as np
import json

BASE_FEATURES = ["population","road","powerline","radiation","temperature",
                 "landcover"]

FEATURES_W_SLOPE = ["population","road","powerline","radiation","temperature",
                  "landcover","slope"]

SQL_URL = 'postgresql://postgres:lewagon@localhost/geospatial_db'

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(\
            os.path.abspath(__file__)),"../.."))

RAW_DATA_PATH = os.path.join(ROOT_PATH,"raw_data")
ML_MODEL_PATH = os.path.join(RAW_DATA_PATH,"models")


def get_population(country,border_gdf):
    population_dir = os.path.join(RAW_DATA_PATH,"population")
    pop_file_dir = os.path.join(population_dir,\
                     f"{country.lower()}_ppp_2019_1km_ASCII_XYZ.zip")

    if os.path.isfile(pop_file_dir):
        df_zip = zipfile.ZipFile(pop_file_dir)
        file_name = df_zip.namelist()[0]
        pop_df = pd.read_csv(df_zip.open(file_name))
        pop_gdf = gpd.GeoDataFrame(pop_df,
                                   geometry=gpd.points_from_xy(\
                                    pop_df.X,pop_df.Y),
                                    crs="EPSG:4326")
        del pop_df
        gc.collect()

        pop_gdf.drop(columns=["X","Y"],inplace=True)
        pop_gdf.rename(columns={"Z":"population"},inplace=True)
        pop_gdf = pop_gdf.sjoin(border_gdf,how="inner")

        pop_gdf = pop_gdf[["population","geometry"]].reset_index()
        pop_gdf = pop_gdf[["population","geometry"]]

        pop_gdf = calculate_pop_percentile(pop_gdf,"population")
        pop_gdf = pop_gdf[pop_gdf["population_percentile"] >= 80].reset_index().\
                        drop(columns=["index"])
        print("✅ Importing population done")
        return pop_gdf


def get_road(border_gdf,country):

    minx,miny,maxx,maxy = border_gdf.total_bounds

    engine = create_engine(SQL_URL)
    road_dict ={
        "USA" : "road",
        "GHA" : "road_africa"
    }
    query = f"""
    SELECT *
    FROM {road_dict[country]}
    WHERE geometry && ST_MakeEnvelope({minx}, {miny}, {maxx}, {maxy}, 4326);
    """
    print("Importing road data......")
    road_gdf = gpd.read_postgis(query, engine, geom_col='geometry')
    print("✅ Importing road data done")
    return road_gdf



def get_powerline(border_gdf,country):

    minx,miny,maxx,maxy = border_gdf.total_bounds

    engine = create_engine(SQL_URL)

    powerline_dict ={
        "USA" : "powerline",
        "GHA" : "powerline_africa"
    }

    query = f"""
    SELECT *
    FROM {powerline_dict[country]}
    WHERE geometry && ST_MakeEnvelope({minx}, {miny}, {maxx}, {maxy}, 4326);
    """
    print("Importing power line data....")
    power_gdf = gpd.read_postgis(query, engine, geom_col='geometry')
    print("✅ Importing powerline data done")

    return power_gdf


def get_radiation(border_gdf):

    print("Importing radiation data....")
    radiation_db=get_jaxa_dataset(border_gdf,
                                  "sun_radiation",ppu=100,
                                  start_time = "2019-01-01T00:00:00",
                                  end_time = "2019-12-01T00:00:00")
    radiation_gdf = get_jaxa_average(radiation_db)
    print("✅ Importing radiation data done")

    return radiation_gdf


def get_temperature(border_gdf):

    print("Importing temperature data...")
    temperature_db=get_jaxa_dataset(border_gdf  ,
                                    "daytime_temperature",
                                    ppu=100,
                                    start_time = "2019-01-01T00:00:00",
                                    end_time = "2019-12-01T00:00:00")
    temperature_gdf = get_jaxa_average(temperature_db)
    print("✅ Importing temperature data done")

    return temperature_gdf



def get_temperature_small(border_gdf):

    print("Importing temperature data...")
    temperature_db=get_jaxa_dataset(border_gdf  ,
                                    "daytime_temperature",
                                    ppu=20,
                                    start_time = "2019-01-01T00:00:00",
                                    end_time = "2019-12-01T00:00:00")
    temperature_gdf = get_jaxa_average(temperature_db)
    print("✅ Importing temperature data done")

    return temperature_gdf


def get_landcover(border_gdf):
    print("Importing landcover data...")
    landcover_gdf=get_jaxa_dataset(border_gdf,
                                   "landcover",
                                   ppu=100,
                                   start_time = "2019-01-01T00:00:00",
                                   end_time = "2019-12-01T00:00:00")
    print("✅ Importing landcover data done")
    return landcover_gdf


def get_landcover_small(border_gdf):
    print("Importing landcover data...")
    landcover_gdf=get_jaxa_dataset(border_gdf,
                                   "landcover",
                                   ppu=20,
                                   start_time = "2019-01-01T00:00:00",
                                   end_time = "2019-12-01T00:00:00")
    print("✅ Importing landcover data done")
    return landcover_gdf


# def get_grids_data(border_gdf):

#     minx,miny,maxx,maxy = border_gdf.total_bounds

#     engine = create_engine(SQL_URL)

#     query = f"""
#     SELECT *
#     FROM "USA_five_km"
#     WHERE geometry && ST_MakeEnvelope({minx}, {miny}, {maxx}, {maxy}, 4326);
#     """
#     print("Importing grid data for prediction....")
#     power_gdf = gpd.read_postgis(query, engine, geom_col='geometry')
#     print("✅ Importing grid data done")

#     return power_gdf



def get_datasets(country="USA",
                         state="California",
                         powerplant_loc_file = "power_plant_loc.csv",
                         features:list=BASE_FEATURES):
    ## Get the border data for the target country or state/province
    country_gdf = download_gdf("USA",1)
    if state and (state in country_gdf.NAME_1.to_list()):
        border_gdf = country_gdf[country_gdf.NAME_1== state]

    ## 1. Get the target data for base gdf
    if "slope" in features:
        base_columns = ["Longitude","Latitude",
                        "elevation","slope","slope_bearing","have_plant"]
    else:
        base_columns = ["Longitude","Latitude","have_plant"]
    train_filepath = os.path.join(RAW_DATA_PATH,powerplant_loc_file)

    training_df = pd.read_csv(train_filepath)[base_columns]
    training_gdf = gpd.GeoDataFrame(training_df,
                                    geometry=gpd.points_from_xy(\
                                    training_df.Longitude,
                                    training_df.Latitude),
                                    crs = "EPSG:4326")
    del training_df
    gc.collect()

    # 2. Get population data
    if "population" in features:
        pop_gdf = get_population(country,border_gdf)
        training_gdf = sjoin_nearest_illuminating(training_gdf, pop_gdf,
                                                  "distance_to_population")
        del pop_gdf
        gc.collect()

    #3. Get road data
    if "road" in features:
        road_gdf = get_road(border_gdf,country)
        training_gdf = sjoin_nearest_illuminating(training_gdf,road_gdf,
                                                  "distance_to_road")
        del road_gdf
        gc.collect()


    #4 Get power line
    if "powerline" in features:
        power_gdf = get_powerline(border_gdf,country)
        training_gdf = sjoin_nearest_illuminating(training_gdf,power_gdf,
                                                        "distance_to_powerline")
        del power_gdf
        gc.collect()



 #5 radiationadiation
    if "radiation" in features:
        radiation_gdf = get_radiation(border_gdf)
        training_gdf = sjoin_nearest_illuminating(training_gdf,radiation_gdf)

        del radiation_gdf
        gc.collect()


    #6 Get temperature
    if "temperature" in features:
        temperature_gdf = get_temperature(border_gdf)
        training_gdf = sjoin_nearest_illuminating(training_gdf,temperature_gdf)

        del temperature_gdf
        gc.collect()


    #7 Get landcover
    if "landcover" in features:
        landcover_gdf = get_landcover(border_gdf)
        training_gdf = sjoin_nearest_illuminating(training_gdf,landcover_gdf)

        del landcover_gdf
        gc.collect()

    training_gdf.drop(columns=["Longitude","Latitude",
                               "population","population_percentile"],inplace=True)
    training_gdf.dropna(inplace=True)

    return training_gdf


def preprocessing_and_train(country="USA",
                         state="California",
                         powerplant_loc_file = "power_plant_loc.csv",
                         features:list=BASE_FEATURES):
    training_gdf = get_datasets(country,state,powerplant_loc_file,features)
    #preprocessing_and_train(training_gdf)
    X = training_gdf.drop(columns=["geometry","have_plant"])
    y = training_gdf["have_plant"]

    X_preprocessed = preprocess_features(X)
    ml_model_selection(X_preprocessed,y)

import os
import json

def write_geojson(gdf, file_path, mode='w'):
    # Convert GeoDataFrame to GeoJSON string
    geojson_str = gdf.to_json()

    if mode == 'w':
        # Write to file
        with open(file_path, 'w') as f:
            f.write(geojson_str)
    elif mode == 'a':
        if os.path.exists(file_path):
            # Append to existing file
            with open(file_path, 'r+') as f:
                try:
                    data = json.load(f)
                    if 'features' in data:
                        features = data['features']
                    else:
                        features = []
                    features.extend(json.loads(geojson_str)['features'])
                    data['features'] = features
                except json.JSONDecodeError:
                    # If the file is empty or corrupted, start fresh
                    data = json.loads(geojson_str)

                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()  # In case the new content is shorter than the old content
        else:
            # If the file does not exist, create it and write the GeoJSON data
            with open(file_path, 'w') as f:
                f.write(geojson_str)



def predict_per_chunk(pred_gdf,border_gdf,radiation_gdf,
                      temperature_gdf,pop_gdf,road_gdf,power_gdf,landcover_gdf,
                      country="USA",
            model_file = "Random Forest_model.pkl",
            features:list=BASE_FEATURES):


        # 2. Get population data
    #pred_gdf_raw = pred_gdf.copy()
    print(f"Initial size: {pred_gdf.shape}")
    pred_gdf["geometry"] = pred_gdf["geometry"].to_crs(epsg=32733).centroid.to_crs(epsg=4326)
    print(f"After centroid: {pred_gdf.shape}")

    pred_gdf = pred_gdf.drop_duplicates(subset=["geometry"])
    print(f"After dropping after centroid: {pred_gdf.shape}")

    print(f"1. Combining with population")

    pred_gdf = sjoin_nearest_illuminating(pred_gdf, pop_gdf,
                                            "distance_to_population")
    print(f"Size after population combined: {pred_gdf.shape}")

    pred_gdf = pred_gdf.drop_duplicates(subset=["geometry"])
    print(f"Size after drop duplicate: {pred_gdf.shape}")



    #3. Get road data
    print(f"2. Combining with road")

    pred_gdf = sjoin_nearest_illuminating(pred_gdf,road_gdf,
                                            "distance_to_road")

    print(f"Size after population combined: {pred_gdf.shape}")

    pred_gdf = pred_gdf.drop_duplicates(subset=["geometry"])
    print(f"Size after drop duplicate: {pred_gdf.shape}")



    #4 Get power line
    print(f"3. Combining with powerline")

    pred_gdf = sjoin_nearest_illuminating(pred_gdf,power_gdf,
                                                    "distance_to_powerline")
    print(f"Size after powerline combined: {pred_gdf.shape}")

    pred_gdf = pred_gdf.drop_duplicates(subset=["geometry"])
    print(f"Size after drop duplicate: {pred_gdf.shape}")




#5 radiation
    print(f"4. Combining with radiation")

    pred_gdf = sjoin_nearest_illuminating(pred_gdf,radiation_gdf)
    print(f"Size after radiation combined: {pred_gdf.shape}")

    pred_gdf = pred_gdf.drop_duplicates(subset=["geometry"])
    print(f"Size after drop duplicate: {pred_gdf.shape}")




    #6 Get temperature
    print(f"5. Combining with temperature")

    pred_gdf = sjoin_nearest_illuminating(pred_gdf,temperature_gdf)
    print(f"Size after temperature combined: {pred_gdf.shape}")

    pred_gdf = pred_gdf.drop_duplicates(subset=["geometry"])
    print(f"Size after drop duplicate: {pred_gdf.shape}")




    #7 Get landcover
    print(f"6. Combining with landcover")
    pred_gdf = sjoin_nearest_illuminating(pred_gdf,landcover_gdf)
    print(f"Size after landcover combined: {pred_gdf.shape}")

    pred_gdf = pred_gdf.drop_duplicates(subset=["geometry"])
    print(f"Size after drop duplicate: {pred_gdf.shape}")


    pred_gdf.drop(columns=["population","population_percentile"],inplace=True)
    pred_gdf = pred_gdf.drop_duplicates(subset=["geometry"])
    if "id"in pred_gdf.columns:
        pred_gdf.drop(columns=["id"],inplace=True)


    pred_gdf.dropna(inplace=True)

    X = pred_gdf.drop(columns=["geometry"])
    #y = pred_gdf["have_plant"]

    pipeline_dir = os.path.join(ML_MODEL_PATH,"preprocessing_pipeline.pkl")
    preprocess_pipline = joblib.load(pipeline_dir)
    landcover_feature_names = preprocess_pipline.named_transformers_['landcover'].named_steps['onehot'].get_feature_names_out(['landcover'])
    numbers_features = preprocess_pipline.named_transformers_['number'].get_feature_names_out()
    all_features = np.concatenate([numbers_features,landcover_feature_names])

    X_preprocessed = preprocess_pipline.fit_transform(X)
    X_preprocessed_df = pd.DataFrame(X_preprocessed,columns = all_features)

    ml_pipeline_dir = os.path.join(ML_MODEL_PATH,"Random Forest_model.pkl")
    ml_model = joblib.load(ml_pipeline_dir)
    y_pred_proba = ml_model.predict_proba(X_preprocessed_df)*100
    y_pred = ml_model.predict(X_preprocessed_df)

    predict_df = pd.DataFrame({"have_plant_proba":y_pred_proba.T[1],
                          "have_plant":y_pred,
                          "geometry":pred_gdf["geometry"]})
    predict_gdf = gpd.GeoDataFrame(predict_df,geometry="geometry")
    return predict_gdf

def read_geojson_in_chunks(session, sql, chunk_size=1000):
    result_proxy = session.execute(sql)
    column_names = list(result_proxy.keys())  # Convert RMKeyView to a list
    while True:
        rows = result_proxy.fetchmany(chunk_size)
        if not rows:
            break
        features = []
        for row in rows:
            try:
                geojson_geom = json.loads(row[0])  # This should be a valid GeoJSON string
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON: {e} - Data: {row[0]}")
                continue
            properties = {key: value for key, value in zip(column_names[1:], row[1:])}  # Accessing other columns
            feature = {
                "type": "Feature",
                "geometry": geojson_geom,
                "properties": properties
            }
            features.append(feature)
        yield gpd.GeoDataFrame.from_features(features)

def predict(country="USA",
            state="California",
            model_file = "Random Forest_model.pkl",
            features:list=BASE_FEATURES):
        ## Get the border data for the target country or state/province
    country_gdf = download_gdf(country,1)
    engine = create_engine(SQL_URL)
    if state and (state in country_gdf.NAME_1.to_list()):
            border_gdf = country_gdf[country_gdf.NAME_1== state]
    else:
        border_gdf=country_gdf


    #pred_gdf = get_grids_data(border_gdf)

    if "population" in features:
        pop_gdf = get_population(country,border_gdf)
    if "road" in features:
        road_gdf = get_road(border_gdf,country)
    if "radiation" in features:
        radiation_gdf = get_radiation(border_gdf)
    if "powerline" in features:
        power_gdf = get_powerline(border_gdf,country)
    if "landcover" in features:
        landcover_gdf = get_landcover(border_gdf)
    if "temperature" in features:
        temperature_gdf = get_temperature(border_gdf)


    minx,miny,maxx,maxy = border_gdf.total_bounds
    # Define your query and batch size
    output_file=os.path.join(RAW_DATA_PATH,"sample.geojson")
    if country=="USA":
        query_loc = "USA"
    else:
        query_loc="Africa"

    query = f"""
    SELECT ST_AsGeoJSON(geometry)
    FROM "{query_loc}_five_km"
    WHERE geometry && ST_MakeEnvelope({minx}, {miny}, {maxx}, {maxy}, 4326);
    """
    sql=text(query)



    Session = sessionmaker(bind=engine)
    session = Session()

    batch_size = 1000  # Number of rows to fetch per batch
    chunk_no = 0
    # Read data in batches
    first_chunk = True
    if not state:
        filename = country
    else:
        filename = state
    output_file = os.path.join(RAW_DATA_PATH,f"{filename}_prediction_result.geojson")

    for chunk in read_geojson_in_chunks(session, sql, chunk_size=1000):
        # Process each chunk
        chunk_no +=1
        print(f"Importing grid data for prediction..{chunk_no}..")
        print(chunk.crs)
        pred_gdf = chunk.set_crs(epsg=4326)
        pred_gdf = pred_gdf[pred_gdf.geometry.within(border_gdf.geometry.unary_union)]

        #print(pred_gdf.head())  # Replace with your processing logic
        if pred_gdf.shape[0]>=1:
            if first_chunk:
                pred_gdf = predict_per_chunk(pred_gdf,border_gdf,radiation_gdf,
                            temperature_gdf,pop_gdf,road_gdf,power_gdf,landcover_gdf,country=country,
                features=features)
                write_geojson(pred_gdf, output_file, mode='w')
                first_chunk = False
            else:
                # Append subsequent chunks to the file
                pred_gdf = predict_per_chunk(pred_gdf,border_gdf,radiation_gdf,
                            temperature_gdf,pop_gdf,road_gdf,power_gdf,landcover_gdf,country=country,
                features=features)
                write_geojson(pred_gdf, output_file, mode='a')

        # Close the engine
    engine.dispose()



if __name__ == '__main__':
    start_time = time.time()
    try:
        #preprocessing_and_train()
        predict(country="USA",state="California")



    except:
        import sys
        import traceback

        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)

    end_time = time.time()
    execution_time = end_time-start_time
    print(f"Execution time : {execution_time}")
