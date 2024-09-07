import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.compose import make_column_selector
from colorama import Fore, Style
import joblib



from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
import os

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(\
            os.path.abspath(__file__)),"../.."))

RAW_DATA_PATH = os.path.join(ROOT_PATH,"raw_data")
ML_MODEL_PATH = os.path.join(RAW_DATA_PATH,"models")

def round_to_tenth(x):
    return np.round(x,-1)

def preprocess_features(X:gpd.GeoDataFrame):
    def create_sklearn_preprocessor():
        ## Function to round the landcover to nearest tenth




        landcover_categories = [[cat for cat in range(10,230,10)]]

        landcover_round = FunctionTransformer(round_to_tenth)
        landcover_onehot = OneHotEncoder(
                          categories=landcover_categories,
                          handle_unknown="ignore",
                          sparse_output=False
                         )

        landcover_pipe = Pipeline([
            ("round",landcover_round),
            ("onehot",landcover_onehot),
            ])

        number_columns = ['distance_to_population', 'distance_to_road',
       'distance_to_powerline', 'sun_radiation', 'daytime_temperature']

        final_preprocessor=ColumnTransformer(
            [
                ("number",RobustScaler(),number_columns),
                ("landcover", landcover_pipe,["landcover"])

            ],
            n_jobs=-1,
            )


        return final_preprocessor

    print(Fore.BLUE + "\nPreprocessing features..." + Style.RESET_ALL)
    number_columns = ['distance_to_population', 'distance_to_road',
       'distance_to_powerline', 'sun_radiation', 'daytime_temperature']
    preprocessor = create_sklearn_preprocessor()
    X_processed = preprocessor.fit_transform(X)
    landcover_feature_names = preprocessor.named_transformers_['landcover'].named_steps['onehot'].get_feature_names_out(['landcover'])
    all_features_names = np.concatenate([number_columns,landcover_feature_names])
    X_processed_df = pd.DataFrame(X_processed,columns=all_features_names)

    file_dir = os.path.join(ML_MODEL_PATH,"preprocessing_pipeline.pkl")
    joblib.dump(preprocessor, file_dir)
    print("✅ X_processed, with shape", X_processed.shape)

    return X_processed_df
