import folium as fl
from streamlit_folium import st_folium
import streamlit as st
from folium import plugins
import requests
from streamlit.components.v1 import html
import json
from folium import GeoJson, LayerControl
import geopandas as gpd
import tempfile

CENTER_START = [36.7783, -119.4179]
ZOOM_START = 5

def create_folium_map():

    if "zoom" not in st.session_state:
        st.session_state["zoom"] = ZOOM_START
    if "center" not in st.session_state:
        st.session_state["center"] = CENTER_START
    if "click" not in st.session_state:
        st.session_state["click"] = 0

    m = fl.Map(location=st.session_state["center"],
            zoom_start=st.session_state["zoom"],
            max_zoom=st.session_state["zoom"]
            # width='100%',
            # height='100%'
            )

    m.add_child(fl.LatLngPopup())

    return m

def main():
    st.title("Illuminating Horizons")

    # Create the Folium map
    folium_map = create_folium_map()

    if st.button("Get prediction") or st.session_state["click"]:
        st.session_state["click"] = 1

        # Load grid data
        with open('random_grid.geojson', 'r') as file1:
            random_grid_data = json.load(file1)

        # Load simple landcover data
        with open('simple_landcover.geojson', 'r') as file2:
            landcover_data = gpd.read_file(file2)

        # landcover_data = landcover_data.drop([210])

        # Create a GeoJSON layer for grid
        geojson_layer_grid = fl.GeoJson(
            random_grid_data,
            name='Grid Layer',
            style_function=lambda feature: {
                "fillColor": "#33b8ff",
                "color": "blue",
                "oppacity": "0",
                "weight": "0",
            }
            )
        # Create a layer for landcover
        geojson_layer_landcover = fl.GeoJson(
            landcover_data,
            name='Landcover Layer',
            style_function=lambda feature: {
                "fillColor": "#FF0000",
                "color": "red",
                "oppacity": "0",
                "weight": "0",
            }
        )

        # Add layers to the map
        geojson_layer_grid.add_to(folium_map)
        geojson_layer_landcover.add_to(folium_map)


        # # Load population data
        # with open('population.geojson', 'r') as file2:
        #     population_data = json.load(file2)

        # # Create a geojson layer for population
        # geojson_layer_population = fl.GeoJson(
        #     population_data,
        #     name='Population Layer'
        #     # style_function=lambda feature: {
        #     #     "fillColor": "#33b8ff",
        #     #     "color": "blue",
        #     #     "oppacity": "0",
        #     #     "weight": "0",
        #     # }
        #     )

        # # Add the population layer to the map
        # geojson_layer_population.add_to(folium_map)

   # Display the Folium map in Streamlit

    display_map = st_folium(folium_map, height=500, width=500)

    if display_map['last_clicked'] is not None:
        st.session_state["zoom"] = 11.4
        st.session_state["center"] = [34.9602, -116.8231]
        # st.session_state["center"] = display_map['last_clicked']['lat'],display_map['last_clicked']['lng']

if __name__ == "__main__":
    main()
