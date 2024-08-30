import folium as fl
from streamlit_folium import st_folium
import streamlit as st
from folium import plugins
import json
from folium import GeoJson, LayerControl
import geopandas as gpd

CENTER_START = [36.7783, -119.4179]
ZOOM_START = 5

def create_folium_map():
    if "zoom" not in st.session_state:
        st.session_state["zoom"] = ZOOM_START
    if "center" not in st.session_state:
        st.session_state["center"] = CENTER_START
    if "click" not in st.session_state:
        st.session_state["click"] = 0

    m = fl.Map(
        location=st.session_state["center"],
        zoom_start=st.session_state["zoom"],
        width='100%',  # Set the map width to 100%
        height='100%'  # Set the map height to 100%
    )

    m.add_child(fl.LatLngPopup())

    return m

def main():
    st.set_page_config(layout="wide")

    # Custom CSS to make the map height dynamic
    st.markdown(
        """
        <style>
        .fullScreenMap {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            height: 100vh;  /* Full viewport height */
            width: 100vw;   /* Full viewport width */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

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

        # Create a GeoJSON layer for grid
        geojson_layer_grid = fl.GeoJson(
            random_grid_data,
            name='Grid Layer',
            style_function=lambda feature: {
                "fillColor": "#33b8ff",
                "color": "blue",
                "opacity": "0",
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
                "opacity": "0",
                "weight": "0",
            }
        )

        # Add layers to the map
        geojson_layer_grid.add_to(folium_map)
        geojson_layer_landcover.add_to(folium_map)

    # Full screen map in Streamlit using custom CSS
    st.markdown(f'<div class="fullScreenMap">{st_folium(folium_map, width="100%")}</div>', unsafe_allow_html=True)

    if st.session_state.get("last_clicked"):
        st.session_state["zoom"] = 11.4
        st.session_state["center"] = [34.9602, -116.8231]

if __name__ == "__main__":
    main()
