import folium as fl
from streamlit_folium import st_folium
import streamlit as st
from folium import plugins
import requests
from streamlit.components.v1 import html

def create_folium_map():

    geo_json_data = requests.get(
        "https://raw.githubusercontent.com/python-visualization/folium-example-data/main/us_states.json"
    ).json()

    CENTER_START = [39.0997, -94.5786]
    ZOOM_START = 3

    # if "click" not in st.session_state:
    #     st.session_state["click"] = 0
    # if "center" not in st.session_state:
    #     st.session_state["center"] = CENTER_START

    m = fl.Map(location=CENTER_START, #st.session_state["center"],
            zoom_start=ZOOM_START,
            width='100%',
            height='100%'
            )

    m.add_child(fl.LatLngPopup())

    fl.GeoJson(geo_json_data).add_to(m)

    return m

def main():
    st.title("Illuminating Horizons")

    # Create the Folium map
    folium_map = create_folium_map()

    # Convert Folium map to HTML
    map_html = folium_map._repr_html_()

    # Display the Folium map in Streamlit
    html(map_html, height=800, width=1200)

if __name__ == "__main__":
    main()
