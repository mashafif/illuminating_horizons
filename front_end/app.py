import folium as fl
from streamlit_folium import st_folium
import streamlit as st
from folium import plugins
import requests

geo_json_data = requests.get(
    "https://raw.githubusercontent.com/python-visualization/folium-example-data/main/us_states.json"
).json()

st.title("Illuminating Horizons")

def get_pos(lat,lng):
    return lat,lng

CENTER_START = [39.0997, -94.5786]
ZOOM_START = 8

if "click" not in st.session_state:
    st.session_state["click"] = 0
if "center" not in st.session_state:
    st.session_state["center"] = CENTER_START

m = fl.Map(location= st.session_state["center"],
           tiles="OpenStreetMap",
           zoom_start=min(4+4*st.session_state["click"],15),
           max_zoom=min(4+4*st.session_state["click"],15))

m.add_child(fl.LatLngPopup())

fl.GeoJson(geo_json_data).add_to(m)

fl.raster_layers.ImageOverlay(
    image="https://upload.wikimedia.org/wikipedia/commons/f/f4/Mercator_projection_SW.jpg",
    name="jpeg",
    bounds=[[-82, -180], [82, 180]],
    opacity=1,
    interactive=False,
    cross_origin=False,
    zindex=1,
    alt="Wikipedia File:Mercator projection SW.jpg",
).add_to(m)

fl.LayerControl().add_to(m)

map = st_folium(m, height=500, width=750)

if map['last_clicked'] is not None:
    st.session_state["click"] += 1
    data = get_pos(map['last_clicked']['lat'],map['last_clicked']['lng'])
    st.session_state["center"] = data
    st.write(data)
