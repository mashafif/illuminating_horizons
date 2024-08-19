import folium as fl
from streamlit_folium import st_folium
import streamlit as st
from folium import plugins

def get_pos(lat,lng):
    return lat,lng

CENTER_START = [39.0997, -94.5786]
ZOOM_START = 8

if "click" not in st.session_state:
    st.session_state["click"] = 0
if "center" not in st.session_state:
    st.session_state["center"] = CENTER_START

m = fl.Map(location= st.session_state["center"],
           zoom_start=min(4+4*st.session_state["click"],15),
           max_zoom=min(4+4*st.session_state["click"],15))

m.add_child(fl.LatLngPopup())

map = st_folium(m, height=500, width=750)

if map['last_clicked'] is not None:
    st.session_state["click"] += 1
    data = get_pos(map['last_clicked']['lat'],map['last_clicked']['lng'])
    st.session_state["center"] = data
    st.write(data)

# Function to create a Folium map centered on given coordinates
def create_map_with_zoom(lat, lon, zoom_level=10):

    lat == map['last_clicked']['lat']
    lon == map['last_clicked']['lng']

    # Create a Folium map centered at the provided coordinates
    map = fl.Map(location=[lat, lon], zoom_start=zoom_level)

    # Add a marker at the given coordinates
    fl.Marker(location=[lat, lon], popup=f"Location: ({lat}, {lon})").add_to(map)

    # Add a marker layer
    # fl.Marker([45.5236, -122.6750], popup='Portland').add_to(m)

    # Optionally, add more layers or plugins
    fl.TileLayer('Stamen Terrain').add_to(map)
    fl.TileLayer('Stamen Toner').add_to(map)
    fl.TileLayer('Stamen Watercolor').add_to(map)

    # Add layer control to switch between layers
    fl.LayerControl().add_to(map)

    return map
