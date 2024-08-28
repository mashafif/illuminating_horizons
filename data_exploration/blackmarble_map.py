import os
import sys
import glob
import io
import base64
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from PIL import Image, ImageEnhance
from shapely.geometry import mapping
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap, Normalize

import folium
from folium import GeoJson, FeatureGroup, Marker
from folium.raster_layers import ImageOverlay

import rasterio
from rasterio import features
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling

import xarray as xr
from blackmarble.raster import bm_raster

from state_dict import *


# Local
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)
from illuminating.data_preparation.geodata_processing import download_gdf

# Constants and parameters
bearer = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6Im1hc2hoYWZpZiIsImV4cCI6MTcyNjAxNDQwMSwiaWF0IjoxNzIwODMwNDAxLCJpc3MiOiJFYXJ0aGRhdGEgTG9naW4ifQ.MffkJ_59FpDWDnES4xntvyRcSTfeVQDPJUwJnIbnk74zy9vbZA6iFz_GmAI5KdNwoVWKcKH_tfP0Byk63X2RwN89RUwShTTKvL86IH4hcc6ZHI3GpheS9M-Pi7_0BphHyDU3Aay7yIjGsd6LupfcGvMalnlnSU_cXCEOoZ_qfm19AQ0b37PwUQGci7snyz1pnb8NlDx-PZpL2-jLyPrndS9X-XDEEH_j2z5rh2nm-LXrq6IScZuAsn4_eD684CQVprT1VqngZOAAZTwL0yWhKVM7VwTr5wH0MXr2s5YKfJ1PnwqVFek0Vvrka4CjoJPZmKQkqAt-UHJVUgeZJM37pg"
PRODUCT = 'VNP46A4'  # Black Marble Annual Nighttime Lights with Cloud-Free Coverage
YEAR = 2023
state = 'Texas'
iso = state_to_iso[state]

# Shape data
USA = download_gdf(country='USA',resolution=1)
shape = USA[USA.NAME_1==state].geometry

# Get the bounding box of the shapefile
gdf = gpd.GeoDataFrame({'geometry': shape}, crs="EPSG:4326")

# Convert the GeoDataFrame to a mask that matches the data shape
shapes = [mapping(geom) for geom in gdf.geometry]

# Load the CSV file and filter for US power plants
data_path = '../raw_data/uspvdb_v2_0_20240801.csv'
solar_data = pd.read_csv(data_path)

# Normalize power capacity based on all available data
global_norm = Normalize(vmin=solar_data['p_cap_ac'].min(), vmax=solar_data['p_cap_ac'].max())

# Filter to get California solar power plants
solar_locations = solar_data[solar_data['p_state'] == iso]


def getStateNightData(state,year=2023, force_dl = False):
    print(state)
    if len(state)!=2:
        state=state_to_iso[state]
    state_name = iso_to_state[state]
    #Check if state is in local
    print(f'../raw_data/bm/blackmarble_{year}_US-{iso}.nc')
    F = glob.glob(f'../raw_data/bm/blackmarble_{year}_US-{iso}.nc')
    if len(F) == 1 and not force_dl:
        print(f'Found local data for {state_name}, reading to memory')
        D = xr.open_dataset(F[0])
    else:
        if force_dl:
            print(f'force_dl is True, force downloading data for {state_name}')
        else:
            print(f'No local data for {state_name}, downloading...')
        D = bm_raster(
        gdf,
        product_id="VNP46A4",
        date_range=pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="YS"),
        bearer=bearer
        )
    return D

nightimg = getStateNightData(state)

# Preprocessing
data_array = nightimg

#Taking data out of container
data = data_array['NearNadir_Composite_Snow_Free'].values[0]
shapes = [mapping(geom) for geom in gdf.geometry]

# Extracting the exact bounds from your data
min_lon, max_lon = data_array.x.values.min(), data_array.x.values.max()
min_lat, max_lat = data_array.y.values.min(), data_array.y.values.max()

# Use these bounds to create a transform
transform = from_bounds(min_lon, min_lat, max_lon, max_lat, data_array.sizes['x'], data_array.sizes['y'])
mask = features.geometry_mask(shapes, transform=transform, invert=True, out_shape=(data_array.sizes['y'], data_array.sizes['x']))

#masked data
data_masked = np.where(mask, data, np.nan)

# Reprojecting function using pyproj
def reproject_to_epsg3857(data_array, x, y):
    # Define the source and target CRS
    src_crs = pyproj.CRS("EPSG:4326")
    dst_crs = pyproj.CRS("EPSG:3857")

    # Create the transform for the target CRS
    x_res = (x[-1] - x[0]) / (len(x) - 1)
    y_res = (y[-1] - y[0]) / (len(y) - 1)
    transform = from_bounds(x[0], y[-1], x[-1], y[0], len(x), len(y))

    # Output arrays for the reprojected data
    dst_data = np.empty_like(data_array)
    dst_x = np.linspace(x[0], x[-1], len(x))
    dst_y = np.linspace(y[0], y[-1], len(y))

    # Reproject
    reproject(
        source=data_array,
        destination=dst_data,
        src_transform=transform,
        src_crs=src_crs,
        dst_crs=dst_crs,
        resampling=Resampling.nearest
    )
    return dst_data, dst_x, dst_y

# Reproject the data to EPSG:3857
data_reprojected, x_reprojected, y_reprojected = reproject_to_epsg3857(
    data_masked, data_array.x.values, data_array.y.values
)

# Define a custom colormap with a stronger emphasis on red
colors = [
    (0, 0, 0),  # Black
    (1, 1, 0),  # Bright yellow
    (1, 0.5, 0),  # Orange
    (1, 0, 0)   # Red
]
n_bins = 100  # Number of bins for interpolation
cmap_name = 'enhanced_black_yellow_red'
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

# Replace NaN values with a small value that will still be considered transparent
data_filled = np.nan_to_num(data_reprojected, nan=0)

# Normalize the reprojected data
data_min = np.min(data_filled)  # Use np.min instead of np.nanmin
data_max = np.max(data_filled)  # Use np.max instead of np.nanmax
data_normalized = (data_filled - data_min) / (data_max - data_min)

# Apply gamma correction (nonlinear scaling) to emphasize lower values and push more to red
gamma = 0.15  # Lower gamma value to push more data into the red range
data_scaled = np.power(data_normalized, gamma)

# Smooth the edges using Gaussian blur
data_smoothed = gaussian_filter(data_scaled, sigma=1)  # Adjust sigma for more or less smoothing

# Apply the custom colormap, ensuring the brightest areas are red
data_colored = custom_cmap(data_smoothed)[..., :3]  # Keep RGB channels only

# Add alpha channel for transparency
alpha_channel = np.where(data_filled > 0, 255, 0).astype(np.uint8)  # Transparent outside California
data_colored_uint8 = np.dstack((data_colored * 255, alpha_channel)).astype(np.uint8)  # RGBA

# Convert to image URL using folium's utilities
image = Image.fromarray(data_colored_uint8)

# Optional: Further enhance the contrast of the image using PIL
image = ImageEnhance.Contrast(image).enhance(1.5)  # Adjust contrast as needed

buffer = io.BytesIO()
image.save(buffer, format="PNG")
img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
img_url = f"data:image/png;base64,{img_str}"

# Create the folium map centered at the approximate center
center_lat = np.mean(y_reprojected)
center_lon = np.mean(x_reprojected)
m = folium.Map(location=[center_lat, center_lon], zoom_start=8, crs="EPSG3857")

# Add the CartoDB Dark Matter tile layer for a dark background
folium.TileLayer(
    tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
    name='CartoDB Dark Matter',
    attr='Map tiles by CartoDB, under CC BY 3.0. Data by OpenStreetMap, under ODbL.'
).add_to(m)

# Create the overlay with reprojected bounds
overlay = ImageOverlay(
    image=img_url,
    bounds=[[y_reprojected.min(), x_reprojected.min()], [y_reprojected.max(), x_reprojected.max()]],
    opacity=0.6,
    name='Light Intensity',
    interactive=True,
    cross_origin=False,
    zindex=1,
    id='light-overlay'
)
overlay.add_to(m)

# Add a thin line around the California border directly to the map
GeoJson(
    gdf.__geo_interface__,
    style_function=lambda x: {
        'color': 'gray',  # Color of the border line
        'weight': 1,  # Thickness of the border line
        'fillOpacity': 0  # Ensure no fill, only the outline
    }
).add_to(folium.map.LayerControl())

# Create a FeatureGroup for the solar power plant markers
solar_fg = FeatureGroup(name=f'{state} Solar Power Plants')

# Add markers for California solar power plants
for _, row in solar_locations.iterrows():
    # Scale the size of the marker based on global power capacity
    size = 5 + 20 * global_norm(row['p_cap_ac'])  # Adjust multiplier for desired size scaling
    # Adjust the color based on global power capacity (higher capacity = more red)
    color = plt.cm.Reds(global_norm(row['p_cap_ac']))
    icon_html = f"""
    <div style="
        background-color: rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, 0.6);  /* Scaled color with transparency */
        border-radius: 50%;  /* Makes the icon circular */
        width: {size}px;  /* Icon width */
        height: {size}px;  /* Icon height */
        border: 2px solid rgba(255, 255, 255, 0.6);  /* White border with transparency */
        box-shadow: 0 0 5px rgba(255, 255, 255, 0.7);  /* Soft shadow for better visibility */
    "></div>
    """
    Marker(
        location=[row['ylat'], row['xlong']],
        popup=row['p_name'],  # Assuming 'name' column exists for the plant names
        icon=folium.DivIcon(html=icon_html)
    ).add_to(solar_fg)

# Add the FeatureGroup to the map
solar_fg.add_to(m)

# Add layer control and display the map
folium.LayerControl().add_to(m)
m
