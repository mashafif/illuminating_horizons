import os
import sys
import folium
import pandas as pd
import geopandas as gpd
from folium import GeoJson, FeatureGroup, Marker
import numpy as np
from folium.raster_layers import ImageOverlay
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image, ImageEnhance
from rasterio.warp import reproject, Resampling
from scipy.ndimage import gaussian_filter
import pyproj
import io
import base64


# Local
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '../..'))
sys.path.append(parent_dir)

from geodata_processing_kaoru import download_gdf
from getData import *
from state_dict import *

#Parameters
bearer = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6Im1hc2hoYWZpZiIsImV4cCI6MTcyNjAxNDQwMSwiaWF0IjoxNzIwODMwNDAxLCJpc3MiOiJFYXJ0aGRhdGEgTG9naW4ifQ.MffkJ_59FpDWDnES4xntvyRcSTfeVQDPJUwJnIbnk74zy9vbZA6iFz_GmAI5KdNwoVWKcKH_tfP0Byk63X2RwN89RUwShTTKvL86IH4hcc6ZHI3GpheS9M-Pi7_0BphHyDU3Aay7yIjGsd6LupfcGvMalnlnSU_cXCEOoZ_qfm19AQ0b37PwUQGci7snyz1pnb8NlDx-PZpL2-jLyPrndS9X-XDEEH_j2z5rh2nm-LXrq6IScZuAsn4_eD684CQVprT1VqngZOAAZTwL0yWhKVM7VwTr5wH0MXr2s5YKfJ1PnwqVFek0Vvrka4CjoJPZmKQkqAt-UHJVUgeZJM37pg"
PRODUCT = 'VNP46A4'  # Black Marble Annual Nighttime Lights with Cloud-Free Coverage
YEAR = 2023

# Control
state = 'IL'
#ISO
if len(state)==2:state=iso_to_state[state]
iso = state_to_iso[state]

################################################################ Geometry
USA = download_gdf(country='USA',resolution=1)
shape = USA[USA.NAME_1==state].geometry
# Get the bounding box of the shapefile
gdf = gpd.GeoDataFrame({'geometry': shape}, crs="EPSG:4326")
# Convert the GeoDataFrame to a mask that matches the data shape
shapes = [mapping(geom) for geom in gdf.geometry]

################################################################ Power plant coordinates
data_path = '../raw_data/uspvdb_v2_0_20240801.csv'
solar_data = pd.read_csv(data_path)
# Normalize power capacity based on all available data
global_norm = Normalize(vmin=solar_data['p_cap_ac'].min(), vmax=solar_data['p_cap_ac'].max())
# Filter to get California solar power plants
solar_locations = solar_data[solar_data['p_state'] == iso]

################################################################ Night data
data_array = getStateNightData(state)


################################################################ Masking Night data
data = data_array['NearNadir_Composite_Snow_Free'].values[0]

# Extracting the exact bounds from your data
min_lon, max_lon = data_array.x.values.min(), data_array.x.values.max()
min_lat, max_lat = data_array.y.values.min(), data_array.y.values.max()

# Use these bounds to create a transform
transform = from_bounds(min_lon, min_lat, max_lon, max_lat, data_array.sizes['x'], data_array.sizes['y'])
mask = False
try:
    mask = features.geometry_mask(shapes, transform=transform, invert=True, out_shape=(data_array.sizes['y'], data_array.sizes['x']))
except:
    pass

if mask is not False:
    #masked data
    data_masked = np.where(mask, data, np.nan)
else:
    data_masked = data

# Reproject night data to EPSG:3857
data_reprojected, x_reprojected, y_reprojected = reproject_to_epsg3857(
    data_masked, data_array.x.values, data_array.y.values
)

# Processing night data further
gdf_bbox = gpd.read_file('../raw_data/bounding_box.geojson')

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

################################################################ Preparing population data
# Load the GeoTIFF file for population data
pop_file_path = f'../raw_data/population/population_2020_US-{iso}.tif'

import rasterio
from rasterio.warp import calculate_default_transform

with rasterio.open(pop_file_path) as dataset:
    # Reproject the population data to EPSG:3857
    transform, width, height = calculate_default_transform(
        dataset.crs, 'EPSG:3857', dataset.width, dataset.height, *dataset.bounds)

    kwargs = dataset.meta.copy()
    kwargs.update({
        'crs': 'EPSG:3857',
        'transform': transform,
        'width': width,
        'height': height
    })

    reprojected_population = np.empty((height, width), dtype=np.float32)

    with rasterio.MemoryFile() as memfile:
        with memfile.open(**kwargs) as dst:
            reproject(
                source=rasterio.band(dataset, 1),
                destination=reprojected_population,
                src_transform=dataset.transform,
                src_crs=dataset.crs,
                dst_transform=transform,
                dst_crs='EPSG:3857',
                resampling=Resampling.nearest
            )

# Load the GeoDataFrame for the state boundary and reproject it to EPSG:3857
gdf_shape = gdf.to_crs("EPSG:3857")

# Create the mask using the state's boundary
shapes = [mapping(geom) for geom in gdf_shape.geometry]
mask = features.geometry_mask(shapes, transform=transform, invert=True, out_shape=reprojected_population.shape)

# Apply the mask to the reprojected population data
population_masked = np.where(mask, reprojected_population, np.nan)

print(population_masked)

# Replace any negative or NaN values with a small positive number to avoid issues with log1p
population_masked_clean = np.where(np.isnan(population_masked) | (population_masked < 0), 0, population_masked)

# Apply log1p to the cleaned data
population_log = np.log1p(population_masked_clean)

# Logarithmic scaling to handle narrow range
#population_log = np.log1p(population_masked)

# Aggressive clipping to remove outliers
clipped_population = np.clip(population_log, np.nanmin(population_log), np.nanpercentile(population_log, 90))

# Normalize the clipped data
population_normalized = (clipped_population - np.nanmin(clipped_population)) / (np.nanmax(clipped_population) - np.nanmin(clipped_population))

# Apply a more sensitive colormap
pop_colormap = plt.cm.inferno
population_colored = pop_colormap(population_normalized)
population_rgb = (population_colored[..., :3] * 255).astype(np.uint8)

# Create an alpha channel (fully transparent where NaN)
alpha_channel = np.where(np.isnan(population_masked), 0, 255).astype(np.uint8)
population_rgba = np.dstack((population_rgb, alpha_channel))

# Convert the RGBA array to an image
pop_image = Image.fromarray(population_rgba)

# Convert image to a base64 string to use with folium
pop_buffer = io.BytesIO()
pop_image.save(pop_buffer, format="PNG")
pop_img_str = base64.b64encode(pop_buffer.getvalue()).decode("utf-8")
pop_img_url = f"data:image/png;base64,{pop_img_str}"


####################### PLOT MAP

# File path for population data
pop_file_path = f'../raw_data/population/population_2020_US-{iso}.tif'

def raster_to_xr(file_path):
    with rasterio.open(file_path) as dataset:
        population_data = dataset.read(1)
        transform = dataset.transform
        rows, cols = np.indices(population_data.shape)
        xs, ys = rasterio.transform.xy(transform, rows, cols)
        xs = np.array(xs)[0, :]
        ys = np.array(ys)[:, 0]
    XR = xr.Dataset(
        {
            "population": (["y", "x"], population_data)
        },
        coords={
            "x": ("x", xs),
            "y": ("y", ys)
        }
    )
    return XR

# Load the GeoDataFrame for the bounding shape (e.g., California)
gdf_shape = gdf
gdf_shape = gdf_shape.to_crs("EPSG:4326")  # Ensure the CRS matches

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

# Create the overlay with reprojected bounds for light data
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
    gdf_shape.__geo_interface__,
    style_function=lambda x: {
        'color': 'gray',  # Color of the border line
        'weight': 1,  # Thickness of the border line
        'fillOpacity': 0  # Ensure no fill, only the outline
    }
).add_to(folium.map.LayerControl())

# Create a FeatureGroup for the solar power plant markers
solar_fg = FeatureGroup(name='California Solar Power Plants')

# Add markers for California solar power plants
for _, row in solar_locations.iterrows():
    size = 5 + 25 * global_norm(row['p_cap_ac'])
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
        popup=row['p_name'],
        icon=folium.DivIcon(html=icon_html)
    ).add_to(solar_fg)

# Add the GeoJSON layer for bounding boxes to the map
folium.GeoJson(
    gdf_bbox.to_json(),
    name='10km x 10km Bounding Box',
    style_function=lambda x: {
        'fillColor': 'White',  # Fill color
        'color': 'red',         # Border color
        'weight': 0,            # Border weight
        'fillOpacity': 0.0,     # Fill opacity
    }
).add_to(m)

gdf_boxes = gpd.read_file('../raw_data/random_grid.geojson')
# Add the GeoJSON layer for the bounding boxes to the map
geojson_layer = folium.GeoJson(
    gdf_boxes.to_json(),
    name='Bounding Boxes',
    style_function=lambda x: {
        'fillColor': 'grey',  # Transparent fill
        'color': 'grey',               # Border color
        'weight': 0,                  # Border weight
        'fillOpacity': 0.2,           # Fill opacity
    }
).add_to(m)

# Add the FeatureGroup for solar power plants to the map
solar_fg.add_to(m)

# Add the population density overlay to the map
pop_overlay = folium.raster_layers.ImageOverlay(
    image=pop_img_url,
    bounds=[[dataset.bounds.bottom, dataset.bounds.left], [dataset.bounds.top, dataset.bounds.right]],
    opacity=0.6,
    name="Population Density"
)
pop_overlay.add_to(m)

# Add layer control to toggle layers
folium.LayerControl().add_to(m)

# Display the map
m
