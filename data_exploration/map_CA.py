import os
import sys
import folium
import pandas as pd
import geopandas as gpd
import numpy as np
from folium import GeoJson, FeatureGroup, Marker
from folium.raster_layers import ImageOverlay
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image, ImageEnhance
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio import features
from rasterio.transform import from_bounds
from scipy.ndimage import gaussian_filter
import io
import base64
import rasterio
from shapely.geometry import mapping
from state_dict import alpha3_to_alpha2

data_precision = 3

# Local imports
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '../..'))
sys.path.append(parent_dir)

from geodata_processing_kaoru import download_gdf
from getData import getStateNightData_global, getStateNightData, reproject_to_epsg3857
from state_dict import iso_to_state, state_to_iso

# Parameters
PRODUCT = 'VNP46A4'
YEAR = 2023
#country='CONUS'
"""country='USA'
state = 'CA'
STATE = 'California'"""
country='GHA'
state = ''
STATE = ''

if state=='':
    STATE = country

# Geometry
if state != '':
    gdf = download_gdf(country=country, resolution=1)
    print(gdf)
    gdf = gdf[gdf.ISO_1 == f'US-{state}']#download_gdf(country=country, resolution=1)
    print(gdf)
    shape = gdf[gdf.NAME_1 == state].geometry
    if len(state) == 2:
        state = iso_to_state[state]
    iso = state_to_iso[state]
else:
    print('No state specified using the whole country')
    gdf = download_gdf(country=country, resolution=1)

shape = gdf.geometry
gdf = gpd.GeoDataFrame({'geometry': shape}, crs="EPSG:4326")
shapes = [mapping(geom) for geom in gdf.geometry]

# Power Plant Coordinates
data_path = '../raw_data/uspvdb_v2_0_20240801.csv'
solar_data = pd.read_csv(data_path)
global_norm = Normalize(vmin=solar_data['p_cap_ac'].min(), vmax=solar_data['p_cap_ac'].max())
if state != '':
    solar_locations = solar_data[solar_data['p_state'] == iso]
else:
    solar_locations = solar_data

# Night Data
if state != '':
    data_array = getStateNightData(state)
else:
    data_array = getStateNightData_global(country)
data = data_array['NearNadir_Composite_Snow_Free'].values[0]

# Masking Night Data
min_lon, max_lon = data_array.x.values.min(), data_array.x.values.max()
min_lat, max_lat = data_array.y.values.min(), data_array.y.values.max()
transform = from_bounds(min_lon, min_lat, max_lon, max_lat, data_array.sizes['x'], data_array.sizes['y'])

try:
    print('shape for night data')
    print(shapes)
    mask = features.geometry_mask(shapes, transform=transform, invert=True, out_shape=(data_array.sizes['y'], data_array.sizes['x']))
    data_masked = np.where(mask, data, np.nan)
except ValueError:
    data_masked = data

# Reproject Night Data
data_reprojected, x_reprojected, y_reprojected = reproject_to_epsg3857(data_masked, data_array.x.values, data_array.y.values)
data_reprojected = np.round(data_reprojected, decimals=data_precision)

# Custom Colormap
colors = [
    (0, 0, 0),  # Black
    (1, 1, 0),  # Bright yellow
    (1, 0.5, 0),  # Orange
    (1, 0, 0)   # Red
]
custom_cmap = LinearSegmentedColormap.from_list('enhanced_black_yellow_red', colors, N=100)

# Image Processing
data_filled = np.nan_to_num(data_reprojected, nan=0)
data_normalized = (data_filled - np.min(data_filled)) / (np.max(data_filled) - np.min(data_filled))
data_scaled = np.power(data_normalized, 0.15)
data_smoothed = gaussian_filter(data_scaled, sigma=1)
data_colored = custom_cmap(data_smoothed)[..., :3]
alpha_channel = np.where(data_filled > 0, 255, 0).astype(np.uint8)
data_colored_uint8 = np.dstack((data_colored * 255, alpha_channel)).astype(np.uint8)

image = Image.fromarray(data_colored_uint8)
image = ImageEnhance.Contrast(image).enhance(1.5)

buffer = io.BytesIO()
image.save(buffer, format="PNG")
img_url = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"


# Population Data Processing
if state != '':
    pop_file_path = f'../raw_data/population/population_2020_US-{iso}.tif'
else:
    if country == 'CONUS':
        pop_file_path = f'../raw_data/population/population_2020_CONUS_.tif'
    else:
        pop_file_path = f'../raw_data/population/population_2020_{alpha3_to_alpha2[country]}.tif'


with rasterio.open(pop_file_path) as dataset:
    transform, width, height = calculate_default_transform(dataset.crs, 'EPSG:3857', dataset.width, dataset.height, *dataset.bounds)
    reprojected_population = np.empty((height, width), dtype=np.float32)
    kwargs = dataset.meta.copy()
    kwargs.update({'crs': 'EPSG:3857', 'transform': transform, 'width': width, 'height': height})
    with rasterio.MemoryFile() as memfile:
        with memfile.open(**kwargs) as dst:
            reproject(source=rasterio.band(dataset, 1), destination=reprojected_population,
                      src_transform=dataset.transform, src_crs=dataset.crs,
                      dst_transform=transform, dst_crs='EPSG:3857', resampling=Resampling.nearest)

gdf_shape = gdf.to_crs("EPSG:3857")
'''shapes = [mapping(geom) for geom in gdf_shape.geometry]
mask = features.geometry_mask(shapes, transform=transform, invert=True, out_shape=reprojected_population.shape)'''
# Apply a small buffer to ensure islands and small features are included
buffered_shapes = [geom.buffer(0.01) for geom in gdf_shape.geometry]  # Adjust the buffer size as needed
mask = features.geometry_mask(
    [mapping(geom) for geom in buffered_shapes],
    transform=transform,
    invert=True,
    out_shape=reprojected_population.shape
)
population_masked = np.where(mask, reprojected_population, np.nan)
population_masked[population_masked == 0] = 0.01

# Replace any negative or NaN values with a small positive number to avoid issues with log1p
population_masked_clean = np.where(np.isnan(population_masked) | (population_masked < 0), 0.01, population_masked)

# Apply log1p to the cleaned data
population_log = np.log1p(population_masked_clean)

# Aggressive clipping to remove outliers
clipped_population = np.clip(population_log, np.nanmin(population_log), np.nanpercentile(population_log, 90))

# Normalize the clipped data
if np.nanmin(clipped_population) != np.nanmax(clipped_population):
    population_normalized = (clipped_population - np.nanmin(clipped_population)) / (np.nanmax(clipped_population) - np.nanmin(clipped_population))
else:
    population_normalized = clipped_population  # If no variation, keep the data as is

#Lighten
population_normalized = np.round(population_normalized, decimals=data_precision)

'''#pop_colormap = plt.cm.inferno
pop_colormap = plt.cm.Blues
#pop_colormap = plt.cm.twilight
population_colored = pop_colormap(population_normalized)
population_rgb = (population_colored[..., :3] * 255).astype(np.uint8)

alpha_channel = np.where(population_masked > 0, 255, 0).astype(np.uint8)

population_rgba = np.dstack((population_rgb, alpha_channel))

# Convert the RGBA array to an image
pop_image = Image.fromarray(population_rgba)

# Convert the image to a base64 string to use with folium
pop_buffer = io.BytesIO()
pop_image.save(pop_buffer, format="PNG")
pop_img_str = base64.b64encode(pop_buffer.getvalue()).decode("utf-8")
pop_img_url = f"data:image/png;base64,{pop_img_str}"'''

# Create a modified 'Blues' colormap with black at the lowest values
blues = plt.cm.Blues(np.linspace(0, 1, 256))
blues[:50, :] = np.array([0, 0, 0, 1])  # Set the first 50 values to black (background color)

# Create a new colormap object
custom_blues = LinearSegmentedColormap.from_list("custom_blues", blues)

# Apply the custom colormap
population_colored = custom_blues(population_normalized)
population_rgb = (population_colored[..., :3] * 255).astype(np.uint8)

# Create alpha channel based on the population data
alpha_channel = np.where(population_masked > 0, 255, 0).astype(np.uint8)
population_rgba = np.dstack((population_rgb, alpha_channel))

# Convert to image and save as PNG for Folium overlay
pop_image = Image.fromarray(population_rgba)
pop_buffer = io.BytesIO()
pop_image.save(pop_buffer, format="PNG")
pop_img_str = base64.b64encode(pop_buffer.getvalue()).decode("utf-8")
pop_img_url = f"data:image/png;base64,{pop_img_str}"




# ==============================================================================================================================
# Folium Map Creation ==========================================================================================================
# ==============================================================================================================================
m = folium.Map(location=[np.mean(y_reprojected), np.mean(x_reprojected)], zoom_start=8, crs="EPSG3857")

# Add Dark Matter Tile Layer
folium.TileLayer(
    tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
    name='CartoDB Dark Matter',
    attr='Map tiles by CartoDB, under CC BY 3.0. Data by OpenStreetMap, under ODbL.'
).add_to(m)


# Add Population Density Overlay with matching bounds
pop_overlay = folium.raster_layers.ImageOverlay(
    image=pop_img_url,
    bounds=[[y_reprojected.min(), x_reprojected.min()], [y_reprojected.max(), x_reprojected.max()]],
    opacity=0.5,
    name="Population Density"
)
pop_overlay.add_to(m)

# Apply consistent bounds to both overlays
light_overlay = ImageOverlay(
    image=img_url,
    bounds=[[y_reprojected.min(), x_reprojected.min()], [y_reprojected.max(), x_reprojected.max()]],
    opacity=0.6,
    name='Light Intensity',
    interactive=True,
    cross_origin=False,
    zindex=1,
    id='light-overlay'
)
light_overlay.add_to(m)

#########ROAD
# Load the road GeoJSON data
'''road_gdf = gpd.read_file('../raw_data/additional/road.geojson')

# Reproject to EPSG:3857
road_gdf = road_gdf.to_crs(epsg=3857)
# Define the bounds and resolution for the raster based on the California shape
min_lon, max_lon = gdf.total_bounds[0], gdf.total_bounds[2]
min_lat, max_lat = gdf.total_bounds[1], gdf.total_bounds[3]
# Define the transform and dimensions for the raster
transform = from_bounds(min_lon, min_lat, max_lon, max_lat, 1000, 1000)  # Adjust the resolution as needed
out_shape = (500, 500)

# Rasterize the road data
road_raster = features.rasterize(
    [(geom, 1) for geom in road_gdf.geometry],
    out_shape=out_shape,
    transform=transform,
    fill=0,  # Background value
    all_touched=True,
    dtype='uint8'
)

# Mask the raster with California's borders (already reprojected to EPSG:3857)
shapes = [mapping(geom) for geom in gdf.geometry]
mask = features.geometry_mask(shapes, transform=transform, invert=True, out_shape=out_shape)

road_raster_masked = np.where(mask, road_raster, 0)  # Mask with 0 for background

# Save the raster to a GeoTIFF file
output_tiff = '../raw_data/rasterized_road_CA.tif'
with rasterio.open(
    output_tiff,
    'w',
    driver='GTiff',
    height=road_raster_masked.shape[0],
    width=road_raster_masked.shape[1],
    count=1,
    dtype=road_raster_masked.dtype,
    crs=gdf.crs,
    transform=transform
) as dst:
    dst.write(road_raster_masked, 1)

# Open the GeoTIFF file to get the bounds
with rasterio.open(output_tiff) as src:
    bounds = [[src.bounds.bottom, src.bounds.left], [src.bounds.top, src.bounds.right]]

# Convert the raster to an image
road_image = Image.fromarray(road_raster_masked * 255)  # Convert to 8-bit image for display

# Generate the image URL for the overlay
buffer = io.BytesIO()
road_image.save(buffer, format="PNG")
img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
img_url = f"data:image/png;base64,{img_str}"

# Add the rasterized road data as an overlay to the Folium map
road_overlay = ImageOverlay(
    image=img_url,
    bounds=bounds,
    opacity=0.6,  # Adjust opacity as needed
    name='Roads'
)

road_overlay.add_to(m)'''

#########ROAD


# IRRIDIANCE############################################################################################################################################
# Load the image as a numpy array
image_path = f'../raw_data/additional/{STATE}_cont_radiation_raster.png'
image = Image.open(image_path).convert("RGBA")  # Ensure the image is in RGBA format
image_array = np.array(image)

# Define the bounds and transform for the mask using the original image's bounds
min_lon, max_lon = gdf.total_bounds[0], gdf.total_bounds[2]
min_lat, max_lat = gdf.total_bounds[1], gdf.total_bounds[3]
transform = from_bounds(min_lon, min_lat, max_lon, max_lat, image_array.shape[1], image_array.shape[0])
# Create the mask from the shape
shapes = [mapping(geom) for geom in gdf.geometry]
mask = features.geometry_mask(shapes, transform=transform, invert=True, out_shape=image_array.shape[:2])
# Apply the mask to the image by setting areas outside state to transparent
cropped_image_array = np.dstack((image_array[:, :, :3], mask.astype(np.uint8) * 255))
# Convert back to an image and save (optional)
cropped_image = Image.fromarray(cropped_image_array)
cropped_image.save(f'../raw_data/additional/{STATE}_cont_radiation_raster_cropped_corrected.png')
# Use the cropped image in your Folium map
i_overlay = ImageOverlay(
    image=f'../raw_data/additional/{STATE}_cont_radiation_raster_cropped_corrected.png',
    bounds=[[min_lat, min_lon], [max_lat, max_lon]],
    opacity=0.6,
    name='Irradiance',
    interactive=True,
    cross_origin=False,
    zindex=1,
    id='solar'
)
i_overlay.add_to(m)
########################################################################################################################################################
# IRRIDIANCE############################################################################################################################################
# Load the image as a numpy array
image_path = f'/Users/kaorukure/code/mashafif/illuminating_horizons/raw_data/additional/{STATE}_disc_landcover_raster.png'
image = Image.open(image_path).convert("RGBA")  # Ensure the image is in RGBA format
image_array = np.array(image)

# Define the bounds and transform for the mask using the original image's bounds
min_lon, max_lon = gdf.total_bounds[0], gdf.total_bounds[2]
min_lat, max_lat = gdf.total_bounds[1], gdf.total_bounds[3]
transform = from_bounds(min_lon, min_lat, max_lon, max_lat, image_array.shape[1], image_array.shape[0])
# Create the mask from the shape
shapes = [mapping(geom) for geom in gdf.geometry]
mask = features.geometry_mask(shapes, transform=transform, invert=True, out_shape=image_array.shape[:2])
# Apply the mask to the image by setting areas outside state to transparent
cropped_image_array = np.dstack((image_array[:, :, :3], mask.astype(np.uint8) * 255))
# Convert back to an image and save (optional)
cropped_image = Image.fromarray(cropped_image_array)
cropped_image.save(f'../raw_data/additional/{STATE}_disc_landcover_raster_corrected.png')
# Use the cropped image in your Folium map
lc_overlay = ImageOverlay(
    image=f'../raw_data/additional/{STATE}_disc_landcover_raster_corrected.png',
    bounds=[[min_lat, min_lon], [max_lat, max_lon]],
    opacity=0.6,
    name='Terrain',
    interactive=True,
    cross_origin=False,
    zindex=1,
    id='terrain'
)
lc_overlay.add_to(m)

########################################################################################################################################################
# PREDICTION
p_overlay = ImageOverlay(
    image=f'/Users/kaorukure/code/mashafif/illuminating_horizons/raw_data/additional/{STATE}_cont_raster_strict.png',
    bounds=[[y_reprojected.min(), x_reprojected.min()], [y_reprojected.max(), x_reprojected.max()]],
    opacity=0.6,
    name='Prediction_strict',
    interactive=True,
    cross_origin=False,
    zindex=1,
    id='Prediction_strict'
)
p_overlay.add_to(m)

# PREDICTION
ps_overlay = ImageOverlay(
    image=f'/Users/kaorukure/code/mashafif/illuminating_horizons/raw_data/additional/{STATE}_cont_raster.png',
    bounds=[[y_reprojected.min(), x_reprojected.min()], [y_reprojected.max(), x_reprojected.max()]],
    opacity=0.6,
    name='Prediction_score',
    interactive=True,
    cross_origin=False,
    zindex=1,
    id='Prediction_score'
)
ps_overlay.add_to(m)


if country in ['USA','CONUS']:
    # Add Solar Power Plant Markers
    solar_fg = FeatureGroup(name=f'{state} Solar Power Plants')
    for _, row in solar_locations.iterrows():
        size = 5 + 15 * global_norm(row['p_cap_ac'])
        color = plt.cm.Reds(global_norm(row['p_cap_ac']))
        icon_html = f"""
        <div style="
            background-color: rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, 0.6);
            border-radius: 50%;
            width: {size}px;
            height: {size}px;
            border: 2px solid rgba(255, 255, 255, 0.6);
            box-shadow: 0 0 5px rgba(255, 255, 255, 0.7);
        "></div>
        """
        Marker(location=[row['ylat'], row['xlong']], popup=row['p_name'], icon=folium.DivIcon(html=icon_html)).add_to(solar_fg)
    solar_fg.add_to(m)





#SUG
# Define marker locations
# Define marker locations and corresponding images
locations = [
    {'lat': 6.608169536418913, 'lon': 0.2628256317421865, 'popup': 'Suggestion 1', 'image': '../suggestion1.png'},
    {'lat': 9.581142569773737, 'lon': -0.6013499001133938, 'popup': 'Suggestion 2', 'image': '../suggestion2.png'},
    {'lat': 7.013574944365997, 'lon': -1.4655254926660755, 'popup': 'Suggestion 3', 'image': '../suggestion3.png'}
]

Suggestion_fg = FeatureGroup(name='Suggested points')

# Teardrop SVG style with black fill and yellow stroke
teardrop_icon = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="40" viewBox="0 0 24 24" fill="black" stroke="yellow" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-map-pin">'
    '<path d="M21 10c0 6.075-9 13-9 13S3 16.075 3 10a9 9 0 1118 0z"></path>'
    '<circle cx="12" cy="10" r="3"></circle>'
    '</svg>'
)

# Add custom teardrop markers to the FeatureGroup
for loc in locations:
    # Create the HTML for the popup including the image
    popup_html = f'<div style="width: 620px;"><h4>{loc["popup"]}</h4><img src="{loc["image"]}" width="600px" height="600px"></div>'
    popup = folium.Popup(popup_html, max_width=600)

    # Add the marker with the popup to the map
    icon = folium.DivIcon(html=teardrop_icon)
    folium.Marker(
        location=[loc['lat'], loc['lon']],
        popup=popup,
        icon=icon
    ).add_to(Suggestion_fg)

# Add the FeatureGroup to the map
Suggestion_fg.add_to(m)





# Add Layer Control with custom position on the left
folium.LayerControl(position='topright').add_to(m)

# Save the map to an HTML file
output_file = f'MAP_{country}'
if state != '': output_file += f'-{state}'
output_file += '3.html'
m.save(output_file)

# Optionally open the map in the default web browser
import webbrowser
webbrowser.open(output_file)
