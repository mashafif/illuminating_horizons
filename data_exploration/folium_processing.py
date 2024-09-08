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

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(\
            os.path.abspath(__file__)),"../"))

RAW_DATA_PATH = os.path.join(ROOT_PATH,"raw_data")

def process_folium():
    data_precision = 1

    # Local imports
    #parent_dir = os.path.abspath(os.path.join(os.getcwd(), '../..'))
    #sys.path.append(parent_dir)

    #from geodata_processing_kaoru import download_gdf
    from illuminating.data_preparation.geodata_processing import download_gdf
    from illuminating.data_preparation.getData_sat import getStateNightData_global, getStateNightData, reproject_to_epsg3857
    from state_dict import iso_to_state, state_to_iso

    # Parameters
    bearer = "your_bearer_token_here"
    PRODUCT = 'VNP46A4'
    YEAR = 2023
    country='USA'
    #country='GHA'
    state = ''

    # Geometry
    if state != '':
        gdf = download_gdf(country=country, resolution=1)
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
    data_path = os.path.join(RAW_DATA_PATH,'uspvdb_v2_0_20240801.csv')
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
        pop_file_path = os.path.join(RAW_DATA_PATH,f'population/population_2020_US-{iso}.tif')
    else:
        if country == 'CONUS':
            pop_file_path = os.path.join(RAW_DATA_PATH,f'population/population_2020_CONUS_.tif')
        else:
            pop_file_path = os.path.join(RAW_DATA_PATH,f'population/population_2020_{alpha3_to_alpha2[country]}.tif')


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
        opacity=0.6,
        name="Population Density"
    )
    pop_overlay.add_to(m)


    # Apply consistent bounds to both overlays
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

    ############
    '''pred_gdf = gpd.read_file('/Users/kaorukure/code/mashafif/illuminating_horizons/raw_data/additional/GHA_prediction_result.geojson')
    pred_layer = GeoJson(
        pred_gdf.to_json(),
        name='Prediction',
        style_function=lambda x: {
            'color': 'red',  # Set the color for roads
            'weight': 2,      # Set the thickness of the road lines
            'opacity': 0.6    # Set the opacity of the road lines
        }
    )
    pred_layer.add_to(m)'''

    ##########

    # Add Solar Power Plant Markers
    solar_fg = FeatureGroup(name='California Solar Power Plants')
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

    # Add Layer Control with custom position on the left
    folium.LayerControl(position='topright').add_to(m)

    # Save the map to an HTML file
    output_file = f'MAP_{country}.html'
    m.save(output_file)

    # Optionally open the map in the default web browser
    import webbrowser
    webbrowser.open(output_file)

if __name__ == '__main__':
   # start_time = time.time()
    try:
        #preprocessing_and_train()
        process_folium()


    except:
        import sys
        import traceback

        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
