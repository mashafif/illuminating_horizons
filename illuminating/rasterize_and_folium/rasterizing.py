from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import os
from PIL import Image
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import LineString
import folium
from folium.raster_layers import ImageOverlay
import pdb
from matplotlib.colors import ListedColormap



ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(\
            os.path.abspath(__file__)),"../.."))

RAW_DATA_PATH = os.path.join(ROOT_PATH,"raw_data")
IMG_PATH = os.path.join(RAW_DATA_PATH,"image_and_html")


def hex_to_rgb(value):
    """Convert hexadecimal to RGB tuple normalized between 0 and 1."""
    value = value.lstrip('#')
    lv = len(value)
    return np.array([int(value[i:i + lv // 3], 16) / 255.0 for i in range(0, lv, lv // 3)])



def create_cont_raster(normalized_raster,
                       rgb:list=["#00FF00","#FFFF00","#FFA500","#FF0000"],
                       color_range:list=[0.5,0.75,0.85,0.95],
                       filename = "continous_raster.png"):

    first_rgb = hex_to_rgb(rgb[0])
    second_rgb = hex_to_rgb(rgb[1])
    third_rgb = hex_to_rgb(rgb[2])
    fourth_rgb = hex_to_rgb(rgb[3])


    num_colors = 256
    colors = np.zeros((num_colors, 4))  # Initialize an RGBA array

    # Calculate the indices for the color transitions
    first_start = int(color_range[0] * num_colors)  # 25%
    second_start = int(color_range[1] * num_colors) # 50%
    third_start = int(color_range[2] * num_colors)  # 75%
    fourth_start = int(color_range[3] * num_colors) # 90%

    for i in range(num_colors):
        if i < first_start:
            colors[i] = np.append(first_rgb, 0)  # Transparent up to 25%
        elif i < second_start:
            interpolation = (i - first_start) / (second_start - first_start)
            colors[i, :3] = first_rgb + (second_rgb - first_rgb) * interpolation
            colors[i, 3] = interpolation  # Gradual increase in opacity
        elif i < third_start:
            interpolation = (i - second_start) / (third_start - second_start)
            colors[i, :3] = second_rgb + (third_rgb - second_rgb) * interpolation
            colors[i, 3] = 1  # Full opacity from 50% onwards
        elif i < fourth_start:
            interpolation = (i - third_start) / (fourth_start - third_start)
            colors[i, :3] = third_rgb + (fourth_rgb - third_rgb) * interpolation
            colors[i, 3] = 1  # Full opacity
        else:
            colors[i, :3] = fourth_rgb
            colors[i, 3] = 1  # Bright red with full opacity at the highest levels
    print(colors)
    print(type(colors))
    custom_cmap = LinearSegmentedColormap.from_list("custom_four_color_map", colors)
        # Apply the custom colormap to get RGBA values
    colored_data = custom_cmap(normalized_raster)
    breakpoint()
    if not isinstance(colored_data, np.ndarray):
        raise ValueError("Expected colored_data to be an ndarray but got {}".format(type(colored_data)))

    # Convert to 8-bit by scaling up to 255
    rgba_image = (colored_data[:, :, :4] * 255).astype(np.uint8)  # Assuming colored_data includes RGBA channels correctly


    # Convert to PIL Image and save
    img = Image.fromarray(rgba_image, 'RGBA')
    png_path = os.path.join(IMG_PATH,filename)
    img.save(png_path)


def gpd_to_raster(gdf,border_gdf,filename = "pred_raster.png",
                  pixel_size=5,target_feature="have_plant_proba",discrete=False):

    pixel_size = 111/5  # Approximate size of 5 km

    bounds = border_gdf.total_bounds  # Set region bounds
    width = int((bounds[2] - bounds[0]) / pixel_size)
    height = int((bounds[3] - bounds[1]) / pixel_size)
    transform = from_origin(bounds[0], bounds[3], pixel_size, pixel_size)

    # Rasterize the points with 'have_plant' values


    # Buffer the points in GeoDataFrame, `distance` depends on your spatial resolution needs
    gdf['geometry'] = gdf.geometry.buffer(6000)# Choose an appropriate distance

    def get_features(gdf):
        for _, row in gdf.iterrows():
            geometry = row['geometry']
            value = row[target_feature]
            yield geometry, value
    print(gdf.head())
    # Then rasterize the buffered geometries
    raster = rasterize(
        get_features(gdf),
        out_shape=(height, width),
        transform=transform,
        fill=0,  # Value for non-point areas
        dtype=float
    )
    print(f"Minimum raster {np.min(raster)}")
    print(f"Maximum raster {np.max(raster)}")

    if not discrete:
        data_normalized = (raster - np.min(raster)) / (np.max(raster) - np.min(raster)) * 255
        data_normalized = data_normalized.astype(np.uint8)
        print(f"Minimum normalized raster{np.min(data_normalized)}")
        print(f"Maximum normalized raster{np.max(data_normalized)}")
    else:
        data_normalized = raster.astype(np.uint8)

    # Save as PNG
    img = Image.fromarray(data_normalized)
    png_path = os.path.join(IMG_PATH,filename)
    img.save(png_path)
    return data_normalized


LANDUSE_LIST =  [
    '#000000', '#ffff64', '#aaf0f0','#dcf064', '#c8c864', '#006400', '#00a000',
    '#003c00','#285000', '#788200', '#8ca000', '#be9600', '#966400', '#ffb432',
    '#ffdcd2', '#ffebaf', '#00785a', '#009678', '#00dc82', '#c31400',
    '#fff5d7', '#0046c8', '#ffffff'
]

def create_discrete_raster(normalized_raster,
                       rgb:list=LANDUSE_LIST,
                       filename = "landcover_dist_raster.png"):

    # Create a ListedColormap
    discrete_cmap = ListedColormap(rgb)

    # Assuming `raster` is your data array with values from 0 to 220, stepped by 10
    # Normalize indices for colormap: map range 0-220 to indices 0-22
    indices = np.clip((normalized_raster / 10).astype(int), 0, 22)  # Clip to ensure indices stay in range

    # Apply the colormap to get RGBA values
    colored_data = discrete_cmap(indices)

    # Convert the data to an 8-bit format for image creation
    rgba_image = (colored_data * 255).astype(np.uint8)


    # Convert to PIL Image and save
    img = Image.fromarray(rgba_image, 'RGBA')
    png_path = os.path.join(IMG_PATH,filename)
    img.save(png_path)


def check_linestrings_in_column(gdf, column_name):
    """
    Check if all geometries in the specified column of a GeoDataFrame are LineString.

    Parameters:
    gdf (GeoDataFrame): The GeoDataFrame to check.
    column_name (str): The name of the column containing geometries.

    Returns:
    bool: True if all geometries are LineString, False otherwise.
    """
    return gdf[column_name].apply(lambda geom: isinstance(geom, LineString)).all()

def folium_show(png_path,border_gdf,opacity=0.6,layername="Prediction",
                filename="map_folium.html"):

    bounds=border_gdf.total_bounds
    raster_bounds = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]  # [southwest, northeast]
    png_path = os.path.join(IMG_PATH,png_path)
# Create a Folium map centered at the middle of the bounds
    m = folium.Map(location=[(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2], zoom_start=10)

    # Add the raster image as an overlay
    ImageOverlay(
        image=png_path,
        bounds=raster_bounds,
        opacity=opacity,  # Full opacity as transparency is handled in the image
        interactive=True,
        name=layername,
        crs=
    ).add_to(m)

    # Add layer control to toggle layers
    folium.LayerControl().add_to(m)

    # Save the map
    map_path = os.path.join(IMG_PATH,filename)
    m.save(map_path)
    return m
