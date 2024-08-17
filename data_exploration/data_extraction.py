import os
import datetime

import colorcet as cc
import contextily as cx
import geopandas
import matplotlib.pyplot as plt
import pandas as pd
from bokeh.plotting import figure, output_notebook, show
from bokeh.models import HoverTool, Title

from blackmarble.extract import bm_extract
from blackmarble.raster import bm_raster

%load_ext autoreload
%autoreload 2

plt.rcParams["figure.figsize"] = (18, 10)

# An environment variable can obfuscate to secure a secret
import os

bearer = os.getenv("BLACKMARBLE_TOKEN")

# Define Region of Interest
gdf = geopandas.read_file(
    "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_GHA_1.json.zip"
)
gdf.explore(tiles="CartoDB dark_matter")

# Daily data: raster for February 5, 2021
VNP46A2_20210205 = bm_raster(
    gdf, product_id="VNP46A2", date_range="2021-02-05", bearer=bearer
)
VNP46A2_20210205

fig, ax = plt.subplots(figsize=(16, 8))

VNP46A2_20210205["Gap_Filled_DNB_BRDF-Corrected_NTL"].sel(
    time="2021-02-05"
).plot.pcolormesh(
    ax=ax,
    cmap=cc.cm.bmy,
    robust=True,
)
cx.add_basemap(ax, crs=gdf.crs.to_string())

ax.text(
    0,
    -0.1,
    "Source: NASA Black Marble VNP46A2",
    ha="left",
    va="center",
    transform=ax.transAxes,
    fontsize=10,
    color="black",
    weight="normal",
)
ax.set_title("Ghana: NTL Radiance on Feb 5 2021", fontsize=20);
