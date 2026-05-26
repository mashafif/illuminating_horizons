"""
Illuminating Horizons — Solar Power Station Siting Dashboard
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import streamlit as st
import altair as alt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE          = os.path.dirname(os.path.abspath(__file__))
ROOT          = os.path.dirname(BASE)
USPVDB_PATH   = os.path.join(ROOT, "raw_data", "uspvdb_v2_0_20240801.csv")
GADM_DIR      = os.path.join(BASE, "gadm_gdf")
TRAIN_GEOJSON = os.path.join(ROOT, "data_exploration", "powerplant_to_pop_powerlines.geojson")

# ── Constants ─────────────────────────────────────────────────────────────────
STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY",
}
ABBR_TO_NAME = {v: k for k, v in STATE_ABBR.items()}

CAP_BINS   = [0, 1, 10, 50, float("inf")]
CAP_LABELS = ["< 1 MW", "1–10 MW", "10–50 MW", "> 50 MW"]
CAP_COLORS = ["#3498db", "#f39c12", "#e74c3c", "#8e44ad"]

MODEL_FEATURES = ["distance_to_largepop", "distance_to_powerlines", "lat", "lon"]


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data
def load_plants() -> pd.DataFrame:
    df = pd.read_csv(USPVDB_PATH, encoding="utf-8-sig")
    keep = ["case_id", "p_name", "p_state", "p_county",
            "ylat", "xlong", "p_year", "p_tech_pri", "p_cap_ac", "p_type"]
    df = df[keep].copy()
    df.rename(columns={"ylat": "lat", "xlong": "lon"}, inplace=True)
    df.dropna(subset=["lat", "lon", "p_cap_ac"], inplace=True)
    df["cap_bin"] = pd.cut(
        df["p_cap_ac"], bins=CAP_BINS, labels=CAP_LABELS, right=False
    )
    return df


@st.cache_data
def load_state_gdf(abbr: str) -> gpd.GeoDataFrame | None:
    path = os.path.join(GADM_DIR, f"gdf_US-{abbr}.geojson")
    return gpd.read_file(path) if os.path.exists(path) else None


@st.cache_data
def load_training_data() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(TRAIN_GEOJSON)
    gdf["lat"] = gdf.geometry.y
    gdf["lon"] = gdf.geometry.x
    return gdf


@st.cache_resource
def get_ca_model():
    """Train the CA siting model once per session."""
    gdf = load_training_data()
    X = gdf[MODEL_FEATURES].values
    y = gdf["have_plant"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe = Pipeline([
        ("scaler", RobustScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=150, max_depth=12, random_state=42, n_jobs=-1
        )),
    ])
    pipe.fit(X_tr, y_tr)

    report      = classification_report(y_te, pipe.predict(X_te), output_dict=True)
    auc         = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
    importances = dict(zip(MODEL_FEATURES, pipe.named_steps["clf"].feature_importances_))
    all_proba   = pipe.predict_proba(X)[:, 1]

    return pipe, report, auc, importances, all_proba


# ── Map helpers ───────────────────────────────────────────────────────────────

def _cap_color(cap_ac: float) -> str:
    if cap_ac < 1:
        return CAP_COLORS[0]
    elif cap_ac < 10:
        return CAP_COLORS[1]
    elif cap_ac < 50:
        return CAP_COLORS[2]
    return CAP_COLORS[3]


def build_map(
    state_abbr: str,
    plants_df: pd.DataFrame,
    state_gdf: gpd.GeoDataFrame,
    show_heatmap: bool,
    show_training: bool,
    train_gdf: gpd.GeoDataFrame | None,
    model_proba: np.ndarray | None,
) -> folium.Map:

    bounds  = state_gdf.total_bounds   # [minx, miny, maxx, maxy]
    center  = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    fb      = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]

    m = folium.Map(location=center, zoom_start=6, tiles="CartoDB positron")

    # State boundary
    folium.GeoJson(
        state_gdf.__geo_interface__,
        name="State boundary",
        style_function=lambda _: {
            "fillColor": "transparent",
            "color": "#2c3e50",
            "weight": 2,
        },
    ).add_to(m)

    # Plant markers — clustered, capacity-coloured
    cluster = MarkerCluster(name="Solar plants")
    for _, row in plants_df.iterrows():
        color  = _cap_color(row["p_cap_ac"])
        radius = max(4, min(14, float(np.log1p(row["p_cap_ac"])) * 2.5))
        year   = int(row["p_year"]) if pd.notna(row.get("p_year")) else "?"
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            weight=0.5,
            popup=folium.Popup(
                f"<b>{row['p_name']}</b><br>"
                f"County: {row['p_county']}<br>"
                f"Capacity: {row['p_cap_ac']:.1f} MW AC<br>"
                f"Built: {year}&emsp;Type: {row['p_tech_pri']}",
                max_width=260,
            ),
        ).add_to(cluster)
    cluster.add_to(m)

    # Capacity density heatmap
    if show_heatmap and not plants_df.empty:
        heat_data = plants_df[["lat", "lon", "p_cap_ac"]].values.tolist()
        HeatMap(
            heat_data, name="Capacity density",
            radius=28, blur=18, min_opacity=0.25,
        ).add_to(m)

    # Training data overlay (CA model, two GeoJson layers — no per-row loop)
    if show_training and train_gdf is not None:
        pos = train_gdf[train_gdf["have_plant"] == 1][["have_plant", "geometry"]]
        neg = train_gdf[train_gdf["have_plant"] == 0][["have_plant", "geometry"]]

        folium.GeoJson(
            pos,
            name="Training: plant present",
            marker=folium.CircleMarker(radius=5, fill=True, weight=0),
            style_function=lambda _: {
                "fillColor": "#27ae60", "color": "#27ae60",
                "fillOpacity": 0.7, "opacity": 0,
            },
        ).add_to(m)
        folium.GeoJson(
            neg,
            name="Training: no plant",
            marker=folium.CircleMarker(radius=4, fill=True, weight=0),
            style_function=lambda _: {
                "fillColor": "#e74c3c", "color": "#e74c3c",
                "fillOpacity": 0.45, "opacity": 0,
            },
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds(fb)
    return m


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _year_chart(df: pd.DataFrame) -> alt.Chart:
    yr = df.dropna(subset=["p_year"]).copy()
    yr["p_year"] = yr["p_year"].astype(int)
    counts = yr.groupby("p_year")["p_cap_ac"].sum().reset_index()
    counts.columns = ["Year", "MW"]
    return (
        alt.Chart(counts)
        .mark_bar(color="#f39c12")
        .encode(
            x=alt.X("Year:O", axis=alt.Axis(labelAngle=-45, title=None)),
            y=alt.Y("MW:Q", title="Capacity added (MW AC)"),
            tooltip=["Year", alt.Tooltip("MW:Q", format=".1f", title="MW AC")],
        )
        .properties(title="Capacity commissioned by year", height=210)
        .configure_view(strokeWidth=0)
    )


def _county_chart(df: pd.DataFrame) -> alt.Chart:
    top = (
        df.groupby("p_county")["p_cap_ac"]
        .sum()
        .nlargest(10)
        .reset_index()
    )
    top.columns = ["County", "MW"]
    return (
        alt.Chart(top)
        .mark_bar(color="#3498db")
        .encode(
            y=alt.Y("County:N", sort="-x", title=None),
            x=alt.X("MW:Q", title="Total capacity (MW AC)"),
            tooltip=["County", alt.Tooltip("MW:Q", format=".1f", title="MW AC")],
        )
        .properties(title="Top 10 counties by installed capacity", height=210)
        .configure_view(strokeWidth=0)
    )


def _importance_chart(importances: dict) -> alt.Chart:
    df = pd.DataFrame(importances.items(), columns=["Feature", "Importance"])
    feature_labels = {
        "distance_to_largepop":    "Dist. to population centre",
        "distance_to_powerlines":  "Dist. to powerline",
        "lat":                     "Latitude",
        "lon":                     "Longitude",
    }
    df["Feature"] = df["Feature"].map(feature_labels)
    df = df.sort_values("Importance", ascending=True)
    return (
        alt.Chart(df)
        .mark_bar(color="#8e44ad")
        .encode(
            y=alt.Y("Feature:N", sort="-x", title=None),
            x=alt.X("Importance:Q", title="Mean decrease in impurity"),
            tooltip=["Feature", alt.Tooltip("Importance:Q", format=".3f")],
        )
        .properties(title="Feature importances — Random Forest (CA)", height=160)
        .configure_view(strokeWidth=0)
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Illuminating Horizons",
        page_icon="☀️",
        layout="wide",
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("☀️ Illuminating Horizons")
        st.caption("Solar power station siting · USPVDB + geospatial ML")

        all_plants = load_plants()

        available_abbrs = {
            s for s in all_plants["p_state"].unique()
            if os.path.exists(os.path.join(GADM_DIR, f"gdf_US-{s}.geojson"))
        }
        state_options = sorted(
            ABBR_TO_NAME[a] for a in available_abbrs if a in ABBR_TO_NAME
        )
        default_idx = state_options.index("California") if "California" in state_options else 0
        state_name  = st.selectbox("State", state_options, index=default_idx)
        state_abbr  = STATE_ABBR[state_name]
        is_ca       = state_abbr == "CA"

        st.divider()
        st.subheader("Layers")
        show_heatmap = st.toggle("Capacity density heatmap", value=True)
        show_training = st.toggle(
            "Training data (CA model)",
            value=False,
            disabled=not is_ca,
            help="Training data covers California only.",
        )

        st.divider()
        st.subheader("Filter plants")
        max_cap   = float(all_plants["p_cap_ac"].max())
        cap_range = st.slider("Capacity (MW AC)", 0.0, max_cap, (0.0, max_cap))

        year_min = int(all_plants["p_year"].dropna().min())
        year_max = int(all_plants["p_year"].dropna().max())
        year_range = st.slider("Year commissioned", year_min, year_max, (year_min, year_max))

        st.divider()
        st.subheader("Legend — plant size")
        for label, color in zip(CAP_LABELS, CAP_COLORS):
            st.markdown(
                f'<span style="color:{color};font-size:1.1rem">●</span>&nbsp;{label}',
                unsafe_allow_html=True,
            )

        st.divider()
        with st.expander("About"):
            st.markdown(
                "**Data**: [US Solar PV Database v2](https://uspvdb.ornl.gov/) (ORNL/DOE, 2024)  \n"
                "**CA model**: Random Forest trained on 2,337 candidate locations · "
                "features: distance to population centre + powerline (OSM/WorldPop), lat/lon  \n"
                "**Full pipeline**: adds JAXA satellite radiation & temperature, "
                "Copernicus land cover, 5 km grid prediction via PostGIS"
            )

    # ── Filter plants ─────────────────────────────────────────────────────────
    df_state = all_plants[
        (all_plants["p_state"] == state_abbr)
        & (all_plants["p_cap_ac"].between(cap_range[0], cap_range[1]))
        & (all_plants["p_year"].fillna(0).between(year_range[0], year_range[1]))
    ].copy()

    state_gdf = load_state_gdf(state_abbr)
    if state_gdf is None:
        st.error(f"No boundary data for {state_name}.")
        return

    # ── Load CA model (lazy) ──────────────────────────────────────────────────
    train_gdf = model_proba = None
    pipe = report = auc = importances = None
    if is_ca:
        pipe, report, auc, importances, model_proba = get_ca_model()
        train_gdf = load_training_data()

    # ── Top metrics ───────────────────────────────────────────────────────────
    total_mw    = df_state["p_cap_ac"].sum()
    plant_count = len(df_state)
    avg_mw      = df_state["p_cap_ac"].mean() if plant_count else 0.0
    largest     = df_state.nlargest(1, "p_cap_ac").iloc[0] if plant_count else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Solar installations", f"{plant_count:,}")
    col2.metric("Total capacity (MW AC)", f"{total_mw:,.0f}")
    col3.metric("Avg plant size (MW AC)", f"{avg_mw:.1f}")
    if largest is not None:
        col4.metric(
            "Largest plant",
            f"{largest['p_cap_ac']:.0f} MW",
            delta=largest["p_name"],
            delta_color="off",
        )

    # ── Map ───────────────────────────────────────────────────────────────────
    folium_map = build_map(
        state_abbr, df_state, state_gdf,
        show_heatmap, show_training, train_gdf, model_proba,
    )
    st_folium(folium_map, use_container_width=True, height=520, key=f"map_{state_abbr}")

    # ── Charts ────────────────────────────────────────────────────────────────
    if not df_state.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.altair_chart(_year_chart(df_state), use_container_width=True)
        with c2:
            st.altair_chart(_county_chart(df_state), use_container_width=True)
    else:
        st.info(f"No installations match the current filters in {state_name}.")

    # ── California ML section ─────────────────────────────────────────────────
    if is_ca and report is not None:
        st.divider()
        st.subheader("California Siting Model")
        st.caption(
            "Binary Random Forest classifier (plant / no-plant) trained on 2,337 "
            "candidate locations across California. Toggle **Training data** in the "
            "sidebar to overlay positive (green) and negative (red) training samples."
        )

        prec = report["1"]["precision"]
        rec  = report["1"]["recall"]
        f1   = report["1"]["f1-score"]
        acc  = report["accuracy"]

        ma, mb, mc, md = st.columns(4)
        ma.metric("Accuracy",            f"{acc:.1%}")
        mb.metric("Precision (plant=1)", f"{prec:.1%}")
        mc.metric("Recall (plant=1)",    f"{rec:.1%}")
        md.metric("ROC-AUC",             f"{auc:.3f}")

        st.altair_chart(_importance_chart(importances), use_container_width=True)

        with st.expander("What these features mean"):
            st.markdown(
                "- **Distance to population centre**: closer to demand → stronger business case  \n"
                "- **Distance to powerline**: grid connection cost scales with distance  \n"
                "- **Lat / Lon**: capture regional solar irradiance and regulatory patterns  \n\n"
                "The full pipeline (PostGIS + JAXA) adds solar radiation, daytime temperature, "
                "and land-cover class — see `illuminating/interface/main.py`."
            )


if __name__ == "__main__":
    main()
