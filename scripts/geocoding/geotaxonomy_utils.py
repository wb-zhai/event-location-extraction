from functools import cache
from pathlib import Path

import geopandas as gpd
import pandas as pd
from google.cloud import storage


@cache
def load_zhai_iso_codes(iso_version: int, lower: bool = False):
    if iso_version not in [2, 3]:
        raise ValueError(
            f"ISO version has to be either 2 or 3. ISO version: {iso_version}"
        )

    zhai_iso_codes = pd.read_json(
        "https://raw.githubusercontent.com/wb-zhai/gists/refs/heads/main/zhai_iso3166-1.json"
    )[f"Alpha-{iso_version} code"]
    return (
        zhai_iso_codes.unique().tolist()
        if not lower
        else zhai_iso_codes.lower().unique().tolist()
    )


def create_coordinate_df(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
):
    df = (
        df[[lat_col, lon_col]]
        .drop_duplicates()
        .dropna()
        .rename(
            {
                lat_col: "latitude",
                lon_col: "longitude",
            },
            axis=1,
        )
    )

    for coordinate_col in ["latitude", "longitude"]:
        df[coordinate_col] = pd.to_numeric(df[coordinate_col], errors="coerce")

    df.reset_index(drop=True, inplace=True)

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )

    return gdf


def get_latlon_to_id(
    df: pd.DataFrame,
    geotaxonomy_gdf: gpd.GeoDataFrame,
    lat_col: str,
    lon_col: str,
):
    our_gdf = create_coordinate_df(df, lat_col, lon_col)

    gdf_joined = gpd.sjoin(
        our_gdf,
        geotaxonomy_gdf,
        how="inner",
        predicate="within",  # or "intersects" if you prefer touching boundaries included
    ).drop(["geometry", "index_right"], axis=1)

    return gdf_joined


def load_geotaxonomy(
    path: str,
    keep_geotaxonomy_cols: list[str],
):
    if not Path(path).is_file():
        storage_client = storage.Client()
        bucket = storage_client.bucket("zhai-data-geotaxonomy")
        blob = bucket.blob("geotaxonomy_prewb.geojson")
        blob.download_to_filename(path)
    return gpd.read_file(path, columns=keep_geotaxonomy_cols)


def get_latlon_to_id_from_path(
    df: pd.DataFrame,
    geotaxonomy_path: str,
    keep_geotaxonomy_cols: list[str] | dict[str, str],
    lat_col: str,
    lon_col: str,
):
    geotaxonomy_gdf = load_geotaxonomy(geotaxonomy_path, list(keep_geotaxonomy_cols))
    if isinstance(keep_geotaxonomy_cols, dict):
        geotaxonomy_gdf = geotaxonomy_gdf.rename(keep_geotaxonomy_cols, axis=1)
    return get_latlon_to_id(df, geotaxonomy_gdf, lat_col, lon_col)


def geocode_df_from_path(
    df: pd.DataFrame,
    geotaxonomy_path: str,
    keep_geotaxonomy_cols: list[str] | dict[str, str],
    lat_col: str,
    lon_col: str,
):
    for col in [lat_col, lon_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    latlon_to_id = get_latlon_to_id_from_path(
        df=df,
        geotaxonomy_path=geotaxonomy_path,
        keep_geotaxonomy_cols=keep_geotaxonomy_cols,
        lat_col=lat_col,
        lon_col=lon_col,
    )
    df.rename(
        {
            lat_col: "latitude",
            lon_col: "longitude",
        },
        axis=1,
        inplace=True,
    )
    return df.merge(latlon_to_id, on=["latitude", "longitude"])


# Usage: get_latlon_to_id_from_path(port_df, "World Bank Official Boundaries - Admin 2_zhai.geojson", ["ISO_A3", "NAM_0", "ADM1CD_c", "NAM_1", "ADM2CD_c", "NAM_2"], "lat", "lon")
