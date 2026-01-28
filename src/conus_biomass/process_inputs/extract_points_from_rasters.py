import numpy as np
import pyproj
import xarray as xr
from shapely.geometry import Point


def calculate_point_buffers(
    lats: list,
    lons: list,
    crs_original="EPSG:4326",
    crs_new="EPSG:5070",
    buffer: float = 1609.34 / 2,
):
    """Calculate buffer radius buffers around lat/lon points.
    Args:
        lats (list or array): List or array of latitudes.
        lons (list or array): List or array of longitudes.
        crs_original (str): Original coordinate reference system (default is WGS84).
        crs_new (str): New coordinate reference system for buffering (default is Albers Equal Area).
        buffer (float): Buffer distance in meters (default is 1609.34m, ~1 mile).
    Returns:
        List of shapely Polygon objects representing the buffers.
    """
    projected_points = calculate_transformed_points(
        lats=lats,
        lons=lons,
        crs_original="EPSG:4326",
        crs_new="EPSG:5070",
    )

    buffers = [pt.buffer(buffer) for pt in projected_points]
    return buffers


def calculate_transformed_points(
    lats: list,
    lons: list,
    crs_original="EPSG:4326",
    crs_new="EPSG:5070",
):
    """Calculate transformed lat/lon points.
    Args:
        lats (list or array): List or array of latitudes.
        lons (list or array): List or array of longitudes.
        crs_original (str): Original coordinate reference system (default is WGS84).
        crs_new (str): New coordinate reference system for buffering (default is Albers Equal Area).
        buffer (float): Buffer distance in meters (default is 1609.34m, ~1 mile).
    Returns:
        List of shapely Polygon objects representing the points.
    """

    # Project points to Albers
    proj = pyproj.Transformer.from_crs(crs_original, crs_new, always_xy=True)

    # Vectorized transformation
    x, y = proj.transform(lons, lats)

    # Create point objects
    projected_points = [Point(xi, yi) for xi, yi in zip(x, y)]

    return projected_points


def extract_from_raster(
    buffers: list, raster_to_extract: xr.DataArray, nan_value=1e30, aggfunc: str = "mean"
):
    """
    Extract mean values from a raster within given buffer geometries.

    Args:
        buffers (list): List of buffer geometries (shapely Polygons).
        raster_to_extract (rioxarray.DataArray): Raster data to extract values from.
        nan_value (float, optional): Value to consider as NaN. Defaults to 1e30.

    Returns:
        np.ndarray: Array of mean values for each buffer.
    """
    buffer_agg = np.full(len(buffers), np.nan)

    raster_crs = raster_to_extract.rio.crs

    raster_masked = raster_to_extract.where(raster_to_extract < nan_value)

    for i, buffer_geom in enumerate(buffers):
        if i % 1000 == 0:
            print(i)
        try:
            minx, miny, maxx, maxy = buffer_geom.bounds
            cropped = raster_masked.rio.clip_box(minx, miny, maxx, maxy)
            clipped = cropped.rio.clip([buffer_geom], raster_crs, drop=True, all_touched=True)
            # clipped = raster_masked.rio.clip([buffer_geom], raster_crs, drop=True, all_touched=False)

            if aggfunc == "mean":
                buffer_agg[i] = np.nanmean(clipped)
            elif aggfunc == "max":
                buffer_agg[i] = np.nanmax(clipped)

        except Exception:
            # cases where buffer doesn't intersect raster
            buffer_agg[i] = np.nan

    return buffer_agg
