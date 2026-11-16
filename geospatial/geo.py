"""
Georeferencing helpers for Web-Mercator map tiles (Esri / OSM style, 256 px).

Our satellite tiles are named  <zoom>_<x>_<y>.png  and come from the standard
XYZ tile scheme, so a pixel in the stitched mosaic has an EXACT longitude and
latitude - no guessing. The formulas below are the same ones used by
scripts/generate_labels.py, just applied to the whole mosaic.

CRS: tiles are in EPSG:3857 (Web Mercator); we output GeoJSON in EPSG:4326
(longitude, latitude in degrees), which is what GeoJSON requires.
"""
import math

TILE = 256


def metres_per_pixel(lat_deg: float, zoom: int) -> float:
    """Ground size of one pixel (metres) at a latitude, for 256 px tiles."""
    return 156543.03392804097 * math.cos(math.radians(lat_deg)) / (2 ** zoom)


def px_to_lonlat(px: float, py: float, x0: int, y0: int, z: int):
    """Mosaic pixel (px, py) -> (lon, lat). (x0, y0) = tile index of the mosaic's top-left tile."""
    n = TILE * (2 ** z)
    X = x0 * TILE + px
    Y = y0 * TILE + py
    lon = X / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * Y / n))))
    return lon, lat


def lonlat_to_px(lon: float, lat: float, x0: int, y0: int, z: int):
    """(lon, lat) -> mosaic pixel (px, py)."""
    n = TILE * (2 ** z)
    X = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    Y = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
    return X - x0 * TILE, Y - y0 * TILE


def mosaic_bounds(meta: dict):
    """(west, south, east, north) of the mosaic."""
    west, north = px_to_lonlat(0, 0, meta["x0"], meta["y0"], meta["z"])
    east, south = px_to_lonlat(meta["width"], meta["height"], meta["x0"], meta["y0"], meta["z"])
    return west, south, east, north


def mosaic_mpp(meta: dict) -> float:
    west, south, east, north = mosaic_bounds(meta)
    return metres_per_pixel((south + north) / 2.0, meta["z"])


def to_local_metres(lon: float, lat: float, lon0: float, lat0: float):
    """Small-area flat approximation (fine for a few km): (x_east, y_north) in metres."""
    x = (lon - lon0) * 111320.0 * math.cos(math.radians(lat0))
    y = (lat - lat0) * 110574.0
    return x, y
