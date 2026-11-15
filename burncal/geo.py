"""Grid and area-of-interest (AOI) geometry. Pure numpy, no GIS dependency."""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_GRID_CELLS = 5_000_000
MAX_POLYGON_VERTICES = 5_000


def cell_index(values, cell_deg: float) -> np.ndarray:
    """Integer cell index along one axis. The tiny epsilon makes exact multiples stable."""
    return np.floor(np.asarray(values, dtype="float64") / cell_deg + 1e-9).astype(np.int32)


def cell_center(idx, cell_deg: float) -> np.ndarray:
    return (np.asarray(idx, dtype="float64") + 0.5) * cell_deg


def points_in_polygon(x, y, poly: np.ndarray) -> np.ndarray:
    """Vectorised even-odd ray casting. poly is (n, 2) of (lon, lat)."""
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    px, py = poly[:, 0], poly[:, 1]
    inside = np.zeros(x.shape, dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi, xj, yj = px[i], py[i], px[j], py[j]
        crosses = (yi > y) != (yj > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xcross = (xj - xi) * (y - yi) / (yj - yi) + xi
        inside ^= crosses & (x < xcross)
        j = i
    return inside


def validate_aoi(aoi: dict) -> dict:
    """Return a normalised AOI dict or raise ValueError with a user-readable message."""
    bbox, polygon = aoi.get("bbox"), aoi.get("polygon")
    if (bbox is None) == (polygon is None):
        raise ValueError("Provide exactly one of 'bbox' or 'polygon'.")
    if bbox is not None:
        if len(bbox) != 4:
            raise ValueError("bbox must be [west, south, east, north].")
        w, s, e, n = map(float, bbox)
        if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
            raise ValueError("bbox out of range or not ordered as [west<east, south<north].")
        return {"bbox": [w, s, e, n]}
    if len(polygon) > MAX_POLYGON_VERTICES:
        raise ValueError(f"polygon has too many vertices (limit {MAX_POLYGON_VERTICES}).")
    pts = np.asarray(polygon, dtype="float64")
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        raise ValueError("polygon needs at least 3 [lon, lat] points.")
    if not (np.all(np.abs(pts[:, 0]) <= 180) and np.all(np.abs(pts[:, 1]) <= 90)):
        raise ValueError("polygon has coordinates out of range.")
    return {"polygon": pts.tolist()}


def aoi_bounds(aoi: dict):
    if "bbox" in aoi:
        return tuple(aoi["bbox"])
    pts = np.asarray(aoi["polygon"], dtype="float64")
    return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()


def aoi_cells(aoi: dict, cell_deg: float):
    """Grid cells whose centre lies inside the AOI.

    Returns (DataFrame[cell_i, cell_j], small) where small=True means the AOI was smaller
    than one cell and the cell containing its centre was used instead.
    """
    aoi = validate_aoi(aoi)
    w, s, e, n = aoi_bounds(aoi)
    i0, i1 = int(cell_index(s, cell_deg)), int(cell_index(n, cell_deg))
    j0, j1 = int(cell_index(w, cell_deg)), int(cell_index(e, cell_deg))
    if (i1 - i0 + 1) * (j1 - j0 + 1) > MAX_GRID_CELLS:
        raise ValueError("Area of interest is too large; draw a smaller area.")
    ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
    ii, jj = ii.ravel().astype("int32"), jj.ravel().astype("int32")
    lat, lon = cell_center(ii, cell_deg), cell_center(jj, cell_deg)
    if "polygon" in aoi:
        mask = points_in_polygon(lon, lat, np.asarray(aoi["polygon"]))
    else:
        mask = (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)
    small = not mask.any()
    if small:
        ci = cell_index((s + n) / 2, cell_deg)
        cj = cell_index((w + e) / 2, cell_deg)
        return pd.DataFrame({"cell_i": [int(ci)], "cell_j": [int(cj)]}).astype("int32"), True
    return pd.DataFrame({"cell_i": ii[mask], "cell_j": jj[mask]}), False
