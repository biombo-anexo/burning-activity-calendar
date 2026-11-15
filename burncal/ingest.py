"""Read FIRMS hotspot files (CSV or shapefile), clean them, and collapse to fire cell-days.

Real FIRMS files differ in small ways (column case, presence of TYPE, confidence coding,
satellite codes), so everything is normalised here and nothing downstream needs to care.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CFG, Config
from .geo import cell_index

REQUIRED = ["latitude", "longitude", "acq_date"]
OPTIONAL = ["acq_time", "satellite", "confidence", "type", "frp", "daynight"]


STATIC_GRID = 0.01          # degrees; static sources are matched on this fine grid (+/- 1 cell)
STATIC_TYPES = (1, 2)       # 1 = active volcano, 2 = other static land source (gas flares, industry)


def _encode(lat, lon) -> np.ndarray:
    i = np.floor(np.asarray(lat, dtype="float64") / STATIC_GRID + 1e-9).astype("int64")
    j = np.floor(np.asarray(lon, dtype="float64") / STATIC_GRID + 1e-9).astype("int64")
    return (i + 10_000) * 100_000 + (j + 20_000)


def _expand_neighbours(keys: np.ndarray) -> np.ndarray:
    out = [keys + di * 100_000 + dj for di in (-1, 0, 1) for dj in (-1, 0, 1)]
    return np.unique(np.concatenate(out))


def drop_near_static(df: pd.DataFrame, keys: np.ndarray):
    """Drop detections within +/-0.01 degree of known static sources. Returns (df, n_removed)."""
    near = np.isin(_encode(df["latitude"], df["longitude"]), keys)
    return df[~near], int(near.sum())


def peek_has_type(path) -> bool:
    """True if the file has a TYPE column (archive files do, near-real-time files do not)."""
    path = Path(path)
    if path.suffix.lower() in (".shp", ".dbf"):
        import shapefile

        return "type" in [f[0].lower() for f in shapefile.Reader(str(path.with_suffix(""))).fields[1:]]
    return "type" in [c.strip().lower() for c in pd.read_csv(path, nrows=0).columns]


def _read_raw(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".shp", ".dbf"):
        import shapefile  # pyshp

        r = shapefile.Reader(str(path.with_suffix("")))
        cols = [f[0].lower() for f in r.fields[1:]]
        return pd.DataFrame(r.records(), columns=cols)
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def read_firms_file(path, sensor: str, cfg: Config = CFG, static_keys: np.ndarray | None = None):
    """Return (clean_df, stats). sensor is 'MODIS' or 'VIIRS'.

    static_keys: expanded 0.01-degree keys of known static sources (learned from archive files
    that carry TYPE). Used only for files WITHOUT a TYPE column, to drop flares/volcanoes there too.
    """
    if sensor not in ("MODIS", "VIIRS"):
        raise ValueError("sensor must be 'MODIS' or 'VIIRS'")
    path = Path(path)
    df = _read_raw(path)
    df.columns = [c.lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing required columns {missing}")
    for c in OPTIONAL:
        if c not in df.columns:
            df[c] = np.nan

    stats = {"file": path.name, "sensor": sensor, "rows_in": int(len(df)), "has_type": bool(df["type"].notna().any())}

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["date"] = pd.to_datetime(df["acq_date"], errors="coerce").dt.normalize()
    df["frp"] = pd.to_numeric(df["frp"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", "date"])
    stats["rows_valid"] = int(len(df))

    # Drop duplicates (overlapping downloads).
    before = len(df)
    df = df.drop_duplicates(subset=["date", "acq_time", "latitude", "longitude", "satellite"])
    stats["duplicates_removed"] = int(before - len(df))

    # Hotspot type: keep presumed vegetation fire. NRT shapefiles have no TYPE column.
    if df["type"].notna().any():
        t = pd.to_numeric(df["type"], errors="coerce")
        keep = t.isin(cfg.keep_types)
        stats["removed_by_type"] = int((~keep).sum())
        # Remember where static sources are, so TYPE-less (near-real-time) files can be cleaned the same way.
        st = df[t.isin(STATIC_TYPES)]
        stats["static_keys"] = np.unique(_encode(st["latitude"], st["longitude"])) if len(st) else np.array([], dtype="int64")
        # Collateral: share of genuine type-0 detections that sit next to a static source (lost in NRT cleaning).
        if len(stats["static_keys"]):
            near = np.isin(_encode(df.loc[keep, "latitude"], df.loc[keep, "longitude"]), _expand_neighbours(stats["static_keys"]))
            stats["collateral_pct"] = float(near.mean() * 100)
        df = df[keep]
        stats["type_filter"] = "applied (TYPE column)"
    elif static_keys is not None and len(static_keys):
        near = np.isin(_encode(df["latitude"], df["longitude"]), static_keys)
        stats["removed_by_type"] = int(near.sum())
        df = df[~near]
        stats["type_filter"] = "TYPE column absent: dropped detections next to static sources learned from archive files"
    else:
        stats["removed_by_type"] = 0
        stats["type_filter"] = "TYPE column absent and no archive to learn static sources from: flares NOT removed"

    # Confidence.
    conf = df["confidence"].astype(str).str.strip().str.lower()
    if sensor == "MODIS":
        num = pd.to_numeric(conf, errors="coerce")
        keep = num >= cfg.modis_min_conf
    else:
        num = pd.to_numeric(conf, errors="coerce")
        if num.notna().mean() > 0.9:           # numeric VIIRS confidence (rare)
            keep = num >= cfg.modis_min_conf
        else:
            keep = conf.isin(cfg.viirs_ok_conf)
    stats["removed_by_confidence"] = int((~keep).sum())
    df = df[keep]

    df["satellite"] = df["satellite"].fillna("?").astype(str)
    stats["satellites"] = df["satellite"].value_counts().to_dict()
    stats["rows_out"] = int(len(df))
    stats["date_min"] = str(df["date"].min().date()) if len(df) else None
    stats["date_max"] = str(df["date"].max().date()) if len(df) else None
    if len(df):
        stats["bbox"] = [float(df["longitude"].min()), float(df["latitude"].min()),
                         float(df["longitude"].max()), float(df["latitude"].max())]
    out = df[["date", "latitude", "longitude", "satellite", "frp", "acq_time"]].copy()
    out["sensor"] = sensor
    return out.reset_index(drop=True), stats


def to_cell_days(df: pd.DataFrame, cfg: Config = CFG) -> pd.DataFrame:
    """Collapse pixel detections to one row per (date, 0.1 deg cell, sensor)."""
    d = pd.DataFrame(
        {
            "date": df["date"].values,
            "cell_i": cell_index(df["latitude"].values, cfg.cell_deg),
            "cell_j": cell_index(df["longitude"].values, cfg.cell_deg),
            "sensor": df["sensor"].values,
            "frp": df["frp"].fillna(0).values,
        }
    )
    g = (
        d.groupby(["date", "cell_i", "cell_j", "sensor"], sort=True)
        .agg(n_det=("frp", "size"), frp_sum=("frp", "sum"))
        .reset_index()
    )
    g["n_det"] = g["n_det"].astype("int32")
    g["date"] = g["date"].astype("datetime64[ns]")
    g["sensor"] = g["sensor"].astype(object)
    return g
