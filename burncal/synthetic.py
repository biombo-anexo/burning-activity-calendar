"""Synthetic FIRMS-format data for Colombia, used to test the whole pipeline end to end.

NOT real data. It encodes assumptions (MODIS misses small fires, VIIRS sees more and more
pixels per fire, Terra degrades after 2021) so we can check that the code recovers a known
truth. The hidden truth file contains what VIIRS *would* have recorded before 2012; the
pipeline never reads it - only the test/validation script does.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

START, END = pd.Timestamp("2000-11-01"), pd.Timestamp("2026-11-10")
AQUA_START = pd.Timestamp("2002-07-04")
VIIRS_START = pd.Timestamp("2012-01-20")
TERRA_DRIFT_START = pd.Timestamp("2021-01-01")

# name, lat, lon, sd_lat, sd_lon, events per year, peak day-of-year, sd days
CLUSTERS = [
    ("llanos", 4.8, -71.0, 1.3, 1.6, 6000, 35, 28),
    ("caribe", 9.5, -74.0, 1.2, 1.0, 2500, 60, 30),
    ("amazon_arc", 1.5, -73.5, 1.2, 1.5, 3000, 220, 30),
    ("amazon_jan", 1.8, -73.0, 1.2, 1.5, 1200, 20, 20),
    ("andes", 5.5, -75.5, 1.2, 0.8, 1500, 40, 40),
    ("pacific", 3.5, -77.0, 0.8, 0.4, 500, 200, 60),
]
BOOST_YEARS = {2002: 1.7, 2010: 1.6, 2016: 1.8, 2020: 1.7, 2024: 1.8}
LAT_RANGE, LON_RANGE = (-4.2, 13.4), (-79.0, -66.9)


def _events(rng: np.random.Generator, scale: float) -> pd.DataFrame:
    years = np.arange(START.year, END.year + 1)
    ymult = {y: float(np.exp(rng.normal(0, 0.28)) * BOOST_YEARS.get(y, 1.0)) for y in years}
    parts = []
    for name, lat, lon, slat, slon, n_yr, peak, sd in CLUSTERS:
        for y in years:
            n = rng.poisson(n_yr * scale * ymult[y])
            if n == 0:
                continue
            doy = rng.normal(peak, sd, n)
            d0 = pd.Timestamp(f"{y}-01-01")
            dates = d0 + pd.to_timedelta(np.floor(doy).astype(int), unit="D")
            parts.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "lat": np.clip(rng.normal(lat, slat, n), *LAT_RANGE),
                        "lon": np.clip(rng.normal(lon, slon, n), *LON_RANGE),
                        "size": np.exp(rng.normal(np.log(5.0), 1.3, n)),  # hectares
                    }
                )
            )
    ev = pd.concat(parts, ignore_index=True)
    ev = ev[(ev["date"] >= START) & (ev["date"] <= END)].reset_index(drop=True)
    dur = rng.geometric(0.5, len(ev))  # event lasts 1..n days
    ev = ev.loc[ev.index.repeat(dur)].reset_index(drop=True)
    offs = np.concatenate([np.arange(k) for k in dur])
    ev["date"] = ev["date"] + pd.to_timedelta(offs, unit="D")
    return ev[(ev["date"] >= START) & (ev["date"] <= END)].reset_index(drop=True)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _observe(ev, rng, kind: str, satellite: str = ""):
    """Return pixel-level detections of the events for one satellite."""
    size, date = ev["size"].values, ev["date"]
    if kind == "MODIS":
        p = 0.92 * _sigmoid((np.log(size) - np.log(30)) / 0.9)
        if satellite == "Terra":
            p = p * 0.92
            p = np.where(date >= TERRA_DRIFT_START, p * 0.85, p)
            p = np.where(date >= START, p, 0)
        else:
            p = np.where(date >= AQUA_START, p, 0)
        npx = 1 + rng.poisson(size / 150)
        jit = 0.006
    else:
        p = 0.95 * _sigmoid((np.log(size) - np.log(3.5)) / 0.9)
        npx = 1 + rng.poisson(size / 20)
        jit = 0.002
    det = rng.random(len(ev)) < p
    idx = np.repeat(np.where(det)[0], npx[det])
    n = len(idx)
    sub = ev.iloc[idx]
    s = np.sqrt(npx[idx])
    out = pd.DataFrame(
        {
            "latitude": sub["lat"].values + rng.normal(0, jit, n) * s,
            "longitude": sub["lon"].values + rng.normal(0, jit, n) * s,
            "acq_date": sub["date"].values,
            "satellite": satellite or "N",
            "frp": np.exp(rng.normal(np.log(8) + 0.3 * np.log(sub["size"].values), 0.6)),
        }
    )
    return out


def _decorate(df, kind, rng, vtype=0):
    n = len(df)
    df = df.copy()
    df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.strftime("%Y-%m-%d")
    df["acq_time"] = rng.integers(0, 2400, n).astype(str)
    df["acq_time"] = df["acq_time"].str.zfill(4)
    df["scan"] = 1.0
    df["track"] = 1.0
    df["instrument"] = kind
    df["version"] = "6.1" if kind == "MODIS" else "2.0"
    df["daynight"] = rng.choice(["D", "N"], n, p=[0.8, 0.2])
    df["type"] = vtype
    df["brightness" if kind == "MODIS" else "bright_ti4"] = 330.0
    df["bright_t31" if kind == "MODIS" else "bright_ti5"] = 295.0
    if kind == "MODIS":
        df["confidence"] = np.clip(rng.normal(62, 20, n), 0, 100).astype(int).astype(str)
        cols = ["latitude", "longitude", "brightness", "scan", "track", "acq_date", "acq_time", "satellite",
                "instrument", "confidence", "version", "bright_t31", "frp", "daynight", "type"]
    else:
        df["confidence"] = rng.choice(["l", "n", "h"], n, p=[0.08, 0.80, 0.12])
        cols = ["latitude", "longitude", "bright_ti4", "scan", "track", "acq_date", "acq_time", "satellite",
                "instrument", "confidence", "version", "bright_ti5", "frp", "daynight", "type"]
    return df[cols]


def _junk(rng, kind, sat_list, start):
    """Static gas-flare detections (type 2) and offshore detections (type 3)."""
    flares = [(4.55, -72.1), (4.9, -72.3), (5.2, -71.8), (3.9, -73.0), (7.1, -73.9), (7.4, -73.6), (2.9, -75.3), (8.6, -73.5)]
    days = pd.date_range(max(start, START), END, freq="D")
    rows = []
    for lat, lon in flares:
        keep = rng.random(len(days)) < 0.85
        rows.append(pd.DataFrame({"latitude": lat + rng.normal(0, 0.001, keep.sum()),
                                  "longitude": lon + rng.normal(0, 0.001, keep.sum()),
                                  "acq_date": days[keep], "satellite": rng.choice(sat_list, keep.sum()),
                                  "frp": 30.0}))
    fl = _decorate(pd.concat(rows, ignore_index=True), kind, rng, vtype=2)
    nsea = 400
    sea = pd.DataFrame({"latitude": rng.uniform(12.6, 13.3, nsea), "longitude": rng.uniform(-76, -72, nsea),
                        "acq_date": rng.choice(days, nsea), "satellite": rng.choice(sat_list, nsea), "frp": 10.0})
    sea = _decorate(sea, kind, rng, vtype=3)
    return pd.concat([fl, sea], ignore_index=True)


def generate(out_dir, scale: float = 1.0, seed: int = 42):
    """Write MODIS_synthetic.csv, VIIRS_synthetic.csv and truth_viirs_pre2012.csv into out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ev = _events(rng, scale)

    terra = _observe(ev, rng, "MODIS", "Terra")
    aqua = _observe(ev, rng, "MODIS", "Aqua")
    modis = pd.concat([terra, aqua], ignore_index=True)
    modis = _decorate(modis, "MODIS", rng)
    modis = pd.concat([modis, _junk(rng, "MODIS", ["Terra", "Aqua"], START)], ignore_index=True)

    viirs_all = _observe(ev, rng, "VIIRS", "N")
    d = pd.to_datetime(viirs_all["acq_date"])
    truth = viirs_all[d < pd.Timestamp("2012-02-01")]
    viirs = viirs_all[d >= VIIRS_START]
    viirs = _decorate(viirs, "VIIRS", rng)
    viirs = pd.concat([viirs, _junk(rng, "VIIRS", ["N"], VIIRS_START)], ignore_index=True)
    truth = _decorate(truth, "VIIRS", rng)

    # Shuffle so the files do not look artificially sorted.
    modis.sample(frac=1, random_state=1).to_csv(out / "MODIS_synthetic.csv", index=False)
    viirs.sample(frac=1, random_state=2).to_csv(out / "VIIRS_synthetic.csv", index=False)
    truth.to_csv(out / "truth_viirs_pre2012.csv", index=False)
    return {"events": int(len(ev)), "modis_rows": int(len(modis)), "viirs_rows": int(len(viirs)), "truth_rows": int(len(truth))}
