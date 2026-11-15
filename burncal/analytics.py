"""Queries and statistics over the harmonized parquet, for an arbitrary area of interest (AOI)."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .config import CFG, Config
from .geo import aoi_cells, cell_center, validate_aoi
from .harmonize import MIN_GAP_DAYS_FLAG

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


class Store:
    """Holds the DuckDB view over harmonized.parquet and the calibration metadata."""

    def __init__(self, data_dir):
        self.dir = Path(data_dir)
        pq = self.dir / "harmonized.parquet"
        cj = self.dir / "calibration.json"
        if not (pq.exists() or (self.dir / "harmonized_parts").is_dir()) or not cj.exists():
            raise FileNotFoundError(f"No processed data in {self.dir}. Run: python -m burncal.pipeline ...")
        self.calib = json.loads(cj.read_text())
        self.cfg = Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in self.calib["config"].items()})
        self.min_date = pd.Timestamp(self.calib["data"]["min_date"])
        self.max_date = pd.Timestamp(self.calib["data"]["max_date"])
        d = self.calib["data"]
        self.gap_months = {k: int(v) for k, v in d.get("gap_months", {}).items()}
        self.prov_start = (pd.Timestamp(d["provisional_start"]) if d.get("provisional_start")
                           else (self.max_date.to_period("M") - (self.cfg.provisional_months - 1)).to_timestamp())
        self.con = duckdb.connect(":memory:")
        parts = sorted((self.dir / "harmonized_parts").glob("*.parquet"))
        src = ("[" + ", ".join(f"'{p.as_posix()}'" for p in parts) + "]") if parts else f"'{pq.as_posix()}'"
        self.con.execute(f"CREATE VIEW h AS SELECT * FROM read_parquet({src})")
        self._lock = threading.Lock()

    def cursor(self):
        return self.con.cursor()

    def lock(self):
        return self._lock

    # ---- month flags ------------------------------------------------------------------
    def month_flag(self, year: int, month: int) -> str:
        ms = pd.Timestamp(year=year, month=month, day=1)
        me = ms + pd.offsets.MonthEnd(0)
        if (ms <= self.min_date <= me and self.min_date > ms) or (ms <= self.max_date <= me and self.max_date < me):
            return "partial"
        if self.gap_months.get(f"{year}-{month:02d}", 0) >= MIN_GAP_DAYS_FLAG:
            return "gaps"
        if ms >= self.prov_start:
            return "provisional"
        if ms < pd.Timestamp(self.cfg.terra_only_until).to_period("M").to_timestamp():
            return "terra_only"
        return "ok"


def _bbox_view(cur, cells: pd.DataFrame) -> None:
    """Temp view 'hb' = h restricted to the AOI's cell-index rectangle. The exact AOI join still
    follows; the range filter only lets DuckDB skip parquet row groups (data sorted by cell)."""
    i0, i1 = int(cells["cell_i"].min()), int(cells["cell_i"].max())
    j0, j1 = int(cells["cell_j"].min()), int(cells["cell_j"].max())
    cur.execute(f"CREATE OR REPLACE TEMP VIEW hb AS SELECT * FROM h "
                f"WHERE cell_i BETWEEN {i0} AND {i1} AND cell_j BETWEEN {j0} AND {j1}")


# --------------------------------------------------------------------------- series
def monthly_series(store: Store, cells: pd.DataFrame) -> pd.DataFrame:
    """Full-history monthly harmonized total for the AOI cells (zeros filled inside coverage)."""
    cur = store.cursor()
    cur.register("aoi_cells", cells)
    _bbox_view(cur, cells)
    df = cur.execute(
        """SELECT year(h.date) AS year, month(h.date) AS month,
                  sum(h.w) AS value, sum(h.n_det) AS n_det, count(*) AS cell_days
           FROM hb AS h JOIN aoi_cells USING (cell_i, cell_j)
           GROUP BY 1, 2 ORDER BY 1, 2"""
    ).fetchdf()
    idx = pd.period_range(store.min_date, store.max_date, freq="M")
    full = pd.DataFrame({"year": idx.year.values, "month": idx.month.values})
    full = full.merge(df, on=["year", "month"], how="left").fillna({"value": 0.0, "n_det": 0, "cell_days": 0})
    full["flag"] = [store.month_flag(int(y), int(m)) for y, m in zip(full["year"], full["month"])]
    return full


def _eligible(full: pd.DataFrame, by, cfg: Config) -> pd.Series:
    return full["flag"].isin(["ok", "provisional"]) & (full["year"] >= by[0]) & (full["year"] <= by[1])


def _category(pctile: float) -> str:
    if pctile >= 90:
        return "very high"
    if pctile >= 75:
        return "above normal"
    if pctile <= 10:
        return "very low"
    if pctile <= 25:
        return "below normal"
    return "normal"


def _anom(x: float, base: np.ndarray, cfg: Config):
    if len(base) < cfg.min_baseline_years:
        return None
    med = float(np.median(base))
    mad = float(np.median(np.abs(base - med)))
    pct = (x - med) / med * 100 if med > 0 else None
    z = (x - med) / (1.4826 * mad) if mad > 0 else None
    pr = float((np.sum(base < x) + 0.5 * np.sum(base == x)) / len(base) * 100)
    return {"median": med, "pct": pct, "z": z, "percentile": pr, "category": _category(pr), "n_baseline": int(len(base))}


def analyze(store: Store, aoi: dict, start=None, end=None, baseline_years=None) -> dict:
    cfg = store.cfg
    aoi = validate_aoi(aoi)
    cells, small = aoi_cells(aoi, cfg.cell_deg)
    full = monthly_series(store, cells)
    y0, y1 = int(full["year"].min()), int(full["year"].max())
    by = tuple(baseline_years) if baseline_years else (y0, y1)

    # climatology per calendar month (baseline = eligible months in baseline years)
    elig = _eligible(full, by, cfg)
    clim = []
    for m in range(1, 13):
        v = full.loc[elig & (full["month"] == m), "value"].values
        clim.append({
            "month": m, "label": MONTHS[m - 1], "n": int(len(v)),
            "median": float(np.median(v)) if len(v) else None,
            "q25": float(np.percentile(v, 25)) if len(v) else None,
            "q75": float(np.percentile(v, 75)) if len(v) else None,
            "min": float(v.min()) if len(v) else None, "max": float(v.max()) if len(v) else None,
        })

    # anomaly per month: baseline excludes the month's own year
    rows = []
    for r in full.itertuples(index=False):
        base = full.loc[elig & (full["month"] == r.month) & (full["year"] != r.year), "value"].values
        a = _anom(float(r.value), base, cfg) if r.flag != "partial" else None
        rows.append({
            "year": int(r.year), "month": int(r.month), "value": round(float(r.value), 2), "flag": r.flag,
            "gap_days": int(store.gap_months.get(f"{int(r.year)}-{int(r.month):02d}", 0)), "n_det": int(r.n_det), "baseline_median": a["median"] if a else None,
            "anomaly_pct": a["pct"] if a else None, "z": a["z"] if a else None,
            "percentile": a["percentile"] if a else None, "category": a["category"] if a else None,
        })
    cal = pd.DataFrame(rows)

    # restrict the displayed calendar to the requested date range
    s = pd.Timestamp(start) if start else store.min_date
    e = pd.Timestamp(end) if end else store.max_date
    if e < s:
        raise ValueError("End date is before start date.")
    ms = pd.to_datetime(dict(year=cal["year"], month=cal["month"], day=1))
    me = ms + pd.offsets.MonthEnd(0)
    shown = cal[(me >= s.normalize()) & (ms <= e.normalize())].reset_index(drop=True)
    shown = shown.astype(object).where(shown.notna(), None)   # NaN is not valid JSON -> null

    # month-to-date comparison when the latest month is incomplete
    mtd = None
    md = store.max_date
    month_incomplete = md != md + pd.offsets.MonthEnd(0)
    if month_incomplete and md.day >= 7 and s <= md.normalize() <= e:
        mtd = _month_to_date(store, cells, by, cfg)

    # rankings (complete months / complete years only, within the shown range)
    comp = shown[shown["flag"].isin(["ok", "provisional", "terra_only", "gaps"])].copy()
    comp["value"] = comp["value"].astype(float)
    top_months = comp.sort_values("value", ascending=False).head(5)[["year", "month", "value", "flag"]].to_dict("records")
    yr = comp.groupby("year").agg(total=("value", "sum"), n=("month", "size")).reset_index()
    yr = yr[yr["n"] == 12]
    top_years = yr.sort_values("total", ascending=False).head(5).round({"total": 1}).to_dict("records")
    peak = sorted([c for c in clim if c["median"] is not None], key=lambda c: -c["median"])[:3]

    summary = {
        "total_in_range": round(float(shown["value"].sum()), 1),
        "months_in_range": int(len(shown)),
        "n_cells": int(len(cells)),
        "small_aoi": bool(small),
        "peak_calendar_months": [{"month": p["month"], "label": p["label"], "median": p["median"]} for p in peak],
        "baseline_years": list(by),
    }
    return {
        "summary": summary, "calendar": shown.to_dict("records"), "climatology": clim,
        "month_to_date": mtd, "top_months": top_months, "top_years": top_years,
        "data_range": {"min": str(store.min_date.date()), "max": str(store.max_date.date())},
        "units": "VIIRS-equivalent fire cell-days (0.1 degree cells with >=1 detection per day)",
    }


def _month_to_date(store: Store, cells: pd.DataFrame, by, cfg: Config):
    md = store.max_date
    cur = store.cursor()
    cur.register("aoi_cells", cells)
    _bbox_view(cur, cells)
    df = cur.execute(
        """SELECT year(h.date) AS year, sum(h.w) AS v FROM hb AS h JOIN aoi_cells USING (cell_i, cell_j)
           WHERE month(h.date) = ? AND day(h.date) <= ? GROUP BY 1""",
        [md.month, md.day],
    ).fetchdf().set_index("year")["v"]
    years = [y for y in range(store.min_date.year, md.year + 1)
             if pd.Timestamp(year=y, month=md.month, day=1) >= store.min_date.to_period("M").to_timestamp()
             and store.month_flag(y, md.month) in ("ok", "provisional", "partial")
             and by[0] <= y <= by[1] and y != md.year]
    base = np.array([float(df.get(y, 0.0)) for y in years])
    cur_v = float(df.get(md.year, 0.0))
    a = _anom(cur_v, base, cfg)
    out = {"year": int(md.year), "month": int(md.month), "day": int(md.day), "value": round(cur_v, 2),
           "label": f"{MONTHS[md.month - 1]} 1-{md.day}, {md.year}", "provisional": True}
    if a:
        out.update({"baseline_median": a["median"], "anomaly_pct": a["pct"], "percentile": a["percentile"],
                    "category": a["category"], "n_baseline": a["n_baseline"]})
    return out


# --------------------------------------------------------------------------- map layer + csv
def _check_range(start, end):
    if start and end and pd.Timestamp(end) < pd.Timestamp(start):
        raise ValueError("End date is before start date.")


def density(store: Store, aoi: dict, start=None, end=None, limit: int = 20000) -> dict:
    _check_range(start, end)
    cfg = store.cfg
    cells, _ = aoi_cells(validate_aoi(aoi), cfg.cell_deg)
    s = pd.Timestamp(start) if start else store.min_date
    e = pd.Timestamp(end) if end else store.max_date
    cur = store.cursor()
    cur.register("aoi_cells", cells)
    _bbox_view(cur, cells)
    df = cur.execute(
        """SELECT h.cell_i, h.cell_j, sum(h.w) AS v FROM hb AS h JOIN aoi_cells USING (cell_i, cell_j)
           WHERE h.date BETWEEN ? AND ? GROUP BY 1, 2 ORDER BY v DESC LIMIT ?""",
        [s.to_pydatetime(), (e + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).to_pydatetime(), limit],
    ).fetchdf()
    lat = cell_center(df["cell_i"].values, cfg.cell_deg)
    lon = cell_center(df["cell_j"].values, cfg.cell_deg)
    return {"cell_deg": cfg.cell_deg, "cells": [[round(float(a), 4), round(float(b), 4), round(float(c), 2)]
                                                for a, b, c in zip(lat, lon, df["v"].values)]}


def hotspot_rows(store: Store, aoi: dict, start=None, end=None):
    """Return (cursor, count) over daily cell-day records for the AOI/date range."""
    _check_range(start, end)
    cfg = store.cfg
    cells, _ = aoi_cells(validate_aoi(aoi), cfg.cell_deg)
    s = pd.Timestamp(start) if start else store.min_date
    e = pd.Timestamp(end) if end else store.max_date
    cur = store.cursor()
    cur.register("aoi_cells", cells)
    _bbox_view(cur, cells)
    args = [s.to_pydatetime(), (e + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).to_pydatetime()]
    n = cur.execute("SELECT count(*) FROM hb AS h JOIN aoi_cells USING (cell_i, cell_j) WHERE h.date BETWEEN ? AND ?", args).fetchone()[0]
    q = cur.execute(
        """SELECT strftime(h.date, '%Y-%m-%d') AS date,
                  round((h.cell_i + 0.5) * ?, 4) AS latitude, round((h.cell_j + 0.5) * ?, 4) AS longitude,
                  h.sensor AS source_sensor, h.n_det AS pixel_detections, round(h.frp_sum, 2) AS frp_sum_mw,
                  round(h.w, 4) AS harmonized_weight, h.flag AS quality_flag
           FROM hb AS h JOIN aoi_cells USING (cell_i, cell_j) WHERE h.date BETWEEN ? AND ?
           ORDER BY h.date, h.cell_i, h.cell_j""",
        [cfg.cell_deg, cfg.cell_deg] + args,
    )
    return q, int(n)


def summary_csv_frame(result: dict) -> pd.DataFrame:
    df = pd.DataFrame(result["calendar"])
    df = df.rename(columns={"value": "harmonized_cell_days", "n_det": "pixel_detections", "gap_days": "suspected_outage_days", "baseline_median": "historical_median",
                            "anomaly_pct": "anomaly_pct_vs_median", "z": "robust_z", "flag": "quality_flag"})
    for c in ("historical_median", "anomaly_pct_vs_median", "robust_z", "percentile"):
        if c in df:
            df[c] = df[c].astype(float).round(2)
    return df[["year", "month", "harmonized_cell_days", "pixel_detections", "historical_median",
               "anomaly_pct_vs_median", "robust_z", "percentile", "category", "suspected_outage_days", "quality_flag"]]
