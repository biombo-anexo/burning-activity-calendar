"""FastAPI backend. Run:  BURNCAL_DATA=data/processed uvicorn app.main:app --reload"""
from __future__ import annotations

import io
import json
import os
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from burncal import analytics
from burncal.geo import validate_aoi

STATIC = Path(__file__).parent / "static"
MAX_CSV_ROWS = 2_000_000
# Colombia's border (Natural Earth 1:50m, public domain), simplified to ~0.02 deg: the default area.
_COL = STATIC / "colombia.json"
DEFAULT_AOI = {"polygon": json.loads(_COL.read_text())} if _COL.exists() else {"bbox": [-79.1, -4.3, -66.8, 13.5]}
# Colombian quick areas = their approximate boxes clipped to the border, so neighbours' fires are not counted.
_REG = STATIC / "colombia_regions.json"
REGION_POLYGONS = json.loads(_REG.read_text()) if _REG.exists() else {}


class AOI(BaseModel):
    bbox: Optional[list[float]] = None
    polygon: Optional[list[list[float]]] = None


class AnalyzeRequest(BaseModel):
    aoi: AOI
    start: Optional[date] = None
    end: Optional[date] = None
    baseline_start_year: Optional[int] = None
    baseline_end_year: Optional[int] = None


def _aoi_from_query(bbox: Optional[str], polygon: Optional[str]) -> dict:
    """bbox='w,s,e,n'  polygon='lon,lat;lon,lat;...'"""
    try:
        if bbox:
            return validate_aoi({"bbox": [float(x) for x in bbox.split(",")]})
        if polygon:
            return validate_aoi({"polygon": [[float(v) for v in p.split(",")] for p in polygon.split(";")]})
    except ValueError as e:
        raise HTTPException(422, str(e))
    raise HTTPException(422, "Provide bbox or polygon.")


def create_app(data_dir: Optional[str] = None) -> FastAPI:
    app = FastAPI(title="Burning Activity Calendar", version="1.0")
    app.add_middleware(GZipMiddleware, minimum_size=2000)   # map and calendar payloads compress ~5x
    holder = {"store": None}

    def store() -> analytics.Store:
        if holder["store"] is None:
            try:
                holder["store"] = analytics.Store(data_dir or os.environ.get("BURNCAL_DATA", "data/processed"))
            except FileNotFoundError as e:
                raise HTTPException(503, str(e))
        return holder["store"]

    @app.get("/api/meta")
    def meta():
        s = store()
        c = s.calib
        return {
            "label": c.get("label", ""), "data_range": c["data"], "model": c["model"], "terra_factor": c.get("terra_factor"),
            "validation": c["validation"], "warnings": c.get("warnings", []),
            "cell_deg": s.cfg.cell_deg, "provisional_months": s.cfg.provisional_months,
            "default_bbox": [-79.1, -4.3, -66.8, 13.5],
            "default_aoi": DEFAULT_AOI,
            "region_polygons": REGION_POLYGONS,
            "coverage_boxes": c["data"].get("coverage_boxes", []),
            "units": "VIIRS-equivalent fire cell-days (0.1 degree cells with >=1 detection per day)",
        }

    @app.post("/api/analyze")
    def analyze(req: AnalyzeRequest):
        s = store()
        by = None
        if req.baseline_start_year or req.baseline_end_year:
            lo = req.baseline_start_year or s.min_date.year
            hi = req.baseline_end_year or s.max_date.year
            if lo > hi:
                raise HTTPException(422, "Baseline start year is after end year.")
            by = (lo, hi)
        try:
            return analytics.analyze(s, req.aoi.model_dump(exclude_none=True), req.start, req.end, by)
        except ValueError as e:
            raise HTTPException(422, str(e))

    @app.post("/api/density")
    def density(req: AnalyzeRequest):
        s = store()
        try:
            return analytics.density(s, req.aoi.model_dump(exclude_none=True), req.start, req.end)
        except ValueError as e:
            raise HTTPException(422, str(e))

    @app.get("/api/download/summary.csv")
    def summary_csv(bbox: Optional[str] = None, polygon: Optional[str] = None,
                    start: Optional[date] = None, end: Optional[date] = None):
        s = store()
        aoi = _aoi_from_query(bbox, polygon)
        try:
            res = analytics.analyze(s, aoi, start, end)
        except ValueError as e:
            raise HTTPException(422, str(e))
        buf = io.StringIO()
        analytics.summary_csv_frame(res).to_csv(buf, index=False)
        return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                                 headers={"Content-Disposition": 'attachment; filename="burning_calendar_summary.csv"'})

    @app.get("/api/download/hotspots.csv")
    def hotspots_csv(bbox: Optional[str] = None, polygon: Optional[str] = None,
                     start: Optional[date] = None, end: Optional[date] = None):
        s = store()
        aoi = _aoi_from_query(bbox, polygon)
        try:
            q, n = analytics.hotspot_rows(s, aoi, start, end)
        except ValueError as e:
            raise HTTPException(422, str(e))
        if n > MAX_CSV_ROWS:
            raise HTTPException(413, f"{n:,} rows is too many for one download (limit {MAX_CSV_ROWS:,}). "
                                     "Narrow the area or date range.")
        cols = [d[0] for d in q.description]

        def gen():
            yield ",".join(cols) + "\n"
            while True:
                rows = q.fetchmany(50_000)
                if not rows:
                    break
                yield pd.DataFrame(rows, columns=cols).to_csv(index=False, header=False)

        return StreamingResponse(gen(), media_type="text/csv",
                                 headers={"Content-Disposition": 'attachment; filename="hotspots_cell_days.csv"',
                                          "X-Row-Count": str(n)})

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
