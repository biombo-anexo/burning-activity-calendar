import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from burncal import analytics
from burncal.analytics import Store, _anom, _category, analyze
from burncal.config import CFG
from burncal.geo import aoi_cells

COLOMBIA = {"bbox": [-79.1, -4.3, -66.8, 13.5]}


@pytest.fixture(scope="module")
def store(synth):
    return Store(synth["out"])


@pytest.fixture(scope="module")
def client(synth):
    return TestClient(create_app(str(synth["out"])))


def test_anomaly_math():
    base = np.array([10, 20, 30, 40, 50.0])
    a = _anom(60.0, base, CFG)
    assert a["median"] == 30 and abs(a["pct"] - 100) < 1e-9 and a["percentile"] == 100.0
    assert a["category"] == "very high"
    assert _anom(1.0, np.array([1, 2.0]), CFG) is None                   # too few baseline years
    assert _category(50) == "normal" and _category(5) == "very low" and _category(80) == "above normal"


def test_calendar_sums_equal_raw_parquet(store, synth):
    cells, _ = aoi_cells(COLOMBIA, CFG.cell_deg)
    res = analyze(store, COLOMBIA)
    h = pd.read_parquet(synth["out"] / "harmonized.parquet")
    h = h.merge(cells, on=["cell_i", "cell_j"])
    assert abs(sum(c["value"] for c in res["calendar"]) - h["w"].astype(float).sum()) < 5   # rounding only


def test_flags_and_partial_months(store):
    res = analyze(store, COLOMBIA)
    flags = {(c["year"], c["month"]): c["flag"] for c in res["calendar"]}
    first = "partial" if store.min_date.day > 1 else "terra_only"        # rule: first detection after the 1st
    assert flags[(2000, 11)] == first and flags[(2026, 11)] == "partial"   # data ends mid-November
    assert flags[(2001, 6)] == "terra_only" and flags[(2002, 7)] == "ok" and flags[(2026, 8)] == "provisional"
    partial = [c for c in res["calendar"] if c["flag"] == "partial"]
    assert partial and all(c["anomaly_pct"] is None for c in partial)


def test_baseline_excludes_own_year_and_terra_only(store):
    res = analyze(store, COLOMBIA)
    jan = [c for c in res["calendar"] if c["month"] == 1]
    clim = res["climatology"][0]
    eligible = [c["value"] for c in jan if c["flag"] in ("ok", "provisional")]
    assert clim["n"] == len(eligible)                                    # terra-only Januaries excluded
    row = next(c for c in jan if c["year"] == 2016)
    others = [c["value"] for c in jan if c["flag"] in ("ok", "provisional") and c["year"] != 2016]
    assert abs(row["baseline_median"] - float(np.median(others))) < 1e-6


def test_month_to_date_present_and_sane(store):
    m = analyze(store, COLOMBIA)["month_to_date"]
    md = store.max_date
    assert (m["year"], m["month"], m["day"]) == (md.year, md.month, md.day) and m["provisional"]
    assert m["n_baseline"] >= 20


def test_seasonality_and_el_nino_years_recovered(store):
    res = analyze(store, {"bbox": [-72.5, 3, -69, 7]})                   # Llanos: Jan-Mar peak by construction
    assert {p["month"] for p in res["summary"]["peak_calendar_months"]} == {1, 2, 3}
    assert {t["year"] for t in res["top_years"][:3]} & {2002, 2010, 2016, 2020, 2024}


def test_date_range_and_validation(store):
    res = analyze(store, COLOMBIA, "2019-01-01", "2020-12-31")
    assert len(res["calendar"]) == 24
    with pytest.raises(ValueError):
        analyze(store, COLOMBIA, "2020-01-01", "2019-01-01")


def test_api_meta_and_strict_json(client):
    m = client.get("/api/meta").json()
    assert m["label"] == "TEST" and m["validation"]["selected"] in ("ratio", "power")
    r = client.post("/api/analyze", json={"aoi": COLOMBIA})
    assert r.status_code == 200
    # Regression: partial months used to produce NaN -> invalid JSON -> 500.
    json.loads(r.text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))


@pytest.mark.parametrize("payload", [
    {"aoi": {}}, {"aoi": {"bbox": [-69, 3, -72, 6]}}, {"aoi": {"bbox": [-72, 3, -69, 6], "polygon": [[0, 0], [1, 1], [1, 0]]}},
    {"aoi": COLOMBIA, "start": "2020-01-01", "end": "2019-01-01"}, {"aoi": COLOMBIA, "start": "not-a-date"},
])
def test_api_rejects_bad_input_with_4xx(client, payload):
    assert client.post("/api/analyze", json=payload).status_code in (422, 400)


def test_csv_downloads(client):
    p = {"bbox": "-72,3,-69,6", "start": "2016-01-01", "end": "2016-12-31"}
    s = client.get("/api/download/summary.csv", params=p)
    lines = s.text.strip().splitlines()
    assert s.status_code == 200 and len(lines) == 13 and lines[0].startswith("year,month,harmonized_cell_days")
    hs = client.get("/api/download/hotspots.csv", params=p)
    rows = hs.text.strip().splitlines()
    assert hs.status_code == 200 and len(rows) - 1 == int(hs.headers["x-row-count"]) > 0
    assert rows[0].split(",")[:4] == ["date", "latitude", "longitude", "source_sensor"]
    poly = client.get("/api/download/hotspots.csv", params={"polygon": "-72,3;-69,3;-69,6", "start": "2016-01-01", "end": "2016-03-31"})
    box = client.get("/api/download/hotspots.csv", params={"bbox": "-72,3,-69,6", "start": "2016-01-01", "end": "2016-03-31"})
    assert 0 < int(poly.headers["x-row-count"]) < int(box.headers["x-row-count"])      # triangle < its bbox
    assert client.get("/api/download/hotspots.csv").status_code == 422


def test_missing_data_dir_gives_503(tmp_path):
    assert TestClient(create_app(str(tmp_path / "nope"))).get("/api/meta").status_code == 503
