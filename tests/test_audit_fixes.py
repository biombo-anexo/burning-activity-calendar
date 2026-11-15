"""Regression tests for problems found by auditing the real FIRMS data."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from burncal.geo import validate_aoi
from burncal.harmonize import SeasonalPowerModel, gap_month_counts, suspected_gap_days
from burncal.ingest import drop_near_static, _encode, _expand_neighbours

COLOMBIA = {"bbox": [-79.1, -4.3, -66.8, 13.5]}


def _raw(daily_counts: dict, sensor="VIIRS"):
    rows = [{"date": pd.Timestamp(d), "sensor": sensor} for d, n in daily_counts.items() for _ in range(n)]
    return pd.DataFrame(rows)


def test_outage_detected_but_quiet_season_zeros_are_not():
    days = pd.date_range("2022-06-01", "2022-09-30")
    counts = {d: 60 for d in days}
    for d in pd.date_range("2022-07-27", "2022-08-10"):
        counts[d] = 0                                      # 15-day outage amid active days
    gaps = suspected_gap_days(_raw({d: n for d, n in counts.items() if n}), "VIIRS", days[0], days[-1])
    assert len(gaps) == 15 and gap_month_counts(gaps) == {"2022-07": 5, "2022-08": 10}
    quiet = {d: (1 if d.day % 3 == 0 else 0) for d in days}            # genuinely quiet wet season
    assert len(suspected_gap_days(_raw({d: n for d, n in quiet.items() if n}), "VIIRS", days[0], days[-1])) == 0


def test_single_zero_day_flagged_only_when_neighbours_very_active():
    days = pd.date_range("2020-01-01", "2020-03-31")
    busy = {d: 100 for d in days if d != pd.Timestamp("2020-02-10")}
    assert len(suspected_gap_days(_raw(busy), "VIIRS", days[0], days[-1])) == 1
    modest = {d: 10 for d in days if d != pd.Timestamp("2020-02-10")}
    assert len(suspected_gap_days(_raw(modest), "VIIRS", days[0], days[-1])) == 0


def test_seasonal_model_fits_a_seasonal_relationship_that_a_single_curve_cannot():
    from burncal.harmonize import PowerModel
    rng = np.random.default_rng(1)
    M = rng.integers(2, 300, 4000).astype(float)
    month = rng.integers(1, 13, 4000)
    ratio = 3.0 * np.exp(0.3 * np.cos(2 * np.pi * (month - 0.5) / 12))   # V/M swings ~1.8x through the year
    V = np.round(M * ratio * rng.lognormal(0, 0.05, 4000))
    seasonal = SeasonalPowerModel().fit(M, V, month)
    plain = PowerModel().fit(M, V, month)

    def worst_month_error(model):
        return max(abs(model.predict(M[month == k], month[month == k]).sum() / V[month == k].sum() - 1) for k in range(1, 13))

    assert worst_month_error(seasonal) < 0.06          # recovers the seasonality
    assert worst_month_error(plain) > 0.15             # a single curve cannot
    with pytest.raises(ValueError):
        SeasonalPowerModel().fit(M, V, None)


def test_static_sources_dropped_with_neighbour_tolerance():
    keys = _expand_neighbours(np.unique(_encode([5.2000], [-72.1000])))
    df = pd.DataFrame({"latitude": [5.2000, 5.2095, 5.30], "longitude": [-72.1000, -72.1000, -72.1000]})
    kept, n = drop_near_static(df, keys)
    assert n == 2 and kept["latitude"].tolist() == [5.30]              # exact + within 0.01 deg removed


def test_real_pipeline_removes_same_static_sources_everywhere(synth):
    stats = {s["file"]: s for s in synth["calib"]["ingest"]}
    assert "removed_near_static" in stats["VIIRS_synthetic.csv"]        # archive-type files get the proximity rule too


def test_provisional_and_gap_flags_from_metadata(synth):
    d = synth["calib"]["data"]
    assert "provisional_start" in d and "gap_months" in d
    h = pd.read_parquet(synth["out"] / "harmonized.parquet")
    assert set(h["flag"]) <= {"ok", "terra_only", "provisional", "gaps"}


@pytest.fixture(scope="module")
def client(synth):
    return TestClient(create_app(str(synth["out"])))


def test_end_before_start_is_422_on_every_endpoint(client):
    p = {"start": "2020-01-01", "end": "2019-01-01"}
    assert client.post("/api/density", json={"aoi": COLOMBIA, **p}).status_code == 422
    q = {"bbox": "-72,3,-69,6", **p}
    assert client.get("/api/download/hotspots.csv", params=q).status_code == 422
    assert client.get("/api/download/summary.csv", params=q).status_code == 422


def test_polygon_vertex_limit():
    poly = [[-72 + 0.0001 * i, 3 + (i % 2) * 0.1] for i in range(5001)]
    with pytest.raises(ValueError):
        validate_aoi({"polygon": poly})


def test_month_to_date_needs_at_least_7_days(synth):
    from burncal.analytics import Store, analyze
    st = Store(synth["out"])
    m = analyze(st, COLOMBIA)["month_to_date"]
    assert (m is None) == (st.max_date.day < 7)


def test_coverage_flags_snpp_retirement_and_checks_provisional_boundary():
    import json
    from burncal.coverage import annotate, DEFAULT_SNAPSHOT
    cov = json.loads(DEFAULT_SNAPSHOT.read_text())
    calib = {"data": {"viirs_end": "2026-11-01", "provisional_start": "2026-08-01"}, "warnings": []}
    out = annotate(calib, cov)
    assert out["firms_coverage"]["provisional_start_matches"] is True
    assert any("S-NPP stopped" in w and "NOAA-20" in w for w in out["warnings"])
    # a dataset that ends long before the S-NPP end is not told it was cut off; a wrong boundary is reported
    old = annotate({"data": {"viirs_end": "2024-06-30", "provisional_start": "2024-02-01"}, "warnings": []}, cov)
    assert not any("S-NPP stopped" in w for w in old["warnings"])
    assert old["firms_coverage"]["provisional_start_matches"] is False and any("Provisional months" in w for w in old["warnings"])
    # idempotent: annotating twice does not duplicate warnings
    assert annotate(out, cov)["warnings"].count(out["warnings"][0]) == 1


def test_regional_factors_fix_blocks_that_a_single_curve_misconverts():
    from burncal.harmonize import regional_factors, apply_regional
    rng = np.random.default_rng(3)
    n = 600
    ov = pd.DataFrame({"bi": np.repeat([0, 1, 2], n), "bj": 0, "ym": np.tile(np.arange(n), 3),
                       "MODIS": rng.integers(20, 200, 3 * n)})
    true = np.repeat([2.0, 3.0, 4.0], n)                    # VIIRS/MODIS differs by region
    ov["VIIRS"] = np.round(ov["MODIS"] * true)
    pred = ov["MODIS"].values * 3.0                         # one country-wide ratio
    f = regional_factors(ov, pred)
    fixed = pred * apply_regional(ov["bi"].values, ov["bj"].values, f)
    for b, t in zip([0, 1, 2], [2.0, 3.0, 4.0]):
        sel = ov["bi"].values == b
        assert abs(fixed[sel].sum() / ov["VIIRS"].values[sel].sum() - 1) < 0.05
    assert abs(fixed.sum() / pred.sum() - 1) < 1e-6          # overall total untouched
    # a block with almost no data keeps a factor near 1 (shrinkage), and unknown blocks get exactly 1
    tiny = pd.DataFrame({"bi": [9], "bj": [9], "ym": [0], "MODIS": [1], "VIIRS": [9]})
    assert abs(list(regional_factors(tiny, np.array([3.0])).values())[0] - 1) < 0.05
    assert apply_regional([7], [7], f)[0] == 1.0


def test_regional_step_is_reported_and_only_used_when_it_helps(synth):
    rg = synth["calib"]["validation"]["regional"]
    assert "block_year_nmae_with" in rg and isinstance(rg["used"], bool)
    assert rg["used"] == (rg["block_year_nmae_with"] < rg["block_year_nmae_without"] * 0.95)


def test_regional_sparse_block_borrows_its_parent_region_factor():
    """A block with almost no data must take its region's factor, not the global one (boreal fix)."""
    import pandas as pd
    from burncal.harmonize import regional_factors
    rows = []
    # Region A (blocks 0..4): VIIRS is half the prediction, lots of data in block (0,0), one sparse block (0,1).
    rows += [(0, 0, 10_000, 20_000), (0, 1, 5, 10)]
    # Region B (blocks 10..14): VIIRS equals the prediction.
    rows += [(10, 10, 20_000, 20_000)]
    ov = pd.DataFrame(rows, columns=["bi", "bj", "VIIRS", "P"])
    f_h = regional_factors(ov, ov["P"].values.astype(float), k=300, parent=5)
    f_flat = regional_factors(ov, ov["P"].values.astype(float), k=300, parent=None)
    # Single-level: the sparse block is pulled to ~1 (times the global rescale); hierarchical: to ~0.5.
    assert f_h[(0, 1)] < 0.75 * f_flat[(0, 1)]
    assert abs(f_h[(0, 1)] / f_h[(0, 0)] - 1) < 0.1
    assert f_h[(10, 10)] > 1.5 * f_h[(0, 0)]


def test_sharded_data_gives_identical_results(synth, tmp_path):
    """Deployment splits harmonized.parquet into parts (GitHub web limit); the app must read them identically."""
    import shutil
    from burncal import analytics, shard
    d = tmp_path / "sharded"
    shutil.copytree(synth["out"], d)
    shard.shard(d, max_mb=0.05, remove_original=True)
    assert len(list((d / "harmonized_parts").glob("*.parquet"))) > 1 and not (d / "harmonized.parquet").exists()
    aoi = {"bbox": [-75.0, 2.0, -70.0, 7.0]}
    a = analytics.analyze(analytics.Store(synth["out"]), aoi)
    b = analytics.analyze(analytics.Store(d), aoi)
    assert abs(a["summary"]["total_in_range"] - b["summary"]["total_in_range"]) < 1e-6
    assert a["calendar"] == b["calendar"]
