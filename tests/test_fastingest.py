"""The DuckDB (out-of-core) ingestion must give exactly the same result as the pandas one."""
import pandas as pd

from burncal import fastingest
from burncal.config import CFG
from burncal.pipeline import _ingest_pandas, build


def _files(synth):
    return [synth["raw"] / "MODIS_synthetic.csv"], [synth["raw"] / "VIIRS_synthetic.csv"]


def test_duckdb_and_pandas_give_identical_cell_days(synth, tmp_path):
    m, v = _files(synth)
    a = _ingest_pandas(m, v, CFG, lambda *_: None)
    b = fastingest.ingest(m, v, CFG, log=lambda *_: None, workdir=tmp_path / "w")
    k = ["date", "cell_i", "cell_j", "sensor"]
    ra = a[0].sort_values(k).reset_index(drop=True)
    rb = b[0].sort_values(k).reset_index(drop=True)
    assert len(ra) == len(rb) > 0
    pd.testing.assert_frame_equal(ra[k + ["n_det"]], rb[k + ["n_det"]], check_dtype=False)
    assert (ra["frp_sum"] - rb["frp_sum"]).abs().max() < 1e-6
    assert a[2] == pytest_approx_dict(b[2]) and abs(a[3] - b[3]) < 1e-9
    sa = {s["file"]: s for s in a[1]}
    for s in b[1]:
        for key in ("rows_in", "rows_valid", "removed_by_type", "removed_by_confidence", "rows_out", "has_type"):
            assert s[key] == sa[s["file"]][key], (s["file"], key)


def pytest_approx_dict(d):
    import pytest
    return {k: pytest.approx(v) for k, v in d.items()}


def test_full_build_with_duckdb_engine(synth, tmp_path):
    m, v = _files(synth)
    calib = build(m, v, tmp_path / "out", label="TEST", log=lambda *_: None, engine="duckdb")
    assert calib["model"]["name"] == synth["calib"]["model"]["name"]
    assert calib["data"]["n_cell_days_raw"] == synth["calib"]["data"]["n_cell_days_raw"]
    h_new = pd.read_parquet(tmp_path / "out" / "harmonized.parquet")
    h_old = pd.read_parquet(synth["out"] / "harmonized.parquet")
    assert abs(h_new["w"].sum() - h_old["w"].sum()) < 1e-3 * h_old["w"].sum()
