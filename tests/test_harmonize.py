import numpy as np
import pandas as pd

from burncal.harmonize import PowerModel, RatioModel, month_index
from burncal.ingest import read_firms_file, to_cell_days


def test_ratio_model_recovers_known_ratio():
    M = np.array([10, 20, 30, 40.0]); V = 2.5 * M
    m = RatioModel().fit(M, V)
    assert abs(m.r - 2.5) < 1e-9 and np.allclose(m.predict(M), V)


def test_power_model_recovers_known_curve_and_zero_maps_to_zero():
    rng = np.random.default_rng(0)
    M = rng.integers(1, 500, 2000).astype(float)
    V = np.round(1.3 * M ** 1.1)
    m = PowerModel().fit(M, V)
    assert abs(m.b - 1.1) < 0.05
    assert m.predict(np.array([0.0]))[0] == 0
    assert abs(m.predict(M).sum() - V.sum()) / V.sum() < 0.05


def test_pipeline_validation_report(synth):
    v = synth["calib"]["validation"]
    assert v["selected"] in ("ratio", "power")
    assert v["test_years"][0] > v["train_years"][1]                      # genuinely held out
    best = v["models"][v["selected"]]["region_month"]["nmae"]
    assert best < v["no_harmonization"]["nmae"] / 3                      # harmonizing clearly helps
    assert v["models"][v["selected"]]["region_month"]["r"] > 0.95


def test_harmonized_pre2012_matches_hidden_truth(synth):
    """The pipeline never sees truth_viirs_pre2012.csv: it is what VIIRS would have recorded."""
    truth_px, _ = read_firms_file(synth["raw"] / "truth_viirs_pre2012.csv", "VIIRS")
    tcd = to_cell_days(truth_px)
    h = pd.read_parquet(synth["out"] / "harmonized.parquet")
    h = h[h["date"] < "2012-02-01"]
    truth = pd.Series(1.0, index=tcd.index).groupby(month_index(tcd["date"])).sum()
    harm = h["w"].astype(float).groupby(month_index(h["date"])).sum()
    idx = truth.index.union(harm.index)
    df = pd.DataFrame({"t": truth.reindex(idx, fill_value=0), "h": harm.reindex(idx, fill_value=0)})
    df = df[df.index >= 2000 * 12 + 11]                                  # skip partial Nov 2000
    nmae = (df.t - df.h).abs().mean() / df.t.mean()
    assert nmae < 0.15 and np.corrcoef(df.t, df.h)[0, 1] > 0.95
    terra = df[df.index < 2002 * 12 + 6]                                 # Terra-only era
    assert abs((terra.h.sum() - terra.t.sum()) / terra.t.sum()) < 0.2    # the correction works


def test_no_junk_and_flags(synth):
    h = pd.read_parquet(synth["out"] / "harmonized.parquet")
    assert not h.duplicated(["date", "cell_i", "cell_j", "sensor"]).any()
    assert (h["w"] >= 0).all()
    assert h.loc[h["sensor"] == "VIIRS", "date"].min() >= pd.Timestamp("2012-02-01")
    assert h.loc[h["sensor"] == "MODIS", "date"].max() < pd.Timestamp("2012-02-01")
    assert set(h["flag"]) <= {"ok", "terra_only", "provisional"}
    assert (h.loc[h["flag"] == "terra_only", "date"] < "2002-07-04").all()


def test_static_flares_removed(synth):
    """Gas flares are fixed points detected nearly every day; none may survive the type filter."""
    h = pd.read_parquet(synth["out"] / "harmonized.parquet")
    per_cell = h[h["date"] >= "2016-01-01"].groupby(["cell_i", "cell_j"]).size()
    assert per_cell.max() < 1500        # a flare cell would have ~3000 cell-days since 2016
