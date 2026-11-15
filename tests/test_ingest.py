import pandas as pd
import shapefile

from burncal.ingest import read_firms_file, to_cell_days


def _csv(tmp_path, name, rows, cols):
    p = tmp_path / name
    pd.DataFrame(rows, columns=cols).to_csv(p, index=False)
    return p


MODIS_COLS = ["latitude", "longitude", "acq_date", "acq_time", "satellite", "confidence", "frp", "type"]


def test_modis_filters_type_confidence_and_duplicates(tmp_path):
    rows = [
        [4.01, -72.01, "2016-02-01", "0100", "Terra", 80, 5, 0],   # kept
        [4.01, -72.01, "2016-02-01", "0100", "Terra", 80, 5, 0],   # exact duplicate
        [4.02, -72.02, "2016-02-01", "0100", "Aqua", 29, 5, 0],    # low confidence
        [4.03, -72.03, "2016-02-01", "0100", "Aqua", 90, 5, 2],    # static source (gas flare)
        [4.04, -72.04, "2016-02-01", "0100", "Aqua", 90, 5, 3],    # offshore
        [4.05, -72.05, "2016-02-02", "0100", "Aqua", 30, 5, 0],    # kept (boundary 30)
    ]
    df, st = read_firms_file(_csv(tmp_path, "m.csv", rows, MODIS_COLS), "MODIS")
    assert len(df) == 2 and st["duplicates_removed"] == 1 and st["removed_by_type"] == 2
    assert st["removed_by_confidence"] == 1


def test_viirs_letter_confidence(tmp_path):
    rows = [[4.0, -72.0, "2020-01-01", "0100", "N", c, 3, 0] for c in ["l", "n", "h", "H", "x"]]
    for i, r in enumerate(rows):
        r[2] = f"2020-01-0{i+1}"   # distinct so none are duplicates
    df, st = read_firms_file(_csv(tmp_path, "v.csv", rows, MODIS_COLS), "VIIRS")
    assert len(df) == 3             # n, h, H


def test_missing_required_column(tmp_path):
    p = _csv(tmp_path, "bad.csv", [[1, 2]], ["latitude", "longitude"])
    try:
        read_firms_file(p, "MODIS")
        assert False
    except ValueError as e:
        assert "acq_date" in str(e)


def test_shapefile_uppercase_without_type_column(tmp_path):
    """Mimics the real NRT shapefiles: uppercase names, no TYPE column, string dates."""
    base = tmp_path / "fire_nrt_test"
    w = shapefile.Writer(str(base))
    for name, kind in [("LATITUDE", "N"), ("LONGITUDE", "N"), ("ACQ_DATE", "C"), ("ACQ_TIME", "C"),
                       ("SATELLITE", "C"), ("CONFIDENCE", "C"), ("FRP", "N")]:
        w.field(name, kind, size=20, decimal=5 if kind == "N" else 0)
    w.point(-72.0, 4.0); w.record(4.0, -72.0, "2026-11-11", "0000", "N20", "n", 2.0)
    w.point(-72.0, 4.0); w.record(4.001, -72.001, "2026-11-11", "0010", "N20", "l", 2.0)
    w.close()
    df, st = read_firms_file(base.with_suffix(".shp"), "VIIRS")
    assert len(df) == 1 and "TYPE column absent" in st["type_filter"]
    assert df["satellite"].iloc[0] == "N20"


def test_cell_days_collapse_pixels(tmp_path):
    rows = [[4.01, -72.01, "2016-02-01", "0100", "Terra", 80, 5, 0],
            [4.02, -72.02, "2016-02-01", "0200", "Aqua", 80, 7, 0],     # same 0.1 cell, same day
            [4.51, -72.01, "2016-02-01", "0100", "Terra", 80, 1, 0]]    # different cell
    df, _ = read_firms_file(_csv(tmp_path, "m.csv", rows, MODIS_COLS), "MODIS")
    cd = to_cell_days(df)
    assert len(cd) == 2 and sorted(cd["n_det"]) == [1, 2]
