"""Cross-check a processed dataset against FIRMS' published coverage (snapshot of /api/map_dates/).

Adds a `firms_coverage` block and warnings to calibration.json. It never changes the data itself.
Usage: python -m burncal.coverage data/processed [data/firms_coverage.json]
"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

DEFAULT_SNAPSHOT = Path(__file__).resolve().parent.parent / "data" / "firms_coverage.json"


def _d(s):
    return date.fromisoformat(s)


def annotate(calib: dict, cov: dict) -> dict:
    md, data = cov["map_dates"], calib["data"]
    warns, info = [], {"snapshot_date": cov["retrieved"]}

    # 1) S-NPP retirement: the reference sensor of this series can no longer be extended.
    cease = cov.get("snpp_cease_date")
    if cease:
        info["snpp_cease_date"] = cease
        if _d(data["viirs_end"]) >= _d(cease) - timedelta(days=3):
            n20, n21 = md["jpss1_viirs_c2"][0], md["jpss2_viirs_c2"][0]
            warns.append(
                f"VIIRS S-NPP stopped delivering data on {cease}, so this series ends on {data['viirs_end']} and cannot be "
                f"extended with S-NPP. To keep it updating, add NOAA-20 (archive from {n20}) or NOAA-21 (from {n21}) "
                "and calibrate them against S-NPP on their overlap.")

    # 2) Our provisional boundary should match the end of FIRMS' standard-processing (SP) archive.
    sp_end = md["suomi_viirs_c2_sp"][1]
    expected = (_d(sp_end) + timedelta(days=1)).isoformat()
    info["firms_sp_end_snpp"] = sp_end
    info["provisional_start_expected"] = expected
    info["provisional_start_matches"] = data.get("provisional_start") == expected
    if not info["provisional_start_matches"]:
        warns.append(f"Provisional months start {data.get('provisional_start')}, but FIRMS' S-NPP standard archive ends {sp_end}.")

    # 3) Overlap available to calibrate NOAA-20 against S-NPP.
    j1_start = _d(md["jpss1_viirs_c2"][0])
    snpp_end = _d(cease) if cease else _d(md["suomi_viirs_c2_nrt"][1])
    info["noaa20_overlap_years"] = round((snpp_end - j1_start).days / 365.25, 1)

    calib["firms_coverage"] = info
    have = set(calib.get("warnings", []))
    calib["warnings"] = list(calib.get("warnings", [])) + [w for w in warns if w not in have]
    return calib


def annotate_dir(out_dir, snapshot=DEFAULT_SNAPSHOT):
    p = Path(out_dir) / "calibration.json"
    calib = json.loads(p.read_text(encoding="utf-8"))
    cov = json.loads(Path(snapshot).read_text(encoding="utf-8"))
    calib.pop("firms_coverage", None)
    calib["warnings"] = [w for w in calib.get("warnings", []) if not w.startswith(("VIIRS S-NPP stopped", "Provisional months start"))]
    p.write_text(json.dumps(annotate(calib, cov), indent=2, ensure_ascii=False), encoding="utf-8")
    return calib


if __name__ == "__main__":
    c = annotate_dir(sys.argv[1], *(sys.argv[2:3]))
    print(json.dumps(c["firms_coverage"], indent=2)); [print("WARNING:", w) for w in c["warnings"]]
