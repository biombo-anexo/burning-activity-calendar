"""End-to-end build: FIRMS files -> cleaned cell-days -> calibration -> harmonized parquet.

Usage:
    python -m burncal.pipeline --modis data/raw/MODIS/*.csv --viirs data/raw/VIIRS_SNPP/*.csv --out data/processed
"""
from __future__ import annotations

import argparse
import glob
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CFG, Config
from .harmonize import (MIN_GAP_DAYS_FLAG, block_monthly, gap_month_counts, harmonize,
                        suspected_gap_days, validate_and_fit)
from .ingest import _expand_neighbours, drop_near_static, peek_has_type, read_firms_file, to_cell_days


def _expand(patterns):
    files = []
    for p in patterns:
        hits = sorted(glob.glob(p))
        if not hits:
            raise FileNotFoundError(f"No files match {p}")
        files += hits
    return files


def coverage_boxes(ingest_stats):
    """Areas covered by the downloads: bounding box of each MODIS archive file (full period),
    rounded outward to whole degrees. Falls back to every file when there is no MODIS archive."""
    import math
    sel = [st for st in ingest_stats if st.get("bbox") and st["sensor"] == "MODIS" and st.get("has_type")] \
        or [st for st in ingest_stats if st.get("bbox")]
    boxes = []
    for st in sel:
        w, s, e, n = st["bbox"]
        b = [max(math.floor(w), -180), max(math.floor(s), -90), min(math.ceil(e), 180), min(math.ceil(n), 90)]
        if b not in boxes:
            boxes.append(b)
    return boxes


def _ingest_pandas(modis_files, viirs_files, cfg: Config, log):
    """In-memory ingestion (small areas, shapefiles). Returns raw cell-days, stats, Terra factors."""
    jobs = [(sensor, f) for sensor, files in (("MODIS", modis_files), ("VIIRS", viirs_files)) for f in files]

    # Pass 1: files WITH a TYPE column (archive). They also teach us where static sources are.
    typed = [j for j in jobs if peek_has_type(j[1])]
    untyped = [j for j in jobs if j not in typed]
    entries, static_parts = [], []
    for sensor, f in typed:
        df, st = read_firms_file(f, sensor, cfg)
        sk = st.pop("static_keys", None)
        if sk is not None and len(sk):
            static_parts.append(sk)
        entries.append([sensor, df, st])
    keys = _expand_neighbours(np.unique(np.concatenate(static_parts))) if static_parts else None

    # Same static-source removal for archive and near-real-time files, so there is no step when
    # the record switches from one to the other.
    if keys is not None:
        for e in entries:
            before = len(e[1])
            e[1], n = drop_near_static(e[1], keys)
            e[2]["removed_near_static"] = n
            e[2]["collateral_pct"] = 100.0 * n / max(before, 1)
            e[2]["rows_out"] = int(len(e[1]))
    # Pass 2: files WITHOUT TYPE (near-real-time).
    for sensor, f in untyped:
        df, st = read_firms_file(f, sensor, cfg, static_keys=keys)
        st.pop("static_keys", None)
        entries.append([sensor, df, st])

    frames, ingest_stats = [], []
    for sensor, df, st in entries:
        extra = f" near-static removed={st.get('removed_near_static', st['removed_by_type']):,}"
        if "collateral_pct" in st:
            extra += f" ({st['collateral_pct']:.1f}% of type-0)"
        log(f"  {st['file']}: {st['rows_in']:,} rows -> {st['rows_out']:,} kept "
            f"({st['date_min']} .. {st['date_max']}){extra} satellites={st['satellites']}")
        frames.append(df)
        ingest_stats.append(st)

    pix = pd.concat(frames, ignore_index=True)
    pix = pix.drop_duplicates(subset=["sensor", "date", "acq_time", "latitude", "longitude", "satellite"])
    raw = to_cell_days(pix, cfg)

    # Terra-only correction, month by month (the Terra/Aqua ratio depends on the season).
    first = pd.Timestamp(cfg.viirs_first_full_month)
    both = pix[(pix["sensor"] == "MODIS") & (pix["date"] >= pd.Timestamp(cfg.terra_only_until))
               & (pix["date"] < first)]
    terra_factors, terra_global = {}, 1.0
    if len(both) and (both["satellite"] == "Terra").any():
        cd_all = to_cell_days(both, cfg)
        cd_terra = to_cell_days(both[both["satellite"] == "Terra"], cfg)
        n_all = cd_all.groupby(cd_all["date"].dt.month).size()
        n_terra = cd_terra.groupby(cd_terra["date"].dt.month).size()
        terra_global = float(len(cd_all) / max(len(cd_terra), 1))
        terra_factors = {int(m): float(n_all.get(m, 0) / n_terra[m]) if n_terra.get(m, 0) else terra_global for m in range(1, 13)}
    return raw, ingest_stats, terra_factors, terra_global


def build(modis_files, viirs_files, out_dir, cfg: Config = CFG, log=print, label: str = "", engine: str = "pandas",
          resume: bool = False):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "ingest_cache.json"
    if resume:   # reuse a previous ingestion (cell_days_raw.parquet + ingest_cache.json in out_dir)
        c = json.loads(cache.read_text())
        raw = pd.read_parquet(out / "cell_days_raw.parquet")
        ingest_stats, terra_global = c["ingest_stats"], c["terra_global"]
        terra_factors = {int(k): v for k, v in c["terra_factors"].items()}
        log(f"  resumed ingestion from {cache.name}")
    elif engine == "duckdb":
        from . import fastingest
        raw, ingest_stats, terra_factors, terra_global = fastingest.ingest(modis_files, viirs_files, cfg, log=log)
    else:
        raw, ingest_stats, terra_factors, terra_global = _ingest_pandas(modis_files, viirs_files, cfg, log)

    # A consistent series needs ONE VIIRS satellite; warn loudly otherwise.
    viirs_sats = {s for st in ingest_stats if st["sensor"] == "VIIRS" for s in st["satellites"]}
    warnings = []
    if len(viirs_sats) > 1:
        warnings.append(f"Multiple VIIRS satellites {sorted(viirs_sats)}: cell-day counts mix satellites; "
                        "use S-NPP only for a consistent 2012+ series.")
    if "N20" in viirs_sats and len(viirs_sats) == 1:
        warnings.append("VIIRS is NOAA-20 only (starts 2018): the calibration overlap is much shorter than with S-NPP.")

    log(f"  cell-days: {len(raw):,}  ({raw.date.min().date()} .. {raw.date.max().date()})")
    if not resume:
        raw.to_parquet(out / "cell_days_raw.parquet", index=False, compression="zstd")
        cache.write_text(json.dumps({"ingest_stats": ingest_stats, "terra_factors": terra_factors,
                                     "terra_global": terra_global}, default=str))

    # The harmonized series ends where VIIRS ends (MODIS after 2012 is only used for calibration).
    viirs_end = raw.loc[raw["sensor"] == "VIIRS", "date"].max()
    modis_end = raw.loc[raw["sensor"] == "MODIS", "date"].max()
    series_end = viirs_end
    if abs((viirs_end - modis_end).days) > 7:
        warnings.append(f"MODIS ends {modis_end.date()} but VIIRS ends {viirs_end.date()}; the series ends at the VIIRS date.")
    overlap_end = min(viirs_end, modis_end)
    last_complete = overlap_end.to_period("M") if overlap_end == overlap_end.to_period("M").to_timestamp(how="end").normalize() \
        else overlap_end.to_period("M") - 1
    last_complete_ym = last_complete.year * 12 + last_complete.month - 1

    # Near-real-time data = VIIRS files without TYPE. Months touching it are 'provisional'.
    nrt_starts = [pd.Timestamp(st["date_min"]) for st in ingest_stats
                  if st["sensor"] == "VIIRS" and not st.get("has_type", True) and st["date_min"]]
    provisional_start = min(nrt_starts) if nrt_starts else (series_end.to_period("M") - (cfg.provisional_months - 1)).to_timestamp()
    provisional_start = provisional_start.to_period("M").to_timestamp()

    # Suspected satellite outages (zero-detection days amid active periods).
    first = pd.Timestamp(cfg.viirs_first_full_month)
    gaps_v = suspected_gap_days(raw, "VIIRS", first, viirs_end)
    gaps_m = suspected_gap_days(raw, "MODIS", raw["date"].min(), modis_end)
    series_gap_months = gap_month_counts(gaps_v)
    for k, v in gap_month_counts(gaps_m[gaps_m < first]).items():
        series_gap_months[k] = v
    exclude = {int(k[:4]) * 12 + int(k[5:]) - 1 for k, v in
               {**{k: v for k, v in gap_month_counts(gaps_v).items()},
                **{k: max(v, gap_month_counts(gaps_v).get(k, 0)) for k, v in gap_month_counts(gaps_m).items()}}.items()
               if v >= MIN_GAP_DAYS_FLAG}
    log(f"  suspected outage days: VIIRS {len(gaps_v)}, MODIS {len(gaps_m)}; months excluded from calibration: {len(exclude)}")

    bm = block_monthly(raw, cfg)
    model, report = validate_and_fit(bm, last_complete_ym, cfg, exclude_ym=exclude)
    log(f"  selected model: {model.name}  (held-out nMAE "
        f"{report['models'][model.name]['region_month']['nmae']*100:.1f}%)" if "models" in report and report["models"] else f"  selected model: {model.name}")

    log("  Terra-only factor by month: " + " ".join(f"{terra_factors.get(m, 1.0):.2f}" for m in range(1, 13)))
    h = harmonize(raw, model, series_end, cfg, terra_factors=terra_factors or 1.0,
                  provisional_start=provisional_start, gap_months=series_gap_months,
                  regional=report.get("regional", {}).get("factors"))
    rg = report.get("regional", {})
    if "block_year_nmae_with" in rg:
        log(f"  regional factors: held-out block-year nMAE {rg['block_year_nmae_without']*100:.1f}% -> "
            f"{rg['block_year_nmae_with']*100:.1f}%  ({'USED' if rg['used'] else 'not used'})")
    # Sorted by cell so the app can skip row groups outside the area of interest.
    h = h.sort_values(["cell_i", "cell_j", "date"], kind="stable").reset_index(drop=True)
    h.to_parquet(out / "harmonized.parquet", index=False, compression="zstd", row_group_size=100_000)

    calib = {
        "model": {"name": model.name, "params": model.params()},
        "terra_factor": terra_global,
        "terra_factors_by_month": terra_factors,
        "label": label,
        "validation": report,
        "config": asdict(cfg),
        "data": {"min_date": str(raw["date"].min().date()), "max_date": str(series_end.date()),
                 "modis_end": str(modis_end.date()), "viirs_end": str(viirs_end.date()),
                 "provisional_start": str(provisional_start.date()),
                 "gap_months": series_gap_months,
                 "gap_days_viirs": [str(d.date()) for d in gaps_v],
                 "gap_days_modis": [str(d.date()) for d in gaps_m],
                 "n_cell_days_raw": int(len(raw)), "n_harmonized_rows": int(len(h)),
                 "coverage_boxes": coverage_boxes(ingest_stats)},
        "ingest": ingest_stats,
        "warnings": warnings,
    }
    (out / "calibration.json").write_text(json.dumps(calib, indent=2, default=str))
    from . import coverage
    if not calib.get("label") and coverage.DEFAULT_SNAPSHOT.exists():   # real data only: cross-check against FIRMS' published coverage
        calib = coverage.annotate_dir(out)
        warnings = calib["warnings"]
    for w in warnings:
        log("  WARNING: " + w)
    return calib


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modis", nargs="+", default=[], help="MODIS files (csv or shp), globs allowed")
    ap.add_argument("--viirs", nargs="+", default=[], help="VIIRS S-NPP files (csv or shp), globs allowed")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--label", default="", help="Shown as a banner in the app, e.g. 'DEMO: synthetic data'")
    ap.add_argument("--resume", action="store_true",
                    help="skip reading the CSVs; reuse cell_days_raw.parquet + ingest_cache.json from --out")
    ap.add_argument("--engine", choices=["pandas", "duckdb"], default="pandas",
                    help="duckdb = out-of-core ingestion for large (continental) CSV downloads")
    a = ap.parse_args()
    print("Reading and cleaning...")
    try:
        calib = build([] if a.resume else _expand(a.modis), [] if a.resume else _expand(a.viirs), a.out,
                      label=a.label, engine=a.engine, resume=a.resume)
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(
            f"\nERROR: {e}\n\nThe calendar needs the ARCHIVE, not the 'last 7 days' files: MODIS from 2000-11 and\n"
            "VIIRS S-NPP from 2012-01, both up to a recent complete month. Request them at\n"
            "https://firms.modaps.eosdis.nasa.gov/download/ (see README)."
        )
    print(f"Done. Selected model: {calib['model']['name']}")


if __name__ == "__main__":
    main()
