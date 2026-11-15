"""Out-of-core ingestion with DuckDB, for continental downloads that do not fit in RAM.

Produces exactly what the pandas path in ingest.py + pipeline.py produces (same filters, same
order of definitions, same cell-day table), but streams the CSVs through DuckDB with spilling to
disk. tests/test_fastingest.py checks that both paths give identical cell-days on the same files.

Filters (identical to ingest.read_firms_file):
  * invalid lat/lon/date dropped; duplicates (date, acq_time, lat, lon, satellite) dropped per file
  * TYPE column present -> keep cfg.keep_types; TYPE 1/2 locations become "static source" keys
  * every detection within +/-0.01 deg of a static source is dropped (archive and NRT alike)
  * confidence: MODIS numeric >= cfg.modis_min_conf; VIIRS 'n'/'h' (or numeric if the file is numeric)
  * duplicates across files dropped on (sensor, date, acq_time, lat, lon, satellite)
CSV only (shapefiles go through the pandas path).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import duckdb
import pandas as pd

from .config import CFG, Config
from .ingest import STATIC_GRID, STATIC_TYPES


def _q(p) -> str:
    return "'" + str(Path(p).as_posix()).replace("'", "''") + "'"


def _key(lat: str, lon: str) -> str:
    return (f"((CAST(floor({lat} / {STATIC_GRID} + 1e-9) AS BIGINT) + 10000) * 100000 "
            f"+ (CAST(floor({lon} / {STATIC_GRID} + 1e-9) AS BIGINT) + 20000))")


def ingest(modis_files, viirs_files, cfg: Config = CFG, log=print, workdir=None,
           memory_limit: str = "3GB", threads: int | None = None):
    """Return (raw cell-days DataFrame, ingest_stats list, terra_factors dict, terra_global float)."""
    tmpdir = Path(workdir or tempfile.mkdtemp(prefix="burncal_"))
    tmpdir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(tmpdir / "ingest.duckdb"))
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET temp_directory={_q(tmpdir / 'spill')}")
    con.execute("SET preserve_insertion_order=false")
    if threads:
        con.execute(f"SET threads={int(threads)}")

    jobs = [(s, Path(f)) for s, fs in (("MODIS", modis_files), ("VIIRS", viirs_files)) for f in fs]
    keep_types = ",".join(str(int(t)) for t in cfg.keep_types)
    static_types = ",".join(str(int(t)) for t in STATIC_TYPES)
    viirs_ok = ",".join("'" + c + "'" for c in cfg.viirs_ok_conf)

    # 1) Parse every file once into a narrow table; drop invalid rows and per-file duplicates.
    tables = []
    for k, (sensor, f) in enumerate(jobs):
        if f.suffix.lower() != ".csv":
            raise ValueError(f"{f.name}: the DuckDB engine reads CSV only; use the pandas engine for shapefiles")
        orig = [r[0] for r in con.execute(
            f"DESCRIBE SELECT * FROM read_csv({_q(f)}, all_varchar=true, header=true)").fetchall()]
        cmap = {c.strip().lower(): '"' + c.replace('"', '""') + '"' for c in orig}
        cols = list(cmap)
        for c in ("latitude", "longitude", "acq_date"):
            if c not in cols:
                raise ValueError(f"{f.name}: missing required column {c}")
        col = lambda c: cmap.get(c, "NULL")   # noqa: E731
        t = f"f{k}"
        con.execute(f"""
            CREATE TABLE {t}_in AS SELECT
                TRY_CAST({col('latitude')} AS DOUBLE) AS lat, TRY_CAST({col('longitude')} AS DOUBLE) AS lon,
                TRY_CAST({col('acq_date')} AS DATE) AS date, CAST({col('acq_time')} AS VARCHAR) AS acq_time,
                CAST({col('satellite')} AS VARCHAR) AS satellite,
                TRY_CAST({col('frp')} AS DOUBLE) AS frp,
                TRY_CAST({col('type')} AS INTEGER) AS type,
                lower(trim(CAST({col('confidence')} AS VARCHAR))) AS conf
            FROM read_csv({_q(f)}, all_varchar=true, header=true)""")
        rows_in = con.execute(f"SELECT count(*) FROM {t}_in").fetchone()[0]
        con.execute(f"""
            CREATE TABLE {t} AS SELECT lat, lon, date, acq_time, satellite,
                   first(frp) AS frp, first(type) AS type, first(conf) AS conf
            FROM {t}_in WHERE lat IS NOT NULL AND lon IS NOT NULL AND date IS NOT NULL
            GROUP BY ALL""")
        rows_valid = con.execute(f"SELECT count(*) FROM {t}_in WHERE lat IS NOT NULL AND lon IS NOT NULL "
                                 "AND date IS NOT NULL").fetchone()[0]
        con.execute(f"DROP TABLE {t}_in")
        n = con.execute(f"SELECT count(*), count(type) FROM {t}").fetchone()
        st = {"file": f.name, "sensor": sensor, "rows_in": int(rows_in), "has_type": bool(n[1] > 0),
              "rows_valid": int(rows_valid), "duplicates_removed": int(rows_valid - n[0])}
        tables.append((sensor, t, st))
        log(f"  read {f.name}: {rows_in:,} rows")

    # 2) Static sources learned from files that carry TYPE, expanded to the 3x3 neighbourhood.
    typed = [t for _, t, st in tables if st["has_type"]]
    if typed:
        union = " UNION ALL ".join(f"SELECT lat, lon FROM {t} WHERE type IN ({static_types})" for t in typed)
        con.execute(f"CREATE TABLE static0 AS SELECT DISTINCT {_key('lat', 'lon')} AS k FROM ({union})")
        con.execute("""CREATE TABLE static AS SELECT DISTINCT k + di * 100000 + dj AS k FROM static0,
                       (SELECT unnest([-1,0,1]) AS di), (SELECT unnest([-1,0,1]) AS dj)""")
    else:
        con.execute("CREATE TABLE static (k BIGINT)")
    n_static = con.execute("SELECT count(*) FROM static").fetchone()[0]

    # 3) Filter each file and collect stats.
    stats, parts = [], []
    for sensor, t, st in tables:
        if sensor == "MODIS":
            conf_ok = f"TRY_CAST(conf AS DOUBLE) >= {cfg.modis_min_conf}"
        else:
            num_share = con.execute(f"SELECT avg(CASE WHEN TRY_CAST(conf AS DOUBLE) IS NOT NULL THEN 1 ELSE 0 END) "
                                    f"FROM {t}").fetchone()[0] or 0
            conf_ok = (f"TRY_CAST(conf AS DOUBLE) >= {cfg.modis_min_conf}" if num_share > 0.9
                       else f"conf IN ({viirs_ok})")
        near = f"({_key('lat', 'lon')} IN (SELECT k FROM static))"
        if st["has_type"]:
            type_ok = f"type IN ({keep_types})"
            c = con.execute(f"""SELECT count(*) FILTER (WHERE NOT coalesce({type_ok}, false)),
                count(*) FILTER (WHERE coalesce({type_ok}, false) AND NOT coalesce({conf_ok}, false)),
                count(*) FILTER (WHERE coalesce({type_ok}, false) AND coalesce({conf_ok}, false) AND {near}),
                count(*) FILTER (WHERE coalesce({type_ok}, false) AND coalesce({conf_ok}, false))
                FROM {t}""").fetchone()
            st.update(removed_by_type=int(c[0]), removed_by_confidence=int(c[1]),
                      removed_near_static=int(c[2]), collateral_pct=100.0 * c[2] / max(c[3], 1),
                      type_filter="applied (TYPE column)")
            keep = f"coalesce({type_ok}, false) AND coalesce({conf_ok}, false) AND NOT {near}"
        else:
            c = con.execute(f"""SELECT count(*) FILTER (WHERE {near}),
                count(*) FILTER (WHERE NOT {near} AND NOT coalesce({conf_ok}, false)) FROM {t}""").fetchone()
            st.update(removed_by_type=int(c[0]), removed_by_confidence=int(c[1]),
                      type_filter=("TYPE column absent: dropped detections next to static sources learned from "
                                   "archive files") if n_static else
                                  "TYPE column absent and no archive to learn static sources from: flares NOT removed")
            keep = f"NOT {near} AND coalesce({conf_ok}, false)"
        con.execute(f"CREATE TABLE {t}_ok AS SELECT '{sensor}' AS sensor, date, lat, lon, "
                    f"coalesce(satellite, '?') AS satellite, frp, acq_time FROM {t} WHERE {keep}")
        con.execute(f"DROP TABLE {t}")
        r = con.execute(f"SELECT count(*), min(date), max(date), min(lon), min(lat), max(lon), max(lat) FROM {t}_ok").fetchone()
        if r[0]:
            st["bbox"] = [float(x) for x in r[3:7]]
        sats = dict(con.execute(f"SELECT satellite, count(*) FROM {t}_ok GROUP BY 1 ORDER BY 2 DESC").fetchall())
        st.update(rows_out=int(r[0]), date_min=str(r[1]) if r[1] else None, date_max=str(r[2]) if r[2] else None,
                  satellites={k: int(v) for k, v in sats.items()})
        stats.append(st)
        parts.append(f"{t}_ok")
        log(f"  {st['file']}: {st['rows_in']:,} rows -> {st['rows_out']:,} kept ({st['date_min']} .. {st['date_max']})"
            f" type/static removed={st['removed_by_type'] + st.get('removed_near_static', 0):,}"
            f" low-confidence removed={st['removed_by_confidence']:,}")

    # 4) Cross-file de-duplication, then cell-days. Done one year at a time so memory stays bounded
    #    (a duplicate always shares its date, so per-year processing gives the same result).
    union = " UNION ALL ".join(f"SELECT * FROM {p}" for p in parts)
    con.execute(f"CREATE VIEW allpix AS {union}")
    years = [r[0] for r in con.execute("SELECT DISTINCT year(date) FROM allpix ORDER BY 1").fetchall()]
    con.execute("""CREATE TABLE pix (sensor VARCHAR, date DATE, lat DOUBLE, lon DOUBLE, satellite VARCHAR,
                   acq_time VARCHAR, frp DOUBLE)""")
    cd = cfg.cell_deg
    cell = f"CAST(floor(lat / {cd} + 1e-9) AS INTEGER) AS cell_i, CAST(floor(lon / {cd} + 1e-9) AS INTEGER) AS cell_j"
    chunks = []
    for y in years:
        con.execute(f"""INSERT INTO pix SELECT sensor, date, lat, lon, satellite, acq_time, any_value(frp)
                        FROM allpix WHERE year(date) = {y}
                        GROUP BY sensor, date, acq_time, lat, lon, satellite""")
        chunks.append(con.execute(f"""
            SELECT CAST(date AS TIMESTAMP) AS date, cell_i, cell_j, sensor,
                   CAST(count(*) AS INTEGER) AS n_det, sum(coalesce(frp, 0)) AS frp_sum
            FROM (SELECT date, sensor, frp, {cell} FROM pix WHERE year(date) = {y})
            GROUP BY date, cell_i, cell_j, sensor""").df())
    con.execute("DROP VIEW allpix")
    for p in parts:
        con.execute(f"DROP TABLE {p}")
    raw = (pd.concat(chunks, ignore_index=True)
           .sort_values(["date", "cell_i", "cell_j", "sensor"], kind="stable").reset_index(drop=True))
    del chunks
    raw["date"] = raw["date"].astype("datetime64[ns]")
    raw["sensor"] = raw["sensor"].astype(object)

    # 5) Terra-only correction by calendar month (period when Terra and Aqua both fly, before VIIRS).
    lo, hi = pd.Timestamp(cfg.terra_only_until).date(), pd.Timestamp(cfg.viirs_first_full_month).date()
    tf = con.execute(f"""
        WITH b AS (SELECT date, satellite, {cell} FROM pix
                   WHERE sensor = 'MODIS' AND date >= DATE '{lo}' AND date < DATE '{hi}')
        SELECT month(date) AS m, count(DISTINCT (date, cell_i, cell_j)) AS n_all,
               count(DISTINCT (date, cell_i, cell_j)) FILTER (WHERE satellite = 'Terra') AS n_terra
        FROM b GROUP BY 1 ORDER BY 1""").fetchall()
    con.close()
    terra_factors, terra_global = {}, 1.0
    if tf and sum(r[2] for r in tf):
        n_all, n_terra = {r[0]: r[1] for r in tf}, {r[0]: r[2] for r in tf}
        terra_global = float(sum(n_all.values()) / max(sum(n_terra.values()), 1))
        terra_factors = {m: float(n_all.get(m, 0) / n_terra[m]) if n_terra.get(m, 0) else terra_global
                         for m in range(1, 13)}
    return raw, stats, terra_factors, terra_global
