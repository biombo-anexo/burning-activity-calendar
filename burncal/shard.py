"""Split harmonized.parquet into parts below a size limit (GitHub's web upload takes <= 25 MB per file).

    python -m burncal.shard data/processed --max-mb 20 --remove-original

The app reads data/processed/harmonized_parts/*.parquet when that folder exists, else harmonized.parquet.
Rows stay sorted by cell, so each part covers a band of the map and DuckDB can skip parts outside an area.
"""
import argparse
import math
from pathlib import Path

import pyarrow.parquet as pq


def shard(data_dir, max_mb: float = 20.0, remove_original: bool = False) -> list:
    d = Path(data_dir)
    src = d / "harmonized.parquet"
    t = pq.read_table(src)
    n_parts = max(1, math.ceil(src.stat().st_size / (max_mb * 1e6) * 1.15))
    out = d / "harmonized_parts"
    out.mkdir(exist_ok=True)
    for old in out.glob("*.parquet"):
        old.unlink()
    step = math.ceil(t.num_rows / n_parts)
    files = []
    for k in range(n_parts):
        f = out / f"part_{k:02d}.parquet"
        pq.write_table(t.slice(k * step, step), f, compression="zstd", row_group_size=100_000)
        files.append(f)
    assert sum(pq.ParquetFile(f).metadata.num_rows for f in files) == t.num_rows
    if remove_original:
        src.unlink()
    return files


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir")
    ap.add_argument("--max-mb", type=float, default=20.0)
    ap.add_argument("--remove-original", action="store_true")
    a = ap.parse_args()
    for f in shard(a.data_dir, a.max_mb, a.remove_original):
        print(f, f"{f.stat().st_size / 1e6:.1f} MB")
