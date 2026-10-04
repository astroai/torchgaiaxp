"""Convert the archive's XpContinuousMeanSpectrum_*.csv.gz files to Parquet.

    python scripts/bulk_to_parquet.py \
        /arc/projects/k-pop/spectra/gaia/dr3/xp_continuous_mean_spectrum \
        /arc/projects/k-pop/spectra/gaia/dr3/xp_continuous_parquet --workers 16

One output per input, skipped when it already exists (resumable); a JSON line
per file on stdout. Parsing is the cost; run it once and train from Parquet.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from torchgaiaxp.io import read_xp_continuous, to_parquet


def convert_one(source: Path, target_dir: Path) -> dict:
    target = target_dir / (source.name.removesuffix(".csv.gz").removesuffix(".csv") + ".parquet")
    if target.exists():
        return {"file": source.name, "skipped": True}
    start = time.perf_counter()
    batch = read_xp_continuous(source)
    partial = target.with_suffix(".parquet.partial")
    to_parquet(batch, partial)
    partial.rename(target)
    return {
        "file": source.name,
        "rows": len(batch),
        "seconds": round(time.perf_counter() - start, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("target_dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pattern", default="XpContinuousMeanSpectrum_*.csv*")
    args = parser.parse_args()
    args.target_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.source_dir.glob(args.pattern))
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(convert_one, f, args.target_dir) for f in files]
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)


if __name__ == "__main__":
    main()
