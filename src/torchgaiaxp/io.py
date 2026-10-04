"""Fast readers for the Gaia DR3 bulk XP continuous files.

The archive's ``XpContinuousMeanSpectrum_*.csv.gz`` files are ECSV: ``#``
header lines, then CSV whose array columns are strings like ``"[1.0, 2.0]"``.
GaiaXPy (and most scripts) parse each array with ``ast.literal_eval`` per row.
Here each array column is parsed in one vectorised call: the strings are joined
and handed to ``numpy`` once, which is what makes 3,386 files (2.2e8 sources) a
CPU-hours job rather than days. ``to_parquet`` writes fixed-size-list columns
so training reads them zero-copy.
"""

from __future__ import annotations

import gzip
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import torch

BANDS = ("bp", "rp")
N_BASES = 55
N_CORR = N_BASES * (N_BASES - 1) // 2


@dataclass
class XPBatch:
    source_id: np.ndarray  # [B] int64
    coefficients: np.ndarray  # [B, 2, 55], NaN for a missing band
    errors: np.ndarray  # [B, 2, 55]
    correlations: np.ndarray  # [B, 2, 1485]
    standard_deviation: np.ndarray  # [B, 2]
    n_relevant: np.ndarray  # [B, 2] int

    def __len__(self) -> int:
        return int(self.source_id.size)

    def tensors(self, device=None, dtype=torch.float64) -> dict[str, torch.Tensor]:
        return {
            "coefficients": torch.as_tensor(self.coefficients, dtype=dtype, device=device),
            "errors": torch.as_tensor(self.errors, dtype=dtype, device=device),
            "correlations": torch.as_tensor(self.correlations, dtype=dtype, device=device),
            "n_relevant": torch.as_tensor(self.n_relevant, device=device),
        }


def _header_lines(path: Path) -> int:
    opener = gzip.open if path.suffix == ".gz" else open
    count = 0
    with opener(path, "rt") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            count += 1
    return count


def parse_array_column(values: pa.Array | pa.ChunkedArray, width: int) -> np.ndarray:
    """``"[a, b, ...]"`` strings (or nulls/empties) -> ``[n, width]`` float64, NaN rows."""
    strings = values.to_pylist() if not isinstance(values, list) else values
    out = np.full((len(strings), width), np.nan)
    good = [i for i, s in enumerate(strings) if s and len(s) > 2]
    if not good:
        return out
    text = ",".join(strings[i].strip()[1:-1] for i in good)
    flat = (
        np.array(text.replace("null", "nan").split(","), dtype=np.float64)
        if ("null" in text)
        else np.fromstring(text, sep=",", dtype=np.float64)
    )
    if flat.size != len(good) * width:
        raise ValueError(f"array column has {flat.size} values for {len(good)} rows of {width}")
    out[good] = flat.reshape(len(good), width)
    return out


def read_xp_continuous(path: str | Path) -> XPBatch:
    """One bulk ``XpContinuousMeanSpectrum_*.csv(.gz)`` file -> ``XPBatch``."""
    path = Path(path)
    skip = _header_lines(path)
    array_columns = [
        f"{b}_{k}"
        for b in BANDS
        for k in ("coefficients", "coefficient_errors", "coefficient_correlations")
    ]
    table = pacsv.read_csv(
        path,
        read_options=pacsv.ReadOptions(skip_rows=skip, block_size=1 << 26),
        convert_options=pacsv.ConvertOptions(
            column_types={name: pa.string() for name in array_columns},
            strings_can_be_null=True,
        ),
    )
    n = table.num_rows
    coefficients = np.stack(
        [parse_array_column(table[f"{b}_coefficients"], N_BASES) for b in BANDS], axis=1
    )
    errors = np.stack(
        [parse_array_column(table[f"{b}_coefficient_errors"], N_BASES) for b in BANDS],
        axis=1,
    )
    correlations = np.stack(
        [parse_array_column(table[f"{b}_coefficient_correlations"], N_CORR) for b in BANDS],
        axis=1,
    )

    def column(name: str, fill: float) -> np.ndarray:
        if name not in table.column_names:
            return np.full(n, fill)
        return np.asarray(table[name].to_numpy(zero_copy_only=False), dtype=np.float64)

    std = np.stack([column(f"{b}_standard_deviation", np.nan) for b in BANDS], axis=1)
    relevant = np.stack([column(f"{b}_n_relevant_bases", 0) for b in BANDS], axis=1)
    return XPBatch(
        source_id=np.asarray(table["source_id"].to_numpy(), dtype=np.int64),
        coefficients=coefficients,
        errors=errors,
        correlations=correlations.astype(np.float32),
        standard_deviation=std,
        n_relevant=np.nan_to_num(relevant).astype(np.int16),
    )


def to_parquet(batch: XPBatch, path: str | Path, *, float_dtype=np.float32) -> Path:
    """Write fixed-size-list columns: coefficients/errors [2*55], correlations [2*1485]."""

    def fixed(array: np.ndarray, dtype) -> pa.FixedSizeListArray:
        flat = np.ascontiguousarray(array.reshape(len(batch), -1), dtype=dtype)
        return pa.FixedSizeListArray.from_arrays(pa.array(flat.reshape(-1)), flat.shape[1])

    table = pa.table(
        {
            "source_id": pa.array(batch.source_id, pa.int64()),
            "coefficients": fixed(batch.coefficients, np.float64),
            "errors": fixed(batch.errors, float_dtype),
            "correlations": fixed(batch.correlations, float_dtype),
            "standard_deviation": fixed(batch.standard_deviation, np.float64),
            "n_relevant": fixed(batch.n_relevant, np.int16),
        }
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    return path


def read_parquet(path: str | Path) -> XPBatch:
    table = pq.read_table(path)

    def matrix(name, shape, dtype):
        values = table[name].combine_chunks().flatten().to_numpy(zero_copy_only=False)
        return np.asarray(values, dtype=dtype).reshape(table.num_rows, *shape)

    return XPBatch(
        source_id=table["source_id"].to_numpy(),
        coefficients=matrix("coefficients", (2, N_BASES), np.float64),
        errors=matrix("errors", (2, N_BASES), np.float64),
        correlations=matrix("correlations", (2, N_CORR), np.float32),
        standard_deviation=matrix("standard_deviation", (2,), np.float64),
        n_relevant=matrix("n_relevant", (2,), np.int16),
    )


def write_ecsv_like(batch: XPBatch, path: str | Path) -> None:
    """Write a file in the archive's bulk layout (tests and fixtures)."""
    buffer = io.StringIO()
    buffer.write("# %ECSV 1.0\n# ---\n# fixture written by torchgaiaxp\n")
    cols = ["source_id"]
    for b in BANDS:
        cols += [
            f"{b}_standard_deviation",
            f"{b}_n_relevant_bases",
            f"{b}_coefficients",
            f"{b}_coefficient_errors",
            f"{b}_coefficient_correlations",
        ]
    buffer.write(",".join(cols) + "\n")

    def arr(values) -> str:
        if not np.all(np.isfinite(values)):
            return ""
        return '"[' + ", ".join(repr(float(v)) for v in values) + ']"'

    for i in range(len(batch)):
        row = [str(int(batch.source_id[i]))]
        for k in range(2):
            present = np.all(np.isfinite(batch.coefficients[i, k]))
            row += [
                repr(float(batch.standard_deviation[i, k])) if present else "",
                str(int(batch.n_relevant[i, k])) if present else "",
                arr(batch.coefficients[i, k]),
                arr(batch.errors[i, k]),
                arr(batch.correlations[i, k]),
            ]
        buffer.write(",".join(row) + "\n")
    data = buffer.getvalue().encode()
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "wb") as handle:
            handle.write(data)
    else:
        path.write_bytes(data)
