"""Fast readers/writers for Gaia DR3 & DR4 XP continuous files (ECSV, Parquet, Archive, FITS, VOTable).

The archive's ``XpContinuousMeanSpectrum_*.csv.gz`` files are ECSV: ``#``
header lines, then CSV whose array columns are strings like ``"[1.0, 2.0]"``.
Here each array column is parsed in one vectorised call: the strings are joined
and handed to ``numpy`` once. Also supports:

* :func:`read_archive_parquet` -- ESA Gaia Archive / DataLink Parquet & VOParquet
  with native Arrow list/fixed-size-list columns;
* :func:`from_dataframe` / :func:`to_dataframe` -- GaiaXPy-compatible DataFrames;
* :func:`read_fits` / :func:`read_votable` -- optional Astropy readers;
* :func:`read_xp` -- format auto-dispatcher.
"""

from __future__ import annotations

import gzip
import io
import math
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
    coefficients: np.ndarray  # [B, 2, N], NaN for a missing band
    errors: np.ndarray | None = None  # [B, 2, N]
    correlations: np.ndarray | None = None  # [B, 2, N*(N-1)//2]
    standard_deviation: np.ndarray | None = None  # [B, 2]
    n_relevant: np.ndarray | None = None  # [B, 2] int
    solution_id: np.ndarray | None = None  # [B] int64
    basis_function_id: np.ndarray | None = None  # [B, 2] int64
    n_parameters: np.ndarray | None = None  # [B, 2] int
    release: str | None = None

    def __len__(self) -> int:
        return int(self.source_id.size)

    def tensors(self, device=None, dtype=torch.float64) -> dict[str, torch.Tensor]:
        def _t(arr, dt):
            return torch.as_tensor(np.array(arr, copy=True), dtype=dt, device=device)

        out = {
            "source_id": _t(self.source_id, torch.int64),
            "coefficients": _t(self.coefficients, dtype),
        }
        for name, dt in (
            ("errors", dtype),
            ("correlations", dtype),
            ("standard_deviation", dtype),
            ("n_relevant", torch.int64),
            ("solution_id", torch.int64),
            ("basis_function_id", torch.int64),
            ("n_parameters", torch.int64),
        ):
            val = getattr(self, name)
            if val is not None:
                out[name] = _t(val, dt)
        return out


def _header_lines(path: Path) -> int:
    opener = gzip.open if path.suffix == ".gz" else open
    count = 0
    with opener(path, "rt") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            count += 1
    return count


def _strip_brackets(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and ((s[0] == "[" and s[-1] == "]") or (s[0] == "(" and s[-1] == ")")):
        return s[1:-1].strip()
    return ""


def parse_array_column(values: pa.Array | pa.ChunkedArray | list, width: int | None = None) -> np.ndarray:
    """``"[a, b, ...]"`` strings (or nulls/empties/NaNs) -> ``[n, width]`` float64, NaN rows."""
    strings = values if isinstance(values, list) else values.to_pylist()
    cleaned: list[tuple[int, str]] = []
    for i, raw in enumerate(strings):
        if not isinstance(raw, str):
            continue
        inner = _strip_brackets(raw)
        if inner:
            cleaned.append((i, inner))
    if not cleaned:
        w = 0 if width is None else width
        return np.full((len(strings), w), np.nan)
    if width is None:
        width = len(cleaned[0][1].split(","))
    out = np.full((len(strings), width), np.nan)
    text = ",".join(inner for _, inner in cleaned)
    if "null" in text or "NaN" in text or "nan" in text:
        tokens = text.replace("null", "nan").split(",")
        flat = np.array(tokens, dtype=np.float64)
    else:
        flat = np.fromstring(text, sep=",", dtype=np.float64)
    if flat.size != len(cleaned) * width:
        raise ValueError(f"array column has {flat.size} values for {len(cleaned)} rows of {width}")
    idx = [i for i, _ in cleaned]
    out[idx] = flat.reshape(len(cleaned), width)
    return out


def _pad_and_stack_bands(bp: np.ndarray, rp: np.ndarray, *, corr: bool = False) -> np.ndarray:
    """Stack ``[B, N_bp]`` and ``[B, N_rp]`` into ``[B, 2, N_max]`` (handles unequal widths)."""
    if bp.shape[1] == rp.shape[1]:
        return np.stack([bp, rp], axis=1)
    if not corr:
        w = max(bp.shape[1], rp.shape[1])
        out = np.full((bp.shape[0], 2, w), np.nan, dtype=np.result_type(bp, rp))
        for k, arr in enumerate((bp, rp)):
            if arr.shape[1] == 0:
                continue
            present = np.isfinite(arr).all(-1)
            out[present, k, :] = 0.0
            out[present, k, : arr.shape[1]] = arr[present]
        return out
    n_bp = int((math.isqrt(1 + 8 * bp.shape[1]) + 1) // 2) if bp.shape[1] else 0
    n_rp = int((math.isqrt(1 + 8 * rp.shape[1]) + 1) // 2) if rp.shape[1] else 0
    n_max = max(n_bp, n_rp)
    w_max = n_max * (n_max - 1) // 2
    out = np.full((bp.shape[0], 2, w_max), np.nan, dtype=np.result_type(bp, rp))
    r_max, c_max = np.tril_indices(n_max, -1)
    for k, (arr, nk) in enumerate(((bp, n_bp), (rp, n_rp))):
        if nk == 0:
            continue
        present = np.isfinite(arr).all(-1)
        out[present, k, :] = 0.0
        if nk == n_max:
            out[present, k, :] = arr[present]
        else:
            sub_idx = np.flatnonzero((r_max < nk) & (c_max < nk))
            out[np.ix_(present, [k], sub_idx)] = arr[present][:, None, :]
    return out


def _extract_optional_meta(names, getter, n: int):
    solution_id = None
    if "solution_id" in names:
        solution_id = np.asarray(getter("solution_id"), dtype=np.int64)
    elif "solutionId" in names:
        solution_id = np.asarray(getter("solutionId"), dtype=np.int64)

    bf_id = None
    if "bp_basis_function_id" in names or "rp_basis_function_id" in names:
        bp_bf = getter("bp_basis_function_id") if "bp_basis_function_id" in names else np.full(n, -1)
        rp_bf = getter("rp_basis_function_id") if "rp_basis_function_id" in names else np.full(n, -1)
        bf_id = np.stack(
            [np.nan_to_num(bp_bf, nan=-1).astype(np.int64), np.nan_to_num(rp_bf, nan=-1).astype(np.int64)],
            axis=1,
        )

    n_params = None
    if "bp_n_parameters" in names or "rp_n_parameters" in names:
        bp_np = getter("bp_n_parameters") if "bp_n_parameters" in names else np.full(n, 0)
        rp_np = getter("rp_n_parameters") if "rp_n_parameters" in names else np.full(n, 0)
        n_params = np.stack(
            [np.nan_to_num(bp_np, nan=0).astype(np.int64), np.nan_to_num(rp_np, nan=0).astype(np.int64)],
            axis=1,
        )
    return solution_id, bf_id, n_params


def _assemble_batch(
    names: set[str],
    n: int,
    source_id: np.ndarray,
    arr_fn,
    scalar_fn,
    *,
    infer_widths: bool = False,
    require_std_relevant: bool = False,
    corr_dtype=None,
) -> XPBatch:
    """Shared column assembly for CSV/ECSV, Archive Parquet, DataFrame, and Astropy Tables."""
    solution_id, bf_id, n_params = _extract_optional_meta(names, lambda k: scalar_fn(k, np.nan), n)
    widths: list[int | None] = [None, None]
    if infer_widths and n_params is not None:
        for k in range(2):
            pos = n_params[:, k][n_params[:, k] > 0]
            if pos.size:
                widths[k] = int(pos[0])

    c_bp = arr_fn("bp_coefficients", widths[0])
    c_rp = arr_fn("rp_coefficients", widths[1])
    coefficients = _pad_and_stack_bands(c_bp, c_rp)

    errors = None
    if "bp_coefficient_errors" in names and "rp_coefficient_errors" in names:
        e_bp = arr_fn("bp_coefficient_errors", c_bp.shape[1] or None)
        e_rp = arr_fn("rp_coefficient_errors", c_rp.shape[1] or None)
        errors = _pad_and_stack_bands(e_bp, e_rp)

    correlations = None
    if "bp_coefficient_correlations" in names and "rp_coefficient_correlations" in names:
        w_r_bp = (
            c_bp.shape[1] * (c_bp.shape[1] - 1) // 2 if (infer_widths and c_bp.shape[1]) else None
        )
        w_r_rp = (
            c_rp.shape[1] * (c_rp.shape[1] - 1) // 2 if (infer_widths and c_rp.shape[1]) else None
        )
        r_bp = arr_fn("bp_coefficient_correlations", w_r_bp)
        r_rp = arr_fn("rp_coefficient_correlations", w_r_rp)
        correlations = _pad_and_stack_bands(r_bp, r_rp, corr=True)
        if corr_dtype is not None:
            correlations = correlations.astype(corr_dtype)

    has_std = require_std_relevant or any(f"{b}_standard_deviation" in names for b in BANDS)
    std = (
        np.stack([scalar_fn(f"{b}_standard_deviation", np.nan) for b in BANDS], axis=1)
        if has_std
        else None
    )
    has_rel = require_std_relevant or any(f"{b}_n_relevant_bases" in names for b in BANDS)
    relevant = (
        np.nan_to_num(
            np.stack([scalar_fn(f"{b}_n_relevant_bases", 0) for b in BANDS], axis=1)
        ).astype(np.int16)
        if has_rel
        else None
    )
    if require_std_relevant and n_params is None:
        n_params = np.broadcast_to(
            np.array([[c_bp.shape[1], c_rp.shape[1]]], dtype=np.int64), (n, 2)
        ).copy()

    return XPBatch(
        source_id=source_id,
        coefficients=coefficients,
        errors=errors,
        correlations=correlations,
        standard_deviation=std,
        n_relevant=relevant,
        solution_id=solution_id,
        basis_function_id=bf_id,
        n_parameters=n_params,
    )


def read_xp_continuous(path: str | Path, *, corr_dtype=np.float32) -> XPBatch:
    """One bulk ``XpContinuousMeanSpectrum_*.csv(.gz)`` or ECSV file -> ``XPBatch``."""
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
    names = set(table.column_names)

    def _col(name: str, fill: float) -> np.ndarray:
        if name not in names:
            return np.full(n, fill)
        return np.asarray(table[name].to_numpy(zero_copy_only=False), dtype=np.float64)

    return _assemble_batch(
        names,
        n,
        np.asarray(table["source_id"].to_numpy(), dtype=np.int64),
        lambda col, w=None: parse_array_column(table[col], w),
        _col,
        infer_widths=True,
        require_std_relevant=True,
        corr_dtype=corr_dtype,
    )


def to_parquet(batch: XPBatch, path: str | Path, *, float_dtype=np.float32) -> Path:
    """Write fixed-size-list columns: coefficients/errors [2*N], correlations [2*N*(N-1)/2]."""

    def fixed(array: np.ndarray, dtype) -> pa.FixedSizeListArray:
        flat = np.ascontiguousarray(array.reshape(len(batch), -1), dtype=dtype)
        return pa.FixedSizeListArray.from_arrays(pa.array(flat.reshape(-1)), flat.shape[1])

    cols: dict[str, pa.Array] = {
        "source_id": pa.array(batch.source_id, pa.int64()),
        "coefficients": fixed(batch.coefficients, np.float64),
    }
    if batch.errors is not None:
        cols["errors"] = fixed(batch.errors, float_dtype)
    if batch.correlations is not None:
        cols["correlations"] = fixed(batch.correlations, float_dtype)
    if batch.standard_deviation is not None:
        cols["standard_deviation"] = fixed(batch.standard_deviation, np.float64)
    if batch.n_relevant is not None:
        cols["n_relevant"] = fixed(batch.n_relevant, np.int16)
    if batch.solution_id is not None:
        cols["solution_id"] = pa.array(batch.solution_id, pa.int64())
    if batch.basis_function_id is not None:
        cols["basis_function_id"] = fixed(batch.basis_function_id, np.int64)
    if batch.n_parameters is not None:
        cols["n_parameters"] = fixed(batch.n_parameters, np.int16)

    table = pa.table(cols)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    return path


def _arrow_list_to_numpy(col: pa.ChunkedArray | pa.Array, width: int | None = None) -> np.ndarray:
    """Extract Arrow List / FixedSizeList / string column to dense ``[B, width]`` float64."""
    arr = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    if pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type):
        return parse_array_column(arr, width)
    if pa.types.is_fixed_size_list(arr.type):
        w = arr.type.list_size
        flat = arr.flatten().to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        out = flat.reshape(len(arr), w).copy()
        if arr.null_count:
            null_mask = arr.is_null().to_numpy(zero_copy_only=False)
            out[null_mask] = np.nan
        return out
    if pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type):
        offsets = arr.offsets.to_numpy(zero_copy_only=False)
        lengths = np.diff(offsets)
        pos = lengths[lengths > 0]
        w = int(pos[0]) if (width is None and pos.size) else (width or 0)
        out = np.full((len(arr), w), np.nan, dtype=np.float64)
        if not pos.size:
            return out
        valid = (lengths == w) & (~arr.is_null().to_numpy(zero_copy_only=False))
        values = arr.values.to_numpy(zero_copy_only=False)
        if valid.all():
            out[:] = values[offsets[0] : offsets[-1]].reshape(len(arr), w)
        else:
            # Fast slice copy for variable-validity lists (missing band rows have length 0 or null).
            for i in np.flatnonzero(valid):
                out[i] = values[offsets[i] : offsets[i + 1]]
        return out
    raise TypeError(f"unsupported Arrow array column type: {arr.type}")


def read_parquet(path: str | Path) -> XPBatch:
    """Read a torchgaiaxp Parquet file (or fall back to Archive Parquet schema)."""
    table = pq.read_table(path)
    names = set(table.column_names)
    if "bp_coefficients" in names and "coefficients" not in names:
        return read_archive_parquet(path)

    n = table.num_rows

    def matrix(name, shape_fn, dtype=None):
        col = table[name].combine_chunks()
        flat = col.flatten()
        np_flat = flat.to_numpy(zero_copy_only=False)
        if dtype is not None:
            np_flat = np.asarray(np_flat, dtype=dtype)
        width = col.type.list_size
        return np_flat.reshape(n, *shape_fn(width))

    coeff = matrix("coefficients", lambda w: (2, w // 2), np.float64)
    n_bases = coeff.shape[-1]
    n_corr = n_bases * (n_bases - 1) // 2
    return XPBatch(
        source_id=table["source_id"].to_numpy(),
        coefficients=coeff,
        errors=matrix("errors", lambda w: (2, w // 2), np.float64) if "errors" in names else None,
        correlations=(
            matrix("correlations", lambda w: (2, n_corr)) if "correlations" in names else None
        ),
        standard_deviation=(
            matrix("standard_deviation", lambda w: (2,), np.float64)
            if "standard_deviation" in names
            else None
        ),
        n_relevant=(
            matrix("n_relevant", lambda w: (2,), np.int16) if "n_relevant" in names else None
        ),
        solution_id=table["solution_id"].to_numpy().astype(np.int64) if "solution_id" in names else None,
        basis_function_id=(
            matrix("basis_function_id", lambda w: (2,), np.int64)
            if "basis_function_id" in names
            else None
        ),
        n_parameters=(
            matrix("n_parameters", lambda w: (2,), np.int64) if "n_parameters" in names else None
        ),
    )


def read_archive_parquet(path: str | Path, *, corr_dtype=None) -> XPBatch:
    """Read official ESA Gaia Archive Parquet / VOParquet XP files."""
    table = pq.read_table(path)
    n = table.num_rows
    names = set(table.column_names)

    def _col(name: str, fill: float) -> np.ndarray:
        if name not in names:
            return np.full(n, fill)
        return np.asarray(table[name].to_numpy(zero_copy_only=False), dtype=np.float64)

    return _assemble_batch(
        names,
        n,
        np.asarray(table["source_id"].to_numpy(), dtype=np.int64),
        lambda col, w=None: _arrow_list_to_numpy(table[col], w),
        _col,
        corr_dtype=corr_dtype,
    )


def _df_array_col(series, width: int | None = None) -> np.ndarray:
    vals = series.tolist() if hasattr(series, "tolist") else list(series)
    arrays: list[tuple[int, np.ndarray]] = []
    for i, v in enumerate(vals):
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        if isinstance(v, str):
            inner = _strip_brackets(v)
            if not inner:
                continue
            arr = np.fromstring(inner.replace("null", "nan"), sep=",", dtype=np.float64)
        else:
            arr = np.asarray(v, dtype=np.float64)
        if arr.size == 0:
            continue
        arrays.append((i, arr))
    if not arrays:
        return np.full((len(vals), width or 0), np.nan, dtype=np.float64)
    if width is None:
        width = arrays[0][1].size
    out = np.full((len(vals), width), np.nan, dtype=np.float64)
    for i, arr in arrays:
        if arr.size == width:
            out[i] = arr
    return out


def from_dataframe(df) -> XPBatch:
    """Convert a GaiaXPy-style ``pandas.DataFrame`` -> ``XPBatch``."""
    n = len(df)
    cols = set(df.columns)

    def _num(name: str, fill: float) -> np.ndarray:
        if name not in cols:
            return np.full(n, fill)
        return df[name].to_numpy(dtype=np.float64, na_value=np.nan)

    return _assemble_batch(
        cols,
        n,
        df["source_id"].to_numpy(dtype=np.int64),
        lambda col, w=None: _df_array_col(df[col], w),
        _num,
    )


def to_dataframe(batch_or_xp):
    """Convert an :class:`XPBatch` or :class:`XPSpectra` -> GaiaXPy-compatible ``pandas.DataFrame``."""
    import pandas as pd

    if hasattr(batch_or_xp, "coefficients") and isinstance(batch_or_xp.coefficients, torch.Tensor):
        xp = batch_or_xp
        n_b = xp.n_bases
        batch = XPBatch(
            source_id=(
                xp.source_id.detach().cpu().numpy()
                if xp.source_id is not None
                else np.arange(len(xp), dtype=np.int64)
            ),
            coefficients=xp.coefficients.detach().cpu().numpy(),
            errors=None if xp.errors is None else xp.errors.detach().cpu().numpy(),
            correlations=None if xp.correlations is None else xp.correlations.detach().cpu().numpy(),
            standard_deviation=(
                None
                if xp.standard_deviation is None
                else xp.standard_deviation.detach().cpu().numpy()
            ),
            n_relevant=None if xp.n_relevant is None else xp.n_relevant.detach().cpu().numpy(),
            solution_id=None if xp.solution_id is None else xp.solution_id.detach().cpu().numpy(),
            basis_function_id=(
                None
                if xp.basis_function_id is None
                else xp.basis_function_id.detach().cpu().numpy()
            ),
            n_parameters=(
                None if xp.n_parameters is None else xp.n_parameters.detach().cpu().numpy()
            ),
        )
    else:
        batch = batch_or_xp
        n_b = batch.coefficients.shape[-1]

    rows = []
    for i in range(len(batch)):
        row: dict = {"source_id": int(batch.source_id[i])}
        if batch.solution_id is not None:
            row["solution_id"] = int(batch.solution_id[i])
        for k, b in enumerate(BANDS):
            present = bool(np.isfinite(batch.coefficients[i, k]).all())
            nk = (
                int(batch.n_parameters[i, k])
                if batch.n_parameters is not None
                else (n_b if present else 0)
            )
            w_slice = nk if (present and 0 < nk < n_b) else n_b
            row[f"{b}_coefficients"] = batch.coefficients[i, k, :w_slice] if present else np.nan
            if batch.errors is not None:
                row[f"{b}_coefficient_errors"] = (
                    batch.errors[i, k, :w_slice] if present else np.nan
                )
            if batch.correlations is not None:
                if present:
                    if 0 < nk < n_b:
                        r_max, c_max = np.tril_indices(n_b, -1)
                        sub_idx = np.flatnonzero((r_max < nk) & (c_max < nk))
                        corr_val = batch.correlations[i, k, sub_idx].astype(np.float64)
                    else:
                        corr_val = batch.correlations[i, k].astype(np.float64)
                else:
                    corr_val = np.nan
                row[f"{b}_coefficient_correlations"] = corr_val
            row[f"{b}_n_parameters"] = (
                int(batch.n_parameters[i, k]) if batch.n_parameters is not None else (n_b if present else np.nan)
            )
            if batch.standard_deviation is not None:
                row[f"{b}_standard_deviation"] = (
                    float(batch.standard_deviation[i, k]) if present else np.nan
                )
            if batch.n_relevant is not None:
                row[f"{b}_n_relevant_bases"] = int(batch.n_relevant[i, k]) if present else np.nan
            if batch.basis_function_id is not None:
                row[f"{b}_basis_function_id"] = (
                    int(batch.basis_function_id[i, k]) if present else np.nan
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _astropy_table_to_batch(table) -> XPBatch:
    names = set(table.colnames)
    n = len(table)

    def _masked_col(name: str, fill: float) -> np.ndarray:
        if name not in names:
            return np.full(n, fill, dtype=np.float64)
        col = table[name]
        if hasattr(col, "filled"):
            return np.asarray(col.filled(fill), dtype=np.float64)
        return np.asarray(col, dtype=np.float64)

    def _masked_arr(name: str) -> np.ndarray:
        col = table[name]
        if hasattr(col, "filled"):
            arr = np.asarray(col.filled(np.nan), dtype=np.float64)
            if arr.ndim == 2:
                return arr
        return _df_array_col(table[name].tolist() if hasattr(table[name], "tolist") else list(col))

    return _assemble_batch(
        names,
        n,
        np.asarray(table["source_id"], dtype=np.int64),
        lambda col, _w=None: _masked_arr(col),
        _masked_col,
    )


def read_fits(path: str | Path) -> XPBatch:
    """Read an Archive FITS XP table (requires optional ``astropy``)."""
    try:
        from astropy.table import Table
    except ImportError as exc:
        raise ImportError("astropy is required to read FITS files: pip install astropy") from exc
    return _astropy_table_to_batch(Table.read(path, format="fits"))


def read_votable(path: str | Path) -> XPBatch:
    """Read an Archive VOTable/XML XP table (requires optional ``astropy``)."""
    try:
        from astropy.io.votable import parse_single_table
    except ImportError as exc:
        raise ImportError("astropy is required to read VOTable/XML files: pip install astropy") from exc
    return _astropy_table_to_batch(parse_single_table(str(path)).to_table())


def read_xp(path: str | Path) -> XPBatch:
    """Auto-dispatch Parquet (torchgaiaxp or Archive), CSV/ECSV(.gz), FITS, or VOTable."""
    p = Path(path)
    name = p.name.lower()
    if name.endswith(".parquet"):
        return read_parquet(p)
    if name.endswith((".fits", ".fits.gz", ".fit", ".fit.gz")):
        return read_fits(p)
    if name.endswith((".xml", ".xml.gz", ".vot", ".vot.gz")):
        return read_votable(p)
    return read_xp_continuous(p)


def write_ecsv_like(batch: XPBatch, path: str | Path) -> None:
    """Write a file in the archive's bulk layout (tests and fixtures)."""
    buffer = io.StringIO()
    buffer.write("# %ECSV 1.0\n# ---\n# fixture written by torchgaiaxp\n")
    cols = ["source_id"]
    if batch.solution_id is not None:
        cols.append("solution_id")
    for b in BANDS:
        if batch.basis_function_id is not None:
            cols.append(f"{b}_basis_function_id")
        if batch.n_parameters is not None:
            cols.append(f"{b}_n_parameters")
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
        if batch.solution_id is not None:
            row.append(str(int(batch.solution_id[i])))
        for k in range(2):
            present = bool(np.all(np.isfinite(batch.coefficients[i, k])))
            if batch.basis_function_id is not None:
                row.append(str(int(batch.basis_function_id[i, k])) if present else "")
            if batch.n_parameters is not None:
                row.append(str(int(batch.n_parameters[i, k])) if present else "")
            row += [
                repr(float(batch.standard_deviation[i, k])) if (present and batch.standard_deviation is not None) else "",
                str(int(batch.n_relevant[i, k])) if (present and batch.n_relevant is not None) else "",
                arr(batch.coefficients[i, k]),
                arr(batch.errors[i, k]) if batch.errors is not None else "",
                arr(batch.correlations[i, k]) if batch.correlations is not None else "",
            ]
        buffer.write(",".join(row) + "\n")
    data = buffer.getvalue().encode()
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "wb") as handle:
            handle.write(data)
    else:
        path.write_bytes(data)
