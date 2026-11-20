"""Multi-format output writers (CSV, ECSV, FITS via torchfits, Parquet, VOTable, Avro)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .io import write_fits, write_votable

_SUPPORTED_FORMATS = {
    "csv": "csv",
    "ecsv": "ecsv",
    "fits": "fits",
    "fit": "fits",
    "parquet": "parquet",
    "pq": "parquet",
    "votable": "votable",
    "vot": "votable",
    "xml": "votable",
    "avro": "avro",
}

_EXTENSIONS = {
    "csv": ".csv",
    "ecsv": ".ecsv",
    "fits": ".fits",
    "parquet": ".parquet",
    "votable": ".xml",
    "avro": ".avro",
}


def standardise_output_format(fmt: str | None, *, default: str = "ecsv") -> str:
    """Normalize a GaiaXPy output format string or extension."""
    key = default if fmt is None else str(fmt).strip().lstrip(".").lower()
    if key not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported output format {fmt!r}; supported: {sorted(set(_SUPPORTED_FORMATS.values()))}"
        )
    return _SUPPORTED_FORMATS[key]


def _check_overwrite(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists and overwrite=False: {path}")


def _is_array_cell(val) -> bool:
    return isinstance(val, (np.ndarray, list, tuple))


def _format_cell_csv(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    if _is_array_cell(val):
        arr = np.asarray(val)
        return '"[' + ", ".join(repr(float(x)) if np.isfinite(x) else "nan" for x in arr.reshape(-1)) + ']"'
    if isinstance(val, (bool, np.bool_)):
        return "True" if val else "False"
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, (float, np.floating)):
        return repr(float(val)) if np.isfinite(val) else ""
    s = str(val)
    if "," in s or '"' in s or "\n" in s:
        return '"' + s.replace('"', '""') + '"'
    return s


def _ecsv_datatype(series) -> tuple[str, str | None]:
    vals = series.tolist()
    sample = next((v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))), None)
    if _is_array_cell(sample):
        return "string", "json"
    dt = np.asarray(series).dtype
    if dt.kind in ("i", "u"):
        return "int64", None
    if dt.kind == "b":
        return "bool", None
    if dt.kind == "f":
        return "float64", None
    return "string", None


def _write_csv_or_ecsv(
    df,
    path: Path,
    *,
    is_ecsv: bool,
    metadata: dict | None = None,
) -> Path:
    lines: list[str] = []
    if is_ecsv:
        lines.append("# %ECSV 1.0")
        lines.append("# ---")
        lines.append("# datatype:")
        for col in df.columns:
            dtype_name, subtype = _ecsv_datatype(df[col])
            lines.append(f"# - {{name: {col}, datatype: {dtype_name}}}" if subtype is None else f"# - {{name: {col}, datatype: {dtype_name}, subtype: {subtype}}}")
        meta = dict(getattr(df, "attrs", {}))
        if metadata:
            meta.update(metadata)
        if meta:
            lines.append("# meta:")
            for mk, mv in meta.items():
                lines.append(f"#   {mk}: {json.dumps(mv, default=lambda o: np.asarray(o).tolist())}")
        lines.append("# delimiter: ','")
    col_lists = [df[c].tolist() for c in df.columns]
    lines.append(",".join(str(c) for c in df.columns))
    for i in range(len(df)):
        lines.append(",".join(_format_cell_csv(col[i]) for col in col_lists))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_parquet_output(df, path: Path, *, metadata: dict | None = None) -> Path:
    cols: dict[str, pa.Array] = {}
    for c in df.columns:
        vals = df[c].tolist()
        sample = next((v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))), None)
        if _is_array_cell(sample):
            rows = [
                None if (v is None or (isinstance(v, float) and np.isnan(v))) else np.asarray(v, np.float64).tolist()
                for v in vals
            ]
            cols[c] = pa.array(rows, type=pa.list_(pa.float64()))
        else:
            cols[c] = pa.array(df[c])
    meta_dict = dict(getattr(df, "attrs", {}))
    if metadata:
        meta_dict.update(metadata)
    table = pa.table(cols)
    if meta_dict:
        table = table.replace_schema_metadata(
            {"gaiaxpy_metadata": json.dumps(meta_dict, default=lambda o: np.asarray(o).tolist()).encode("utf-8")}
        )
    pq.write_table(table, path, compression="zstd")
    return path


def _write_avro_output(df, path: Path, *, metadata: dict | None = None) -> Path:
    try:
        import fastavro
    except ImportError as exc:
        raise ImportError("fastavro is required for output_format='avro': pip install fastavro") from exc
    fields = []
    col_lists: list[tuple[str, list]] = []
    for c in df.columns:
        vals = df[c].tolist()
        col_lists.append((str(c), vals))
        sample = next((v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))), None)
        if c == "source_id" or isinstance(sample, (int, np.integer)):
            ftype: object = ["null", "long"]
        elif isinstance(sample, (bool, np.bool_)) or "_within_range_" in c:
            ftype = ["null", "boolean"]
        elif _is_array_cell(sample):
            ftype = ["null", {"type": "array", "items": "double"}]
        elif isinstance(sample, (float, np.floating)):
            ftype = ["null", "double"]
        else:
            ftype = ["null", "string"]
        fields.append({"name": str(c), "type": ftype})
    schema = {"type": "record", "name": "GaiaXPOutput", "fields": fields}
    records = []
    for i in range(len(df)):
        rec = {}
        for c, vals in col_lists:
            v = vals[i]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                rec[c] = None
            elif _is_array_cell(v):
                rec[c] = [float(x) for x in np.asarray(v).reshape(-1)]
            elif isinstance(v, (np.integer,)) or (c == "source_id" and isinstance(v, (int, float))):
                rec[c] = int(v)
            elif isinstance(v, (np.floating,)):
                rec[c] = float(v)
            elif isinstance(v, (np.bool_,)):
                rec[c] = bool(v)
            else:
                rec[c] = v
        records.append(rec)
    meta = {k: json.dumps(v, default=lambda o: np.asarray(o).tolist()) for k, v in (metadata or {}).items()}
    with open(path, "wb") as handle:
        fastavro.writer(handle, schema, records, metadata=meta or None)
    return path


def write_output(
    df,
    output_path: str | Path = ".",
    output_file: str = "output_spectra",
    output_format: str | None = None,
    *,
    sampling=None,
    metadata: dict | None = None,
    overwrite: bool = True,
) -> Path:
    """Write calibrated/converted spectra or photometry DataFrame in GaiaXPy formats."""
    fmt = standardise_output_format(output_format, default="ecsv")
    ext = _EXTENSIONS[fmt]
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(output_file).name
    target = out_dir / (stem if stem.lower().endswith(ext) else f"{stem}{ext}")
    _check_overwrite(target, overwrite)

    meta = dict(getattr(df, "attrs", {}))
    if metadata:
        meta.update(metadata)
    if sampling is not None:
        meta["sampling"] = np.asarray(sampling).tolist()

    if fmt in ("csv", "ecsv"):
        return _write_csv_or_ecsv(df, target, is_ecsv=(fmt == "ecsv"), metadata=meta)
    if fmt == "parquet":
        return _write_parquet_output(df, target, metadata=meta)
    if fmt == "fits":
        hdr = {}
        if "data_type" in meta:
            hdr["DATATYPE"] = str(meta["data_type"])
        if "data_release" in meta:
            hdr["RELEASE"] = str(meta["data_release"])
        return write_fits(df, target, header=hdr or None, overwrite=overwrite)
    if fmt == "votable":
        params = {k: json.dumps(v, default=lambda o: np.asarray(o).tolist()) for k, v in meta.items()}
        return write_votable(df, target, params=params, overwrite=overwrite)
    return _write_avro_output(df, target, metadata=meta)
