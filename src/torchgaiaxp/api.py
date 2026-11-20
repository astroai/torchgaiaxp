"""GaiaXPy-compatible top-level entry points and input requirement descriptors."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import _assets
from .archive import DEFAULT_ARCHIVE_SERVER, fetch_adql, fetch_source_ids
from .io import XPBatch, from_dataframe, read_xp
from .photometry import generate as _generate_photometry
from .spectrum import XPSpectra, pack_correlations
from .writers import write_output

_MANDATORY_XP_COLS = [
    "source_id",
    "bp_n_parameters",
    "bp_coefficients",
    "bp_standard_deviation",
    "rp_n_parameters",
    "rp_coefficients",
    "rp_standard_deviation",
]
_CORR_COLS = [
    "bp_coefficient_errors",
    "bp_coefficient_correlations",
    "rp_coefficient_errors",
    "rp_coefficient_correlations",
]
_TRUNCATION_COLS = ["bp_n_relevant_bases", "rp_n_relevant_bases"]
_SUPPORTED_REQ_FUNCTIONS = {
    "calibrate",
    "convert",
    "generate",
    "apply_colour_equation",
    "apply_error_correction",
    "get_inverse_covariance_matrix",
    "get_inverse_square_root_covariance_matrix",
    "plot_spectra",
}


@dataclass
class ColumnRequirements:
    """Structured description of required/optional columns matching GaiaXPy ``get_required_columns``."""

    function: str
    required: list[str] = field(default_factory=list)
    alternatives: dict[str, list[str]] = field(default_factory=dict)
    conditional: dict[str, list[str]] = field(default_factory=dict)
    optional: dict[str, list[str]] = field(default_factory=dict)
    column_info: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "function": self.function,
            "required": list(self.required),
            "alternatives": dict(self.alternatives),
            "conditional": dict(self.conditional),
            "optional": dict(self.optional),
            "column_info": dict(self.column_info),
            "notes": list(self.notes),
        }


def get_required_columns(function) -> ColumnRequirements:
    """Return input table column requirements for a GaiaXPy / torchgaiaxp function."""
    name = function.__name__ if callable(function) else str(function)
    if name not in _SUPPORTED_REQ_FUNCTIONS:
        raise NotImplementedError(f"Input-column description not implemented for {name!r}.")

    if name == "plot_spectra":
        return ColumnRequirements(
            function=name,
            required=["source_id", "flux", "flux_error"],
            conditional={"multi=True with internal spectra": ["xp"]},
            notes=[
                "Pass sampling explicitly when plotting a DataFrame.",
                "Retain attrs['data_type'] from calibrate or convert when plotting DataFrames.",
            ],
        )
    if name == "apply_error_correction":
        return ColumnRequirements(
            function=name,
            required=["{system}_flux_error_{band}", "DR3_GaiaVega_mag_G"],
            optional={"source identification": ["source_id"]},
            notes=["DR3 correction tables are selected by default unless another release is specified."],
        )

    req = list(_MANDATORY_XP_COLS)
    if name != "apply_colour_equation":
        req = req + _CORR_COLS
    cond = {"truncation=True": list(_TRUNCATION_COLS)} if name in {"calibrate", "convert", "generate"} else {}
    return ColumnRequirements(
        function=name,
        required=req,
        conditional=cond,
        optional={"release identification": ["solution_id", "bp_basis_function_id", "rp_basis_function_id"]},
        notes=[
            "Use NaN/empty values for missing bands rather than zero coefficients.",
            "Include solution_id (or release=...) to select non-DR3 calibrations cleanly.",
        ],
    )


def _is_dataframe(obj) -> bool:
    return hasattr(obj, "columns") and hasattr(obj, "iterrows") and not isinstance(obj, XPBatch)


def _is_series(obj) -> bool:
    return hasattr(obj, "index") and not hasattr(obj, "columns") and type(obj).__name__ == "Series"


def _load_xp_spectra(
    input_object,
    *,
    truncation: bool = False,
    release: str | None = None,
    basis_type: str = "basic",
    username: str | None = None,
    password: str | None = None,
    credentials_file: str | Path | None = None,
    retrieval_type: str | None = None,
    data_release: str | None = None,
    archive_server: str = DEFAULT_ARCHIVE_SERVER,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> XPSpectra:
    """Resolve any GaiaXPy input object (file, zip, hdfs, DataFrame, ADQL, source IDs, XPBatch, XPSpectra)."""
    if isinstance(input_object, XPSpectra):
        xp = input_object.truncated(truncation) if truncation != input_object.truncation else input_object
        if device is not None or dtype != xp.dtype:
            xp = xp.to(device=device, dtype=dtype)
        return xp

    if isinstance(input_object, XPBatch):
        batch = input_object
    elif _is_series(input_object):
        import pandas as pd

        df = pd.DataFrame([input_object.tolist()], columns=input_object.index)
        batch = from_dataframe(df)
    elif _is_dataframe(input_object):
        batch = from_dataframe(input_object)
    elif isinstance(input_object, (list, tuple)):
        batch = fetch_source_ids(
            input_object,
            release=release or _assets.DEFAULT_RELEASE,
            username=username,
            password=password,
            credentials_file=credentials_file,
            archive_server=archive_server,
            retrieval_type=retrieval_type or "XP_CONTINUOUS",
            data_release=data_release,
        )
    elif isinstance(input_object, (str, Path)):
        text = str(input_object)
        if text.startswith("hdfs://") or Path(text).is_file():
            batch = read_xp(text)
        elif isinstance(input_object, str) and text.lstrip().lower().startswith("select"):
            batch = fetch_adql(
                text,
                release=release or _assets.DEFAULT_RELEASE,
                username=username,
                password=password,
                credentials_file=credentials_file,
                archive_server=archive_server,
                retrieval_type=retrieval_type or "XP_CONTINUOUS",
                data_release=data_release,
            )
        else:
            raise FileNotFoundError(f"File {input_object!r} does not exist")
    else:
        raise TypeError(f"Unsupported input_object type: {type(input_object).__name__}")

    return XPSpectra.from_batch(
        batch,
        device=device,
        dtype=dtype,
        truncation=truncation,
        release=release,
        basis_type=basis_type,
    )


def _calibrate_or_convert_to_df(xp: XPSpectra, res: dict, *, is_convert: bool, with_correlation: bool | str):
    import pandas as pd

    sids = (
        xp.source_id.detach().cpu().numpy()
        if xp.source_id is not None
        else np.arange(len(xp), dtype=np.int64)
    )
    flux = res["flux"].detach().cpu().numpy()
    err = res["flux_error"].detach().cpu().numpy() if "flux_error" in res else None
    has_cov = "covariance" in res

    rows = []
    if not is_convert:
        for i in range(len(xp)):
            row: dict = {"source_id": int(sids[i]), "flux": flux[i]}
            if err is not None:
                row["flux_error"] = err[i]
            if has_cov:
                row["correlation"] = pack_correlations(res["covariance"][i]).detach().cpu().numpy()
            elif with_correlation == "factor" and "factor" in res:
                row["factor"] = res["factor"][i].detach().cpu().numpy()
            rows.append(row)
        df = pd.DataFrame(rows)
        df.attrs["data_type"] = "AbsoluteSampledSpectrum"
        df.attrs["data_release"] = xp.release.upper()
        grid = res["wavelength"].detach().cpu().numpy()
        return df, grid

    for i in range(len(xp)):
        for k, band in enumerate(("BP", "RP")):
            if not bool(xp.present[i, k]):
                continue
            row = {"source_id": int(sids[i]), "xp": band, "flux": flux[i, k]}
            if err is not None:
                row["flux_error"] = err[i, k]
            if has_cov:
                row["correlation"] = pack_correlations(res["covariance"][i, k]).detach().cpu().numpy()
            elif with_correlation == "factor" and "factor" in res:
                row["factor"] = res["factor"][i, k].detach().cpu().numpy()
            rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["data_type"] = "XpSampledSpectrum"
    df.attrs["data_release"] = xp.release.upper()
    grid = res["pseudo_wavelength"].detach().cpu().numpy()
    return df, grid


def _photometry_to_df(xp: XPSpectra, results: dict[str, dict]):
    import pandas as pd

    sids = (
        xp.source_id.detach().cpu().numpy()
        if xp.source_id is not None
        else np.arange(len(xp), dtype=np.int64)
    )
    data: dict[str, object] = {"source_id": sids}
    meta_rel = xp.release
    from .photometry import _resolve_system

    for name, res in results.items():
        _, _, _, info, _ = _resolve_system(name, meta_rel)
        label = info.get("full_label", info["label"])
        flux = res["flux"].detach().cpu().numpy()
        mag = res["mag"].detach().cpu().numpy()
        err = res["flux_error"].detach().cpu().numpy() if res.get("flux_error") is not None else None
        for j, band in enumerate(res["bands"]):
            data[f"{label}_mag_{band}"] = mag[:, j]
            data[f"{label}_flux_{band}"] = flux[:, j]
            if err is not None:
                data[f"{label}_flux_error_{band}"] = err[:, j]
    df = pd.DataFrame(data)
    df.attrs["data_type"] = "Photometry"
    df.attrs["data_release"] = xp.release.upper()
    return df


def calibrate(
    input_object,
    sampling=None,
    truncation: bool = False,
    output_path: str | Path = ".",
    output_file: str = "output_spectra",
    output_format: str | None = None,
    save_file: bool = False,
    with_correlation: bool | str = False,
    username: str | None = None,
    password: str | None = None,
    credentials_file: str | Path | None = None,
    release: str | None = None,
    *,
    disable_info: bool = True,
    basis_type: str = "basic",
    overwrite: bool = True,
    retrieval_type: str | None = None,
    data_release: str | None = None,
    archive_server: str = DEFAULT_ARCHIVE_SERVER,
    device=None,
    dtype: torch.dtype = torch.float64,
    return_tensors: bool | None = None,
):
    """GaiaXPy-compatible top-level ``calibrate`` (accepts files, ZIP, DataFrames, ADQL, source IDs, or ``XPSpectra``)."""
    del disable_info
    is_xp = isinstance(input_object, XPSpectra)
    xp = _load_xp_spectra(
        input_object,
        truncation=truncation,
        release=release,
        basis_type=basis_type,
        username=username,
        password=password,
        credentials_file=credentials_file,
        retrieval_type=retrieval_type,
        data_release=data_release,
        archive_server=archive_server,
        device=device,
        dtype=dtype,
    )
    res = xp.calibrate(sampling=sampling, with_correlation=with_correlation, basis_type=basis_type)
    want_tensors = is_xp if return_tensors is None else return_tensors
    if save_file or not want_tensors:
        df, grid = _calibrate_or_convert_to_df(
            xp, res, is_convert=False, with_correlation=with_correlation
        )
        if save_file:
            write_output(
                df,
                output_path=output_path,
                output_file=output_file,
                output_format=output_format,
                sampling=grid,
                overwrite=overwrite,
            )
    if want_tensors:
        return res
    return df, grid


def convert(
    input_object,
    sampling=None,
    truncation: bool = False,
    output_path: str | Path = ".",
    output_file: str = "output_spectra",
    output_format: str | None = None,
    save_file: bool = False,
    with_correlation: bool | str = False,
    username: str | None = None,
    password: str | None = None,
    credentials_file: str | Path | None = None,
    release: str | None = None,
    *,
    disable_info: bool = True,
    overwrite: bool = True,
    retrieval_type: str | None = None,
    data_release: str | None = None,
    archive_server: str = DEFAULT_ARCHIVE_SERVER,
    device=None,
    dtype: torch.dtype = torch.float64,
    return_tensors: bool | None = None,
):
    """GaiaXPy-compatible top-level ``convert`` (accepts files, ZIP, DataFrames, ADQL, source IDs, or ``XPSpectra``)."""
    del disable_info
    is_xp = isinstance(input_object, XPSpectra)
    xp = _load_xp_spectra(
        input_object,
        truncation=truncation,
        release=release,
        username=username,
        password=password,
        credentials_file=credentials_file,
        retrieval_type=retrieval_type,
        data_release=data_release,
        archive_server=archive_server,
        device=device,
        dtype=dtype,
    )
    res = xp.convert(sampling=sampling, with_correlation=with_correlation)
    want_tensors = is_xp if return_tensors is None else return_tensors
    if save_file or not want_tensors:
        df, grid = _calibrate_or_convert_to_df(
            xp, res, is_convert=True, with_correlation=with_correlation
        )
        if save_file:
            write_output(
                df,
                output_path=output_path,
                output_file=output_file,
                output_format=output_format,
                sampling=grid,
                overwrite=overwrite,
            )
    if want_tensors:
        return res
    return df, grid


def generate(
    input_object,
    photometric_system=None,
    truncation: bool = False,
    output_path: str | Path = ".",
    output_file: str = "output_synthetic_photometry",
    output_format: str | None = None,
    save_file: bool = False,
    error_correction: bool = False,
    username: str | None = None,
    password: str | None = None,
    credentials_file: str | Path | None = None,
    release: str | None = None,
    *,
    systems_=None,
    disable_info: bool = True,
    overwrite: bool = True,
    retrieval_type: str | None = None,
    data_release: str | None = None,
    archive_server: str = DEFAULT_ARCHIVE_SERVER,
    device=None,
    dtype: torch.dtype = torch.float64,
    return_tensors: bool | None = None,
):
    """GaiaXPy-compatible top-level ``generate`` (also supports direct ``generate(xp, 'DR3_SDSS')``)."""
    del disable_info
    sys_arg = photometric_system if photometric_system is not None else systems_
    if sys_arg is None:
        raise ValueError("photometric_system is required")
    is_xp = isinstance(input_object, XPSpectra)
    xp = _load_xp_spectra(
        input_object,
        truncation=truncation,
        release=release,
        username=username,
        password=password,
        credentials_file=credentials_file,
        retrieval_type=retrieval_type,
        data_release=data_release,
        archive_server=archive_server,
        device=device,
        dtype=dtype,
    )
    res = _generate_photometry(xp, sys_arg, error_correction=error_correction, truncation=truncation)
    want_tensors = is_xp if return_tensors is None else return_tensors
    if save_file or not want_tensors:
        df = _photometry_to_df(xp, res)
        if save_file:
            write_output(
                df,
                output_path=output_path,
                output_file=output_file,
                output_format=output_format,
                overwrite=overwrite,
            )
    if want_tensors:
        return res
    return df
