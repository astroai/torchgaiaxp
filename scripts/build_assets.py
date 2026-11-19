"""Extract GaiaXPy configuration tables per release into torchgaiaxp npz/json assets.

Run with GaiaXPy (>=2.2) installed (build time only). The runtime never imports it.
GaiaXPy's configuration is BSD-3 (DPAC CU5/DPCI); see LICENSE-GAIAXPY.
"""

from __future__ import annotations

import json
from ast import literal_eval
from pathlib import Path

import gaiaxpy
import numpy as np
import pandas as pd
import yaml
from gaiaxpy import PhotometricSystem
from gaiaxpy.calibrator.external_instrument_model import ExternalInstrumentModel
from gaiaxpy.config.paths import config_path
from gaiaxpy.converter.config import get_bands_config
from gaiaxpy.core import satellite
from gaiaxpy.core.config import load_xpmerge_from_file, load_xpsampling_from_file
from gaiaxpy.core.config_dispatcher import (
    CalibratorDispatcher,
    ConverterDispatcher,
    ErrorCorrectorDispatcher,
)
from gaiaxpy.core.generic_functions import parse_config
from gaiaxpy.core.generic_variables import (
    DEFAULT_MODELS,
    INV_BASIS_TYPES,
    RELEASES,
    n_params_dict,
    release_solution_id,
)
from scipy import interpolate

OUT = Path(__file__).resolve().parents[1] / "src" / "torchgaiaxp" / "_assets"
CONFIG = Path(config_path)

LEGACY_DR3_ALIASES = {
    "Gaia_DR3_Vega": "DR3_Gaia_Vega",
}


def _wl_ranges_for_release(release: str) -> tuple[list[float], list[float]]:
    bp_attr = getattr(satellite, f"BP_WL_{release.upper()}", satellite.BP_WL_DR3)
    rp_attr = getattr(satellite, f"RP_WL_{release.upper()}", satellite.RP_WL_DR3)
    return [float(bp_attr.low), float(bp_attr.high)], [float(rp_attr.low), float(rp_attr.high)]


def instrument_for_release(release: str) -> tuple[dict[str, np.ndarray], dict]:
    arrays: dict[str, np.ndarray] = {}
    basis_types = list(INV_BASIS_TYPES)
    models = DEFAULT_MODELS[release]
    n_bases_map = n_params_dict[release]

    for basis_type in basis_types:
        cal_disp = CalibratorDispatcher()
        try:
            samp_files = cal_disp.build(sampling=np.array([400.0, 500.0]), release=release, basis_type=basis_type)
        except ValueError:
            continue
        for band in ("bp", "rp"):
            model_ = ExternalInstrumentModel.from_config_npz(
                samp_files[band]["dispersion"],
                samp_files[band]["response"],
                samp_files[band]["bases"],
            )
            for name, table, x, y in (
                ("dispersion", model_.dispersion, "wavelength", "pseudo-wavelength"),
                ("response", model_.response, "wavelength", "response"),
            ):
                t, c, k = interpolate.splrep(table[x], table[y], s=0)
                arrays[f"{basis_type}/{band}_{name}_knots"] = np.asarray(t, np.float64)
                arrays[f"{basis_type}/{band}_{name}_coeffs"] = np.asarray(c, np.float64)
                arrays[f"{basis_type}/{band}_{name}_degree"] = np.array(k)
                arrays[f"{basis_type}/{band}_{name}_x"] = np.asarray(table[x], np.float64)
                arrays[f"{basis_type}/{band}_{name}_y"] = np.asarray(table[y], np.float64)
            bases = model_.bases
            n_b = int(bases["nBases"])
            inv = np.asarray(bases["inverseBasesCoefficients"], np.float64).reshape(
                n_b, int(bases["nInverseBasesCoefficients"])
            )
            tr = np.asarray(bases["transformationMatrix"], np.float64).reshape(
                n_b, int(bases["nTransformedBases"])
            )
            arrays[f"{basis_type}/{band}_inverse_coefficients"] = inv
            arrays[f"{basis_type}/{band}_inverse_transform"] = tr
            # Pre-combined contiguous [n_inv, n_bases] for single-matmul & MPS safety.
            arrays[f"{basis_type}/{band}_inverse_matrix"] = np.ascontiguousarray((tr @ inv).T)
            arrays[f"{basis_type}/{band}_pwl_range"] = np.array(
                [bases["pwlRangeMin"], bases["pwlRangeMax"]], np.float64
            )
            arrays[f"{basis_type}/{band}_norm_range"] = np.array(
                [bases["normRangeMin"], bases["normRangeMax"]], np.float64
            )

    conv_file = ConverterDispatcher().build(release=release)
    internal = get_bands_config(parse_config(conv_file))
    for band, cfg in (("bp", internal.bpConfig), ("rp", internal.rpConfig)):
        n = int(cfg.dimension)
        tr = np.asarray(cfg.transformationMatrix, np.float64).reshape(
            n, int(cfg.transformedSetDimension)
        )
        arrays[f"{band}_internal_range"] = np.array(cfg.range, np.float64)
        arrays[f"{band}_internal_norm_range"] = np.array(cfg.normalizedRange, np.float64)
        arrays[f"{band}_internal_transform"] = tr
        arrays[f"{band}_internal_transform_T"] = np.ascontiguousarray(tr.T)
        arrays[f"{band}_basis_function_id"] = np.array(int(cfg.uniqueId))

    with open(CONFIG / "config.yaml", encoding="utf-8") as f:
        cfg_yaml = yaml.safe_load(f)
    disp_cfg = cfg_yaml["core"][release]
    disp_path = CONFIG / disp_cfg["path"] / disp_cfg["dispersion_function"]
    table = pd.read_csv(disp_path, comment="#", float_precision="round_trip")
    arrays["dispersion_function_wl"] = table["wl_nm"].to_numpy(np.float64)
    arrays["dispersion_function_bp_pwl"] = table["bp_pwl"].to_numpy(np.float64)
    arrays["dispersion_function_rp_pwl"] = table["rp_pwl"].to_numpy(np.float64)

    nosamp_file = CalibratorDispatcher().build(sampling=None, release=release, basis_type="basic")
    grid, merge = load_xpmerge_from_file(nosamp_file)
    sampling = load_xpsampling_from_file(nosamp_file)
    arrays["default_grid"] = np.asarray(grid, np.float64)
    # Store design already in [2, G, N] contiguous layout.
    arrays["default_design"] = np.ascontiguousarray(
        np.stack([np.asarray(sampling[b], np.float64).T for b in ("bp", "rp")])
    )
    arrays["default_merge"] = np.stack([np.asarray(merge[b], np.float64) for b in ("bp", "rp")])

    bp_wl, rp_wl = _wl_ranges_for_release(release)
    meta = {
        "release": release,
        "bp_wl": bp_wl,
        "rp_wl": rp_wl,
        "n_bases": {b: int(n_bases_map[b]) for b in ("bp", "rp")},
        "models": dict(models),
        "basis_types": basis_types,
        "solution_ids": list(release_solution_id.get(release, [])),
        "gaiaxpy_version": gaiaxpy.__version__,
    }
    return arrays, meta


def systems_for_release(release: str) -> tuple[dict[str, np.ndarray], dict]:
    arrays: dict[str, np.ndarray] = {}
    meta: dict = {}
    aliases: dict[str, str] = {}
    prefix = f"{release.upper()}_"

    for member in PhotometricSystem:
        system = member.value
        if getattr(system, "release", release) != release:
            continue
        sys_file = system.get_system_file()
        _, merge = load_xpmerge_from_file(sys_file)
        sampling = load_xpsampling_from_file(sys_file)
        # Store design in [2, nb, N] contiguous layout.
        arrays[f"{member.name}/design"] = np.ascontiguousarray(
            np.stack([np.asarray(sampling[b], np.float64).T for b in ("bp", "rp")])
        )
        arrays[f"{member.name}/merge"] = np.stack(
            [np.asarray(merge[b], np.float64) for b in ("bp", "rp")]
        )
        arrays[f"{member.name}/zero_points"] = np.asarray(member.get_zero_points(), np.float64)
        offsets = member.get_offsets()
        arrays[f"{member.name}/offsets"] = (
            np.zeros(len(member.get_bands()), np.float64)
            if offsets is None
            else np.asarray(offsets, np.float64)
        )
        full_label = member.get_system_full_label()
        meta[member.name] = {
            "label": member.get_system_label(),
            "full_label": full_label,
            "bands": list(member.get_bands()),
            "standardised": type(system).__name__.startswith("Standardised"),
            "release": release,
        }
        if member.name.startswith(prefix):
            aliases[member.name[len(prefix) :]] = member.name

    if release == "dr3":
        aliases.update(LEGACY_DR3_ALIASES)

    with open(CONFIG / "config.yaml", encoding="utf-8") as f:
        cfg_yaml = yaml.safe_load(f)

    correction = {}
    err_disp = ErrorCorrectorDispatcher()
    corr_dir, _ = err_disp.resolve(release)
    for path in sorted(Path(corr_dir).glob("DIDREQ-465-*-correction-factors.csv")):
        label = path.name.split("-")[2]
        frame = pd.read_csv(path, comment="#", float_precision="round_trip")
        factor_columns = [c for c in frame.columns if c.startswith("factor_")]
        arrays[f"correction/{label}/min"] = frame["min_Gmag_bin"].to_numpy(np.float64)
        arrays[f"correction/{label}/max"] = frame["max_Gmag_bin"].to_numpy(np.float64)
        arrays[f"correction/{label}/factors"] = frame[factor_columns].to_numpy(np.float64)
        correction[label] = [c.removeprefix("factor_") for c in factor_columns]

    colour = {}
    ce_cfg = cfg_yaml.get("colour_equation", {}).get(release)
    if ce_cfg is not None:
        ce_dir = CONFIG / ce_cfg["path"]
        for path in sorted(ce_dir.glob("*_colour_eq.yaml")):
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            label = next(k for k in data if k != "gaiaxpy_metadata")
            section = data[label]
            poly = section["POLY_COEFFICIENTS"]
            if isinstance(poly, str):
                poly = literal_eval(poly)
            crange = section["COLOUR_RANGE"]
            if isinstance(crange, str):
                crange = literal_eval(crange)
            colour[label] = {
                "filter": section["FILTER"],
                "colour_index": section["COLOUR_INDEX"],
                "colour_range": [float(v) for v in crange],
                # numpy.poly1d order (highest power first), as GaiaXPy uses it.
                "poly": [float(v) for v in list(poly)[::-1]],
            }

    gaia_vega_name = f"{release.upper()}_Gaia_Vega"
    meta["_aliases"] = aliases
    meta["_correction_bands"] = correction
    meta["_colour_equations"] = colour
    meta["_gaia_vega_system"] = gaia_vega_name if gaia_vega_name in meta else None
    meta["_gaiaxpy_version"] = gaiaxpy.__version__
    return arrays, meta


if __name__ == "__main__":
    for rel in RELEASES:
        rel_dir = OUT / rel
        rel_dir.mkdir(parents=True, exist_ok=True)
        inst_arrays, inst_meta = instrument_for_release(rel)
        np.savez_compressed(rel_dir / "instrument.npz", **inst_arrays)
        (rel_dir / "instrument.json").write_text(json.dumps(inst_meta, indent=1, sort_keys=True))
        sys_arrays, sys_meta = systems_for_release(rel)
        np.savez_compressed(rel_dir / "systems.npz", **sys_arrays)
        (rel_dir / "systems.json").write_text(json.dumps(sys_meta, indent=1, sort_keys=True))
        print(f"assets written for {rel} to {rel_dir}")
