"""Extract every GaiaXPy configuration table torchgaiaxp needs into npz assets.

Run with GaiaXPy installed (build time only). The runtime never imports it.
GaiaXPy's configuration is BSD-3 (DPAC CU5/DPCI); see LICENSE-GAIAXPY.
"""

from __future__ import annotations

import configparser
import json
from ast import literal_eval
from pathlib import Path

import gaiaxpy
import numpy as np
import pandas as pd
from gaiaxpy import PhotometricSystem
from gaiaxpy.calibrator.external_instrument_model import ExternalInstrumentModel
from gaiaxpy.config.paths import config_path, hermite_bases_file
from gaiaxpy.converter.config import get_bands_config, parse_config
from gaiaxpy.core.config import load_xpmerge_from_xml, load_xpsampling_from_xml
from scipy import interpolate

OUT = Path(__file__).resolve().parents[1] / "src" / "torchgaiaxp" / "_assets"
CONFIG = Path(config_path)
MODELS = {"bp": "v375wi", "rp": "v142r"}


def instrument() -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for band, model in MODELS.items():
        stem = f"{band}C03_{model}"
        model_ = ExternalInstrumentModel.from_config_csv(
            str(CONFIG / f"{stem}_dispersion.csv"),
            str(CONFIG / f"{stem}_response.csv"),
            str(CONFIG / f"{stem}_bases.csv"),
        )
        for name, table, x, y in (
            ("dispersion", model_.dispersion, "wavelength", "pseudo-wavelength"),
            ("response", model_.response, "wavelength", "response"),
        ):
            t, c, k = interpolate.splrep(table[x], table[y], s=0)
            arrays[f"{band}_{name}_knots"] = np.asarray(t, np.float64)
            arrays[f"{band}_{name}_coeffs"] = np.asarray(c, np.float64)
            arrays[f"{band}_{name}_degree"] = np.array(k)
            arrays[f"{band}_{name}_x"] = np.asarray(table[x], np.float64)
            arrays[f"{band}_{name}_y"] = np.asarray(table[y], np.float64)
        bases = model_.bases
        arrays[f"{band}_inverse_coefficients"] = np.asarray(
            bases["inverseBasesCoefficients"], np.float64
        )
        arrays[f"{band}_inverse_transform"] = np.asarray(bases["transformationMatrix"], np.float64)
        arrays[f"{band}_pwl_range"] = np.array(
            [bases["pwlRangeMin"], bases["pwlRangeMax"]], np.float64
        )
        arrays[f"{band}_norm_range"] = np.array(
            [bases["normRangeMin"], bases["normRangeMax"]], np.float64
        )
    internal = get_bands_config(parse_config(hermite_bases_file))
    for band, cfg in (("bp", internal.bpConfig), ("rp", internal.rpConfig)):
        n = int(cfg.dimension)
        arrays[f"{band}_internal_range"] = np.array(cfg.range, np.float64)
        arrays[f"{band}_internal_norm_range"] = np.array(cfg.normalizedRange, np.float64)
        arrays[f"{band}_internal_transform"] = np.asarray(
            cfg.transformationMatrix, np.float64
        ).reshape(n, int(cfg.transformedSetDimension))
        arrays[f"{band}_basis_function_id"] = np.array(int(cfg.uniqueId))
    cfg = configparser.ConfigParser()
    cfg.read(CONFIG / "config.ini")
    table = pd.read_csv(CONFIG / cfg.get("core", "dispersion_function"))
    arrays["dispersion_function_wl"] = table["wl_nm"].to_numpy(np.float64)
    arrays["dispersion_function_bp_pwl"] = table["bp_pwl"].to_numpy(np.float64)
    arrays["dispersion_function_rp_pwl"] = table["rp_pwl"].to_numpy(np.float64)
    grid, merge = load_xpmerge_from_xml()
    sampling = load_xpsampling_from_xml()
    arrays["default_grid"] = np.asarray(grid, np.float64)
    arrays["default_design"] = np.stack([np.asarray(sampling[b], np.float64) for b in MODELS])
    arrays["default_merge"] = np.stack([np.asarray(merge[b], np.float64) for b in MODELS])
    return arrays


def systems() -> tuple[dict[str, np.ndarray], dict]:
    arrays: dict[str, np.ndarray] = {}
    meta: dict = {}
    for member in PhotometricSystem:
        system = member.value
        label = system.get_system_label()
        sampling = system.load_xpsampling_from_xml()
        _, merge = system.load_xpmerge_from_xml()
        arrays[f"{member.name}/design"] = np.stack(
            [np.asarray(sampling[b], np.float64) for b in MODELS]
        )
        arrays[f"{member.name}/merge"] = np.stack(
            [np.asarray(merge[b], np.float64) for b in MODELS]
        )
        arrays[f"{member.name}/zero_points"] = np.asarray(system.get_zero_points(), np.float64)
        offsets = system.get_offsets()
        arrays[f"{member.name}/offsets"] = (
            np.zeros(len(system.get_bands()))
            if offsets is None
            else np.asarray(offsets, np.float64)
        )
        meta[member.name] = {
            "label": label,
            "bands": list(system.get_bands()),
            "standardised": type(system).__name__.startswith("Standardised"),
        }
    correction = {}
    for path in sorted((CONFIG / "correction_tables").glob("DIDREQ-465-*-correction-factors.csv")):
        label = path.name.split("-")[2]
        frame = pd.read_csv(path, float_precision="round_trip")
        factor_columns = [c for c in frame.columns if c.startswith("factor_")]
        arrays[f"correction/{label}/min"] = frame["min_Gmag_bin"].to_numpy(np.float64)
        arrays[f"correction/{label}/max"] = frame["max_Gmag_bin"].to_numpy(np.float64)
        arrays[f"correction/{label}/factors"] = frame[factor_columns].to_numpy(np.float64)
        correction[label] = [c.removeprefix("factor_") for c in factor_columns]
    colour = {}
    for path in sorted((CONFIG / "colour_eq_files").glob("*_colour_eq.ini")):
        parser = configparser.ConfigParser()
        parser.read(path)
        label = parser.sections()[0]
        section = parser[label]
        colour[label] = {
            "filter": section["FILTER"],
            "colour_index": section["COLOUR_INDEX"],
            "colour_range": list(literal_eval(section["COLOUR_RANGE"])),
            # numpy.poly1d order (highest power first), as GaiaXPy uses it.
            "poly": list(literal_eval(section["POLY_COEFFICIENTS"]))[::-1],
        }
    meta["_correction_bands"] = correction
    meta["_colour_equations"] = colour
    meta["_gaiaxpy_version"] = gaiaxpy.__version__
    return arrays, meta


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "instrument.npz", **instrument())
    arrays, meta = systems()
    np.savez_compressed(OUT / "systems.npz", **arrays)
    (OUT / "systems.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    print("assets written to", OUT)
