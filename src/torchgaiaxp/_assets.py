"""Packaged GaiaXPy configuration, loaded once and cached per device and dtype."""

from __future__ import annotations

import json
from functools import cache
from importlib import resources

import numpy as np
import torch

BANDS = ("bp", "rp")
#: Band wavelength ranges (nm) GaiaXPy uses for merging and blanking.
BP_WL = (330.0, 643.0)
RP_WL = (635.0, 1020.0)
TELESCOPE_PUPIL_AREA = 0.7278  # m^2
#: hc in W s nm (GaiaXPy: 1e9 * C * PLANCK).
HC = 1.0e9 * 2.99792458e8 * 6.62607004e-34


@cache
def _npz(name: str) -> dict[str, np.ndarray]:
    asset = resources.files("torchgaiaxp") / "_assets" / name
    with resources.as_file(asset) as path, np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


@cache
def systems_meta() -> dict:
    with resources.as_file(resources.files("torchgaiaxp") / "_assets" / "systems.json") as path:
        return json.loads(path.read_text())


def instrument_array(key: str) -> np.ndarray:
    return _npz("instrument.npz")[key]


def system_array(key: str) -> np.ndarray:
    return _npz("systems.npz")[key]


_TENSORS: dict[tuple, torch.Tensor] = {}


def tensor(store: str, key: str, device=None, dtype=torch.float64) -> torch.Tensor:
    """An asset as a tensor on ``device``/``dtype``, converted once and cached."""
    device = torch.device(device) if device is not None else torch.device("cpu")
    cache_key = (store, key, str(device), dtype)
    hit = _TENSORS.get(cache_key)
    if hit is None:
        array = instrument_array(key) if store == "instrument" else system_array(key)
        hit = torch.as_tensor(array, dtype=dtype).to(device)
        _TENSORS[cache_key] = hit
    return hit
