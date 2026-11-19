"""Pseudo-wavelength <-> wavelength (GaiaXPy ``pwl_to_wl``/``wl_to_pwl``), in torch.

Linear interpolation of DPAC's tabulated dispersion function with linear
extrapolation, exactly as GaiaXPy's ``interp1d(..., fill_value='extrapolate')``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._assets import (
    BANDS,
    DEFAULT_RELEASE,
    band_wl_range,
    parse_release,
    register_cache_clearer,
    tensor,
)

_SORTED_TABLES: dict[tuple, tuple[Tensor, Tensor]] = {}
register_cache_clearer(_SORTED_TABLES.clear)


def _check_band(band: str) -> str:
    b = band.lower()
    if b not in BANDS:
        raise ValueError(f"invalid band {band!r}; expected 'bp' or 'rp'")
    return b


def _as_coord_tensor(x) -> Tensor:
    if isinstance(x, Tensor):
        if x.is_floating_point():
            return x
        dtype = torch.float32 if x.device.type == "mps" else torch.float64
        return x.to(dtype=dtype)
    return torch.as_tensor(x, dtype=torch.float64)


def _table(band: str, like: Tensor, release: str, inverse: bool = False) -> tuple[Tensor, Tensor]:
    b = _check_band(band)
    rel = parse_release(release)
    key = (rel, b, inverse, str(like.device), like.dtype)
    hit = _SORTED_TABLES.get(key)
    if hit is None:
        kw = {"device": like.device, "dtype": like.dtype, "release": rel}
        wl = tensor("instrument", "dispersion_function_wl", **kw)
        pwl = tensor("instrument", f"dispersion_function_{b}_pwl", **kw)
        keep = torch.isfinite(pwl)
        xp, fp = (pwl[keep], wl[keep]) if inverse else (wl[keep], pwl[keep])
        order = torch.argsort(xp)
        hit = (xp[order].contiguous(), fp[order].contiguous())
        _SORTED_TABLES[key] = hit
    return hit


def _interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    i = torch.searchsorted(xp, x.detach().contiguous()).clamp(1, xp.numel() - 1)
    x0, x1, f0, f1 = xp[i - 1], xp[i], fp[i - 1], fp[i]
    return f0 + (f1 - f0) * (x - x0) / (x1 - x0)


def wl_to_pwl(band: str, wavelength, release: str = DEFAULT_RELEASE) -> Tensor:
    x = _as_coord_tensor(wavelength)
    xp, fp = _table(band, x, release, inverse=False)
    return _interp(x, xp, fp)


def pwl_to_wl(band: str, pseudo_wavelength, release: str = DEFAULT_RELEASE) -> Tensor:
    x = _as_coord_tensor(pseudo_wavelength)
    xp, fp = _table(band, x, release, inverse=True)
    return _interp(x, xp, fp)


def wl_range(band: str, release: str = DEFAULT_RELEASE) -> list[float]:
    return list(band_wl_range(_check_band(band), release))


def pwl_range(band: str, release: str = DEFAULT_RELEASE) -> list[float]:
    low, high = wl_range(band, release)
    return [
        float(wl_to_pwl(band, low, release=release)),
        float(wl_to_pwl(band, high, release=release)),
    ]
