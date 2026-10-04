"""Pseudo-wavelength <-> wavelength (GaiaXPy ``pwl_to_wl``/``wl_to_pwl``), in torch.

Linear interpolation of DPAC's tabulated dispersion function with linear
extrapolation, exactly as GaiaXPy's ``interp1d(..., fill_value='extrapolate')``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._assets import BP_WL, RP_WL, tensor


def _interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    order = torch.argsort(xp)
    xp, fp = xp[order], fp[order]
    i = torch.searchsorted(xp.contiguous(), x.detach().contiguous()).clamp(1, xp.numel() - 1)
    x0, x1, f0, f1 = xp[i - 1], xp[i], fp[i - 1], fp[i]
    return f0 + (f1 - f0) * (x - x0) / (x1 - x0)


def _table(band: str, like: Tensor) -> tuple[Tensor, Tensor]:
    kw = {"device": like.device, "dtype": like.dtype if like.is_floating_point() else torch.float64}
    wl = tensor("instrument", "dispersion_function_wl", **kw)
    pwl = tensor("instrument", f"dispersion_function_{band.lower()}_pwl", **kw)
    keep = torch.isfinite(pwl)
    return wl[keep], pwl[keep]


def wl_to_pwl(band: str, wavelength) -> Tensor:
    wavelength = torch.as_tensor(wavelength, dtype=torch.float64)
    wl, pwl = _table(band, wavelength)
    return _interp(wavelength, wl, pwl)


def pwl_to_wl(band: str, pseudo_wavelength) -> Tensor:
    pseudo_wavelength = torch.as_tensor(pseudo_wavelength, dtype=torch.float64)
    wl, pwl = _table(band, pseudo_wavelength)
    return _interp(pseudo_wavelength, pwl, wl)


def wl_range(band: str) -> list[float]:
    return list(BP_WL if band.lower() == "bp" else RP_WL)


def pwl_range(band: str) -> list[float]:
    low, high = wl_range(band)
    return [float(wl_to_pwl(band, low)), float(wl_to_pwl(band, high))]
