"""Basis functions of the XP continuous representation, evaluated anywhere.

Everything here is torch, differentiable, and runs on the device of its input:

* :func:`hermite_functions` -- orthonormal Hermite functions by recurrence;
* :func:`bspline` -- FITPACK ``splev`` (de Boor, with its end-polynomial
  extrapolation) for the ``splrep`` tables of the instrument model;
* :func:`absolute_design` -- the externally calibrated ("inverse") bases at any
  wavelengths: what GaiaXPy's ``calibrate`` builds for a custom sampling, here
  for a tensor of wavelengths, so an XP spectrum can be evaluated on another
  spectrum's own grid;
* :func:`internal_design` -- the internally calibrated bases on a
  pseudo-wavelength grid (``convert``).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ._assets import BP_WL, HC, RP_WL, TELESCOPE_PUPIL_AREA, tensor


def hermite_functions(x: Tensor, n: int) -> Tensor:
    """``psi_k(x)``, k < n, orthonormal Hermite functions: ``[..., n]``."""
    psi0 = math.pi ** (-0.25) * torch.exp(-0.5 * x * x)
    out = [psi0]
    if n > 1:
        out.append(math.sqrt(2.0) * x * psi0)
    for k in range(2, n):
        out.append(math.sqrt(2.0 / k) * x * out[k - 1] - math.sqrt((k - 1) / k) * out[k - 2])
    return torch.stack(out[:n], dim=-1)


def bspline(x: Tensor, knots: Tensor, coeffs: Tensor, degree: int) -> Tensor:
    """FITPACK ``splev(x, (t, c, k), ext=0)``: de Boor, extrapolating the end pieces."""
    t, c, k = knots, coeffs, int(degree)
    n = t.numel() - k - 1
    # Interval index i with t[i] <= x < t[i+1], restricted to [k, n-1].
    i = torch.searchsorted(t.contiguous(), x.detach().contiguous(), right=True) - 1
    i = i.clamp(k, n - 1)
    d = [c[(i - k + j).clamp(0, c.numel() - 1)] for j in range(k + 1)]
    for r in range(1, k + 1):
        for j in range(k, r - 1, -1):
            left = t[i + j - k]
            right = t[i + 1 + j - r]
            alpha = (x - left) / (right - left)
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
    return d[k]


def _spline(band: str, name: str, x: Tensor) -> Tensor:
    kw = {"device": x.device, "dtype": x.dtype}
    return bspline(
        x,
        tensor("instrument", f"{band}_{name}_knots", **kw),
        tensor("instrument", f"{band}_{name}_coeffs", **kw),
        int(tensor("instrument", f"{band}_{name}_degree")),
    )


def band_range(band: str) -> tuple[float, float]:
    return BP_WL if band == "bp" else RP_WL


def merge_weights(wavelength: Tensor) -> Tensor:
    """GaiaXPy's BP/RP blending weights at ``wavelength``: ``[2, ...]``."""
    low, high = RP_WL[0], BP_WL[1]
    ramp = ((wavelength - low) / (high - low)).clamp(0.0, 1.0)
    return torch.stack([1.0 - ramp, ramp])


def absolute_design(wavelength: Tensor, band: str, weight: Tensor | None = None) -> Tensor:
    """Externally calibrated bases ``[..., 55]`` at ``wavelength`` (nm).

    ``weight`` (same shape) marks where the band contributes; the basis is zero
    where it is not positive, as in GaiaXPy. By default GaiaXPy's merge weight
    for the band (BP below 643 nm, RP above 635 nm); pass ``torch.ones_like``
    to evaluate one band's bases beyond the blend.
    Multiplying by the coefficients gives flux in W m^-2 nm^-1.
    """
    wl = wavelength
    kw = {"device": wl.device, "dtype": wl.dtype}
    if weight is None:
        weight = merge_weights(wl)[0 if band == "bp" else 1]
    pwl_range = tensor("instrument", f"{band}_pwl_range", **kw)
    norm_range = tensor("instrument", f"{band}_norm_range", **kw)
    scale = (norm_range[1] - norm_range[0]) / (pwl_range[1] - pwl_range[0])
    offset = norm_range[0] - pwl_range[0] * scale
    x = _spline(band, "dispersion", wl) * scale + offset
    inverse = tensor("instrument", f"{band}_inverse_coefficients", **kw)
    transform = tensor("instrument", f"{band}_inverse_transform", **kw)
    n_bases = transform.shape[0] if transform.dim() == 2 else math.isqrt(transform.numel())
    inverse = inverse.reshape(n_bases, -1)
    transform = transform.reshape(n_bases, -1)
    hermite = hermite_functions(x, inverse.shape[1]) * (weight > 0).unsqueeze(-1)
    design = hermite @ inverse.T @ transform.T  # [..., 55]
    response = _spline(band, "response", wl)
    # GaiaXPy: norm = hc / (A r lambda) where r > 0, else 0. The safe-where keeps
    # gradients finite where the basis is masked out anyway.
    use = (response > 0) & (weight > 0)
    safe = torch.where(use, response, torch.ones_like(response))
    norm = torch.where(use, HC / (TELESCOPE_PUPIL_AREA * safe * wl), 0.0)
    return design * norm.unsqueeze(-1)


def internal_design(pseudo_wavelength: Tensor, band: str) -> Tensor:
    """Internally calibrated Hermite bases ``[..., 55]`` at pseudo-wavelengths."""
    pwl = pseudo_wavelength
    kw = {"device": pwl.device, "dtype": pwl.dtype}
    low, high = tensor("instrument", f"{band}_internal_range", **kw)
    nlow, nhigh = tensor("instrument", f"{band}_internal_norm_range", **kw)
    scale = (nhigh - nlow) / (high - low)
    x = pwl * scale + (nlow - low * scale)
    transform = tensor("instrument", f"{band}_internal_transform", **kw)
    return hermite_functions(x, transform.shape[0]) @ transform.T
