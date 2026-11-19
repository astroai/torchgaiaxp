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

from ._assets import (
    BANDS,
    DEFAULT_RELEASE,
    HC,
    TELESCOPE_PUPIL_AREA,
    band_wl_range,
    instrument_meta,
    parse_release,
    tensor,
)


def _check_band(band: str) -> str:
    b = band.lower()
    if b not in BANDS:
        raise ValueError(f"invalid band {band!r}; expected 'bp' or 'rp'")
    return b


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


def _spline(
    band: str,
    name: str,
    x: Tensor,
    *,
    release: str = DEFAULT_RELEASE,
    basis_type: str = "basic",
) -> Tensor:
    kw = {"device": x.device, "dtype": x.dtype, "release": release}
    deg = int(tensor("instrument", f"{basis_type}/{band}_{name}_degree", release=release))
    return bspline(
        x,
        tensor("instrument", f"{basis_type}/{band}_{name}_knots", **kw),
        tensor("instrument", f"{basis_type}/{band}_{name}_coeffs", **kw),
        deg,
    )


def band_range(band: str, release: str = DEFAULT_RELEASE) -> tuple[float, float]:
    return band_wl_range(_check_band(band), release)


def merge_weights(wavelength: Tensor, release: str = DEFAULT_RELEASE) -> Tensor:
    """GaiaXPy's BP/RP blending weights at ``wavelength``: ``[2, ...]``."""
    low = band_wl_range("rp", release)[0]
    high = band_wl_range("bp", release)[1]
    ramp = ((wavelength - low) / (high - low)).clamp(0.0, 1.0)
    return torch.stack([1.0 - ramp, ramp])


def absolute_design(
    wavelength: Tensor,
    band: str,
    weight: Tensor | None = None,
    *,
    release: str = DEFAULT_RELEASE,
    basis_type: str = "basic",
) -> Tensor:
    """Externally calibrated bases ``[..., N]`` at ``wavelength`` (nm).

    ``weight`` (same shape) marks where the band contributes; the basis is zero
    where it is not positive, as in GaiaXPy. By default GaiaXPy's merge weight
    for the band (BP below 643 nm, RP above 635 nm in DR3); pass
    ``torch.ones_like`` to evaluate one band's bases beyond the blend.
    Multiplying by the coefficients gives flux in W m^-2 nm^-1.
    """
    b = _check_band(band)
    rel = parse_release(release)
    bt = basis_type.lower()
    allowed = instrument_meta(rel).get("basis_types", ["basic"])
    if bt not in allowed:
        raise ValueError(f"basis_type {basis_type!r} not supported for release {rel!r}; allowed: {allowed}")
    wl = wavelength
    kw = {"device": wl.device, "dtype": wl.dtype, "release": rel}
    if weight is None:
        weight = merge_weights(wl, release=rel)[0 if b == "bp" else 1]
    pwl_range = tensor("instrument", f"{bt}/{b}_pwl_range", **kw)
    norm_range = tensor("instrument", f"{bt}/{b}_norm_range", **kw)
    scale = (norm_range[1] - norm_range[0]) / (pwl_range[1] - pwl_range[0])
    offset = norm_range[0] - pwl_range[0] * scale
    x = _spline(b, "dispersion", wl, release=rel, basis_type=bt) * scale + offset
    inv_mat = tensor("instrument", f"{bt}/{b}_inverse_matrix", **kw)  # [n_inv, n_bases], contiguous
    hermite = hermite_functions(x, inv_mat.shape[0]) * (weight > 0).unsqueeze(-1)
    design = hermite @ inv_mat  # [..., n_bases]
    response = _spline(b, "response", wl, release=rel, basis_type=bt)
    # GaiaXPy: norm = hc / (A r lambda) where r > 0, else 0. Guard both response and wl
    # so autograd stays finite even if wl == 0 or response <= 0.
    use = (response > 0) & (weight > 0) & (wl > 0)
    safe_r = torch.where(use, response, torch.ones_like(response))
    safe_wl = torch.where(use, wl, torch.ones_like(wl))
    norm = torch.where(use, HC / (TELESCOPE_PUPIL_AREA * safe_r * safe_wl), 0.0)
    return design * norm.unsqueeze(-1)


def internal_design(
    pseudo_wavelength: Tensor,
    band: str,
    *,
    release: str = DEFAULT_RELEASE,
) -> Tensor:
    """Internally calibrated Hermite bases ``[..., N]`` at pseudo-wavelengths."""
    b = _check_band(band)
    rel = parse_release(release)
    pwl = pseudo_wavelength
    kw = {"device": pwl.device, "dtype": pwl.dtype, "release": rel}
    low_high = tensor("instrument", f"{b}_internal_range", **kw)
    nlow_nhigh = tensor("instrument", f"{b}_internal_norm_range", **kw)
    scale = (nlow_nhigh[1] - nlow_nhigh[0]) / (low_high[1] - low_high[0])
    x = pwl * scale + (nlow_nhigh[0] - low_high[0] * scale)
    transform_t = tensor("instrument", f"{b}_internal_transform_T", **kw)
    return hermite_functions(x, transform_t.shape[0]) @ transform_t
