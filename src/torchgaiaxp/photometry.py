"""Synthetic photometry: GaiaXPy ``generate`` for all its photometric systems.

For each system: band fluxes ``sum_b w_b D_b c_b`` (W m^-2 Hz^-1 for AB-like
systems, W m^-2 nm^-1 for Vega-like ones, as GaiaXPy), errors from the
coefficient covariance, magnitudes ``-2.5 log10(f) + zp`` (NaN for f <= 0).
Then, as GaiaXPy does, in order:

1. standardised systems (``*_Std``): flux + offset, error x |(f + off) / f|;
2. colour equations (JKC_Std U from B-V, SDSS_Std u from g-i): polynomial
   inside the colour range, linear extrapolation outside, error propagated
   through the polynomial's derivative;
3. optional error correction (Montegriffo et al. 2023): flux errors scaled by
   factors tabulated in Gaia_DR3_Vega G.

Photometry needs both bands: a source with one band missing is NaN, as in
GaiaXPy.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from . import _assets
from .design import XPDesign

LN10 = math.log(10.0)


def systems() -> list[str]:
    """The available photometric system names (GaiaXPy ``PhotometricSystem``)."""
    return sorted(k for k in _assets.systems_meta() if not k.startswith("_"))


def _raw(xp, system: str):
    design = XPDesign.photometric(system, xp.device, xp.dtype)
    flux_b, factor_b = xp._band_samples(design, xp.errors is not None)
    w = design.merge.unsqueeze(0)  # [1, 2, nb]
    flux = (w * flux_b).sum(1)
    error = None
    if factor_b is not None:
        # GaiaXPy: sqrt(w_bp^2 e_bp^2 + w_rp^2 e_rp^2) with per-band diagonal errors.
        band_err = factor_b.square().sum(-1)  # [B, 2, nb]
        error = (w.square() * band_err).sum(1).sqrt()
    both = xp.present.all(-1, keepdim=True)
    flux = torch.where(both, flux, torch.nan)
    if error is not None:
        error = torch.where(both, error, torch.nan)
    return design, flux, error


def _mag(flux: Tensor, zp: Tensor) -> Tensor:
    return torch.where(flux > 0, -2.5 * torch.log10(flux.clamp_min(1e-300)) + zp, torch.nan)


def _poly(coeffs: list[float], x: Tensor) -> Tensor:
    out = torch.zeros_like(x)
    for c in coeffs:  # highest power first
        out = out * x + c
    return out


def _poly_deriv(coeffs: list[float], x: Tensor) -> Tensor:
    n = len(coeffs) - 1
    return _poly([c * (n - i) for i, c in enumerate(coeffs[:-1])], x)


def _colour_equation(result: dict, label: str, spec: dict, zp: dict[str, float]):
    bands = result["bands"]
    target = bands.index(spec["filter"])
    b0, b1 = spec["colour_index"].split("-")
    i0, i1 = bands.index(b0), bands.index(b1)
    flux, err, mag = result["flux"], result["flux_error"], result["mag"]
    colour = mag[..., i0] - mag[..., i1]
    low, high = min(spec["colour_range"]), max(spec["colour_range"])
    coeffs = spec["poly"]
    limit = colour.clamp(low, high)
    inside = (colour >= low) & (colour <= high)
    correction = torch.where(
        inside,
        _poly(coeffs, colour),
        _poly(coeffs, limit) + _poly_deriv(coeffs, limit) * (colour - limit),
    )
    corrected = mag[..., target] + correction
    out_flux = 10.0 ** (-0.4 * (corrected - zp[spec["filter"]]))
    result["mag"] = _set(mag, target, corrected)
    result["flux"] = _set(flux, target, out_flux)
    if err is not None:

        def mag_err(i):
            return 2.5 * err[..., i] / (flux[..., i] * LN10)

        colour_err = torch.sqrt(mag_err(i0) ** 2 + mag_err(i1) ** 2)
        # GaiaXPy evaluates the derivative at the colour itself, even outside the range.
        corr_err = colour_err * _poly_deriv(coeffs, colour).abs()
        total = torch.sqrt(mag_err(target) ** 2 + corr_err**2)
        result["flux_error"] = _set(err, target, total * out_flux * LN10 / 2.5)


def _set(x: Tensor, index: int, value: Tensor) -> Tensor:
    x = x.clone()
    x[..., index] = value
    return x


def correction_factors(label: str, g_mag: Tensor) -> Tensor:
    """Error-correction factors ``[B, nb]`` for a system label at Gaia G (Vega)."""
    kw = {"device": g_mag.device, "dtype": g_mag.dtype}
    lo = _assets.tensor("systems", f"correction/{label}/min", **kw)
    hi = _assets.tensor("systems", f"correction/{label}/max", **kw)
    factors = _assets.tensor("systems", f"correction/{label}/factors", **kw)
    centre = 0.5 * (lo + hi)
    floor = torch.floor(g_mag)
    # Row whose min equals floor(G) (GaiaXPy's lookup), clamped to the table.
    row = torch.searchsorted(lo.contiguous(), floor.contiguous(), right=True) - 1
    row = row.clamp(0, lo.numel() - 1)
    f_row = factors[row]
    nxt = (row + 1).clamp(max=lo.numel() - 1)
    has_next = (row + 1 < lo.numel()) & (lo[nxt] == hi[row])
    t = ((g_mag - centre[row]) / (hi[row] - centre[row])).unsqueeze(-1)
    interp = f_row + t * (factors[nxt] - f_row)
    above = (g_mag > centre[row]) & (g_mag < hi[row]) & has_next
    out = torch.where(above.unsqueeze(-1), interp, f_row)
    out = torch.where((g_mag > hi[-1]).unsqueeze(-1), factors[-1], out)
    out = torch.where((g_mag < lo[0]).unsqueeze(-1), factors[0], out)
    return torch.where(torch.isfinite(g_mag).unsqueeze(-1), out, torch.nan)


def generate(
    xp,
    systems_: str | list[str],
    *,
    error_correction: bool = False,
    truncation: bool | None = None,
) -> dict[str, dict]:
    """Synthetic photometry for ``xp`` in each system.

    Returns ``{system: {"bands", "flux", "flux_error", "mag"}}`` with ``[B, nb]``
    tensors on ``xp``'s device. ``truncation`` overrides ``xp.truncation``.
    """
    if truncation is not None:
        xp = xp.truncated(truncation)
    names = [systems_] if isinstance(systems_, str) else list(systems_)
    meta = _assets.systems_meta()
    results: dict[str, dict] = {}
    for name in names:
        _, flux, error = _raw(xp, name)
        info = meta[name]
        kw = {"device": xp.device, "dtype": xp.dtype}
        zp = _assets.tensor("systems", f"{name}/zero_points", **kw)
        if info["standardised"]:
            off = _assets.tensor("systems", f"{name}/offsets", **kw)
            if error is not None:
                error = error * ((flux + off) / flux).abs()
            flux = flux + off
        result = {
            "bands": list(info["bands"]),
            "flux": flux,
            "flux_error": error,
            "mag": _mag(flux, zp),
        }
        eq = meta["_colour_equations"].get(info["label"])
        if eq is not None:
            zp_map = dict(zip(info["bands"], zp.tolist()))
            _colour_equation(result, info["label"], eq, zp_map)
        results[name] = result
    if error_correction and xp.errors is not None:
        g = results.get("Gaia_DR3_Vega") or generate(xp, "Gaia_DR3_Vega")["Gaia_DR3_Vega"]
        g_mag = g["mag"][..., g["bands"].index("G")]
        for name, result in results.items():
            label = meta[name]["label"]
            if label not in meta["_correction_bands"]:
                continue  # GaiaXPy warns and skips systems without a table
            factors = correction_factors(label, g_mag)
            if factors.shape[-1] != result["flux_error"].shape[-1]:
                raise ValueError(f"{name}: correction table and bands disagree")
            result["flux_error"] = result["flux_error"] * factors
    return results
