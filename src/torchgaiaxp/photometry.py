"""Synthetic photometry: GaiaXPy ``generate``, colour equations, and error correction.

For each system: band fluxes ``sum_b w_b D_b c_b`` (W m^-2 Hz^-1 for AB-like
systems, W m^-2 nm^-1 for Vega-like ones, as GaiaXPy), errors from the
coefficient covariance, magnitudes ``-2.5 log10(f) + zp`` (NaN for f <= 0).
Then, as GaiaXPy does, in order:

1. standardised systems (``*_Std``): flux + offset, error x |(f + off) / f|;
2. colour equations (e.g. DR3_JKC_Std U from B-V, DR3_SDSS_Std u from g-i):
   polynomial inside the colour range, linear extrapolation outside, error
   propagated through the polynomial's derivative;
3. optional error correction (Montegriffo et al. 2023): flux errors scaled by
   factors tabulated in Gaia Vega G.

Photometry needs both bands: a source with one band missing is NaN, as in
GaiaXPy.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import torch
from torch import Tensor

from . import _assets
from .design import XPDesign

LN10 = math.log(10.0)
_USER_SYSTEMS: dict[str, tuple[XPDesign, dict]] = {}
_FILTER_NAME_RE = re.compile(
    r"(?P<name>[a-zA-Z0-9_-]+)\.gaiaxpy_(?P<release>dr\d+)_(?P<version>[a-zA-Z0-9-]+)\.(?:xml|ya?ml)",
    re.IGNORECASE,
)


def systems(release: str | None = None) -> list[str]:
    """Available photometric system names (GaiaXPy ``PhotometricSystem``)."""
    releases = (_assets.parse_release(release),) if release is not None else _assets.available_releases()
    names = [
        k
        for rel in releases
        for k in _assets.systems_meta(rel)
        if not k.startswith("_")
    ]
    names.extend(_USER_SYSTEMS.keys())
    return sorted(names)


def load_additional_systems(
    path: str | Path,
    *,
    release: str = _assets.DEFAULT_RELEASE,
) -> list[str]:
    """Load custom GaiaXPy filter file(s) (XML or YAML) into the system registry."""
    rel = _assets.parse_release(release)
    p = Path(path)
    files = [p] if p.is_file() else sorted(
        f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in (".xml", ".yaml", ".yml")
    )
    if not files:
        raise ValueError(f"No filter files found at {p}")
    loaded = []
    for f in files:
        design, info = XPDesign.from_filter(f, release=rel)
        m = _FILTER_NAME_RE.fullmatch(f.name)
        if m:
            key = f"USER_{m.group('release').upper()}_{m.group('name')}_{m.group('version')}"
            info["standardised"] = m.group("name").endswith("Std") or info["standardised"]
            info["release"] = m.group("release").lower()
        else:
            key = f.stem
            info["standardised"] = key.endswith("Std") or info["standardised"]
            info["release"] = rel
        info["full_label"] = key
        info["label"] = key
        design.system = key
        _USER_SYSTEMS[key] = (design, info)
        loaded.append(key)
    return loaded


def remove_additional_systems() -> None:
    """Clear all user-loaded additional photometric systems."""
    _USER_SYSTEMS.clear()


def _resolve_system(name: str, release: str | None):
    if name in _USER_SYSTEMS:
        design, info = _USER_SYSTEMS[name]
        return info["release"], name, design, info, True
    rel, canonical = _assets.resolve_system_name(name, release=release)
    info = _assets.systems_meta(rel)[canonical]
    return rel, canonical, None, info, False


def _raw(xp, design: XPDesign, packed_cov: Tensor | None):
    des = design.to(xp.device, xp.dtype)
    flux_b = xp._band_fluxes(des)
    w = des.merge.unsqueeze(0)  # [1, 2, nb]
    flux = (w * flux_b).sum(1)
    error = None
    var_b = xp._band_variances(des, packed_cov=packed_cov)
    if var_b is not None:
        # GaiaXPy: sqrt(w_bp^2 e_bp^2 + w_rp^2 e_rp^2) with per-band diagonal errors.
        error = (w.square() * var_b).sum(1).clamp_min(0.0).sqrt()
    both = xp.present.all(-1, keepdim=True)
    flux = torch.where(both, flux, torch.nan)
    if error is not None:
        error = torch.where(both, error, torch.nan)
    return flux, error


def _mag(flux: Tensor, zp: Tensor) -> Tensor:
    tiny = torch.finfo(flux.dtype).tiny
    return torch.where(flux > 0, -2.5 * torch.log10(flux.clamp_min(tiny)) + zp, torch.nan)


def _poly(coeffs: list[float], x: Tensor) -> Tensor:
    out = torch.zeros_like(x)
    for c in coeffs:  # highest power first
        out = out * x + c
    return out


def _poly_deriv(coeffs: list[float], x: Tensor) -> Tensor:
    n = len(coeffs) - 1
    return _poly([c * (n - i) for i, c in enumerate(coeffs[:-1])], x)


def _colour_equation(result: dict, spec: dict, zp_map: dict[str, float]):
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
    out_flux = 10.0 ** (-0.4 * (corrected - zp_map[spec["filter"]]))
    result["mag"] = _set(mag, target, corrected)
    result["flux"] = _set(flux, target, out_flux)
    if err is not None:

        def mag_err(i):
            f_i = flux[..., i]
            safe = torch.where(f_i != 0, f_i, torch.ones_like(f_i))
            return torch.where(f_i != 0, 2.5 * err[..., i] / (safe * LN10), torch.nan)

        colour_err = torch.sqrt(mag_err(i0) ** 2 + mag_err(i1) ** 2)
        # GaiaXPy evaluates the derivative at the colour itself, even outside the range.
        corr_err = colour_err * _poly_deriv(coeffs, colour).abs()
        total = torch.sqrt(mag_err(target) ** 2 + corr_err**2)
        result["flux_error"] = _set(err, target, total * out_flux * LN10 / 2.5)


def _set(x: Tensor, index: int, value: Tensor) -> Tensor:
    x = x.clone()
    x[..., index] = value
    return x


def correction_factors(
    label: str,
    g_mag: Tensor,
    *,
    release: str = _assets.DEFAULT_RELEASE,
) -> Tensor:
    """Error-correction factors ``[B, nb]`` for a system label at Gaia G (Vega)."""
    rel = _assets.parse_release(release)
    kw = {"device": g_mag.device, "dtype": g_mag.dtype, "release": rel}
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
    out = torch.where((g_mag >= hi[-1]).unsqueeze(-1), factors[-1], out)
    out = torch.where((g_mag < lo[0]).unsqueeze(-1), factors[0], out)
    return torch.where(torch.isfinite(g_mag).unsqueeze(-1), out, torch.nan)


def _maybe_apply_colour_equation(result: dict, s_rel: str, canonical: str, info: dict) -> None:
    meta = _assets.systems_meta(s_rel)
    full_label = info.get("full_label", info["label"])
    eq = meta.get("_colour_equations", {}).get(full_label)
    if eq is not None:
        zp_arr = _assets.system_array(f"{canonical}/zero_points", s_rel)
        zp_map = dict(zip(info["bands"], zp_arr.tolist()))
        _colour_equation(result, eq, zp_map)


def _scale_error_correction(
    name: str,
    result: dict,
    g_mag: Tensor,
    default_rel: str,
) -> None:
    if result.get("flux_error") is None:
        return
    s_rel, _, _, info, is_user = _resolve_system(name, default_rel)
    if is_user:
        return
    s_meta = _assets.systems_meta(s_rel)
    full_label = info.get("full_label", info["label"])
    if full_label not in s_meta.get("_correction_bands", {}):
        return
    factors = correction_factors(full_label, g_mag, release=s_rel)
    if factors.shape[-1] != result["flux_error"].shape[-1]:
        raise ValueError(f"{name}: correction table and bands disagree")
    result["flux_error"] = result["flux_error"] * factors


def apply_colour_equation(
    results: dict[str, dict],
    *,
    release: str | None = None,
) -> dict[str, dict]:
    """Apply configured colour equations to a ``generate()`` output dict (returns updated copy)."""
    out: dict[str, dict] = {}
    for name, res in results.items():
        r_copy = dict(res)
        rel, canonical, _, info, is_user = _resolve_system(name, release)
        if not is_user:
            _maybe_apply_colour_equation(r_copy, rel, canonical, info)
        out[name] = r_copy
    return out


def apply_error_correction(
    results: dict[str, dict],
    g_mag: Tensor | None = None,
    *,
    release: str | None = None,
) -> dict[str, dict]:
    """Apply tabulated G-magnitude error correction factors to a ``generate()`` output dict."""
    if release is None:
        rel = next(
            (s_rel for name in results for s_rel, _, _, _, is_user in [_resolve_system(name, None)] if not is_user),
            _assets.DEFAULT_RELEASE,
        )
    else:
        rel = _assets.parse_release(release)
    meta = _assets.systems_meta(rel)
    gaia_sys = meta.get("_gaia_vega_system")
    if g_mag is None:
        if gaia_sys is None:
            raise ValueError(f"No Gaia Vega reference system configured for release {rel!r}")
        for name, res in results.items():
            _, canonical, _, _, is_user = _resolve_system(name, rel)
            if not is_user and canonical == gaia_sys:
                g_mag = res["mag"][..., res["bands"].index("G")]
                break
        if g_mag is None:
            raise ValueError(
                f"g_mag is required when {gaia_sys!r} is not present in photometry results"
            )
    out: dict[str, dict] = {}
    for name, res in results.items():
        r_copy = dict(res)
        _scale_error_correction(name, r_copy, g_mag, rel)
        out[name] = r_copy
    return out


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
    if not names:
        raise ValueError("at least one photometric system is required")

    rel_default = getattr(xp, "release", _assets.DEFAULT_RELEASE)
    # Unpack/pack covariance once across all requested systems (+ Gaia Vega if needed).
    packed_cov = xp._packed_covariance() if (xp.errors is not None and xp.correlations is not None) else None
    kw = {"device": xp.device, "dtype": xp.dtype}
    results: dict[str, dict] = {}
    canonical_to_key: dict[str, str] = {}

    for name in names:
        rel, canonical, user_design, info, is_user = _resolve_system(name, rel_default)
        canonical_to_key[canonical] = name
        if is_user:
            design = user_design
            zp = torch.as_tensor(info["zero_points"], device=xp.device, dtype=xp.dtype)
            off = torch.as_tensor(info["offsets"], device=xp.device, dtype=xp.dtype)
        else:
            design = XPDesign.photometric(canonical, xp.device, xp.dtype, release=rel)
            zp = _assets.tensor("systems", f"{canonical}/zero_points", release=rel, **kw)
            off = _assets.tensor("systems", f"{canonical}/offsets", release=rel, **kw)

        flux, error = _raw(xp, design, packed_cov)
        if info["standardised"]:
            if error is not None:
                safe_f = torch.where(flux != 0, flux, torch.ones_like(flux))
                error = torch.where(flux != 0, error * ((flux + off) / safe_f).abs(), torch.nan)
            flux = flux + off
        result = {
            "bands": list(info["bands"]),
            "flux": flux,
            "flux_error": error,
            "mag": _mag(flux, zp),
        }
        if not is_user:
            _maybe_apply_colour_equation(result, rel, canonical, info)
        results[name] = result

    if error_correction and xp.errors is not None:
        meta = _assets.systems_meta(rel_default)
        gaia_sys = meta.get("_gaia_vega_system")
        if gaia_sys is not None:
            if gaia_sys in canonical_to_key:
                g_res = results[canonical_to_key[gaia_sys]]
            else:
                g_des = XPDesign.photometric(gaia_sys, xp.device, xp.dtype, release=rel_default)
                g_zp = _assets.tensor("systems", f"{gaia_sys}/zero_points", release=rel_default, **kw)
                g_flux, _ = _raw(xp, g_des, packed_cov)
                g_res = {"bands": list(meta[gaia_sys]["bands"]), "mag": _mag(g_flux, g_zp)}
            g_mag = g_res["mag"][..., g_res["bands"].index("G")]
            for name, result in results.items():
                _scale_error_correction(name, result, g_mag, rel_default)
    return results
