"""Continuous algebra, projection, devices, dtypes, and bug regression tests."""

from __future__ import annotations

import math

import pytest
import torch
from conftest import DEVICES, drop_bands, make_batch, planck

import torchgaiaxp as tg
from torchgaiaxp import Constant, Sampled, XPDesign, XPSpectra


def test_xp_is_continuous_and_matches_calibrate(batch):
    xp = XPSpectra.from_batch(batch)
    wl = torch.linspace(400, 900, 51, dtype=torch.float64)
    assert torch.allclose(xp(wl), xp.calibrate(wl)["flux"])


def test_per_source_grids(batch):
    xp = XPSpectra.from_batch(batch)
    wl = torch.linspace(400, 900, 51, dtype=torch.float64)
    grids = wl * torch.linspace(1.0, 1.1, len(xp), dtype=torch.float64).unsqueeze(-1)
    flux = xp(grids)
    for i in range(len(xp)):
        assert torch.allclose(flux[i], xp[i](grids[i])[0])


def test_algebra_and_self_correlation(batch):
    xp = XPSpectra.from_batch(batch)
    wl = torch.linspace(450, 850, 41, dtype=torch.float64)
    hi_wl = torch.linspace(300, 1100, 40001, dtype=torch.float64)
    trans = Sampled(hi_wl, 0.5 + 0.4 * torch.sin(hi_wl / 3.0))  # R >> XP
    expr = 2.0 * xp * trans + Constant(torch.ones(len(xp))) - xp / 4
    ev = expr.evaluate(wl)
    ref = xp.evaluate(wl)
    t = trans(wl)
    assert torch.allclose(ev.flux, 2 * ref.flux * t + 1 - ref.flux / 4)
    # Same XP factor appears twice (correlated): factor stays rank-110 and ev.error is exact.
    assert ev.factor.shape == ref.factor.shape
    jac = (2 * t - 0.25).unsqueeze(-1)
    combined = (jac * ref.factor).square().sum(-1).sqrt()
    assert torch.allclose(ev.error, combined)

    # (xp - xp) and (xp / xp) cancel correlated uncertainty to 0.
    ev_diff = (xp - xp).evaluate(wl)
    assert torch.allclose(ev_diff.flux, torch.zeros_like(ev_diff.flux))
    assert float(ev_diff.error.abs().max()) == 0.0
    ev_ratio = (xp / xp).evaluate(wl)
    assert torch.allclose(ev_ratio.error, torch.zeros_like(ev_ratio.error), atol=1e-14)


def test_sampled_errors_and_smoothing():
    wl = torch.linspace(500, 600, 10001, dtype=torch.float64)
    flux = torch.ones_like(wl)
    flux[(wl > 549.95) & (wl < 550.05)] = 0.0  # unresolved line
    s = Sampled(wl, flux, 0.01 * torch.ones_like(wl))
    low = s.smooth(1000.0)
    depth = 1 - low(torch.tensor([550.0], dtype=torch.float64))
    assert 0 < float(depth) < 0.3  # diluted by a 0.55 nm FWHM LSF
    eq_width = torch.trapezoid(1 - torch.nan_to_num(low.flux, nan=1.0), low.wavelength)
    eq_in = torch.trapezoid(1 - flux, wl)
    assert math.isclose(float(eq_width), float(eq_in), rel_tol=0.02)  # EW conserved
    assert float(low.error[torch.isfinite(low.error)].max()) < 0.01

    with pytest.raises(ValueError):
        Sampled(torch.tensor([600.0, 500.0]), torch.ones(2)).smooth(100.0)


def test_integrate_against_passband_and_missing_band(batch):
    xp = XPSpectra.from_batch(batch)
    wl = torch.linspace(500, 700, 401, dtype=torch.float64)
    box = Sampled(wl, torch.ones_like(wl))
    flux, err = xp.integrate(box, wl, photon=False)
    ref = xp(wl)
    assert torch.allclose(flux, torch.trapezoid(ref, wl) / 200.0)
    assert (err > 0).all()

    # Missing RP band integrated over 700-900 nm must return NaN (not 0.0).
    xp_miss = XPSpectra.from_batch(drop_bands(make_batch(4, realistic=False)))
    wl_red = torch.linspace(700, 900, 51, dtype=torch.float64)
    box_red = Sampled(wl_red, torch.ones_like(wl_red))
    f_red, e_red = xp_miss.integrate(box_red, wl_red, photon=False)
    assert torch.isnan(f_red[1]) and torch.isnan(e_red[1])
    assert torch.isfinite(f_red[0]) and torch.isfinite(e_red[0])


def test_indexing_band_scalar_dispersion_and_autograd():
    batch = drop_bands(make_batch(4, realistic=False))
    xp = XPSpectra.from_batch(batch)

    # Negative indexing & bounds check
    last = xp[-1]
    assert len(last) == 1
    assert torch.allclose(last.coefficients[0], xp.coefficients[-1])
    with pytest.raises(IndexError):
        _ = xp[10]
    with pytest.raises(IndexError):
        _ = xp[-10]

    # band() case-insensitivity & missing band -> NaN
    wl = torch.linspace(700, 900, 21, dtype=torch.float64)
    assert torch.allclose(xp.band("BP", wl), xp.band("bp", wl), equal_nan=True)
    assert torch.isnan(xp.band("rp", wl)[1]).all()
    with pytest.raises(ValueError):
        xp.band("invalid", wl)

    # 0D scalar wavelength
    f0 = xp(torch.tensor(550.0, dtype=torch.float64))
    assert f0.shape == (len(xp),)
    assert torch.allclose(f0, xp(torch.tensor([550.0], dtype=torch.float64)).squeeze(-1), equal_nan=True)

    # Dispersion preserves dtype/device & validates band
    wl32 = torch.tensor([450.0, 600.0], dtype=torch.float32)
    pwl32 = tg.wl_to_pwl("bp", wl32)
    assert pwl32.dtype == torch.float32
    assert tg.pwl_to_wl("bp", pwl32).dtype == torch.float32
    with pytest.raises(ValueError):
        tg.wl_range("bad")

    # float32 _mag autograd stays finite even when another source in batch has negative flux
    xp32 = XPSpectra.from_batch(make_batch(3, realistic=False), dtype=torch.float32)
    xp32.coefficients.requires_grad_(True)
    xp32.coefficients.data[0] = -xp32.coefficients.data[0].abs()
    res = xp32.generate("DR3_SDSS")["DR3_SDSS"]
    torch.nan_to_num(res["mag"][1:]).sum().backward()
    assert torch.isfinite(xp32.coefficients.grad).all()


def test_fit_recovers_xp_coefficients():
    truth = XPSpectra.from_batch(make_batch(n=3))
    wl = torch.linspace(330, 1050, 721, dtype=torch.float64)
    fitted = XPSpectra.fit(truth, wl, rcond=1e-12)
    assert (fitted(wl) - truth(wl)).abs().max() < 1e-6 * truth(wl).abs().max()


def test_fit_of_smooth_sed_is_close():
    sed = planck(5800.0)
    wl = torch.linspace(340, 1000, 331, dtype=torch.float64)
    xp = XPSpectra.fit(sed)
    rel = (xp(wl) - sed(wl)).abs().max() / sed(wl).max()
    assert rel < 1e-2


def test_shift_per_source(batch):
    xp = XPSpectra.from_batch(batch)
    z = torch.linspace(0, 0.1, len(xp), dtype=torch.float64)
    wl = torch.linspace(450, 850, 21, dtype=torch.float64)
    shifted = xp.shift(z)(wl)
    assert torch.allclose(shifted, xp(wl / (1 + z).unsqueeze(-1)))


def test_gradients_flow_to_coefficients_and_wavelength(batch):
    xp = XPSpectra.from_batch(batch)
    xp.coefficients.requires_grad_(True)
    wl = torch.linspace(450, 850, 21, dtype=torch.float64, requires_grad=True)
    xp(wl).sum().backward()
    assert torch.isfinite(xp.coefficients.grad).all()
    assert torch.isfinite(wl.grad).all() and (wl.grad != 0).any()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_devices_and_dtypes_are_honoured(batch, device, dtype):
    if device == "mps" and dtype == torch.float64:
        pytest.skip("MPS has no float64")
    ref = XPSpectra.from_batch(batch)
    xp = XPSpectra.from_batch(batch, device=device, dtype=dtype)
    wl = torch.linspace(400, 900, 51, device=device, dtype=dtype)
    outputs = [
        xp(wl),
        xp.calibrate()["flux"],
        xp.calibrate(wl)["flux_error"],
        xp.convert()["flux"],
        xp.generate("DR3_SDSS_Std")["DR3_SDSS_Std"]["flux"],
        (xp * Sampled(wl, torch.ones_like(wl))).evaluate(wl).factor,
        XPDesign.absolute(wl).design,
        tg.wl_to_pwl("bp", wl),
    ]
    for out in outputs:
        assert out.device.type == torch.device(device).type and out.dtype == dtype
    tol = 1e-3 if dtype == torch.float32 else 1e-10
    want = ref(wl.cpu().double())
    got = xp(wl).cpu().double()
    assert torch.allclose(got, want, rtol=tol, atol=tol * want.abs().max())
    moved = ref.to(device, dtype)
    assert moved.coefficients.device.type == torch.device(device).type
    assert moved.n_relevant.dtype == torch.int64
