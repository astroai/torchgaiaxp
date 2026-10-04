"""Parity with GaiaXPy 2.1 (skipped when GaiaXPy is not installed)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import drop_bands, frame, make_batch

import torchgaiaxp as tg
from torchgaiaxp import XPDesign, XPSpectra

gaiaxpy = pytest.importorskip("gaiaxpy")
from gaiaxpy.calibrator import calibrator as _cal


def _xp(batch, truncation=False):
    return XPSpectra.from_batch(batch, truncation=truncation)


@pytest.mark.parametrize("sampling", [np.linspace(320, 1050, 300), np.geomspace(400, 900, 77)])
def test_native_design_matches_gaiaxpy(sampling):
    build = getattr(_cal, "__generate_xp_matrices_and_merge")
    matrices, merge = build("calibrator", sampling, "v375wi", "v142r")
    ours = XPDesign.absolute(sampling)
    for i, b in enumerate(("bp", "rp")):
        ref = matrices[b].get_design_matrix()
        np.testing.assert_allclose(
            ours.design[i].T.numpy(), ref, rtol=0, atol=1e-12 * abs(ref).max()
        )
        np.testing.assert_allclose(ours.merge[i].numpy(), merge[b])


@pytest.mark.parametrize("truncation", [False, True])
@pytest.mark.parametrize("sampling", [None, np.linspace(400, 900, 101)])
def test_calibrate(truncation, sampling):
    batch = drop_bands(make_batch(realistic=False))
    ref, grid = gaiaxpy.calibrate(
        frame(batch), sampling=sampling, truncation=truncation, save_file=False
    )
    out = _xp(batch).truncated(truncation).calibrate(sampling)
    np.testing.assert_allclose(out["wavelength"].numpy(), grid)
    for i in range(len(batch)):
        f, e = np.asarray(ref["flux"][i]), np.asarray(ref["flux_error"][i])
        np.testing.assert_allclose(
            out["flux"][i].numpy(), f, rtol=1e-9, atol=1e-9 * np.nanmax(abs(f))
        )
        np.testing.assert_allclose(
            out["flux_error"][i].numpy(), e, rtol=1e-6, atol=1e-9 * np.nanmax(e)
        )


@pytest.mark.parametrize("truncation", [False, True])
def test_convert(truncation):
    batch = drop_bands(make_batch(realistic=False))
    sampling = np.linspace(-5, 65, 141)
    ref, grid = gaiaxpy.convert(
        frame(batch), sampling=sampling, truncation=truncation, save_file=False
    )
    out = _xp(batch).truncated(truncation).convert(sampling)
    np.testing.assert_allclose(out["pseudo_wavelength"].numpy(), grid)
    for _, row in ref.iterrows():
        i = int(np.where(batch.source_id == row["source_id"])[0][0])
        k = 0 if row["xp"].lower() == "bp" else 1
        if not isinstance(row["flux"], np.ndarray):  # GaiaXPy: missing band
            assert torch.isnan(out["flux"][i, k]).all()
            continue
        np.testing.assert_allclose(out["flux"][i, k].numpy(), row["flux"], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(out["flux_error"][i, k].numpy(), row["flux_error"], rtol=1e-6)


@pytest.mark.parametrize("system", tg.systems())
def test_generate_every_system(system):
    batch = make_batch()
    ps = getattr(gaiaxpy.PhotometricSystem, system)
    ref = gaiaxpy.generate(frame(batch), photometric_system=ps, save_file=False)
    out = _xp(batch).generate(system)[system]
    label = ps.get_system_label()
    for j, band in enumerate(out["bands"]):
        for key, col in (("flux", "flux"), ("flux_error", "flux_error"), ("mag", "mag")):
            want = ref[f"{label}_{col}_{band}"].to_numpy(np.float64)
            np.testing.assert_allclose(
                out[key][:, j].numpy(),
                want,
                rtol=1e-7,
                atol=1e-12 * np.nanmax(abs(want)),
                err_msg=f"{band} {key}",
            )


@pytest.mark.parametrize(
    "system", ["JKC_Std", "SDSS_Std", "SDSS", "Gaia_DR3_Vega", "PanSTARRS1_Std", "Stromgren_Std"]
)
def test_generate_error_correction(system):
    batch = make_batch(n=12, seed=3)
    ps = getattr(gaiaxpy.PhotometricSystem, system)
    # Spread G over the table so every branch (clamp, centre, interpolation) is hit.
    scale = 10 ** (-0.4 * np.linspace(-8, 10, 12))
    for name in ("coefficients", "errors"):
        getattr(batch, name)[:] *= scale[:, None, None]
    ref = gaiaxpy.generate(
        frame(batch), photometric_system=ps, save_file=False, error_correction=True
    )
    out = _xp(batch).generate(system, error_correction=True)[system]
    label = ps.get_system_label()
    for j, band in enumerate(out["bands"]):
        want = ref[f"{label}_flux_error_{band}"].to_numpy(np.float64)
        np.testing.assert_allclose(out["flux_error"][:, j].numpy(), want, rtol=1e-7, err_msg=band)


def test_missing_band_photometry_is_nan():
    batch = drop_bands(make_batch())
    out = _xp(batch).generate("SDSS")["SDSS"]
    assert torch.isnan(out["flux"][1:3]).all()
    assert torch.isfinite(out["flux"][0]).all()


def test_cholesky_tools():
    from gaiaxpy import (
        get_chi2,
        get_inverse_covariance_matrix,
        get_inverse_square_root_covariance_matrix,
    )

    batch = make_batch(realistic=False)
    df = frame(batch)
    ref_isr = get_inverse_square_root_covariance_matrix(df, band="bp")
    ref_inv = get_inverse_covariance_matrix(df, band="rp")
    t = {
        k: torch.as_tensor(getattr(batch, k))
        for k in ("errors", "correlations", "standard_deviation")
    }
    isr = tg.inverse_square_root_covariance(
        t["errors"][:, 0], t["correlations"][:, 0], t["standard_deviation"][:, 0]
    )
    inv = tg.inverse_covariance(
        t["errors"][:, 1], t["correlations"][:, 1], t["standard_deviation"][:, 1]
    )
    for i in range(len(batch)):
        np.testing.assert_allclose(isr[i].numpy(), ref_isr.iloc[i, 1], rtol=1e-8)
        np.testing.assert_allclose(inv[i].numpy(), ref_inv.iloc[i, 1], rtol=1e-7)
        r = np.random.default_rng(i).normal(size=55)
        assert np.isclose(
            float(tg.chi2(isr[i], torch.as_tensor(r))), get_chi2(ref_isr.iloc[i, 1], r)
        )


def test_dispersion_conversions():
    from gaiaxpy import pwl_range, pwl_to_wl, wl_range, wl_to_pwl

    for band in ("bp", "rp"):
        x = np.linspace(-2, 62, 50)
        np.testing.assert_allclose(tg.pwl_to_wl(band, x).numpy(), pwl_to_wl(band, x), rtol=1e-12)
        wl = np.linspace(300, 1100, 50)
        np.testing.assert_allclose(tg.wl_to_pwl(band, wl).numpy(), wl_to_pwl(band, wl), rtol=1e-12)
        np.testing.assert_allclose(tg.pwl_range(band), pwl_range(band), rtol=1e-12)
        np.testing.assert_allclose(tg.wl_range(band), wl_range(band))
