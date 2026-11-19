"""Parity with GaiaXPy 2.2+ (skipped when GaiaXPy is not installed)."""

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
    cal_disp = gaiaxpy.core.config_dispatcher.CalibratorDispatcher()
    cfg = cal_disp.build(sampling=sampling, release="dr3", basis_type="basic")
    matrices, merge, _ = build(sampling, cfg, release="dr3")
    ours = XPDesign.absolute(sampling, release="dr3")
    for i, b in enumerate(("bp", "rp")):
        ref = matrices[b].get_design_matrix()
        np.testing.assert_allclose(
            ours.design[i].T.numpy(), ref, rtol=0, atol=1e-12 * abs(ref).max()
        )
        np.testing.assert_allclose(ours.merge[i].numpy(), merge[b])


@pytest.mark.parametrize("truncation", [False, True])
@pytest.mark.parametrize("sampling", [None, np.linspace(400, 900, 101)])
@pytest.mark.parametrize("with_correlation", [False, True])
def test_calibrate(truncation, sampling, with_correlation):
    batch = drop_bands(make_batch(realistic=False))
    ref, grid = gaiaxpy.calibrate(
        frame(batch),
        sampling=sampling,
        truncation=truncation,
        with_correlation=with_correlation,
        save_file=False,
        disable_info=True,
    )
    out = _xp(batch).truncated(truncation).calibrate(sampling, with_correlation=with_correlation)
    np.testing.assert_allclose(out["wavelength"].numpy(), grid)
    for i in range(len(batch)):
        f, e = np.asarray(ref["flux"][i]), np.asarray(ref["flux_error"][i])
        np.testing.assert_allclose(
            out["flux"][i].numpy(), f, rtol=1e-9, atol=1e-9 * np.nanmax(abs(f))
        )
        np.testing.assert_allclose(
            out["flux_error"][i].numpy(), e, rtol=1e-6, atol=1e-9 * np.nanmax(e)
        )
        if with_correlation:
            corr_ref = np.asarray(ref["correlation"][i], dtype=np.float64)
            corr_ours = tg.pack_correlations(out["covariance"][i]).numpy()
            np.testing.assert_allclose(
                corr_ours, corr_ref, rtol=1e-5, atol=1e-6, equal_nan=True
            )


@pytest.mark.parametrize("truncation", [False, True])
def test_convert(truncation):
    batch = drop_bands(make_batch(realistic=False))
    sampling = np.linspace(-5, 65, 141)
    ref, grid = gaiaxpy.convert(
        frame(batch),
        sampling=sampling,
        truncation=truncation,
        with_correlation=True,
        save_file=False,
        disable_info=True,
    )
    out = _xp(batch).truncated(truncation).convert(sampling, with_correlation=True)
    np.testing.assert_allclose(out["pseudo_wavelength"].numpy(), grid)
    for _, row in ref.iterrows():
        i = int(np.where(batch.source_id == row["source_id"])[0][0])
        k = 0 if row["xp"].lower() == "bp" else 1
        if not isinstance(row["flux"], np.ndarray):  # GaiaXPy: missing band
            assert torch.isnan(out["flux"][i, k]).all()
            continue
        np.testing.assert_allclose(out["flux"][i, k].numpy(), row["flux"], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(out["flux_error"][i, k].numpy(), row["flux_error"], rtol=1e-6)
        np.testing.assert_allclose(
            tg.pack_correlations(out["covariance"][i, k]).numpy(),
            row["correlation"],
            rtol=1e-5,
            atol=1e-6,
        )


@pytest.mark.parametrize("system", tg.systems(release="dr3"))
def test_generate_every_system(system):
    batch = make_batch()
    ps = getattr(gaiaxpy.PhotometricSystem, system)
    ref = gaiaxpy.generate(frame(batch), photometric_system=ps, save_file=False, disable_info=True)
    out = _xp(batch).generate(system)[system]
    label = ps.get_system_full_label()
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
    "system,legacy",
    [
        ("DR3_JKC_Std", "JKC_Std"),
        ("DR3_SDSS_Std", "SDSS_Std"),
        ("DR3_SDSS", "SDSS"),
        ("DR3_Gaia_Vega", "Gaia_DR3_Vega"),
        ("DR3_PanSTARRS1_Std", "PanSTARRS1_Std"),
        ("DR3_Stromgren_Std", "Stromgren_Std"),
    ],
)
def test_generate_error_correction_and_aliases(system, legacy):
    batch = make_batch(n=12, seed=3)
    ps = getattr(gaiaxpy.PhotometricSystem, system)
    # Spread G over the table so every branch (clamp, centre, interpolation) is hit.
    scale = 10 ** (-0.4 * np.linspace(-8, 10, 12))
    for name in ("coefficients", "errors"):
        getattr(batch, name)[:] *= scale[:, None, None]
    ref = gaiaxpy.generate(
        frame(batch),
        photometric_system=ps,
        save_file=False,
        error_correction=True,
        disable_info=True,
    )
    xp = _xp(batch)
    out = xp.generate(system, error_correction=True)[system]
    out_alias = xp.generate(legacy, error_correction=True)[legacy]
    label = ps.get_system_full_label()
    for j, band in enumerate(out["bands"]):
        want = ref[f"{label}_flux_error_{band}"].to_numpy(np.float64)
        np.testing.assert_allclose(out["flux_error"][:, j].numpy(), want, rtol=1e-7, err_msg=band)
        np.testing.assert_allclose(
            out_alias["flux_error"][:, j].numpy(), want, rtol=1e-7, err_msg=band
        )


def test_missing_band_photometry_is_nan():
    batch = drop_bands(make_batch())
    out = _xp(batch).generate("DR3_SDSS")["DR3_SDSS"]
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
    xp = _xp(batch)
    isr = xp.inverse_square_root_covariance("bp")
    inv = xp.inverse_covariance("rp")
    for i in range(len(batch)):
        np.testing.assert_allclose(isr[i].numpy(), ref_isr.iloc[i, 1], rtol=1e-8)
        np.testing.assert_allclose(inv[i].numpy(), ref_inv.iloc[i, 1], rtol=1e-7)
        r = np.random.default_rng(i).normal(size=55)
        assert np.isclose(
            float(xp.chi2("bp", torch.as_tensor(r))[i]), get_chi2(ref_isr.iloc[i, 1], r)
        )


def test_dispersion_conversions():
    from gaiaxpy import pwl_range, pwl_to_wl, wl_range, wl_to_pwl

    for band in ("bp", "rp"):
        x = np.linspace(-2, 62, 50)
        np.testing.assert_allclose(
            tg.pwl_to_wl(band, x, release="dr3").numpy(), pwl_to_wl(band, x, "dr3"), rtol=1e-12
        )
        wl = np.linspace(300, 1100, 50)
        np.testing.assert_allclose(
            tg.wl_to_pwl(band, wl, release="dr3").numpy(), wl_to_pwl(band, wl, "dr3"), rtol=1e-12
        )
        np.testing.assert_allclose(
            tg.pwl_range(band, release="dr3"), pwl_range(band, "dr3"), rtol=1e-12
        )
        ref_range = wl_range(band, "dr3")
        np.testing.assert_allclose(tg.wl_range(band, release="dr3"), [ref_range.low, ref_range.high])
