"""Gaia DR4 readiness, custom systems, standalone corrections, and epoch spectra tests."""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from conftest import make_batch

import torchgaiaxp as tg
from torchgaiaxp import (
    XPDesign,
    XPEpochSpectra,
    XPSpectra,
    apply_colour_equation,
    apply_error_correction,
    load_additional_systems,
    register_release,
    remove_additional_systems,
)
from torchgaiaxp.io import XPBatch


def test_custom_filter_xml_and_additional_systems(tmp_path):
    # Build a synthetic 2-band XML filter from DR3 JKC U/B design slices
    base = XPDesign.photometric("DR3_JKC")
    bp_vals = base.design[0, :2].reshape(-1).tolist()
    rp_vals = base.design[1, :2].reshape(-1).tolist()
    bp_w = base.merge[0, :2].tolist()
    rp_w = base.merge[1, :2].tolist()
    xml = f"""<filter>
      <bands><band>U</band><band>B</band></bands>
      <zeropoints><zp>25.0</zp><zp>25.5</zp></zeropoints>
      <XpMerge>
        <sampleMeanWavelengths><w>360</w><w>440</w></sampleMeanWavelengths>
        <bpWeights>{"".join(f"<w>{v}</w>" for v in bp_w)}</bpWeights>
        <rpWeights>{"".join(f"<w>{v}</w>" for v in rp_w)}</rpWeights>
      </XpMerge>
      <XpSampling>
        <bpSampledBases dimension="55">{"".join(f"<v>{v}</v>" for v in bp_vals)}</bpSampledBases>
        <rpSampledBases dimension="55">{"".join(f"<v>{v}</v>" for v in rp_vals)}</rpSampledBases>
      </XpSampling>
    </filter>"""
    fpath = tmp_path / "MySys.gaiaxpy_dr3_v1.xml"
    fpath.write_text(xml, encoding="utf-8")
    try:
        loaded = load_additional_systems(fpath)
        assert loaded == ["USER_DR3_MySys_v1"]
        xp = XPSpectra.from_batch(make_batch(2))
        out = xp.generate("USER_DR3_MySys_v1")["USER_DR3_MySys_v1"]
        assert out["bands"] == ["U", "B"]
        assert torch.isfinite(out["flux"]).all()
    finally:
        remove_additional_systems()


def test_standalone_colour_eq_and_error_correction():
    xp = XPSpectra.from_batch(make_batch(4))
    res = xp.generate(["DR3_SDSS_Std", "DR3_Gaia_Vega"], error_correction=True)
    # Standalone error correction against g_mag reproduces error_correction=True
    raw = xp.generate(["DR3_SDSS_Std", "DR3_Gaia_Vega"], error_correction=False)
    corrected = apply_error_correction(raw)
    assert torch.allclose(
        corrected["DR3_SDSS_Std"]["flux_error"], res["DR3_SDSS_Std"]["flux_error"]
    )
    # apply_colour_equation runs cleanly on a raw photometry dict
    ce = apply_colour_equation({"DR3_SDSS_Std": raw["DR3_SDSS_Std"]})
    assert "flux" in ce["DR3_SDSS_Std"]


def test_variable_n_bases_and_release_registry(tmp_path):
    # Variable N = 60 (>55) coefficients, errors, correlations, cholesky
    n_b = 60
    rng = np.random.default_rng(7)
    coeffs = rng.normal(size=(3, 2, n_b))
    errs = np.abs(rng.normal(size=(3, 2, n_b))) + 0.1
    corr = np.zeros((3, 2, n_b * (n_b - 1) // 2))
    batch = XPBatch(
        source_id=np.arange(3, dtype=np.int64),
        coefficients=coeffs,
        errors=errs,
        correlations=corr,
        standard_deviation=np.ones((3, 2)),
        n_relevant=np.full((3, 2), 40, dtype=np.int16),
        n_parameters=np.full((3, 2), n_b, dtype=np.int64),
    )
    xp = XPSpectra.from_batch(batch)
    assert xp.n_bases == 60
    isr = xp.inverse_square_root_covariance("bp")
    assert isr.shape == (3, 60, 60)

    # Register synthetic DR4 release copied from DR3 assets with distinct solution_id
    dr3_dir = tg._assets._asset_path("dr3", "instrument.json").parent
    dr4_dir = tmp_path / "dr4"
    dr4_dir.mkdir()
    for fname in ("instrument.npz", "systems.npz", "systems.json"):
        (dr4_dir / fname).write_bytes((dr3_dir / fname).read_bytes())
    meta = json.loads((dr3_dir / "instrument.json").read_text())
    meta["release"] = "dr4"
    meta["solution_ids"] = ["8999999999999999999"]
    (dr4_dir / "instrument.json").write_text(json.dumps(meta))
    register_release("dr4", dr4_dir)
    assert "dr4" in tg.available_releases()

    dr4_batch = make_batch(2, realistic=False)
    dr4_batch.solution_id = np.array([8999999999999999999, 8999999999999999999], dtype=np.int64)
    xp4 = XPSpectra.from_batch(dr4_batch)
    assert xp4.release == "dr4"
    assert torch.isfinite(xp4.calibrate()["flux"]).all()
    with pytest.raises(ValueError, match="conflicts"):
        XPSpectra.from_batch(dr4_batch, release="dr3")
    tg._assets._EXTRA_RELEASES.pop("dr4", None)


def test_epoch_spectra_workflow(tmp_path):
    wl = torch.linspace(400.0, 900.0, 11, dtype=torch.float64)
    # 5 transits across 2 sources (3 transits for sid=10, 2 for sid=20), 1 crowded transit
    sids = torch.tensor([20, 10, 10, 20, 10], dtype=torch.int64)
    times = torch.tensor([2.0, 3.0, 1.0, 1.0, 2.0], dtype=torch.float64)
    flux = torch.ones((5, 11), dtype=torch.float64)
    flux[1] = 10.0  # crowded outlier on sid=10 at t=3
    errs = torch.full((5, 11), 0.1, dtype=torch.float64)
    crowd = torch.tensor([False, True, False, False, False])

    ep = XPEpochSpectra(
        flux=flux, grid=wl, errors=errs, source_id=sids, time=times, crowding_flag=crowd
    )
    usid, f_pad, _, t_pad, valid_pad = ep.group_by_source()
    assert usid.tolist() == [10, 20]
    assert f_pad.shape == (2, 3, 11)
    assert torch.allclose(t_pad[0], torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    assert not valid_pad[0, 2].any()  # crowded transit masked

    _, mf, me = ep.mean_flux(weights="inverse_variance")
    assert torch.allclose(mf, torch.ones((2, 11), dtype=torch.float64))
    assert torch.allclose(me, torch.full((2, 11), 0.1 / np.sqrt(2), dtype=torch.float64))

    sampled = ep.to_sampled()
    assert sampled(torch.tensor([500.0], dtype=torch.float64)).shape == (2, 1)

    # Parquet round-trip
    pq_path = tmp_path / "epoch.parquet"
    pq.write_table(
        pa.table(
            {
                "source_id": pa.array(sids.numpy(), pa.int64()),
                "obs_time": pa.array(times.numpy(), pa.float64()),
                "flux": pa.array(flux.numpy().tolist(), type=pa.list_(pa.float64())),
                "flux_error": pa.array(errs.numpy().tolist(), type=pa.list_(pa.float64())),
                "crowding_flag": pa.array(crowd.numpy(), pa.bool_()),
            }
        ),
        pq_path,
    )
    ep2 = XPEpochSpectra.from_parquet(pq_path, grid=wl)
    _, mf2, _ = ep2.mean_flux()
    assert torch.allclose(mf2, mf)


def test_dr4_official_epoch_spectrum_schema(tmp_path):
    """Exact DR4 xp_epoch_spectrum columns: per-transit bp/rp_pseudo_wavelength[60],
    masked_samples, ccd_proc_flags, obs_time, and transit_id bitfield decoding."""
    n_t, n_pix = 4, 60
    sids = np.array([101, 101, 101, 102], dtype=np.int64)
    # Encode OBMT=5000*204800 ns, fov=1, ccd_row=3, af1_ac_pixel=42
    tid = (5000 << 17) | (0 << 15) | (3 << 12) | 42
    tids = np.full(n_t, tid, dtype=np.int64)
    decoded = tg.decode_transit_id(torch.tensor([tid], dtype=torch.int64))
    assert int(decoded["fov"][0]) == 1
    assert int(decoded["ccd_row"][0]) == 3
    assert int(decoded["af1_ac_pixel"][0]) == 42

    pwl_base = np.linspace(0.0, 60.0, n_pix, dtype=np.float32)
    bp_pwl = [pwl_base + 0.05 * i for i in range(n_t)]
    rp_pwl = [pwl_base - 0.03 * i for i in range(n_t)]
    bp_flux = [(g + 1.0).astype(np.float32) for g in bp_pwl]
    rp_flux = [(2.0 * g + 3.0).astype(np.float32) for g in rp_pwl]
    bp_err = [np.full(n_pix, 0.1, dtype=np.float32) for _ in range(n_t)]
    rp_err = [np.full(n_pix, 0.2, dtype=np.float32) for _ in range(n_t)]
    # Mask pixel 10 on transit 0 BP, and flag transit 2 BP with bit 8 (rejected) + bit 13 (crowding)
    bp_masked = [[10], [], [], []]
    rp_masked = [[], [], [], []]
    bp_proc = np.array([1 << 13, 0, (1 << 8) | (1 << 13), 0], dtype=np.int16)
    rp_proc = np.zeros(n_t, dtype=np.int16)

    pq_path = tmp_path / "dr4_xp_epoch.parquet"
    pq.write_table(
        pa.table(
            {
                "source_id": pa.array(sids, pa.int64()),
                "transit_id": pa.array(tids, pa.int64()),
                "bp_obs_time": pa.array(np.linspace(100.0, 103.0, n_t), pa.float64()),
                "rp_obs_time": pa.array(np.linspace(100.0, 103.0, n_t), pa.float64()),
                "bp_pseudo_wavelength": pa.array(bp_pwl, type=pa.list_(pa.float32())),
                "rp_pseudo_wavelength": pa.array(rp_pwl, type=pa.list_(pa.float32())),
                "bp_flux": pa.array(bp_flux, type=pa.list_(pa.float32())),
                "rp_flux": pa.array(rp_flux, type=pa.list_(pa.float32())),
                "bp_flux_error": pa.array(bp_err, type=pa.list_(pa.float32())),
                "rp_flux_error": pa.array(rp_err, type=pa.list_(pa.float32())),
                "bp_masked_samples": pa.array(bp_masked, type=pa.list_(pa.int8())),
                "rp_masked_samples": pa.array(rp_masked, type=pa.list_(pa.int8())),
                "bp_ccd_proc_flags": pa.array(bp_proc, pa.int16()),
                "rp_ccd_proc_flags": pa.array(rp_proc, pa.int16()),
            }
        ),
        pq_path,
    )
    ep = XPEpochSpectra.from_parquet(pq_path)
    assert ep.flux.shape == (n_t, 2, n_pix)
    assert ep.grid.shape == (n_t, 2, n_pix)
    assert ep.has_crowding_entry()[:, 0].tolist() == [True, False, True, False]
    vm = ep.valid_mask()
    assert not vm[0, 0, 10]  # bp_masked_samples
    assert not vm[2, 0].any()  # ccd_proc_flags bit 8

    # Resample onto a shared pseudo-wavelength grid and coadd per source
    tgt = torch.linspace(1.0, 59.0, 30, dtype=torch.float64)
    usid, mf, me = ep.mean_flux(target_grid=tgt)
    assert usid.tolist() == [101, 102]
    assert torch.allclose(mf[:, 0], tgt + 1.0, atol=1e-5)
    assert torch.allclose(mf[:, 1], 2.0 * tgt + 3.0, atol=1e-5)
    assert torch.isfinite(me).all()


def test_factor_only_correlation_mode():
    batch = make_batch(4, realistic=False)
    xp = XPSpectra.from_batch(batch)
    cal = xp.calibrate(with_correlation=True)
    cal_f = xp.calibrate(with_correlation="factor")
    assert "factor" in cal_f and "covariance" not in cal_f
    assert torch.allclose(cal_f["factor"], cal["factor"])
    conv = xp.convert(with_correlation=True)
    conv_f = xp.convert(with_correlation="factor")
    assert "factor" in conv_f and "covariance" not in conv_f
    assert torch.allclose(conv_f["factor"], conv["factor"])


def test_review_edge_case_regressions(tmp_path):
    from importlib import resources

    from torchgaiaxp.io import (
        _arrow_list_to_numpy,
        _pad_and_stack_bands,
        from_dataframe,
        to_dataframe,
    )

    # 1. All-missing band with inferred width 0 must remain NaN, not 0.0
    bp_only = np.ones((3, 5), dtype=np.float64)
    rp_empty = np.empty((3, 0), dtype=np.float64)
    stacked = _pad_and_stack_bands(bp_only, rp_empty)
    assert stacked.shape == (3, 2, 5)
    assert np.isfinite(stacked[:, 0]).all()
    assert np.isnan(stacked[:, 1]).all()

    r_bp = np.zeros((3, 10), dtype=np.float64)
    r_rp = np.empty((3, 0), dtype=np.float64)
    r_stacked = _pad_and_stack_bands(r_bp, r_rp, corr=True)
    assert np.isfinite(r_stacked[:, 0]).all()
    assert np.isnan(r_stacked[:, 1]).all()

    # 2. Sliced Arrow ListArray with offsets[0] > 0
    sliced = pa.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], type=pa.list_(pa.float64())).slice(1, 2)
    np.testing.assert_allclose(_arrow_list_to_numpy(sliced), [[3.0, 4.0], [5.0, 6.0]])

    # 3. Unequal n_bp != n_rp XML filter + to_dataframe/from_dataframe round-trip
    xml = """<GenericPhotometricSystem>
  <bands><band>A</band></bands>
  <zeropoints><zeroPoint>25.0</zeroPoint></zeropoints>
  <XpMerge><bpWeights><weight>1.0</weight></bpWeights><rpWeights><weight>1.0</weight></rpWeights></XpMerge>
  <XpSampling>
    <bpSampledBases dimension="4"><val>1</val><val>2</val><val>3</val><val>4</val></bpSampledBases>
    <rpSampledBases dimension="2"><val>5</val><val>6</val></rpSampledBases>
  </XpSampling>
</GenericPhotometricSystem>"""
    xml_path = tmp_path / "unequal.xml"
    xml_path.write_text(xml, encoding="utf-8")
    udes, _ = XPDesign.from_filter(xml_path)
    assert udes.design.shape == (2, 1, 4)
    assert torch.allclose(udes.design[1, 0, 2:], torch.zeros(2, dtype=torch.float64))

    c_bp = np.arange(8, dtype=np.float64).reshape(2, 4)
    c_rp = np.arange(4, dtype=np.float64).reshape(2, 2)
    ubatch = XPBatch(
        source_id=np.array([10, 11], dtype=np.int64),
        coefficients=_pad_and_stack_bands(c_bp, c_rp),
        n_parameters=np.tile([4, 2], (2, 1)),
    )
    rt = from_dataframe(to_dataframe(ubatch))
    np.testing.assert_allclose(rt.coefficients, ubatch.coefficients, equal_nan=True)

    # 4. Overwriting register_release clears derived dispersion/design caches, and
    # apply_error_correction infers non-default release automatically
    dr3_dir = resources.files("torchgaiaxp") / "_assets" / "dr3"
    reg_dir = tmp_path / "dr4_cache_test"
    reg_dir.mkdir()
    with resources.as_file(dr3_dir) as src:
        for fname in ("instrument.json", "instrument.npz", "systems.json", "systems.npz"):
            if fname == "systems.json":
                sdata = json.loads((src / fname).read_text())
                sdata["DR4C_Gaia_Vega"] = dict(sdata["DR3_Gaia_Vega"])
                sdata["DR4C_SDSS_Std"] = dict(sdata["DR3_SDSS_Std"])
                sdata["_gaia_vega_system"] = "DR4C_Gaia_Vega"
                (reg_dir / fname).write_text(json.dumps(sdata))
            elif fname == "systems.npz":
                with np.load(src / fname) as z:
                    arrays = {k: z[k] for k in z.files}
                    for suf in ("design", "merge", "zero_points", "offsets"):
                        arrays[f"DR4C_Gaia_Vega/{suf}"] = z[f"DR3_Gaia_Vega/{suf}"]
                        arrays[f"DR4C_SDSS_Std/{suf}"] = z[f"DR3_SDSS_Std/{suf}"]
                    np.savez_compressed(reg_dir / fname, **arrays)
            else:
                (reg_dir / fname).write_bytes((src / fname).read_bytes())

    register_release("dr4c", reg_dir)
    try:
        _ = tg.wl_to_pwl("bp", [400.0, 500.0], release="dr4c")
        _ = XPDesign.photometric("DR4C_SDSS_Std", release="dr4c")

        xp_c = XPSpectra.from_batch(make_batch(3, realistic=False), release="dr4c")
        raw_res = xp_c.generate(["DR4C_SDSS_Std", "DR4C_Gaia_Vega"], error_correction=False)
        auto_ec = apply_error_correction(raw_res)
        direct_ec = xp_c.generate(["DR4C_SDSS_Std", "DR4C_Gaia_Vega"], error_correction=True)
        assert torch.allclose(
            auto_ec["DR4C_SDSS_Std"]["flux_error"],
            direct_ec["DR4C_SDSS_Std"]["flux_error"],
            equal_nan=True,
        )

        # Overwrite dr4c with shifted dispersion PWL -> cache must be invalidated
        with np.load(reg_dir / "instrument.npz") as z:
            inst = {k: z[k].copy() for k in z.files}
        inst["dispersion_function_bp_pwl"] = inst["dispersion_function_bp_pwl"] + 5.0
        np.savez_compressed(reg_dir / "instrument.npz", **inst)
        register_release("dr4c", reg_dir)
        shifted = tg.wl_to_pwl("bp", [400.0, 500.0], release="dr4c")
        assert torch.allclose(
            shifted, tg.wl_to_pwl("bp", [400.0, 500.0], release="dr3") + 5.0, atol=1e-9
        )
    finally:
        tg._assets._EXTRA_RELEASES.pop("dr4c", None)

    # 5. XPEpochSpectra [2, G] shared-band grid indexing & source_id=None valid_mask in to_sampled
    ep = XPEpochSpectra(
        flux=torch.ones((5, 2, 10), dtype=torch.float64),
        grid=torch.linspace(0, 10, 10, dtype=torch.float64).unsqueeze(0).expand(2, 10).clone(),
        crowding_flag=torch.tensor([False, True, False, False, False]),
    )
    ep1 = ep[-1]
    assert ep1.flux.shape == (1, 2, 10) and ep1.grid.shape == (2, 10)
    ep_1d = XPEpochSpectra(
        flux=torch.ones((2, 4), dtype=torch.float64),
        grid=torch.linspace(400, 700, 4, dtype=torch.float64),
        quality_mask=torch.tensor([True, False]),
    )
    sampled = ep_1d.to_sampled()
    assert torch.isfinite(sampled.flux[0]).all() and torch.isnan(sampled.flux[1]).all()


def test_cuda_end_to_end_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    batch = make_batch(8, realistic=False)
    for dtype in (torch.float32, torch.float64):
        xp = XPSpectra.from_batch(batch, device="cuda", dtype=dtype)
        cal = xp.calibrate(with_correlation=True)
        cal_f = xp.calibrate(with_correlation="factor")
        conv = xp.convert(with_correlation=True)
        conv_f = xp.convert(with_correlation="factor")
        gen = xp.generate(["DR3_SDSS_Std", "DR3_JKC_Std"], error_correction=True)
        isr = xp.inverse_square_root_covariance("bp")
        c2 = xp.chi2("bp", torch.zeros(xp.n_bases, device="cuda", dtype=dtype))
        assert cal["flux"].is_cuda and cal["covariance"].is_cuda
        assert "factor" in cal_f and "covariance" not in cal_f
        assert "factor" in conv_f and "covariance" not in conv_f
        assert torch.allclose(cal_f["factor"], cal["factor"])
        assert conv["flux"].is_cuda and gen["DR3_SDSS_Std"]["mag"].is_cuda
        assert isr.is_cuda and c2.is_cuda

    ep = XPEpochSpectra(
        flux=torch.ones((6, 2, 60), device="cuda", dtype=torch.float32),
        grid=torch.linspace(0, 60, 60, device="cuda", dtype=torch.float32)
        .view(1, 1, 60)
        .expand(6, 2, 60)
        .clone(),
        errors=torch.full((6, 2, 60), 0.1, device="cuda", dtype=torch.float32),
        source_id=torch.tensor([1, 1, 2, 2, 1, 2], device="cuda", dtype=torch.int64),
    )
    usid, mf, me = ep.mean_flux(
        target_grid=torch.linspace(5, 55, 25, device="cuda", dtype=torch.float32)
    )
    assert usid.is_cuda and mf.is_cuda and torch.allclose(mf, torch.ones_like(mf))
    assert torch.isfinite(me).all()


