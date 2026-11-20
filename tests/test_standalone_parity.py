"""Standalone GaiaXPy-parity tests (FITS via torchfits, VOTable XML, ZIP, writers, Archive HTTP mocks, plotter)."""

from __future__ import annotations

import importlib.util
import zipfile

import numpy as np
import pytest
import torch
from conftest import frame, make_batch

import torchgaiaxp as tg
from torchgaiaxp import XPSpectra, calibrate, convert, generate, get_required_columns, plot_spectra
from torchgaiaxp.io import (
    read_fits,
    read_votable,
    read_xp,
    read_zip,
    to_parquet,
    write_ecsv_like,
    write_fits,
    write_votable,
)


def test_fits_round_trip(tmp_path):
    """FITS BINTABLE write + read using torchfits (or astropy fallback when torchfits unavailable)."""
    batch = make_batch(4, realistic=False)
    batch.coefficients[1, 1] = np.nan
    batch.solution_id = np.full(4, 4545469030156206080, dtype=np.int64)
    batch.n_parameters = np.full((4, 2), 55, dtype=np.int64)
    batch.n_parameters[1, 1] = 0

    fits_path = tmp_path / "xp_continuous.fits"
    write_fits(batch, fits_path, extname="GAIA_XP", header={"ORIGIN": "torchgaiaxp"})
    back = read_fits(fits_path)
    np.testing.assert_array_equal(back.source_id, batch.source_id)
    np.testing.assert_array_equal(back.solution_id, batch.solution_id)
    np.testing.assert_allclose(back.coefficients, batch.coefficients, equal_nan=True)
    np.testing.assert_allclose(back.errors, batch.errors, equal_nan=True)
    np.testing.assert_allclose(back.correlations, batch.correlations, rtol=1e-6, equal_nan=True)

    via_auto = read_xp(fits_path)
    np.testing.assert_allclose(via_auto.coefficients, batch.coefficients, equal_nan=True)


def test_native_xml_votable_round_trip(tmp_path):
    """XML TABLEDATA VOTable read/write works cleanly with stdlib XML (even uncompressed or .gz)."""
    batch = make_batch(3, realistic=False)
    batch.coefficients[2, 0] = np.nan
    vot_path = tmp_path / "xp_votable.vot.gz"
    write_votable(batch, vot_path, params={"RELEASE": "DR3"})
    back = read_votable(vot_path)
    np.testing.assert_array_equal(back.source_id, batch.source_id)
    np.testing.assert_allclose(back.coefficients, batch.coefficients, equal_nan=True)
    np.testing.assert_allclose(back.errors, batch.errors, equal_nan=True)
    np.testing.assert_allclose(back.correlations, batch.correlations, rtol=1e-6, equal_nan=True)

    via_auto = read_xp(vot_path)
    np.testing.assert_allclose(via_auto.coefficients, batch.coefficients, equal_nan=True)


def test_zip_archive_reader(tmp_path):
    b1 = make_batch(2, seed=1, realistic=False)
    b2 = make_batch(3, seed=2, realistic=False)
    f1 = tmp_path / "part1.csv"
    f2 = tmp_path / "part2.parquet"
    f3 = tmp_path / "part3.xml"
    write_ecsv_like(b1, f1)
    to_parquet(b2, f2)
    write_votable(b1, f3)

    zpath = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.write(f1, arcname="part1.csv")
        zf.write(f2, arcname="part2.parquet")
        zf.write(f3, arcname="part3.xml")

    combined = read_zip(zpath)
    assert len(combined) == len(b1) + len(b2) + len(b1)
    assert len(read_xp(zpath)) == len(combined)


def test_top_level_api_dataframe_series_and_writers(tmp_path):
    batch = make_batch(3, realistic=False)
    ecsv_path = tmp_path / "sample.csv.gz"
    write_ecsv_like(batch, ecsv_path)
    df_in = frame(batch)
    series_in = df_in.iloc[0]

    # Top-level calibrate -> (DataFrame, grid) + file writers across every format
    df_cal, wl = calibrate(
        ecsv_path,
        sampling=np.linspace(400.0, 900.0, 51),
        with_correlation=True,
        save_file=True,
        output_path=tmp_path,
        output_file="cal_out",
        output_format="ecsv",
    )
    assert (tmp_path / "cal_out.ecsv").is_file()
    assert df_cal.attrs["data_type"] == "AbsoluteSampledSpectrum"
    assert len(df_cal) == 3 and len(wl) == 51

    # Top-level convert -> (DataFrame, pwl)
    df_conv, pwl = convert(df_in, with_correlation=False)
    assert df_conv.attrs["data_type"] == "XpSampledSpectrum"
    assert len(df_conv) == 6 and len(pwl) == 600

    # Single pd.Series input
    df_s, _ = calibrate(series_in, sampling=wl)
    assert len(df_s) == 1

    # Top-level generate + standalone DataFrame error correction
    df_phot = generate(
        ecsv_path,
        photometric_system=["DR3_SDSS_Std", "DR3_Gaia_Vega"],
        error_correction=False,
        save_file=True,
        output_path=tmp_path,
        output_file="phot_out",
        output_format="votable",
    )
    assert (tmp_path / "phot_out.xml").is_file()
    assert "DR3_SdssStd_mag_g" in df_phot.columns
    df_ec = tg.apply_error_correction(df_phot)
    df_ec_direct = generate(
        ecsv_path,
        photometric_system=["DR3_SDSS_Std", "DR3_Gaia_Vega"],
        error_correction=True,
    )
    np.testing.assert_allclose(
        df_ec["DR3_SdssStd_flux_error_g"].to_numpy(),
        df_ec_direct["DR3_SdssStd_flux_error_g"].to_numpy(),
        equal_nan=True,
    )

    # Extra output formats: csv, parquet, fits (+ avro if fastavro installed)
    for fmt, ext in (("csv", ".csv"), ("parquet", ".parquet"), ("fits", ".fits")):
        generate(
            df_in,
            photometric_system="DR3_SDSS",
            save_file=True,
            output_path=tmp_path,
            output_file=f"phot_{fmt}",
            output_format=fmt,
        )
        assert (tmp_path / f"phot_{fmt}{ext}").is_file()

    if importlib.util.find_spec("fastavro") is not None:
        generate(
            df_in,
            photometric_system="DR3_SDSS",
            save_file=True,
            output_path=tmp_path,
            output_file="phot_avro",
            output_format="avro",
        )
        assert (tmp_path / "phot_avro.avro").is_file()

    # Overwrite=False guard
    with pytest.raises(FileExistsError):
        generate(
            df_in,
            photometric_system="DR3_SDSS",
            save_file=True,
            output_path=tmp_path,
            output_file="phot_csv",
            output_format="csv",
            overwrite=False,
        )

    # Top-level generate(xp, "DR3_SDSS") keeps tensor-dict return by default
    xp = XPSpectra.from_batch(batch)
    t_res = generate(xp, "DR3_SDSS")
    assert isinstance(t_res, dict) and isinstance(t_res["DR3_SDSS"]["flux"], torch.Tensor)


def test_archive_client_datalink_adql_and_auth(tmp_path, monkeypatch):
    """Mock ESA TAP/DataLink + CAS login endpoints and test ADQL & source_id retrieval."""
    batch = make_batch(3, realistic=False)
    ecsv_path = tmp_path / "dl_payload.csv"
    write_ecsv_like(batch, ecsv_path)
    payload_bytes = ecsv_path.read_bytes()

    cred_file = tmp_path / "creds.txt"
    cred_file.write_text("user1\npass1\n", encoding="utf-8")

    calls: list[tuple[str, str]] = []

    class _FakeHeaders:
        def get(self, key, default=""):
            return "text/csv" if key.lower() == "content-type" else default

    class _FakeResp:
        def __init__(self, data: bytes):
            self._data = data
            self.headers = _FakeHeaders()

        def read(self):
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def _fake_open(self, req, timeout=None):
        body = (req.data or b"").decode("utf-8")
        calls.append((req.full_url, body))
        if req.full_url.endswith("/tap-server/login"):
            return _FakeResp(b"OK")
        if "SELECT+source_id" in body or "SELECT%20source_id" in body:
            return _FakeResp(b"source_id\n" + "\n".join(str(int(s)) for s in batch.source_id).encode() + b"\n")
        return _FakeResp(payload_bytes)

    import urllib.request

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", _fake_open)

    # 1. Source IDs via top-level calibrate()
    df_cal, wl = calibrate(
        batch.source_id.tolist(),
        sampling=np.linspace(450, 750, 11),
        credentials_file=cred_file,
    )
    assert len(df_cal) == 3 and len(wl) == 11
    assert any(url.endswith("/tap-server/login") for url, _ in calls)
    assert any("data-server/datalink/links" in url for url, _ in calls)

    # 2. ADQL selecting source_id -> falls back to DataLink seamlessly
    df_phot = generate("SELECT source_id FROM gaiadr3.gaia_source WHERE ...", photometric_system="DR3_SDSS")
    assert len(df_phot) == 3 and "DR3_Sdss_mag_g" in df_phot.columns


def test_get_required_columns_and_plot_spectra(tmp_path, monkeypatch):
    req = get_required_columns("calibrate")
    assert "bp_coefficients" in req.required
    assert "bp_n_relevant_bases" in req.conditional["truncation=True"]
    with pytest.raises(NotImplementedError):
        get_required_columns("nonexistent_fn")

    import matplotlib

    matplotlib.use("Agg")
    batch = make_batch(2, realistic=False)
    xp = XPSpectra.from_batch(batch)
    df_cal, wl = calibrate(xp, sampling=np.linspace(400, 900, 21), return_tensors=False)
    df_conv, pwl = convert(xp, return_tensors=False)

    # DataFrame multi & individual
    plot_spectra(df_cal, wl, multi=True, show_plot=False, output_path=tmp_path, output_file="abs_multi", format="png")
    plot_spectra(df_cal, wl, multi=False, show_plot=False, output_path=tmp_path, output_file="abs_single", format="png")
    plot_spectra(df_conv, pwl, multi=True, show_plot=False, output_path=tmp_path, output_file="xp_multi", format="png")
    # XPSpectra & tensor dict directly
    plot_spectra(xp, wl, multi=True, show_plot=False, output_path=tmp_path, output_file="from_xp", format="png")
    plot_spectra(xp.convert(), show_plot=False, output_path=tmp_path, output_file="from_dict", format="png")

    assert (tmp_path / "abs_multi.png").is_file()
    assert (tmp_path / "abs_single_0.png").is_file()
    assert (tmp_path / "abs_single_1.png").is_file()
    assert (tmp_path / "xp_multi.png").is_file()
    assert (tmp_path / "from_xp.png").is_file()
    assert (tmp_path / "from_dict_0.png").is_file()
