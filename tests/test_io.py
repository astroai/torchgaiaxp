from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from conftest import frame, make_batch

from torchgaiaxp import XPSpectra
from torchgaiaxp.io import (
    from_dataframe,
    parse_array_column,
    read_archive_parquet,
    read_parquet,
    read_xp,
    read_xp_continuous,
    to_dataframe,
    to_parquet,
    write_ecsv_like,
)


def test_bulk_file_round_trip(tmp_path):
    batch = make_batch(4, realistic=False)
    batch.coefficients[1, 1] = np.nan
    batch.solution_id = np.full(4, 4545469030156206080, dtype=np.int64)
    batch.basis_function_id = np.tile(np.array([[56, 57]], dtype=np.int64), (4, 1))
    batch.n_parameters = np.full((4, 2), 55, dtype=np.int64)

    path = tmp_path / "XpContinuousMeanSpectrum_000-003.csv.gz"
    write_ecsv_like(batch, path)
    read = read_xp_continuous(path)
    np.testing.assert_array_equal(read.source_id, batch.source_id)
    np.testing.assert_array_equal(read.solution_id, batch.solution_id)
    want_np = batch.n_parameters.copy()
    want_np[1, 1] = 0
    np.testing.assert_array_equal(read.n_parameters, want_np)
    np.testing.assert_allclose(read.coefficients, batch.coefficients, equal_nan=True)
    np.testing.assert_allclose(read.correlations, batch.correlations, rtol=1e-6, equal_nan=True)

    pq_path = to_parquet(read, tmp_path / "x.parquet", float_dtype=np.float64)
    again = read_parquet(pq_path)
    assert again.correlations.dtype == np.float64
    xp = XPSpectra.from_batch(again)
    assert torch.isfinite(xp(torch.linspace(400, 600, 5, dtype=torch.float64))[0]).all()

    tensors = again.tensors()
    assert "standard_deviation" in tensors and "source_id" in tensors and "solution_id" in tensors


def test_parse_array_column_edge_cases():
    out = parse_array_column(["   ", "[] ", "NaN", None, "[1.0, 2.0, null]"], width=3)
    assert out.shape == (5, 3)
    assert np.isnan(out[:4]).all()
    np.testing.assert_allclose(out[4], [1.0, 2.0, np.nan], equal_nan=True)


def test_archive_parquet_and_dataframe_round_trip(tmp_path):
    batch = make_batch(3, realistic=False)
    batch.coefficients[1, 1] = np.nan
    df = frame(batch)

    from_df = from_dataframe(df)
    assert len(from_df) == 3
    xp_df = XPSpectra.from_dataframe(df)
    xp_ref = XPSpectra.from_batch(batch)
    assert torch.allclose(xp_df.calibrate()["flux"], xp_ref.calibrate()["flux"], equal_nan=True)
    back_df = to_dataframe(xp_df)
    assert len(back_df) == 3

    # Write an Archive-style Parquet with native Arrow list<double> and null rows.
    def list_col(arr):
        rows = [None if not np.isfinite(r).all() else r.tolist() for r in arr]
        return pa.array(rows, type=pa.list_(pa.float64()))

    arch_table = pa.table(
        {
            "source_id": pa.array(batch.source_id, pa.int64()),
            "bp_coefficients": list_col(batch.coefficients[:, 0]),
            "rp_coefficients": list_col(batch.coefficients[:, 1]),
            "bp_coefficient_errors": list_col(batch.errors[:, 0]),
            "rp_coefficient_errors": list_col(batch.errors[:, 1]),
            "bp_coefficient_correlations": list_col(batch.correlations[:, 0]),
            "rp_coefficient_correlations": list_col(batch.correlations[:, 1]),
            "bp_standard_deviation": pa.array(batch.standard_deviation[:, 0]),
            "rp_standard_deviation": pa.array(batch.standard_deviation[:, 1]),
            "bp_n_relevant_bases": pa.array(batch.n_relevant[:, 0]),
            "rp_n_relevant_bases": pa.array(batch.n_relevant[:, 1]),
        }
    )
    arch_path = tmp_path / "archive_xp.parquet"
    pq.write_table(arch_table, arch_path)
    arch_batch = read_xp(arch_path)
    assert isinstance(read_archive_parquet(arch_path).coefficients, np.ndarray)
    np.testing.assert_allclose(arch_batch.coefficients, batch.coefficients, equal_nan=True)
    assert torch.allclose(
        XPSpectra.from_batch(arch_batch).calibrate()["flux"],
        xp_ref.calibrate()["flux"],
        equal_nan=True,
    )
