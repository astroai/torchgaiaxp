from __future__ import annotations

import numpy as np
import torch
from conftest import make_batch

from torchgaiaxp import XPSpectra
from torchgaiaxp.io import read_parquet, read_xp_continuous, to_parquet, write_ecsv_like


def test_bulk_file_round_trip(tmp_path):
    batch = make_batch(4, realistic=False)
    batch.coefficients[1, 1] = np.nan
    path = tmp_path / "XpContinuousMeanSpectrum_000-003.csv.gz"
    write_ecsv_like(batch, path)
    read = read_xp_continuous(path)
    np.testing.assert_array_equal(read.source_id, batch.source_id)
    np.testing.assert_allclose(read.coefficients, batch.coefficients, equal_nan=True)
    np.testing.assert_allclose(read.correlations, batch.correlations, rtol=1e-6, equal_nan=True)
    pq = to_parquet(read, tmp_path / "x.parquet", float_dtype=np.float64)
    again = read_parquet(pq)
    xp = XPSpectra.from_batch(again)
    assert torch.isfinite(xp(torch.linspace(400, 600, 5, dtype=torch.float64))[0]).all()
