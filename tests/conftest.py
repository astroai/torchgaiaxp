from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from torchgaiaxp import Function, XPSpectra
from torchgaiaxp.io import N_BASES, N_CORR, XPBatch


def planck(temperature: float):
    def f(wl):
        x = 1.4387769e7 / (wl * temperature)  # hc/(lambda k T), lambda in nm
        return 1e-18 * (500.0 / wl) ** 5 / torch.expm1(x)

    return Function(f)


def make_batch(n: int = 6, seed: int = 0, realistic: bool = True) -> XPBatch:
    rng = np.random.default_rng(seed)
    if realistic:
        temps = np.linspace(3500, 12000, n)
        scales = 10 ** rng.uniform(-1, 1, n)
        coeffs = []
        for t, s in zip(temps, scales):
            xp = XPSpectra.fit(planck(float(t)) * float(s))
            coeffs.append(xp.coefficients[0].numpy())
        coefficients = np.stack(coeffs)
    else:
        coefficients = rng.normal(size=(n, 2, N_BASES)) * np.exp(-np.arange(N_BASES) / 8) * 50
    scale = np.abs(coefficients).max(-1, keepdims=True)
    errors = (np.abs(rng.normal(size=(n, 2, N_BASES))) * 0.01 + 0.002) * scale
    correlations = np.empty((n, 2, N_CORR))
    rows, cols = np.tril_indices(N_BASES, -1)
    for i in range(n):
        for k in range(2):
            a = rng.normal(size=(N_BASES, N_BASES))
            c = a @ a.T + N_BASES * np.eye(N_BASES)
            d = np.sqrt(np.diag(c))
            correlations[i, k] = (c / np.outer(d, d))[rows, cols]
    return XPBatch(
        source_id=np.arange(n, dtype=np.int64) + 1000,
        coefficients=coefficients,
        errors=errors,
        correlations=correlations,
        standard_deviation=rng.uniform(0.8, 2.0, size=(n, 2)),
        n_relevant=rng.integers(20, 55, size=(n, 2)).astype(np.int16),
    )


def drop_bands(batch: XPBatch) -> XPBatch:
    for name in ("coefficients", "errors", "correlations"):
        getattr(batch, name)[1, 1] = np.nan
        getattr(batch, name)[2, 0] = np.nan
    return batch


def frame(batch: XPBatch) -> pd.DataFrame:
    rows = []
    for i in range(len(batch)):
        row = {"source_id": int(batch.source_id[i])}
        for k, b in enumerate(("bp", "rp")):
            present = np.isfinite(batch.coefficients[i, k]).all()
            row[f"{b}_coefficients"] = batch.coefficients[i, k] if present else np.nan
            row[f"{b}_coefficient_errors"] = batch.errors[i, k] if present else np.nan
            row[f"{b}_coefficient_correlations"] = (
                batch.correlations[i, k].astype(np.float64) if present else np.nan
            )
            row[f"{b}_n_parameters"] = N_BASES if present else np.nan
            row[f"{b}_standard_deviation"] = batch.standard_deviation[i, k] if present else np.nan
            row[f"{b}_n_relevant_bases"] = int(batch.n_relevant[i, k]) if present else np.nan
            row[f"{b}_basis_function_id"] = (56 if b == "bp" else 57) if present else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def batch() -> XPBatch:
    return make_batch()


DEVICES = (
    ["cpu"]
    + (["cuda"] if torch.cuda.is_available() else [])
    + (
        ["mps"]
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else []
    )
)
