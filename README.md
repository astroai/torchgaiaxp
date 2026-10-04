# torchgaiaxp

Gaia XP spectra in PyTorch: continuous, batched, differentiable, and on whatever
device you put them. A port of [GaiaXPy](https://github.com/gaia-dpci/GaiaXPy)
2.1 with feature parity, plus an algebra for combining XP spectra with spectra
of any other resolution.

GaiaXPy is not needed at runtime: its configuration tables are packaged under
`src/torchgaiaxp/_assets` (BSD-3, DPAC CU5/DPCI; see `LICENSE-GAIAXPY`) and the
bases are evaluated natively in torch.

```python
import torch
from torchgaiaxp import XPSpectra, Sampled
from torchgaiaxp.io import read_xp_continuous

xp = XPSpectra.from_batch(read_xp_continuous("XpContinuousMeanSpectrum_000000-003111.csv.gz"),
                          device="cuda", dtype=torch.float32)

xp.calibrate()                       # GaiaXPy calibrate: flux, flux_error on 336-1020 nm
xp.calibrate(sampling, with_correlation=True)  # + exact low-rank factor and covariance
xp.convert()                         # internal, per band, pseudo-wavelengths
xp.generate(["SDSS_Std", "JKC_Std"], error_correction=True)  # 28 systems

wl = torch.linspace(380, 950, 5000, device="cuda")
xp(wl)                               # evaluate anywhere (per-source grids [B, G] too)
model = xp.shift(z) * Sampled(hi_wl, transmission)   # lazy; any resolution
ev = model.evaluate(wl)              # flux, factor (correlated), variance (diag)
flux, err = xp.integrate(passband, wl)               # any filter curve
XPSpectra.fit(Sampled(wl_model, flux_model).smooth(100.0))  # XP-basis view of a model
```

## What matches GaiaXPy, and how closely

Tested against GaiaXPy 2.1.4 (`tests/test_against_gaiaxpy.py`, float64):

| Function | Agreement |
|---|---|
| `calibrate` (default and custom grids, truncation, missing bands) | flux 1e-9, errors 1e-6 |
| native absolute bases at any wavelength vs GaiaXPy's custom-grid design | 1e-12 of peak |
| `convert` (custom pseudo-wavelengths, truncation, missing bands) | 1e-9 / 1e-6 |
| `generate`, all 28 systems: flux, flux error, mag | 1e-7 |
| standardised offsets, JKC_Std/SDSS_Std colour equations | 1e-7 |
| error correction (all G branches) | 1e-7 |
| `get_inverse_square_root_covariance_matrix`, `get_inverse_covariance_matrix`, `get_chi2` | 1e-8 |
| `pwl_to_wl`, `wl_to_pwl`, `pwl_range`, `wl_range` | 1e-12 |

float32 agrees with float64 to ~1e-4 of the peak.

## Things worth knowing

- **Two default designs in GaiaXPy.** `calibrate(sampling=None)` uses DPAC's
  packaged XML matrix; any explicit sampling evaluates the instrument model.
  On the default grid they differ by up to 0.3% of the BP peak. `XPDesign.default()`
  reproduces the former, `XPDesign.absolute(wl)` the latter.
- **Merged covariance.** GaiaXPy's `with_correlation=True` merges band
  covariances with the merge weights applied once (`w C`, not `w C w`), while its
  errors use `w^2`. Inside the 635-643 nm blend its correlations are therefore
  inconsistent with its errors (off by up to 0.31). Here the covariance is
  `F F^T` with the exact factor `F = [w_bp D_bp diag(e_bp) L_bp, w_rp ...]`
  (`[B, G, 110]`), whose diagonal equals GaiaXPy's errors.
- **Cholesky tools use `e / sigma`** (GaiaXPy convention); sampled errors use `e`.
- **`fit` is a projection, not the DPAC forward model.** It is the least-squares
  projection onto the span of the absolute bases (what a spectrum looks like at
  XP band limit). The merged bases are conditioned at ~1e15, so coefficients are
  not unique; `rcond` trades flux fidelity against coefficient size (default
  1e-8: ~0.5% on a blackbody, O(1) coefficients). DPAC's LSF/calibration kernels
  are not in GaiaXPy.
- **Continuous algebra** propagates uncertainty to first order: XP covariance as
  a low-rank factor (correlated across wavelength, and correctly correlated when
  the same XP spectrum appears twice in an expression), tabulated errors as an
  independent diagonal (interpolation keeps it diagonal: an approximation).
- **Devices.** Assets are converted once per (device, dtype) and cached; every
  output lands on the input's device. Tests run on CPU, CUDA and MPS when present.

## Throughput (2-core cloud CPU; GPU numbers belong to a GPU node)

| | torchgaiaxp f32 | torchgaiaxp f64 | GaiaXPy |
|---|---|---|---|
| calibrate (errors) | 2,100/s | 1,070/s | 870/s |
| generate SDSS_Std | 10,600/s | 7,700/s | 1,360/s |

`python scripts/benchmark.py 20000 cuda` on a GPU node.

## Bulk archive files

`scripts/bulk_to_parquet.py` converts the archive's `XpContinuousMeanSpectrum_*.csv.gz`
to Parquet once (resumable, parallel); `torchgaiaxp.io.read_parquet` reads them back.

## Rebuilding the assets

`pip install gaiaxpy scipy pandas && python scripts/build_assets.py`.
