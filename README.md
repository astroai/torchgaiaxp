# torchgaiaxp

Gaia XP spectra in PyTorch: continuous, batched, differentiable, and on whatever
device you put them. A port of [GaiaXPy](https://github.com/gaia-dpci/GaiaXPy)
2.2 with feature parity, multi-release / DR4 readiness (`XPEpochSpectra`,
variable `n_bases`, `basis_type`, Archive Parquet / FITS / VOTable / DataFrame
I/O), and an algebra for combining XP spectra with spectra of any other resolution.

GaiaXPy is not needed at runtime: its configuration tables are packaged under
`src/torchgaiaxp/_assets/<release>` (BSD-3, DPAC CU5/DPCI; see `LICENSE-GAIAXPY`)
and the bases are evaluated natively in torch.

```python
import torch
from torchgaiaxp import XPSpectra, XPEpochSpectra, Sampled
from torchgaiaxp.io import read_xp

xp = XPSpectra.from_batch(read_xp("XpContinuousMeanSpectrum_000000-003111.csv.gz"),
                          device="cuda", dtype=torch.float32)

xp.calibrate()                       # GaiaXPy calibrate: flux, flux_error on 336-1020 nm
xp.calibrate(sampling, with_correlation=True)      # + exact low-rank factor and covariance
xp.calibrate(sampling, with_correlation="factor")  # low-rank factor only (low VRAM)
xp.convert()                         # internal, per band, pseudo-wavelengths
xp.generate(["DR3_SDSS_Std", "DR3_JKC_Std"], error_correction=True)  # 28 systems (+ legacy aliases)

wl = torch.linspace(380, 950, 5000, device="cuda")
xp(wl)                               # evaluate anywhere (per-source grids [B, G] too)
model = xp.shift(z) * Sampled(hi_wl, transmission)   # lazy; any resolution
ev = model.evaluate(wl)              # flux, factor (correlated), variance (diag)
flux, err = xp.integrate(passband, wl)               # any filter curve
XPSpectra.fit(Sampled(wl_model, flux_model).smooth(100.0))  # XP-basis view of a model
```

## What matches GaiaXPy 2.2, and how closely

Tested against GaiaXPy 2.2.0 (`tests/test_against_gaiaxpy.py`, float64):

| Function | Agreement |
|---|---|
| `calibrate` (default and custom grids, truncation, missing bands) | flux 1e-9, errors 1e-6 |
| `calibrate` / `convert` sample correlations (`pack_correlations`) | 1e-5 |
| native absolute bases at any wavelength vs GaiaXPy's custom-grid design | 1e-12 of peak |
| `convert` (custom pseudo-wavelengths, truncation, missing bands) | 1e-9 / 1e-6 |
| `generate`, all 28 systems + legacy aliases: flux, flux error, mag | 1e-7 |
| standardised offsets, JKC_Std/SDSS_Std colour equations | 1e-7 |
| error correction (all G branches) | 1e-7 |
| `inverse_square_root_covariance`, `inverse_covariance`, `chi2` | 1e-8 |
| `pwl_to_wl`, `wl_to_pwl`, `pwl_range`, `wl_range` | 1e-12 |

float32 agrees with float64 to ~1e-4 of the peak.

## Things worth knowing

- **Two default designs in GaiaXPy.** `calibrate(sampling=None)` uses DPAC's
  packaged matrix; any explicit sampling evaluates the instrument model. On the
  default grid they differ by up to 0.3% of the BP peak. `XPDesign.default()`
  reproduces the former, `XPDesign.absolute(wl)` the latter.
- **Merged covariance.** The covariance is `F F^T` with the exact factor
  `F = [w_bp D_bp diag(e_bp) L_bp, w_rp ...]` (`[B, G, 2N]`), matching
  GaiaXPy 2.2's `with_correlation=True` covariance and errors across the
  635-643 nm blend. When `with_correlation=False`, `calibrate`, `convert`, and
  `generate` evaluate exact variances via a Cholesky-free packed symmetric form.
- **Cholesky tools use `e / sigma`** (GaiaXPy convention); sampled errors use `e`.
- **`fit` is a projection, not the DPAC forward model.** It is the least-squares
  projection onto the span of the absolute bases (what a spectrum looks like at
  XP band limit).
- **Continuous algebra** propagates uncertainty to first order: XP covariance as
  a low-rank factor (correlated across wavelength, and coherent when the same
  XP spectrum appears multiple times in an expression), tabulated errors as an
  independent diagonal.
- **Gaia DR4 readiness.** Multi-release assets (`register_release`, `release`,
  `basis_type`), variable `n_bases` (and unequal per-band widths), provenance
  tracking (`solution_id`, `basis_function_id`, `n_parameters`), Archive
  Parquet/FITS/VOTable/DataFrame readers (`read_xp`, `read_archive_parquet`),
  custom XML/YAML filters (`load_additional_systems`), and `XPEpochSpectra` for
  transit-level `xp_epoch_spectrum` + crowding masks.
- **Devices.** Assets are converted once per `(release, device, dtype)` and
  cached; outputs land on the input's device. Tests run on CPU, CUDA, and MPS.

## Bulk archive files

`scripts/bulk_to_parquet.py` converts the archive's `XpContinuousMeanSpectrum_*.csv.gz`
to Parquet once (resumable, parallel); `torchgaiaxp.io.read_xp` reads them back
(and directly reads Archive Parquet/VOParquet files via `read_archive_parquet`).

## Rebuilding the assets

`pip install gaiaxpy scipy pandas pyyaml && python scripts/build_assets.py`.
