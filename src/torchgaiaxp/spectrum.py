"""Batches of Gaia XP mean spectra in their continuous representation.

:class:`XPSpectra` holds what the archive delivers per source and band
(coefficients, formal errors, correlations, standard deviation, number of
relevant bases, and optional release/provenance identifiers) as tensors on one
device, and is a :class:`Continuous`: call it at any wavelengths, combine it
with other spectra, shift it, integrate it.

GaiaXPy parity: :meth:`calibrate` (absolute sampled spectra), :meth:`convert`
(internal), :meth:`generate` (synthetic photometry, see
:mod:`torchgaiaxp.photometry`), and the Cholesky products
(:mod:`torchgaiaxp.cholesky`).

The merged covariance (``factor``, ``covariance``) is ``F F^T`` with
``F = [w_bp D_bp diag(e_bp) L_bp, w_rp D_rp diag(e_rp) L_rp]``: exact, and its
diagonal equals GaiaXPy's errors and GaiaXPy >=2.2's merged covariance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace

import numpy as np
import torch
from torch import Tensor

from . import _assets
from .continuous import Continuous, Evaluation, _wl
from .design import XPDesign

N_BASES = 55
_TRIL_CACHE: dict[tuple[int, str], tuple[Tensor, Tensor]] = {}


def _mm(a: Tensor, b: Tensor) -> Tensor:
    """2D matmul safe on MPS (PyTorch MPS ``mm`` aborts on non-contiguous or offset views)."""
    if a.device.type == "mps":
        a = a.clone() if a.storage_offset() else a.contiguous()
        b = b.clone() if b.storage_offset() else b.contiguous()
    return a @ b


def _tril_indices(n: int, device: torch.device) -> tuple[Tensor, Tensor]:
    key = (n, str(device))
    hit = _TRIL_CACHE.get(key)
    if hit is None:
        hit = tuple(torch.tril_indices(n, n, offset=-1, device=device))  # type: ignore[assignment]
        _TRIL_CACHE[key] = hit
    return hit


def _infer_matrix_size(packed_len: int) -> int:
    n = int((math.isqrt(1 + 8 * packed_len) + 1) // 2)
    if n * (n - 1) // 2 != packed_len:
        raise ValueError(f"packed length {packed_len} is not n*(n-1)/2 for any integer n")
    return n


def unpack_correlations(packed: Tensor, n: int | None = None) -> Tensor:
    """Archive packed correlations ``[..., n(n-1)/2]`` -> ``[..., n, n]``.

    The strictly lower triangle in ``np.tril_indices(n, -1)`` order; diagonal 1.
    A ``[..., n, n]`` input is returned unchanged.
    """
    if packed.dim() >= 2 and packed.shape[-1] == packed.shape[-2] and (
        n is None or packed.shape[-1] == n
    ):
        return packed
    if n is None:
        n = _infer_matrix_size(packed.shape[-1])
    rows, cols = _tril_indices(n, packed.device)
    if packed.shape[-1] != rows.numel():
        raise ValueError(f"expected {rows.numel()} packed values, got {packed.shape[-1]}")
    matrix = packed.new_zeros((*packed.shape[:-1], n, n))
    matrix[..., rows, cols] = packed
    matrix = matrix + matrix.transpose(-1, -2)
    matrix.diagonal(dim1=-2, dim2=-1).fill_(1.0)
    return matrix


def pack_correlations(covariance: Tensor) -> Tensor:
    """Convert dense covariance ``[..., G, G]`` -> packed lower-triangle correlations."""
    g = covariance.shape[-1]
    rows, cols = _tril_indices(g, covariance.device)
    stdev = covariance.diagonal(dim1=-2, dim2=-1).clamp_min(0.0).sqrt()
    denom = stdev[..., rows] * stdev[..., cols]
    cov_off = covariance[..., rows, cols]
    safe = torch.where(denom != 0, denom, torch.ones_like(denom))
    return torch.where(denom != 0, cov_off / safe, 0.0)


@dataclass
class XPSpectra(Continuous):
    """``B`` sources. Per-band tensors are ``[B, 2, ...]`` with band order (BP, RP).

    A missing band is NaN in ``coefficients``. ``correlations`` may be packed
    ``[B, 2, N*(N-1)/2]`` or full ``[B, 2, N, N]``.
    """

    coefficients: Tensor
    errors: Tensor | None = None
    correlations: Tensor | None = None
    standard_deviation: Tensor | None = None  # [B, 2]
    n_relevant: Tensor | None = None  # [B, 2]
    source_id: Tensor | None = None  # [B] int64
    solution_id: Tensor | None = None  # [B] int64
    basis_function_id: Tensor | None = None  # [B, 2] int64
    n_parameters: Tensor | None = None  # [B, 2] int64
    truncation: bool = False
    release: str = _assets.DEFAULT_RELEASE
    basis_type: str = "basic"

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_batch(
        cls,
        batch,
        device=None,
        dtype=torch.float64,
        truncation: bool = False,
        release: str | None = None,
        basis_type: str = "basic",
    ) -> XPSpectra:
        """From :class:`torchgaiaxp.io.XPBatch` (or anything with its fields)."""

        def t(x, kind=dtype):
            if x is None:
                return None
            x = x if isinstance(x, torch.Tensor) else np.array(x)  # writable copy
            return torch.as_tensor(x).to(device=device, dtype=kind)

        sol_raw = getattr(batch, "solution_id", None)
        bf_raw = getattr(batch, "basis_function_id", None)
        np_raw = getattr(batch, "n_parameters", None)
        resolved_release = _assets.resolve_release(
            solution_id=sol_raw,
            n_parameters=np_raw,
            release=release if release is not None else getattr(batch, "release", None),
        )
        return cls(
            coefficients=t(batch.coefficients),
            errors=t(batch.errors),
            correlations=t(batch.correlations),
            standard_deviation=t(batch.standard_deviation),
            n_relevant=t(batch.n_relevant, torch.int64),
            source_id=t(batch.source_id, torch.int64),
            solution_id=t(sol_raw, torch.int64),
            basis_function_id=t(bf_raw, torch.int64),
            n_parameters=t(np_raw, torch.int64),
            truncation=truncation,
            release=resolved_release,
            basis_type=basis_type.lower(),
        )

    @classmethod
    def from_dataframe(
        cls,
        df,
        *,
        device=None,
        dtype=torch.float64,
        truncation: bool = False,
        release: str | None = None,
        basis_type: str = "basic",
    ) -> XPSpectra:
        """Build directly from a GaiaXPy-style ``pandas.DataFrame``."""
        from .io import from_dataframe

        batch = from_dataframe(df)
        return cls.from_batch(
            batch,
            device=device,
            dtype=dtype,
            truncation=truncation,
            release=release,
            basis_type=basis_type,
        )

    # -- basics ---------------------------------------------------------------
    @property
    def n_bases(self) -> int:
        return int(self.coefficients.shape[-1])

    @property
    def device(self) -> torch.device:
        return self.coefficients.device

    @property
    def dtype(self) -> torch.dtype:
        return self.coefficients.dtype

    def __len__(self) -> int:
        return self.coefficients.shape[0]

    def __getitem__(self, index) -> XPSpectra:
        if isinstance(index, int):
            n = len(self)
            if index < 0:
                index += n
            if not 0 <= index < n:
                raise IndexError(f"index {index} out of range for batch size {n}")
            index = slice(index, index + 1)
        values = {}
        for f in fields(self):
            v = getattr(self, f.name)
            values[f.name] = v[index] if isinstance(v, Tensor) else v
        return XPSpectra(**values)

    def to(self, device=None, dtype=None) -> XPSpectra:
        values = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, Tensor):
                v = v.to(device=device, dtype=dtype if v.is_floating_point() else None)
            values[f.name] = v
        return XPSpectra(**values)

    def truncated(self, truncation: bool = True) -> XPSpectra:
        """GaiaXPy ``truncation=True``: keep the first ``n_relevant`` bases."""
        return replace(self, truncation=truncation)

    @property
    def present(self) -> Tensor:
        """``[B, 2]``: which bands exist."""
        return torch.isfinite(self.coefficients).all(-1)

    # -- linear maps -----------------------------------------------------------
    def _mask(self) -> Tensor | None:
        if not self.truncation or self.n_relevant is None:
            return None
        index = torch.arange(self.n_bases, device=self.device)
        n = self.n_relevant.to(self.device)
        keep = index < n.unsqueeze(-1)
        return torch.where((n > 0).unsqueeze(-1), keep, torch.ones_like(keep))

    def _masked_errors(self) -> Tensor:
        e = torch.nan_to_num(self.errors)
        mask = self._mask()
        if mask is not None:
            e = torch.where(mask, e, 0.0)
        return e

    def _corr_matrices(self) -> Tensor:
        assert self.correlations is not None
        corr = unpack_correlations(torch.nan_to_num(self.correlations), self.n_bases)
        mask = self._mask()
        if mask is not None:
            keep = mask.to(corr.dtype)
            corr = corr * keep.unsqueeze(-1) * keep.unsqueeze(-2) + torch.diag_embed(1.0 - keep)
        return corr

    def _coefficient_factor(self) -> Tensor | None:
        """``diag(e) L`` per band ``[B, 2, N, N]`` (or diagonal), truncation applied."""
        if self.errors is None:
            return None
        e = self._masked_errors()
        if self.correlations is None:
            return torch.diag_embed(e)
        corr = self._corr_matrices()
        chol, info = torch.linalg.cholesky_ex(corr)
        chol = torch.where((info == 0)[..., None, None], chol, torch.nan)
        return e.unsqueeze(-1) * chol

    def _packed_covariance(self) -> Tensor | None:
        """Packed symmetric ``[B, 2, N*(N+1)//2]`` covariance ``[diag, tril(-1)]``."""
        if self.errors is None or self.correlations is None:
            return None
        e = self._masked_errors()
        n = e.shape[-1]
        rows, cols = _tril_indices(n, e.device)
        corr = torch.nan_to_num(self.correlations)
        if corr.dim() == 4:
            corr = corr[..., rows, cols]
        sig_diag = e.square()
        sig_off = e[..., rows] * e[..., cols] * corr
        return torch.cat([sig_diag, sig_off], dim=-1)

    def _band_fluxes(self, design: XPDesign) -> Tensor:
        """Per-band flux ``[B, 2, G]`` (or ``[B, 2]`` for 0D grid)."""
        d = design.to(self.device, self.dtype).design  # [2, (B,) G, N] or [2, N]
        scalar = d.dim() == 2
        if scalar:
            d = d.unsqueeze(1)  # [2, 1, N]
        c = torch.nan_to_num(self.coefficients)
        mask = self._mask()
        if mask is not None:
            c = torch.where(mask, c, 0.0)
        if d.dim() == 3:
            # Shared grid: batched 2-band GEMM [2, B, N] @ [2, N, G].
            dT = d.transpose(-1, -2)
            if dT.device.type == "mps":
                dT = dT.contiguous()
            c_b = c.transpose(0, 1)
            if c_b.device.type == "mps":
                c_b = c_b.contiguous()
            flux = torch.bmm(c_b, dT).transpose(0, 1)
        else:
            d_b = d.transpose(0, 1)  # [B, 2, G, N]
            flux = (d_b @ c.unsqueeze(-1)).squeeze(-1)
        return flux.squeeze(-1) if scalar else flux

    def _band_variances(
        self, design: XPDesign, packed_cov: Tensor | None = None
    ) -> Tensor | None:
        """Exact per-band diagonal variance ``[B, 2, G]`` without Cholesky or ``[B, G, N]``."""
        if self.errors is None:
            return None
        des = design.to(self.device, self.dtype)
        d = des.design
        scalar = d.dim() == 2
        if scalar:
            des = replace(des, design=d.unsqueeze(1), merge=des.merge.unsqueeze(1), _packed=None)
            d = des.design
        if self.correlations is None:
            e2 = self._masked_errors().square()  # [B, 2, N]
            if d.dim() == 3:
                d2T = d.square().transpose(-1, -2)
                e2_b = e2.transpose(0, 1)
                if d2T.device.type == "mps":
                    d2T = d2T.contiguous()
                    e2_b = e2_b.contiguous()
                var_b = torch.bmm(e2_b, d2T).transpose(0, 1)
            else:
                var_b = (d.transpose(0, 1).square() @ e2.unsqueeze(-1)).squeeze(-1)
        elif d.dim() == 3:
            p = des.packed_basis_products()  # [2, N*(N+1)//2, G]
            sig = packed_cov if packed_cov is not None else self._packed_covariance()
            assert p is not None and sig is not None
            sig_b = sig.transpose(0, 1)
            if sig_b.device.type == "mps":
                sig_b = sig_b.contiguous()
            var_b = torch.bmm(sig_b, p).transpose(0, 1)
        else:
            e = self._masked_errors()
            r = self._corr_matrices()
            de = d.transpose(0, 1) * e.unsqueeze(-2)
            var_b = ((de @ r) * de).sum(-1)
        return var_b.squeeze(-1) if scalar else var_b

    def _band_samples(self, design: XPDesign, with_factor: bool):
        """Per-band flux ``[B, 2, G]`` and optional factor ``[B, 2, G, N]``."""
        flux = self._band_fluxes(design)
        factor = None
        if with_factor:
            cf = self._coefficient_factor()
            if cf is not None:
                d = design.to(self.device, self.dtype).design
                scalar = d.dim() == 2
                if scalar:
                    d = d.unsqueeze(1)
                b, _, n, _ = cf.shape
                if d.dim() == 3:
                    g = d.shape[1]
                    cf_r = cf.permute(1, 2, 0, 3).contiguous().reshape(2, n, b * n)
                    factor = (
                        torch.bmm(d.contiguous(), cf_r)
                        .view(2, g, b, n)
                        .permute(2, 0, 1, 3)
                        .contiguous()
                    )
                else:
                    g = d.shape[2]
                    d_r = d.contiguous().reshape(2 * b, g, n)
                    cf_r = cf.transpose(0, 1).contiguous().reshape(2 * b, n, n)
                    factor = torch.bmm(d_r, cf_r).view(2, b, g, n).transpose(0, 1).contiguous()
                if scalar:
                    factor = factor.squeeze(-2)
        return flux, factor

    def _merge_and_blank(
        self,
        design: XPDesign,
        flux_b: Tensor,
        factor_b: Tensor | None = None,
        var_b: Tensor | None = None,
    ) -> Evaluation:
        rel = design.release or self.release
        rp_low = _assets.band_wl_range("rp", rel)[0]
        bp_high = _assets.band_wl_range("bp", rel)[1]
        grid = design.grid.to(self.device, self.dtype)
        scalar = grid.dim() == 0
        w = design.merge.to(self.device, self.dtype)  # [2, (B,) G] or [2]
        if scalar:
            grid = grid.unsqueeze(0)
            w = w.unsqueeze(1)
            flux_b = flux_b.unsqueeze(-1)
            if factor_b is not None:
                factor_b = factor_b.unsqueeze(-2)
            if var_b is not None:
                var_b = var_b.unsqueeze(-1)
        w = w.unsqueeze(1) if w.dim() == 2 else w
        w = w.transpose(0, 1)  # [B|1, 2, G]
        present = self.present
        both = present.all(-1)
        weight = torch.where(both[:, None, None], w, present.to(w.dtype)[:, :, None])
        flux = (weight * flux_b).sum(1)
        only_bp = present[:, 0] & ~present[:, 1]
        only_rp = present[:, 1] & ~present[:, 0]
        blank = (only_rp[:, None] & (grid <= rp_low)) | (only_bp[:, None] & (grid >= bp_high))
        blank = blank | ~present.any(-1, keepdim=True)
        flux = torch.where(blank, torch.nan, flux)
        f = None
        terms = None
        if factor_b is not None:
            f = weight.unsqueeze(-1) * factor_b
            f = torch.cat([f[:, 0], f[:, 1]], dim=-1)
            f = torch.where(blank.unsqueeze(-1), torch.nan, f)
            terms = {id(self.coefficients): f}
        v = None
        if var_b is not None:
            v = (weight.square() * var_b).sum(1)
            v = torch.where(blank, torch.nan, v)
        if scalar:
            flux = flux.squeeze(-1)
            if f is not None:
                f = f.squeeze(-2)
                terms = {id(self.coefficients): f}
            if v is not None:
                v = v.squeeze(-1)
        return Evaluation(flux, f, v, terms)

    def evaluate_design(
        self, design: XPDesign | Tensor | None = None, *, factor: bool = True
    ) -> Evaluation:
        """Merged absolute flux on ``design`` (or wavelengths), GaiaXPy semantics.

        Both bands: blended with the merge weights. One band: that band alone
        (unweighted), NaN where only the missing band would contribute. No band:
        NaN. ``factor`` gives the ``[B, G, 2N]`` covariance factor.
        """
        if design is None:
            design = XPDesign.default(self.device, self.dtype, release=self.release)
        elif not isinstance(design, XPDesign):
            design = XPDesign.absolute(
                _wl(design).to(self.device, self.dtype),
                release=self.release,
                basis_type=self.basis_type,
            )
        if design.kind != "absolute":
            raise ValueError("evaluate_design needs an absolute design; use convert() for internal")
        flux_b, factor_b = self._band_samples(design, factor)
        return self._merge_and_blank(design, flux_b, factor_b=factor_b)

    # Continuous interface: evaluate at arbitrary wavelengths [G] or [B, G].
    def evaluate(self, wavelength: Tensor, uncertainty: bool = True) -> Evaluation:
        wl = _wl(wavelength).to(self.device, self.dtype)
        return self.evaluate_design(wl, factor=uncertainty)

    def band(self, band: str, wavelength) -> Tensor:
        """One band's own reconstruction ``[B, G]`` with no merge and no blanking
        (its bases evaluated everywhere the response is positive). Missing bands
        return NaN."""
        from .basis import _check_band, absolute_design

        b = _check_band(band)
        wl = _wl(wavelength).to(self.device, self.dtype)
        scalar = wl.dim() == 0
        if scalar:
            wl = wl.unsqueeze(0)
        i = 0 if b == "bp" else 1
        d = absolute_design(
            wl, b, weight=torch.ones_like(wl), release=self.release, basis_type=self.basis_type
        )
        c = torch.nan_to_num(self.coefficients[:, i])
        mask = self._mask()
        if mask is not None:
            c = torch.where(mask[:, i], c, 0.0)
        out = (d @ c.unsqueeze(-1)).squeeze(-1) if d.dim() == 3 else _mm(c, d.T)
        present = self.present[:, i].unsqueeze(-1)
        out = torch.where(present, out, torch.nan)
        return out.squeeze(-1) if scalar else out

    # -- GaiaXPy entry points --------------------------------------------------
    def calibrate(
        self,
        sampling=None,
        *,
        with_correlation: bool | str = False,
        basis_type: str | None = None,
    ) -> dict[str, Tensor]:
        """GaiaXPy ``calibrate``: ``wavelength``, ``flux``, ``flux_error`` ``[B, G]``
        (W m^-2 nm^-1); with ``with_correlation="factor"`` adds ``factor`` ``[B, 2N, G]``,
        and with ``with_correlation=True`` also adds dense ``covariance`` ``[B, G, G]``."""
        if isinstance(with_correlation, str) and with_correlation != "factor":
            raise ValueError(f"with_correlation must be bool or 'factor', got {with_correlation!r}")
        bt = (basis_type or self.basis_type).lower()
        design = (
            XPDesign.default(self.device, self.dtype, release=self.release)
            if sampling is None
            else XPDesign.absolute(
                _wl(sampling).to(self.device, self.dtype),
                release=self.release,
                basis_type=bt,
            )
        )
        if sampling is None and bt != "basic":
            raise ValueError("Non-basic inverse basis functions can only be used with explicit sampling")
        if with_correlation:
            ev = self.evaluate_design(design, factor=self.errors is not None)
            out: dict[str, Tensor] = {"wavelength": design.grid, "flux": ev.flux}
            if ev.factor is not None:
                out["flux_error"] = ev.error
                out["factor"] = ev.factor
                if with_correlation != "factor":
                    out["covariance"] = ev.covariance()
            return out

        flux_b = self._band_fluxes(design)
        var_b = self._band_variances(design)
        ev = self._merge_and_blank(design, flux_b, var_b=var_b)
        out = {"wavelength": design.grid, "flux": ev.flux}
        if ev.variance is not None:
            out["flux_error"] = ev.error
        return out

    def convert(
        self,
        sampling=None,
        *,
        with_correlation: bool | str = False,
    ) -> dict[str, Tensor]:
        """GaiaXPy ``convert``: internal flux per band ``[B, 2, G]`` (e-/s) at
        pseudo-wavelengths (default ``linspace(0, 60, 600)``); ``with_correlation="factor"``
        omits dense ``[B, 2, G, G]`` covariance materialisation."""
        if isinstance(with_correlation, str) and with_correlation != "factor":
            raise ValueError(f"with_correlation must be bool or 'factor', got {with_correlation!r}")
        design = XPDesign.internal(
            None if sampling is None else _wl(sampling),
            device=self.device,
            dtype=self.dtype,
            release=self.release,
        )
        present = self.present[..., None]
        if with_correlation:
            flux, factor = self._band_samples(design, self.errors is not None)
            out: dict[str, Tensor] = {
                "pseudo_wavelength": design.grid,
                "flux": torch.where(present, flux, torch.nan),
            }
            if factor is not None:
                factor = torch.where(present[..., None], factor, torch.nan)
                out["flux_error"] = factor.square().sum(-1).clamp_min(0.0).sqrt()
                out["factor"] = factor
                if with_correlation != "factor":
                    out["covariance"] = factor @ factor.transpose(-1, -2)
            return out

        flux = self._band_fluxes(design)
        var_b = self._band_variances(design)
        out = {
            "pseudo_wavelength": design.grid,
            "flux": torch.where(present, flux, torch.nan),
        }
        if var_b is not None:
            var_b = torch.where(present, var_b, torch.nan)
            out["flux_error"] = var_b.clamp_min(0.0).sqrt()
        return out

    def generate(self, systems, **kwargs):
        """GaiaXPy ``generate``; see :func:`torchgaiaxp.photometry.generate`."""
        from .photometry import generate

        return generate(self, systems, **kwargs)

    # -- Cholesky convenience methods ------------------------------------------
    def inverse_square_root_covariance(self, band: str) -> Tensor:
        """Inverse Cholesky factor ``[B, N, N]`` for ``'bp'`` or ``'rp'``."""
        from .basis import _check_band
        from .cholesky import inverse_square_root_covariance

        if self.errors is None or self.correlations is None:
            raise ValueError("errors and correlations are required for inverse covariance")
        i = 0 if _check_band(band) == "bp" else 1
        sd = None if self.standard_deviation is None else self.standard_deviation[:, i]
        return inverse_square_root_covariance(self.errors[:, i], self.correlations[:, i], sd)

    def inverse_covariance(self, band: str) -> Tensor:
        """Inverse coefficient covariance ``[B, N, N]`` for ``'bp'`` or ``'rp'``."""
        w = self.inverse_square_root_covariance(band)
        return w.transpose(-1, -2) @ w

    def chi2(self, band: str, residuals: Tensor) -> Tensor:
        """Batched ``|L^-1 r|^2`` ``[B]`` against ``residuals`` for ``'bp'`` or ``'rp'``."""
        from .cholesky import chi2

        return chi2(self.inverse_square_root_covariance(band), residuals)

    # -- the other direction ---------------------------------------------------
    @classmethod
    def fit(
        cls,
        spectrum: Continuous,
        wavelength=None,
        *,
        rcond: float = 1e-8,
        ridge: float = 0.0,
        device=None,
        dtype=torch.float64,
        release: str = _assets.DEFAULT_RELEASE,
        basis_type: str = "basic",
    ) -> XPSpectra:
        """XP-basis view of any spectrum: least-squares coefficients on ``wavelength``.

        Minimises ``sum_lambda (f(lambda) - sum_b w_b D_b c_b)^2`` over both
        bands, per source (NaN samples are dropped), by SVD. When all sources
        share the same finite-sample mask, a single SVD is computed and shared
        across the whole batch.
        """
        rel = _assets.parse_release(release)
        if wavelength is None:
            bp_lo = _assets.band_wl_range("bp", rel)[0]
            rp_hi = _assets.band_wl_range("rp", rel)[1]
            wavelength = torch.linspace(bp_lo, rp_hi + 30.0, 721, dtype=dtype, device=device)
        design = XPDesign.absolute(
            _wl(wavelength).to(device=device, dtype=dtype),
            release=rel,
            basis_type=basis_type,
        )
        target = spectrum.evaluate(design.grid, uncertainty=False).flux
        if target.dim() == 1:
            target = target.unsqueeze(0)
        d, w = design.design, design.merge
        n_bp, n_rp = d[0].shape[-1], d[1].shape[-1]
        a = torch.cat([w[0].unsqueeze(-1) * d[0], w[1].unsqueeze(-1) * d[1]], dim=-1)  # [G, N_tot]
        m = torch.isfinite(target).to(a.dtype)  # [B, G]
        y = m * torch.nan_to_num(target)

        if target.shape[0] == 1 or bool((m == m[:1]).all()):
            am0 = m[0].unsqueeze(-1) * a  # [G, N_tot]
            u, s, vh = torch.linalg.svd(am0, full_matrices=False)
            smax = s[:1]
            keep = s > rcond * smax
            inv = torch.where(keep, s / (s.square() + ridge * smax.square()), 0.0)
            uty = y @ u  # [B, K]
            c = (inv * uty) @ vh  # [B, N_tot]
        else:
            am = m.unsqueeze(-1) * a  # [B, G, N_tot]
            u, s, vh = torch.linalg.svd(am, full_matrices=False)
            smax = s[..., :1]
            keep = s > rcond * smax
            inv = torch.where(keep, s / (s.square() + ridge * smax.square()), 0.0)
            uty = (u.transpose(-1, -2) @ y.unsqueeze(-1)).squeeze(-1)
            c = (vh.transpose(-1, -2) @ (inv * uty).unsqueeze(-1)).squeeze(-1)

        if n_bp == n_rp:
            coeffs = c.reshape(-1, 2, n_bp)
        else:
            n_max = max(n_bp, n_rp)
            coeffs = c.new_zeros((c.shape[0], 2, n_max))
            coeffs[:, 0, :n_bp] = c[:, :n_bp]
            coeffs[:, 1, :n_rp] = c[:, n_bp:]
        return cls(coefficients=coeffs, release=rel, basis_type=basis_type.lower())
