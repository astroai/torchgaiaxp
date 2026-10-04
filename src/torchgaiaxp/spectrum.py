"""Batches of Gaia XP mean spectra in their continuous representation.

:class:`XPSpectra` holds what the archive delivers per source and band
(coefficients, formal errors, correlations, standard deviation, number of
relevant bases) as tensors on one device, and is a :class:`Continuous`: call it
at any wavelengths, combine it with other spectra, shift it, integrate it.

GaiaXPy parity: :meth:`calibrate` (absolute sampled spectra), :meth:`convert`
(internal), :meth:`generate` (synthetic photometry, see
:mod:`torchgaiaxp.photometry`), and the Cholesky products
(:mod:`torchgaiaxp.cholesky`).

The merged covariance (``factor``, ``covariance``) is ``F F^T`` with
``F = [w_bp D_bp diag(e_bp) L_bp, w_rp D_rp diag(e_rp) L_rp]``: exact, and its
diagonal equals GaiaXPy's errors. GaiaXPy's ``with_correlation`` merges band
covariances with the weights applied once (``w C``, not ``w C w``), which differs
inside the 635-643 nm blend; this package uses the consistent form.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np
import torch
from torch import Tensor

from .continuous import Continuous, Evaluation, _wl
from .design import XPDesign

N_BASES = 55
BP_HIGH, RP_LOW = 643.0, 635.0


def unpack_correlations(packed: Tensor, n: int = N_BASES) -> Tensor:
    """Archive packed correlations ``[..., n(n-1)/2]`` -> ``[..., n, n]``.

    The strictly lower triangle in ``np.tril_indices(n, -1)`` order; diagonal 1.
    A ``[..., n, n]`` input is returned unchanged.
    """
    if packed.dim() >= 2 and packed.shape[-1] == packed.shape[-2] == n:
        return packed
    rows, cols = torch.tril_indices(n, n, offset=-1, device=packed.device)
    if packed.shape[-1] != rows.numel():
        raise ValueError(f"expected {rows.numel()} packed values, got {packed.shape[-1]}")
    matrix = packed.new_zeros((*packed.shape[:-1], n, n))
    matrix[..., rows, cols] = packed
    matrix = matrix + matrix.transpose(-1, -2)
    matrix.diagonal(dim1=-2, dim2=-1).fill_(1.0)
    return matrix


@dataclass
class XPSpectra(Continuous):
    """``B`` sources. Per-band tensors are ``[B, 2, ...]`` with band order (BP, RP).

    A missing band is NaN in ``coefficients``. ``correlations`` may be packed
    ``[B, 2, 1485]`` or full ``[B, 2, 55, 55]``.
    """

    coefficients: Tensor
    errors: Tensor | None = None
    correlations: Tensor | None = None
    standard_deviation: Tensor | None = None  # [B, 2]
    n_relevant: Tensor | None = None  # [B, 2]
    source_id: Tensor | None = None  # [B] int64
    truncation: bool = False

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_batch(cls, batch, device=None, dtype=torch.float64, truncation=False) -> XPSpectra:
        """From :class:`torchgaiaxp.io.XPBatch` (or anything with its fields)."""

        def t(x, kind=dtype):
            if x is None:
                return None
            x = x if isinstance(x, torch.Tensor) else np.array(x)  # writable copy
            return torch.as_tensor(x).to(device=device, dtype=kind)

        return cls(
            coefficients=t(batch.coefficients),
            errors=t(batch.errors),
            correlations=t(batch.correlations),
            standard_deviation=t(batch.standard_deviation),
            n_relevant=t(batch.n_relevant, torch.int64),
            source_id=t(batch.source_id, torch.int64),
            truncation=truncation,
        )

    # -- basics ---------------------------------------------------------------
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
        index = torch.arange(self.coefficients.shape[-1], device=self.device)
        n = self.n_relevant.to(self.device)
        keep = index < n.unsqueeze(-1)
        return torch.where((n > 0).unsqueeze(-1), keep, torch.ones_like(keep))

    def _coefficient_factor(self) -> Tensor | None:
        """``diag(e) L`` per band ``[B, 2, 55, 55]`` (or diagonal), truncation applied."""
        if self.errors is None:
            return None
        mask = self._mask()
        e = torch.nan_to_num(self.errors)
        if mask is not None:
            e = torch.where(mask, e, 0.0)
        if self.correlations is None:
            return torch.diag_embed(e)
        corr = unpack_correlations(torch.nan_to_num(self.correlations), e.shape[-1])
        if mask is not None:
            keep = mask.to(corr.dtype)
            corr = corr * keep.unsqueeze(-1) * keep.unsqueeze(-2) + torch.diag_embed(1.0 - keep)
        chol, info = torch.linalg.cholesky_ex(corr)
        chol = torch.where((info == 0)[..., None, None], chol, torch.nan)
        return e.unsqueeze(-1) * chol

    def _band_samples(self, design: XPDesign, with_factor: bool):
        """Per-band flux ``[B, 2, G]`` and factor ``[B, 2, G, 55]``."""
        d = design.to(self.device, self.dtype).design  # [2, (B,) G, 55]
        if d.dim() == 3:
            d = d.unsqueeze(1)  # [2, 1, G, 55]
        d = d.transpose(0, 1)  # [B|1, 2, G, 55]
        c = torch.nan_to_num(self.coefficients)
        mask = self._mask()
        if mask is not None:
            c = torch.where(mask, c, 0.0)
        flux = (d @ c.unsqueeze(-1)).squeeze(-1)
        factor = None
        if with_factor:
            cf = self._coefficient_factor()
            if cf is not None:
                factor = d @ cf
        return flux, factor

    def evaluate_design(
        self, design: XPDesign | Tensor | None = None, *, factor: bool = True
    ) -> Evaluation:
        """Merged absolute flux on ``design`` (or wavelengths), GaiaXPy semantics.

        Both bands: blended with the merge weights. One band: that band alone
        (unweighted), NaN where only the missing band would contribute. No band:
        NaN. ``factor`` gives the ``[B, G, 110]`` covariance factor.
        """
        if design is None:
            design = XPDesign.default(self.device, self.dtype)
        elif not isinstance(design, XPDesign):
            design = XPDesign.absolute(_wl(design).to(self.device, self.dtype))
        if design.kind != "absolute":
            raise ValueError("evaluate_design needs an absolute design; use convert() for internal")
        flux_b, factor_b = self._band_samples(design, factor)
        w = design.merge.to(self.device, self.dtype)  # [2, (B,) G]
        w = w.unsqueeze(1) if w.dim() == 2 else w
        w = w.transpose(0, 1)  # [B|1, 2, G]
        present = self.present
        both = present.all(-1)
        weight = torch.where(both[:, None, None], w, present.to(w.dtype)[:, :, None])
        flux = (weight * flux_b).sum(1)
        grid = design.grid.to(self.device, self.dtype)
        only_bp = present[:, 0] & ~present[:, 1]
        only_rp = present[:, 1] & ~present[:, 0]
        blank = (only_rp[:, None] & (grid <= RP_LOW)) | (only_bp[:, None] & (grid >= BP_HIGH))
        blank = blank | ~present.any(-1, keepdim=True)
        f = None
        if factor_b is not None:
            f = weight.unsqueeze(-1) * factor_b
            f = torch.cat([f[:, 0], f[:, 1]], dim=-1)
            f = torch.where(blank.unsqueeze(-1), torch.nan, f)
        return Evaluation(torch.where(blank, torch.nan, flux), f)

    # Continuous interface: evaluate at arbitrary wavelengths [G] or [B, G].
    def evaluate(self, wavelength: Tensor, uncertainty: bool = True) -> Evaluation:
        wl = _wl(wavelength).to(self.device, self.dtype)
        return self.evaluate_design(wl, factor=uncertainty)

    def band(self, band: str, wavelength) -> Tensor:
        """One band's own reconstruction ``[B, G]`` with no merge and no blanking
        (its bases evaluated everywhere the response is positive)."""
        from .basis import absolute_design

        wl = _wl(wavelength).to(self.device, self.dtype)
        i = 0 if band == "bp" else 1
        d = absolute_design(wl, band, weight=torch.ones_like(wl))
        c = torch.nan_to_num(self.coefficients[:, i])
        mask = self._mask()
        if mask is not None:
            c = torch.where(mask[:, i], c, 0.0)
        return (d @ c.unsqueeze(-1)).squeeze(-1) if d.dim() == 3 else c @ d.T

    # -- GaiaXPy entry points --------------------------------------------------
    def calibrate(self, sampling=None, *, with_correlation: bool = False) -> dict[str, Tensor]:
        """GaiaXPy ``calibrate``: ``wavelength``, ``flux``, ``flux_error`` ``[B, G]``
        (W m^-2 nm^-1), and with ``with_correlation`` the ``factor`` and
        ``covariance`` (consistent merge, see module docstring)."""
        design = (
            XPDesign.default(self.device, self.dtype)
            if sampling is None
            else XPDesign.absolute(_wl(sampling).to(self.device, self.dtype))
        )
        ev = self.evaluate_design(design, factor=self.errors is not None)
        out = {"wavelength": design.grid, "flux": ev.flux}
        if ev.factor is not None:
            out["flux_error"] = ev.error
            if with_correlation:
                out["factor"] = ev.factor
                out["covariance"] = ev.covariance()
        return out

    def convert(self, sampling=None, *, with_correlation: bool = False) -> dict[str, Tensor]:
        """GaiaXPy ``convert``: internal flux per band ``[B, 2, G]`` (e-/s) at
        pseudo-wavelengths (default ``linspace(0, 60, 600)``)."""
        design = XPDesign.internal(
            None if sampling is None else _wl(sampling), device=self.device, dtype=self.dtype
        )
        flux, factor = self._band_samples(design, self.errors is not None)
        present = self.present[..., None]
        out = {"pseudo_wavelength": design.grid, "flux": torch.where(present, flux, torch.nan)}
        if factor is not None:
            factor = torch.where(present[..., None], factor, torch.nan)
            out["flux_error"] = factor.square().sum(-1).sqrt()
            if with_correlation:
                out["factor"] = factor
                out["covariance"] = factor @ factor.transpose(-1, -2)
        return out

    def generate(self, systems, **kwargs):
        """GaiaXPy ``generate``; see :func:`torchgaiaxp.photometry.generate`."""
        from .photometry import generate

        return generate(self, systems, **kwargs)

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
    ) -> XPSpectra:
        """XP-basis view of any spectrum: least-squares coefficients on ``wavelength``.

        Minimises ``sum_lambda (f(lambda) - sum_b w_b D_b c_b)^2`` over both
        bands, per source (NaN samples are dropped), by a batched SVD:
        singular values below ``rcond`` x the largest are discarded (the
        default keeps coefficients O(1) so float32 evaluation stays at ~1e-4;
        1e-12 reproduces an XP spectrum to ~1e-7 with wild coefficients), and
        ``ridge`` (relative to the largest squared singular value) adds Tikhonov
        damping. The merged absolute bases are badly conditioned (~1e15: high-
        order bases carry almost no flux), so coefficients are not unique; the
        reconstructed flux is.

        This is a projection onto the span of the XP absolute bases -- what a
        spectrum looks like at XP resolution and band limit -- not DPAC's forward
        model (its LSF and calibration kernels are not public in GaiaXPy).
        """
        if wavelength is None:
            wavelength = torch.linspace(330.0, 1050.0, 721, dtype=dtype, device=device)
        design = XPDesign.absolute(_wl(wavelength).to(device=device, dtype=dtype))
        target = spectrum.evaluate(design.grid, uncertainty=False).flux
        if target.dim() == 1:
            target = target.unsqueeze(0)
        d, w = design.design, design.merge
        a = torch.cat([w[0].unsqueeze(-1) * d[0], w[1].unsqueeze(-1) * d[1]], dim=-1)  # [G, 110]
        m = torch.isfinite(target).to(a.dtype)  # [B, G]
        am = m.unsqueeze(-1) * a  # [B, G, 110]
        u, s, vh = torch.linalg.svd(am, full_matrices=False)
        smax = s[..., :1]
        keep = s > rcond * smax
        inv = torch.where(keep, s / (s.square() + ridge * smax.square()), 0.0)
        uty = (u.transpose(-1, -2) @ (m * torch.nan_to_num(target)).unsqueeze(-1)).squeeze(-1)
        c = (vh.transpose(-1, -2) @ (inv * uty).unsqueeze(-1)).squeeze(-1)
        return cls(coefficients=c.reshape(-1, 2, N_BASES))
