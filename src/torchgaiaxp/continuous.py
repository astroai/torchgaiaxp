"""Continuous spectra you can evaluate anywhere and combine.

A :class:`Continuous` is anything that returns flux at the wavelengths you ask
for. XP spectra (:class:`torchgaiaxp.XPSpectra`) are continuous by nature; a
tabulated spectrum at any resolution becomes one through :class:`Sampled`
(linear interpolation, optionally smoothed to a lower resolution first); a
callable or a tensor becomes one through :class:`Function`/:class:`Constant`.

They combine with ``+ - * /`` (and with scalars, tensors and callables), shift
in redshift, and integrate against passbands. Combination is lazy: nothing is
sampled until you evaluate on a grid, so an XP spectrum times an R ~ 20,000
transmission is evaluated on whatever grid the comparison needs.

Uncertainties propagate to first order through every operation:

* ``factor`` ``[..., G, K]`` -- correlated part (XP: the exact covariance
  factor, ``cov = F F^T``); independent operands concatenate along ``K``;
* ``variance`` ``[..., G]`` -- independent per-point part (a tabulated
  spectrum's errors; interpolation keeps it diagonal, an approximation).

Shapes: a batch of ``B`` spectra evaluated at ``[G]`` or ``[B, G]`` returns
``[B, G]``; an unbatched one returns ``[G]``. Devices: each operand evaluates on
its own device; operands must share one (``.to()`` moves a whole expression).
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Evaluation:
    flux: Tensor
    factor: Tensor | None = None  # [..., G, K]
    variance: Tensor | None = None  # [..., G]

    @property
    def error(self) -> Tensor | None:
        if self.factor is None and self.variance is None:
            return None
        total = 0.0
        if self.factor is not None:
            total = total + self.factor.square().sum(-1)
        if self.variance is not None:
            total = total + self.variance
        return torch.as_tensor(total).sqrt()

    def covariance(self) -> Tensor | None:
        """Dense ``[..., G, G]``: ``F F^T + diag(variance)``."""
        if self.factor is None and self.variance is None:
            return None
        cov = None
        if self.factor is not None:
            cov = self.factor @ self.factor.transpose(-1, -2)
        if self.variance is not None:
            diag = torch.diag_embed(self.variance)
            cov = diag if cov is None else cov + diag
        return cov


class Continuous:
    """Base class: implement :meth:`evaluate`; everything else follows."""

    # -- to implement ----------------------------------------------------------
    def evaluate(self, wavelength: Tensor, uncertainty: bool = True) -> Evaluation:
        """Flux (and, with ``uncertainty``, its factor/variance) at ``wavelength``."""
        raise NotImplementedError  # pragma: no cover

    def to(self, device=None, dtype=None) -> Continuous:  # pragma: no cover
        raise NotImplementedError

    # -- derived ---------------------------------------------------------------
    def __call__(self, wavelength) -> Tensor:
        return self.evaluate(_wl(wavelength), uncertainty=False).flux

    def sample(self, wavelength) -> Sampled:
        """Freeze onto a grid (keeps flux and the diagonal error)."""
        wl = _wl(wavelength)
        ev = self.evaluate(wl)
        err = ev.error
        return Sampled(wl, ev.flux, err)

    def shift(self, z) -> Continuous:
        """Observed-frame view of a rest-frame spectrum: ``f(lambda / (1 + z))``.

        ``z`` scalar or ``[B]`` (one per spectrum). The flux density is not
        rescaled: multiply by ``1 / (1 + z)`` yourself if you need f_lambda
        conservation.
        """
        return Shifted(self, z)

    def integrate(self, passband: Continuous, wavelength, *, photon: bool = True):
        """Passband-averaged flux ``int f T w / int T w`` on ``wavelength``.

        ``w = lambda`` for photon-counting detectors (``photon=True``), else 1.
        Returns ``(flux, error)`` with shapes ``[...]``; error is ``None`` when
        the spectrum carries no uncertainty.
        """
        wl = _wl(wavelength)
        trans = passband.evaluate(wl, uncertainty=False).flux
        weight = trans * (wl if photon else torch.ones_like(wl))
        norm = torch.trapezoid(weight, wl, dim=-1)
        ev = self.evaluate(wl)
        flux = torch.trapezoid(torch.nan_to_num(ev.flux) * weight, wl, dim=-1) / norm
        var = None
        q = _trapezoid_weights(wl) * weight / norm.unsqueeze(-1)  # [..., G]
        if ev.factor is not None:
            f = (q.unsqueeze(-1) * torch.nan_to_num(ev.factor)).sum(-2)
            var = f.square().sum(-1)
        if ev.variance is not None:
            v = (q.square() * torch.nan_to_num(ev.variance)).sum(-1)
            var = v if var is None else var + v
        return flux, None if var is None else var.sqrt()

    # -- algebra ---------------------------------------------------------------
    def __add__(self, other):
        return Combined(operator.add, self, as_continuous(other))

    def __radd__(self, other):
        return Combined(operator.add, as_continuous(other), self)

    def __sub__(self, other):
        return Combined(operator.sub, self, as_continuous(other))

    def __rsub__(self, other):
        return Combined(operator.sub, as_continuous(other), self)

    def __mul__(self, other):
        return Combined(operator.mul, self, as_continuous(other))

    def __rmul__(self, other):
        return Combined(operator.mul, as_continuous(other), self)

    def __truediv__(self, other):
        return Combined(operator.truediv, self, as_continuous(other))

    def __rtruediv__(self, other):
        return Combined(operator.truediv, as_continuous(other), self)

    def __neg__(self):
        return Combined(operator.mul, Constant(-1.0), self)


def as_continuous(x) -> Continuous:
    if isinstance(x, Continuous):
        return x
    if callable(x):
        return Function(x)
    return Constant(x)


def _wl(wavelength) -> Tensor:
    if isinstance(wavelength, Tensor):
        return wavelength if wavelength.is_floating_point() else wavelength.double()
    return torch.as_tensor(wavelength, dtype=torch.float64)


def _trapezoid_weights(x: Tensor) -> Tensor:
    dx = x.diff(dim=-1)
    w = torch.zeros_like(x)
    w[..., :-1] += 0.5 * dx
    w[..., 1:] += 0.5 * dx
    return w


def _scale_factor(factor: Tensor | None, scale: Tensor) -> Tensor | None:
    return None if factor is None else factor * scale.unsqueeze(-1)


def _cat_factors(*factors: Tensor | None) -> Tensor | None:
    present = [f for f in factors if f is not None]
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    shape = torch.broadcast_shapes(*(f.shape[:-1] for f in present))
    return torch.cat([f.expand(*shape, f.shape[-1]) for f in present], dim=-1)


def _add_var(*vs: Tensor | None) -> Tensor | None:
    present = [v for v in vs if v is not None]
    if not present:
        return None
    out = present[0]
    for v in present[1:]:
        out = out + v
    return out


class Combined(Continuous):
    """``op(a, b)`` with first-order propagation of both operands' uncertainty."""

    def __init__(self, op: Callable, a: Continuous, b: Continuous):
        self.op, self.a, self.b = op, a, b

    def evaluate(self, wavelength: Tensor, uncertainty: bool = True) -> Evaluation:
        ea = self.a.evaluate(wavelength, uncertainty)
        eb = self.b.evaluate(wavelength, uncertainty)
        fa, fb = ea.flux, eb.flux
        if self.op is operator.add or self.op is operator.sub:
            sign = 1.0 if self.op is operator.add else -1.0
            return Evaluation(
                self.op(fa, fb),
                _cat_factors(ea.factor, None if eb.factor is None else sign * eb.factor),
                _add_var(ea.variance, eb.variance),
            )
        if self.op is operator.mul:
            return Evaluation(
                fa * fb,
                _cat_factors(_scale_factor(ea.factor, fb), _scale_factor(eb.factor, fa)),
                _add_var(
                    None if ea.variance is None else ea.variance * fb.square(),
                    None if eb.variance is None else eb.variance * fa.square(),
                ),
            )
        if self.op is operator.truediv:
            inv = 1.0 / fb
            return Evaluation(
                fa * inv,
                _cat_factors(
                    _scale_factor(ea.factor, inv),
                    _scale_factor(eb.factor, -fa * inv.square()),
                ),
                _add_var(
                    None if ea.variance is None else ea.variance * inv.square(),
                    None if eb.variance is None else eb.variance * (fa * inv.square()).square(),
                ),
            )
        raise ValueError(f"unsupported operator {self.op}")

    def to(self, device=None, dtype=None) -> Combined:
        return Combined(self.op, self.a.to(device, dtype), self.b.to(device, dtype))


class Constant(Continuous):
    """A scalar or tensor, constant in wavelength. ``[B]`` broadcasts per spectrum."""

    def __init__(self, value):
        self.value = torch.as_tensor(value)

    def evaluate(self, wavelength: Tensor, uncertainty: bool = True) -> Evaluation:
        v = self.value.to(device=wavelength.device, dtype=wavelength.dtype)
        if v.dim() >= 1:
            v = v.unsqueeze(-1)  # [B] -> [B, 1]: one value per spectrum
        return Evaluation(v.expand(*torch.broadcast_shapes(v.shape, wavelength.shape)))

    def to(self, device=None, dtype=None) -> Constant:
        return Constant(self.value.to(device=device, dtype=dtype))


class Function(Continuous):
    """Any ``f(wavelength) -> flux`` callable (torch ops keep it differentiable)."""

    def __init__(self, fn: Callable[[Tensor], Tensor]):
        self.fn = fn

    def evaluate(self, wavelength: Tensor, uncertainty: bool = True) -> Evaluation:
        return Evaluation(torch.as_tensor(self.fn(wavelength)))

    def to(self, device=None, dtype=None) -> Function:
        return self


class Shifted(Continuous):
    def __init__(self, base: Continuous, z):
        self.base, self.z = base, torch.as_tensor(z)

    def evaluate(self, wavelength: Tensor, uncertainty: bool = True) -> Evaluation:
        z = self.z.to(device=wavelength.device, dtype=wavelength.dtype)
        if z.dim() >= 1:
            z = z.unsqueeze(-1)  # [B, 1]: per-spectrum shift
        return self.base.evaluate(wavelength / (1.0 + z), uncertainty)

    def to(self, device=None, dtype=None) -> Shifted:
        return Shifted(self.base.to(device, dtype), self.z.to(device=device, dtype=dtype))


def _interp(x: Tensor, xp: Tensor, fp: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Batched linear interpolation indices/weights: ``x [..., G]``, ``xp [..., N]``."""
    batch = torch.broadcast_shapes(x.shape[:-1], xp.shape[:-1], fp.shape[:-1])
    xp_b = xp.expand(*batch, xp.shape[-1]).contiguous()
    x_b = x.expand(*batch, x.shape[-1]).contiguous()
    i = torch.searchsorted(xp_b.detach(), x_b.detach()).clamp(1, xp.shape[-1] - 1)
    x0, x1 = xp_b.gather(-1, i - 1), xp_b.gather(-1, i)
    t = (x_b - x0) / (x1 - x0)
    inside = (x_b >= xp_b[..., :1]) & (x_b <= xp_b[..., -1:])
    return i, t, inside, x_b


class Sampled(Continuous):
    """A tabulated spectrum at any resolution: ``wavelength [N]`` or ``[B, N]``.

    Evaluates by linear interpolation, ``fill`` (default NaN) outside its range.
    ``error`` (optional) propagates as an independent per-point variance.
    """

    def __init__(self, wavelength, flux, error=None, *, fill: float = math.nan):
        self.wavelength = _wl(wavelength)
        self.flux = torch.as_tensor(flux).to(self.wavelength)
        self.error = None if error is None else torch.as_tensor(error).to(self.wavelength)
        self.fill = fill

    def evaluate(self, wavelength: Tensor, uncertainty: bool = True) -> Evaluation:
        wl = wavelength.to(self.wavelength)
        i, t, inside, _ = _interp(wl, self.wavelength, self.flux)
        batch = i.shape[:-1]
        fp = self.flux.expand(*batch, self.flux.shape[-1])
        f0, f1 = fp.gather(-1, i - 1), fp.gather(-1, i)
        flux = torch.where(inside, f0 + t * (f1 - f0), self.fill)
        var = None
        if uncertainty and self.error is not None:
            ep = self.error.expand(*batch, self.error.shape[-1])
            e0, e1 = ep.gather(-1, i - 1), ep.gather(-1, i)
            var = torch.where(
                inside, (1 - t).square() * e0.square() + t.square() * e1.square(), math.nan
            )
        return Evaluation(flux, None, var)

    def smooth(self, resolution: float, *, oversample: int = 2) -> Sampled:
        """Convolve with a Gaussian LSF of resolving power ``R = lambda / FWHM``.

        Resamples onto a log-uniform grid (``oversample`` points per FWHM at
        least, never coarser than the input) and returns that grid. Use it to
        bring a high-resolution spectrum to XP (R ~ 30-100) or RVS (~11,500).
        """
        wl = self.wavelength
        lo, hi = wl[..., 0].min(), wl[..., -1].max()
        dlnl_in = torch.diff(torch.log(wl), dim=-1).min()
        sigma = 1.0 / (resolution * 2.0 * math.sqrt(2.0 * math.log(2.0)))
        step = min(float(dlnl_in), sigma * 2.3548 / oversample)
        n = math.ceil(float(torch.log(hi / lo)) / step) + 1
        grid = torch.exp(torch.linspace(float(torch.log(lo)), float(torch.log(hi)), n)).to(wl)
        ev = self.evaluate(grid)
        half = math.ceil(4 * sigma / step)
        k = torch.exp(-0.5 * (torch.arange(-half, half + 1).to(wl) * step / sigma) ** 2)
        k = k / k.sum()
        flux = _convolve(torch.nan_to_num(ev.flux), k)
        good = _convolve(torch.isfinite(ev.flux).to(wl), k)
        flux = torch.where(good > 0.99, flux / good, math.nan)
        err = None
        if ev.variance is not None:
            var = _convolve(torch.nan_to_num(ev.variance), k.square())
            err = torch.where(good > 0.99, var.sqrt(), math.nan)
        return Sampled(grid, flux, err, fill=self.fill)

    def to(self, device=None, dtype=None) -> Sampled:
        return Sampled(
            self.wavelength.to(device=device, dtype=dtype),
            self.flux.to(device=device, dtype=dtype),
            None if self.error is None else self.error.to(device=device, dtype=dtype),
            fill=self.fill,
        )


def _convolve(y: Tensor, kernel: Tensor) -> Tensor:
    shape = y.shape
    flat = y.reshape(-1, 1, shape[-1])
    out = torch.nn.functional.conv1d(
        flat, kernel.flip(0).view(1, 1, -1), padding=kernel.numel() // 2
    )
    return out.reshape(shape)
