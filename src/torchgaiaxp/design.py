"""Design matrices: the linear map from coefficients to a sampled quantity.

An :class:`XPDesign` holds, per band, the bases evaluated at some positions
(``design``: ``[2, ..., G, 55]``) and GaiaXPy's merge weights (``merge``:
``[2, ..., G]``). Positions may be shared (``[G]``) or per source (``[B, G]``,
e.g. a redshifted rest-frame grid). Four kinds:

* ``"absolute"`` -- externally calibrated flux (W m^-2 nm^-1) at wavelengths;
  built natively in torch for any wavelengths (:func:`XPDesign.absolute`), or
  GaiaXPy's own packaged default grid (:func:`XPDesign.default`);
* ``"internal"`` -- internally calibrated flux at pseudo-wavelengths (convert);
* ``"photometric"`` -- synthetic photometry in one of GaiaXPy's 28 systems.

Everything is created on the requested device and dtype, and ``.to()`` moves it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor

from . import _assets
from .basis import absolute_design, internal_design, merge_weights


@dataclass
class XPDesign:
    grid: Tensor  # [..., G] nm, pseudo-wavelength, or band index
    design: Tensor  # [2, ..., G, 55]
    merge: Tensor  # [2, ..., G]
    kind: str  # absolute | internal | photometric
    system: str | None = None
    bands: tuple[str, ...] | None = None

    # -- construction ---------------------------------------------------------
    @classmethod
    def default(cls, device=None, dtype=torch.float64) -> XPDesign:
        """GaiaXPy's default ``calibrate`` grid (336-1020 nm, 2 nm), from its XML.

        DPAC's packaged matrix differs from the instrument model evaluated on the
        same grid by up to 0.3% of the peak in BP (GaiaXPy's own two paths
        disagree there); this one reproduces ``calibrate(sampling=None)``.
        """
        kw = {"device": device, "dtype": dtype}
        grid = _assets.tensor("instrument", "default_grid", **kw)
        design = _assets.tensor("instrument", "default_design", **kw).transpose(-1, -2).contiguous()
        merge = _assets.tensor("instrument", "default_merge", **kw)
        return cls(grid, design, merge, "absolute")

    @classmethod
    def absolute(cls, wavelength, device=None, dtype=None) -> XPDesign:
        """Externally calibrated bases at any wavelengths (nm), natively in torch.

        Equals GaiaXPy's ``calibrate(sampling=wavelength)`` design to ~1e-14.
        Differentiable in ``wavelength``.
        """
        wl = _as_tensor(wavelength, device, dtype)
        merge = merge_weights(wl)
        design = torch.stack(
            [absolute_design(wl, b, weight=merge[i]) for i, b in enumerate(("bp", "rp"))]
        )
        return cls(wl, design, merge, "absolute")

    @classmethod
    def internal(cls, pseudo_wavelength=None, device=None, dtype=None) -> XPDesign:
        """Internally calibrated bases (``convert``); default ``linspace(0, 60, 600)``."""
        if pseudo_wavelength is None:
            pseudo_wavelength = torch.linspace(
                0, 60, 600, dtype=dtype or torch.float64, device=device
            )
        pwl = _as_tensor(pseudo_wavelength, device, dtype)
        design = torch.stack([internal_design(pwl, b) for b in ("bp", "rp")])
        return cls(pwl, design, torch.ones_like(design[..., 0]), "internal")

    @classmethod
    def photometric(cls, system: str, device=None, dtype=torch.float64) -> XPDesign:
        """Synthetic photometry design for one of GaiaXPy's photometric systems."""
        meta = _assets.systems_meta()
        if system not in meta or system.startswith("_"):
            known = sorted(k for k in meta if not k.startswith("_"))
            raise KeyError(f"unknown photometric system {system!r}; known: {known}")
        kw = {"device": device, "dtype": dtype}
        design = _assets.tensor("systems", f"{system}/design", **kw).transpose(-1, -2).contiguous()
        merge = _assets.tensor("systems", f"{system}/merge", **kw)
        grid = torch.arange(design.shape[-2], device=design.device, dtype=dtype)
        return cls(grid, design, merge, "photometric", system, tuple(meta[system]["bands"]))

    # -- device ---------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self.design.device

    @property
    def dtype(self) -> torch.dtype:
        return self.design.dtype

    def to(self, device=None, dtype=None) -> XPDesign:
        return replace(
            self,
            grid=self.grid.to(device=device, dtype=dtype),
            design=self.design.to(device=device, dtype=dtype),
            merge=self.merge.to(device=device, dtype=dtype),
        )


def _as_tensor(x, device=None, dtype=None) -> Tensor:
    if isinstance(x, Tensor):
        if dtype is None and not x.is_floating_point():
            dtype = torch.float64
        return x.to(device=device or x.device, dtype=dtype or x.dtype)
    return torch.as_tensor(x, dtype=dtype or torch.float64, device=device)
