"""Design matrices: the linear map from coefficients to a sampled quantity.

An :class:`XPDesign` holds, per band, the bases evaluated at some positions
(``design``: ``[2, ..., G, N]``) and GaiaXPy's merge weights (``merge``:
``[2, ..., G]``). Positions may be shared (``[G]``) or per source (``[B, G]``,
e.g. a redshifted rest-frame grid). Four kinds:

* ``"absolute"`` -- externally calibrated flux (W m^-2 nm^-1) at wavelengths;
  built natively in torch for any wavelengths (:func:`XPDesign.absolute`), or
  GaiaXPy's own packaged default grid (:func:`XPDesign.default`);
* ``"internal"`` -- internally calibrated flux at pseudo-wavelengths (convert);
* ``"photometric"`` -- synthetic photometry in one of GaiaXPy's photometric
  systems or a user-supplied filter file (:func:`XPDesign.from_filter`).

Everything is created on the requested device and dtype, and ``.to()`` moves it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import torch
from torch import Tensor

from . import _assets
from .basis import absolute_design, internal_design, merge_weights

_PACKED_DESIGN_CACHE: dict[tuple, Tensor] = {}
_assets.register_cache_clearer(_PACKED_DESIGN_CACHE.clear)


def _stack_padded_designs(mats: list[Tensor]) -> Tensor:
    """Stack per-band designs ``[..., N_k]`` into ``[2, ..., max(N_k)]`` with right zero-pad."""
    n_bp, n_rp = mats[0].shape[-1], mats[1].shape[-1]
    if n_bp == n_rp:
        return torch.stack(mats)
    n_max = max(n_bp, n_rp)
    out = mats[0].new_zeros((2, *mats[0].shape[:-1], n_max))
    out[0, ..., :n_bp] = mats[0]
    out[1, ..., :n_rp] = mats[1]
    return out


def _stack_padded_np(mats: list[np.ndarray]) -> np.ndarray:
    """Stack per-band NumPy filter matrices ``[B_k, N_k]`` into ``[2, B_k, max(N_k)]``."""
    n_bp, n_rp = mats[0].shape[-1], mats[1].shape[-1]
    if n_bp == n_rp:
        return np.ascontiguousarray(np.stack(mats))
    n_max = max(n_bp, n_rp)
    out = np.zeros((2, *mats[0].shape[:-1], n_max), dtype=mats[0].dtype)
    out[0, ..., :n_bp] = mats[0]
    out[1, ..., :n_rp] = mats[1]
    return out


def _tril_diag_indices(n: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    rows, cols = torch.tril_indices(n, n, offset=-1, device=device)
    diag = torch.arange(n, device=device)
    return rows, cols, diag


def _build_packed_design(design: Tensor) -> Tensor:
    """For shared grid ``design [2, G, N]``, build ``[2, N*(N+1)//2, G]`` symmetric basis products."""
    n = design.shape[-1]
    rows, cols, diag = _tril_diag_indices(n, design.device)
    p_diag = design[:, :, diag].square()
    p_off = 2.0 * design[:, :, rows] * design[:, :, cols]
    return torch.cat([p_diag, p_off], dim=-1).transpose(-1, -2).contiguous()


@dataclass
class XPDesign:
    grid: Tensor  # [..., G] nm, pseudo-wavelength, or band index
    design: Tensor  # [2, ..., G, N]
    merge: Tensor  # [2, ..., G]
    kind: str  # absolute | internal | photometric
    system: str | None = None
    bands: tuple[str, ...] | None = None
    release: str = _assets.DEFAULT_RELEASE
    _cache_key: tuple | None = field(default=None, repr=False, compare=False)
    _packed: Tensor | None = field(default=None, repr=False, compare=False)

    # -- construction ---------------------------------------------------------
    @classmethod
    def default(
        cls,
        device=None,
        dtype=torch.float64,
        *,
        release: str = _assets.DEFAULT_RELEASE,
    ) -> XPDesign:
        """GaiaXPy's default ``calibrate`` grid (336-1020 nm, 2 nm for DR3)."""
        rel = _assets.parse_release(release)
        kw = {"device": device, "dtype": dtype, "release": rel}
        grid = _assets.tensor("instrument", "default_grid", **kw)
        design = _assets.tensor("instrument", "default_design", **kw)
        if design.shape[1] != grid.shape[0]:
            design = design.transpose(-1, -2).contiguous()
        merge = _assets.tensor("instrument", "default_merge", **kw)
        key = (rel, "default", str(design.device), design.dtype)
        return cls(grid, design, merge, "absolute", release=rel, _cache_key=key)

    @classmethod
    def absolute(
        cls,
        wavelength,
        device=None,
        dtype=None,
        *,
        release: str = _assets.DEFAULT_RELEASE,
        basis_type: str = "basic",
    ) -> XPDesign:
        """Externally calibrated bases at any wavelengths (nm), natively in torch.

        Equals GaiaXPy's ``calibrate(sampling=wavelength)`` design to ~1e-14.
        Differentiable in ``wavelength``.
        """
        rel = _assets.parse_release(release)
        wl = _as_tensor(wavelength, device, dtype)
        merge = merge_weights(wl, release=rel)
        design = _stack_padded_designs(
            [
                absolute_design(wl, b, weight=merge[i], release=rel, basis_type=basis_type)
                for i, b in enumerate(_assets.BANDS)
            ]
        )
        return cls(wl, design, merge, "absolute", release=rel)

    @classmethod
    def internal(
        cls,
        pseudo_wavelength=None,
        device=None,
        dtype=None,
        *,
        release: str = _assets.DEFAULT_RELEASE,
    ) -> XPDesign:
        """Internally calibrated bases (``convert``); default ``linspace(0, 60, 600)``."""
        rel = _assets.parse_release(release)
        if pseudo_wavelength is None:
            dt = dtype or (torch.float32 if str(device).startswith("mps") else torch.float64)
            pseudo_wavelength = torch.linspace(0, 60, 600, dtype=dt, device=device)
        pwl = _as_tensor(pseudo_wavelength, device, dtype)
        design = _stack_padded_designs([internal_design(pwl, b, release=rel) for b in _assets.BANDS])
        return cls(pwl, design, torch.ones_like(design[..., 0]), "internal", release=rel)

    @classmethod
    def photometric(
        cls,
        system: str,
        device=None,
        dtype=torch.float64,
        *,
        release: str | None = None,
    ) -> XPDesign:
        """Synthetic photometry design for one of GaiaXPy's photometric systems."""
        rel, canonical = _assets.resolve_system_name(system, release=release)
        meta = _assets.systems_meta(rel)
        kw = {"device": device, "dtype": dtype, "release": rel}
        design = _assets.tensor("systems", f"{canonical}/design", **kw)
        merge = _assets.tensor("systems", f"{canonical}/merge", **kw)
        if design.shape[1] != merge.shape[1]:
            design = design.transpose(-1, -2).contiguous()
        grid = torch.arange(design.shape[-2], device=design.device, dtype=dtype)
        key = (rel, "photometric", canonical, str(design.device), design.dtype)
        return cls(
            grid,
            design,
            merge,
            "photometric",
            canonical,
            tuple(meta[canonical]["bands"]),
            release=rel,
            _cache_key=key,
        )

    @classmethod
    def from_filter(
        cls,
        path: str | Path,
        *,
        device=None,
        dtype=torch.float64,
        release: str = _assets.DEFAULT_RELEASE,
    ) -> tuple[XPDesign, dict]:
        """Load a custom GaiaXPy filter file (XML or YAML) -> ``(XPDesign, metadata)``."""
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Filter file not found: {file_path}")
        suffix = file_path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            design, merge, info = _read_filter_yaml(file_path)
        elif suffix == ".xml":
            design, merge, info = _read_filter_xml(file_path)
        else:
            raise ValueError(f"Unsupported filter extension {suffix!r}; expected .xml, .yaml, or .yml")
        d_t = torch.as_tensor(design, dtype=dtype).to(device).contiguous()
        m_t = torch.as_tensor(merge, dtype=dtype).to(device).contiguous()
        grid = torch.arange(d_t.shape[-2], device=d_t.device, dtype=dtype)
        rel = _assets.parse_release(release)
        xp_design = cls(grid, d_t, m_t, "photometric", file_path.stem, tuple(info["bands"]), release=rel)
        return xp_design, info

    def packed_basis_products(self) -> Tensor | None:
        """Cached ``[2, N*(N+1)//2, G]`` symmetric basis products for shared-grid fast variance."""
        if self.design.dim() != 3:
            return None
        if self._cache_key is not None:
            hit = _PACKED_DESIGN_CACHE.get(self._cache_key)
            if hit is None:
                hit = _build_packed_design(self.design)
                _PACKED_DESIGN_CACHE[self._cache_key] = hit
            return hit
        if self._packed is None:
            self._packed = _build_packed_design(self.design)
        return self._packed

    # -- device ---------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self.design.device

    @property
    def dtype(self) -> torch.dtype:
        return self.design.dtype

    def to(self, device=None, dtype=None) -> XPDesign:
        target_dev = self.device if device is None else torch.device(device)
        target_dt = self.dtype if dtype is None else dtype
        if target_dev == self.device and target_dt == self.dtype:
            return self
        return replace(
            self,
            grid=self.grid.to(device=target_dev, dtype=target_dt),
            design=self.design.to(device=target_dev, dtype=target_dt),
            merge=self.merge.to(device=target_dev, dtype=target_dt),
            _cache_key=(
                None
                if self._cache_key is None
                else (*self._cache_key[:-2], str(target_dev), target_dt)
            ),
            _packed=None,
        )


def _as_tensor(x, device=None, dtype=None) -> Tensor:
    if isinstance(x, Tensor):
        if dtype is None and not x.is_floating_point():
            dtype = torch.float32 if (device or x.device).type == "mps" else torch.float64
        return x.to(device=device or x.device, dtype=dtype or x.dtype)
    if dtype is None and device is not None and torch.device(device).type == "mps":
        dtype = torch.float32
    return torch.as_tensor(x, dtype=dtype or torch.float64, device=device)


def _read_filter_xml(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    root = ElementTree.parse(path).getroot()
    bands = [el.text.strip() for el in root.find("bands")]
    zp = np.array([float(el.text) for el in root.find("zeropoints")], dtype=np.float64)
    fb_node = root.find("fluxBias")
    offsets = (
        np.array([float(el.text) for el in fb_node], dtype=np.float64)
        if fb_node is not None
        else np.zeros(len(bands), dtype=np.float64)
    )
    xp_merge = root.find("XpMerge")
    bp_w = np.array([float(el.text) for el in xp_merge.find("bpWeights")], dtype=np.float64)
    rp_w = np.array([float(el.text) for el in xp_merge.find("rpWeights")], dtype=np.float64)
    xp_samp = root.find("XpSampling")
    mats = []
    for b in _assets.BANDS:
        node = xp_samp.find(f"{b}SampledBases")
        dim = int(node.attrib["dimension"])
        vals = np.array([float(el.text) for el in node], dtype=np.float64)
        mats.append(vals.reshape(len(bands), dim))
    design = _stack_padded_np(mats)
    merge = np.ascontiguousarray(np.stack([bp_w, rp_w]))
    info = {
        "bands": bands,
        "zero_points": zp,
        "offsets": offsets,
        "standardised": np.any(offsets != 0),
    }
    return design, merge, info


def _read_filter_yaml(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load .yaml filter files: pip install pyyaml") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = data.get("config", data)

    def _arr(node):
        if isinstance(node, dict):
            if "values" in node:
                return np.asarray(node["values"], dtype=np.float64)
            if "npz_key" in node:
                npz_path = path.parent / cfg["arrayFile"]
                with np.load(npz_path, allow_pickle=False) as z:
                    return np.asarray(z[node["npz_key"]], dtype=np.float64)
        return np.asarray(node, dtype=np.float64)

    bands_node = cfg["bands"]
    bands = list(bands_node["values"] if isinstance(bands_node, dict) else bands_node)
    zp = np.asarray(cfg["zeropoints"], dtype=np.float64)
    offsets = np.asarray(cfg.get("fluxBias", np.zeros(len(bands))), dtype=np.float64)
    bp_w = _arr(cfg["XpMerge"]["bpWeights"])
    rp_w = _arr(cfg["XpMerge"]["rpWeights"])
    mats = []
    for b in _assets.BANDS:
        node = cfg["XpSampling"][f"{b}SampledBases"]
        dim = int(node["dimension"])
        vals = _arr(node)
        mats.append(vals.reshape(len(bands), dim))
    design = _stack_padded_np(mats)
    merge = np.ascontiguousarray(np.stack([bp_w, rp_w]))
    info = {
        "bands": bands,
        "zero_points": zp,
        "offsets": offsets,
        "standardised": np.any(offsets != 0),
    }
    return design, merge, info
