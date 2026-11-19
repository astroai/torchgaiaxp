"""Gaia DR4 epoch/transit-level XP spectra (``xp_epoch_spectrum`` & ``xp_epoch_crowding``)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import Tensor

from .continuous import Sampled, _interp
from .io import _arrow_list_to_numpy, _df_array_col

#: Bits in DR4 ``bp/rp_ccd_proc_flags`` indicating invalid/excluded transit spectra:
#: bit 0 (no data), 1 (unrecoverable truncation), 4 (no calibration), 7 (pre-filtered), 8 (rejected).
DEFAULT_EXCLUDE_PROC_BITS = (1 << 0) | (1 << 1) | (1 << 4) | (1 << 7) | (1 << 8)
CROWDING_ENTRY_BIT = 1 << 13


def decode_transit_id(transit_id: Tensor) -> dict[str, Tensor]:
    """Decode Gaia DR4 ``transit_id`` bitfields (OBMT [ns], FoV [1/2], CCD row [1..7], AF1 pixel)."""
    return {
        "obmt_ns": 204800 * ((transit_id >> 17) & 0x000003FFFFFFFFFF),
        "fov": 1 + ((transit_id >> 15) & 0x03),
        "ccd_row": (transit_id >> 12) & 0x07,
        "af1_ac_pixel": transit_id & 0x0FFF,
    }


def _masked_indices_to_mask(indices_col, shape: tuple[int, ...]) -> np.ndarray:
    """Convert variable-length ``bp/rp_masked_samples`` index lists to boolean ``True=valid`` mask."""
    mask = np.ones(shape, dtype=bool)
    for i, idxs in enumerate(indices_col):
        if idxs is None:
            continue
        if isinstance(idxs, float) and np.isnan(idxs):
            continue
        arr = np.asarray(idxs, dtype=np.int64).reshape(-1)
        valid_idx = arr[(arr >= 0) & (arr < shape[-1])]
        if valid_idx.size:
            mask[i, valid_idx] = False
    return mask


@dataclass
class XPEpochSpectra:
    """``T`` transit-level BP/RP spectra (DR4 ``xp_epoch_spectrum``).

    ``flux`` has shape ``[T, 2, G]`` (per-band BP/RP) or ``[T, G]`` (single/merged).
    ``grid`` has shape ``[G]``, ``[T, G]``, or ``[T, 2, G]`` (per-transit DR4
    ``bp/rp_pseudo_wavelength``). Supports vectorized sample/flag masking,
    pseudo-wavelength resampling, grouping by ``source_id``, and inverse-variance
    coaddition on CPU, CUDA, or MPS.
    """

    flux: Tensor  # [T, 2, G] or [T, G]
    grid: Tensor  # [G], [T, G], or [T, 2, G]
    errors: Tensor | None = None
    source_id: Tensor | None = None  # [T] int64
    transit_id: Tensor | None = None  # [T] int64
    time: Tensor | None = None  # [T] or [T, 2]
    sample_mask: Tensor | None = None  # [T, 2, G] or [T, G] bool (True = valid)
    ccd_proc_flags: Tensor | None = None  # [T, 2] or [T] int
    ccd_info_flags: Tensor | None = None  # [T, 2] or [T] int
    crowding_flag: Tensor | None = None  # [T, 2] or [T] bool (True = crowded)
    quality_mask: Tensor | None = None  # [T, 2] or [T] bool (True = valid)
    release: str = "dr4"

    @property
    def device(self) -> torch.device:
        return self.flux.device

    @property
    def dtype(self) -> torch.dtype:
        return self.flux.dtype

    def __len__(self) -> int:
        return int(self.flux.shape[0])

    def __getitem__(self, index) -> XPEpochSpectra:
        if isinstance(index, int):
            n = len(self)
            if index < 0:
                index += n
            if not 0 <= index < n:
                raise IndexError(f"index {index} out of range for {n} transits")
            index = slice(index, index + 1)
        values = {}
        for f in fields(self):
            v = getattr(self, f.name)
            slice_grid = (
                f.name == "grid"
                and isinstance(v, Tensor)
                and (
                    v.dim() == self.flux.dim()
                    or (
                        self.flux.dim() == 3
                        and v.dim() == 2
                        and v.shape[0] == len(self)
                        and v.shape[0] != 2
                    )
                )
            )
            if isinstance(v, Tensor) and (f.name != "grid" or slice_grid):
                values[f.name] = v[index]
            else:
                values[f.name] = v
        return XPEpochSpectra(**values)

    def to(self, device=None, dtype=None) -> XPEpochSpectra:
        values = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, Tensor):
                v = v.to(device=device, dtype=dtype if v.is_floating_point() else None)
            values[f.name] = v
        return XPEpochSpectra(**values)

    def has_crowding_entry(self) -> Tensor | None:
        """DR4 ``xp_epoch_crowding`` presence per transit/band (bit 13 of ``ccd_proc_flags``)."""
        if self.ccd_proc_flags is None:
            return None
        return (self.ccd_proc_flags & CROWDING_ENTRY_BIT) != 0

    def valid_mask(self, *, exclude_proc_bits: int = DEFAULT_EXCLUDE_PROC_BITS) -> Tensor:
        """Boolean validity mask matching ``self.flux.shape``."""
        valid = torch.isfinite(self.flux)
        if self.errors is not None:
            valid = valid & torch.isfinite(self.errors) & (self.errors > 0)
        if self.sample_mask is not None:
            valid = valid & self.sample_mask
        if self.ccd_proc_flags is not None and exclude_proc_bits:
            pf = (self.ccd_proc_flags & exclude_proc_bits) == 0
            while pf.dim() < valid.dim():
                pf = pf.unsqueeze(-1)
            valid = valid & pf
        if self.crowding_flag is not None:
            cf = self.crowding_flag
            while cf.dim() < valid.dim():
                cf = cf.unsqueeze(-1)
            valid = valid & ~cf
        if self.quality_mask is not None:
            qm = self.quality_mask
            while qm.dim() < valid.dim():
                qm = qm.unsqueeze(-1)
            valid = valid & qm
        return valid

    def resample(self, target_grid: Tensor) -> XPEpochSpectra:
        """Interpolate per-transit ``grid`` onto a shared ``target_grid [G']``."""
        tgt = torch.as_tensor(target_grid, device=self.device, dtype=self.dtype)
        valid = self.valid_mask()
        xp = self.grid.expand_as(self.flux)
        # Mask invalid flux/error samples to NaN before interpolation so neighbors don't bleed.
        f_masked = torch.where(valid, self.flux, torch.nan)
        i, t, inside, _ = _interp(tgt, xp, f_masked)
        fp = f_masked.expand(*i.shape[:-1], f_masked.shape[-1])
        f0, f1 = fp.gather(-1, i - 1), fp.gather(-1, i)
        flux_out = torch.where(inside, f0 + t * (f1 - f0), torch.nan)
        err_out = None
        if self.errors is not None:
            e_masked = torch.where(valid, self.errors, torch.nan)
            ep = e_masked.expand(*i.shape[:-1], e_masked.shape[-1])
            e0, e1 = ep.gather(-1, i - 1), ep.gather(-1, i)
            var_out = torch.where(
                inside, (1.0 - t).square() * e0.square() + t.square() * e1.square(), torch.nan
            )
            err_out = var_out.clamp_min(0.0).sqrt()
        return XPEpochSpectra(
            flux=flux_out,
            grid=tgt,
            errors=err_out,
            source_id=self.source_id,
            transit_id=self.transit_id,
            time=self.time,
            sample_mask=torch.isfinite(flux_out),
            ccd_proc_flags=self.ccd_proc_flags,
            ccd_info_flags=self.ccd_info_flags,
            crowding_flag=self.crowding_flag,
            quality_mask=self.quality_mask,
            release=self.release,
        )

    def group_by_source(self) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor]:
        """Pack irregular transit counts into ``[S, M_max, ...]`` padded tensors.

        Returns ``(unique_source_id, flux_pad, errors_pad, time_pad, valid_pad)``,
        sorted by ``source_id`` (and ``time`` within each source when present).
        """
        if self.source_id is None:
            raise ValueError("source_id is required to group epoch spectra by source")
        sid = self.source_id
        if self.time is not None:
            t_key = self.time[:, 0] if self.time.dim() > 1 else self.time
            order = torch.argsort(t_key, stable=True)
            order = order[torch.argsort(sid[order], stable=True)]
        else:
            order = torch.argsort(sid, stable=True)

        sid_s = sid[order]
        unique_sid, counts = torch.unique_consecutive(sid_s, return_counts=True)
        s_count = unique_sid.numel()
        m_max = int(counts.max().item()) if s_count else 0

        starts = counts.cumsum(0) - counts
        group_idx = torch.repeat_interleave(torch.arange(s_count, device=sid.device), counts)
        within_idx = torch.arange(sid_s.numel(), device=sid.device) - starts[group_idx]

        flux_s = self.flux[order]
        valid_s = self.valid_mask()[order]
        flux_pad = flux_s.new_full((s_count, m_max, *flux_s.shape[1:]), torch.nan)
        flux_pad[group_idx, within_idx] = flux_s

        valid_pad = valid_s.new_zeros((s_count, m_max, *valid_s.shape[1:]))
        valid_pad[group_idx, within_idx] = valid_s

        err_pad = None
        if self.errors is not None:
            err_s = self.errors[order]
            err_pad = err_s.new_full((s_count, m_max, *err_s.shape[1:]), torch.nan)
            err_pad[group_idx, within_idx] = err_s

        time_pad = None
        if self.time is not None:
            time_s = self.time[order]
            time_pad = time_s.new_full((s_count, m_max, *time_s.shape[1:]), torch.nan)
            time_pad[group_idx, within_idx] = time_s

        return unique_sid, flux_pad, err_pad, time_pad, valid_pad

    def mean_flux(
        self,
        *,
        weights: str = "inverse_variance",
        target_grid: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Vectorized per-source coadd ``(unique_source_id, mean_flux, mean_error)``.

        If ``target_grid`` is provided (recommended when DR4 transits have per-epoch
        ``bp/rp_pseudo_wavelength`` grids), transits are resampled onto ``target_grid``
        before coaddition.
        """
        if target_grid is not None:
            return self.resample(target_grid).mean_flux(weights=weights)
        if self.source_id is None:
            raise ValueError("source_id is required to compute per-source mean flux")
        if weights not in ("inverse_variance", "uniform"):
            raise ValueError(f"weights must be 'inverse_variance' or 'uniform', got {weights!r}")
        unique_sid, inv = torch.unique(self.source_id, sorted=True, return_inverse=True)
        valid = self.valid_mask()
        f_clean = torch.where(valid, self.flux, 0.0)
        if weights == "inverse_variance" and self.errors is not None:
            e_safe = torch.where(valid, self.errors, torch.ones_like(self.errors))
            w = torch.where(valid, 1.0 / e_safe.square(), 0.0)
        else:
            w = valid.to(self.flux.dtype)

        s_count = unique_sid.numel()
        tail = self.flux.shape[1:]
        w_sum = self.flux.new_zeros((s_count, *tail))
        wf_sum = self.flux.new_zeros((s_count, *tail))
        idx = inv.view(-1, *([1] * len(tail))).expand_as(w)
        w_sum.scatter_add_(0, idx, w)
        wf_sum.scatter_add_(0, idx, w * f_clean)

        has_data = w_sum > 0
        safe_w = torch.where(has_data, w_sum, torch.ones_like(w_sum))
        mean_f = torch.where(has_data, wf_sum / safe_w, torch.nan)
        mean_e = None
        if self.errors is not None:
            if weights == "inverse_variance":
                mean_e = torch.where(has_data, (1.0 / safe_w).sqrt(), torch.nan)
            else:
                we2_sum = self.flux.new_zeros((s_count, *tail))
                e_clean = torch.where(valid, self.errors, 0.0)
                we2_sum.scatter_add_(0, idx, (w * e_clean).square())
                mean_e = torch.where(has_data, we2_sum.sqrt() / safe_w, torch.nan)
        return unique_sid, mean_f, mean_e

    def to_sampled(
        self, *, weights: str = "inverse_variance", target_grid: Tensor | None = None
    ) -> Sampled:
        """Coadd transits per source (or wrap grid) as :class:`Sampled`."""
        ep = self.resample(target_grid) if target_grid is not None else self
        if ep.flux.dim() == 3:
            raise ValueError("to_sampled expects 2D [T, G] flux; slice a band or pass merged flux")
        if ep.source_id is None:
            valid = ep.valid_mask()
            flux_v = torch.where(valid, ep.flux, torch.nan)
            err_v = None if ep.errors is None else torch.where(valid, ep.errors, torch.nan)
            return Sampled(ep.grid, flux_v, err_v)
        _, mf, me = ep.mean_flux(weights=weights)
        return Sampled(ep.grid, mf, me)

    @classmethod
    def _from_column_store(
        cls,
        cols: set[str],
        n: int,
        get_arr,
        get_pylist,
        get_scalar,
        grid=None,
        *,
        device=None,
        dtype=torch.float64,
        release: str = "dr4",
    ) -> XPEpochSpectra:
        if "bp_flux" in cols and "rp_flux" in cols:
            flux_np = np.stack([get_arr("bp_flux"), get_arr("rp_flux")], axis=1)
            err_np = (
                np.stack([get_arr("bp_flux_error"), get_arr("rp_flux_error")], axis=1)
                if ("bp_flux_error" in cols and "rp_flux_error" in cols)
                else None
            )
            if grid is None and "bp_pseudo_wavelength" in cols and "rp_pseudo_wavelength" in cols:
                grid_np = np.stack(
                    [get_arr("bp_pseudo_wavelength"), get_arr("rp_pseudo_wavelength")],
                    axis=1,
                )
                grid_t = torch.as_tensor(grid_np, dtype=dtype, device=device)
            elif grid is not None:
                grid_t = torch.as_tensor(grid, dtype=dtype, device=device)
            else:
                grid_t = torch.arange(flux_np.shape[-1], dtype=dtype, device=device)
        else:
            flux_np = get_arr("flux")
            err_np = get_arr("flux_error") if "flux_error" in cols else None
            if grid is not None:
                grid_t = torch.as_tensor(grid, dtype=dtype, device=device)
            elif "wavelength" in cols:
                grid_t = torch.as_tensor(get_arr("wavelength"), dtype=dtype, device=device)
            else:
                grid_t = torch.arange(flux_np.shape[-1], dtype=dtype, device=device)

        sample_mask = None
        if "bp_masked_samples" in cols and "rp_masked_samples" in cols:
            m_bp = _masked_indices_to_mask(get_pylist("bp_masked_samples"), (n, flux_np.shape[-1]))
            m_rp = _masked_indices_to_mask(get_pylist("rp_masked_samples"), (n, flux_np.shape[-1]))
            sample_mask = torch.as_tensor(
                np.stack([m_bp, m_rp], axis=1), dtype=torch.bool, device=device
            )

        def _opt(name: str, dt):
            if name not in cols:
                return None
            return torch.as_tensor(np.array(get_scalar(name), copy=True), dtype=dt, device=device)

        def _pair(b_col: str, r_col: str, dt, np_dt):
            if b_col in cols and r_col in cols:
                arr = np.stack(
                    [
                        np.asarray(get_scalar(b_col), dtype=np_dt),
                        np.asarray(get_scalar(r_col), dtype=np_dt),
                    ],
                    axis=1,
                )
                return torch.as_tensor(arr, dtype=dt, device=device)
            return None

        time_t = _pair("bp_obs_time", "rp_obs_time", dtype, np.float64)
        if time_t is None and ("obs_time" in cols or "time" in cols):
            time_t = _opt("obs_time" if "obs_time" in cols else "time", dtype)

        crowd = _pair("bp_crowding_flag", "rp_crowding_flag", torch.bool, bool)
        if crowd is None and "crowding_flag" in cols:
            crowd = _opt("crowding_flag", torch.bool)

        return cls(
            flux=torch.as_tensor(flux_np, dtype=dtype, device=device),
            grid=grid_t,
            errors=None if err_np is None else torch.as_tensor(err_np, dtype=dtype, device=device),
            source_id=_opt("source_id", torch.int64),
            transit_id=_opt("transit_id", torch.int64),
            time=time_t,
            sample_mask=sample_mask,
            ccd_proc_flags=_pair("bp_ccd_proc_flags", "rp_ccd_proc_flags", torch.int64, np.int64),
            ccd_info_flags=_pair("bp_ccd_info_flags", "rp_ccd_info_flags", torch.int64, np.int64),
            crowding_flag=crowd,
            quality_mask=_opt("quality_mask", torch.bool),
            release=release,
        )

    @classmethod
    def from_dataframe(
        cls,
        df,
        grid=None,
        *,
        device=None,
        dtype=torch.float64,
        release: str = "dr4",
    ) -> XPEpochSpectra:
        """Construct from a Gaia DR4 ``xp_epoch_spectrum``-style DataFrame."""
        return cls._from_column_store(
            set(df.columns),
            len(df),
            lambda c: _df_array_col(df[c]),
            lambda c: df[c].tolist(),
            lambda c: df[c].to_numpy(),
            grid=grid,
            device=device,
            dtype=dtype,
            release=release,
        )

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        grid=None,
        *,
        device=None,
        dtype=torch.float64,
        release: str = "dr4",
    ) -> XPEpochSpectra:
        """Construct from a Parquet ``xp_epoch_spectrum`` file."""
        table = pq.read_table(path)
        return cls._from_column_store(
            set(table.column_names),
            table.num_rows,
            lambda c: _arrow_list_to_numpy(table[c]),
            lambda c: table[c].to_pylist(),
            lambda c: table[c].to_numpy(zero_copy_only=False),
            grid=grid,
            device=device,
            dtype=dtype,
            release=release,
        )
