"""Sampled spectrum plotting (``plot_spectra``), optional ``matplotlib`` dependency."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _normalise_plot_inputs(spectra, sampling):
    """Return ``(kind, grid, entries)`` where ``kind in ('absolute', 'internal')``."""
    # Direct XPSpectra convenience: evaluate on provided or default sampling.
    if hasattr(spectra, "calibrate") and hasattr(spectra, "coefficients"):
        data = spectra.calibrate(sampling) if sampling is not None else spectra.calibrate()
        sampling = data["wavelength"]
        spectra = data

    # Tensor dict returned by XPSpectra.calibrate() or XPSpectra.convert()
    if isinstance(spectra, dict):
        if "wavelength" in spectra:
            grid = _to_numpy(spectra["wavelength"] if sampling is None else sampling)
            flux = _to_numpy(spectra["flux"])
            err = _to_numpy(spectra["flux_error"]) if "flux_error" in spectra else None
            entries = [
                (
                    f"source_{i}",
                    flux[i],
                    None if err is None else err[i],
                    None,
                )
                for i in range(flux.shape[0])
            ]
            return "absolute", grid, entries
        if "pseudo_wavelength" in spectra:
            grid = _to_numpy(spectra["pseudo_wavelength"] if sampling is None else sampling)
            flux = _to_numpy(spectra["flux"])
            err = _to_numpy(spectra["flux_error"]) if "flux_error" in spectra else None
            entries = []
            for i in range(flux.shape[0]):
                for k, band in enumerate(("BP", "RP")):
                    entries.append(
                        (
                            f"source_{i}",
                            flux[i, k],
                            None if err is None else err[i, k],
                            band,
                        )
                    )
            return "internal", grid, entries
        raise ValueError("Unrecognised spectrum dict; expected 'wavelength' or 'pseudo_wavelength'")

    # GaiaXPy-style DataFrame
    if sampling is None:
        raise ValueError("sampling must be provided when plotting a DataFrame of sampled spectra")
    grid = _to_numpy(sampling)
    cols = set(spectra.columns)
    data_type = str(getattr(spectra, "attrs", {}).get("data_type", ""))
    is_internal = "xp" in cols or data_type == "XpSampledSpectrum"
    entries = []
    for i, (_, row) in enumerate(spectra.iterrows()):
        sid = row["source_id"] if "source_id" in cols else i
        label = str(sid)
        band = str(row["xp"]).upper() if is_internal and "xp" in cols else None
        raw_f = row["flux"]
        if isinstance(raw_f, (float, np.floating)) and np.isnan(raw_f):
            f_arr = np.full_like(grid, np.nan)
        else:
            f_arr = np.asarray(raw_f, dtype=np.float64)
        e_arr = None
        if "flux_error" in cols:
            raw_e = row["flux_error"]
            if not (isinstance(raw_e, (float, np.floating)) and np.isnan(raw_e)) and raw_e is not None:
                e_arr = np.asarray(raw_e, dtype=np.float64)
        entries.append((label, f_arr, e_arr, band))
    return ("internal" if is_internal else "absolute"), grid, entries


def plot_spectra(
    spectra,
    sampling=None,
    multi: bool = False,
    show_plot: bool = True,
    output_path: str | Path | None = None,
    output_file: str | None = None,
    format: str | None = None,
    legend: bool = True,
    use_tex: bool = False,
):
    """Display or save plots of sampled XP spectra (matches GaiaXPy ``plot_spectra``).

    Accepts a GaiaXPy-style ``pandas.DataFrame``, a ``calibrate()``/``convert()``
    tensor dictionary, or an :class:`~torchgaiaxp.XPSpectra` instance.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot_spectra: pip install 'torchgaiaxp[plot]'"
        ) from exc

    kind, grid, entries = _normalise_plot_inputs(spectra, sampling)
    save_file = output_path is not None or (output_file is not None and output_file != "")
    ext = (format or "png").lstrip(".").lower()
    out_dir = Path("." if output_path is None else output_path)
    if save_file:
        out_dir.mkdir(parents=True, exist_ok=True)
    stem = output_file or ("XpSampledSpectrum" if kind == "internal" else "AbsoluteSampledSpectrum")

    xlabel = "Pseudo-wavelength" if kind == "internal" else "Wavelength [nm]"
    ylabel = "Flux [e-/s]" if kind == "internal" else r"Flux [W m$^{-2}$ nm$^{-1}$]"

    figs = []
    with plt.rc_context({"text.usetex": bool(use_tex)}):
        if multi:
            if kind == "internal":
                fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=False)
                for ax, band in zip(axes, ("BP", "RP")):
                    for label, flux, _err, b in entries:
                        if b == band and np.isfinite(flux).any():
                            ax.plot(grid, flux, label=label)
                    ax.set_ylabel(f"{band} {ylabel}")
                    ax.set_title(band)
                    if legend and entries:
                        ax.legend(loc="best", fontsize="small")
                axes[-1].set_xlabel(xlabel)
            else:
                fig, ax = plt.subplots(figsize=(8, 4.5))
                for label, flux, _err, _b in entries:
                    if np.isfinite(flux).any():
                        ax.plot(grid, flux, label=label)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                if legend and entries:
                    ax.legend(loc="best", fontsize="small")
            fig.tight_layout()
            figs.append(fig)
            if save_file:
                fig.savefig(out_dir / f"{stem}.{ext}")
            if not show_plot:
                plt.close(fig)
        else:
            multi_figs = len(entries) > 1
            for idx, (label, flux, err, band) in enumerate(entries):
                fig, ax = plt.subplots(figsize=(8, 4.5))
                ax.plot(grid, flux, label=label if band is None else f"{label} ({band})")
                if err is not None:
                    ax.fill_between(grid, flux - err, flux + err, alpha=0.25, label=r"$\pm 1\sigma$")
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(label if band is None else f"{label} ({band})")
                if legend:
                    ax.legend(loc="best", fontsize="small")
                fig.tight_layout()
                figs.append(fig)
                if save_file:
                    suffix = f"_{idx}" if multi_figs else ""
                    fig.savefig(out_dir / f"{stem}{suffix}.{ext}")
                if not show_plot:
                    plt.close(fig)
        if show_plot:
            plt.show()
    return figs
