"""Packaged GaiaXPy configuration per data release, cached per device and dtype."""

from __future__ import annotations

import json
from functools import cache
from importlib import resources
from pathlib import Path

import numpy as np
import torch

BANDS = ("bp", "rp")
DEFAULT_RELEASE = "dr3"
BP_WL = (330.0, 643.0)
RP_WL = (635.0, 1020.0)
TELESCOPE_PUPIL_AREA = 0.7278  # m^2
#: hc in W s nm (GaiaXPy 2.2+: SI 2019 Planck constant 6.62607015e-34).
HC = 1.0e9 * 2.99792458e8 * 6.62607015e-34

_EXTRA_RELEASES: dict[str, Path] = {}
_TENSORS: dict[tuple, torch.Tensor] = {}
_CACHE_CLEARERS: list = []


def register_cache_clearer(fn) -> None:
    """Register a zero-arg callback to clear derived asset caches on release changes."""
    _CACHE_CLEARERS.append(fn)


def register_release(release: str, asset_dir: str | Path) -> None:
    """Register an external directory of release assets (e.g. a DR4 pre-release)."""
    rel = release.lower()
    path = Path(asset_dir)
    if not (path / "instrument.json").is_file():
        raise FileNotFoundError(f"Expected instrument.json in {path}")
    _EXTRA_RELEASES[rel] = path
    _npz.cache_clear()
    _json.cache_clear()
    _TENSORS.clear()
    for clear in _CACHE_CLEARERS:
        clear()


def available_releases() -> tuple[str, ...]:
    """Return all packaged (and registered) Gaia data release labels."""
    root = resources.files("torchgaiaxp") / "_assets"
    found = {
        entry.name
        for entry in root.iterdir()
        if entry.is_dir() and (entry / "instrument.json").is_file()
    }
    found.update(_EXTRA_RELEASES.keys())
    return tuple(sorted(found))


def parse_release(release: str | None) -> str:
    """Normalize and validate a Gaia data release label (default ``'dr3'``)."""
    rel = DEFAULT_RELEASE if release is None else str(release).lower()
    known = available_releases()
    if rel not in known:
        raise ValueError(f"unknown Gaia release {release!r}; available: {list(known)}")
    return rel


def _asset_path(release: str, filename: str):
    rel = parse_release(release)
    if rel in _EXTRA_RELEASES:
        return _EXTRA_RELEASES[rel] / filename
    return resources.files("torchgaiaxp") / "_assets" / rel / filename


@cache
def _npz(release: str, name: str) -> dict[str, np.ndarray]:
    asset = _asset_path(release, name)
    with resources.as_file(asset) as path, np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


@cache
def _json(release: str, name: str) -> dict:
    asset = _asset_path(release, name)
    with resources.as_file(asset) as path:
        return json.loads(Path(path).read_text(encoding="utf-8"))


def instrument_meta(release: str = DEFAULT_RELEASE) -> dict:
    return _json(parse_release(release), "instrument.json")


def systems_meta(release: str = DEFAULT_RELEASE) -> dict:
    return _json(parse_release(release), "systems.json")


def band_wl_range(band: str, release: str = DEFAULT_RELEASE) -> tuple[float, float]:
    b = band.lower()
    if b not in BANDS:
        raise ValueError(f"invalid band {band!r}; expected 'bp' or 'rp'")
    meta = instrument_meta(release)
    lo, hi = meta[f"{b}_wl"]
    return float(lo), float(hi)


def n_bases_for_release(band: str | None = None, release: str = DEFAULT_RELEASE):
    meta = instrument_meta(release)
    nb = meta["n_bases"]
    if band is None:
        return int(nb["bp"]), int(nb["rp"])
    b = band.lower()
    if b not in BANDS:
        raise ValueError(f"invalid band {band!r}; expected 'bp' or 'rp'")
    return int(nb[b])


def resolve_release(
    solution_id=None,
    n_parameters=None,
    release: str | None = None,
) -> str:
    """Resolve and validate Gaia release from explicit arg, solution_id, or n_parameters."""
    explicit = None if release is None else parse_release(release)
    releases = available_releases()

    if solution_id is not None:
        arr = np.asarray(solution_id)
        vals = {str(int(v)) for v in arr.reshape(-1) if np.isfinite(v)}
        if vals:
            sol_map = {
                rel: set(instrument_meta(rel).get("solution_ids", ()))
                for rel in releases
            }
            known_sols = set().union(*sol_map.values())
            unknown = vals - known_sols
            if not unknown:
                matched = {rel for rel, sids in sol_map.items() if vals & sids}
                if len(matched) > 1:
                    raise ValueError(f"input mixes solution_ids from multiple releases: {sorted(matched)}")
                if len(matched) == 1:
                    detected = next(iter(matched))
                    if explicit is not None and detected != explicit:
                        raise ValueError(
                            f"input solution_id corresponds to {detected!r}, conflicts with release={explicit!r}"
                        )
                    return detected

    if n_parameters is not None:
        arr = np.asarray(n_parameters)
        if arr.ndim == 1 and arr.size == 2:
            arr = arr[None, :]
        if arr.ndim >= 2 and arr.shape[-1] == 2:
            bp_vals = {int(v) for v in arr[..., 0].reshape(-1) if np.isfinite(v) and v > 0}
            rp_vals = {int(v) for v in arr[..., 1].reshape(-1) if np.isfinite(v) and v > 0}
            if bp_vals or rp_vals:
                matched = set()
                for rel in releases:
                    nb = instrument_meta(rel)["n_bases"]
                    if (not bp_vals or bp_vals == {int(nb["bp"])}) and (
                        not rp_vals or rp_vals == {int(nb["rp"])}
                    ):
                        matched.add(rel)
                if len(matched) == 1:
                    detected = next(iter(matched))
                    if explicit is not None and detected != explicit:
                        raise ValueError(
                            f"input n_parameters corresponds to {detected!r}, conflicts with release={explicit!r}"
                        )
                    return detected

    return explicit if explicit is not None else DEFAULT_RELEASE


def resolve_system_name(name: str, release: str | None = None) -> tuple[str, str]:
    """Resolve canonical or legacy-alias system name -> ``(release, canonical_name)``."""
    search_releases = (parse_release(release),) if release is not None else available_releases()
    for rel in search_releases:
        meta = systems_meta(rel)
        if name in meta and not name.startswith("_"):
            return rel, name
        aliases = meta.get("_aliases", {})
        if name in aliases:
            return rel, aliases[name]
    known = sorted(
        k
        for rel in search_releases
        for k in systems_meta(rel)
        if not k.startswith("_")
    )
    raise KeyError(f"unknown photometric system {name!r}; known: {known}")


def instrument_array(key: str, release: str = DEFAULT_RELEASE) -> np.ndarray:
    return _npz(parse_release(release), "instrument.npz")[key]


def system_array(key: str, release: str = DEFAULT_RELEASE) -> np.ndarray:
    return _npz(parse_release(release), "systems.npz")[key]


def tensor(
    store: str,
    key: str,
    device=None,
    dtype=torch.float64,
    release: str = DEFAULT_RELEASE,
) -> torch.Tensor:
    """An asset as a contiguous tensor on ``device``/``dtype``, converted once and cached."""
    rel = parse_release(release)
    device = torch.device(device) if device is not None else torch.device("cpu")
    cache_key = (rel, store, key, str(device), dtype)
    hit = _TENSORS.get(cache_key)
    if hit is None:
        array = instrument_array(key, rel) if store == "instrument" else system_array(key, rel)
        hit = torch.as_tensor(array, dtype=dtype).to(device).contiguous()
        _TENSORS[cache_key] = hit
    return hit
