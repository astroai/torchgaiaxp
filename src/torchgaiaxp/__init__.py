"""torchgaiaxp: Gaia XP spectra in torch -- continuous, batched, on any device.

Feature parity with GaiaXPy 2.2 (calibrate, convert, generate with all 28
photometric systems + aliases + custom XML/YAML filters, colour equations, error
correction, Cholesky tools, pseudo-wavelength conversions, multi-release and
DR4 epoch/variable-basis support), plus a continuous-spectrum algebra for
combining XP spectra with spectra of any other resolution.
"""

from ._assets import available_releases, register_release
from .cholesky import chi2, inverse_covariance, inverse_square_root_covariance
from .continuous import Constant, Continuous, Evaluation, Function, Sampled
from .design import XPDesign
from .dispersion import pwl_range, pwl_to_wl, wl_range, wl_to_pwl
from .epoch import XPEpochSpectra, decode_transit_id
from .photometry import (
    apply_colour_equation,
    apply_error_correction,
    generate,
    load_additional_systems,
    remove_additional_systems,
    systems,
)
from .spectrum import XPSpectra, pack_correlations, unpack_correlations

__version__ = "0.3.0"

__all__ = [
    "Constant",
    "Continuous",
    "Evaluation",
    "Function",
    "Sampled",
    "XPDesign",
    "XPEpochSpectra",
    "XPSpectra",
    "apply_colour_equation",
    "apply_error_correction",
    "available_releases",
    "chi2",
    "decode_transit_id",
    "generate",
    "inverse_covariance",
    "inverse_square_root_covariance",
    "load_additional_systems",
    "pack_correlations",
    "pwl_range",
    "pwl_to_wl",
    "register_release",
    "remove_additional_systems",
    "systems",
    "unpack_correlations",
    "wl_range",
    "wl_to_pwl",
]
