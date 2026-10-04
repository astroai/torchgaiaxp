"""torchgaiaxp: Gaia XP spectra in torch -- continuous, batched, on any device.

Feature parity with GaiaXPy 2.1 (calibrate, convert, generate with all 28
photometric systems, colour equations, error correction, Cholesky tools,
pseudo-wavelength conversions), plus a continuous-spectrum algebra for
combining XP spectra with spectra of any other resolution.
"""

from .cholesky import chi2, inverse_covariance, inverse_square_root_covariance
from .continuous import Constant, Continuous, Evaluation, Function, Sampled
from .design import XPDesign
from .dispersion import pwl_range, pwl_to_wl, wl_range, wl_to_pwl
from .photometry import generate, systems
from .spectrum import XPSpectra, unpack_correlations

__version__ = "0.2.0"

__all__ = [
    "Constant",
    "Continuous",
    "Evaluation",
    "Function",
    "Sampled",
    "XPDesign",
    "XPSpectra",
    "chi2",
    "generate",
    "inverse_covariance",
    "inverse_square_root_covariance",
    "pwl_range",
    "pwl_to_wl",
    "systems",
    "unpack_correlations",
    "wl_range",
    "wl_to_pwl",
]
