"""GaiaXPy's Cholesky products, batched: inverse square root, inverse, chi^2.

As in GaiaXPy, the coefficient covariance here is built from the formal errors
divided by the standard deviation of the least-squares solution,
``Sigma = diag(e / sigma) R diag(e / sigma)``. (Sampled errors use ``e`` itself:
GaiaXPy rescales those by ``sigma`` afterwards, which cancels.) Sources whose
correlation matrix is not positive definite give NaN where GaiaXPy gives None.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .spectrum import unpack_correlations


def inverse_square_root_covariance(
    errors: Tensor, correlations: Tensor, standard_deviation: Tensor | None = None
) -> Tensor:
    """``L^-1 diag(sigma / e)`` ``[..., 55, 55]`` (GaiaXPy
    ``get_inverse_square_root_covariance_matrix``)."""
    e = errors if standard_deviation is None else errors / standard_deviation.unsqueeze(-1)
    corr = unpack_correlations(correlations, errors.shape[-1])
    chol, info = torch.linalg.cholesky_ex(corr)
    eye = torch.eye(chol.shape[-1], dtype=chol.dtype, device=chol.device).expand_as(chol)
    l_inv = torch.linalg.solve_triangular(chol, eye, upper=False)
    out = l_inv / e.unsqueeze(-2)
    return torch.where((info == 0)[..., None, None], out, torch.nan)


def inverse_covariance(
    errors: Tensor, correlations: Tensor, standard_deviation: Tensor | None = None
) -> Tensor:
    """``(L^-1 E^-1)^T (L^-1 E^-1)`` (GaiaXPy ``get_inverse_covariance_matrix``)."""
    w = inverse_square_root_covariance(errors, correlations, standard_deviation)
    return w.transpose(-1, -2) @ w


def chi2(inverse_square_root: Tensor, residuals: Tensor) -> Tensor:
    """``|L^-1 r|^2`` (GaiaXPy ``get_chi2``), batched over leading dimensions."""
    x = (inverse_square_root @ residuals.unsqueeze(-1)).squeeze(-1)
    return x.square().sum(-1)
