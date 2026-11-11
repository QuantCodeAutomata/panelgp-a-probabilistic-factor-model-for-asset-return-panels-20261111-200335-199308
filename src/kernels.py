"""Elementary covariance kernels used to build the loading and factor kernels.

Implements the three elementary kernels of Section 3.3 and Appendix C.2:
the RBF kernel with automatic relevance determination (ARD) lengthscales
(Equation C.2), the affine kernel ``1 + x^T x'`` and the constant kernel 1.
A dedicated Gaussian-process library (GPyTorch/GPflow) was considered, but
the bilinear likelihood and the block-structured inducing state in Section 4
have no library equivalent, so the kernels themselves are implemented
directly in PyTorch to keep full control of gradients through the custom
variational objective in src/panelgp_model.py.
"""

import torch


def rbf_kernel(x1: torch.Tensor, x2: torch.Tensor, lengthscale: torch.Tensor) -> torch.Tensor:
    """ARD RBF kernel, paper Equation (C.2).

    k(x, x'; ell) = exp(-0.5 * sum_p (x_p - x'_p)^2 / ell_p^2)

    Args:
        x1: (N1, P) inputs.
        x2: (N2, P) inputs.
        lengthscale: (P,) positive per-coordinate lengthscales.

    Returns:
        (N1, N2) kernel matrix.
    """
    assert x1.shape[-1] == x2.shape[-1] == lengthscale.shape[-1]
    diff = (x1[:, None, :] - x2[None, :, :]) / lengthscale[None, None, :]
    sq_dist = (diff ** 2).sum(-1)
    return torch.exp(-0.5 * sq_dist)


def affine_kernel(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Affine kernel 1 + x^T x', paper Appendix C.2."""
    return 1.0 + x1 @ x2.T


def constant_kernel(n1: int, n2: int, device=None, dtype=None) -> torch.Tensor:
    """Constant kernel equal to 1 everywhere, paper Appendix C.2."""
    return torch.ones(n1, n2, device=device, dtype=dtype)


def safe_cholesky(matrix: torch.Tensor, jitter_init: float, jitter_growth: float,
                   max_tries: int) -> tuple[torch.Tensor, float]:
    """Cholesky factorization with growing diagonal jitter, paper Appendix C.3.

    Starts at ``jitter_init`` and multiplies the jitter by ``jitter_growth``
    until the factorization succeeds, as described for the near-singular
    affine inducing covariance.
    """
    n = matrix.shape[-1]
    eye = torch.eye(n, device=matrix.device, dtype=matrix.dtype)
    jitter = jitter_init
    for _ in range(max_tries):
        try:
            chol = torch.linalg.cholesky(matrix + jitter * eye)
            return chol, jitter
        except RuntimeError:
            jitter *= jitter_growth
    raise RuntimeError(f"Cholesky failed after {max_tries} tries, last jitter={jitter}")
