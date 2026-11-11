"""IPCA benchmark, estimated by alternating least squares (paper Sec. 6.2).

Instrumented Principal Component Analysis (Kelly, Pruitt and Su): loadings
are linear in the characteristics, Lambda_t = [X_t, 1] @ Gamma, and factors
are re-estimated jointly with Gamma by alternating least squares until
convergence. The intercept variant adds a second loading column
[X_t, 1] @ gamma_alpha on a factor fixed at one (paper Sec. 6.2). No
off-the-shelf package implements exactly this unbalanced-panel ALS variant
with a training/validation/evaluation split matching the paper's
description, so it is implemented directly with ``numpy``/``scipy.linalg``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IPCAFit:
    """Fitted IPCA model (paper Sec. 6.2).

    Gamma: (P+1, H) loading-generating coefficients on [characteristics, 1].
    factors: (T, H) estimated factor realizations at the training dates.
    with_intercept: whether an intercept loading/factor-fixed-at-one column
        was added.
    gamma_alpha: (P+1,) intercept loading coefficients, or None.
    sigma_e2: pooled in-sample residual variance.
    """

    Gamma: np.ndarray
    factors: np.ndarray
    with_intercept: bool
    gamma_alpha: np.ndarray | None
    sigma_e2: float


def _loadings(X_t: np.ndarray, Gamma: np.ndarray) -> np.ndarray:
    """[X_t, 1] @ Gamma for one month's characteristic matrix X_t (N_t, P)."""
    ones = np.ones((X_t.shape[0], 1))
    return np.concatenate([X_t, ones], axis=1) @ Gamma


def fit_ipca(
    X_by_month: list[np.ndarray], r_by_month: list[np.ndarray], H: int,
    with_intercept: bool, n_iter: int = 500, tol: float = 1e-6,
    seed: int = 0,
) -> IPCAFit:
    """Estimates IPCA by alternating least squares (paper Sec. 6.2).

    Args:
        X_by_month: list of (N_t, P) characteristic matrices, one per
            training month t.
        r_by_month: list of (N_t,) excess-return vectors aligned with
            ``X_by_month``.
        H: number of factors.
        with_intercept: if True, adds an intercept loading
            [X_t, 1] @ gamma_alpha on a factor fixed at one (IPCA+int).
        n_iter: maximum ALS iterations.
        tol: relative change in Gamma (Frobenius norm) for convergence.
        seed: random seed for the Gamma initialization.

    Returns:
        An :class:`IPCAFit`.
    """
    T = len(X_by_month)
    P = X_by_month[0].shape[1]
    rng = np.random.default_rng(seed)

    # Orthonormal random initialization of Gamma (P+1, H), paper Sec. 6.2.
    Gamma, _ = np.linalg.qr(rng.standard_normal((P + 1, H)))
    gamma_alpha = rng.standard_normal(P + 1) * 0.01 if with_intercept else None

    factors = np.zeros((T, H))
    prev_Gamma = Gamma.copy()
    for iteration in range(n_iter):
        # --- Step 1: given Gamma (and gamma_alpha), re-estimate factors f_t
        # by OLS of r_t on Lambda_t = [X_t,1]@Gamma (net of the intercept
        # loading, which has a factor fixed at one). ---
        for t in range(T):
            X_t, r_t = X_by_month[t], r_by_month[t]
            Lambda_t = _loadings(X_t, Gamma)  # (N_t, H)
            y_t = r_t
            if with_intercept:
                alpha_t = _loadings(X_t, gamma_alpha[:, None]).ravel()
                y_t = r_t - alpha_t
            f_t, *_ = np.linalg.lstsq(Lambda_t, y_t, rcond=None)
            factors[t] = f_t

        # --- Step 2: given factors, re-estimate Gamma (and gamma_alpha) by
        # pooled OLS of stacked returns on stacked regressors
        # [X_t,1] kron f_t (and [X_t,1] for the intercept). ---
        rows_lhs, rows_rhs = [], []
        for t in range(T):
            X_t, r_t, f_t = X_by_month[t], r_by_month[t], factors[t]
            ones = np.ones((X_t.shape[0], 1))
            Z_t = np.concatenate([X_t, ones], axis=1)  # (N_t, P+1)
            design = np.einsum("np,h->nph", Z_t, f_t).reshape(Z_t.shape[0], -1)
            if with_intercept:
                design = np.concatenate([design, Z_t], axis=1)
            rows_lhs.append(design)
            rows_rhs.append(r_t)
        design_all = np.concatenate(rows_lhs, axis=0)
        r_all = np.concatenate(rows_rhs, axis=0)
        theta, *_ = np.linalg.lstsq(design_all, r_all, rcond=None)
        Gamma = theta[: (P + 1) * H].reshape(P + 1, H)
        if with_intercept:
            gamma_alpha = theta[(P + 1) * H:]

        # --- Identification: orthonormalize Gamma columns (paper Sec. 6.2
        # leaves the rotation of (Gamma, factors) unidentified up to an
        # invertible H x H transform; forecasts and Sigma_hat are invariant
        # to this choice, so any fixed normalization is valid here). ---
        U, S, Vt = np.linalg.svd(Gamma, full_matrices=False)
        Gamma = U * S
        factors = factors @ Vt.T

        change = np.linalg.norm(Gamma - prev_Gamma) / max(np.linalg.norm(prev_Gamma), 1e-12)
        prev_Gamma = Gamma.copy()
        if change < tol:
            break

    resid_sq, n_total = 0.0, 0
    for t in range(T):
        X_t, r_t, f_t = X_by_month[t], r_by_month[t], factors[t]
        Lambda_t = _loadings(X_t, Gamma)
        pred = Lambda_t @ f_t
        if with_intercept:
            pred = pred + _loadings(X_t, gamma_alpha[:, None]).ravel()
        resid_sq += float(np.sum((r_t - pred) ** 2))
        n_total += X_t.shape[0]
    sigma_e2 = resid_sq / n_total

    return IPCAFit(Gamma=Gamma, factors=factors, with_intercept=with_intercept,
                    gamma_alpha=gamma_alpha, sigma_e2=sigma_e2)


def ipca_forecast(fit: IPCAFit, X_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predictive mean and covariance for month t's characteristics, paper
    Sec. 6.2:

    mu_hat = [X_t,1] @ Gamma_hat @ phi_bar (+ intercept loading),
    Sigma_hat = [X_t,1] @ Gamma_hat @ Sigma_phi_hat @ Gamma_hat^T @ [X_t,1]^T
                 + sigma_e_hat^2 * I,

    with phi_bar and Sigma_phi_hat the sample mean and covariance of the
    estimated training factors.
    """
    phi_bar = fit.factors.mean(axis=0)
    Sigma_phi = np.cov(fit.factors.T, bias=True) if fit.factors.shape[1] > 1 else np.array([[np.var(fit.factors)]])
    Lambda_t = _loadings(X_t, fit.Gamma)
    mu_hat = Lambda_t @ phi_bar
    if fit.with_intercept:
        mu_hat = mu_hat + _loadings(X_t, fit.gamma_alpha[:, None]).ravel()
    Sigma_hat = Lambda_t @ Sigma_phi @ Lambda_t.T + fit.sigma_e2 * np.eye(X_t.shape[0])
    return mu_hat, Sigma_hat
