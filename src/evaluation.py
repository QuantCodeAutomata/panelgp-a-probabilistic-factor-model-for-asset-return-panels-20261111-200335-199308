"""Evaluation utilities for the simulation study (paper Section 5, Table 1).

Implements the factor-alignment transformation, the function-R^2 metric on
dense grids, the oracle conditional variance of Equation (6), and the
predicted-variance decomposition of Equations (25)-(27).
"""

from __future__ import annotations

import numpy as np
import torch


def r_squared(true: np.ndarray, fitted: np.ndarray) -> float:
    """R^2 = 1 - sum((true-fitted)^2) / sum(true^2), matching the paper's
    pooled predictive-R^2 definition (Section 6.3) applied here to function
    or state recovery (Section 5.2)."""
    true = np.asarray(true).ravel()
    fitted = np.asarray(fitted).ravel()
    ss_res = np.sum((true - fitted) ** 2)
    ss_tot = np.sum(true ** 2)
    return 1.0 - ss_res / ss_tot


def align_factors(F_true: np.ndarray, F_est: np.ndarray) -> np.ndarray:
    """Least-squares alignment matrix Upsilon minimizing
    ||Upsilon F_est^T - F_true^T||_F (paper Section 5.1), solved by ordinary
    least squares: Upsilon^T = argmin_A ||F_est A - F_true||_F.

    Args:
        F_true: (T, H) true realized factors.
        F_est: (T, H) estimated (variational-mean) realized factors.

    Returns:
        Upsilon: (H, H) such that Upsilon @ F_est.T approximates F_true.T.
    """
    # Solve F_est @ A = F_true for A (T,H)@(H,H)=(T,H); Upsilon = A.T
    A, *_ = np.linalg.lstsq(F_est, F_true, rcond=None)
    return A.T


def oracle_conditional_variance(sigma_beta2: float, sigma_phi2: float,
                                 sigma_r2: float, beta_x: np.ndarray,
                                 phi_c: np.ndarray) -> np.ndarray:
    """Oracle variance of r_{i,t+1} given the true smooth maps, Equation (6):

    Var(r | beta(x), phi(c)) = phi(c)^T D_beta phi(c) + beta(x)^T D_phi beta(x)
                                 + tr(D_beta D_phi) + sigma_r^2,

    with isotropic D_beta = sigma_beta^2 I, D_phi = sigma_phi^2 I (Sec. 5.1).

    Args:
        beta_x: (N, H) true smooth loadings at each row's characteristics.
        phi_c: (N, H) true smooth factors at each row's date.

    Returns:
        (N,) oracle variances.
    """
    H = beta_x.shape[1]
    systematic_beta = sigma_beta2 * (phi_c ** 2).sum(-1)
    systematic_phi = sigma_phi2 * (beta_x ** 2).sum(-1)
    cross = H * sigma_beta2 * sigma_phi2
    return systematic_beta + systematic_phi + cross + sigma_r2


def variance_decomposition(m_beta: np.ndarray, S_beta: np.ndarray,
                            m_phi: np.ndarray, S_phi: np.ndarray,
                            tau_beta: np.ndarray, rho_phi: np.ndarray,
                            sigma_r2: float, V0: np.ndarray) -> dict:
    """Predicted-variance decomposition, paper Equations (25)-(27).

    Args:
        m_beta: (N,H) loading posterior mean at each row's characteristics.
        S_beta: (N,H,H) loading posterior covariance (smooth, no D_beta).
        m_phi: (N,H) factor posterior mean at each row's date.
        S_phi: (N,H,H) factor posterior covariance (smooth, no D_phi).
        tau_beta: (H,) diagonal of D_beta.
        rho_phi: (H,) diagonal of D_phi.
        sigma_r2: observation noise variance.
        V0: (N,) Var_q(beta(x)^T phi(c)) from posterior uncertainty about the
            smooth maps (computed by the caller via the joint predictive
            covariance of beta and phi).

    Returns:
        dict with V0, V_phi, V_beta, cross, sigma_r2, and total variance.
    """
    D_phi = np.diag(rho_phi)
    D_beta = np.diag(tau_beta)
    V_phi = np.einsum("nh,hg,ng->n", m_beta, D_phi, m_beta) + np.einsum(
        "nhh->n", S_beta @ D_phi)
    V_beta = np.einsum("nh,hg,ng->n", m_phi, D_beta, m_phi) + np.einsum(
        "nhh->n", S_phi @ D_beta)
    cross = np.trace(D_beta @ D_phi) * np.ones_like(V0)
    total = V0 + V_phi + V_beta + cross + sigma_r2
    return {"V0": V0, "V_phi": V_phi, "V_beta": V_beta, "cross": cross,
            "sigma_r2": sigma_r2 * np.ones_like(V0), "total": total}


def joint_loading_factor_variance(model, X: torch.Tensor,
                                   date_idx: np.ndarray) -> np.ndarray:
    """Computes V0_j = Var_q(beta(x_j)^T phi(c_t(j))) for each row j, using
    the joint predictive covariance of beta(x_j) and phi(c_t(j)) under the
    variational posterior (paper Eq. 24-26): for jointly Gaussian
    (beta,phi) with mean (m_b,m_p) and covariance blocks (Sbb,Sbp,Spp),
    Var(beta^T phi) = m_b^T Spp m_b + m_p^T Sbb m_p + 2 m_b^T Sbp^T m_p
                        + tr(Sbb Spp) + tr(Sbp Sbp^T).

    This uses the same building blocks as the training likelihood
    (`PanelGPModel._gather_blocks`), evaluated row by row with ``date_idx``
    giving each row's training-date index.
    """
    with torch.no_grad():
        L_beta, L_phi, a, b, rho = model.prior_cholesky()
        m_v, S_v = model.variational_moments(L_beta, L_phi)
        G, R = model._row_projection(X, L_beta)
        date_idx_t = torch.as_tensor(date_idx, dtype=torch.long)
        m_beta, m_phi, S_bb, S_bphi, S_phiphi = model._gather_blocks(
            m_v, S_v, G, R, date_idx_t)
        m_b = m_beta.numpy()
        m_p = m_phi.numpy()
        Sbb = S_bb.numpy()
        Spp = S_phiphi.numpy()
        Sbp = S_bphi.numpy()
        term1 = np.einsum("nh,nhg,ng->n", m_b, Spp, m_b)
        term2 = np.einsum("nh,nhg,ng->n", m_p, Sbb, m_p)
        # Cross term Cov(mb^T dphi, dbeta^T mp) = mp^T Sbp mb (Sbp[h,g]=Cov(beta_h,phi_g)
        # is not symmetric, so mp on the left and mb on the right is not interchangeable
        # with the reverse order; derived via Isserlis/Wick's theorem for the variance
        # of a bilinear form in jointly Gaussian variables).
        term3 = 2 * np.einsum("nh,nhg,ng->n", m_p, Sbp, m_b)
        term4 = np.einsum("nhh->n", Sbb @ Spp)
        term5 = np.einsum("nhg,ngh->n", Sbp, Sbp)
        return term1 + term2 + term3 + term4 + term5
