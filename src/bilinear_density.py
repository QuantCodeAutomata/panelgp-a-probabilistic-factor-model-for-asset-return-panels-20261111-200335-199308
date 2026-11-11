"""Predictive density of a single bilinear return via one-dimensional Fourier
inversion of its cumulant generating function (paper Appendix B.3, Eq.
B.2, B.7, B.14-B.15).

No existing library implements the CGF of a bilinear form of two independent
Gaussian vectors, so it is implemented directly from the closed-form
cumulant generating function of a quadratic form of a Gaussian vector
(Mathai & Provost, 1992, as used in paper Eq. B.7).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import brentq

from src import config as cfg


def _build_joint(m_beta: np.ndarray, S_beta: np.ndarray, m_phi: np.ndarray,
                  S_phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stacks w = (beta_j, phi_t) into one Gaussian vector and builds the
    bilinear-form matrix A with off-diagonal 1/2 I blocks (paper Eq. B.2),
    so that beta_j^T phi_t = w^T A w."""
    H = m_beta.shape[0]
    mu = np.concatenate([m_beta, m_phi])
    Sigma = np.zeros((2 * H, 2 * H))
    Sigma[:H, :H] = S_beta
    Sigma[H:, H:] = S_phi
    A = np.zeros((2 * H, 2 * H))
    A[:H, H:] = 0.5 * np.eye(H)
    A[H:, :H] = 0.5 * np.eye(H)
    return mu, Sigma, A


def _cgf(s: complex, mu: np.ndarray, Sigma: np.ndarray, A: np.ndarray) -> complex:
    """K(s) = -0.5 log det(I - 2sA*Sigma) + s * mu^T A (I-2sA*Sigma)^-1 mu
    (paper Eq. B.7), the CGF of the quadratic form w^T A w, w ~ N(mu,Sigma)."""
    d = mu.shape[0]
    M = np.eye(d) - 2.0 * s * (A @ Sigma)
    Minv = np.linalg.inv(M)
    logdet = np.log(np.linalg.det(M).astype(complex))
    quad = mu @ Minv @ (A @ mu)
    return -0.5 * logdet + s * quad


def _cgf_domain(Sigma: np.ndarray, A: np.ndarray) -> tuple[float, float]:
    """Real interval of s on which K(s) is finite: eigenvalues of A*Sigma
    must satisfy 1 - 2*s*lambda > 0 for every eigenvalue lambda."""
    eigs = np.linalg.eigvals(A @ Sigma).real
    upper = min([1.0 / (2.0 * e) for e in eigs if e > 1e-12], default=np.inf)
    lower = max([1.0 / (2.0 * e) for e in eigs if e < -1e-12], default=-np.inf)
    margin = 0.999
    return lower * margin if np.isfinite(lower) else -10.0, upper * margin if np.isfinite(upper) else 10.0


def _derivatives(s: float, mu: np.ndarray, Sigma: np.ndarray, A: np.ndarray, h: float = 1e-4):
    """K(s), K'(s), K''(s) at real s by central finite differences of the
    closed-form CGF (Eq. B.7); K is holomorphic and smooth on its domain, so
    this is numerically stable."""
    k_m2, k_m1, k0, k_p1, k_p2 = (
        _cgf(s + i * h, mu, Sigma, A).real for i in (-2, -1, 0, 1, 2)
    )
    k_prime = (k_p1 - k_m1) / (2 * h)
    k_double_prime = (k_p1 - 2 * k0 + k_m1) / (h ** 2)
    return k0, k_prime, k_double_prime


def log_density_bilinear(r: float, m_beta: np.ndarray, S_beta: np.ndarray,
                          m_phi: np.ndarray, S_phi: np.ndarray, sigma_r2: float) -> float:
    """log p_bil(r) for one row, r = beta_j^T phi_t + sigma_r * eps (paper
    Sec. 4.3, Eq. B.14-B.15): saddlepoint-style Fourier inversion of the
    joint CGF of the bilinear form plus the independent Gaussian noise term.

    The CGF of the full return adds the noise term 0.5*sigma_r2*s^2 to the
    bilinear-form CGF (independent sum of cumulant generating functions).
    """
    mu, Sigma, A = _build_joint(m_beta, S_beta, m_phi, S_phi)
    s_lo, s_hi = _cgf_domain(Sigma, A)

    def K_total(s):
        return _cgf(s, mu, Sigma, A) + 0.5 * sigma_r2 * s ** 2

    def K_prime_total(s):
        _, kp, _ = _derivatives(s, mu, Sigma, A)
        return kp + sigma_r2 * s

    # Solve K'(s) = r for s strictly inside the open CGF domain.
    lo, hi = s_lo + 1e-6, s_hi - 1e-6
    f_lo, f_hi = K_prime_total(lo) - r, K_prime_total(hi) - r
    if f_lo * f_hi > 0:
        # r outside the reachable mean range at this precision: fall back to
        # the domain boundary closest to the sign change.
        grid = np.linspace(lo, hi, 50)
        vals = np.array([K_prime_total(x) - r for x in grid])
        sign_changes = np.where(np.diff(np.sign(vals)) != 0)[0]
        assert len(sign_changes) > 0, "no sign change found for K'(s)=r in the CGF domain"
        i = sign_changes[0]
        lo, hi = grid[i], grid[i + 1]
    s_star = brentq(lambda s: K_prime_total(s) - r, lo, hi, xtol=1e-10)

    k0, _, k_double_prime_bilinear = _derivatives(s_star, mu, Sigma, A)
    k_double_prime = k_double_prime_bilinear + sigma_r2
    assert k_double_prime > 0, "K''(s*) must be positive at the saddlepoint"
    d = np.sqrt(k_double_prime)
    k_s_star = (k0 + 0.5 * sigma_r2 * s_star ** 2).real

    def integrand(w):
        s_c = s_star + 1j * w / d
        val = K_total(s_c) - k_s_star - 1j * w * r / d
        return np.real(np.exp(val))

    integral, _ = quad_vec(integrand, 0.0, np.inf,
                            epsabs=cfg.ROLL_CGF_ABS_TOL, epsrel=cfg.ROLL_CGF_REL_TOL)
    density = np.exp(k_s_star - s_star * r) * integral / (np.pi * d)
    assert density > 0, "predictive density must be positive"
    return float(np.log(density))


def gaussian_log_density(r: float, mu: float, sigma2: float) -> float:
    """log N(r | mu, sigma2), the Gaussian reference used for the log-score
    gain g = log p_bil(r) - log N(r|mu,sigma2) (paper Sec. 6.7)."""
    return float(-0.5 * (np.log(2 * np.pi * sigma2) + (r - mu) ** 2 / sigma2))
