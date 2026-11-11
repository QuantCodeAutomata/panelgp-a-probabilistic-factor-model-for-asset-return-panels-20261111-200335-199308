"""Pairwise model comparisons with Newey-West variances and Holm adjustment
(paper Appendix D.3)."""

from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


def newey_west_long_run_variance(series: np.ndarray, n_lags: int = 12) -> float:
    """Newey-West (1987) long-run variance of a demeaned time series with
    ``n_lags`` lags (paper Appendix D.3)."""
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    n = len(x)
    gamma0 = float(np.dot(x, x) / n)
    lrv = gamma0
    for lag in range(1, n_lags + 1):
        weight = 1.0 - lag / (n_lags + 1)
        gamma_l = float(np.dot(x[lag:], x[:-lag]) / n)
        lrv += 2.0 * weight * gamma_l
    return max(lrv, 1e-12)


def paired_difference_test(series_a: np.ndarray, series_b: np.ndarray, n_lags: int = 12) -> tuple[float, float]:
    """Two-sided test of equal mean of ``series_a`` and ``series_b`` for a
    paired monthly score (e.g. squared errors, log scores), using the
    Newey-West long-run variance of the difference series (paper
    Appendix D.3).

    Returns:
        (mean_difference, p_value)
    """
    diff = np.asarray(series_a) - np.asarray(series_b)
    n = len(diff)
    lrv = newey_west_long_run_variance(diff, n_lags=n_lags)
    se = np.sqrt(lrv / n)
    mean_diff = diff.mean()
    z = mean_diff / se if se > 0 else 0.0
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
    return float(mean_diff), p_value


def sharpe_ratio_delta_method_test(
    returns_a: np.ndarray, returns_b: np.ndarray, n_lags: int = 12,
) -> tuple[float, float]:
    """Delta-method test of equal (non-annualized) Sharpe ratios of two
    paired monthly return series (Ledoit and Wolf, 2008, as used in paper
    Appendix D.3).

    The Sharpe ratio SR = mu / sigma has influence function (per
    observation t) psi_t = (r_t - mu)/sigma - mu*((r_t-mu)^2 - sigma^2)/(2*sigma^3);
    the HAC variance of the difference psi_a - psi_b gives the test
    statistic for SR_a - SR_b.
    """
    a, b = np.asarray(returns_a, dtype=float), np.asarray(returns_b, dtype=float)
    mu_a, mu_b = a.mean(), b.mean()
    sigma_a, sigma_b = a.std(ddof=0), b.std(ddof=0)
    assert sigma_a > 0 and sigma_b > 0

    def influence(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        return (x - mu) / sigma - mu * ((x - mu) ** 2 - sigma ** 2) / (2.0 * sigma ** 3)

    psi = influence(a, mu_a, sigma_a) - influence(b, mu_b, sigma_b)
    n = len(psi)
    lrv = newey_west_long_run_variance(psi, n_lags=n_lags)
    se = np.sqrt(lrv / n)
    sr_diff = mu_a / sigma_a - mu_b / sigma_b
    z = sr_diff / se if se > 0 else 0.0
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
    return float(sr_diff), p_value


def holm_adjust(p_values: list[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Holm step-down adjustment of a family of p-values (paper
    Appendix D.3: "15 comparisons per criterion").

    Returns:
        (adjusted_p_values, reject) with ``reject`` the boolean array of
        rejections at level ``alpha``.
    """
    reject, p_adj, _, _ = multipletests(p_values, alpha=alpha, method="holm")
    return p_adj, reject
