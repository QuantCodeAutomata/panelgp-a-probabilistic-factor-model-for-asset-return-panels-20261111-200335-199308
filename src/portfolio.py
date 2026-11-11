"""Portfolio construction and trading-cost accounting (paper Sec. 6.3, 6.6).

Implements the three monthly portfolios used to evaluate forecasts (GMV,
tangency, long-short decile) and the weight-drift/turnover/net-return
accounting used for the trading-cost analysis of the rolling experiment.
"""

from __future__ import annotations

import numpy as np


def gmv_weights(Sigma_hat: np.ndarray) -> np.ndarray:
    """Global-minimum-variance weights w = Sigma^-1 1 / (1^T Sigma^-1 1)
    (paper Sec. 6.3)."""
    ones = np.ones(Sigma_hat.shape[0])
    inv_ones = np.linalg.solve(Sigma_hat, ones)
    return inv_ones / (ones @ inv_ones)


def tangency_weights(mu_hat: np.ndarray, Sigma_hat: np.ndarray) -> np.ndarray:
    """Tangency weights w = Sigma^-1 mu / ||Sigma^-1 mu||_1 (paper Sec. 6.3).

    Normalizing by the L1 norm gives unit gross exposure (sum of |w| = 1);
    short positions are allowed.
    """
    raw = np.linalg.solve(Sigma_hat, mu_hat)
    denom = np.sum(np.abs(raw))
    assert denom > 0, "tangency weights: Sigma^-1 mu is degenerate (all zero)"
    return raw / denom


def long_short_weights(mu_hat: np.ndarray, gross_exposure: float = 0.5) -> np.ndarray:
    """Equal-weighted top-minus-bottom decile portfolio by predicted return
    (paper Sec. 6.3): long the top decile, short the bottom decile, each
    side with gross exposure ``gross_exposure`` (0.5 by default, for total
    gross exposure 1)."""
    n = len(mu_hat)
    decile = max(1, n // 10)
    order = np.argsort(mu_hat)
    weights = np.zeros(n)
    short_idx, long_idx = order[:decile], order[-decile:]
    weights[long_idx] = gross_exposure / decile
    weights[short_idx] = -gross_exposure / decile
    return weights


def annualized_sharpe(monthly_returns: np.ndarray) -> float:
    """Annualized Sharpe ratio from monthly portfolio returns:
    sqrt(12) * mean / std (paper Sec. 6.3)."""
    monthly_returns = np.asarray(monthly_returns)
    std = monthly_returns.std(ddof=1)
    assert std >= 0
    if std == 0:
        return 0.0
    return float(np.sqrt(12.0) * monthly_returns.mean() / std)


def annualized_volatility(monthly_returns: np.ndarray) -> float:
    """Annualized volatility: sqrt(12) * std of monthly returns."""
    return float(np.sqrt(12.0) * np.asarray(monthly_returns).std(ddof=1))


def drifted_previous_weights(
    w_prev: np.ndarray, r_t: np.ndarray, rf_t: float,
) -> np.ndarray:
    """Weight drift before rebalancing (paper Sec. 6.6):

    w_tilde_{i,t-1} = w_{i,t-1} * (1 + r_f,t + r_i,t) / (1 + r_f,t + w_{t-1}^T r_t)
    """
    port_return = float(w_prev @ r_t)
    return w_prev * (1.0 + rf_t + r_t) / (1.0 + rf_t + port_return)


def turnover(w_t: np.ndarray, w_tilde_prev: np.ndarray) -> float:
    """TO_t = sum_i |w_{i,t} - w_tilde_{i,t-1}| (paper Sec. 6.6)."""
    return float(np.sum(np.abs(w_t - w_tilde_prev)))


def net_return(gross_return: float, turnover_t: float, cost_bps: float) -> float:
    """r_net = w_t^T r_{t+1} - c * TO_t, with c in decimal units (paper
    Sec. 6.6)."""
    return gross_return - (cost_bps / 1e4) * turnover_t


def break_even_cost_bps(monthly_net_excess_at_zero_cost: np.ndarray, turnovers: np.ndarray) -> float:
    """Cost (in bps) at which the average net excess return is zero (paper
    Sec. 6.6): c* solves mean(r_gross - c*TO) = 0, i.e.
    c* = mean(r_gross) / mean(TO)."""
    mean_turnover = np.mean(turnovers)
    assert mean_turnover > 0
    return float(10_000.0 * np.mean(monthly_net_excess_at_zero_cost) / mean_turnover)
