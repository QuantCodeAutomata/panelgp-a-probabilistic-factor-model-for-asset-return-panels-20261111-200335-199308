"""Unit tests for the empirical-application shared components (paper
Section 6): characteristic/context preprocessing, IPCA, portfolio
construction, and pairwise significance testing."""

import numpy as np
import pandas as pd
import pytest

from src.characteristics import (
    compute_return_scale,
    filter_characteristics_by_coverage,
    filter_rows_by_row_coverage,
    rank_transform_characteristics,
    standardize_context,
)
from src.ipca import fit_ipca, ipca_forecast
from src.portfolio import (
    annualized_sharpe,
    annualized_volatility,
    break_even_cost_bps,
    drifted_previous_weights,
    gmv_weights,
    long_short_weights,
    net_return,
    tangency_weights,
    turnover,
)
from src.significance import holm_adjust, newey_west_long_run_variance, paired_difference_test, sharpe_ratio_delta_method_test


def test_rank_transform_bounds_and_missing():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "date": np.repeat(["2020-01", "2020-02"], 5),
        "c1": [1.0, 2.0, np.nan, 4.0, 5.0, 10.0, 20.0, 30.0, np.nan, np.nan],
    })
    out = rank_transform_characteristics(df, ["c1"])
    assert out["c1"].between(-0.5, 0.5).all()
    # originally-missing entries are exactly zero
    assert out.loc[2, "c1"] == 0.0
    assert out.loc[8, "c1"] == 0.0 and out.loc[9, "c1"] == 0.0


def test_coverage_filters():
    df = pd.DataFrame({
        "c1": [1.0, 2.0, 3.0, np.nan],
        "c2": [np.nan, np.nan, np.nan, 1.0],
    })
    train_mask = pd.Series([True, True, True, True])
    kept = filter_characteristics_by_coverage(df, ["c1", "c2"], train_mask, min_char_coverage=0.7)
    assert kept == ["c1"]
    row_df = pd.DataFrame({"c1": [1.0, np.nan], "c2": [1.0, 1.0]})
    kept_rows = filter_rows_by_row_coverage(row_df, ["c1", "c2"], min_row_coverage=0.7)
    assert len(kept_rows) == 1


def test_standardize_context_training_moments_only():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 100.0]})
    train_mask = pd.Series([True, True, True, False])
    standardized, mean, std = standardize_context(df, train_mask)
    assert np.isclose(mean["x"], 2.0)
    assert np.isclose(standardized.loc[train_mask, "x"].mean(), 0.0, atol=1e-8)


def test_return_scale_uses_training_rows_only():
    r = np.array([1.0, 1.0, 1.0, 100.0])
    mask = np.array([True, True, True, False])
    s_r = compute_return_scale(r, mask)
    assert np.isclose(s_r, 1.0)


def test_ipca_recovers_known_factor_structure():
    """Synthetic check: generate returns from a known one-factor IPCA model
    and verify the fit recovers forecasts that explain most of the
    in-sample variance (verifies the ALS implementation on data with a
    known answer)."""
    rng = np.random.default_rng(1)
    T, N, P, H = 40, 25, 3, 1
    true_Gamma = rng.standard_normal((P + 1, H))
    X_by_month, r_by_month = [], []
    true_factors = rng.standard_normal((T, H)) * 0.5
    for t in range(T):
        X_t = rng.standard_normal((N, P))
        ones = np.ones((N, 1))
        Lambda_t = np.concatenate([X_t, ones], axis=1) @ true_Gamma
        r_t = Lambda_t @ true_factors[t] + 0.01 * rng.standard_normal(N)
        X_by_month.append(X_t)
        r_by_month.append(r_t)

    fit = fit_ipca(X_by_month, r_by_month, H=H, with_intercept=False, n_iter=200, seed=0)
    # In-sample R^2 using the *realized* fitted factors f_t (not ipca_forecast,
    # which deliberately uses the unconditional mean phi_bar as the forecast
    # mean per paper Sec. 6.2 since the next-period factor is unobserved at
    # forecast time -- this checks the ALS estimator itself, not the forecast
    # rule built on top of it).
    sse, sst = 0.0, 0.0
    for t in range(T):
        Lambda_t = np.concatenate([X_by_month[t], np.ones((N, 1))], axis=1) @ fit.Gamma
        mu_hat = Lambda_t @ fit.factors[t]
        sse += np.sum((r_by_month[t] - mu_hat) ** 2)
        sst += np.sum(r_by_month[t] ** 2)
    r2 = 1 - sse / sst
    assert r2 > 0.9, f"IPCA ALS failed to recover known one-factor structure, R2={r2}"

    # The forecast function itself must run and return the documented shapes,
    # using phi_bar as the forecast mean (paper Sec. 6.2).
    mu_hat, Sigma_hat = ipca_forecast(fit, X_by_month[0])
    assert mu_hat.shape == (N,)
    assert Sigma_hat.shape == (N, N)
    assert np.allclose(Sigma_hat, Sigma_hat.T, atol=1e-8)


def test_ipca_with_intercept_runs():
    rng = np.random.default_rng(2)
    T, N, P, H = 20, 15, 2, 2
    X_by_month = [rng.standard_normal((N, P)) for _ in range(T)]
    r_by_month = [rng.standard_normal(N) * 0.05 for _ in range(T)]
    fit = fit_ipca(X_by_month, r_by_month, H=H, with_intercept=True, n_iter=50, seed=0)
    mu_hat, Sigma_hat = ipca_forecast(fit, X_by_month[0])
    assert mu_hat.shape == (N,)
    assert Sigma_hat.shape == (N, N)
    assert np.all(np.linalg.eigvalsh(Sigma_hat) > 0)


def test_portfolio_weights_properties():
    rng = np.random.default_rng(3)
    n = 10
    mu = rng.standard_normal(n) * 0.01
    A = rng.standard_normal((n, n))
    Sigma = A @ A.T + np.eye(n)

    w_gmv = gmv_weights(Sigma)
    assert np.isclose(w_gmv.sum(), 1.0)

    w_tan = tangency_weights(mu, Sigma)
    assert np.isclose(np.abs(w_tan).sum(), 1.0)

    w_ls = long_short_weights(mu, gross_exposure=0.5)
    assert np.isclose(w_ls.sum(), 0.0, atol=1e-8)
    assert np.isclose(np.abs(w_ls).sum(), 1.0)


def test_sharpe_and_volatility_annualization():
    monthly = np.array([0.01] * 24)
    # zero-variance series: Sharpe is defined as 0 by convention here
    assert annualized_sharpe(monthly) == 0.0
    rng = np.random.default_rng(4)
    monthly_random = rng.standard_normal(120) * 0.02 + 0.01
    sr = annualized_sharpe(monthly_random)
    vol = annualized_volatility(monthly_random)
    assert vol > 0
    assert np.isclose(sr, np.sqrt(12) * monthly_random.mean() / monthly_random.std(ddof=1))


def test_turnover_and_net_return_and_breakeven():
    w_prev = np.array([0.5, 0.5])
    r_t = np.array([0.1, -0.1])
    w_tilde = drifted_previous_weights(w_prev, r_t, rf_t=0.0)
    assert np.isclose(w_tilde.sum(), w_prev.sum(), atol=1e-6)

    w_t = np.array([0.6, 0.4])
    to = turnover(w_t, w_tilde)
    assert to >= 0
    net = net_return(gross_return=0.02, turnover_t=to, cost_bps=10.0)
    assert net < 0.02

    be = break_even_cost_bps(np.array([0.01, 0.02, 0.0]), np.array([0.1, 0.2, 0.05]))
    assert be > 0


def test_newey_west_matches_sample_variance_at_zero_lags():
    rng = np.random.default_rng(5)
    x = rng.standard_normal(500)
    lrv = newey_west_long_run_variance(x, n_lags=0)
    assert np.isclose(lrv, np.var(x), rtol=0.05)


def test_paired_difference_detects_known_shift():
    rng = np.random.default_rng(6)
    a = rng.standard_normal(500) + 1.0
    b = rng.standard_normal(500)
    mean_diff, p_value = paired_difference_test(a, b, n_lags=12)
    assert mean_diff > 0.5
    assert p_value < 0.01


def test_sharpe_delta_method_runs_and_sign():
    rng = np.random.default_rng(7)
    a = rng.standard_normal(300) * 0.02 + 0.02
    b = rng.standard_normal(300) * 0.02
    sr_diff, p_value = sharpe_ratio_delta_method_test(a, b, n_lags=12)
    assert sr_diff > 0
    assert 0 <= p_value <= 1


def test_holm_adjust_controls_family_wise_error():
    p_values = [0.001, 0.02, 0.5, 0.8]
    p_adj, reject = holm_adjust(p_values, alpha=0.05)
    assert len(p_adj) == 4
    assert reject[0]  # strongly significant p-value should survive adjustment
    assert all(p_adj[i] >= p_values[i] for i in range(4))
