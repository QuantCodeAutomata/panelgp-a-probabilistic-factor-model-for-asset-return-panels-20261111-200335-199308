"""Shared data-preparation, estimation and forecasting pipeline for the
empirical application (paper Section 6), used by both the fixed-training
experiment (exp2) and the rolling re-estimation experiment (exp3).

Implements: input transforms (Sec. 6.1), PanelGP training-tensor assembly,
inducing-point initialization (App. C.3-C.4), the per-month out-of-sample
forecast of the predictive mean and covariance for both PanelGP (Eq. 19-20,
24-27) and IPCA (Sec. 6.2), and the evaluation criteria of Section 6.3
(pooled predictive R^2, monthly rank correlation, Gaussian log score) plus
the portfolio formation of Section 6.3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from scipy import stats

from src import config as cfg
from src.characteristics import (
    rank_transform_characteristics, filter_characteristics_by_coverage,
    filter_rows_by_row_coverage, standardize_context, compute_return_scale,
)
from src.panelgp_model import PanelGPModel
from src.training import train_model
from src.evaluation import variance_decomposition
from src.ipca import fit_ipca, ipca_forecast, IPCAFit
from src.portfolio import (
    gmv_weights, tangency_weights, long_short_weights, annualized_sharpe,
    annualized_volatility,
)


@dataclass
class PreparedPanel:
    """Merged, transformed (date, ticker) panel ready for estimation.

    ``panel`` has columns date, ticker, <char_cols> (ranked to [-0.5,0.5]),
    excess_return (original units), scaled_return (= excess_return / s_r).
    ``context`` is indexed by date with standardized columns in
    ``context_cols``. All dates are month-end timestamps.
    """

    panel: pd.DataFrame
    context: pd.DataFrame
    char_cols: list[str]
    context_cols: list[str]
    s_r: float
    train_dates: np.ndarray
    valid_dates: np.ndarray
    eval_dates: np.ndarray


def prepare_panel(raw: dict, train_start: str, train_end: str,
                   valid_start: str, valid_end: str,
                   eval_start: str, eval_end: str) -> PreparedPanel:
    """Applies the Section 6.1 input transforms to the raw panel returned by
    ``data.data_loader.build_empirical_panel`` and splits it into
    train/validation/evaluation date ranges."""
    char_cols_raw = raw["char_cols"]
    context_cols = raw["context_cols"]

    merged = raw["characteristics"].merge(raw["returns"], on=["date", "ticker"], how="inner")
    merged["date"] = pd.to_datetime(merged["date"])
    train_mask_all = (merged["date"] >= pd.Timestamp(train_start)) & (merged["date"] <= pd.Timestamp(train_end))

    ranked = rank_transform_characteristics(merged, char_cols_raw)
    char_cols = filter_characteristics_by_coverage(
        ranked, char_cols_raw, train_mask_all, cfg.APP_MIN_CHAR_COVERAGE)
    ranked = filter_rows_by_row_coverage(ranked, char_cols, cfg.APP_MIN_ROW_COVERAGE)

    context = raw["context"].copy()
    context.index = pd.to_datetime(context.index)
    context_train_mask = (context.index >= pd.Timestamp(train_start)) & (context.index <= pd.Timestamp(train_end))
    context_std, _, _ = standardize_context(context[context_cols], context_train_mask)

    train_mask = (ranked["date"] >= pd.Timestamp(train_start)) & (ranked["date"] <= pd.Timestamp(train_end))
    s_r = compute_return_scale(ranked["excess_return"].values, train_mask.values)
    ranked["scaled_return"] = ranked["excess_return"] / s_r

    all_dates = np.sort(ranked["date"].unique())
    train_dates = all_dates[(all_dates >= np.datetime64(train_start)) & (all_dates <= np.datetime64(train_end))]
    valid_dates = all_dates[(all_dates >= np.datetime64(valid_start)) & (all_dates <= np.datetime64(valid_end))]
    eval_dates = all_dates[(all_dates >= np.datetime64(eval_start)) & (all_dates <= np.datetime64(eval_end))]

    return PreparedPanel(panel=ranked.reset_index(drop=True), context=context_std,
                          char_cols=char_cols, context_cols=context_cols, s_r=s_r,
                          train_dates=train_dates, valid_dates=valid_dates, eval_dates=eval_dates)


def build_training_tensors(prep: PreparedPanel):
    """Assembles the training-row tensor X, per-row training-date index,
    scaled-return tensor r, and the (T,Q) training context matrix C_train
    used as the PanelGP model's realized-factor state (paper Sec. 4.1)."""
    train_panel = prep.panel[prep.panel["date"].isin(prep.train_dates)].sort_values(["date", "ticker"])
    date_to_idx = {d: i for i, d in enumerate(prep.train_dates)}
    X = train_panel[prep.char_cols].values.astype(np.float64)
    date_idx = train_panel["date"].map(date_to_idx).values.astype(np.int64)
    r = train_panel["scaled_return"].values.astype(np.float64)
    C_train = prep.context.loc[prep.train_dates, prep.context_cols].values.astype(np.float64)
    return X, date_idx, r, C_train, date_to_idx


def init_inducing_points(X: np.ndarray, H: int, M: int, seed: int) -> np.ndarray:
    """Places M trainable inducing inputs per factor at distinct training
    rows (paper Appendix C.3)."""
    rng = np.random.default_rng(seed)
    rows = rng.choice(X.shape[0], size=(H, M), replace=X.shape[0] < M)
    return X[rows]


def fit_panelgp(X: np.ndarray, date_idx: np.ndarray, r: np.ndarray, C_train: np.ndarray,
                 H: int, loading_kernel: str, factor_kernel: str, seed: int,
                 M: int, n_epochs: int, batch_size: int, n_mc: int,
                 lr_variational: float, lr_model: float,
                 decay_fractions: tuple, grad_clip_norm: float) -> PanelGPModel:
    """Trains one PanelGP kernel-pair model on the training tensors (paper
    Sec. 6.1, App. C.4). Loading/factor innovation variances are
    component-specific (``isotropic_innovations=False``, App. C.2 "In the
    application")."""
    torch.set_default_dtype(torch.float64)
    P, Q, T = X.shape[1], C_train.shape[1], C_train.shape[0]
    Z_init = torch.as_tensor(init_inducing_points(X, H, M, seed), dtype=torch.float64)
    if loading_kernel == "affine":
        loading_diag_init = 1.0 + float(np.mean(np.sum(X ** 2, axis=1)))
    else:
        loading_diag_init = 1.0

    model = PanelGPModel(
        P=P, Q=Q, T=T, H=H, M=M, loading_kernel=loading_kernel,
        factor_kernel=factor_kernel, isotropic_innovations=False,
        Z_init=Z_init, C_train=torch.as_tensor(C_train, dtype=torch.float64),
        loading_diag_init=loading_diag_init,
    )
    train_model(
        model, torch.as_tensor(X, dtype=torch.float64), torch.as_tensor(date_idx, dtype=torch.long),
        torch.as_tensor(r, dtype=torch.float64), n_epochs=n_epochs, batch_size=batch_size,
        n_mc=n_mc, lr_variational=lr_variational, lr_model=lr_model,
        decay_fractions=decay_fractions, grad_clip_norm=grad_clip_norm, seed=seed,
    )
    model.eval()
    return model


def _d_phi_d_beta(model: PanelGPModel):
    rho = model.factor_shares()[2].detach().numpy()
    tau_beta = model.tau_beta().detach().numpy()
    sigma_r2 = float(model.sigma_r2().item())
    return rho, tau_beta, sigma_r2


def forecast_panelgp_month(model: PanelGPModel, X_t: np.ndarray, c_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predictive mean and covariance for one out-of-sample month (paper
    Eq. 19-20, 24-27), in SCALED return units.

    ``Sigma_hat = B D_phi B^T + diag(V0 + tr(S_beta D_phi) + V_beta + cross + sigma_r2)``,
    with ``B`` the (N,H) matrix of loading posterior means. The paper gives
    this decomposition for the variance of a single row (Eq. 25-27); the
    off-diagonal term ``B D_phi B^T`` extends it to the full cross-sectional
    covariance needed for portfolio construction by the shared realized
    factor innovation D_phi (common to every asset in month t), which is the
    only source of the paper's variance terms that is shared across assets
    rather than asset-specific (see results/RESULTS.md, "Deviations from the
    paper").
    """
    with torch.no_grad():
        X_t_th = torch.as_tensor(X_t, dtype=torch.float64)
        c_t_th = torch.as_tensor(c_t, dtype=torch.float64).reshape(1, -1)
        m_beta, S_beta = model.predict_loading(X_t_th)
        m_phi, S_phi = model.predict_factor(c_t_th, add_innovation=False)
        m_beta, S_beta = m_beta.numpy(), S_beta.numpy()
        m_phi, S_phi = m_phi.numpy()[0], S_phi.numpy()[0]
    rho, tau_beta, sigma_r2 = _d_phi_d_beta(model)
    D_phi = np.diag(rho)
    N = m_beta.shape[0]

    mu_hat = m_beta @ m_phi
    # V0 = Var_q(beta(x)^T phi(c)) ignoring the posterior cross-covariance
    # between beta(x) and phi(c) (ignored here because predict_loading and
    # predict_factor are evaluated independently out of sample; the exact
    # joint term is used in-sample via evaluation.joint_loading_factor_variance,
    # see results/RESULTS.md, "Deviations from the paper").
    V0 = (np.einsum("h,nhg,g->n", m_phi, S_beta, m_phi)
          + np.einsum("nh,hg,ng->n", m_beta, S_phi, m_beta)
          + np.einsum("nhg,gh->n", S_beta, S_phi))
    decomp = variance_decomposition(
        m_beta, S_beta, np.tile(m_phi, (N, 1)), np.tile(S_phi, (N, 1, 1)),
        tau_beta, rho, sigma_r2, V0)
    diag_no_phi_cross = decomp["total"] - decomp["V_phi"] + np.einsum("nhh->n", S_beta @ D_phi)
    Sigma_hat = m_beta @ D_phi @ m_beta.T + np.diag(diag_no_phi_cross)
    return mu_hat, Sigma_hat


def pooled_predictive_r2(residuals: np.ndarray, returns: np.ndarray) -> float:
    """Pooled predictive R^2 = 1 - sum(resid^2)/sum(r^2) (paper Sec. 6.3)."""
    return float(1.0 - np.sum(residuals ** 2) / np.sum(returns ** 2))


def monthly_rank_correlation(mu_by_month: list[np.ndarray], r_by_month: list[np.ndarray]) -> float:
    """Average monthly Spearman rank correlation between predicted and
    realized returns (paper Sec. 6.3)."""
    corrs = []
    for mu, r in zip(mu_by_month, r_by_month):
        if len(mu) > 1 and np.std(mu) > 0:
            corrs.append(stats.spearmanr(mu, r).correlation)
    return float(np.nanmean(corrs))


def gaussian_log_score(mu_hat: np.ndarray, Sigma_hat: np.ndarray, r: np.ndarray) -> float:
    """Per-month Gaussian log score l_t (paper Sec. 6.3, Eq. in Section 6.3):
    l_t = -(N_t log(2 pi) + log|Sigma_hat_t| + (r-mu)^T Sigma_hat^-1 (r-mu)) / (2 N_t)."""
    N_t = len(r)
    sign, logdet = np.linalg.slogdet(Sigma_hat)
    assert sign > 0, "Sigma_hat must be positive definite"
    resid = r - mu_hat
    quad = resid @ np.linalg.solve(Sigma_hat, resid)
    return float(-(N_t * np.log(2 * np.pi) + logdet + quad) / (2 * N_t))


@dataclass
class MonthEvaluation:
    dates: list
    mu_hat: list = field(default_factory=list)       # per-month (N_t,) in ORIGINAL return units
    Sigma_hat: list = field(default_factory=list)     # per-month (N_t,N_t) in ORIGINAL return units
    r: list = field(default_factory=list)             # per-month realized excess returns (original units)
    tickers: list = field(default_factory=list)       # per-month ticker order


def evaluate_months(forecast_fn, prep: PreparedPanel, dates: np.ndarray) -> MonthEvaluation:
    """Runs ``forecast_fn(X_t, c_t) -> (mu_hat_scaled, Sigma_hat_scaled)`` for
    every month in ``dates`` and rescales to original return units
    (paper Sec. 6.1: "multiply predictive means by s_r, covariances by
    s_r^2")."""
    out = MonthEvaluation(dates=list(dates))
    for d in dates:
        month_rows = prep.panel[prep.panel["date"] == d].sort_values("ticker")
        X_t = month_rows[prep.char_cols].values.astype(np.float64)
        c_t = prep.context.loc[d, prep.context_cols].values.astype(np.float64)
        mu_s, Sigma_s = forecast_fn(X_t, c_t)
        out.mu_hat.append(mu_s * prep.s_r)
        out.Sigma_hat.append(Sigma_s * prep.s_r ** 2)
        out.r.append(month_rows["excess_return"].values.astype(np.float64))
        out.tickers.append(month_rows["ticker"].values)
    return out


def summarize_evaluation(ev: MonthEvaluation, cost_bps: float = 0.0,
                          risk_free: pd.Series | None = None) -> dict:
    """Computes the Section 6.3 evaluation criteria and the three monthly
    portfolios, pooling squared errors/log scores over months."""
    resid_all = np.concatenate([r - mu for r, mu in zip(ev.r, ev.mu_hat)])
    r_all = np.concatenate(ev.r)
    r2 = pooled_predictive_r2(resid_all, r_all)
    rank_corr = monthly_rank_correlation(ev.mu_hat, ev.r)
    log_scores = [gaussian_log_score(mu, S, r) for mu, S, r in zip(ev.mu_hat, ev.Sigma_hat, ev.r)]
    log_score = float(np.mean(log_scores))

    gmv_rets, tan_rets, ls_rets = [], [], []
    gmv_net, tan_net, ls_net = [], [], []
    turnovers = {"gmv": [], "tan": [], "ls": []}
    prev_w = {"gmv": None, "tan": None, "ls": None}
    for i, (mu, S, r, tickers, d) in enumerate(zip(ev.mu_hat, ev.Sigma_hat, ev.r, ev.tickers, ev.dates)):
        w_gmv = gmv_weights(S)
        w_tan = tangency_weights(mu, S)
        w_ls = long_short_weights(mu, gross_exposure=cfg.APP_LONG_SHORT_GROSS)
        gmv_rets.append(float(w_gmv @ r))
        tan_rets.append(float(w_tan @ r))
        ls_rets.append(float(w_ls @ r))
        if risk_free is not None:
            from src.portfolio import drifted_previous_weights, turnover, net_return
            rf_t = float(risk_free.get(d, 0.0))
            for name, w, w_prev, gross in (
                ("gmv", w_gmv, prev_w["gmv"], gmv_rets[-1]),
                ("tan", w_tan, prev_w["tan"], tan_rets[-1]),
                ("ls", w_ls, prev_w["ls"], ls_rets[-1]),
            ):
                if w_prev is not None and len(w_prev) == len(r):
                    w_tilde = drifted_previous_weights(w_prev, r, rf_t)
                    to = turnover(w, w_tilde)
                else:
                    to = float(np.sum(np.abs(w)))
                turnovers[name].append(to)
                net = net_return(gross, to, cost_bps)
                {"gmv": gmv_net, "tan": tan_net, "ls": ls_net}[name].append(net)
            prev_w = {"gmv": w_gmv, "tan": w_tan, "ls": w_ls}

    result = {
        "predictive_r2_pct": r2 * 100.0,
        "rank_correlation_pct": rank_corr * 100.0,
        "log_score": log_score,
        "tangency_sharpe": annualized_sharpe(np.array(tan_rets)),
        "long_short_sharpe": annualized_sharpe(np.array(ls_rets)),
        "gmv_volatility_pct": annualized_volatility(np.array(gmv_rets)) * 100.0,
        "gmv_returns": gmv_rets, "tangency_returns": tan_rets, "long_short_returns": ls_rets,
    }
    if risk_free is not None:
        result["tangency_net_sharpe"] = annualized_sharpe(np.array(tan_net))
        result["long_short_net_sharpe"] = annualized_sharpe(np.array(ls_net))
        result["tangency_turnover"] = turnovers["tan"]
        result["long_short_turnover"] = turnovers["ls"]
        result["tangency_net_returns"] = tan_net
        result["long_short_net_returns"] = ls_net
    return result


def ipca_month_forecasts(fit: IPCAFit, prep: PreparedPanel, dates: np.ndarray) -> MonthEvaluation:
    out = MonthEvaluation(dates=list(dates))
    for d in dates:
        month_rows = prep.panel[prep.panel["date"] == d].sort_values("ticker")
        X_t = month_rows[prep.char_cols].values.astype(np.float64)
        mu_s, Sigma_s = ipca_forecast(fit, X_t)
        out.mu_hat.append(mu_s * prep.s_r)
        out.Sigma_hat.append(Sigma_s * prep.s_r ** 2)
        out.r.append(month_rows["excess_return"].values.astype(np.float64))
        out.tickers.append(month_rows["ticker"].values)
    return out


def fit_ipca_on_training(prep: PreparedPanel, H: int, with_intercept: bool, seed: int) -> IPCAFit:
    X_by_month, r_by_month = [], []
    for d in prep.train_dates:
        rows = prep.panel[prep.panel["date"] == d].sort_values("ticker")
        X_by_month.append(rows[prep.char_cols].values.astype(np.float64))
        r_by_month.append(rows["scaled_return"].values.astype(np.float64))
    return fit_ipca(X_by_month, r_by_month, H, with_intercept=with_intercept, seed=seed)
