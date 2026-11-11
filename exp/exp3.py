"""Experiment 3: annual rolling re-estimation of PanelGP (RBF loadings, RBF
factors, H=5) on S&P 500 constituents, with trading costs, higher-moment
log-score decomposition, and variance-decomposition calibration (paper
Section 6.6-6.8, Tables 3-4).

Run: python -m exp.exp3 [--full]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats

from data.data_loader import build_empirical_panel, load_sp500_universe
from src import config as cfg
from src.bilinear_density import gaussian_log_density, log_density_bilinear
from src.empirical_pipeline import (
    prepare_panel, build_training_tensors, fit_panelgp, fit_ipca_on_training,
    evaluate_months, ipca_month_forecasts, summarize_evaluation,
    forecast_panelgp_month, _d_phi_d_beta,
)
from src.evaluation import variance_decomposition
from src.portfolio import (
    annualized_sharpe, drifted_previous_weights, turnover, net_return,
    break_even_cost_bps, gmv_weights, tangency_weights, long_short_weights,
)
from src.significance import paired_difference_test, sharpe_ratio_delta_method_test, holm_adjust

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def rolling_windows(full: bool):
    """Yields (train_start, train_end, forecast_year) for each annual
    re-estimation (paper Sec. 6.6): window length ``ROLL_WINDOW_MONTHS_*``,
    first/last forecast year ``ROLL_FIRST/LAST_FORECAST_YEAR_*``."""
    window_months = cfg.ROLL_WINDOW_MONTHS_PAPER if full else cfg.ROLL_WINDOW_MONTHS_RUN
    first_year = cfg.ROLL_FIRST_FORECAST_YEAR_PAPER if full else cfg.ROLL_FIRST_FORECAST_YEAR_RUN
    last_year = cfg.ROLL_LAST_FORECAST_YEAR_PAPER if full else cfg.ROLL_LAST_FORECAST_YEAR_RUN
    for year in range(first_year, last_year + 1, cfg.ROLL_REESTIMATION_FREQ_YEARS):
        train_end = pd.Timestamp(f"{year - 1}-12-31")
        train_start = (train_end - pd.DateOffset(months=window_months - 1)).replace(day=1)
        yield train_start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d"), year


def run_one_window(raw, train_start, train_end, forecast_year, H, M, n_epochs, seed):
    eval_start, eval_end = f"{forecast_year}-01-01", f"{forecast_year}-12-31"
    prep = prepare_panel(raw, train_start, train_end, train_start, train_end, eval_start, eval_end)
    X, date_idx, r, C_train, _ = build_training_tensors(prep)
    model = fit_panelgp(X, date_idx, r, C_train, H, "rbf", "rbf", seed, M,
                         n_epochs=n_epochs, batch_size=cfg.APP_BATCH_SIZE, n_mc=cfg.APP_N_MC_SAMPLES,
                         lr_variational=cfg.APP_LR_VARIATIONAL, lr_model=cfg.APP_LR_MODEL,
                         decay_fractions=cfg.APP_LR_DECAY_FRACTIONS, grad_clip_norm=cfg.APP_GRAD_CLIP_NORM)

    def forecast_fn(X_t, c_t):
        return forecast_panelgp_month(model, X_t, c_t)

    ev = evaluate_months(forecast_fn, prep, prep.eval_dates)

    ipca_fit = fit_ipca_on_training(prep, H, False, seed)
    ipca_int_fit = fit_ipca_on_training(prep, H, True, seed)
    ev_ipca = ipca_month_forecasts(ipca_fit, prep, prep.eval_dates)
    ev_ipca_int = ipca_month_forecasts(ipca_int_fit, prep, prep.eval_dates)
    return prep, model, ev, ev_ipca, ev_ipca_int


def trading_cost_table(ev, risk_free: pd.Series, cost_bps: float) -> dict:
    """Net-of-cost Sharpe ratios and break-even costs for the tangency and
    long-short portfolios (paper Sec. 6.6, Eq. for TO_t and r_net)."""
    prev = {"tan": None, "ls": None}
    gross, net, to_hist = {"tan": [], "ls": []}, {"tan": [], "ls": []}, {"tan": [], "ls": []}
    for mu, S, r, d in zip(ev.mu_hat, ev.Sigma_hat, ev.r, ev.dates):
        w_tan = tangency_weights(mu, S)
        w_ls = long_short_weights(mu, gross_exposure=cfg.APP_LONG_SHORT_GROSS)
        rf_t = float(risk_free.get(pd.Timestamp(d), 0.0))
        for name, w in (("tan", w_tan), ("ls", w_ls)):
            g = float(w @ r)
            if prev[name] is not None and len(prev[name]) == len(r):
                w_tilde = drifted_previous_weights(prev[name], r, rf_t)
                to = turnover(w, w_tilde)
            else:
                to = float(np.sum(np.abs(w)))
            gross[name].append(g)
            to_hist[name].append(to)
            net[name].append(net_return(g, to, cost_bps))
            prev[name] = w
    out = {}
    for name in ("tan", "ls"):
        out[f"{name}_gross_sharpe"] = annualized_sharpe(np.array(gross[name]))
        out[f"{name}_net_sharpe"] = annualized_sharpe(np.array(net[name]))
        out[f"{name}_breakeven_bps"] = break_even_cost_bps(np.array(gross[name]), np.array(to_hist[name]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    full = args.full

    H, M = cfg.APP_H_SELECTED, cfg.APP_M_INDUCING
    n_epochs = cfg.APP_EPOCHS_PAPER if full else cfg.APP_EPOCHS_RUN
    universe_size = cfg.APP_UNIVERSE_SIZE_PAPER if full else cfg.APP_UNIVERSE_SIZE_RUN
    window_months = cfg.ROLL_WINDOW_MONTHS_PAPER if full else cfg.ROLL_WINDOW_MONTHS_RUN

    windows = list(rolling_windows(full))
    overall_start = windows[0][0]
    overall_end = f"{windows[-1][2]}-12-31"
    uni = load_sp500_universe(universe_size, overall_start, overall_end,
                               lookback_months=cfg.APP_LIQUIDITY_LOOKBACK_MONTHS)
    tickers = uni["ticker"].tolist()
    raw = build_empirical_panel(tickers, overall_start, overall_end)
    print(f"[exp3] universe={len(tickers)} tickers, {len(windows)} rolling windows "
          f"({windows[0][2]}-{windows[-1][2]}), window length={window_months} months", flush=True)

    all_ev, all_ev_ipca, all_ev_ipca_int = [], [], []
    higher_moment_gains, abs_z_scores = [], []
    var_beta_by_month, var_phi_rows, err_sq_by_month = [], [], []
    t0 = time.time()
    for train_start, train_end, year in windows:
        print(f"[exp3] window train=[{train_start},{train_end}] forecast={year} ...", flush=True)
        seed = cfg.RANDOM_SEED_BASE
        prep, model, ev, ev_ipca, ev_ipca_int = run_one_window(
            raw, train_start, train_end, year, H, M, n_epochs, seed)
        all_ev.append(ev)
        all_ev_ipca.append(ev_ipca)
        all_ev_ipca_int.append(ev_ipca_int)

        rho, tau_beta, sigma_r2 = _d_phi_d_beta(model)
        rng = np.random.default_rng(seed)
        for d, mu, Sigma, r in zip(ev.dates, ev.mu_hat, ev.Sigma_hat, ev.r):
            month_rows = prep.panel[prep.panel["date"] == d].sort_values("ticker")
            X_t_rows = month_rows[prep.char_cols].values.astype(np.float64)
            c_t = prep.context.loc[d, prep.context_cols].values.astype(np.float64)
            with torch.no_grad():
                X_t_th = torch.as_tensor(X_t_rows, dtype=torch.float64)
                c_t_th = torch.as_tensor(c_t, dtype=torch.float64).reshape(1, -1)
                m_beta, S_beta = model.predict_loading(X_t_th)
                m_phi, S_phi = model.predict_factor(c_t_th, add_innovation=False)
                m_beta, S_beta = m_beta.numpy(), S_beta.numpy()
                m_phi, S_phi = m_phi.numpy()[0], S_phi.numpy()[0]

            # V^phi_j (factor-innovation contribution, row-specific:
            # m_beta_j^T D_phi m_beta_j + tr(S_beta_j D_phi)) and V^beta_t
            # (loading-innovation contribution, common to the month:
            # m_phi_t^T D_beta m_phi_t + tr(S_phi_t D_beta)), paper Sec. 6.8,
            # using D_phi=diag(rho) and D_beta=diag(tau_beta) as in
            # src.evaluation.variance_decomposition (Eq. 26-27).
            v_phi_row = m_beta ** 2 @ rho + np.einsum("nhh->n", S_beta @ np.diag(rho))
            var_phi_rows.append(v_phi_row)
            v_beta_month = float(m_phi @ np.diag(tau_beta) @ m_phi + np.trace(S_phi @ np.diag(tau_beta)))
            var_beta_by_month.append(v_beta_month)
            err_sq_by_month.append((r - mu) ** 2)

            n_sample = min(cfg.ROLL_N_QUINTILES * 2, len(r))
            sample_idx = (rng.choice(len(r), size=n_sample, replace=False)
                          if len(r) > n_sample else np.arange(len(r)))
            for i in sample_idx:
                sigma2_i = Sigma[i, i]
                r_orig, mu_i = r[i], mu[i]
                r_scaled = r_orig / prep.s_r
                log_bil = (log_density_bilinear(r_scaled, m_beta[i], S_beta[i], m_phi, S_phi, float(sigma_r2))
                           - float(np.log(prep.s_r)))
                log_gauss = gaussian_log_density(r_orig, mu_i, sigma2_i)
                higher_moment_gains.append(log_bil - log_gauss)
                abs_z_scores.append(abs(r_orig - mu_i) / np.sqrt(max(sigma2_i, 1e-12)))

    summary = summarize_evaluation_pooled(all_ev)
    summary_ipca = summarize_evaluation_pooled(all_ev_ipca)
    summary_ipca_int = summarize_evaluation_pooled(all_ev_ipca_int)

    risk_free = raw["risk_free"]
    ev_concat = concat_eval(all_ev)
    ev_ipca_concat = concat_eval(all_ev_ipca)
    ev_ipca_int_concat = concat_eval(all_ev_ipca_int)
    costs = {
        "panelgp": trading_cost_table(ev_concat, risk_free, cfg.ROLL_TRADING_COST_BPS),
        "ipca": trading_cost_table(ev_ipca_concat, risk_free, cfg.ROLL_TRADING_COST_BPS),
        "ipca_int": trading_cost_table(ev_ipca_int_concat, risk_free, cfg.ROLL_TRADING_COST_BPS),
    }

    evals = {"panelgp_rolling": ev_concat, "ipca_rolling": ev_ipca_concat, "ipca_int_rolling": ev_ipca_int_concat}
    summaries = {"panelgp_rolling": summarize_evaluation(ev_concat),
                 "ipca_rolling": summarize_evaluation(ev_ipca_concat),
                 "ipca_int_rolling": summarize_evaluation(ev_ipca_int_concat)}
    pairwise = run_pairwise(evals, summaries)

    var_beta_arr = np.array(var_beta_by_month)
    err_sq_month_avg = np.array([np.mean(e) for e in err_sq_by_month])
    spearman_var_beta = float(stats.spearmanr(var_beta_arr, err_sq_month_avg).correlation)
    var_phi_flat = np.concatenate(var_phi_rows)
    err_sq_flat = np.concatenate(err_sq_by_month)
    within_month_corrs = []
    for v_row, e_row in zip(var_phi_rows, err_sq_by_month):
        if len(v_row) > 2 and np.std(v_row) > 0:
            within_month_corrs.append(stats.spearmanr(v_row, e_row).correlation)
    spearman_var_phi = float(np.nanmean(within_month_corrs))

    total_time = time.time() - t0
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "config": {"full": full, "n_windows": len(windows), "window_months": window_months,
                   "H": H, "M": M, "n_epochs": n_epochs, "universe_size": len(tickers)},
        "table3": {"panelgp_rolling": summary, "ipca_rolling": summary_ipca, "ipca_int_rolling": summary_ipca_int},
        "table4_trading_costs": costs,
        "pairwise_tests": pairwise,
        "higher_moments": {
            "mean_log_score_gain": float(np.mean(higher_moment_gains)) if higher_moment_gains else None,
            "n_observations": len(higher_moment_gains),
            "mean_gain_abs_z_gt_3": float(np.mean([g for g, z in zip(higher_moment_gains, abs_z_scores) if z > 3])) if any(z > 3 for z in abs_z_scores) else None,
        },
        "variance_calibration": {
            "spearman_var_beta_vs_monthly_avg_err2": spearman_var_beta,
            "spearman_var_phi_vs_err2_within_month": spearman_var_phi,
        },
        "total_time_sec": total_time,
    }
    with open(RESULTS_DIR / "exp3_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("[exp3] table3:", json.dumps(out["table3"], indent=2))
    print(f"[exp3] total time: {total_time:.1f}s")


def concat_eval(ev_list):
    from src.empirical_pipeline import MonthEvaluation
    out = MonthEvaluation(dates=[])
    for ev in ev_list:
        out.dates.extend(ev.dates)
        out.mu_hat.extend(ev.mu_hat)
        out.Sigma_hat.extend(ev.Sigma_hat)
        out.r.extend(ev.r)
        out.tickers.extend(ev.tickers)
    return out


def summarize_evaluation_pooled(ev_list):
    return summarize_evaluation(concat_eval(ev_list))


def run_pairwise(evals, summaries):
    from src.empirical_pipeline import gaussian_log_score
    names = list(evals.keys())
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]

    def monthly_sq_err(ev):
        return np.array([np.sum((r - mu) ** 2) for r, mu in zip(ev.r, ev.mu_hat)])

    def monthly_log_scores(ev):
        return np.array([gaussian_log_score(mu, S, r) for mu, S, r in zip(ev.mu_hat, ev.Sigma_hat, ev.r)])

    criteria = {
        "predictive_r2": lambda a, b: paired_difference_test(-monthly_sq_err(evals[a]), -monthly_sq_err(evals[b]), cfg.APP_NW_LAGS),
        "log_score": lambda a, b: paired_difference_test(monthly_log_scores(evals[a]), monthly_log_scores(evals[b]), cfg.APP_NW_LAGS),
        "tangency_sharpe": lambda a, b: sharpe_ratio_delta_method_test(
            np.array(summaries[a]["tangency_returns"]), np.array(summaries[b]["tangency_returns"]), cfg.APP_NW_LAGS),
    }
    results = {}
    for crit_name, test_fn in criteria.items():
        pvals, diffs = [], []
        for a, b in pairs:
            diff, p = test_fn(a, b)
            diffs.append(diff)
            pvals.append(p)
        p_adj, reject = holm_adjust(pvals, alpha=cfg.APP_SIGNIFICANCE_LEVEL)
        results[crit_name] = [
            {"a": a, "b": b, "diff": float(d), "p_adj": float(p), "significant": bool(r)}
            for (a, b), d, p, r in zip(pairs, diffs, p_adj, reject)
        ]
    return results


if __name__ == "__main__":
    main()
