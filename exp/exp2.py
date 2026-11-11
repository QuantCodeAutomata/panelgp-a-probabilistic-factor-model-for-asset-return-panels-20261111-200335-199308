"""Experiment 2: fixed-training kernel comparison and factor-number analysis
on S&P 500 constituents (paper Section 6.1-6.5, Table 2, Figures 2-3).

Run: python -m exp.exp2 [--full]

``--full`` selects the paper-scale universe, date split, inducing points,
epochs and seeds (``*_PAPER`` in ``src/config.py``); by default the executed
(``*_RUN``) configuration is used, documented in ``results/RESULTS.md``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from data.data_loader import build_empirical_panel, load_sp500_universe
from src import config as cfg
from src.empirical_pipeline import (
    prepare_panel, build_training_tensors, fit_panelgp, fit_ipca_on_training,
    evaluate_months, ipca_month_forecasts, summarize_evaluation,
    forecast_panelgp_month,
)
from src.significance import (
    paired_difference_test, sharpe_ratio_delta_method_test, holm_adjust,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

KERNEL_PAIRS = [
    ("affine", "constant"), ("affine", "rbf"),
    ("rbf", "constant"), ("rbf", "rbf"),
]


def load_prepared(full: bool):
    universe_size = cfg.APP_UNIVERSE_SIZE_PAPER if full else cfg.APP_UNIVERSE_SIZE_RUN
    train_start = cfg.APP_TRAIN_START_PAPER if full else cfg.APP_TRAIN_START_RUN
    train_end = cfg.APP_TRAIN_END_PAPER if full else cfg.APP_TRAIN_END_RUN
    valid_start = cfg.APP_VALID_START_PAPER if full else cfg.APP_VALID_START_RUN
    valid_end = cfg.APP_VALID_END_PAPER if full else cfg.APP_VALID_END_RUN
    eval_start = cfg.APP_EVAL_START_PAPER if full else cfg.APP_EVAL_START_RUN
    eval_end = cfg.APP_EVAL_END_PAPER if full else cfg.APP_EVAL_END_RUN

    uni = load_sp500_universe(universe_size, train_start, eval_end,
                               lookback_months=cfg.APP_LIQUIDITY_LOOKBACK_MONTHS)
    tickers = uni["ticker"].tolist()
    raw = build_empirical_panel(tickers, train_start, eval_end)
    prep = prepare_panel(raw, train_start, train_end, valid_start, valid_end, eval_start, eval_end)
    print(f"universe: {len(tickers)} tickers; characteristics kept: {len(prep.char_cols)}/"
          f"{len(raw['char_cols'])}; train months: {len(prep.train_dates)}; "
          f"valid months: {len(prep.valid_dates)}; eval months: {len(prep.eval_dates)}")
    return prep, raw


def fit_and_evaluate_panelgp(prep, H, loading_kernel, factor_kernel, seed, n_epochs, M):
    X, date_idx, r, C_train, _ = build_training_tensors(prep)
    t0 = time.time()
    model = fit_panelgp(
        X, date_idx, r, C_train, H, loading_kernel, factor_kernel, seed, M,
        n_epochs=n_epochs, batch_size=cfg.APP_BATCH_SIZE, n_mc=cfg.APP_N_MC_SAMPLES,
        lr_variational=cfg.APP_LR_VARIATIONAL, lr_model=cfg.APP_LR_MODEL,
        decay_fractions=cfg.APP_LR_DECAY_FRACTIONS, grad_clip_norm=cfg.APP_GRAD_CLIP_NORM,
    )
    fit_time = time.time() - t0

    def forecast_fn(X_t, c_t):
        return forecast_panelgp_month(model, X_t, c_t)

    ev_valid = evaluate_months(forecast_fn, prep, prep.valid_dates)
    ev_eval = evaluate_months(forecast_fn, prep, prep.eval_dates)
    summary_valid = summarize_evaluation(ev_valid)
    summary_eval = summarize_evaluation(ev_eval)
    return model, summary_valid, summary_eval, ev_eval, fit_time


def fit_and_evaluate_ipca(prep, H, with_intercept, seed):
    fit = fit_ipca_on_training(prep, H, with_intercept, seed)

    def forecast_fn(dates):
        return ipca_month_forecasts(fit, prep, dates)

    ev_valid = forecast_fn(prep.valid_dates)
    ev_eval = forecast_fn(prep.eval_dates)
    return summarize_evaluation(ev_valid), summarize_evaluation(ev_eval), ev_eval


def run_pairwise_tests(evals: dict, summaries: dict) -> dict:
    """Pairwise Newey-West + Holm tests among all models for each criterion
    (paper Sec. 6.3, App. D.3): R^2 (monthly squared-error sums), log score,
    tangency/long-short Sharpe (delta method)."""
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
        "long_short_sharpe": lambda a, b: sharpe_ratio_delta_method_test(
            np.array(summaries[a]["long_short_returns"]), np.array(summaries[b]["long_short_returns"]), cfg.APP_NW_LAGS),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Use the paper-scale configuration.")
    args = parser.parse_args()
    full = args.full

    n_seeds = cfg.APP_N_SEEDS_PAPER if full else cfg.APP_N_SEEDS_RUN
    n_epochs = cfg.APP_EPOCHS_PAPER if full else cfg.APP_EPOCHS_RUN
    M = cfg.APP_M_INDUCING
    H_sel = cfg.APP_H_SELECTED
    H_grid = cfg.APP_H_GRID if full else cfg.APP_H_GRID_RUN

    prep, raw = load_prepared(full)

    table2 = {}
    extra_evals, extra_summaries = {}, {}
    t_start = time.time()
    for loading_kernel, factor_kernel in KERNEL_PAIRS:
        name = f"{loading_kernel}_{factor_kernel}"
        seed_rows = []
        for s in range(n_seeds):
            seed = cfg.RANDOM_SEED_BASE + s
            print(f"[exp2] fitting PanelGP {name} H={H_sel} seed={seed} ...", flush=True)
            _, _, summary_eval, ev_eval, fit_time = fit_and_evaluate_panelgp(
                prep, H_sel, loading_kernel, factor_kernel, seed, n_epochs, M)
            print(f"[exp2]  -> {name}: R2={summary_eval['predictive_r2_pct']:.3f}% "
                  f"rank={summary_eval['rank_correlation_pct']:.2f}% "
                  f"log={summary_eval['log_score']:.3f} ({fit_time:.1f}s)", flush=True)
            seed_rows.append(summary_eval)
            if s == 0:
                extra_evals[name] = ev_eval
                extra_summaries[name] = summary_eval
        table2[name] = {k: {"mean": float(np.mean([r[k] for r in seed_rows])),
                             "std": float(np.std([r[k] for r in seed_rows]))}
                        for k in seed_rows[0] if not k.endswith("_returns")}

    for name, with_intercept in (("ipca", False), ("ipca_int", True)):
        seed_rows = []
        for s in range(n_seeds):
            seed = cfg.RANDOM_SEED_BASE + s
            print(f"[exp2] fitting {name} H={H_sel} seed={seed} ...", flush=True)
            _, summary_eval, ev_eval = fit_and_evaluate_ipca(prep, H_sel, with_intercept, seed)
            seed_rows.append(summary_eval)
            if s == 0:
                extra_evals[name] = ev_eval
                extra_summaries[name] = summary_eval
        table2[name] = {k: {"mean": float(np.mean([r[k] for r in seed_rows])),
                             "std": float(np.std([r[k] for r in seed_rows]))}
                        for k in seed_rows[0] if not k.endswith("_returns")}

    pairwise = run_pairwise_tests(extra_evals, extra_summaries)

    figure3 = {}
    for loading_kernel, factor_kernel in [("rbf", "rbf")]:
        name = f"{loading_kernel}_{factor_kernel}"
        for H in H_grid:
            key = f"{name}_H{H}"
            if H == H_sel:
                figure3[key] = table2[name]
                continue
            print(f"[exp2] fitting PanelGP {name} H={H} (Figure 3 sensitivity) ...", flush=True)
            _, summary_valid, summary_eval, _, fit_time = fit_and_evaluate_panelgp(
                prep, H, loading_kernel, factor_kernel, cfg.RANDOM_SEED_BASE, n_epochs, M)
            figure3[key] = {k: {"mean": float(v), "std": 0.0} for k, v in summary_eval.items()
                             if not isinstance(v, list)}
            figure3[key]["validation"] = {k: float(v) for k, v in summary_valid.items() if not isinstance(v, list)}

    import torch.nn.functional as _F
    ard = {}
    rbf_rbf_name = "rbf_rbf"
    if rbf_rbf_name in table2:
        # re-fit once more to extract ARD lengthscales for reporting (paper Fig. 2)
        X, date_idx, r, C_train, _ = build_training_tensors(prep)
        model = fit_panelgp(X, date_idx, r, C_train, H_sel, "rbf", "rbf", cfg.RANDOM_SEED_BASE, M,
                             n_epochs=n_epochs, batch_size=cfg.APP_BATCH_SIZE, n_mc=cfg.APP_N_MC_SAMPLES,
                             lr_variational=cfg.APP_LR_VARIATIONAL, lr_model=cfg.APP_LR_MODEL,
                             decay_fractions=cfg.APP_LR_DECAY_FRACTIONS, grad_clip_norm=cfg.APP_GRAD_CLIP_NORM)
        ell_beta = _F.softplus(model.raw_ell_beta).detach().numpy()  # (H,P)
        ell_phi = _F.softplus(model.raw_ell_phi).detach().numpy()    # (H,Q)
        inv_ell_beta = (1.0 / ell_beta).mean(axis=0)
        inv_ell_phi = (1.0 / ell_phi).mean(axis=0)
        ard["loading"] = dict(zip(prep.char_cols, inv_ell_beta.tolist()))
        ard["factor"] = dict(zip(prep.context_cols, inv_ell_phi.tolist()))

    total_time = time.time() - t_start
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "config": {"full": full, "n_seeds": n_seeds, "n_epochs": n_epochs, "M": M,
                   "H_selected": H_sel, "H_grid": H_grid, "universe_size": len(raw["char_cols"]) and None},
        "table2": table2, "figure3": figure3, "pairwise_tests": pairwise, "ard_lengthscales": ard,
        "total_time_sec": total_time,
    }
    with open(RESULTS_DIR / "exp2_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("[exp2] table2:", json.dumps(table2, indent=2))
    print(f"[exp2] total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
