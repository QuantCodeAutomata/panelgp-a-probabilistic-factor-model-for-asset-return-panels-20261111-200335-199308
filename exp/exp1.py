"""Experiment 1: recovery of smooth maps, realized states, and variances in
simulated PanelGP data (paper Section 5, Table 1, Figure 1, Figure S1).

Run: python -m exp.exp1 [--replications N] [--epochs N]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from data.generate_synthetic import (
    generate_panel, panel_to_rows, true_loading_maps, true_factor_maps,
)
from src import config as cfg
from src.evaluation import (
    r_squared, align_factors, oracle_conditional_variance,
    joint_loading_factor_variance,
)
from src.panelgp_model import PanelGPModel
from src.training import train_model

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def dense_grid(resolution: int) -> np.ndarray:
    """Regular grid on [0,1]^2 for loading-function R^2 (paper Sec. 5.2)."""
    axis = np.linspace(0.0, 1.0, resolution)
    g1, g2 = np.meshgrid(axis, axis, indexing="ij")
    return np.stack([g1.ravel(), g2.ravel()], axis=-1)


def dense_context_grid(resolution: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, resolution).reshape(-1, 1)


def run_one_replication(seed: int, n_epochs: int) -> dict:
    panel = generate_panel(seed)
    X_rows, date_idx, r_rows = panel_to_rows(panel)
    X_t = torch.as_tensor(X_rows, dtype=torch.float64)
    date_idx_t = torch.as_tensor(date_idx, dtype=torch.long)
    r_t = torch.as_tensor(r_rows, dtype=torch.float64)
    C_train = torch.as_tensor(panel["C"], dtype=torch.float64)

    torch.set_default_dtype(torch.float64)
    P, Q, H = 2, 1, cfg.SIM_N_FACTORS
    T = cfg.SIM_N_MONTHS
    M = cfg.SIM_M_INDUCING

    rng = np.random.default_rng(seed)
    z_rows = rng.choice(X_rows.shape[0], size=(H, M), replace=True)
    Z_init = torch.as_tensor(X_rows[z_rows], dtype=torch.float64)  # (H,M,P)

    model = PanelGPModel(
        P=P, Q=Q, T=T, H=H, M=M, loading_kernel="rbf", factor_kernel="rbf",
        isotropic_innovations=True, Z_init=Z_init, C_train=C_train,
        loading_diag_init=1.0,
    )

    log = train_model(
        model, X_t, date_idx_t, r_t, n_epochs=n_epochs,
        batch_size=cfg.SIM_BATCH_SIZE, n_mc=cfg.SIM_N_MC_SAMPLES,
        lr_variational=cfg.SIM_LR_VARIATIONAL, lr_model=cfg.SIM_LR_MODEL,
        decay_fractions=cfg.SIM_LR_DECAY_FRACTIONS,
        grad_clip_norm=cfg.SIM_GRAD_CLIP_NORM, seed=seed,
    )

    model.eval()
    with torch.no_grad():
        # --- Factor function recovery on a dense grid (paper Sec. 5.1-5.2) ---
        c_grid = dense_context_grid(cfg.SIM_GRID_RESOLUTION)
        c_grid_t = torch.as_tensor(c_grid, dtype=torch.float64)
        phi_true_grid = true_factor_maps(c_grid)
        m_phi_grid, _ = model.predict_factor(c_grid_t, add_innovation=False)
        m_phi_grid_np = m_phi_grid.numpy()

        # --- Loading function recovery on a dense grid ---
        x_grid = dense_grid(cfg.SIM_GRID_RESOLUTION)
        x_grid_t = torch.as_tensor(x_grid, dtype=torch.float64)
        beta_true_grid = true_loading_maps(x_grid)
        m_beta_grid, _ = model.predict_loading(x_grid_t)
        m_beta_grid_np = m_beta_grid.numpy()

        # --- Align estimated factors to true ones by least squares (Sec. 5.1) ---
        m_phi_train, S_phi_train = model.realized_factors_at_training_dates()
        Upsilon = align_factors(panel["phi_tilde"], m_phi_train.numpy())
        Upsilon_inv_T = np.linalg.inv(Upsilon).T

        m_phi_grid_aligned = m_phi_grid_np @ Upsilon.T
        m_beta_grid_aligned = m_beta_grid_np @ Upsilon_inv_T

        factor_r2 = np.mean([
            r_squared(phi_true_grid[:, h], m_phi_grid_aligned[:, h]) for h in range(H)
        ])
        loading_r2 = np.mean([
            r_squared(beta_true_grid[:, h], m_beta_grid_aligned[:, h]) for h in range(H)
        ])

        # --- Innovation variances, transformed by the alignment (Sec. 5.1):
        # phi_aligned = Upsilon @ phi_est, beta_aligned = Upsilon^{-T} @ beta_est,
        # so D_phi (factor's own innovation) transforms by Upsilon on both
        # sides and D_beta (loading's own innovation) by Upsilon^{-T}. ---
        tau_beta = model.tau_beta().numpy()  # estimated D_beta diagonal
        rho = model.factor_shares()[2].numpy()  # estimated D_phi diagonal
        D_beta_aligned = Upsilon_inv_T @ np.diag(tau_beta) @ Upsilon_inv_T.T
        D_phi_aligned = Upsilon @ np.diag(rho) @ Upsilon.T
        sigma_beta2_hat = np.mean(np.diag(D_beta_aligned))
        sigma_phi2_hat = np.mean(np.diag(D_phi_aligned))
        sigma_r2_hat = model.sigma_r2().item()

        # --- Predictive mean / variance over the training panel (Sec. 5.2) ---
        X_rows_t = torch.as_tensor(X_rows, dtype=torch.float64)
        m_beta_rows, S_beta_rows = model.predict_loading(X_rows_t)
        m_phi_rows_all, S_phi_rows_all = model.realized_factors_at_training_dates()
        m_phi_rows = m_phi_rows_all[date_idx]
        S_phi_rows = S_phi_rows_all[date_idx]

        pred_mean = (m_beta_rows * m_phi_rows).sum(-1).numpy()
        true_cond_mean = np.einsum(
            "nh,nh->n", true_loading_maps(X_rows), panel["phi_tilde"][date_idx])
        mean_r2 = r_squared(true_cond_mean, pred_mean)

        V0 = joint_loading_factor_variance(model, X_rows_t, date_idx)
        V_phi = np.einsum("nh,h,nh->n", m_beta_rows.numpy(), rho, m_beta_rows.numpy()) \
            + np.einsum("nhh->n", S_beta_rows.numpy() * rho[None, :, None])
        V_beta = np.einsum("nh,h,nh->n", m_phi_rows.numpy(), tau_beta, m_phi_rows.numpy()) \
            + np.einsum("nhh->n", S_phi_rows.numpy() * tau_beta[None, :, None])
        cross = np.sum(tau_beta * rho)
        pred_var = V0 + V_phi + V_beta + cross + sigma_r2_hat

        oracle_var = oracle_conditional_variance(
            panel["sigma_beta2"], panel["sigma_phi2"], panel["sigma_r2"],
            true_loading_maps(X_rows), panel["phi_tilde"][date_idx])
        var_corr = np.corrcoef(pred_var, oracle_var)[0, 1]
        var_ratio = pred_var.mean() / oracle_var.mean()

        # --- Realized factor innovation recovery at training dates (Sec. 5.2) ---
        m_phi_train_aligned = m_phi_train.numpy() @ Upsilon.T
        true_innovation = panel["phi_tilde"] - panel["phi_smooth"]
        fitted_smooth_at_train, _ = model.predict_factor(C_train, add_innovation=False)
        fitted_smooth_at_train_aligned = fitted_smooth_at_train.numpy() @ Upsilon.T
        recovered_innovation = m_phi_train_aligned - fitted_smooth_at_train_aligned
        innovation_r2 = [
            r_squared(true_innovation[:, h], recovered_innovation[:, h]) for h in range(H)
        ]

    return {
        "seed": seed,
        "factor_function_r2": float(factor_r2),
        "loading_function_r2": float(loading_r2),
        "predictive_mean_r2": float(mean_r2),
        "predictive_variance_correlation": float(var_corr),
        "predictive_variance_ratio": float(var_ratio),
        "sigma_beta2_hat": float(sigma_beta2_hat),
        "sigma_phi2_hat": float(sigma_phi2_hat),
        "sigma_r2_hat": float(sigma_r2_hat),
        "innovation_r2_factor1": float(innovation_r2[0]),
        "innovation_r2_factor2": float(innovation_r2[1]),
        "wall_time_sec": log["wall_time_sec"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=cfg.SIM_N_REPLICATIONS)
    parser.add_argument("--epochs", type=int, default=cfg.SIM_EPOCHS_RUN)
    parser.add_argument("--full", action="store_true",
                         help="Use the paper-scale epoch count.")
    args = parser.parse_args()
    n_epochs = cfg.SIM_EPOCHS_PAPER if args.full else args.epochs

    results = []
    for rep in range(args.replications):
        seed = cfg.RANDOM_SEED_BASE + rep
        t0 = time.time()
        res = run_one_replication(seed, n_epochs)
        print(f"replication {rep}: {res}  ({time.time()-t0:.1f}s)")
        results.append(res)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "exp1_replications.json", "w") as f:
        json.dump(results, f, indent=2)

    keys = [k for k in results[0] if k not in ("seed",)]
    summary = {k: {"mean": float(np.mean([r[k] for r in results])),
                    "std": float(np.std([r[k] for r in results]))}
               for k in keys}
    with open(RESULTS_DIR / "exp1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
