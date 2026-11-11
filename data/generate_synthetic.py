"""Synthetic panel generator for the simulation study (paper Section 5.1).

Generates a balanced panel of 80 assets and 50 months from the PanelGP
generative model itself (Equation 3, 28), with known smooth loading and
factor maps featuring several local extrema (paper Section 5.1).
"""

from __future__ import annotations

import numpy as np

from src import config as cfg


def asset_characteristics() -> np.ndarray:
    """Builds the 9x9 grid in [0,1]^2 with the upper-right corner omitted,
    giving 80 fixed asset characteristic vectors (paper Section 5.1)."""
    side = cfg.SIM_GRID_SIDE
    coords = np.linspace(0.0, 1.0, side)
    grid = np.array([[x1, x2] for x1 in coords for x2 in coords])
    # Omit the single upper-right corner point (1,1): 81 - 1 = 80 assets.
    mask = ~np.all(np.isclose(grid, 1.0), axis=1)
    X = grid[mask]
    assert X.shape[0] == cfg.SIM_N_ASSETS
    return X


def month_contexts(n_months: int) -> np.ndarray:
    """Scalar monthly context c_t = (t-1)/(T-1), t=1,...,T (paper Sec. 5.1)."""
    t = np.arange(1, n_months + 1)
    return ((t - 1) / (n_months - 1)).reshape(-1, 1)


def true_factor_maps(c: np.ndarray) -> np.ndarray:
    """Smooth factor maps phi_1(c), phi_2(c), paper Section 5.1.

    phi_1(c) = 1/2 + sin(2*pi*c) + 1/2*cos(4*pi*c) + 1/3*sin(6*pi*c)
    phi_2(c) = -1/2 + cos(2*pi*c) - 1/2*sin(4*pi*c) + 1/3*cos(6*pi*c)
    """
    c = c.ravel()
    phi1 = 0.5 + np.sin(2 * np.pi * c) + 0.5 * np.cos(4 * np.pi * c) + (1 / 3) * np.sin(6 * np.pi * c)
    phi2 = -0.5 + np.cos(2 * np.pi * c) - 0.5 * np.sin(4 * np.pi * c) + (1 / 3) * np.cos(6 * np.pi * c)
    return np.stack([phi1, phi2], axis=-1)


def true_loading_maps(x: np.ndarray) -> np.ndarray:
    """Smooth loading maps beta_1(x), beta_2(x), paper Section 5.1.

    beta_1(x) = sin(2*pi*x1)*cos(pi*x2) + 1/2*cos(2*pi*(x1+x2))
                 + 1/3*sin(4*pi*x1)*sin(2*pi*x2)
    beta_2(x) = cos(2*pi*x2)*sin(pi*x1) - 1/2*sin(2*pi*(x1-x2))
                 + 1/3*cos(2*pi*x1)*cos(4*pi*x2)
    """
    x1, x2 = x[:, 0], x[:, 1]
    beta1 = (np.sin(2 * np.pi * x1) * np.cos(np.pi * x2)
             + 0.5 * np.cos(2 * np.pi * (x1 + x2))
             + (1 / 3) * np.sin(4 * np.pi * x1) * np.sin(2 * np.pi * x2))
    beta2 = (np.cos(2 * np.pi * x2) * np.sin(np.pi * x1)
             - 0.5 * np.sin(2 * np.pi * (x1 - x2))
             + (1 / 3) * np.cos(2 * np.pi * x1) * np.cos(4 * np.pi * x2))
    return np.stack([beta1, beta2], axis=-1)


def generate_panel(seed: int) -> dict:
    """Simulates one balanced panel from the model (paper Section 3.1, Eq. 3-8).

    Every input month t = 1,...,T has a realized loading per asset,
    beta_tilde_{i,t} = beta(x_i) + xi_{i,t} with xi_{i,t} ~ N(0, sigma_beta^2 I),
    and a realized factor beta_tilde_t = phi(c_t) + zeta_t with
    zeta_t ~ N(0, sigma_phi^2 I). Each input month predicts one return row,
    r_{i,t+1} = beta_tilde_{i,t}^T phi_tilde_t + sigma_r * epsilon_{i,t+1}
    (paper Section 5.1): both realized states are evaluated at the *input*
    month t, giving exactly N*T = 80*50 = 4,000 rows with no additional
    month needed for the last prediction (paper's general row notation
    r_n = r_{i(n),t(n)+1}, Equation 4).

    Returns a dict with assets (N,2), input-month contexts (T,1), the true
    smooth maps, the realized states at every input month, and the return
    rows.
    """
    rng = np.random.default_rng(seed)
    N, T, H = cfg.SIM_N_ASSETS, cfg.SIM_N_MONTHS, cfg.SIM_N_FACTORS
    X = asset_characteristics()
    C = month_contexts(T)
    beta_smooth = true_loading_maps(X)       # (N,H)
    phi_smooth = true_factor_maps(C)          # (T,H)

    sigma_beta = np.sqrt(cfg.SIM_SIGMA_BETA2)
    sigma_phi = np.sqrt(cfg.SIM_SIGMA_PHI2)
    sigma_r = np.sqrt(cfg.SIM_SIGMA_R2)

    # Independent innovation at every input month (paper Eq. 10-11: the
    # realized states are smooth map plus one fresh innovation, not a
    # cumulative random walk).
    beta_tilde = beta_smooth[:, None, :] + sigma_beta * rng.standard_normal((N, T, H))
    phi_tilde = phi_smooth + sigma_phi * rng.standard_normal((T, H))

    eps = sigma_r * rng.standard_normal((N, T))
    # r_{i,t+1} = beta_tilde_{i,t}^T phi_tilde_t + sigma_r * eps_{i,t+1}.
    returns = np.einsum("ith,th->it", beta_tilde, phi_tilde) + eps  # (N,T)

    return {
        "X": X, "C": C, "beta_smooth": beta_smooth, "phi_smooth": phi_smooth,
        "beta_tilde": beta_tilde, "phi_tilde": phi_tilde, "returns": returns,
        "sigma_beta2": cfg.SIM_SIGMA_BETA2, "sigma_phi2": cfg.SIM_SIGMA_PHI2,
        "sigma_r2": cfg.SIM_SIGMA_R2, "seed": seed,
    }


def panel_to_rows(panel: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flattens the simulated panel to training rows (paper Sec. 5.1: 4,000
    rows = 80 assets x 50 input months, each predicting its own next return).

    Returns:
        X_rows: (N*T, 2) characteristic vectors, repeated per input month.
        date_idx: (N*T,) integer index into the T training dates (0..T-1).
        r_rows: (N*T,) returns.
    """
    N, T = panel["X"].shape[0], panel["C"].shape[0]
    X_rows = np.tile(panel["X"], (T, 1))
    date_idx = np.repeat(np.arange(T), N)
    r_rows = panel["returns"].T.reshape(-1)  # (T,N) -> (T*N,), matches date_idx order
    assert X_rows.shape[0] == cfg.SIM_N_ASSETS * cfg.SIM_N_MONTHS == 4000
    return X_rows, date_idx, r_rows
