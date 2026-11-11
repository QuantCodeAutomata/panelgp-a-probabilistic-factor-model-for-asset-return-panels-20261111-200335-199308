"""Experiment parameters for the PanelGP replication.

All numeric parameters used by src/ and exp/ live here as module-level
UPPER_CASE variables. Each has a one-line source note: a paper section or
equation, or "not specified in the paper" when the paper leaves the value
open. Values reduced to keep the replication inside its available compute
budget are given alongside their paper-scale counterparts
(``*_PAPER`` vs ``*_RUN``); the executed value is used unless an
experiment's ``--full`` flag selects the paper value.
"""

import numpy as np

# ---------------------------------------------------------------------------
# General numerics (paper Appendix C.3 "Numerical stabilization")
# ---------------------------------------------------------------------------
JITTER_INIT = 1e-7          # initial diagonal jitter added before Cholesky, paper App. C.3
JITTER_GROWTH = 10.0        # jitter multiplier on Cholesky failure, paper App. C.3
JITTER_MAX_TRIES = 8        # not specified in the paper; safety bound on the jitter loop

# ===========================================================================
# Experiment 1: simulation (paper Section 5.1 "Design")
# ===========================================================================
SIM_N_ASSETS = 80            # assets on a 9x9 grid minus one corner, paper Sec. 5.1
SIM_GRID_SIDE = 9             # grid side length, paper Sec. 5.1 ("regular 9x9 grid")
SIM_N_MONTHS = 50             # months, paper Sec. 5.1
SIM_N_FACTORS = 2             # H, paper Sec. 5.1 ("With H = 2")
SIM_SIGMA_BETA2 = 0.05        # loading-innovation variance sigma_beta^2, paper Sec. 5.1 / Eq. (28)
SIM_SIGMA_PHI2 = 0.05         # factor-innovation variance sigma_phi^2, paper Sec. 5.1 / Eq. (28)
SIM_SIGMA_R2 = 0.05           # observation-noise variance sigma_r^2, paper Sec. 5.1 / Eq. (28)
SIM_N_REPLICATIONS = 5        # independent simulated panels / seeds, paper Sec. 5.1 ("Repeat for 5 replications")

SIM_M_INDUCING = 64           # trainable inducing inputs per component, paper Sec. 5.1 / App. C.3
SIM_EPOCHS_PAPER = 5000       # training epochs, paper Sec. 5.1
SIM_EPOCHS_RUN = 5000         # executed epochs (full panel is small: 4,000 rows, CPU-feasible)
SIM_BATCH_SIZE = 2048         # minibatch rows, paper App. C.4 "Optimization"
SIM_N_MC_SAMPLES = 64         # factor Monte Carlo samples per likelihood evaluation (32 draws + antithetic negatives), paper App. C.4
SIM_LR_VARIATIONAL = 0.01     # Adam learning rate, variational parameters, paper App. C.4
SIM_LR_MODEL = 0.005          # Adam learning rate, model parameters (incl. inducing inputs), paper App. C.4
SIM_LR_DECAY_FRACTIONS = (0.3, 0.5, 0.6, 0.7, 0.8, 0.9)  # learning-rate halving schedule, paper App. C.4
SIM_GRAD_CLIP_NORM = 1.0      # gradient clipping to unit Euclidean norm, paper App. C.4
SIM_GRID_RESOLUTION = 41      # dense grid points per axis for function R^2; not specified in the paper, chosen for a fine but tractable grid

# Initial loading output scale, paper App. C.4: gamma_beta = 0.1, eta_h = 0.9 / (H*(d+0.1))
SIM_GAMMA_BETA = 0.1

# ===========================================================================
# Experiment 2 & 3: empirical application (paper Section 6)
# ===========================================================================
APP_TRAIN_START_PAPER = "1960-01-01"   # fixed-training estimation start, paper Sec. 6.1
APP_TRAIN_END_PAPER = "1989-12-31"     # fixed-training estimation end, paper Sec. 6.1
APP_VALID_START_PAPER = "1990-01-01"   # validation window start, paper Sec. 6.1
APP_VALID_END_PAPER = "1994-12-31"     # validation window end, paper Sec. 6.1
APP_EVAL_START_PAPER = "1995-01-01"    # evaluation window start, paper Sec. 6.1
APP_EVAL_END_PAPER = "2024-12-31"      # evaluation window end, paper Sec. 6.1

# Executed dates: the QuantX Dataloader's Massive-backed coverage for the
# candidate universe was found unreliable before approximately 2003 (many
# "no ticker record" errors for 1960s-1990s requests); see
# results/RESULTS.md "Deviations from the paper". The executed split keeps
# the paper's train > valid < eval ordering and relative proportions on the
# longest window with verified data coverage through the end of 2024.
APP_TRAIN_START_RUN = "2005-01-01"
APP_TRAIN_END_RUN = "2015-12-31"       # 132 months
APP_VALID_START_RUN = "2016-01-01"
APP_VALID_END_RUN = "2017-12-31"       # 24 months
APP_EVAL_START_RUN = "2018-01-01"
APP_EVAL_END_RUN = "2024-12-31"        # 84 months

APP_N_CHARACTERISTICS_PAPER = 137  # characteristics kept (observed >=70% of 1960-1989 stock-months), paper Sec. 6.1
APP_MIN_CHAR_COVERAGE = 0.70        # minimum per-characteristic coverage in the training period, paper Sec. 6.1
APP_MIN_ROW_COVERAGE = 0.70         # minimum fraction of characteristics observed to keep a stock-month, paper Sec. 6.1
APP_N_CONTEXT = 6                   # macro-financial context variables, paper Sec. 6.1

APP_H_GRID = list(range(1, 9))     # number of factors H = 1..8 evaluated in the validation grid, paper Sec. 6
APP_H_GRID_RUN = [1, 3, 5]          # executed H grid for the RBF/RBF factor-number sensitivity (Fig. 3); reduced for CPU runtime, see results/RESULTS.md
APP_H_SELECTED = 5                  # H used for Tables 2-4, paper Sec. 6.4 ("H=5 as compromise")
APP_KERNEL_PAIRS = [("affine", "constant"), ("affine", "rbf"),
                     ("rbf", "constant"), ("rbf", "rbf")]  # (loading, factor) kernels compared, paper Sec. 6.1
APP_N_SEEDS_PAPER = 5                # optimization seeds per kernel pair / H, paper Sec. 6.1
APP_N_SEEDS_RUN = 1                  # executed seeds (reduced universe/compute; see results/RESULTS.md)
APP_M_INDUCING = 160                 # trainable inducing inputs per component, paper Sec. 6.1 / App. C.3
APP_EPOCHS_PAPER = 500                # training epochs, paper Sec. 6.1
APP_EPOCHS_RUN = 500                  # executed epochs (kept at the paper value; universe/seeds are reduced instead)
APP_BATCH_SIZE = 2048                 # minibatch rows, paper App. C.4
APP_N_MC_SAMPLES = 64                 # factor Monte Carlo samples per likelihood evaluation, paper App. C.4
APP_LR_VARIATIONAL = 0.01             # Adam learning rate, variational parameters, paper App. C.4
APP_LR_MODEL = 0.005                  # Adam learning rate, model parameters, paper App. C.4
APP_LR_DECAY_FRACTIONS = (0.3, 0.5, 0.6, 0.7, 0.8, 0.9)  # learning-rate halving schedule, paper App. C.4
APP_GRAD_CLIP_NORM = 1.0              # gradient clipping, paper App. C.4

APP_UNIVERSE_SIZE_PAPER = 500         # approximate number of S&P 500 constituents in the historical panel, paper Sec. 6.1
APP_UNIVERSE_SIZE_RUN = 60            # executed universe size; see results/RESULTS.md for the time estimate and cut rule
APP_LIQUIDITY_LOOKBACK_MONTHS = 12    # window used to rank stocks by dollar volume for the universe cut; not specified in the paper

APP_NW_LAGS = 12                      # Newey-West lags for pairwise significance tests, paper Sec. 6.3 ("Newey-West (1987)... with 12 lags")
APP_HOLM_N_COMPARISONS = 15           # pairwise comparisons per criterion among 6 models (C(6,2)=15), paper Sec. 6.3
APP_SIGNIFICANCE_LEVEL = 0.05         # Holm-adjusted significance threshold, paper Sec. 6.3

APP_LONG_SHORT_DECILE = 0.10          # top/bottom decile for the long-short portfolio, paper Sec. 6.3
APP_LONG_SHORT_GROSS = 0.5            # gross exposure on each side of the long-short portfolio, paper Sec. 6.3
APP_ANNUALIZATION_MONTHS = 12          # months per year used to annualize Sharpe ratios and volatilities, paper Sec. 6.3

# ===========================================================================
# Experiment 3: rolling re-estimation (paper Section 6.6-6.8)
# ===========================================================================
ROLL_WINDOW_MONTHS_PAPER = 360          # rolling training window length, paper Sec. 6.6
ROLL_WINDOW_MONTHS_RUN = 120            # executed window length (120 months; data coverage and compute, see RESULTS.md)
ROLL_FIRST_FORECAST_YEAR_PAPER = 1995   # first forecast year, paper Sec. 6.6
ROLL_FIRST_FORECAST_YEAR_RUN = 2018     # executed first forecast year, matching APP_EVAL_START_RUN
ROLL_LAST_FORECAST_YEAR_PAPER = 2024    # last forecast year in the paper, paper Sec. 6.6
ROLL_LAST_FORECAST_YEAR_RUN = 2024      # executed last forecast year
ROLL_REESTIMATION_FREQ_YEARS = 1        # annual re-estimation, paper Sec. 6.6

ROLL_TRADING_COST_BPS = 10.0            # trading cost per dollar traded, paper Sec. 6.6 ("10 basis points")
ROLL_BREAKEVEN_SEARCH_BPS = (0.0, 200.0)  # bracket for the break-even-cost search; not specified in the paper

ROLL_CGF_ABS_TOL = 1e-9                 # quad_vec absolute tolerance, paper App. B.3
ROLL_CGF_REL_TOL = 1e-9                 # quad_vec relative tolerance, paper App. B.3
ROLL_N_QUINTILES = 5                    # temporal and cross-sectional risk quintiles, paper Sec. 6.8

RANDOM_SEED_BASE = 20240521             # not specified in the paper; fixed to make synthetic draws reproducible across runs

# Macro context series substituted for Goyal et al. (2024); see results/RESULTS.md "Deviations from the paper".
GOYAL_CONTEXT_COLUMNS = [
    "market_return", "term_spread", "default_yield_spread",
    "log_realized_market_variance", "average_correlation", "inflation",
]
