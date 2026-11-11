# PanelGP: A Probabilistic Factor Model for Asset-Return Panels

Replication of selected experiments from *PanelGP: A Probabilistic Factor
Model for Asset-Return Panels*.

This repository reproduces three experiments from the paper:

1. **Experiment 1** (Section 5): a simulation study verifying that the
   PanelGP estimator recovers the smooth loading and factor maps, the
   realized latent states, and the variance parameters from data generated
   by the model itself (Table 1).
2. **Experiment 2** (Section 6.1-6.5): a fixed-training empirical
   application to monthly returns of historical S&P 500 constituents,
   comparing four PanelGP kernel pairs against IPCA benchmarks across
   H = 1,...,8 factors (Table 2, Figure 2, Figure 3).
3. **Experiment 3** (Section 6.6-6.8): annual rolling re-estimation of the
   RBF/RBF PanelGP model (H=5), evaluating portfolio performance net of
   trading costs, the log-score contribution of the bilinear predictive
   distribution's higher moments, and the calibration of the predicted
   variance decomposition (Tables 3-4).

## Repository layout

```
src/     model, kernels, training, evaluation, IPCA, portfolio and
         significance-testing code shared by every experiment
exp/     one script per experiment, calling only the src/ API
data/    data_loader.py (market data) and generate_synthetic.py
         (simulation panels); no downloaded data is stored here
tests/   pytest unit and integration tests
results/ metrics, tables and plots written by the experiment scripts
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY
```

Experiment 1 uses only simulated data and needs no API keys. Experiments 2
and 3 load monthly stock prices through the QuantX Dataloader and reference
data through Massive; see `results/RESULTS.md` for the characteristics and
universe substitutions used because the paper's WRDS-distributed data
(Jensen et al., 2023 characteristics; CRSP point-in-time S&P 500
membership; Goyal et al., 2024 macro-financial context) is not available
through these services.

## Running

```bash
python -m pytest -m "not market_data"      # unit tests (no network access)
python -m pytest                           # full suite, including market-data tests

python -m exp.exp1                         # Experiment 1 (paper-scale: 5 replications, 5000 epochs)
python -m exp.exp2                         # Experiment 2 (executed configuration, see results/RESULTS.md)
python -m exp.exp2 --full                  # Experiment 2, full paper configuration (universe, seeds, H grid)
python -m exp.exp3                         # Experiment 3 (executed configuration)
python -m exp.exp3 --full                  # Experiment 3, full paper configuration

bash reproduce.sh                          # tests, then every experiment at its executed configuration
```

Every run overwrites the corresponding files in `results/`. The numbers and
verdicts in `results/RESULTS.md` are produced by hand from the files saved
by the most recent run; see that file for the full comparison against the
paper, every deviation, and every result not reproduced.
