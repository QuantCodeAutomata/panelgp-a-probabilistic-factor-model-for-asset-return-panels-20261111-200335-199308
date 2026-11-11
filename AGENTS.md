# AGENTS.md

Guide for coding agents and contributors working in this repository.

## What this repository implements

A replication of *PanelGP: A Probabilistic Factor Model for Asset-Return
Panels*: a bilinear variational Gaussian-process factor model for panels of
asset returns. Three experiments are implemented, see README.md for their
titles and the paper sections they reproduce.

## Layout

- `src/` — flat modules: kernels, the PanelGP model, training loop,
  evaluation metrics, IPCA benchmark, portfolio formation, significance
  tests, and `config.py` (every experiment parameter as an UPPER_CASE
  module-level variable with a source comment).
- `exp/` — one script per experiment (`exp1.py`, `exp2.py`, `exp3.py`),
  calling only the `src/` API.
- `data/` — `data_loader.py` (market data through the QuantX Dataloader and
  Massive) and `generate_synthetic.py` (the Experiment 1 simulation panel).
  Downloaded data is never committed.
- `tests/` — pytest tests for `src/` and for the experiments.
- `results/` — metrics, tables and plots written directly here, plus
  `RESULTS.md` with the comparison against the paper.

## Commands

```bash
pip install -r requirements.txt
python -m pytest -m "not market_data"   # unit tests, no network access
python -m pytest                        # full suite, including market-data tests
python -m exp.exp1
python -m exp.exp2            # executed configuration
python -m exp.exp2 --full     # full paper configuration
python -m exp.exp3
python -m exp.exp3 --full     # full paper configuration
```

## Conventions

- All experiment parameters live in `src/config.py` only, as module-level
  variables; no classes or dicts. Deviations and reduced-scale runs are
  documented only in `results/RESULTS.md`, never as prose in `config.py`.
- Market data is loaded only through `data/data_loader.py`; the Experiment 1
  synthetic panel is generated only through `data/generate_synthetic.py`. No
  other module performs data loading or generation.
- Experiments run as `python -m exp.<name>` from the repository root; tests
  run as `python -m pytest`. Neither modifies `sys.path`.
- Market-data tests are marked `@pytest.mark.market_data` and are skipped
  automatically when `DATALOADER_API_KEY` / `MASSIVE_API_KEY` are not set.
  Run them only after a change to the data path, and otherwise use
  `python -m pytest -m "not market_data"`.
- API keys are read from an uncommitted `.env` file (see `.env.example`);
  they are never hardcoded or logged.
- Re-running an experiment overwrites its files in `results/`. The numbers
  and verdicts in `results/RESULTS.md` are maintained by hand and must be
  checked against the newly written files after every rerun.
