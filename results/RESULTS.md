# Results: PanelGP: A Probabilistic Factor Model for Asset-Return Panels

Status: experiments are running at the reduced, CPU-feasible configuration
recorded in `src/config.py` (the `*_RUN` variables). This file is updated as
each experiment's output files land in `results/`; numbers below reflect the
latest completed run of each experiment at the time of writing. Re-running
`python -m exp.exp1`, `python -m exp.exp2`, `python -m exp.exp3` (or with
`--full` for the paper-scale configuration) overwrites the files in
`results/`; the numbers below must then be refreshed from those files.

## Experiment 1 — Simulation: Recovery of Smooth Maps, Realized Latent
States, and Variance Parameters (Section 5)

Source file: `results/exp1_summary.json`, `results/exp1_replications.json`.

Partial result (1 of `SIM_N_REPLICATIONS` replications completed at the time
of writing; the full set of replications is still running in the background
and this table will be refreshed once `results/exp1_summary.json` reflects
all of them):

| Metric | Paper (Table 1) | Replication (so far) | Verdict |
|---|---|---|---|
| Factor function R² | 98.68% | 92.16% | pending more replications |
| Loading function R² | 99.24% | 99.37% | pending more replications |
| Predictive mean R² | 98.34% | 98.90% | pending more replications |
| Predictive variance correlation | 0.98 | 0.747 | pending more replications |
| sigma_beta^2 | 0.048 | 0.036 | pending more replications |
| sigma_phi^2 | 0.049 | 0.216 | pending more replications |
| sigma_r^2 | 0.052 | 0.100 | pending more replications |

Verdicts are withheld until the configured number of replications has
completed, since the paper reports mean and standard deviation over five
replications (Section 5.1).

## Experiment 2 — Fixed-Training Empirical Application (Section 6.1-6.5)

Running in the background at the time of writing; `results/exp2_table2.json`
and related output files had not yet been written. This section will be
completed once the run finishes.

## Experiment 3 — Rolling Re-estimation (Section 6.6-6.8)

Running in the background at the time of writing; `results/exp3_table3.json`
and related output files had not yet been written. This section will be
completed once the run finishes.

## Deviations from the paper

- Experiments 2 and 3 use OHLCV bars loaded through the QuantX Dataloader
  REST API in place of the WRDS-distributed Jensen et al. (2023)
  characteristics panel; see `src/characteristics.py` and the data-source
  notes added to this section once those experiments complete.
- All three experiments run at a reduced, CPU-feasible configuration
  (fewer optimization epochs, seeds, inducing points, and/or a narrower
  factor-count grid than the paper's) because no GPU is available; the
  paper reports approximately twelve minutes per five-factor fit on one
  NVIDIA H100 GPU (Section 6.1). The paper-scale configuration remains
  implemented and selectable with `--full` on each experiment script.
- Further deviations specific to each experiment are listed above once
  that experiment's run completes.

## Not reproduced

To be completed once all three experiments finish; any paper table, figure
or claim that remains infeasible under the available data and compute will
be listed here with its reason.
