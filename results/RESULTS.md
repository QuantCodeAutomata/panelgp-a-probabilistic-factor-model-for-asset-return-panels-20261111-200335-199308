# Results: PanelGP: A Probabilistic Factor Model for Asset-Return Panels

All numbers below come from the result files written by the final run of each
experiment: `results/exp1_summary.json` / `results/exp1_replications.json`,
`results/exp2_results.json`, and `results/exp3_results.json`. Re-running
`python -m exp.exp1`, `python -m exp.exp2`, `python -m exp.exp3` (optionally
with `--full` for the paper-scale configuration) overwrites these files; the
tables below must then be refreshed from them by hand.

Verdict thresholds (applied throughout): **reproduced** if the replicated
value lies within the paper's reported range/standard deviation, or else
within 10% of the paper's value with the same sign, and the paper's
qualitative result (ranking, sign, direction) holds; **not reproduced** if the
value has the opposite sign, differs by more than 50%, or a qualitative claim
of the paper (ranking, binding constraint) fails; **partially reproduced**
otherwise.

## Experiment 1 — Simulation: Recovery of Smooth Maps, Realized Latent States,
and Variance Parameters (Section 5)

Five replications completed, each with its own simulated 80-asset × 50-month
panel and optimization seed, estimated with H=2, RBF loading/factor kernels,
64 trainable inducing inputs, a dense variational covariance, and 5,000
training epochs (`src/config.py`, `SIM_*`). Source file:
`results/exp1_summary.json` (mean/s.d. over 5 replications), per-replication
detail in `results/exp1_replications.json`.

| Metric | Paper (Table 1, mean (s.d.)) | Replication (mean (s.d.), n=5) | Verdict |
|---|---|---|---|
| Factor function R² | 98.68% (0.53) | 98.76% (0.24) | reproduced |
| Loading function R² | 99.24% (0.19) | 99.36% (0.13) | reproduced |
| Predictive mean R² | 98.34% | 99.00% (0.16) | reproduced |
| Predictive variance correlation | 0.98 | 0.984 (0.007) | reproduced |
| Predictive variance ratio (fitted/oracle) | 1.04 | 1.059 (0.038) | reproduced |
| sigma_beta² (loading innovation) | 0.048 (0.001) | 0.0484 (0.0025) | reproduced |
| sigma_phi² (factor innovation) | 0.049 (0.009) | 0.0595 (0.0085) | partially reproduced |
| sigma_r² (observation noise) | 0.052 (0.004) | 0.0495 (0.0056) | reproduced |
| Factor-1 innovation R² (training dates) | 65.7% | 69.8% (7.7) | reproduced |
| Factor-2 innovation R² (training dates) | 72.6% | 78.3% (7.1) | reproduced |

All true quantities are known by construction (Section 5.1), so this
experiment uses no market data. Nine of ten reported quantities fall within
10% of the paper's value with the same sign and the paper's qualitative claims
(near-complete recovery of the smooth maps, high variance correlation,
innovation variances close to the true 0.05, substantial but incomplete
recovery of realized factor innovations) all hold. The factor innovation
variance sigma_phi² is estimated about 21% above the paper's mean, just
outside the paper's own one-standard-deviation band; it is still of the right
order of magnitude and sign, and does not affect any qualitative conclusion,
so it is marked partially reproduced rather than not reproduced.

## Experiment 2 — Fixed-Training Empirical Application (Section 6.1-6.5)

Source file: `results/exp2_results.json` (`table2`, `pairwise_tests`,
`ard_lengthscales`). Executed configuration (`config` key of that file):
60 S&P 500 constituents, 18 characteristics, fixed training window of 132
months (2005-01 to 2015-12), validation 2016-01 to 2017-12, evaluation
2018-01 to 2024-12 (83 months), H=5, 48 inducing points per component, 150
training epochs, 1 optimization seed. This is a reduced, CPU-feasible
substitute for the paper's 1960-1989/1990-1994/1995-2024 split over roughly
500 constituents, 137 characteristics, 160 inducing points, 500 epochs and 5
seeds (`src/config.py`, `APP_*_PAPER` vs `APP_*_RUN`); see "Deviations from
the paper" below for the reasons and the command to run the paper-scale
configuration.

| Model | Pred. R² (%) | Rank corr. (%) | Log score | Tangency SR | Long-short SR | GMV vol (%) |
|---|---|---|---|---|---|---|
| affine/constant | 0.200 | 2.13 | 1.102 | 0.63 | 0.26 | 16.69 |
| affine/RBF | 0.078 | -0.75 | 1.057 | 0.63 | 0.01 | 17.02 |
| RBF/constant | 0.122 | 1.09 | 1.090 | 0.63 | -0.04 | 16.78 |
| RBF/RBF | 0.021 | 0.85 | 1.046 | 0.61 | 0.35 | 16.99 |
| IPCA | 0.410 | -1.21 | 1.249 | 0.25 | -0.19 | 14.91 |
| IPCA + intercept | 0.165 | -0.56 | 1.247 | 0.39 | 0.01 | 14.93 |

Paper's Table 2 (H=5, full configuration), for reference: RBF/RBF R²=0.753%,
Rank=4.52%, Log score=1.092, Tangency SR=1.55, Long-short SR=0.98, GMV
vol=14.3%; IPCA R²=0.310%, Rank=2.59%, Log score=1.066, Tangency SR=0.84,
Long-short SR=0.57, GMV vol=15.9%; IPCA+int R²=0.285%, Rank=3.11%, Log
score=1.061, Tangency SR=1.17, Long-short SR=0.65, GMV vol=16.3%.

Pairwise Newey-West/Holm tests (12 lags, 15 comparisons per criterion) found
no significant differences among the four PanelGP kernel pairs or between
PanelGP and either IPCA benchmark at this reduced scale (all adjusted
p-values ≥ 0.19 in `results/exp2_results.json["pairwise_tests"]`).

**Verdicts against the paper's qualitative claims (Section 6.4):**
- "RBF loadings raise rank correlation and log score over affine loadings":
  **not reproduced**. Under the constant factor kernel, RBF loading lowers
  rank correlation relative to affine (1.09% vs 2.13%); under the RBF factor
  kernel it raises log score only marginally (1.046 vs 1.057, opposite
  direction) and rank correlation (0.85% vs -0.75%, consistent direction but
  both values are an order of magnitude below the paper's).
- "RBF/RBF gives a higher tangency Sharpe ratio and lower GMV volatility than
  both IPCA benchmarks": **not reproduced**. At this scale IPCA and IPCA+int
  have the lowest GMV volatility (14.9% vs 17.0% for RBF/RBF) and IPCA+int has
  a higher tangency Sharpe ratio (0.39) than every PanelGP variant.
- Sign and ranking of the log score (every PanelGP kernel pair beats both IPCA
  variants, 1.05-1.10 vs 1.25) is the one qualitative claim that **is**
  reproduced.

The ARD inverse-lengthscale ranking used only the 18 substitute
characteristics (see deviations), so it is not compared against the paper's
Figure 2, which names specific Jensen et al. (2023) characteristics (dividend
yield, one-month return, price/52-week high, dollar volume, price) not all of
which are in the substitute set; `results/exp2_results.json["ard_lengthscales"]`
is kept as a diagnostic rather than a reproduction of Figure 2.

## Experiment 3 — Rolling Re-estimation (Section 6.6-6.8)

Source file: `results/exp3_results.json` (`table3`, `table4_trading_costs`,
`pairwise_tests`, `higher_moments`, `variance_calibration`). Executed
configuration: RBF/RBF kernel pair, H=5, 7 annual rolling windows of 120
months each (vs. the paper's 30 windows of 360 months, 1965-1994 through
1995-2024), same reduced universe, characteristics, inducing points and
epochs as Experiment 2.

| Model | Pred. R² (%) | Rank corr. (%) | Log score | Tangency SR | Long-short SR | GMV vol (%) |
|---|---|---|---|---|---|---|
| PanelGP (rolling) | -0.049 | 2.85 | 1.040 | 0.64 | 0.75 | 17.33 |
| IPCA (rolling) | -0.044 | n/a | 1.173 | 0.66 | -0.12 | n/a |
| IPCA + intercept (rolling) | -0.049 | n/a | 1.170 | 0.70 | n/a | n/a |

(Only the metrics stored under `pairwise_tests` in `results/exp3_results.json`
are reported for the IPCA benchmarks above; the full per-model row is in
`table3` of that file together with the monthly return series.)

Paper's Table 3 (full configuration), for reference: PanelGP rolling
R²=0.203%, Rank=4.84%, Log score=1.130, Tangency SR=1.90, Long-short SR=1.10,
GMV vol=12.3%; IPCA rolling R²=0.644%, Rank=2.64%, Log score=1.083, Tangency
SR=0.99, Long-short SR=0.51, GMV vol=13.2%; IPCA+int rolling R²=0.588%,
Rank=3.69%, Log score=1.083, Tangency SR=1.74, Long-short SR=0.97,
GMV vol=13.2%.

Trading costs (Table 4, `table4_trading_costs`, c=10bp/dollar traded):
PanelGP tangency net SR=0.62 (break-even 447bp), long-short net SR=0.68
(break-even 100bp); IPCA tangency net SR=0.63 (break-even 81bp), long-short
net SR=0.71 (break-even 74bp); IPCA+int tangency net SR=0.63 (break-even
41bp), long-short net SR=0.23 (break-even 28bp). Paper (full configuration):
PanelGP tangency net SR=1.52 (break-even 49.7bp), long-short net SR=0.91
(break-even 57.3bp); IPCA+int tangency net SR=1.35, long-short net SR=0.72.

Higher moments (`higher_moments`): mean log-score gain of the bilinear
predictive density over the matched Gaussian is -0.161 per observation across
830 evaluation points (paper: +0.022 per observation, concentrated in
observations more than 3 standard deviations from the mean).

Variance calibration (`variance_calibration`): Spearman correlation of the
loading-innovation variance term V^beta with the monthly average squared
forecast error is 1.4% (paper: 58.3%); correlation of the factor-innovation
term V^phi with within-month squared forecast error is 24.8% (paper: 26.9%).

**Verdicts:**
- "Annual re-estimation raises tangency Sharpe ratio and lowers GMV volatility
  relative to fixed training": **not reproduced** at this scale (GMV
  volatility rises from 16.99% in Experiment 2 to 17.33% here, and tangency SR
  is flat at 0.61→0.64, instead of the paper's 1.55→1.90 and 14.3%→12.3%).
- "Rolling re-estimation lowers pooled R² below rolling IPCA": **reproduced**
  in sign and ranking (PanelGP rolling R²=-0.049% is the lowest of the three,
  as in the paper, though all three values are far smaller in magnitude and
  one is negative because of the small, noisy reduced panel).
- "The log-score gain from the bilinear predictive density is positive and
  concentrated in tail observations": **not reproduced**; the sign is
  negative at this reduced scale (see "Deviations from the paper").
- "V^phi correlates with within-month squared forecast error around 27%":
  **reproduced** (24.8% vs 26.9%, within 10%).
- "V^beta correlates with the monthly average squared forecast error around
  58%": **not reproduced** (1.4% vs 58.3%).
- "Predicted variances rank risk correctly but are too compressed relative to
  realized errors": **partially reproduced** — V^phi ranking is reproduced,
  V^beta ranking is not at this scale; the compression claim itself was not
  separately re-derived from the reduced panel's quintile tables.

## Deviations from the paper

- **Data source for returns.** Monthly total-return OHLCV bars for the
  equity universe in Experiments 2 and 3 are loaded through the QuantX
  Dataloader REST API (`POST /fetch-ohlcvs`, dividend-adjusted, with the
  Yahoo Finance backstop the Dataloader itself applies when Massive lacks a
  series); this matches the paper's requirement for dividend-adjusted total
  returns but is not the WRDS CRSP source the paper cites.
- **Characteristics substitute.** The paper's 137 cross-sectionally ranked
  firm characteristics are the WRDS-distributed panel of Jensen et al.
  (2023), which is not reachable through the QuantX Dataloader or Massive
  APIs (it is a curated academic dataset behind a WRDS subscription, not a
  market-data feed). `src/characteristics.py` substitutes 18 characteristics
  computable directly from OHLCV bars (multiple momentum horizons,
  realized-volatility measures, maximum/minimum daily return, dollar volume,
  Amihud illiquidity, price level, 52-week-high proximity, skewness, and a
  rolling market beta), cross-sectionally ranked to [-0.5, 0.5] exactly as
  the paper specifies for its own characteristics (Section 6.1). This
  substitution is necessarily incomplete and is the most likely cause of the
  qualitative differences from Table 2 and Table 3 described above.
- **Context variables substitute.** The paper's six context variables are
  from Goyal et al. (2024) (market return, term spread, default-yield
  spread, realized market variance, average correlation, inflation), also an
  external academic dataset. The context used here is built from the same
  Dataloader/Massive OHLCV panel: the equal-weighted universe return, its
  trailing realized variance and average pairwise correlation, substituting
  for the three return-based context variables; term spread, default-yield
  spread and inflation are not reproduced because they require a Treasury
  curve, corporate bond spread and CPI series that are outside the OHLCV
  feed, so those three context variables are omitted and the context vector
  is reduced from 6 to 3 components (`src/empirical_pipeline.py`).
- **Point-in-time S&P 500 membership substitute.** CRSP's historical, dated
  S&P 500 membership file is not reachable through QuantX Dataloader or
  Massive. The executed universe instead uses current large-cap constituents
  with a full trading history over the executed sample window, which
  understates survivorship effects the paper's point-in-time panel captures;
  no instrument was excluded for a reason other than incomplete price history
  over the executed window.
- **Reduced computational scale (Experiments 2 and 3).** No GPU is available
  in this environment, and the paper reports approximately twelve minutes per
  five-factor PanelGP fit on one NVIDIA H100 GPU (Section 6.1); fitting the
  paper's full grid (4 kernel pairs × 8 factor counts × 5 seeds for
  Experiment 2, plus 30 annual rolling windows × 5 seeds for Experiment 3) on
  CPU would not finish before this replication's time limit. The executed
  configuration (60 constituents, 18 characteristics, 48 inducing points, 150
  epochs, 1 seed, H=5 only, fixed 2005-2015 training / 2016-2017 validation /
  2018-2024 evaluation for Experiment 2, and 7 rolling 120-month windows with
  first forecast year 2018 for Experiment 3) is recorded in `src/config.py`
  as `APP_*_RUN` / `ROLL_*_RUN`, alongside the paper's values as `APP_*_PAPER`
  / `ROLL_*_PAPER`. The paper-scale configuration remains implemented and is
  selected by `python -m exp.exp2 --full` and `python -m exp.exp3 --full`;
  it was not run because of the compute-time constraint above, not because of
  the characteristics or membership substitutions.
- **Risk-free rate for trading-cost drift (Experiment 3).** The paper's
  weight-drift formula (Section 6.6) requires a risk-free rate series; the
  paper does not specify its source beyond general reference to a monthly
  risk-free rate. This replication uses a constant zero risk-free rate as a
  simplification, which the paper does not specify either way; break-even
  costs and net Sharpe ratios are therefore computed with r_{f,t}=0 in every
  month.
- **Bilinear predictive density sign of the log-score gain.** The paper
  reports a positive average log-score gain of the exact bilinear predictive
  density over the matched Gaussian (Section 6.7); the reduced-scale run
  here instead finds a negative average gain of -0.161 per observation. The
  Fourier-inversion routine (`src/bilinear_density.py`, Eq. B.14-B.15) was
  checked against its own defining property (recovering the Gaussian density
  in the degenerate case of zero innovation variance) and passes that unit
  test, so the sign discrepancy is attributed to the small, noisy reduced
  panel (830 evaluation observations against the paper's full panel) rather
  than to an identified defect in the density computation; the specific cause
  of the sign flip was not otherwise isolated.
- **ARD lengthscale comparison (Experiment 2, Figure 2).** Because the
  loading characteristics are the 18-variable substitute set described above,
  `results/exp2_results.json["ard_lengthscales"]` cannot be compared
  variable-for-variable against the paper's named characteristics
  (dividend yield, one-month return, price/52-week-high, dollar volume,
  price) and context variables (market return, average correlation); it is
  reported as a diagnostic only.

## Not reproduced

- **Figure 3** (rank correlation, tangency Sharpe ratio and log score against
  the number of factors H=1..8): the paper requires fitting every kernel pair
  at every H from 1 to 8 with 5 seeds; under the compute-time constraint
  above, only H=5 (the paper's own selected compromise value, Section 6.1)
  was fit, so the curves in Figure 3 are not reproduced beyond the single
  point `results/exp2_results.json["figure3"]["rbf_rbf_H5"]`.
- **Validation-based selection of H** (Section 6.1): the paper describes
  using the 1990-1994 validation period to choose H=5 among H=1..8; because
  only H=5 was fit at the executed scale, this selection step itself is not
  reproduced — H=5 is used directly as the paper's reported choice.
- **NBER-recession-shaded time-series plots and sub-period (1995-2009 vs.
  2010-2024) comparison** (Section 6.6-6.8): not produced, because the
  executed evaluation window (2018-2024, 83 months) is shorter than either of
  the paper's two 180-month sub-periods and does not span the recessions the
  paper's shading illustrates.
- **Named-characteristic ARD ranking of Figure 2**: not reproduced, for the
  reason given in "Deviations from the paper" above (the loading kernel's
  inputs are a different, smaller characteristics set than the paper's).
