#!/usr/bin/env bash
# Reproduce the PanelGP replication: tests, then every experiment.
# See README.md for details and for the full-scale (--full) commands.
set -uo pipefail
cd "$(dirname "$0")"

echo "== Unit tests (excluding market-data tests) =="
python -m pytest -m "not market_data" -q

echo "== Experiment 1: simulation recovery study =="
python -m exp.exp1

echo "== Experiment 2: fixed-training empirical application =="
python -m exp.exp2

echo "== Experiment 3: rolling re-estimation =="
python -m exp.exp3
