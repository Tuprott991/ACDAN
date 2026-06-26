#!/usr/bin/env bash
# Reproduce the headline offline results into results/.
# Usage: bash scripts/reproduce.sh
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results

echo ">> Full ACDAN evaluation (seed 0)"
python -m acdan.cli eval --config configs/default.yaml --seed 0 --out results/full_seed0.json

echo ">> Baseline CoT evaluation (seed 0)"
python -m acdan.cli eval --config configs/ablations/baseline_cot.yaml --seed 0 --out results/baseline_seed0.json

echo ">> Ablation grid (seed 0)"
python -m acdan.cli ablation --config configs/default.yaml --seed 0 --out results/ablation_seed0.json

echo ">> Demo trace (one task)"
python -m acdan.cli demo --config configs/default.yaml --seed 0 --task-index 0

echo
echo "Done. JSON artefacts written to results/."
