#!/usr/bin/env bash
set -euo pipefail

# Approval matrix for AAAI-facing ACDAN experiments.
# Run from the repo root after installing .venv and generating candidate files.

PY=${PY:-.venv/bin/python}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
TAG=${TAG:-qwen7b}
SEEDS=${SEEDS:-"0 1 2"}
N_VALUES=${N_VALUES:-"1 2 4 8 16"}

mkdir -p results/approval

run_math() {
  local dataset=$1
  local data_path=$2
  for seed in $SEEDS; do
    for method in cot sc asc bon acdan; do
      for n in $N_VALUES; do
        if [[ "$method" == "cot" && "$n" != "1" ]]; then
          continue
        fi
        $PY -m acdan.run_experiment \
          --method "$method" --dataset "$dataset" --data-path "$data_path" \
          --policy vllm --policy-model "$MODEL" --prm llm \
          --math-evidence none --n "$n" --seed "$seed" --save-per-task \
          --out "results/approval/${dataset}_${TAG}_${method}_n${n}_s${seed}.json"
      done
    done
  done

  # Evidence attribution for ACDAN at N=8.
  for evidence in none prompt prm dto all; do
    $PY -m acdan.run_experiment \
      --method acdan --dataset "$dataset" --data-path "$data_path" \
      --policy vllm --policy-model "$MODEL" --prm llm \
      --math-evidence "$evidence" --math-count-weight 0.35 \
      --n 8 --seed 0 --save-per-task \
      --out "results/approval/${dataset}_${TAG}_acdan_evidence_${evidence}.json"
  done
}

run_bfcl() {
  local test_path=${BFCL_TEST:-data/bfcl_full.jsonl}
  local train_path=${BFCL_TRAIN:-data/bfcl_dev.jsonl}
  for seed in $SEEDS; do
    for method in cot sc asc bon acdan; do
      for n in $N_VALUES; do
        if [[ "$method" == "cot" && "$n" != "1" ]]; then
          continue
        fi
        $PY -m acdan.run_experiment \
          --method "$method" --dataset bfcl --data-path "$test_path" \
          --policy vllm --policy-model "$MODEL" --prm llm \
          --n "$n" --seed "$seed" --save-per-task \
          --fit-inertia --inertia-fit-path "$train_path" \
          --out "results/approval/bfcl_${TAG}_${method}_n${n}_s${seed}.json"
      done
    done
  done

  for abl in no_dto no_graph no_inertia no_verification no_latent no_ttt; do
    $PY -m acdan.run_experiment \
      --method acdan --disable "$abl" --dataset bfcl --data-path "$test_path" \
      --policy vllm --policy-model "$MODEL" --prm llm \
      --n 8 --seed 0 --save-per-task \
      --fit-inertia --inertia-fit-path "$train_path" \
      --out "results/approval/bfcl_${TAG}_abl_${abl}.json"
  done
}

run_math gsm8k "data/gsm8k_${TAG}_k8_sample.jsonl"
run_math math "data/math500_${TAG}_k8_sample.jsonl"
run_bfcl

$PY scripts/collect_pareto.py --glob "results/approval/*.json" \
  --csv results/approval/pareto.csv \
  --svg results/approval/pareto.svg
