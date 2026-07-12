#!/usr/bin/env bash
set -euo pipefail

# Blind ACDAN/baseline selection followed by a separate immutable-score join.
PY=${PY:-.venv/bin/python}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
TAG=${TAG:-qwen7b}
K=${K:-8}
METHODS=${METHODS:-"acdan bon asc sc cot"}
DATASETS=${DATASETS:-"browsecomp mathhay swe_bench_verified terminal_bench tau2_bench mcp_bench"}
TASK_DIR=${TASK_DIR:-data/agentbench}
ARTIFACT_DIR=${ARTIFACT_DIR:-runs/agentbench/artifacts}
OUT_DIR=${OUT_DIR:-results/agentbench/reportable}
CALIBRATOR_DIR=${CALIBRATOR_DIR:-results/agentbench/calibrators}
ENCODER_ARGS=${ENCODER_ARGS:-"--encoder hash"}
LATENT_ARGS=${LATENT_ARGS:-"--no-latent"}
MONITOR_ARGS=${MONITOR_ARGS:-"--monitor --progress-every 25"}
SELECTOR_ARGS=${SELECTOR_ARGS:-"--task-preview-chars 4096 --candidate-preview-chars 2048"}

read -r -a ENCODER_ARGV <<< "$ENCODER_ARGS"
read -r -a LATENT_ARGV <<< "$LATENT_ARGS"
read -r -a MONITOR_ARGV <<< "$MONITOR_ARGS"
read -r -a SELECTOR_ARGV <<< "$SELECTOR_ARGS"

mkdir -p "$OUT_DIR"

for dataset in $DATASETS; do
  task_path="${TASK_DIR}/${dataset}_tasks.jsonl"
  cand_path="${ARTIFACT_DIR}/${dataset}_${TAG}_k${K}_candidates.jsonl"
  score_path="${ARTIFACT_DIR}/${dataset}_${TAG}_k${K}_scores.jsonl"

  "$PY" experiments/validate_agentbench.py \
    --tasks "$task_path" \
    --candidates "$cand_path" \
    --scores "$score_path" \
    --min-candidates "$K" \
    --require-blind --require-provenance

  for method in $METHODS; do
    selection_path="${OUT_DIR}/${dataset}_${TAG}_${method}_selection.json"
    result_path="${OUT_DIR}/${dataset}_${TAG}_${method}.json"

    "$PY" experiments/run_agentbench_selection.py \
      --method "$method" \
      --candidates-path "$cand_path" \
      --selection-only \
      --policy vllm --policy-model "$MODEL" --prm llm \
      "${ENCODER_ARGV[@]}" \
      "${LATENT_ARGV[@]}" \
      "${MONITOR_ARGV[@]}" \
      "${SELECTOR_ARGV[@]}" \
      --n "$K" --seed 0 --save-per-task \
      --out "$selection_path"

    eval_args=(
      --selection "$selection_path"
      --candidates "$cand_path"
      --scores "$score_path"
      --out "$result_path"
    )
    calibrator="${CALIBRATOR_DIR}/${dataset}_${TAG}_${method}.json"
    if [[ -s "$calibrator" ]]; then
      eval_args+=(--calibrator "$calibrator")
    fi
    "$PY" experiments/evaluate_agentbench_selection.py "${eval_args[@]}"
  done
done
