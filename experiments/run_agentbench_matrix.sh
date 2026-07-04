#!/usr/bin/env bash
set -euo pipefail

# Run ACDAN and selector baselines on pre-scored AgentBench candidate files.
# Candidate files must be produced by build_agentbench_candidates.py and contain
# score/is_correct, or pass evaluator commands through EVALUATOR_ARGS.

PY=${PY:-.venv/bin/python}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
TAG=${TAG:-qwen7b}
K=${K:-8}
METHODS=${METHODS:-"acdan bon asc sc cot"}
DATASETS=${DATASETS:-"browsecomp webvoyager swe_bench_verified tau2_bench"}
TASK_DIR=${TASK_DIR:-data/agentbench}
CAND_DIR=${CAND_DIR:-results/agentbench}
OUT_DIR=${OUT_DIR:-results/agentbench}
ENCODER_ARGS=${ENCODER_ARGS:-"--encoder hash"}
LATENT_ARGS=${LATENT_ARGS:-"--no-latent"}
MONITOR_ARGS=${MONITOR_ARGS:-"--monitor --progress-every 25"}
EVALUATOR_ARGS=${EVALUATOR_ARGS:-""}
SELECTOR_ARGS=${SELECTOR_ARGS:-"--task-preview-chars 4096 --candidate-preview-chars 2048"}
VALIDATOR_ARGS=${VALIDATOR_ARGS:-""}
ALLOW_UNEVALUATED=${ALLOW_UNEVALUATED:-0}

read -r -a ENCODER_ARGV <<< "$ENCODER_ARGS"
read -r -a LATENT_ARGV <<< "$LATENT_ARGS"
read -r -a MONITOR_ARGV <<< "$MONITOR_ARGS"
read -r -a EVALUATOR_ARGV <<< "$EVALUATOR_ARGS"
read -r -a SELECTOR_ARGV <<< "$SELECTOR_ARGS"
read -r -a VALIDATOR_ARGV <<< "$VALIDATOR_ARGS"
if [[ -n "$EVALUATOR_ARGS" ]]; then
  VALIDATOR_ARGV+=("--allow-external-unscored")
fi
if compgen -A variable ACDAN_EXTERNAL_ >/dev/null; then
  VALIDATOR_ARGV+=("--allow-external-unscored")
fi
if [[ "$ALLOW_UNEVALUATED" == "1" || "$ALLOW_UNEVALUATED" == "true" ]]; then
  VALIDATOR_ARGV+=("--allow-external-unscored")
  SELECTOR_ARGV+=("--allow-unevaluated")
fi

mkdir -p "$OUT_DIR"

for dataset in $DATASETS; do
  task_path="${TASK_DIR}/${dataset}_tasks.jsonl"
  cand_path="${CAND_DIR}/${dataset}_${TAG}_k${K}_candidates.jsonl"

  if [[ ! -s "$task_path" ]]; then
    echo "missing task manifest: $task_path" >&2
    exit 1
  fi
  if [[ ! -s "$cand_path" ]]; then
    echo "missing or empty candidate file: $cand_path" >&2
    echo "Build it with experiments/build_agentbench_candidates.py first." >&2
    exit 1
  fi

  "$PY" experiments/validate_agentbench.py \
    --tasks "$task_path" \
    --candidates "$cand_path" \
    --min-candidates "$K" \
    "${VALIDATOR_ARGV[@]}"

  for method in $METHODS; do
    "$PY" experiments/run_agentbench_selection.py \
      --method "$method" \
      --candidates-path "$cand_path" \
      --policy vllm --policy-model "$MODEL" --prm llm \
      "${ENCODER_ARGV[@]}" \
      "${LATENT_ARGV[@]}" \
      "${MONITOR_ARGV[@]}" \
      "${EVALUATOR_ARGV[@]}" \
      "${SELECTOR_ARGV[@]}" \
      --n "$K" --seed 0 --save-per-task \
      --out "${OUT_DIR}/${dataset}_${TAG}_${method}.json"
  done
done
