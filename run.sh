#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

M=${M:-Qwen/Qwen2.5-7B-Instruct}
TAG=${TAG:-qwen7b}

mkdir -p results logs

# 1. Generate candidates for Omni-MATH k8.
if [ ! -f "data/omni_math_${TAG}_k8.jsonl" ]; then
  .venv/bin/python experiments/gen_candidates.py \
    --model "$M" \
    --in data/omni_math.jsonl \
    --out "data/omni_math_${TAG}_k8.jsonl" \
    --k 8 \
    --order sample \
    --temperature 0.8 \
    --max-tokens 1024 \
    2>&1 | tee "logs/omni_math_${TAG}_k8_gen.log"
else
  echo "Candidate file exists: data/omni_math_${TAG}_k8.jsonl"
fi

# 2. Run the full matrix for the remaining math datasets.
for DATASET in math500 aime2025 omni_math; do
  if [ "$DATASET" = "aime2025" ]; then
    D="data/aime2025_${TAG}_k16.jsonl"
  else
    D="data/${DATASET}_${TAG}_k8.jsonl"
  fi

  if [ ! -f "$D" ]; then
    echo "Skip $DATASET because missing $D"
    continue
  fi

  echo "=============================="
  echo "Running dataset: $DATASET"
  echo "Data path: $D"
  echo "=============================="

  for METHOD in acdan bon asc sc cot; do
    .venv/bin/python -m acdan.run_experiment \
      --method "$METHOD" \
      --dataset "$DATASET" \
      --data-path "$D" \
      --policy vllm \
      --policy-model "$M" \
      --prm llm \
      --math-evidence none \
      --n 8 \
      --seed 0 \
      --out "results/${DATASET}_${TAG}_${METHOD}.json"
  done

  for EVID in none prompt prm dto all; do
    .venv/bin/python -m acdan.run_experiment \
      --method acdan \
      --dataset "$DATASET" \
      --data-path "$D" \
      --policy vllm \
      --policy-model "$M" \
      --prm llm \
      --math-evidence "$EVID" \
      --math-count-weight 0.35 \
      --out "results/${DATASET}_${TAG}_evidence_${EVID}.json"
  done

  for ABL in no_dto no_verification no_latent no_ttt; do
    .venv/bin/python -m acdan.run_experiment \
      --method acdan \
      --disable "$ABL" \
      --dataset "$DATASET" \
      --data-path "$D" \
      --policy vllm \
      --policy-model "$M" \
      --prm llm \
      --math-evidence none \
      --out "results/${DATASET}_${TAG}_abl_${ABL}.json"
  done
done
