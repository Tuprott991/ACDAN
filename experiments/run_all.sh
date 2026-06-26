#!/usr/bin/env bash
# Validate the full experiment matrix OFFLINE (mock backends, no GPU), then show
# the real commands. Run from repo root:  bash experiments/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results

echo "==> Offline dry-run (mock policy + PRM, synthetic data) — proves every script runs"
for METHOD in acdan cot sc bon; do
  python -m acdan.run_experiment --method "$METHOD" --dataset synthetic --limit 24 \
    --out "results/dryrun_${METHOD}.json"
done
for ABL in no_dto no_graph no_inertia no_verification; do
  python -m acdan.run_experiment --method acdan --disable "$ABL" --dataset synthetic \
    --limit 24 --out "results/dryrun_abl_${ABL}.json"
done

echo
echo "==> Dry-run accuracies:"
python - <<'PY'
import glob, json
for f in sorted(glob.glob("results/dryrun_*.json")):
    s = json.load(open(f))["summary"]
    print(f"{f:45s} acc={s['accuracy']:.3f} ece={s['ece']:.3f} wall={s['wall_s']:.2f}s")
PY

echo
echo "Offline validation complete. For REAL runs on the GPU VM, see experiments/PLAN.md"
echo "  e.g.: python -m acdan.run_experiment --method acdan --dataset bfcl \\"
echo "          --data-path data/bfcl_test.jsonl --policy vllm \\"
echo "          --policy-model Qwen/Qwen2.5-7B-Instruct --prm llm --fit-inertia \\"
echo "          --out results/bfcl_qwen7b_acdan.json"
