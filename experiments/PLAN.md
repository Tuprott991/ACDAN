# ACDAN Experiment Plan (VM)

End-to-end plan for running ACDAN after the evaluator cleanup. The framing is
still **tool/operator-selection DTO**: the policy scores candidate actions into
an `(H, V)` prior, DTO refines that prior, and a PRM supplies reward gradients.
Math runs are `H=1, V=K` answer selection over pre-generated candidates; graph
and inertia should be evaluated on multi-step tool/agent tasks, not sold as math
accuracy modules.

---

## 0. VM Setup

Recommended VM: 1x A100 80GB or H100 for 7B/8B policy + LLM-as-PRM.

```bash
git clone <your-repo> && cd ACDAN
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install -r requirements-data.txt
pip install -r requirements-gpu.txt

# Put HF_TOKEN in .env if needed. scripts/setup_datasets.py calls load_env().
# Put ANTHROPIC_API_KEY in the shell only for --verifier claude / GAIA judging.

.venv/bin/python -B -m pytest -q
.venv/bin/python -m acdan.run_experiment --method acdan --dataset synthetic --limit 24
```

Measure real throughput before full runs:

```bash
.venv/bin/python -m acdan.run_experiment --method acdan --dataset bfcl \
  --data-path data/bfcl_full.jsonl --limit 50 \
  --policy vllm --policy-model Qwen/Qwen2.5-7B-Instruct --prm llm \
  --out results/_smoke_bfcl.json --save-per-task
```

Use `summary.mean_latency_s` from this smoke run to rescale estimates:
`benchmark_eval_seconds ~= model_load + N_tasks * mean_latency_s`.

---

## 1. Dataset Setup

The setup script prepares stable local files and raw snapshots:

```bash
.venv/bin/python scripts/setup_datasets.py --suite all --overwrite
```

Outputs currently prepared:

| File | Rows | Use |
|---|---:|---|
| `data/gsm8k_train.jsonl` | 7473 | optional candidate-generation dev/train source |
| `data/gsm8k_test.jsonl` | 1319 | GSM8K candidate generation |
| `data/math500.jsonl` | 500 | MATH-500 candidate generation |
| `data/aime2025.jsonl` | 30 | harder math candidate generation |
| `data/omni_math.jsonl` | 4428 | broad hard math candidate generation |
| `data/bfcl_full.jsonl` | 2600 | BFCL tool-name selection with gold call metadata |
| `data/raw/bfcl/` | snapshot | official BFCL source files |
| `data/raw/tau2_bench_data/` | snapshot | stateful tau2 source material |
| `data/raw/tau2_bench_hud/` | snapshot | HUD-format tau2 tasks |

BFCL final paper numbers should still prefer official AST/executable evaluation.
The current ACDAN adapter preserves `gold_calls`, but the runner still selects
tool names, not full argument ASTs.

---

## 2. Candidate Generation

Generate math candidates with neutral ordering by default. Do not use plurality
order except as a named self-consistency control.

```bash
M=Qwen/Qwen2.5-7B-Instruct
TAG=qwen7b

.venv/bin/python experiments/gen_candidates.py --model $M \
  --in data/gsm8k_test.jsonl --out data/gsm8k_${TAG}_k8.jsonl \
  --k 8 --order sample

.venv/bin/python experiments/gen_candidates.py --model $M \
  --in data/math500.jsonl --out data/math500_${TAG}_k8.jsonl \
  --k 8 --order sample

.venv/bin/python experiments/gen_candidates.py --model $M \
  --in data/aime2025.jsonl --out data/aime2025_${TAG}_k16.jsonl \
  --k 16 --order sample
```

Useful controls:

```bash
# Neutral randomized order control.
--order shuffle --shuffle-seed 0

# Explicit self-consistency/plurality baseline artifact.
--order plurality
```

Every math result summary now reports `oracle_candidate_accuracy`,
`always_first_accuracy`, `always_last_accuracy`, and `highest_count_accuracy`.

---

## 3. Main Matrix

Policies: `Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`,
`Qwen/Qwen2.5-3B-Instruct`.

Methods: `acdan`, `bon`, `asc`, `sc`, `cot`.  `asc` is an early-stopping
self-consistency baseline: it stops once the plurality answer reaches a
confidence threshold, giving ACDAN a direct adaptive-compute competitor.

### 3a. Math

Default rigorous math run: no candidate counts in prompts, no PRM count bonus, no
DTO count prior.

```bash
M=Qwen/Qwen2.5-7B-Instruct; TAG=qwen7b; D=data/gsm8k_${TAG}_k8.jsonl
for METHOD in acdan bon asc sc cot; do
  .venv/bin/python -m acdan.run_experiment --method $METHOD --dataset gsm8k \
    --data-path $D --policy vllm --policy-model $M --prm llm \
    --math-evidence none --n 8 --seed 0 \
    --out results/gsm8k_${TAG}_${METHOD}.json
done
```

Self-consistency evidence ablation, ACDAN only:

```bash
for EVID in none prompt prm dto all; do
  .venv/bin/python -m acdan.run_experiment --method acdan --dataset gsm8k \
    --data-path $D --policy vllm --policy-model $M --prm llm \
    --math-evidence $EVID --math-count-weight 0.35 \
    --out results/gsm8k_${TAG}_evidence_${EVID}.json
done
```

Full approval matrix with per-task artifacts and Pareto collection:

```bash
MODEL=Qwen/Qwen2.5-7B-Instruct TAG=qwen7b \
  experiments/run_approval_matrix.sh
```

This writes `results/approval/*.json`, `results/approval/pareto.csv`, and
`results/approval/pareto.svg`.

Module ablations on math should focus on DTO/verification/calibration. Graph and
inertia are structurally weak on `H=1` tasks.

```bash
for ABL in no_dto no_verification no_latent no_ttt; do
  .venv/bin/python -m acdan.run_experiment --method acdan --disable $ABL \
    --dataset gsm8k --data-path $D --policy vllm --policy-model $M --prm llm \
    --math-evidence none --out results/gsm8k_${TAG}_abl_${ABL}.json
done
```

Repeat the same block for `math500`, `aime2025`, and later `omni_math` once
candidate files exist.

### 3b. BFCL / Tool Use

Use BFCL first for graph/inertia cost behavior. Fit inertia from a separate file
or omit it; the runner no longer fits from evaluation gold.

```bash
M=Qwen/Qwen2.5-7B-Instruct; TAG=qwen7b
TRAIN=data/bfcl_dev.jsonl
TEST=data/bfcl_full.jsonl

for METHOD in acdan bon asc sc cot; do
  .venv/bin/python -m acdan.run_experiment --method $METHOD --dataset bfcl \
    --data-path $TEST --policy vllm --policy-model $M --prm llm \
    --n 8 --seed 0 \
    --fit-inertia --inertia-fit-path $TRAIN \
    --out results/bfcl_${TAG}_${METHOD}.json
done
```

Calibration study:

```bash
.venv/bin/python -m acdan.run_experiment --method acdan --dataset bfcl \
  --data-path $TEST --policy vllm --policy-model $M --prm llm \
  --verifier claude --judge-model claude-opus-4-8 \
  --fit-inertia --inertia-fit-path $TRAIN \
  --out results/bfcl_${TAG}_acdan_claudeverify.json
```

### 3c. Stateful Agentic Raw Data

`data/raw/tau2_bench_data/` and `data/raw/tau2_bench_hud/` are available for the
next adapter/executor milestone. Do not report tau2/GAIA as final ACDAN numbers
until the runner produces final answers through an execution layer.

---

## 4. Recommended Order

1. Offline tests and synthetic run.
2. `scripts/setup_datasets.py --suite all`.
3. BFCL smoke run to measure throughput.
4. BFCL full matrix, including graph/inertia ablations and calibration.
5. GSM8K/MATH-500 with neutral candidate order and `--math-evidence none`.
6. Math evidence ablation: `none`, `prompt`, `prm`, `dto`, `all`.
7. AIME 2025 and Omni-MATH after candidate generation budget is available.
8. tau2/GAIA only after adding real execution/final-answer generation.

---

## 5. Claims And Guardrails

| Result | What It Can Claim |
|---|---|
| `acdan` vs `bon/sc/cot` at same model+PRM | DTO vs discrete decision rules |
| `acdan` vs `asc` | adaptive candidate/verifier allocation vs early-stop SC |
| `no_dto` | DTO is the accuracy driver |
| `no_graph`, `no_inertia` on BFCL | cost/pruning/call-saving effects |
| `no_verification` / Claude verifier | calibration and abstention effects |
| math `highest_count_accuracy` | strength of self-consistency plurality |
| evidence ablation | whether counts help through prompt, PRM, DTO, or all |

Do not claim graph/inertia improve GSM8K/MATH accuracy; those tasks are one-step
answer selection. Do not claim BFCL argument correctness from the current runner;
use official AST/executable evaluation for that.

Report real `total_real_prompt_tokens`, `mean_latency_s`, mean +- std over at
least three seeds, and the candidate diagnostics in every math table.

---

## 6. Training Study

Offline PS-GRPO remains available for module learning curves:

```bash
.venv/bin/python -m acdan.train --iters 80 --out results/psgrpo_full.json
.venv/bin/python -m acdan.train --iters 80 --no-process --out results/psgrpo_noproc.json
.venv/bin/python -m acdan.train --iters 80 --no-confidence-margin --out results/psgrpo_nocm.json
.venv/bin/python -m acdan.train --iters 80 --no-baseline --out results/psgrpo_nobaseline.json
```

Report `eval_acc_final` and the learning curve. Real LLM policy training still
requires an `LLMPolicyHead`/LoRA implementation behind the existing trainer
interfaces.
