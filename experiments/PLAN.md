# ACDAN Experiment Plan (VM)

End-to-end plan for the current repo. The runner is currently strongest as a
candidate/action selector: math datasets are `H=1, V=K` answer selection over
pre-generated candidates, and BFCL is tool-name sequence selection over a fixed
tool vocabulary. DTO is the main decision module. Latent reasoning and in-place
TTT are now treated as experimental modules, not the default claim path.

Current main-run default: disable latent reasoning with `--no-latent`. This
keeps encoder features available to the shallow projected state but disables the
latent recurrent unroll and in-place TTT. After the primary matrix is stable,
run the latent revisit commands in Section 3.

Use `--encoder hash` for the current no-latent main matrix. Use
`--encoder vllm_hidden --encoder-pooling last` only for latent/TTT revisit runs
where the hidden-state features can actually affect the module under test.

---

## 0. VM Setup

Recommended VM: 1x A100 80GB or H100 for a 7B/8B vLLM policy plus LLM-as-PRM.

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
PY=.venv/bin/python
M=Qwen/Qwen2.5-7B-Instruct
ENCODER_ARGS="--encoder hash"
LATENT_ARGS="--no-latent"
MONITOR_ARGS="--monitor --progress-every 10"

$PY -m acdan.run_experiment --method acdan --dataset bfcl \
  --data-path data/bfcl_test.jsonl --limit 50 \
  --policy vllm --policy-model $M --prm llm \
  $ENCODER_ARGS $LATENT_ARGS $MONITOR_ARGS \
  --out results/_smoke_bfcl.json --save-per-task
```

Use `summary.mean_latency_s` from the smoke run to estimate full runtime:

```text
benchmark_eval_seconds ~= model_load_seconds + N_tasks * mean_latency_s
```

If vLLM startup fails with `RuntimeError: Engine core initialization failed`,
first lower `--vllm-gpu-memory-utilization` or `--vllm-max-model-len`. If the
installed vLLM build cannot expose prompt hidden states, use this named fallback
only for a low-memory diagnostic:

```bash
$PY -m acdan.run_experiment --method acdan --dataset bfcl \
  --data-path data/bfcl_test.jsonl --limit 25 \
  --policy vllm --policy-model $M --prm llm \
  --encoder hf --encoder-mode input_emb --encoder-pooling mean \
  --encoder-dtype bfloat16 --encoder-device cpu --encoder-max-length 2048 \
  --no-latent \
  --out results/_smoke_bfcl_hfencoder_lowmem.json
```

For no-latent main runs, `hash` is intentional because the latent/TTT module is
off and `vllm_hidden` would only add overhead. For latent/TTT claims, switch
back to `--encoder vllm_hidden --encoder-pooling last`.

---

## 1. Dataset Setup

The commands below are the exact setup commands supported by the current repo.

### 1a. Runner-Ready Datasets

Prepare all datasets that have current repo support:

```bash
PY=.venv/bin/python
$PY scripts/setup_datasets.py --suite all --overwrite
```

Equivalent explicit commands:

```bash
$PY scripts/setup_datasets.py --suite math --overwrite
$PY scripts/setup_datasets.py --suite bfcl --overwrite
$PY scripts/setup_datasets.py --suite agentic_raw --overwrite
```

Expected primary files:

| Domain | Dataset | Setup command | Output | Runner status |
|---|---|---|---|---|
| Reason | GSM8K | `--suite math` | `data/gsm8k_train.jsonl`, `data/gsm8k_test.jsonl` | runnable after candidate generation |
| Reason | MATH-500 | `--suite math` | `data/math500.jsonl` | runnable after candidate generation |
| Reason | AIME 2025 | `--suite math` | `data/aime2025.jsonl` | runnable after candidate generation |
| Reason | Omni-MATH | `--suite math` | `data/omni_math.jsonl` | runnable after candidate generation |
| Tool-Calling | BFCL | `--suite bfcl` | `data/bfcl_full.jsonl`, `data/bfcl_dev.jsonl`, `data/bfcl_test.jsonl` | runnable as tool-name selection |
| Tool-Calling | Tau2 raw | `--suite agentic_raw` | `data/raw/agentic_benchmarks/tau2_bench_*` | raw only; no runner adapter yet |

Sanity checks:

```bash
wc -l data/gsm8k_test.jsonl data/math500.jsonl data/aime2025.jsonl data/bfcl_dev.jsonl data/bfcl_test.jsonl
$PY -m acdan.run_experiment --method acdan --dataset synthetic --limit 24 \
  --out results/_smoke_synthetic.json
```

BFCL caveat: the current adapter preserves `gold_calls`, but the runner selects
tool names, not full function-call argument ASTs. Final BFCL claims should still
be checked against the official AST/executable evaluation when implemented.

### 1b. Raw Roadmap Datasets

These commands prepare sources for future agentic adapters. Raw setup success is
not the same as runnable ACDAN evaluation.

Dry run:

```bash
$PY scripts/setup_datasets.py --suite agentic_benchmarks --dry-run \
  --out-dir data/roadmap_dryrun
```

Download all raw roadmap datasets:

```bash
$PY scripts/setup_datasets.py --suite agentic_benchmarks --overwrite
```

Download only the requested roadmap set:

```bash
$PY scripts/setup_datasets.py --suite agentic_benchmarks --overwrite \
  --benchmarks browsecomp,webvoyager,swe_bench_verified,terminal_bench,mathhay,tau2_bench_data,tau2_bench_hud,mcp_bench
```

Roadmap status:

| Domain | Dataset | Original | Setup key | Current repo status |
|---|---:|---|---|---|
| Search | BrowseComp | 1266 | `browsecomp` | task manifest + trajectory self-choice |
| Search | WebVoyager | 643 | `webvoyager` | task manifest + external evaluator |
| Coding | SWE-Bench Verified | 500 | `swe_bench_verified` | task manifest + external evaluator |
| Coding | Terminal-Bench | 230 | `terminal_bench` | task manifest + external evaluator |
| Reason | MathHay | 602 | `mathhay` | task manifest + trajectory self-choice |
| Tool-Calling | Tau2-Bench | 278 | `tau2_bench_data`, `tau2_bench_hud` | task manifest + external evaluator |
| Tool-Calling | MCP-Bench | 104 | `mcp_bench` | task manifest + external evaluator |

Before reporting final numbers, candidate trajectories must come from the
appropriate agent environment and environment benchmarks must use official or
reproducible evaluator outputs (`is_correct`, `score`, or `--evaluator-command`).

---

## 2. Candidate Generation

Generate neutral-order math candidates. Do not use plurality order for primary
results; keep it as a named self-consistency control.

```bash
PY=.venv/bin/python
M=Qwen/Qwen2.5-7B-Instruct
TAG=qwen7b

$PY experiments/gen_candidates.py --model $M \
  --in data/gsm8k_test.jsonl --out data/gsm8k_${TAG}_k8_sample.jsonl \
  --k 8 --order sample

$PY experiments/gen_candidates.py --model $M \
  --in data/math500.jsonl --out data/math500_${TAG}_k8_sample.jsonl \
  --k 8 --order sample

$PY experiments/gen_candidates.py --model $M \
  --in data/aime2025.jsonl --out data/aime2025_${TAG}_k16_sample.jsonl \
  --k 16 --order sample

$PY experiments/gen_candidates.py --model $M \
  --in data/omni_math.jsonl --out data/omni_math_${TAG}_k8_sample.jsonl \
  --k 8 --order sample
```

Useful controls:

```bash
# Neutral randomized order control.
$PY experiments/gen_candidates.py --model $M \
  --in data/gsm8k_test.jsonl --out data/gsm8k_${TAG}_k8_shuffle0.jsonl \
  --k 8 --order shuffle --shuffle-seed 0

# Explicit self-consistency/plurality artifact.
$PY experiments/gen_candidates.py --model $M \
  --in data/gsm8k_test.jsonl --out data/gsm8k_${TAG}_k8_plurality.jsonl \
  --k 8 --order plurality
```

Every math result summary reports `oracle_candidate_accuracy`,
`always_first_accuracy`, `always_last_accuracy`, and `highest_count_accuracy`.
If first/last accuracy is suspiciously high, regenerate with `--order shuffle`.

---

## 3. Experiment Matrix And Commands

Shared VM variables:

```bash
PY=.venv/bin/python
M=Qwen/Qwen2.5-7B-Instruct
TAG=qwen7b
ENCODER_ARGS="--encoder hash"
LATENT_ARGS="--no-latent"
MONITOR_ARGS="--monitor --progress-every 25"
mkdir -p results results/approval
```

The result summary now includes an `ablation` object. Check it before merging
tables; main runs should show `latent_reasoning=false` and `in_place_ttt=false`.

### 3a. Reason Domain: GSM8K, MATH-500, AIME 2025, Omni-MATH

Primary methods for a single dataset:

```bash
D=data/gsm8k_${TAG}_k8_sample.jsonl
DATASET=gsm8k

for METHOD in acdan bon asc sc cot; do
  $PY -m acdan.run_experiment --method $METHOD --dataset $DATASET \
    --data-path $D --policy vllm --policy-model $M --prm llm \
    $ENCODER_ARGS $LATENT_ARGS $MONITOR_ARGS \
    --math-evidence none --n 8 --seed 0 --save-per-task \
    --out results/${DATASET}_${TAG}_${METHOD}.json
done
```

Repeat by changing `DATASET` and `D`:

```bash
DATASET=math500; D=data/math500_${TAG}_k8_sample.jsonl
DATASET=aime2025; D=data/aime2025_${TAG}_k16_sample.jsonl
DATASET=omni_math; D=data/omni_math_${TAG}_k8_sample.jsonl
```

Secondary search/refinement baselines:

```bash
for METHOD in tot rap refine s1; do
  $PY -m acdan.run_experiment --method $METHOD --dataset $DATASET \
    --data-path $D --policy vllm --policy-model $M --prm llm \
    $ENCODER_ARGS $LATENT_ARGS $MONITOR_ARGS \
    --math-evidence none --n 8 --seed 0 --save-per-task \
    --out results/${DATASET}_${TAG}_${METHOD}.json
done
```

Math evidence ablation, ACDAN only:

```bash
for EVID in none prompt prm dto all; do
  $PY -m acdan.run_experiment --method acdan --dataset $DATASET \
    --data-path $D --policy vllm --policy-model $M --prm llm \
    $ENCODER_ARGS $LATENT_ARGS $MONITOR_ARGS \
    --math-evidence $EVID --math-count-weight 0.35 \
    --n 8 --seed 0 --save-per-task \
    --out results/${DATASET}_${TAG}_evidence_${EVID}.json
done
```

DTO-focused ablations under the current no-latent main setting:

```bash
for ABL in no_dto no_verification; do
  $PY -m acdan.run_experiment --method acdan --disable $ABL \
    --dataset $DATASET --data-path $D \
    --policy vllm --policy-model $M --prm llm \
    $ENCODER_ARGS $LATENT_ARGS $MONITOR_ARGS \
    --math-evidence none --n 8 --seed 0 --save-per-task \
    --out results/${DATASET}_${TAG}_abl_${ABL}_no_latent.json
done
```

Latent revisit after the primary results are complete:

```bash
LATENT_ENCODER_ARGS="--encoder vllm_hidden --encoder-pooling last"

# Full latent + TTT.
$PY -m acdan.run_experiment --method acdan --dataset $DATASET \
  --data-path $D --policy vllm --policy-model $M --prm llm \
  $LATENT_ENCODER_ARGS $MONITOR_ARGS \
  --math-evidence none --n 8 --seed 0 --save-per-task \
  --out results/${DATASET}_${TAG}_latent_full.json

# Latent recurrent state without in-place TTT.
$PY -m acdan.run_experiment --method acdan --no-ttt --dataset $DATASET \
  --data-path $D --policy vllm --policy-model $M --prm llm \
  $LATENT_ENCODER_ARGS $MONITOR_ARGS \
  --math-evidence none --n 8 --seed 0 --save-per-task \
  --out results/${DATASET}_${TAG}_latent_no_ttt.json

# Current main setting, repeated for direct comparison.
$PY -m acdan.run_experiment --method acdan --no-latent --dataset $DATASET \
  --data-path $D --policy vllm --policy-model $M --prm llm \
  $LATENT_ENCODER_ARGS $MONITOR_ARGS \
  --math-evidence none --n 8 --seed 0 --save-per-task \
  --out results/${DATASET}_${TAG}_latent_off.json
```

Full approval matrix:

```bash
MODEL=Qwen/Qwen2.5-7B-Instruct TAG=qwen7b \
  ENCODER_ARGS="--encoder hash" \
  LATENT_ARGS="--no-latent" \
  MONITOR_ARGS="--monitor --progress-every 25" \
  experiments/run_approval_matrix.sh
```

For a full-latent rerun, set both `LATENT_ARGS=""` and
`ENCODER_ARGS="--encoder vllm_hidden --encoder-pooling last"`.

### 3b. Tool-Calling Domain: BFCL

```bash
TRAIN=data/bfcl_dev.jsonl
TEST=data/bfcl_test.jsonl

for METHOD in acdan bon asc sc cot; do
  $PY -m acdan.run_experiment --method $METHOD --dataset bfcl \
    --data-path $TEST --policy vllm --policy-model $M --prm llm \
    $ENCODER_ARGS $LATENT_ARGS $MONITOR_ARGS \
    --n 8 --seed 0 --save-per-task \
    --fit-inertia --inertia-fit-path $TRAIN \
    --out results/bfcl_${TAG}_${METHOD}.json
done
```

Secondary BFCL baselines:

```bash
for METHOD in tot rap refine s1; do
  $PY -m acdan.run_experiment --method $METHOD --dataset bfcl \
    --data-path $TEST --policy vllm --policy-model $M --prm llm \
    $ENCODER_ARGS $LATENT_ARGS $MONITOR_ARGS \
    --n 8 --seed 0 --save-per-task \
    --fit-inertia --inertia-fit-path $TRAIN \
    --out results/bfcl_${TAG}_${METHOD}.json
done
```

BFCL module ablations under the no-latent main setting:

```bash
for ABL in no_dto no_graph no_inertia no_verification; do
  $PY -m acdan.run_experiment --method acdan --disable $ABL \
    --dataset bfcl --data-path $TEST \
    --policy vllm --policy-model $M --prm llm \
    $ENCODER_ARGS $LATENT_ARGS $MONITOR_ARGS \
    --n 8 --seed 0 --save-per-task \
    --fit-inertia --inertia-fit-path $TRAIN \
    --out results/bfcl_${TAG}_abl_${ABL}_no_latent.json
done
```

BFCL latent revisit:

```bash
LATENT_ENCODER_ARGS="--encoder vllm_hidden --encoder-pooling last"

$PY -m acdan.run_experiment --method acdan --dataset bfcl \
  --data-path $TEST --policy vllm --policy-model $M --prm llm \
  $LATENT_ENCODER_ARGS $MONITOR_ARGS \
  --n 8 --seed 0 --save-per-task \
  --fit-inertia --inertia-fit-path $TRAIN \
  --out results/bfcl_${TAG}_latent_full.json

$PY -m acdan.run_experiment --method acdan --no-ttt --dataset bfcl \
  --data-path $TEST --policy vllm --policy-model $M --prm llm \
  $LATENT_ENCODER_ARGS $MONITOR_ARGS \
  --n 8 --seed 0 --save-per-task \
  --fit-inertia --inertia-fit-path $TRAIN \
  --out results/bfcl_${TAG}_latent_no_ttt.json

$PY -m acdan.run_experiment --method acdan --no-latent --dataset bfcl \
  --data-path $TEST --policy vllm --policy-model $M --prm llm \
  $LATENT_ENCODER_ARGS $MONITOR_ARGS \
  --n 8 --seed 0 --save-per-task \
  --fit-inertia --inertia-fit-path $TRAIN \
  --out results/bfcl_${TAG}_latent_off.json
```

Optional verifier/calibration study:

```bash
$PY -m acdan.run_experiment --method acdan --dataset bfcl \
  --data-path $TEST --policy vllm --policy-model $M --prm llm \
  $ENCODER_ARGS $LATENT_ARGS $MONITOR_ARGS \
  --verifier claude --judge-model claude-opus-4-8 \
  --fit-inertia --inertia-fit-path $TRAIN \
  --out results/bfcl_${TAG}_acdan_claudeverify.json
```

### 3c. General AgentBench: Search, Coding, Reason, Tool-Calling

Use this path for the paper-style datasets: BrowseComp, WebVoyager, SWE-Bench
Verified, Terminal-Bench, MathHay, Tau2-Bench, and MCP-Bench. This path prepares
tasks, consumes K candidate trajectories from an agent/executor, then uses ACDAN
as the self-choice selector. It reports selected score, pass@K, oracle score,
and verification gap.

Prepare raw sources and sampled task manifests:

```bash
$PY scripts/setup_datasets.py --suite agentic_benchmarks --overwrite \
  --benchmarks browsecomp,webvoyager,swe_bench_verified,terminal_bench,mathhay,tau2_bench_data,tau2_bench_hud,mcp_bench

$PY experiments/prepare_agentbench.py --datasets browsecomp,webvoyager,swe_bench_verified,tau2_bench \
  --out-dir data/agentbench --seed 0

# Archive-backed datasets use their raw extracted folders.
$PY experiments/prepare_agentbench.py --datasets terminal_bench \
  --out-dir data/agentbench --source-path data/raw/agentic_benchmarks/terminal_bench --seed 0
$PY experiments/prepare_agentbench.py --datasets mathhay \
  --out-dir data/agentbench --source-path data/raw/agentic_benchmarks/mathhay --seed 0
$PY experiments/prepare_agentbench.py --datasets mcp_bench \
  --out-dir data/agentbench --source-path data/raw/agentic_benchmarks/mcp_bench --seed 0
```

Generate or import K candidate trajectories per task into this schema:

```json
{
  "task": {
    "task_id": "browsecomp-00001",
    "dataset": "browsecomp",
    "domain": "search",
    "instruction": "...",
    "evaluator": "semantic_qa",
    "gold": "..."
  },
  "candidates": [
    {
      "candidate_id": "0",
      "final_answer": "...",
      "trajectory": [{"role": "assistant", "content": "..."}],
      "is_correct": false
    }
  ]
}
```

If your agent/executor writes one prediction per line, group it with:

```bash
$PY experiments/build_agentbench_candidates.py \
  --tasks data/agentbench/${DATASET}_tasks.jsonl \
  --predictions results/agentbench/${DATASET}_${TAG}_predictions.jsonl \
  --out results/agentbench/${DATASET}_${TAG}_k8_candidates.jsonl \
  --min-candidates 8
```

For official environment benchmarks, fill `is_correct` or `score` from the
official harness, or pass an evaluator command. Run ACDAN self-choice:

```bash
DATASET=browsecomp
CAND=results/agentbench/${DATASET}_${TAG}_k8_candidates.jsonl

for METHOD in acdan bon asc sc cot; do
  $PY experiments/run_agentbench_selection.py --method $METHOD \
    --candidates-path $CAND \
    --policy vllm --policy-model $M --prm llm \
    $ENCODER_ARGS $LATENT_ARGS $MONITOR_ARGS \
    --n 8 --seed 0 --save-per-task \
    --out results/agentbench/${DATASET}_${TAG}_${METHOD}.json
done
```

External evaluator examples:

```bash
$PY experiments/run_agentbench_selection.py --method acdan \
  --candidates-path results/agentbench/swe_bench_verified_${TAG}_k8_candidates.jsonl \
  --policy vllm --policy-model $M --prm llm \
  $ENCODER_ARGS $LATENT_ARGS $MONITOR_ARGS \
  --evaluator-command external_swe_bench="python official_swe_eval.py --input {input} --output {output}" \
  --out results/agentbench/swe_bench_verified_${TAG}_acdan.json
```

Do not report `browsecomp_proxy`/`mathhay_proxy` as General AgentBench. Those
names are selector-only smoke/proxy datasets for debugging DTO behavior.

---

## 4. Recommended Run Order

1. VM setup, tests, and synthetic smoke run.
2. `scripts/setup_datasets.py --suite all --overwrite`.
3. Generate GSM8K, MATH-500, AIME 2025, and Omni-MATH candidate files.
4. BFCL and GSM8K `--limit 50` smoke runs with `--no-latent`.
5. GSM8K/MATH-500 primary matrix with `--math-evidence none`.
6. Math evidence ablation: `none`, `prompt`, `prm`, `dto`, `all`.
7. BFCL full matrix with graph/inertia/verifier ablations.
8. Secondary baselines: `tot`, `rap`, `refine`, `s1`.
9. AIME 2025 and Omni-MATH after the main runs are stable.
10. Latent revisit: full latent, `--no-ttt`, and `--no-latent`.
11. Raw roadmap setup for BrowseComp/WebVoyager/SWE-Bench/Terminal-Bench/MathHay/Tau2/MCP.
12. Add executors/adapters before claiming roadmap benchmark numbers.

---

## 5. Claims And Guardrails

| Result | What it can claim |
|---|---|
| `acdan` vs `bon/sc/cot` at same model+PRM | DTO versus discrete selection rules |
| `acdan` vs `asc` | adaptive allocation versus early-stop self-consistency |
| `no_dto` | whether DTO is the accuracy driver |
| `no_graph`, `no_inertia` on BFCL | cost/pruning/call-saving effects |
| `no_verification` / Claude verifier | calibration and abstention effects |
| `latent_full` vs `latent_no_ttt` vs `latent_off` | whether latent reasoning or TTT helps after the main matrix |
| math `highest_count_accuracy` | strength of self-consistency plurality |
| evidence ablation | whether counts help through prompt, PRM, DTO, or all |
| `tot`, `rap`, `refine`, `s1` | search/refinement decision rules under the same prior+PRM |

Do not claim graph/inertia improve GSM8K/MATH accuracy; those are one-step
answer-selection tasks. Do not claim BFCL argument correctness from the current
runner until official AST/executable evaluation is connected.

Report `ablation`, `total_real_prompt_tokens`, `mean_latency_s`,
`mean_samples`, `mean_verified_candidates`, `mean_prm_passes`, mean +- std over
seeds, and math candidate diagnostics in every table.

---

## 6. Training Study

Offline PS-GRPO remains available for module learning curves:

```bash
$PY -m acdan.train --iters 80 --out results/psgrpo_full.json
$PY -m acdan.train --iters 80 --no-process --out results/psgrpo_noproc.json
$PY -m acdan.train --iters 80 --no-confidence-margin --out results/psgrpo_nocm.json
$PY -m acdan.train --iters 80 --no-baseline --out results/psgrpo_nobaseline.json
```

Report `eval_acc_final` and the learning curve. Real LLM policy training still
requires an `LLMPolicyHead`/LoRA implementation behind the existing trainer
interfaces.
