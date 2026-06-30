# ACDAN Experiment Plan (VM)

End-to-end plan for running ACDAN after the evaluator cleanup. The framing is
still **tool/operator-selection DTO**: the policy scores candidate actions into
an `(H, V)` prior, DTO refines that prior, and a PRM supplies reward gradients.
Math runs are `H=1, V=K` answer selection over pre-generated candidates; graph
and inertia should be evaluated on multi-step tool/agent tasks, not sold as math
accuracy modules. Latent reasoning gates the LLM-PRM target sharpness through
`quality(latent)`. The default `--encoder hash` is only an offline smoke-test
feature encoder. For real claims, prefer `--encoder hf`, which extracts prompt
features from a causal LLM's pooled input embeddings or final hidden state before
the LM head. This is still controller-side feature extraction; it does not
perform hidden-state TTT inside the base LLM weights.

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
ENCODER_ARGS="--encoder hf --encoder-mode last_hidden --encoder-pooling last \
  --encoder-dtype bfloat16 --encoder-device cpu --encoder-max-length 2048"

.venv/bin/python -m acdan.run_experiment --method acdan --dataset bfcl \
  --data-path data/bfcl_full.jsonl --limit 50 \
  --policy vllm --policy-model Qwen/Qwen2.5-7B-Instruct --prm llm \
  $ENCODER_ARGS \
  --out results/_smoke_bfcl.json --save-per-task
```

Use `summary.mean_latency_s` from this smoke run to rescale estimates:
`benchmark_eval_seconds ~= model_load + N_tasks * mean_latency_s`.

Faster offline/diagnostic smoke tests may still use the default hash encoder, but
do not use hash-encoder numbers as the main latent/TTT evidence.

Alternative low-memory latent encoder smoke test:

```bash
.venv/bin/python -m acdan.run_experiment --method acdan --dataset bfcl \
  --data-path data/bfcl_full.jsonl --limit 25 \
  --policy vllm --policy-model Qwen/Qwen2.5-7B-Instruct --prm llm \
  --encoder hf --encoder-mode input_emb --encoder-pooling mean \
  --encoder-dtype bfloat16 --encoder-device cpu --encoder-max-length 2048 \
  --out results/_smoke_bfcl_hfencoder_lowmem.json
```

Memory caveat: with `--policy vllm --encoder hf`, the model may be loaded once
by vLLM for action scoring and once by Transformers for hidden-state features.
The commands above use `--encoder-device cpu` so the HF encoder does not steal
GPU memory before vLLM starts. If you intentionally run the HF encoder on GPU,
also reduce vLLM's reservation, for example
`--vllm-gpu-memory-utilization 0.55`, or use a smaller `--encoder-model`.

If vLLM fails at startup with `RuntimeError: Engine core initialization failed`,
first rerun with `--encoder-device cpu`; then try lowering
`--vllm-gpu-memory-utilization`, `--vllm-max-model-len`, or using
`--encoder-mode input_emb`. Use `--encoder hash` only as a diagnostic smoke test,
not for headline latent/TTT claims.

---

## 1. Dataset Setup

The setup script prepares stable local files and raw snapshots:

```bash
.venv/bin/python scripts/setup_datasets.py --suite all --overwrite
```

Roadmap stateful benchmarks are prepared separately so the normal math/BFCL
workflow stays fast and stable:

```bash
# Inspect sources and manifest shape without downloading anything.
.venv/bin/python scripts/setup_datasets.py --suite agentic_benchmarks --dry-run \
  --out-dir data/roadmap_dryrun

# Download all raw roadmap benchmark sources into data/raw/agentic_benchmarks/.
.venv/bin/python scripts/setup_datasets.py --suite agentic_benchmarks --overwrite

# Or download a targeted subset.
.venv/bin/python scripts/setup_datasets.py --suite agentic_benchmarks \
  --benchmarks webvoyager,swe_bench_verified,tau2_bench_data,tau2_bench_hud
```

Fast sanity checks before candidate generation:

```bash
wc -l data/gsm8k_test.jsonl data/math500.jsonl data/aime2025.jsonl data/bfcl_full.jsonl
.venv/bin/python -m acdan.run_experiment --method acdan --dataset synthetic --limit 24 \
  --out results/_smoke_synthetic.json
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
  --in data/gsm8k_test.jsonl --out data/gsm8k_${TAG}_k8_sample.jsonl \
  --k 8 --order sample

.venv/bin/python experiments/gen_candidates.py --model $M \
  --in data/math500.jsonl --out data/math500_${TAG}_k8_sample.jsonl \
  --k 8 --order sample

.venv/bin/python experiments/gen_candidates.py --model $M \
  --in data/aime2025.jsonl --out data/aime2025_${TAG}_k16_sample.jsonl \
  --k 16 --order sample
```

Optional broad math file after the primary runs are stable:

```bash
.venv/bin/python experiments/gen_candidates.py --model $M \
  --in data/omni_math.jsonl --out data/omni_math_${TAG}_k8_sample.jsonl \
  --k 8 --order sample
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
If `always_first_accuracy` or `always_last_accuracy` is suspiciously close to
selected-method accuracy, regenerate candidates with `--order shuffle` and a
fixed `--shuffle-seed` before making claims.

---

## 3. Main Matrix

Policies: `Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`,
`Qwen/Qwen2.5-3B-Instruct`.

Use HF causal-LLM features for all real reported runs:

```bash
ENCODER_ARGS="--encoder hf --encoder-mode last_hidden --encoder-pooling last \
  --encoder-dtype bfloat16 --encoder-device cpu --encoder-max-length 2048"
```

If runtime or memory is tight, use `--encoder-mode input_emb --encoder-pooling
mean`, lower `--vllm-gpu-memory-utilization`, use a smaller `--encoder-model`,
or run a named low-memory ablation. Do not silently fall back to `--encoder hash`
for headline latent/TTT claims.

Primary methods: `acdan`, `bon`, `asc`, `sc`, `cot`. `asc` is an early-stopping
self-consistency baseline: it stops once the plurality answer reaches a
confidence threshold, giving ACDAN a direct adaptive-compute competitor.

Secondary search/refinement baselines now available in the same runner:
`tot`, `rap`, `refine`, `s1`. Here `rap` is the MCTS-style baseline. Include
them in the appendix matrix or targeted BFCL/math robustness table after the
primary Pareto gates pass. Process Reward Agents should be cited as a recent
PRM-guided agent comparison family, but this repo does not yet implement a
dedicated PRA executor baseline.

### 3a. Math

Default rigorous math run: no candidate counts in prompts, no PRM count bonus, no
DTO count prior.

```bash
M=Qwen/Qwen2.5-7B-Instruct; TAG=qwen7b; D=data/gsm8k_${TAG}_k8_sample.jsonl
ENCODER_ARGS="--encoder hf --encoder-mode last_hidden --encoder-pooling last --encoder-dtype bfloat16 --encoder-device cpu --encoder-max-length 2048"
for METHOD in acdan bon asc sc cot; do
  .venv/bin/python -m acdan.run_experiment --method $METHOD --dataset gsm8k \
    --data-path $D --policy vllm --policy-model $M --prm llm \
    $ENCODER_ARGS \
    --math-evidence none --n 8 --seed 0 \
    --out results/gsm8k_${TAG}_${METHOD}.json
done
```

Optional secondary baselines at matched `N=8`:

```bash
for METHOD in tot rap refine s1; do
  .venv/bin/python -m acdan.run_experiment --method $METHOD --dataset gsm8k \
    --data-path $D --policy vllm --policy-model $M --prm llm \
    $ENCODER_ARGS \
    --math-evidence none --n 8 --seed 0 \
    --out results/gsm8k_${TAG}_${METHOD}.json
done
```

Self-consistency evidence ablation, ACDAN only:

```bash
for EVID in none prompt prm dto all; do
  .venv/bin/python -m acdan.run_experiment --method acdan --dataset gsm8k \
    --data-path $D --policy vllm --policy-model $M --prm llm \
    $ENCODER_ARGS \
    --math-evidence $EVID --math-count-weight 0.35 \
    --out results/gsm8k_${TAG}_evidence_${EVID}.json
done
```

Full approval matrix with per-task artifacts and Pareto collection:

```bash
MODEL=Qwen/Qwen2.5-7B-Instruct TAG=qwen7b \
  ENCODER_ARGS="--encoder hf --encoder-mode last_hidden --encoder-pooling last --encoder-dtype bfloat16 --encoder-device cpu --encoder-max-length 2048" \
  experiments/run_approval_matrix.sh
```

This writes `results/approval/*.json`, `results/approval/pareto.csv`, and
`results/approval/pareto.svg`.

Module ablations on math should focus on DTO/verification/calibration. Graph and
inertia are structurally weak on `H=1` tasks. `no_latent` and `no_ttt` should now
move only through latent-quality-conditioned PRM sharpness and calibration; if
they remain flat, report them as weak/negative evidence rather than overstating
latent reasoning.

```bash
for ABL in no_dto no_verification no_latent no_ttt; do
  .venv/bin/python -m acdan.run_experiment --method acdan --disable $ABL \
    --dataset gsm8k --data-path $D --policy vllm --policy-model $M --prm llm \
    $ENCODER_ARGS \
    --math-evidence none --out results/gsm8k_${TAG}_abl_${ABL}.json
done
```

Repeat the same block for `math500`, `aime2025`, and later `omni_math` once
candidate files exist. Use dataset names `math500` or `math` for MATH-500;
the approval script currently uses `math` with
`data/math500_${TAG}_k8_sample.jsonl`.

### 3b. BFCL / Tool Use

Use BFCL first for graph/inertia cost behavior. Fit inertia from a separate file
or omit it; the runner no longer fits from evaluation gold.

```bash
M=Qwen/Qwen2.5-7B-Instruct; TAG=qwen7b
ENCODER_ARGS="--encoder hf --encoder-mode last_hidden --encoder-pooling last --encoder-dtype bfloat16 --encoder-device cpu --encoder-max-length 2048"
TRAIN=data/bfcl_dev.jsonl 
TEST=data/bfcl_full.jsonl

for METHOD in acdan bon asc sc cot; do
  .venv/bin/python -m acdan.run_experiment --method $METHOD --dataset bfcl \
    --data-path $TEST --policy vllm --policy-model $M --prm llm \
    $ENCODER_ARGS \
    --n 8 --seed 0 \
    --fit-inertia --inertia-fit-path $TRAIN \
    --out results/bfcl_${TAG}_${METHOD}.json
done
```

Secondary BFCL baselines:

```bash
for METHOD in tot rap refine s1; do
  .venv/bin/python -m acdan.run_experiment --method $METHOD --dataset bfcl \
    --data-path $TEST --policy vllm --policy-model $M --prm llm \
    $ENCODER_ARGS \
    --n 8 --seed 0 \
    --out results/bfcl_${TAG}_${METHOD}.json
done
```

BFCL module ablations for the main table:

```bash
for ABL in no_dto no_graph no_inertia no_verification no_latent no_ttt; do
  .venv/bin/python -m acdan.run_experiment --method acdan --disable $ABL \
    --dataset bfcl --data-path $TEST --policy vllm --policy-model $M --prm llm \
    $ENCODER_ARGS \
    --n 8 --seed 0 --fit-inertia --inertia-fit-path $TRAIN \
    --out results/bfcl_${TAG}_abl_${ABL}.json
done
```

Calibration study:

```bash
.venv/bin/python -m acdan.run_experiment --method acdan --dataset bfcl \
  --data-path $TEST --policy vllm --policy-model $M --prm llm \
  $ENCODER_ARGS \
  --verifier claude --judge-model claude-opus-4-8 \
  --fit-inertia --inertia-fit-path $TRAIN \
  --out results/bfcl_${TAG}_acdan_claudeverify.json
```

### 3c. Stateful Agentic Raw Data

`data/raw/tau2_bench_data/` and `data/raw/tau2_bench_hud/` are available for the
next adapter/executor milestone. They are raw snapshots only: `build_dataset`
currently has no `tau`, `tau2`, or `taubench` adapter, and the runner has no
tau-bench stateful executor/judge. Do not report tau2/GAIA as final ACDAN
numbers until the runner produces final answers through an execution layer.

Recommended stateful benchmark roadmap:

| Domain | Dataset | Original size | Setup key | Raw source | Current repo status |
|---|---|---:|---|---|---|
| Search | BrowseComp | 1266 | `browsecomp` | `smolagents/browse_comp` | raw snapshot only |
| Search | WebVoyager | 643 | `webvoyager` | `agentorg/webvoyager` | raw snapshot only |
| Coding | SWE-Bench Verified | 500 | `swe_bench_verified` | `SWE-bench/SWE-bench_Verified` | raw snapshot only |
| Coding | Terminal-Bench | 230 | `terminal_bench` | `laude-institute/terminal-bench-datasets` | raw archive only |
| Reason | MathHay | 602 | `mathhay` | `cxcscmu/General-AgentBench` | raw archive only |
| Tool-Calling | Tau2-Bench | 278 | `tau2_bench_data`, `tau2_bench_hud` | `HuggingFaceH4/tau2-bench-data`, `Genteki/tau2-bench` | raw snapshot only |
| Tool-Calling | MCP-Bench | 104 | `mcp_bench` | `Accenture/mcp-bench` | raw archive only |

Raw setup success does not mean ACDAN can report those benchmark numbers. Before
reporting a stateful benchmark, add:

1. A dataset adapter that converts the raw task format into candidate actions or
   an agent state.
2. An executor for browser, terminal, repository, tau2, or MCP state transitions.
3. A reproducible judge using the official evaluator when one exists.
4. Budget-matched baselines: CoT/greedy, BoN+PRM, RAP/MCTS, ToT/Self-Refine/s1,
   plus Process Reward Agents if implementing a direct PRM-guided agent baseline.

---

## 4. Recommended Order

1. Offline tests and synthetic run.
2. `scripts/setup_datasets.py --suite all`.
3. Generate neutral-order candidate files matching the approval script names:
  `gsm8k_${TAG}_k8_sample`, `math500_${TAG}_k8_sample`, and
  `aime2025_${TAG}_k16_sample`.
4. BFCL and GSM8K smoke runs with `--limit 50` to measure throughput and catch
  prompt/token regressions.
5. GSM8K/MATH-500 primary matrix with `--math-evidence none`.
6. Math evidence ablation: `none`, `prompt`, `prm`, `dto`, `all`.
7. BFCL full matrix, including graph/inertia ablations and calibration.
8. Secondary baselines (`tot`, `rap`, `refine`, `s1`) on GSM8K and BFCL if the
  primary matrix is stable.
9. AIME 2025 and Omni-MATH after candidate generation budget is available.
10. tau2/GAIA only after adding real execution/final-answer generation.

---

## 5. Claims And Guardrails

| Result | What It Can Claim |
|---|---|
| `acdan` vs `bon/sc/cot` at same model+PRM | DTO vs discrete decision rules |
| `acdan` vs `asc` | adaptive candidate/verifier allocation vs early-stop SC |
| `no_dto` | DTO is the accuracy driver |
| `no_graph`, `no_inertia` on BFCL | cost/pruning/call-saving effects |
| `no_verification` / Claude verifier | calibration and abstention effects |
| `no_latent`, `no_ttt` | effect of latent-quality-gated PRM sharpness/calibration |
| math `highest_count_accuracy` | strength of self-consistency plurality |
| evidence ablation | whether counts help through prompt, PRM, DTO, or all |
| `tot`, `rap`, `refine`, `s1` | search/refinement decision rules under the same prior+PRM |

Do not claim graph/inertia improve GSM8K/MATH accuracy; those tasks are one-step
answer selection. Do not claim BFCL argument correctness from the current runner;
use official AST/executable evaluation for that.

Report real `total_real_prompt_tokens`, `mean_latency_s`, mean +- std over at
least three seeds, and the candidate diagnostics in every math table.

Treat `evidence_all` as ACDAN with all count-evidence channels enabled, not as a
separate competing method. If it becomes the chosen math default, name it clearly
as an ACDAN evidence setting in tables.

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
