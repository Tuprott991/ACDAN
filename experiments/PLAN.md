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
| Tool-Calling | Tau2 raw | `--suite agentic_raw` | `data/raw/agentic_benchmarks/tau2_bench_*` | r  aw only; no runner adapter yet |

Sanity checks:

```bash
wc -l data/gsm8k_test.jsonl data/math500.jsonl data/aime2025.jsonl data/bfcl_dev.jsonl data/bfcl_test.jsonl
$PY -m acdan.run_experiment --method acdan --dataset synthetic --limit 24 \
  --out results/_smoke_synthetic.json
```

BFCL caveat: the current adapter preserves `gold_calls`, but the runner selects
tool names, not full function-call argument ASTs. Final BFCL claims should still
be checked against the official AST/executable evaluation when implemented.

### 1b. Official AgentBench Sources

Do not use `scripts/setup_datasets.py --suite agentic_benchmarks` for paper
results. That command is only a raw-source roadmap downloader. Reportable runs
use the pinned commits in `configs/agentbench.lock.json` and the official
General AgentBench execution layer.

```bash
PY=.venv/bin/python
ROOT=$(pwd)
GAB=data/external/General-AgentBench
WV=data/external/WebVoyager
GAB_REV=35f5c027c31ddcb3366b28674c6cb2957460c0e2
WV_REV=5a7896738c10bfb8b9edccce6bb0e0411f8ae569

# Preview, then clone the pinned repositories.
$PY scripts/setup_agentbench_official.py --dry-run
$PY scripts/setup_agentbench_official.py

# Install the unified runner and Tau simulator into this VM environment.
# The General AgentBench pyproject is intentionally minimal; its requirements
# file is required for the actual runner.
pip install -r "$GAB/general_agent/requirements.txt"
pip install -e "$GAB/general_agent"
pip install -e "$GAB/benchmarks/tau2-bench"

# MCP-Bench launches its own local MCP servers. Install each server's pinned
# dependencies, then fill the upstream key file without committing secrets.
cd "$GAB/benchmarks/mcp-bench/mcp_servers"
bash ./install.sh
cd "$ROOT"
${EDITOR:-nano} "$GAB/benchmarks/mcp-bench/mcp_servers/api_key"
# Required by the pinned suite: NPS_API_KEY, NASA_API_KEY, HF_TOKEN,
# GOOGLE_MAPS_API_KEY, and NCI_API_KEY.

# WebVoyager is separate from General AgentBench and needs Chrome/Selenium.
pip install -r "$WV/requirements.txt"
docker info
```

Create score-blind task manifests from exact known files. The adapters never
recursively scan benchmark repositories.

```bash
$PY experiments/prepare_agentbench.py \
  --datasets browsecomp,mathhay,swe_bench_verified,terminal_bench,tau2_bench,mcp_bench \
  --source-path "$GAB" --source-revision "$GAB_REV" \
  --out-dir data/agentbench

$PY experiments/prepare_agentbench.py --datasets webvoyager \
  --source-path "$WV" --source-revision "$WV_REV" \
  --out-dir data/agentbench

$PY experiments/validate_agentbench.py \
  --tasks-dir data/agentbench --require-provenance

# Fail before spending GPU/API budget if a harness, key, or container is absent.
$PY experiments/preflight_agentbench.py --strict
```

Expected paper subsets:

| Domain | Dataset | Tasks | Native executor/evaluator |
|---|---|---:|---|
| Search | BrowseComp | 124 | General AgentBench search + Serper/native evaluator |
| Search | WebVoyager | 65 | pinned Selenium runner + multimodal judge |
| Coding | SWE-Bench Verified | 50 | General AgentBench/OpenHands + Docker tests |
| Coding | Terminal-Bench | 80 | General AgentBench/Terminal-Bench + Docker tests |
| Reason | MathHay | 75 | General AgentBench MathHay evaluator |
| Tool-Calling | Tau2-Bench | 50 | General AgentBench + Tau user simulator/native reward |
| Tool-Calling | MCP-Bench | 52 | General AgentBench MCP servers + native judge |

The checked-in manifests already match these pinned sources. Regenerate them on
the VM anyway so source hashes and checkout revisions are verified before spend.

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
# To enable vLLM hidden-state features and latent/TTT:
# ENCODER_ARGS="--encoder vllm_hidden --encoder-pooling last"
# LATENT_ARGS=""
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

The reportable flow has four immutable stages:

```text
native K rollouts -> blind candidate file -> blind selection -> score join
                         |                       |               |
                    no outcomes             no outcomes     official metrics
```

#### A. Start Local vLLM For The Unified Agent

General AgentBench calls models through LiteLLM/OpenAI-compatible APIs. Run the
generation model as a server; do not use ACDAN's offline vLLM object as the
environment agent.

```bash
PY=.venv/bin/python
M=Qwen/Qwen2.5-7B-Instruct
TAG=qwen7b
K=8

vllm serve "$M" --host 0.0.0.0 --port 8000 \
  --api-key EMPTY --enable-auto-tool-choice --tool-call-parser hermes

# In the experiment shell:
export OPENAI_API_KEY=EMPTY
export OPENAI_API_BASE=http://127.0.0.1:8000/v1
export SERPER_API_KEY="YOUR_SERPER_API_KEY"
GEN_MODEL="openai/$M"

# The pinned runner receives the same values through --env-file.
cat > .env <<EOF
OPENAI_API_KEY=$OPENAI_API_KEY
OPENAI_API_BASE=$OPENAI_API_BASE
SERPER_API_KEY=$SERPER_API_KEY
EOF
chmod 600 .env
```

Use a model with reliable native tool calling. A text-only Qwen model cannot run
the visual WebVoyager protocol; use a compatible multimodal model for that
separate track.

#### B. Native K-Trajectory Generation And Official Scoring

Run one dataset first. Omitting `--execute` prints the exact upstream command.

```bash
DATASET=browsecomp
NATIVE=runs/agentbench/native/${DATASET}_${TAG}_k${K}

$PY experiments/run_agentbench_official.py \
  --dataset "$DATASET" --model "$GEN_MODEL" --model-name "$TAG" \
  --k "$K" --base-seed 42 --output-dir "$NATIVE"

$PY experiments/run_agentbench_official.py \
  --dataset "$DATASET" --model "$GEN_MODEL" --model-name "$TAG" \
  --k "$K" --base-seed 42 --output-dir "$NATIVE" --execute
```

Dataset mapping used by this command:

| ACDAN key | Native key | Required execution layer |
|---|---|---|
| `browsecomp` | `search` | Serper search and native answer evaluator |
| `mathhay` | `mathhay` | long-context MathHay evaluator |
| `swe_bench_verified` | `swebench` | OpenHands, repository container, tests |
| `terminal_bench` | `terminalbench` | terminal container and task tests |
| `tau2_bench` | `tau2bench` | stateful simulator and user model |
| `mcp_bench` | `mcpbench` | official MCP servers and native judge |

For a one-task harness smoke, make a native subset and pass it to the runner:

```bash
UPSTREAM=data/external/General-AgentBench/general_agent/data
$PY experiments/subset_agentbench_native_tasks.py \
  --source "$UPSTREAM/search_benchmark.json" \
  --out runs/agentbench/smoke/search_1.json --limit 1

$PY experiments/run_agentbench_official.py \
  --dataset browsecomp --model "$GEN_MODEL" --model-name "$TAG" \
  --k 2 --task-file runs/agentbench/smoke/search_1.json \
  --output-dir runs/agentbench/native/browsecomp_smoke --execute
```

#### C. Import Blind Candidates And Immutable Scores

```bash
ART=runs/agentbench/artifacts
mkdir -p "$ART"

$PY experiments/import_general_agentbench.py \
  --dataset "$DATASET" --source "$NATIVE" --k "$K" \
  --tasks data/agentbench/${DATASET}_tasks.jsonl \
  --candidates-out "$ART/${DATASET}_${TAG}_k${K}_candidates.jsonl" \
  --trajectories-out "$ART/${DATASET}_${TAG}_k${K}_trajectories.jsonl" \
  --scores-out "$ART/${DATASET}_${TAG}_k${K}_scores.jsonl" \
  --evaluator-version general-agentbench@35f5c027

$PY experiments/validate_agentbench.py \
  --tasks data/agentbench/${DATASET}_tasks.jsonl \
  --candidates "$ART/${DATASET}_${TAG}_k${K}_candidates.jsonl" \
  --scores "$ART/${DATASET}_${TAG}_k${K}_scores.jsonl" \
  --min-candidates "$K" --require-blind --require-provenance
```

The candidate file contains trajectories and generation cost but no score,
correctness flag, or gold answer. The score file is loaded only after selection.

#### D. Blind ACDAN Selection And Score Join

```bash
ENCODER_ARGS="--encoder hash"
LATENT_ARGS="--no-latent"
MONITOR_ARGS="--monitor --progress-every 25"
SELECTOR_ARGS="--task-preview-chars 4096 --candidate-preview-chars 2048"
CAND="$ART/${DATASET}_${TAG}_k${K}_candidates.jsonl"
SCORES="$ART/${DATASET}_${TAG}_k${K}_scores.jsonl"

for METHOD in acdan bon asc sc cot; do
  SEL=results/agentbench/reportable/${DATASET}_${TAG}_${METHOD}_selection.json
  OUT=results/agentbench/reportable/${DATASET}_${TAG}_${METHOD}.json

  $PY experiments/run_agentbench_selection.py --method "$METHOD" \
    --candidates-path "$CAND" --selection-only \
    --policy vllm --policy-model "$M" --prm llm \
    $ENCODER_ARGS $LATENT_ARGS $MONITOR_ARGS $SELECTOR_ARGS \
    --n "$K" --seed 0 --save-per-task --out "$SEL"

  $PY experiments/evaluate_agentbench_selection.py \
    --selection "$SEL" --candidates "$CAND" --scores "$SCORES" \
    --out "$OUT"
done
```

Preview limits preserve both the beginning and end of each trajectory, including
the final response and late tool observations. Use `0` only when the model
context can safely score complete trajectories for every method.

After all six native datasets have artifacts, the equivalent matrix command is:

```bash
DATASETS="browsecomp mathhay swe_bench_verified terminal_bench tau2_bench mcp_bench" \
MODEL="$M" TAG="$TAG" K="$K" \
ARTIFACT_DIR=runs/agentbench/artifacts \
ENCODER_ARGS="--encoder hash" LATENT_ARGS="--no-latent" \
bash experiments/run_agentbench_matrix.sh
```

#### E. Held-Out Calibration

Never fit confidence on the reported test tasks. Run the same selector on a
disjoint calibration task-ID file, join its official scores, then fit:

```bash
$PY experiments/fit_agentbench_calibrator.py \
  --calibration-result results/agentbench/calibration/${DATASET}_${TAG}_acdan.json \
  --out results/agentbench/calibrators/${DATASET}_${TAG}_acdan.json

$PY experiments/evaluate_agentbench_selection.py \
  --selection results/agentbench/reportable/${DATASET}_${TAG}_acdan_selection.json \
  --candidates "$CAND" --scores "$SCORES" \
  --calibrator results/agentbench/calibrators/${DATASET}_${TAG}_acdan.json \
  --out results/agentbench/reportable/${DATASET}_${TAG}_acdan.json
```

Without a disjoint calibration split, report raw-confidence ECE and label it as
uncalibrated. Do not fit the calibrator on test outcomes.

#### F. WebVoyager Separate Track

WebVoyager is not supported by the General AgentBench unified runner. Run its
pinned Selenium implementation K times with a multimodal model, preserving one
output directory per pass, then import and score the screenshots:

```bash
WV=data/external/WebVoyager
WV_TAG=gpt4o
WEBVOYAGER_MODEL=gpt-4o
WEBVOYAGER_JUDGE=gpt-4o
WV_NATIVE=runs/agentbench/native/webvoyager_${WV_TAG}_k${K}

$PY experiments/export_webvoyager_subset.py \
  --manifest data/agentbench/webvoyager_tasks.jsonl \
  --native-source "$WV/data/WebVoyager_data.jsonl" \
  --out runs/agentbench/native/webvoyager_tasks_65.jsonl

for PASS in $(seq 1 "$K"); do
  python "$WV/run.py" \
    --test_file runs/agentbench/native/webvoyager_tasks_65.jsonl \
    --api_key "$OPENAI_API_KEY" --api_model "$WEBVOYAGER_MODEL" \
    --seed $((41 + PASS)) --headless \
    --output_dir "$WV_NATIVE/pass_${PASS}"
done

$PY experiments/import_webvoyager.py \
  --source "$WV_NATIVE" --tasks data/agentbench/webvoyager_tasks.jsonl --k "$K" \
  --candidates-out "$ART/webvoyager_${WV_TAG}_k${K}_candidates.jsonl" \
  --trajectories-out "$ART/webvoyager_${WV_TAG}_k${K}_trajectories.jsonl"

$PY experiments/score_webvoyager.py \
  --checkout "$WV" --source "$WV_NATIVE" --k "$K" \
  --judge-model "$WEBVOYAGER_JUDGE" \
  --evaluator-version webvoyager@5a789673 \
  --out "$ART/webvoyager_${WV_TAG}_k${K}_scores.jsonl" --resume
```

The official WebVoyager task file contains 643 tasks. The export command freezes
the exact 65 IDs in the checked-in manifest; do not evaluate a different subset.

#### G. Tables And Confidence Intervals

```bash
$PY experiments/collect_agentbench_results.py \
  --results results/agentbench/reportable/{browsecomp,mathhay,swe_bench_verified,terminal_bench,tau2_bench,mcp_bench}_${TAG}_{acdan,bon,asc,sc,cot}.json \
  --reference bon --bootstrap-samples 2000 --seed 0 \
  --csv-out results/agentbench/reportable/summary.csv \
  --comparisons-out results/agentbench/reportable/paired_vs_bon.json
```

Headline metrics are selected accuracy/official score, pass@K, recovery rate,
verification gap or oracle regret, ECE, Brier, AURC, selector prompt tokens,
selector latency, samples, and actual verified candidates. Generation tokens are
reported separately because every selector shares the same frozen candidate
pool. `mean_token_surrogate` is an appendix diagnostic, not an AgentBench cost.

Do not report `browsecomp_proxy`, `mathhay_proxy`, the old
`data/agentbench/browsecomp_qwen7b_k8_candidates.jsonl`, or
`--allow-unevaluated` runs as General AgentBench results.

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
11. Clone and verify pinned official AgentBench repositories; regenerate all seven manifests.
12. Run one-task native harness smoke tests, then import blind candidates and official scores.
13. Run ACDAN/BoN/ASC/SC/CoT blind selection on shared candidate pools.
14. Join scores, fit calibration only on disjoint tasks, and collect paired/Pareto results.

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

For AgentBench report selected accuracy/official score, pass@K, recovery rate,
verification gap or oracle regret, ECE, Brier, AURC, actual selector prompt
tokens, selector latency, samples, and verified candidates. For the original
math/BFCL runner also report `ablation`, `total_real_prompt_tokens`,
`mean_latency_s`, `mean_samples`, `mean_verified_candidates`,
`mean_prm_passes`, mean +- std over seeds, and candidate diagnostics.

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
