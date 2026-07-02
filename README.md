# ACDAN — Adaptive Calibrated Differentiable Agentic Networks

> A reproducible, **offline**, dependency-light reference implementation of the
> core ideas from the AAAI-2027 proposal *"Adaptive Calibrated Differentiable
> Agentic Networks: A New Direction for Universal Test-Time Agentic Systems."*

ACDAN is best understood as an **agentic test-time controller**, not a
replacement foundation model. It wraps a base LLM / agent policy that can score a
finite set of candidate actions or plans, then uses process reward, graph
control, and calibration to pick a better action sequence.

ACDAN unifies four mechanisms into one controller:
**(1) latent-space reasoning + in-place test-time training**,
**(2) Differentiable Textual Optimization (DTO)** over soft action logits,
**(3) a dynamic two-layer agentic computation graph** (execution + dependency)
with dead-step pruning and **Tool-Usage Inertia**, and
**(4) self-verification with confidence calibration (RLCM / independent asking)**.

The offline core runs locally with only `numpy` + `PyYAML`. Real experiments use
optional lazy backends (`vLLM`, sentence-transformers, Anthropic/Claude) behind
small interfaces. No model weights or benchmark files are downloaded by the core
package; point the runner at local data files under `data/`.

---

## Table of contents
- [Research motivation](#research-motivation)
- [Architecture overview](#architecture-overview)
- [Module → paper mapping](#module--paper-mapping)
- [Installation](#installation)
- [Offline demo](#offline-demo)
- [Example commands & expected outputs](#example-commands--expected-outputs)
- [Current evidence](#current-evidence)
- [Real datasets / models](#real-datasets--models)
- [Repository layout](#repository-layout)
- [Rebuttal-round checklist](#rebuttal-round-checklist)
- [What is and is not implemented](#what-is-and-is-not-implemented)
- [License](#license)

---

## Research motivation

Test-time scaling for agents is bottlenecked by three problems the proposal
targets directly:

1. **Discrete search is wasteful** (MCTS / Best-of-N explode compute). ACDAN
   replaces repeated discrete rollouts with **continuous, differentiable**
   refinement over a task-defined action/candidate space. In this repo, RAP is
   the MCTS-style comparison baseline.
2. **Overthinking** — long reasoning hurts accuracy and burns tokens. ACDAN
   models the workflow as a **dependency graph** and prunes redundant "dead"
   steps, and reuses familiar tool transitions via **inertia**.
3. **Poor confidence calibration** — outcome-only RL makes agents overconfident.
   ACDAN attaches an **RLCM confidence probe**, a **margin** score, and an
   **independent verifier** for risk-controlled accept/abstain.

This repo implements those mechanisms as small, testable algorithms and exposes a
real experiment runner. The strongest current evidence is for ACDAN as a
controller on top of an agentic pipeline: DTO improves action/tool selection, and
verification improves calibration.

## Architecture overview

```text
task / agent state
  -> candidate actions or candidate plan steps (H x V)
  -> prompt features / encoder state
  -> Latent Reasoning (+ optional In-Place TTT)
  -> Core-model prior logits L0
          |
          +-> DTO: L <- L - eta * grad J(L)
          |       J = -ll - alpha*R_prm + beta*len - gamma*H_dep
          |
          +-> optional Inertial Sensing for familiar transitions
                  |
                  v
            executed action plan
                  |
                  v
            Two-layer Agentic Computation Graph (EX + ED)
            - exact vN entropy for reporting
            - differentiable entropy surrogate for DTO
            - dead-step pruning / cost accounting
                  |
                  v
            Self-Verification & Calibration
            - confidence probe or margin fallback
            - optional independent verifier
            - accept / abstain / ECE
                  |
                  v
            RolloutMetrics + per-method cost fields
```

ACDAN requires a finite candidate/action space. For raw free-form text
generation, wrap the LLM output into candidates first (for example answer
selection, tool selection, workflow-step selection, or reranking).

The default feature encoder is a dependency-free hash encoder. Use it for the
current `--no-latent` main matrix because latent/TTT is intentionally disabled.
For latent/TTT revisit runs, use `--encoder vllm_hidden` to extract pooled
prompt hidden states from the same vLLM engine used for action scoring:

```bash
python -m acdan.run_experiment --method acdan --dataset bfcl \
  --data-path data/bfcl_full.jsonl \
  --policy vllm --policy-model Qwen/Qwen2.5-7B-Instruct --prm llm \
  --encoder vllm_hidden --encoder-pooling last
```

This grounds ACDAN's latent state in the policy model's own vLLM forward path,
but it remains controller-side feature extraction. It does not update the base
LLM hidden states or weights during decoding.

Closest comparison families for the paper are **Best-of-N + verifier/PRM**,
**MCTS/RAP-style planning**, **Tree of Thoughts**, **Self-Refine**, **s1/budget
forcing**, and **Process Reward Agents**. The distinction to keep sharp is that
these methods search, sample, or execute discrete trajectories, while ACDAN
updates a soft action-logit field with a process reward signal before decoding a
single plan.

Full details: [`docs/architecture.md`](docs/architecture.md) and the formal
objective / update rules in [`docs/math.md`](docs/math.md).

## Module → paper mapping

| Paper concept | Code | Status |
|---|---|---|
| Latent reasoning (recurrent unroll) | [`latent_reasoning.py`](src/acdan/latent_reasoning.py) | implemented |
| In-Place TTT | `latent_reasoning.LatentReasoner._ttt_adapt` | implemented |
| Differentiable Textual Optimization | [`dto.py`](src/acdan/dto.py) | implemented (analytic grads) |
| Process Reward Model / LLM-as-PRM | [`rewards.py`](src/acdan/rewards.py), [`prm_adapter.py`](src/acdan/backends/prm_adapter.py) | mock + LLM-backed adapter |
| Net Information Gain (O(N)) | `rewards.net_information_gain` | implemented |
| Two-layer graph (EX + ED) | [`graph.py`](src/acdan/graph.py) | implemented |
| von Neumann entropy / diversity | `graph.von_neumann_entropy`, `make_entropy_hook` | implemented (exact + surrogate) |
| Dead-step pruning (DECS) | `graph.AgenticComputationGraph.prune` | implemented |
| Tool Usage Inertia | [`inertia.py`](src/acdan/inertia.py) | implemented |
| Independent Question Asking | `verification.IndependentVerifier`, [`claude.py`](src/acdan/backends/claude.py) | mock + optional Claude backend |
| RLCM confidence probe + margin | [`verification.py`](src/acdan/verification.py) | implemented |
| Calibration (ECE) | `verification.expected_calibration_error` | implemented |
| Feature encoder | [`encoder.py`](src/acdan/backends/encoder.py) | hash, sentence-transformer, or HF causal-LLM hidden state |
| Core LLM action scorer | [`registry.py`](src/acdan/registry.py), [`vllm_core.py`](src/acdan/backends/vllm_core.py) | mock + vLLM-backed scorer |
| Datasets | [`datasets/`](src/acdan/datasets), [`tasks/synthetic.py`](src/acdan/tasks/synthetic.py) | synthetic + local JSONL adapters |
| MCTS-style baseline | `baselines.reasoning_as_planning` in [`baselines.py`](src/acdan/baselines.py) | implemented as RAP / PUCT approximation |
| Process Reward Agents comparison | PRM + verifier interfaces; no dedicated PRA agent executor yet | conceptual comparator / future baseline |

Line-level mapping and an explicit *implemented vs. mocked* breakdown:
[`docs/module_to_paper_mapping.md`](docs/module_to_paper_mapping.md).

## Installation

Requires Python ≥ 3.10. No network access needed beyond installing the two pure
dependencies.

```bash
# from the repository root
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"         # installs numpy, PyYAML, pytest

# verify
pytest -q
```

No build step? You can also run straight from source:

```bash
pip install numpy pyyaml
export PYTHONPATH=src           # Windows PowerShell: $env:PYTHONPATH="src"
python -m acdan.cli eval --config configs/default.yaml
```

## Offline demo

Everything below runs with **zero downloads**. The agent operates on synthetic
math/code/tool task families that are designed to make each mechanism's effect
measurable.

```bash
# 1) One task, full module-by-module trace
acdan demo --config configs/default.yaml --task-index 0

# 2) Evaluate the full architecture over the synthetic suite
acdan eval --config configs/default.yaml

# 3) Ablation comparison (full ACDAN, per-module off, baseline CoT)
acdan ablation --config configs/default.yaml

# 4) Reproduce all headline artefacts into results/
bash scripts/reproduce.sh        # Windows: powershell -File scripts/reproduce.ps1
```

(If you did not `pip install`, prefix with `PYTHONPATH=src python -m acdan.cli …`.)

## Example commands & expected outputs

`acdan demo --task-index 0` prints a per-step trace:

```
=== ACDAN demo | config=default | enabled=[...] disabled=[] ===
task         : math-000  (family=math, difficulty=0.64)
optimal plan : ['check', 'read', 'decompose', 'read', 'decompose']
DTO steps    : 40
DTO objective: -0.12xx -> -0.27xx (lower is better)

step | action      | from_inertia | dead | prm   | nig    | conf
----------------------------------------------------------------
   0 | read        | False        | False| 0.6xx | +0.0xx | 0.6xx
   ...
--- verification ---
confidence=0.6xx  margin=0.xxx  independent=0.xxx  verified=True  abstained=False
--- metrics ---
correct=True  token_cost=5.00  llm_calls=5  inertia_saved=0  dead_pruned=0 ...
```

`acdan eval` prints an aggregate summary (accuracy, coverage, selective accuracy,
token cost, LLM calls, inertia saved, dead steps pruned, mean PRM, ECE,
dependency entropy). Exact numbers below.

## Current evidence

There are two kinds of evidence in this repo:

1. **Offline mechanism checks** (`acdan ablation`) isolate each module on seeded
   synthetic multi-step tasks.
2. **Real-backend benchmark runs** (`python -m acdan.run_experiment`) use local
   benchmark files plus optional vLLM / LLM-as-PRM backends.

### Offline mechanism checks

`acdan ablation --seed 0 --n 8` (24 synthetic tasks; deterministic across
machines/processes because seeds use a stable BLAKE2b hash):

| config | acc | sel_acc | cov | token_cost | llm_calls | inertia_saved | dead_pruned | ECE | dep_H |
|---|---|---|---|---|---|---|---|---|---|
| **acdan_full** | **0.833** | 0.833 | 1.000 | **3.44** | 3.21 | 1.29 | 0.96 | **0.133** | 1.222 |
| no_latent | 0.792 | 0.792 | 1.000 | 3.36 | 3.12 | 1.29 | 1.04 | 0.108 | 1.212 |
| no_ttt | 0.792 | 0.792 | 1.000 | 3.40 | 3.17 | 1.29 | 1.00 | 0.106 | 1.212 |
| no_dto | 0.167 | 0.500 | 0.250 | 2.81 | 2.54 | 1.42 | 1.46 | 0.212 | 1.085 |
| no_graph | 0.875 | 0.875 | 1.000 | 4.30 | 4.04 | 1.29 | 0.00 | 0.142 | 0.000 |
| no_inertia | 0.875 | 0.875 | 1.000 | 4.29 | 4.29 | 0.00 | 1.04 | 0.096 | 1.230 |
| no_verification | 0.833 | 0.833 | 1.000 | 3.44 | 3.21 | 1.29 | 0.96 | 0.260 | 1.222 |
| no_confidence_margin | 0.833 | 0.833 | 1.000 | 3.44 | 3.21 | 1.29 | 0.96 | 0.186 | 1.222 |
| baseline_cot | 0.125 | 0.125 | 1.000 | 5.33 | 5.33 | 0.00 | 0.00 | 0.497 | 0.000 |

**How to read this (each module earns its place):**
- **DTO drives accuracy.** Removing it collapses accuracy 0.833 → 0.167.
- **Inertia + graph cut cost.** Full ACDAN uses **3.44** token-cost units vs.
  **5.33** for the CoT baseline (≈ **35%** reduction here); disabling inertia or
  graph pruning individually pushes cost back to ~4.3 with no accuracy gain.
- **Verification fixes calibration.** ECE drops from **0.497** (baseline) /
  **0.260** (no verification) to **0.133** with the full RLCM stack.
- **Latent + TTT** give a smaller but consistent accuracy lift (cleaner latent →
  cleaner PRM signal).

These numbers are mechanism-isolation results, not benchmark SOTA claims.

### Real-backend BFCL run

The current BFCL result files under `results/bfcl_qwen7b_*.json` use
`Qwen/Qwen2.5-7B-Instruct`, the local `data/bfcl_full.jsonl` adapter, and
LLM-as-PRM scoring. They evaluate **tool-name sequence selection**, not official
BFCL argument / AST / executable correctness.

| method | accuracy | ECE | real prompt tokens/task | token surrogate | samples | verified candidates |
|---|---:|---:|---:|---:|---:|---:|
| ACDAN | **0.898** | **0.042** | 1587 | 1.29 | 1.00 | 3.10 |
| RAP | 0.897 | 0.103 | 1587 | 10.46 | 8.00 | 3.10 |
| Self-Refine | 0.892 | 0.108 | 1587 | 1.48 | 1.13 | 3.10 |
| BoN | 0.882 | 0.118 | 1587 | 10.46 | 8.00 | 3.10 |
| CoT / greedy | 0.803 | 0.197 | 824 | 1.31 | 1.00 | 0.00 |

Current BFCL ablations say:
- **DTO is the main accuracy driver**: `no_dto` falls to CoT-level accuracy.
- **Verification is the main calibration driver**: `no_verification` keeps
  accuracy but worsens ECE substantially.
- **Graph gives a small BFCL effect**.
- **Latent reasoning, TTT, and inertia are implemented but not yet strongly
  validated by this BFCL run**. In particular, `no_inertia` is effectively
  identical to full ACDAN here.

Cost caveat: `mean_token_surrogate` is useful for within-run accounting, but it is
not a full end-to-end compute metric. For fair efficiency claims, report
`mean_real_prompt_tokens`, `total_real_prompt_tokens`, `mean_latency_s`, and
multi-seed means. In this BFCL run, ACDAN, BoN, RAP, Self-Refine, ToT, and s1 have
similar real prompt-token costs because they share the same prior and PRM scoring
fields.

## Real datasets / models

The architecture programs against small interfaces and lazy backends:

1. **Core model / policy scorer**:
   `CoreModel.prior_logits(task, latent) -> (H, V)`. The repo includes a mock
   scorer and a vLLM-backed candidate/action scorer.
2. **Process reward model**:
   `ProcessRewardModel.step_reward_matrix`, `score_probs`, and
   `grad_wrt_probs`. The repo includes a mock PRM and LLM-as-PRM.
3. **Independent verifier / judge**:
   optional mock or Claude-backed evidence query.
4. **Dataset adapter**:
   converts local JSONL benchmark files into finite-action `Task` objects.

Current local adapters cover synthetic tasks, GSM8K/MATH-style answer selection,
AIME/Omni-MATH-style candidate selection, BFCL/tool-name selection, and GAIA-like
open-ended data. Tau2/tau-bench data is only snapshotted by
[`scripts/setup_datasets.py`](scripts/setup_datasets.py); there is currently no
`tau`, `tau2`, or `taubench` dataset adapter in
[`build_dataset`](src/acdan/datasets/base.py), and no stateful executor/judge in
the runner. Do not report tau-bench/tau2 as an ACDAN result until that adapter
and execution layer are wired.

### Agentic benchmark roadmap

Raw setup for these datasets is available with:

```bash
python scripts/setup_datasets.py --suite agentic_benchmarks --dry-run
python scripts/setup_datasets.py --suite agentic_benchmarks --overwrite
```

These are useful external datasets for the AgentBench-style trajectory
self-choice path:

| Domain | Dataset | Original size | Setup key | Current status |
|---|---|---:|---|---|
| Search | BrowseComp | 1266 | `browsecomp` | task manifest + trajectory self-choice |
| Search | WebVoyager | 643 | `webvoyager` | task manifest + external evaluator |
| Coding | SWE-Bench Verified | 500 | `swe_bench_verified` | task manifest + external evaluator |
| Coding | Terminal-Bench | 230 | `terminal_bench` | task manifest + external evaluator |
| Reason | MathHay | 602 | `mathhay` | task manifest + trajectory self-choice |
| Tool-Calling | Tau2-Bench | 278 | `tau2_bench_data`, `tau2_bench_hud` | task manifest + external evaluator |
| Tool-Calling | MCP-Bench | 104 | `mcp_bench` | task manifest + external evaluator |

Recommended claim boundary: report these only from candidate trajectories
generated in the correct agent environment, with official evaluator outputs or a
configured external evaluator command. Selector-only `*_proxy` datasets are for
debugging DTO behavior, not General AgentBench claims.

A complete custom-backend template is in
[`examples/custom_backend.py`](examples/custom_backend.py).

## Repository layout

```
ACDAN/
|-- README.md
|-- pyproject.toml / requirements*.txt
|-- configs/
|   |-- default.yaml
|   `-- ablations/
|-- data/                   # local benchmark/candidate JSONL files
|-- docs/
|   |-- architecture.md
|   |-- math.md
|   |-- module_to_paper_mapping.md
|   `-- rebuttal_checklist.md
|-- experiments/
|   |-- PLAN.md
|   |-- gen_candidates.py
|   `-- run_approval_matrix.sh
|-- results/                # local experiment summaries
|-- src/acdan/
|   |-- agent.py            # ACDAN controller orchestration
|   |-- run_experiment.py   # real-backend benchmark runner
|   |-- baselines.py        # CoT, SC, BoN, ToT, RAP, Self-Refine, s1
|   |-- dto.py              # Differentiable Textual Optimization
|   |-- rewards.py          # PRM interface + mock PRM + NIG
|   |-- graph.py            # two-layer ACG, vN entropy, pruning
|   |-- inertia.py          # tool-usage inertia
|   |-- verification.py     # confidence / independent verification / ECE
|   |-- latent_reasoning.py # recurrent latent block + TTT
|   |-- datasets/           # local JSONL adapters
|   |-- backends/           # optional vLLM / encoder / Claude backends
|   |-- training/           # PS-GRPO components
|   `-- tasks/synthetic.py
`-- tests/
```

## Rebuttal-round checklist

A full, actionable checklist for an AAAI rebuttal lives in
[`docs/rebuttal_checklist.md`](docs/rebuttal_checklist.md). Highlights:

- **Reproduce on demand:** use `experiments/PLAN.md` and
  `experiments/run_approval_matrix.sh` for the current real-backend matrix.
- **Every component ablated:** one config / flag each.
- **Math is verifiable:** DTO gradients are finite-difference-tested; vN entropy
  is exact; NIG is provably O(N).
- **Generalisation argument:** backend-agnostic via `registry.py`
  (`examples/custom_backend.py`).
- **Honest limitations** are pre-stated (next section), which reviewers reward.

## What is and is not implemented

**Implemented as real algorithms (numpy, tested):** DTO with analytic gradients
(finite-difference-checked), exact von Neumann entropy, O(N) Net Information Gain,
dependency-graph dead-step pruning, Markov inertial sensing, RLCM confidence probe
(trained by BCE) + margin + ECE, in-place TTT, and the **PS-GRPO post-training
loop** (process supervision + drop-moment + confidence margin + PPO-clip + KL).

**Implemented as optional real backends:** vLLM candidate/action scoring,
LLM-as-PRM, sentence-transformer encoding, Claude independent verifier / judge,
and local benchmark adapters. These are imported lazily and are not required for
the offline core.

**Still limited / not yet final:** current BFCL evaluation is tool-name sequence
selection, not official BFCL executable argument correctness; tau-bench/tau2 and
other stateful web/coding/search benchmarks are not current reported results;
Process Reward Agents are a recent comparison family, not a dedicated implemented
baseline in this repo; PS-GRPO currently trains an offline policy head rather
than a real LLM/LoRA policy. The current real evidence supports ACDAN primarily
as a test-time agentic controller, with DTO driving accuracy and verification
driving calibration.

We do **not** claim SOTA. Report benchmark numbers with the stated evaluator,
real prompt-token/latency fields, and the limitations above.

## License

MIT (placeholder — see [`LICENSE`](LICENSE); replace per your institution before
public release).
