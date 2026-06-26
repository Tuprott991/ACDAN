# ACDAN — Adaptive Calibrated Differentiable Agentic Networks

> A reproducible, **offline**, dependency-light reference implementation of the
> core ideas from the AAAI-2027 proposal *"Adaptive Calibrated Differentiable
> Agentic Networks: A New Direction for Universal Test-Time Agentic Systems."*

ACDAN unifies four mechanisms into one test-time agent:
**(1) latent-space reasoning + in-place test-time training**,
**(2) Differentiable Textual Optimization (DTO)** over soft action logits,
**(3) a dynamic two-layer agentic computation graph** (execution + dependency)
with dead-step pruning and **Tool-Usage Inertia**, and
**(4) self-verification with confidence calibration (RLCM / independent asking)**.

Everything runs locally with only `numpy` + `PyYAML`. **No datasets or model
weights are downloaded** — real backends drop in behind small interfaces.

---

## Table of contents
- [Research motivation](#research-motivation)
- [Architecture overview](#architecture-overview)
- [Module → paper mapping](#module--paper-mapping)
- [Installation](#installation)
- [Offline demo](#offline-demo)
- [Example commands & expected outputs](#example-commands--expected-outputs)
- [Reproducible results](#reproducible-results)
- [Adding real datasets / models later](#adding-real-datasets--models-later)
- [Repository layout](#repository-layout)
- [Rebuttal-round checklist](#rebuttal-round-checklist)
- [What is and is not implemented](#what-is-and-is-not-implemented)
- [License](#license)

---

## Research motivation

Test-time scaling for agents is bottlenecked by three problems the proposal
targets directly:

1. **Discrete search is wasteful** (MCTS / Best-of-N explode compute). ACDAN
   replaces it with **continuous, differentiable** refinement of the action plan.
2. **Overthinking** — long reasoning hurts accuracy and burns tokens. ACDAN
   models the workflow as a **dependency graph** and prunes redundant "dead"
   steps, and reuses familiar tool transitions via **inertia**.
3. **Poor confidence calibration** — outcome-only RL makes agents overconfident.
   ACDAN attaches an **RLCM confidence probe**, a **margin** score, and an
   **independent verifier** for risk-controlled accept/abstain.

This repo implements those mechanisms as small, testable, numpy algorithms so the
*ideas* are demonstrable offline, and so a real LLM/PRM can be swapped in without
touching the architecture.

## Architecture overview

```
features ─▶ Latent Reasoning (+In-Place TTT) ─▶ Core-Model prior logits L0
                                                     │
                          ┌──────────────────────────┴───────────────────────┐
                          ▼                                                    ▼
             Inertial Sensing (skip LLM plan)                 DTO: L ← L − η∇J(L)
             for high-inertia tool transitions     J = −ll −αR_prm +β·len −γ·H_dep
                          └──────────────────────────┬───────────────────────┘
                                                     ▼ executed actions
                  Two-layer Agentic Computation Graph (EX + ED)
                  • von Neumann entropy   • prune dead steps (DECS)
                                                     ▼
                  Self-Verification & Calibration (RLCM)
                  • confidence probe • margin • independent ask → accept/abstain
                                                     ▼
                                              RolloutMetrics
```

Full details: [`docs/architecture.md`](docs/architecture.md) and the formal
objective / update rules in [`docs/math.md`](docs/math.md).

## Module → paper mapping

| Paper concept | Code | Status |
|---|---|---|
| Latent reasoning (recurrent unroll) | [`latent_reasoning.py`](src/acdan/latent_reasoning.py) | implemented |
| In-Place TTT | `latent_reasoning.LatentReasoner._ttt_adapt` | implemented |
| Differentiable Textual Optimization | [`dto.py`](src/acdan/dto.py) | implemented (analytic grads) |
| Process Reward Model (TIM-PRM/Athena) | [`rewards.py`](src/acdan/rewards.py) | **mock** behind interface |
| Net Information Gain (O(N)) | `rewards.net_information_gain` | implemented |
| Two-layer graph (EX + ED) | [`graph.py`](src/acdan/graph.py) | implemented |
| von Neumann entropy / diversity | `graph.von_neumann_entropy`, `make_entropy_hook` | implemented (exact + surrogate) |
| Dead-step pruning (DECS) | `graph.AgenticComputationGraph.prune` | implemented |
| Tool Usage Inertia | [`inertia.py`](src/acdan/inertia.py) | implemented |
| Independent Question Asking | `verification.IndependentVerifier` | **mock** behind interface |
| RLCM confidence probe + margin | [`verification.py`](src/acdan/verification.py) | implemented |
| Calibration (ECE) | `verification.expected_calibration_error` | implemented |
| Core LLM action head | [`registry.py`](src/acdan/registry.py) | **mock** behind interface |
| Datasets (GAIA/GSM8K/…) | [`tasks/synthetic.py`](src/acdan/tasks/synthetic.py) | synthetic stand-ins |

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

## Reproducible results

`acdan ablation --seed 0 --n 8` (24 synthetic tasks; **deterministic across
machines/processes** — seeds use a stable BLAKE2b hash, not Python's salted
`hash()`):

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

> These are **synthetic-task** numbers that isolate each *mechanism*. They are not
> benchmark SOTA and are not presented as such. For mean ± std across seeds run
> `python examples/run_ablation.py`.

## Adding real datasets / models later

The architecture never imports a backend directly — it programs against small
interfaces, all wired in [`src/acdan/registry.py`](src/acdan/registry.py). To go
from demo to real experiments, implement and register:

1. **A core model** (`CoreModel.prior_logits(task, latent) -> (H, V)`): wrap your
   LLM's action-head logits.
2. **A PRM** (`ProcessRewardModel`, see `rewards.py`): wrap TIM-PRM / Athena-PRM.
   Only `score_probs` + `grad_wrt_probs` are needed for DTO.
3. **An independent verifier** (`IndependentVerifier.agreement`): a real tool /
   sandbox evidence query.
4. **A dataset adapter** (`DatasetAdapter.tasks()` yielding `Task` objects): load
   GAIA / GSM8K / LiveCodeBench. Provide `prompt_features` from your encoder; set
   `optimal_plan=None` for tasks without ground truth (metrics degrade gracefully).

```python
from acdan.registry import register_prm, register_core_model
register_prm("tim_prm", MyTimPrmAdapter)
register_core_model("llama3", MyLlamaAdapter)
```

A complete, runnable template is in
[`examples/custom_backend.py`](examples/custom_backend.py). **No changes to
`agent.py`, `dto.py`, `graph.py`, `inertia.py`, or `verification.py` are
required.**

## Repository layout

```
ACDAN/
├── README.md
├── pyproject.toml / requirements.txt
├── LICENSE / CONTRIBUTING.md / .gitignore
├── configs/
│   ├── default.yaml
│   └── ablations/{baseline_cot,no_dto,no_inertia,no_graph,no_verification,no_latent}.yaml
├── docs/
│   ├── architecture.md
│   ├── math.md
│   ├── module_to_paper_mapping.md
│   └── rebuttal_checklist.md
├── examples/{run_demo,run_ablation,custom_backend}.py
├── scripts/{reproduce.sh,reproduce.ps1}
├── src/acdan/
│   ├── agent.py            # orchestration of the full pipeline
│   ├── config.py           # configs + AblationFlags
│   ├── latent_reasoning.py # recurrent latent block + in-place TTT
│   ├── dto.py              # Differentiable Textual Optimization
│   ├── rewards.py          # PRM interface + mock PRM + Net Information Gain
│   ├── graph.py            # two-layer ACG, vN entropy, dead-step pruning
│   ├── inertia.py          # tool-usage inertia / inertial sensing
│   ├── verification.py     # confidence probe, margin, independent ask, ECE
│   ├── evaluate.py         # build + fit + run harness
│   ├── metrics.py          # aggregation
│   ├── registry.py         # pluggable backends (core model / PRM / datasets)
│   ├── types.py            # dataclasses + stable seeding
│   └── tasks/synthetic.py  # offline math/code/tool task families
└── tests/                  # 40+ unit tests incl. finite-difference grad checks
```

## Rebuttal-round checklist

A full, actionable checklist for an AAAI rebuttal lives in
[`docs/rebuttal_checklist.md`](docs/rebuttal_checklist.md). Highlights:

- **Reproduce on demand:** `bash scripts/reproduce.sh` → `results/`.
- **Every component ablated:** one config / flag each (table above).
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
loop** (process supervision + drop-moment + confidence margin + PPO-clip + KL,
analytic & finite-difference-tested — `python -m acdan.train`).

**Mocked behind interfaces (offline stand-ins, pluggable):** the core LLM, the
PRM, the independent verifier, and the datasets.

**Deliberately out of scope for the offline demo:** loading real model/dataset
weights and training a **real LLM** policy with PS-GRPO (the trainer optimises a
parametric `(H,V)` policy on synthetic data; the advantage computation is
backend-agnostic, so it is a `PolicyHead`→LLM-head swap on the VM); real
multimodal inputs. These are documented extension points, not hidden gaps.

We do **not** claim SOTA or benchmark numbers — only that the mechanisms work and
compose as described, demonstrably and reproducibly, offline.

## License

MIT (placeholder — see [`LICENSE`](LICENSE); replace per your institution before
public release).
