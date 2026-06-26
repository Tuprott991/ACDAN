# AAAI Rebuttal-Round Checklist

A practical checklist for using this repository during an AAAI rebuttal. The goal
is to answer reviewer questions with *runnable evidence* in minutes, not days.

## 0. Reproducibility (have this ready before reviews arrive)

- [ ] `bash scripts/reproduce.sh` (or `scripts/reproduce.ps1`) regenerates every
      headline number into `results/`.
- [ ] Every run is seeded from `config.seed`; `tests/test_agent.py::test_determinism_same_seed_same_result`
      guards bit-for-bit reproducibility.
- [ ] `pip install -e ".[dev]" && pytest` is green (40+ tests).

## 1. "Did you ablate every component?" → yes, one config each

Run `acdan ablation` (or `examples/run_ablation.py` for mean±std over seeds). Each
module maps to a metric it is responsible for:

| Disabled module | Expected, defensible effect |
|---|---|
| `no_dto` | large **accuracy** drop (DTO is the accuracy driver) |
| `no_inertia` | higher **token cost / LLM calls**, accuracy ~unchanged |
| `no_graph` | higher **token cost** (no pruning), `dep_H = 0` |
| `no_verification` | worse **ECE**, coverage → 1.0 (no risk control) |
| `no_confidence_margin` | worse **calibration** |
| `no_latent` / `no_ttt` | modest **accuracy** drop (cleaner latent → cleaner PRM) |
| `baseline_cot` | worst on every axis |

If a reviewer proposes a *new* ablation, it is usually a one-line `AblationFlags`
change plus a `run_evaluation` call.

## 2. "Are the math claims real or hand-waved?"

- [ ] DTO gradient is **analytic** and **finite-difference-verified**
      (`tests/test_dto.py`).
- [ ] von Neumann entropy is computed by **exact** eigendecomposition; the
      optimised surrogate is disclosed in [`math.md §3`](math.md). Do not claim
      you back-prop the exact spectral entropy — you optimise a consistent proxy.
- [ ] NIG is genuinely **O(N)** (`rewards.net_information_gain`).

## 3. "Your gains might be cherry-picked / seed luck."

- [ ] Report **mean ± std across seeds** via `examples/run_ablation.py`
      (`SEEDS = [0,1,2]`, extend as needed).
- [ ] Add seeds: `for s in 0 1 2 3 4; do acdan eval --seed $s; done`.
- [ ] Selective metrics (`coverage`, `selective_accuracy`) show the
      risk-controlled behaviour, not just raw accuracy.

## 4. "Does it generalise / would it work with a real model?"

- [ ] The architecture is **backend-agnostic**: `examples/custom_backend.py`
      swaps in custom PRM + core model with **no change** to the core modules.
- [ ] Point reviewers to the `registry.py` seam and the
      [mapping table](module_to_paper_mapping.md) ("Implemented vs. mocked").
- [ ] Be explicit: the offline numbers are on **synthetic** tasks designed to
      isolate each mechanism; they demonstrate *mechanism*, not benchmark SOTA.

## 5. "What is NOT implemented?" (pre-empt this honestly)

State these up front to build credibility:

- [ ] No real LLM / PRM weights are loaded (mock interfaces only).
- [x] PS-GRPO **training** loop is implemented (offline, numpy, analytic +
      finite-difference-tested) over the parametric `(H,V)` policy, with process
      supervision + drop-moment + RLCM confidence margin + PPO-clip + KL
      (`acdan.training`, `python -m acdan.train`). What's *not* yet done:
      training a **real LLM** policy — the advantages are backend-agnostic, so
      this is a `PolicyHead`→LLM-head swap on the VM.
- [ ] Multimodality is interface-level, not exercised with real images.
- [ ] The >50% token-reduction claim is illustrated on synthetic workflows
      (~40–50% in the demo); a real-benchmark number requires real backends.

## 6. New-experiment turnaround during rebuttal

- [ ] New task family: add to `tasks/synthetic.py::_FAMILIES`.
- [ ] New metric: extend `RolloutMetrics` + `metrics.summarize`.
- [ ] New baseline: a new YAML in `configs/` (or `AblationFlags`).
- [ ] Sensitivity sweep: loop a hyper-parameter in a 10-line script over
      `run_evaluation`.

## 7. Figures reviewers like

- [ ] DTO objective trace (`Plan.objective_trace`) → "optimisation converges".
- [ ] Accuracy vs. token-cost scatter across ablations → "Pareto improvement".
- [ ] Reliability diagram from `(confidence, correct)` pairs → calibration.
- [ ] Inertia transition heatmap (`InertialSensor.transition_matrix`).

> Keep a one-paragraph "limitations" note in the paper that matches §5 here.
> Reviewers reward honesty far more than they punish a clearly-scoped demo.
