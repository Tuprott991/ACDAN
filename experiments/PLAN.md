# ACDAN Experiment Plan (VM)

End-to-end plan for running ACDAN vs. baselines across **3 policies × 3 benchmark
families** on a rented GPU VM. Every command is copy-pasteable. Durations are
estimates with stated assumptions and a recompute formula — **measure real
throughput in Step 0 and rescale**.

> Framing (locked): **tool/operator-selection DTO** — the policy produces an
> `(H, V)` prior by scoring candidate actions; DTO refines it; PRM supplies the
> gradient. Math benchmarks run as `H=1, V=K` answer-selection over
> pre-generated candidates. No closed API can run DTO (no logits) — Claude is
> used only as the independent verifier / judge.

---

## 0. VM setup + smoke test (do this first)

**Recommended VM:** 1× A100 80GB or H100 (7–8B policy + PRM fit comfortably).
~$1.5–3/GPU-hr spot.

```bash
git clone <your-repo> && cd ACDAN
python -m venv .venv && source .venv/bin/activate
pip install -e .                      # core (numpy, pyyaml)
pip install -r requirements-gpu.txt   # vllm, torch, transformers, sentence-transformers, anthropic
export ANTHROPIC_API_KEY=...          # only for --verifier claude / GAIA judge

# Sanity: offline mock pipeline + unit tests (no GPU needed)
pytest -q
python -m acdan.run_experiment --method acdan --dataset synthetic --limit 24

# Step 0a — measure REAL throughput on a tiny slice (rescale all estimates from this)
python -m acdan.run_experiment --method acdan --dataset bfcl \
  --data-path data/bfcl_dev.jsonl --limit 50 \
  --policy vllm --policy-model Qwen/Qwen2.5-7B-Instruct --prm llm \
  --out results/_smoke_bfcl.json --save-per-task
#   -> read summary.mean_latency_s ; multiply by task counts below.
```

**Throughput assumptions used for the estimates** (override with your Step-0a numbers):

| Symbol | Meaning | Assumed (A100, 7B, vLLM) |
|---|---|---|
| `gen` | full CoT generation throughput | 5,000 tok/s |
| `score` | short scoring/PRM requests (max_tokens=1, batched) | 1,500 req/s |
| `load` | vLLM model load (per model init) | ~2 min |
| `judge` | Claude judge/verifier latency (sequential) | ~3 s/call (or use Batches: ~1 h/whole set, 50% cost) |

Per-task ACDAN cost = `(H·V core scorings + H·V PRM scorings) / score` ≈ tens of ms
for these task shapes, so **for math the candidate-generation step dominates; for
tools/agentic the model-load + judge dominate**, not the ACDAN eval itself.

---

## 1. Data prep (local files — nothing is downloaded by ACDAN)

Place your prepared JSONL under `data/`. Schemas are documented in
`src/acdan/datasets/{base,gsm8k,bfcl,gaia}.py`.

```bash
# Math: generate K candidate answers per question (the only heavy math step)
python experiments/gen_candidates.py --model Qwen/Qwen2.5-7B-Instruct \
  --in data/gsm8k_test.jsonl --out data/gsm8k_qwen7b_k8.jsonl --k 8
python experiments/gen_candidates.py --model meta-llama/Llama-3.1-8B-Instruct \
  --in data/gsm8k_test.jsonl --out data/gsm8k_llama8b_k8.jsonl --k 8
python experiments/gen_candidates.py --model Qwen/Qwen2.5-7B-Instruct \
  --in data/math500.jsonl --out data/math500_qwen7b_k8.jsonl --k 8

# Tools: convert BFCL/ToolBench to the JSONL schema (prompt, tools, gold)
#   -> data/bfcl_test.jsonl   (your conversion; see bfcl.py schema)
# Agentic: convert GAIA validation to (prompt, tools, gold)
#   -> data/gaia_val.jsonl    (see gaia.py schema)
```

**Candidate-generation duration** (per model):
`tasks · K · gen_tokens / gen`. GSM8K(1319)·8·400 / 5000 ≈ **18 min**; MATH-500 ≈ **7 min**.

---

## 2. The experiment matrix

Policies `P`: `Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`,
`Qwen/Qwen2.5-3B-Instruct`. Methods: `acdan`, `bon`, `sc`, `cot` (+ ablations).
PRM: `--prm llm` (LLM-as-PRM, reuses the policy) or a dedicated Math-PRM.

### 2a. Math — GSM8K / MATH-500 (answer-selection)

```bash
M=Qwen/Qwen2.5-7B-Instruct; TAG=qwen7b; D=data/gsm8k_${TAG}_k8.jsonl
for METHOD in acdan bon sc cot; do
  python -m acdan.run_experiment --method $METHOD --dataset gsm8k --data-path $D \
    --policy vllm --policy-model $M --prm llm --n 8 --seed 0 \
    --out results/gsm8k_${TAG}_${METHOD}.json
done
# Ablations (ACDAN only, GSM8K, one model is enough):
for ABL in no_dto no_graph no_inertia no_verification; do
  python -m acdan.run_experiment --method acdan --disable $ABL --dataset gsm8k \
    --data-path $D --policy vllm --policy-model $M --prm llm \
    --out results/gsm8k_${TAG}_abl_${ABL}.json
done
```
Repeat the 4-method block for `llama8b` and `qwen3b`. Add `--seed 1 --seed 2` runs
for mean±std (eval is cheap; candidates are reused).

### 2b. Tools — BFCL / ToolBench (tool-sequence) — best fit, run FIRST

```bash
M=Qwen/Qwen2.5-7B-Instruct; TAG=qwen7b; D=data/bfcl_test.jsonl
for METHOD in acdan bon sc cot; do
  python -m acdan.run_experiment --method $METHOD --dataset bfcl --data-path $D \
    --policy vllm --policy-model $M --prm llm --n 8 --seed 0 \
    --fit-inertia --out results/bfcl_${TAG}_${METHOD}.json
done
# Verification/calibration study: turn the Claude independent verifier on
python -m acdan.run_experiment --method acdan --dataset bfcl --data-path $D \
  --policy vllm --policy-model $M --prm llm --verifier claude \
  --judge-model claude-opus-4-8 --out results/bfcl_${TAG}_acdan_claudeverify.json
```

### 2c. Agentic — GAIA / AgentBench (Claude-judged) — run LAST

```bash
M=Qwen/Qwen2.5-7B-Instruct; TAG=qwen7b; D=data/gaia_val.jsonl
for METHOD in acdan bon cot; do
  python -m acdan.run_experiment --method $METHOD --dataset gaia --data-path $D \
    --policy vllm --policy-model $M --prm llm --n 8 \
    --out results/gaia_${TAG}_${METHOD}.json
done
```

### 2d. Offline dry-run (no GPU) — validate every script first
Swap `--policy vllm --policy-model $M --prm llm` for `--policy mock --prm mock`
and `--dataset synthetic`. This is exactly what the unit tests cover.

---

## 3. Duration & cost estimates

Counts: GSM8K 1319, MATH-500 500, BFCL 500 (subset; scale up later), GAIA 165.
Eval-only ACDAN/baseline passes are **seconds–minutes** (the `(H,V)` scorings are
tiny); the real time sinks are **candidate generation (math)**, **model load**,
and **Claude judging (GAIA)**.

| Phase | Per-run formula | Est. per model | × models | Subtotal |
|---|---|---|---|---|
| Math candidate gen (GSM8K) | `1319·8·400/gen` | ~18 min | 3 | ~55 min |
| Math candidate gen (MATH-500) | `500·8·400/gen` | ~7 min | 3 | ~21 min |
| GSM8K eval (4 methods + 4 abl) | load + `N·(H·V·2)/score` | load 2m + ~3 min | 3 | ~15 min |
| MATH eval | same shape | ~4 min | 3 | ~12 min |
| BFCL eval (4 methods, +inertia) | load 2m + ~3 min | ~5 min | 3 | ~15 min |
| BFCL Claude-verify study | +`500·judge` (or Batches) | ~25 min seq / ~1 h batch | 1 | ~25 min |
| GAIA predictions (3 methods) | load 2m + ~1 min | ~3 min | 3 | ~9 min |
| GAIA Claude judge | `165·methods·judge` (Batches) | ~1 h batch | — | ~1 h |
| 3-seed re-eval (cheap, reuse candidates) | eval only ×2 | ~10 min | — | ~20 min |
| **GPU total** | | | | **≈ 2.5–3.5 GPU-hours** |
| **Wall-clock incl. Claude judging** | | | | **≈ 4–5 hours** |

**Cost (rough):**
- GPU: ~3 GPU-hr × $1.5–3/hr ≈ **$5–10** (spot A100).
- Claude API (verifier + GAIA judge): independent verifier ~1.8k calls + GAIA judge
  ~1k calls, short prompts. At Opus-4.8 rates with the Batches API (50% off):
  **≈ $15–30**. Use Sonnet-4.6 for the high-volume verifier to cut to ~$5–10.

> If you scale BFCL/ToolBench to the full set or add WebArena, multiply the tool
> rows accordingly; GPU eval stays cheap, model-load and judging dominate.

**Recompute formula (drop in your Step-0a `mean_latency_s = L`):**
`benchmark_eval_seconds ≈ model_load + N_tasks · L`. Everything else (candidate
gen, judging) scales linearly in the table above.

---

## 4. Recommended order (by fit + cost)

1. **Offline dry-run** (§2d) — confirm all scripts run with mocks. *(minutes, no GPU)*
2. **BFCL** (§2b) — cleanest fit; validates inertia + graph + the DTO-vs-BoN claim. *(~15 min/model)*
3. **GSM8K → MATH-500** (§2a) — candidate gen then cheap eval. *(~25 min/model incl. gen)*
4. **Verification study** (Claude verifier on BFCL) — calibration/ECE numbers.
5. **GAIA** (§2c) — last; Claude-judged, most expensive per task.
6. **Seeds 1–2** re-eval for mean±std (candidates reused).

---

## 5. The comparisons each run feeds (paper tables)

| Result files | Comparison | Claim |
|---|---|---|
| `*_acdan` vs `*_bon` vs `*_sc` vs `*_cot` | accuracy at matched model+PRM | **DTO > discrete search** |
| `*_acdan` vs `*_abl_no_dto/...` | per-module ablation | each module earns its place |
| `bon` at N=1,4,8,16 vs `acdan` | accuracy vs compute (sweep `--n`) | inference-scaling curve |
| `*_acdan_claudeverify` ECE vs `*_abl_no_verification` | calibration | RLCM calibration |
| train family A → eval family B | cross-benchmark | OOD generalization |
| `qwen3b acdan` vs `qwen7b cot` | small+ACDAN vs larger | efficiency (claim only if it holds) |

Collect with: `python -c "import glob,json; [print(f, json.load(open(f))['summary']['accuracy']) for f in glob.glob('results/*.json')]"`.

---

## 6. Honesty guardrails (carry into the paper)

- In the selection framing, ACDAN and BoN share the prior + PRM, so the **headline
  is accuracy at matched budget**, not raw cost — state this; don't overclaim
  token savings that come from the framing rather than the method.
- Report **real** `total_real_prompt_tokens` and `mean_latency_s` (the runner logs
  both), not the surrogate, for any efficiency claim.
- Report **mean ± std over ≥3 seeds**.
- The LLM-as-PRM is a stand-in for a trained PRM (TIM-PRM/Athena) — say so;
  optionally swap in a real Math-PRM via `--prm-model` and compare.
- PS-GRPO **post-training** is implemented offline (`python -m acdan.train`); the
  open item is training a **real LLM** policy (backend-agnostic advantage → swap
  `PolicyHead` for an LLM head/LoRA on the VM). The §2 eval results above are
  **test-time** unless you first train and load a policy.

### Training study (PS-GRPO learning curve + ablations)

Offline, CPU, minutes — produces the learning curve and the training-time ablation:

```bash
python -m acdan.train --iters 80 --out results/psgrpo_full.json
python -m acdan.train --iters 80 --no-process            --out results/psgrpo_noproc.json
python -m acdan.train --iters 80 --no-confidence-margin  --out results/psgrpo_nocm.json
python -m acdan.train --iters 80 --no-baseline           --out results/psgrpo_nobaseline.json
```
Report `eval_acc_final` and the `curve` (eval_acc vs iter) per config. To move to a
real LLM policy, implement an `LLMPolicyHead` (LLM action head + LoRA) exposing
`probs(x)` / `logprobs_for` / a gradient step, and feed `PSGRPOTrainer` advantages
into your autograd optimiser — the rollout/advantage/PPO logic is reused as-is.
