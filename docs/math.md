# ACDAN Mathematical Model

This document states the objective and update rules implemented in the code, and
notes where the offline implementation uses a tractable surrogate. Symbols follow
the proposal.

## Notation

- Planning step horizon `H`; action vocabulary `V`.
- Soft action logits `L ∈ R^{H×V}`; per-step distribution `P_h = softmax(L_h / τ)`.
- Latent state `h` from the recurrent reasoning block.
- `R_prm(P)` — process reward of a soft plan from the PRM.
- `A_dep` — adjacency / representation of the dependency layer (ED).

## 1. Test-time objective (DTO)

ACDAN minimises, at test time, over the soft logits `L`:

```
J(L) = − w_ll · loglik(P)                  (core-model log-likelihood prior)
       − α   · R_prm(P)                     (PRM process reward)
       + β   · len_pen(P)                   (length / repetition penalty)
       − γ   · H_vN(A_dep(P))               (dependency von Neumann entropy)
```

with `P = softmax(L/τ)`. Coefficients `(w_ll, α, β, γ)` are the adaptive
coefficients; in code `α` and `γ` are scaled by a per-task `difficulty_scale`
(proposal: *hệ số điều phối thích ứng tùy độ khó*). See `DTOConfig` and
`agent._plan`.

Term-by-term implementation (`dto.py`):

- `loglik(P) = (1/H) Σ_h Σ_v P_{h,v} log p0_{h,v}` where `p0 = softmax(L0)` is the
  fixed core-model prior. (Linear in `P`.)
- `R_prm(P) = (1/H) Σ_h ⟨P_h, t_h⟩`, with `t_h` a per-step softmax-sharpened
  reward target from the PRM (`MockProcessReward._reward_target`). Gradient is the
  constant `t/H` — the PRM "back-propagates" into the logits.
- `len_pen(P) = (1/(H−1)) Σ_h ⟨P_h, P_{h−1}⟩` penalises consecutive-step overlap
  (repetition / overthinking).
- `H_vN` — see §3.

## 2. First-order update

```
L^{(k+1)} = L^{(k)} − η · ∂J/∂L
```

for `k = 0 … T−1`, then decode `a*_h = argmax_v softmax(L^{(T)}/τ)_{h,v}`.

The chain rule from `∂J/∂P` to `∂J/∂L` uses the exact per-row softmax
vector–Jacobian product

```
∂P_h/∂L_h applied to g  =  (1/τ) · P_h ⊙ (g − ⟨P_h, g⟩)
```

(`dto._softmax_jacobian_vjp`). This is verified against finite differences in
`tests/test_dto.py::test_gradient_matches_finite_difference`.

## 3. Dependency von Neumann entropy

**Reported (exact).** With step representations `S ∈ R^{H×d}`, form the PSD Gram
matrix `G = S Sᵀ` and density matrix `ρ = G / tr(G)` (eigenvalues ≥ 0, sum 1).
Then

```
H_vN(ρ) = − Σ_i λ_i log λ_i ,   λ_i = eig(ρ).
```

Implemented in `AgenticComputationGraph.von_neumann_entropy`.

**Optimised (surrogate).** The exact `H_vN` requires an eigendecomposition whose
gradient we avoid for stability and zero extra dependencies. The DTO loop instead
maximises a smooth, analytically-differentiable proxy — the representation
variance of the soft step embeddings:

```
H_surr(P) = (1/H) Σ_h ‖ S_h − S̄ ‖²,   S = P·E,  S̄ = mean_h S_h
```

(`graph.make_entropy_hook`). Both `H_vN` and `H_surr` are minimised iff all steps
collapse to one representation and increase as steps diversify, so optimising the
surrogate is consistent with reporting the exact quantity. **We report the exact
entropy and optimise the surrogate — stated explicitly so it can be defended in
review.**

## 4. Net Information Gain (process supervision, O(N))

For a chain with per-step PRM scores `s_1 … s_N`, define an accumulating belief in
the correct answer and take its increments:

```
b_i   = σ( κ · ( mean(s_1..s_i) − ½ ) ),     b_0 = ½
NIG_i = b_i − b_{i−1}
```

A single linear pass (O(N)) versus O(N²) classical Monte-Carlo rollout labelling
(`rewards.net_information_gain`). Steps with `NIG_i ≤ 0` are dead-step candidates.

## 5. Dead-step pruning (DECS)

Step `i` is pruned iff it is **both** (a) disconnected from the final answer in
the dependency layer (no ED path `i → H−1`) **and** (b) unhelpful (`NIG_i ≤ 0`).
The final step is never pruned. This conservative rule removes redundant branches
without dropping any belief-increasing step (`AgenticComputationGraph.dead_steps`).

## 6. Inertial sensing

A first-order Markov model `T(s'|s)` is fit from execution traces with Laplace
smoothing. Inertia fires for state `s` iff `n_obs(s) ≥ m` and `max_{s'} T(s'|s) ≥
θ`; the action is then taken without an LLM planning call (`inertia.py`).

## 7. RLCM confidence & calibration

The probe is a 2-layer MLP `c = σ(W₂ tanh(W₁ x + b₁) + b₂)` over features
`x = [latent mean, std, norm, plan margin, mean PRM]`, trained by binary
cross-entropy. The reported confidence blends the probe with the independent
verifier:

```
conf = (1 − w) · c_probe + w · agree_independent
```

The **margin** is the mean top1−top2 of the per-step action distributions.
**Calibration** is summarised by ECE with equal-width bins
(`verification.expected_calibration_error`).

## 8. PS-GRPO post-training (implemented)

The proposal's post-training objective — Process-Supervised Group Relative Policy
Optimization with confidence margin — is implemented as a complete, offline,
analytic-gradient trainer over the parametric `(H, V)` policy
`π_θ(a|x) = softmax((W x + b))` (`acdan.training`).

For a prompt `x`, sample a group of `G` rollouts `{a^g}` from `π_θ`. Let
`R_g ∈ {0,1}` be the outcome, `s^g_h` the PRM step score, `NIG^g_h` its Net
Information Gain (§4), `τ^g` the **drop-moment** (first `h` with `s^g_h < θ_drop`),
and `c^g_h` the RLCM probe confidence. The per-step advantage is

```
Â_g    = (R_g − mean_g R) / (std_g R + ε)                 (group-relative)
A_{g,h} = Â_g
          + β_proc · NIG^g_h                              (process supervision)
          − β_drop · 1[R_g = 0 ∧ h ≥ τ^g]                 (drop-moment blame)
          + β_cm   · (R_g − c^g_h)                        (confidence margin / RLCM)
```

normalised to zero mean / unit std. The policy is updated with the **PPO-clipped**
surrogate plus a **KL** penalty to the reference (initial) policy and an entropy
bonus:

```
J(θ) = mean_{g,h} min( ρ_{g,h} A_{g,h}, clip(ρ_{g,h}, 1±ε) A_{g,h} )
       − β_KL · KL(π_ref ‖ π_θ) + β_ent · H(π_θ),   ρ_{g,h} = π_θ(a^g_h)/π_old(a^g_h)
```

All gradients are analytic (score-function for the surrogate; `p_θ − p_ref` for
KL; `−p⊙(log p − ⟨log p⟩)` for entropy) and finite-difference-tested
(`tests/test_psgrpo.py`). The confidence probe is **co-trained** each iteration on
rollout outcomes (RLCM), so policy improvement and calibration are coupled. Every
term is ablatable (`PSGRPOConfig.use_process / use_confidence_margin /
use_group_baseline / use_kl`).

> **Scope.** The offline trainer optimises the parametric `(H, V)` policy on the
> learnable synthetic suite (features encode the optimal plan, so there is a real
> learning signal). The advantage computation is backend-agnostic: to train a real
> LLM policy on the VM, replace `PolicyHead` with an LLM action head / LoRA and
> feed the same advantages into an autograd optimiser — nothing else changes.
