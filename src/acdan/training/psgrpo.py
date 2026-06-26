"""PS-GRPO: Process-Supervised Group Relative Policy Optimization + RLCM.

Implements the proposal's post-training objective end-to-end (offline, numpy):

  * **Group Relative** advantage (GRPO): for each prompt, sample a group of G
    rollouts and baseline each reward against the group mean/std — no learned
    critic.
  * **Process supervision**: shape the per-step advantage with the PRM's
    Net-Information-Gain signal, and **localise errors at the drop-moment**
    ``tau_drop`` (the first step where the PRM quality falls below a threshold).
  * **Confidence margin (RLCM)**: adjust the advantage by ``(accuracy - probe
    confidence)`` per step, and co-train the confidence probe on rollout
    outcomes — coupling policy improvement to calibration.
  * **PPO-clipped** surrogate with a **KL** penalty to the reference policy and
    an entropy bonus.

Per-step advantage (see docs/math.md):

  A_{g,h} = Â_g
            + beta_proc * NIG_{g,h}
            - beta_drop * 1[outcome_g = 0 and h >= tau_drop_g]
            + beta_cm   * (outcome_g - conf_{g,h})

with Â_g = (R_g - mean_g R) / (std_g R + eps). All terms are individually
ablatable. Gradients are analytic and finite-difference-tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from acdan.rewards import ProcessRewardModel, net_information_gain
from acdan.training.policy import PolicyHead
from acdan.types import Task
from acdan.verification import ConfidenceProbe

_PROBE_INPUT_DIM = 5  # [latent mean, std, norm, step margin, step PRM]


@dataclass
class PSGRPOConfig:
    # optimisation
    iters: int = 60
    group_size: int = 8
    inner_epochs: int = 2
    lr: float = 0.3
    clip_eps: float = 0.2
    kl_coef: float = 0.02
    entropy_coef: float = 0.01
    # process supervision / RLCM coefficients
    proc_coef: float = 0.5            # NIG weight
    drop_coef: float = 0.5            # drop-moment error penalty
    cm_coef: float = 0.2              # confidence-margin weight
    drop_threshold: float = 0.5       # PRM quality below this marks the drop
    # evaluation / scoring
    success_threshold: float = 0.6
    seed: int = 0
    # ablation switches
    use_group_baseline: bool = True
    use_process: bool = True
    use_confidence_margin: bool = True
    use_kl: bool = True


@dataclass
class _Rollout:
    actions: List[int]
    outcome: float                    # 0/1 correctness
    prm: List[float]
    nig: List[float]
    tau_drop: int
    old_logp: np.ndarray              # (H,) behaviour-policy step logprobs


@dataclass
class TrainHistory:
    iters: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"iters": self.iters}


class PSGRPOTrainer:
    """Trains one :class:`PolicyHead` per task family with PS-GRPO + RLCM."""

    def __init__(self, config: PSGRPOConfig, prm: ProcessRewardModel, reasoner):
        self.cfg = config
        self.prm = prm
        self.reasoner = reasoner  # LatentReasoner (feature_dim must match tasks)
        self.probe = ConfidenceProbe(input_dim=_PROBE_INPUT_DIM, seed=config.seed)
        self.heads: Dict[str, PolicyHead] = {}
        self._ref_probs: Dict[str, np.ndarray] = {}
        self._latent: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------- helpers

    def _family(self, task: Task) -> str:
        return str(task.metadata.get("family", "default"))

    def _latent_of(self, task: Task) -> np.ndarray:
        if task.task_id not in self._latent:
            self._latent[task.task_id] = self.reasoner.reason(task.prompt_features).final_state
        return self._latent[task.task_id]

    def _is_correct(self, task: Task, actions: List[int]) -> float:
        opt = task.optimal_plan
        if not opt:
            return 0.0
        n = min(len(actions), len(opt))
        m = sum(1 for i in range(n) if actions[i] == opt[i % len(opt)])
        return 1.0 if (m / n) >= self.cfg.success_threshold else 0.0

    def _step_features(self, latent: np.ndarray, probs: np.ndarray,
                       actions: List[int], prm: List[float]) -> np.ndarray:
        lat = np.asarray(latent).reshape(-1)
        summ = np.array([lat.mean(), lat.std(),
                         float(np.linalg.norm(lat) / (np.sqrt(lat.size) + 1e-9))])
        out = np.zeros((len(actions), _PROBE_INPUT_DIM))
        for h in range(len(actions)):
            row = np.sort(probs[h])[-2:]
            margin = float(row[-1] - row[0]) if probs.shape[1] >= 2 else float(row[-1])
            out[h] = np.concatenate([summ, [margin, prm[h] if h < len(prm) else 0.0]])
        return out

    # ------------------------------------------------------------ rollouts

    def _rollout_group(self, head: PolicyHead, task: Task, latent: np.ndarray,
                       rng: np.random.Generator) -> Tuple[List[_Rollout], np.ndarray]:
        rollouts: List[_Rollout] = []
        probs = head.probs(task.prompt_features)
        logp = np.log(probs + 1e-12)
        for _ in range(self.cfg.group_size):
            actions = [int(rng.choice(head.V, p=probs[h])) for h in range(head.H)]
            prm = self.prm.score_actions(task, latent, actions)
            nig = net_information_gain(prm)
            tau = next((h for h in range(len(prm)) if prm[h] < self.cfg.drop_threshold),
                       head.H)
            old_lp = np.array([logp[h, actions[h]] for h in range(head.H)])
            rollouts.append(_Rollout(actions, self._is_correct(task, actions),
                                     prm, nig, tau, old_lp))
        return rollouts, probs

    def _advantages(self, head: PolicyHead, rollouts: List[_Rollout],
                    confs: np.ndarray) -> np.ndarray:
        G, H = len(rollouts), head.H
        R = np.array([r.outcome for r in rollouts])
        if self.cfg.use_group_baseline:
            std = R.std()
            ahat = (R - R.mean()) / (std + 1e-6) if std > 1e-8 else R - R.mean()
        else:
            ahat = R - 0.5
        A = np.zeros((G, H))
        for g, r in enumerate(rollouts):
            A[g, :] = ahat[g]
            if self.cfg.use_process:
                A[g, :] += self.cfg.proc_coef * np.asarray(r.nig)
                if r.outcome == 0.0:
                    for h in range(r.tau_drop, H):
                        A[g, h] -= self.cfg.drop_coef
            if self.cfg.use_confidence_margin:
                A[g, :] += self.cfg.cm_coef * (r.outcome - confs[g])
        return (A - A.mean()) / (A.std() + 1e-6)

    # ----------------------------------------------------- objective + grad

    def _objective_and_grad(self, head: PolicyHead, x: np.ndarray,
                            rollouts: List[_Rollout], A: np.ndarray,
                            ref_probs: np.ndarray) -> Tuple[float, np.ndarray]:
        p = head.probs(x)
        logp = np.log(p + 1e-12)
        H, V = head.H, head.V
        eps = self.cfg.clip_eps
        dlogits = np.zeros((H, V))
        obj = 0.0
        G = len(rollouts)
        for g, r in enumerate(rollouts):
            for h in range(H):
                a = r.actions[h]
                ratio = float(np.exp(logp[h, a] - r.old_logp[h]))
                adv = A[g, h]
                unclipped = ratio * adv
                clipped = float(np.clip(ratio, 1 - eps, 1 + eps)) * adv
                if unclipped <= clipped:
                    obj += unclipped
                    coef = adv * ratio
                else:
                    obj += clipped
                    coef = 0.0
                dlogits[h, a] += coef
                dlogits[h, :] -= coef * p[h, :]
        obj /= G
        dlogits /= G
        if self.cfg.use_kl:
            obj -= self.cfg.kl_coef * float(np.sum(ref_probs * (np.log(ref_probs + 1e-12) - logp)))
            dlogits -= self.cfg.kl_coef * (p - ref_probs)
        if self.cfg.entropy_coef > 0:
            ent = -float(np.sum(p * logp))
            obj += self.cfg.entropy_coef * ent
            dlogits += self.cfg.entropy_coef * (-(p * (logp - (p * logp).sum(axis=1, keepdims=True))))
        return obj, dlogits

    # ----------------------------------------------------------------- eval

    def eval_accuracy(self, tasks: List[Task]) -> float:
        if not tasks:
            return 0.0
        acc = []
        for t in tasks:
            head = self.heads[self._family(t)]
            acc.append(self._is_correct(t, head.greedy(t.prompt_features)))
        return float(np.mean(acc))

    # ---------------------------------------------------------------- train

    def _init_heads(self, tasks: List[Task]) -> None:
        for t in tasks:
            fam = self._family(t)
            if fam not in self.heads:
                self.heads[fam] = PolicyHead(t.prompt_features.size, t.horizon,
                                             t.vocab_size, seed=self.cfg.seed)
        # reference policy = initial policy (fixed KL anchor)
        for t in tasks:
            self._ref_probs[t.task_id] = self.heads[self._family(t)].probs(t.prompt_features)

    def train(self, train_tasks: List[Task],
              eval_tasks: Optional[List[Task]] = None) -> TrainHistory:
        eval_tasks = eval_tasks or train_tasks
        self._init_heads(train_tasks + list(eval_tasks))
        rng = np.random.default_rng(self.cfg.seed)
        hist = TrainHistory()

        for it in range(self.cfg.iters):
            probe_X: List[np.ndarray] = []
            probe_y: List[float] = []
            rewards: List[float] = []

            # ---- collect rollouts (behaviour policy = current head) ----
            collected = []  # (task, latent, rollouts, A)
            for t in train_tasks:
                latent = self._latent_of(t)
                head = self.heads[self._family(t)]
                rollouts, probs = self._rollout_group(head, t, latent, rng)
                # per-step confidence from the current probe
                confs = np.full((len(rollouts), head.H), 0.5)
                for g, r in enumerate(rollouts):
                    feats = self._step_features(latent, probs, r.actions, r.prm)
                    if self.probe.is_fitted:
                        confs[g] = [self.probe.predict(feats[h]) for h in range(head.H)]
                    probe_X.extend(feats)
                    probe_y.extend([r.outcome] * head.H)
                    rewards.append(r.outcome)
                A = self._advantages(head, rollouts, confs)
                collected.append((t, rollouts, A))

            # ---- PPO-clipped updates (multiple inner epochs) ----
            for _ in range(self.cfg.inner_epochs):
                for t, rollouts, A in collected:
                    head = self.heads[self._family(t)]
                    _, dlogits = self._objective_and_grad(
                        head, t.prompt_features, rollouts, A, self._ref_probs[t.task_id])
                    head.apply_logit_grad(t.prompt_features, dlogits, self.cfg.lr)

            # ---- co-train the confidence probe (RLCM) ----
            if len(set(probe_y)) >= 2:
                self.probe.fit(np.array(probe_X), np.array(probe_y), epochs=60, lr=0.1)

            hist.iters.append({
                "iter": it,
                "train_reward": float(np.mean(rewards)) if rewards else 0.0,
                "eval_acc": self.eval_accuracy(eval_tasks),
            })
        return hist
