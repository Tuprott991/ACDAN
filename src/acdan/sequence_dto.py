"""Autoregressive lattice DTO for prefix-conditioned tool trajectories.

The frozen core model and PRM are black boxes. They score a bounded set of
discrete prefixes once; DTO then optimizes logit offsets over the resulting
prefix trie with exact dynamic programming. No gradient is taken through the
LLM, and optimization iterations do not issue model calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from acdan.config import SequenceDTOConfig
from acdan.types import Plan, Task, stable_seed


STOP_ACTION = -1
Prefix = Tuple[int, ...]


def _softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(values, dtype=np.float64) / max(float(temperature), 1e-6)
    z = z - float(np.max(z))
    exp = np.exp(z)
    return exp / max(float(exp.sum()), 1e-12)


@dataclass
class LatticeEdge:
    action: int
    child: Prefix | None
    base_logit: float = 0.0
    reward: float = 0.0


@dataclass
class LatticeNode:
    prefix: Prefix
    edges: List[LatticeEdge] = field(default_factory=list)
    offsets: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    probs: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    prior_probs: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    q_values: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    value: float = 0.0
    occupancy: float = 0.0

    @property
    def base_logits(self) -> np.ndarray:
        return np.asarray([edge.base_logit for edge in self.edges], dtype=np.float64)


@dataclass
class TrajectoryLattice:
    nodes: Dict[Prefix, LatticeNode]
    candidates: Tuple[Prefix, ...]
    trajectory_rewards: Dict[Prefix, float]
    sample_counts: Dict[Prefix, int]

    @property
    def evaluated_edges(self) -> int:
        return sum(len(node.edges) for node in self.nodes.values())


class AutoregressiveLatticeOptimizer:
    """Optimize a bounded prefix trie using analytic expected-return gradients."""

    def __init__(self, config: SequenceDTOConfig, core, prm, seed: int = 0):
        self.cfg = config
        self.core = core
        self.prm = prm
        self.seed = int(seed)
        self._prior_cache: Dict[Prefix, np.ndarray] = {}
        self._reward_cache: Dict[Prefix, np.ndarray] = {}
        self._prior_backend_batches = 0
        self._reward_backend_batches = 0

    # --------------------------------------------------------------- backends

    def _conditional_prior(self, task: Task, latent: np.ndarray, prefix: Prefix) -> np.ndarray:
        cached = self._prior_cache.get(prefix)
        if cached is not None:
            return cached.copy()
        self._prefetch(task, latent, [prefix], rewards=False)
        return self._prior_cache[prefix].copy()

    def _extension_rewards(self, task: Task, latent: np.ndarray, prefix: Prefix) -> np.ndarray:
        cached = self._reward_cache.get(prefix)
        if cached is not None:
            return cached.copy()
        self._prefetch(task, latent, [prefix], priors=False)
        return self._reward_cache[prefix].copy()

    def _prefetch(
        self,
        task: Task,
        latent: np.ndarray,
        prefixes: Sequence[Prefix],
        *,
        priors: bool = True,
        rewards: bool = True,
    ) -> None:
        unique = list(dict.fromkeys(tuple(prefix) for prefix in prefixes))
        if priors:
            missing = [prefix for prefix in unique if prefix not in self._prior_cache]
            if missing:
                batch = getattr(self.core, "conditional_prior_logits_batch", None)
                if callable(batch):
                    rows = np.asarray(
                        batch(task, latent, missing, include_stop=True), dtype=np.float64
                    )
                    backend_batches = 1
                else:
                    method = getattr(self.core, "conditional_prior_logits", None)
                    if callable(method):
                        rows = np.asarray(
                            [method(task, latent, prefix, include_stop=True) for prefix in missing],
                            dtype=np.float64,
                        )
                        backend_batches = len(missing)
                    else:
                        fixed = np.asarray(self.core.prior_logits(task, latent), dtype=np.float64)
                        rows = []
                        for prefix in missing:
                            row = fixed[min(len(prefix), fixed.shape[0] - 1)]
                            rows.append(np.concatenate([row, [self.cfg.stop_logit]]))
                        rows = np.asarray(rows, dtype=np.float64)
                        backend_batches = 1
                self._prior_backend_batches += backend_batches
                rows = np.atleast_2d(rows)
                for prefix, row in zip(missing, rows):
                    row = np.asarray(row, dtype=np.float64).reshape(-1)
                    if row.size == task.vocab_size:
                        row = np.concatenate([row, [self.cfg.stop_logit]])
                    if row.size != task.vocab_size + 1:
                        raise ValueError(
                            "conditional prior must return V or V+1 logits; "
                            f"got {row.size} for V={task.vocab_size}"
                        )
                    self._prior_cache[prefix] = row.copy()
        if rewards:
            missing = [prefix for prefix in unique if prefix not in self._reward_cache]
            if missing:
                batch = getattr(self.prm, "extension_rewards_batch", None)
                if callable(batch):
                    rows = np.asarray(batch(task, latent, missing), dtype=np.float64)
                    backend_batches = 1
                else:
                    method = getattr(self.prm, "extension_rewards", None)
                    if callable(method):
                        rows = np.asarray(
                            [method(task, latent, prefix) for prefix in missing],
                            dtype=np.float64,
                        )
                        backend_batches = len(missing)
                    else:
                        matrix = np.asarray(
                            self.prm.step_reward_matrix(task, latent), dtype=np.float64
                        )
                        rows = np.asarray([
                            1.0 / (1.0 + np.exp(-matrix[min(len(prefix), matrix.shape[0] - 1)]))
                            for prefix in missing
                        ])
                        backend_batches = 1
                self._reward_backend_batches += backend_batches
                rows = np.atleast_2d(rows)
                for prefix, row in zip(missing, rows):
                    row = np.asarray(row, dtype=np.float64).reshape(-1)
                    if row.size != task.vocab_size:
                        raise ValueError(
                            f"extension reward must return V={task.vocab_size} values, got {row.size}"
                        )
                    self._reward_cache[prefix] = np.clip(row, 0.0, 1.0)

    def _trajectory_rewards(
        self,
        task: Task,
        latent: np.ndarray,
        trajectories: Sequence[Prefix],
    ) -> Dict[Prefix, float]:
        method = getattr(self.prm, "trajectory_rewards", None)
        if callable(method):
            scores = list(method(task, latent, trajectories))
        else:
            scores = []
            for path in trajectories:
                step_scores = []
                for depth, action in enumerate(path):
                    step_scores.append(
                        self._extension_rewards(task, latent, path[:depth])[action]
                    )
                scores.append(float(np.mean(step_scores)) if step_scores else 0.0)
        if len(scores) != len(trajectories):
            raise ValueError("trajectory reward backend returned the wrong number of scores")
        return {
            tuple(path): float(np.clip(score, 0.0, 1.0))
            for path, score in zip(trajectories, scores)
        }

    # ------------------------------------------------------------ candidates

    def _proposal_scores(self, task: Task, latent: np.ndarray, prefix: Prefix) -> np.ndarray:
        prior = self._conditional_prior(task, latent, prefix)
        rewards = self._extension_rewards(task, latent, prefix)
        scores = prior.copy()
        scores[:task.vocab_size] += self.cfg.step_reward_weight * rewards
        if len(prefix) < self.cfg.min_steps:
            scores[-1] = -np.inf
        return scores

    def _greedy_completions(
        self, task: Task, latent: np.ndarray, max_steps: int
    ) -> Iterable[Prefix]:
        """Cover every root tool, then greedily complete each branch."""
        active = [(root_action,) for root_action in range(task.vocab_size)]
        for prefix in active:
            yield prefix
        while active and len(active[0]) < max_steps:
            self._prefetch(task, latent, active)
            next_active: List[Prefix] = []
            for prefix in active:
                scores = self._proposal_scores(task, latent, prefix)
                action = int(np.argmax(scores))
                if action == task.vocab_size:
                    continue
                child = prefix + (action,)
                yield child
                next_active.append(child)
            active = next_active

    def _beam_candidates(
        self, task: Task, latent: np.ndarray, max_steps: int
    ) -> Iterable[Prefix]:
        frontier: List[Tuple[Prefix, float]] = [((), 0.0)]
        width = max(1, int(self.cfg.beam_width))
        for _ in range(max_steps):
            self._prefetch(task, latent, [prefix for prefix, _ in frontier])
            expanded: List[Tuple[Prefix, float]] = []
            for prefix, cumulative in frontier:
                scores = self._proposal_scores(task, latent, prefix)
                log_probs = np.log(_softmax(scores) + 1e-12)
                if len(prefix) >= self.cfg.min_steps:
                    yield prefix
                for action in range(task.vocab_size):
                    child = prefix + (action,)
                    expanded.append((child, cumulative + float(log_probs[action])))
            expanded.sort(key=lambda item: (-item[1], item[0]))
            frontier = []
            seen: set[Prefix] = set()
            for prefix, score in expanded:
                if prefix in seen:
                    continue
                seen.add(prefix)
                yield prefix
                if len(prefix) < max_steps and len(frontier) < width:
                    frontier.append((prefix, score))
                if len(frontier) >= width and len(seen) >= width * 2:
                    break
            if not frontier:
                break

    def _sample_candidates(
        self, task: Task, latent: np.ndarray, max_steps: int
    ) -> Tuple[List[Prefix], Dict[Prefix, int]]:
        rng = np.random.default_rng(stable_seed(task.task_id, "sequence-dto", self.seed))
        samples: List[Prefix] = []
        counts: Dict[Prefix, int] = {}
        active: List[Prefix] = [()] * max(0, int(self.cfg.samples))
        finished: List[Prefix] = []
        for _depth in range(max_steps):
            if not active:
                break
            self._prefetch(task, latent, active)
            next_active = []
            for prefix in active:
                # Consensus is independent policy evidence; do not feed the PRM
                # back into the trajectories whose frequency becomes R_SC.
                scores = self._conditional_prior(task, latent, prefix)
                if len(prefix) < self.cfg.min_steps:
                    scores[-1] = -np.inf
                action = int(rng.choice(task.vocab_size + 1, p=_softmax(scores)))
                if action == task.vocab_size:
                    finished.append(prefix)
                else:
                    next_active.append(prefix + (action,))
            active = next_active
        finished.extend(active)
        for prefix in finished:
            if not prefix:
                prefix = (int(np.argmax(self._conditional_prior(task, latent, ())[:-1])),)
            samples.append(prefix)
            counts[prefix] = counts.get(prefix, 0) + 1
        return samples, counts

    def build_lattice(self, task: Task, latent: np.ndarray) -> TrajectoryLattice:
        self._prior_cache.clear()
        self._reward_cache.clear()
        self._prior_backend_batches = 0
        self._reward_backend_batches = 0
        max_steps = max(self.cfg.min_steps, int(self.cfg.max_steps))
        candidates = set(self._greedy_completions(task, latent, max_steps))
        candidates.update(self._beam_candidates(task, latent, max_steps))
        samples, sample_counts = self._sample_candidates(task, latent, max_steps)
        candidates.update(samples)
        candidates = {path for path in candidates if path and len(path) <= max_steps}
        ordered = tuple(sorted(candidates, key=lambda path: (len(path), path)))
        trajectory_rewards = self._trajectory_rewards(task, latent, ordered)

        nodes: Dict[Prefix, LatticeNode] = {(): LatticeNode(())}
        for path in ordered:
            prefix: Prefix = ()
            for action in path:
                child = prefix + (int(action),)
                node = nodes.setdefault(prefix, LatticeNode(prefix))
                nodes.setdefault(child, LatticeNode(child))
                if not any(edge.action == action for edge in node.edges):
                    node.edges.append(LatticeEdge(int(action), child))
                prefix = child
            node = nodes.setdefault(prefix, LatticeNode(prefix))
            if not any(edge.action == STOP_ACTION for edge in node.edges):
                node.edges.append(LatticeEdge(STOP_ACTION, None))

        for prefix, node in nodes.items():
            if not node.edges:
                continue
            prior = self._conditional_prior(task, latent, prefix)
            has_tool_edge = any(edge.action != STOP_ACTION for edge in node.edges)
            extension = (
                self._extension_rewards(task, latent, prefix)
                if has_tool_edge else np.zeros(task.vocab_size, dtype=np.float64)
            )
            for edge in node.edges:
                if edge.action == STOP_ACTION:
                    edge.base_logit = float(prior[-1])
                    consensus = np.log1p(float(sample_counts.get(prefix, 0)))
                    edge.reward = (
                        self.cfg.trajectory_reward_weight * trajectory_rewards[prefix]
                        + self.cfg.self_consistency_weight * consensus
                    )
                else:
                    edge.base_logit = float(prior[edge.action])
                    edge.reward = (
                        self.cfg.step_reward_weight
                        * float(extension[edge.action])
                        / max(1, max_steps)
                        - self.cfg.length_cost
                    )
            node.edges.sort(key=lambda edge: edge.action)
            node.offsets = np.zeros(len(node.edges), dtype=np.float64)

        return TrajectoryLattice(nodes, ordered, trajectory_rewards, sample_counts)

    # ----------------------------------------------------------- optimization

    def _evaluate_lattice(
        self, lattice: TrajectoryLattice
    ) -> Tuple[float, Dict[Prefix, np.ndarray]]:
        cfg = self.cfg
        nodes = lattice.nodes
        ordered = sorted(nodes, key=lambda prefix: (-len(prefix), prefix))
        for prefix in ordered:
            node = nodes[prefix]
            if not node.edges:
                node.value = 0.0
                continue
            base = node.base_logits
            node.prior_probs = _softmax(base, cfg.temperature)
            node.probs = _softmax(base + node.offsets, cfg.temperature)
            q_values = []
            for edge in node.edges:
                child_value = nodes[edge.child].value if edge.child is not None else 0.0
                q_values.append(edge.reward + child_value)
            node.q_values = np.asarray(q_values, dtype=np.float64)
            log_ratio = np.log(node.probs + 1e-12) - np.log(node.prior_probs + 1e-12)
            kl = float(np.sum(node.probs * log_ratio))
            node.value = float(np.dot(node.probs, node.q_values) - cfg.kl_weight * kl)

        for node in nodes.values():
            node.occupancy = 0.0
        nodes[()].occupancy = 1.0
        for prefix in sorted(nodes, key=lambda item: (len(item), item)):
            node = nodes[prefix]
            for probability, edge in zip(node.probs, node.edges):
                if edge.child is not None:
                    nodes[edge.child].occupancy += node.occupancy * float(probability)

        gradients: Dict[Prefix, np.ndarray] = {}
        temperature = max(float(cfg.temperature), 1e-6)
        for prefix, node in nodes.items():
            if not node.edges:
                continue
            log_ratio = np.log(node.probs + 1e-12) - np.log(node.prior_probs + 1e-12)
            adjusted_q = node.q_values - cfg.kl_weight * (log_ratio + 1.0)
            baseline = float(np.dot(node.probs, adjusted_q))
            gradients[prefix] = (
                node.occupancy
                * node.probs
                * (adjusted_q - baseline)
                / temperature
            )
        return nodes[()].value, gradients

    def _decode(
        self, task: Task, lattice: TrajectoryLattice, use_dto: bool
    ) -> Tuple[List[int], np.ndarray]:
        actions: List[int] = []
        rows: List[np.ndarray] = []
        prefix: Prefix = ()
        while prefix in lattice.nodes and lattice.nodes[prefix].edges:
            node = lattice.nodes[prefix]
            if use_dto:
                decision_values = (
                    node.q_values
                    + self.cfg.kl_weight * np.log(node.prior_probs + 1e-12)
                )
            else:
                decision_values = np.log(node.prior_probs + 1e-12)
            edge_index = int(np.argmax(decision_values))
            edge = node.edges[edge_index]
            if edge.action == STOP_ACTION:
                break
            row = np.full(task.vocab_size, -1e9, dtype=np.float64)
            for idx, candidate in enumerate(node.edges):
                if candidate.action != STOP_ACTION:
                    row[candidate.action] = decision_values[idx]
            rows.append(row)
            actions.append(edge.action)
            if edge.child is None:
                break
            prefix = edge.child
        logits = np.stack(rows) if rows else np.zeros((0, task.vocab_size), dtype=np.float64)
        return actions, logits

    def optimize(self, task: Task, latent: np.ndarray, use_dto: bool = True) -> Plan:
        lattice = self.build_lattice(task, latent)
        trace: List[float] = []
        steps = max(0, int(self.cfg.iters)) if use_dto else 0
        for _ in range(steps):
            objective, gradients = self._evaluate_lattice(lattice)
            trace.append(float(objective))
            for prefix, gradient in gradients.items():
                node = lattice.nodes[prefix]
                node.offsets = np.clip(
                    node.offsets + self.cfg.lr * gradient,
                    -20.0,
                    20.0,
                )
        objective, _ = self._evaluate_lattice(lattice)
        actions, logits = self._decode(task, lattice, use_dto=use_dto)
        return Plan(
            actions=actions,
            logits=logits,
            dto_steps=steps,
            objective_trace=trace,
            metadata={
                "planner": "autoregressive_lattice_dto",
                "objective": float(objective),
                "expanded_nodes": len(lattice.nodes),
                "evaluated_edges": lattice.evaluated_edges,
                "trajectory_candidates": len(lattice.candidates),
                "self_consistency_samples": int(self.cfg.samples),
                "policy_score_batches": self._prior_backend_batches,
                "prm_score_batches": self._reward_backend_batches + 1,
                "stopped": len(actions) < int(self.cfg.max_steps),
            },
        )
