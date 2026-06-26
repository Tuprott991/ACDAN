"""PS-GRPO: gradient correctness, learning, ablations, determinism."""

import numpy as np

from acdan.config import LatentConfig
from acdan.latent_reasoning import LatentReasoner
from acdan.registry import build_prm
from acdan.training.policy import PolicyHead
from acdan.training.psgrpo import PSGRPOConfig, PSGRPOTrainer, _Rollout
from acdan.training.tasks import FEATURE_DIM, make_learnable_suite


def _trainer(**kw):
    cfg = PSGRPOConfig(iters=kw.pop("iters", 30), group_size=kw.pop("group_size", 6), **kw)
    prm = build_prm("mock", seed=0)
    reasoner = LatentReasoner(LatentConfig(), feature_dim=FEATURE_DIM, seed=0)
    return PSGRPOTrainer(cfg, prm, reasoner)


def test_policy_head_shapes_and_params():
    head = PolicyHead(8, 3, 4, seed=0)
    x = np.ones(8)
    assert head.probs(x).shape == (3, 4)
    np.testing.assert_allclose(head.probs(x).sum(axis=1), 1.0, atol=1e-9)
    assert len(head.greedy(x)) == 3
    W, b = head.get_params()
    head.set_params((W + 1.0, b + 1.0))
    assert not np.allclose(head.get_params()[0], W)


def test_objective_gradient_matches_finite_difference():
    t = _trainer(iters=1)
    head = PolicyHead(6, 3, 4, seed=1)
    rng = np.random.default_rng(0)
    x = rng.normal(size=6)
    H, V, G = head.H, head.V, 4
    rollouts = [
        _Rollout(
            actions=[int(rng.integers(0, V)) for _ in range(H)],
            outcome=float(rng.integers(0, 2)),
            prm=list(rng.uniform(0, 1, size=H)),
            nig=list(rng.uniform(-0.2, 0.2, size=H)),
            tau_drop=int(rng.integers(0, H + 1)),
            old_logp=rng.normal(size=H) - 1.5,
        )
        for _ in range(G)
    ]
    A = rng.normal(size=(G, H))
    ref = head.probs(x)  # fixed reference

    obj, dlogits = t._objective_and_grad(head, x, rollouts, A, ref)

    eps = 1e-6
    W0 = head.W.copy()
    for (j, d) in [(0, 0), (5, 2), (H * V - 1, 5)]:
        pert = W0.copy()
        pert[j, d] += eps
        head.W = pert
        o2, _ = t._objective_and_grad(head, x, rollouts, A, ref)
        head.W = W0
        fd = (o2 - obj) / eps
        analytic = dlogits.reshape(-1)[j] * x[d]
        assert np.isclose(fd, analytic, rtol=1e-3, atol=1e-4)


def test_advantage_shape_and_finite():
    t = _trainer(iters=1, group_size=5)
    tasks = make_learnable_suite(n_per_family=1, seed=0)
    task = tasks[0]
    t._init_heads(tasks)
    head = t.heads[t._family(task)]
    latent = t._latent_of(task)
    rollouts, _ = t._rollout_group(head, task, latent, np.random.default_rng(0))
    A = t._advantages(head, rollouts, np.full((len(rollouts), head.H), 0.5))
    assert A.shape == (len(rollouts), head.H)
    assert np.all(np.isfinite(A))


def test_psgrpo_learns():
    t = _trainer(iters=40, seed=0)
    train = make_learnable_suite(n_per_family=12, seed=0)
    ev = make_learnable_suite(n_per_family=6, seed=1)
    hist = t.train(train, ev)
    assert hist.iters[-1]["train_reward"] > hist.iters[0]["train_reward"] + 0.2
    assert hist.iters[-1]["eval_acc"] > hist.iters[0]["eval_acc"]


def test_process_supervision_helps():
    train = make_learnable_suite(n_per_family=12, seed=0)
    full = _trainer(iters=40, seed=0).train(train, train).iters[-1]["train_reward"]
    nop = _trainer(iters=40, seed=0, use_process=False).train(train, train).iters[-1]["train_reward"]
    assert full >= nop - 1e-6


def test_training_is_deterministic():
    train = make_learnable_suite(n_per_family=8, seed=0)
    a = _trainer(iters=20, seed=0).train(train, train).to_dict()
    b = _trainer(iters=20, seed=0).train(train, train).to_dict()
    assert a == b
