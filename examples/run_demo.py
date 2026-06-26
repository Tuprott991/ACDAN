"""Minimal end-to-end ACDAN demo (no CLI, pure Python).

Run:  python examples/run_demo.py
Requires only numpy + PyYAML. Downloads nothing.
"""

from __future__ import annotations

from acdan.config import ACDANConfig
from acdan.evaluate import build_agent
from acdan.tasks.synthetic import make_suite


def main() -> None:
    config = ACDANConfig(name="demo", seed=0)
    agent = build_agent(config, n_per_family=8)

    # Run the first task of each family and print a compact trace.
    tasks = make_suite(n_per_family=8, seed=config.seed)
    seen = set()
    for task in tasks:
        fam = task.metadata.get("family")
        if fam in seen:
            continue
        seen.add(fam)

        result = agent.run_task(task)
        m, v = result.metrics, result.verification
        print(f"\n[{fam}] {task.task_id}  difficulty={task.difficulty:.2f}")
        print("  optimal :", [task.vocab[a] for a in (task.optimal_plan or [])])
        print("  planned :", [s.action_name for s in result.steps])
        print("  inertia :", [s.from_inertia for s in result.steps])
        print(f"  correct={m.correct}  conf={v.confidence:.3f}  "
              f"token_cost={m.token_cost:.2f}  llm_calls={m.llm_calls}  "
              f"dead_pruned={m.dead_steps_pruned}  dep_H={m.dependency_entropy:.3f}")


if __name__ == "__main__":
    main()
