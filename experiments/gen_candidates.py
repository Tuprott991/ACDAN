"""Pre-generate K candidate answers per answer-selection question.

This is the only expensive GPU step for the math benchmarks: it samples K full
chain-of-thought solutions per question with the policy, extracts each final
answer, and writes the candidates JSONL that ``acdan.datasets.gsm8k`` consumes.
DTO + PRM then *select* among these candidates (the H=1, V=K framing).

Usage (on the VM):
    python experiments/gen_candidates.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --in data/gsm8k_test.jsonl \
        --out data/gsm8k_qwen7b_k8.jsonl --k 8

Input JSONL lines: {"question": "...", "answer": "42"}.
The question field may also be named ``problem`` or ``prompt``.
Output JSONL lines: {"question": ..., "candidates": ["...","..."], "answer": "42"}

The output also preserves lightweight evidence for the selector:
``candidate_solutions`` keeps the first sampled solution for each unique final
answer, ``candidate_counts`` keeps self-consistency counts, and
``candidate_first_indices`` keeps generation order. The dataset adapter remains
backward-compatible with older files that only have ``candidates``.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import random

from acdan.datasets.math_answer import extract_final_answer


def extract_answer(text: str) -> str:
    return extract_final_answer(text)


def _question(row: dict) -> str:
    for key in ("question", "problem", "prompt"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    raise KeyError("input row must contain one of: question, problem, prompt")


def _answer(row: dict) -> str:
    for key in ("answer", "gold", "final_answer"):
        value = row.get(key)
        if value is not None:
            return str(value)
    raise KeyError("input row must contain one of: answer, gold, final_answer")


def _prompt(question: str, task_kind: str) -> str:
    if task_kind == "qa":
        return (
            "Answer the question. Reason briefly if useful, then end with "
            "'The answer is <answer>'.\n\n"
            f"{question}"
        )
    return (
        "Solve the problem step by step and end with 'The answer is <number>'.\n\n"
        f"{question}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--task-kind",
        choices=["math", "qa"],
        default="math",
        help="Prompt style for candidate generation. Use 'qa' for BrowseComp-style answers.",
    )
    ap.add_argument(
        "--order",
        choices=["sample", "plurality", "shuffle"],
        default="sample",
        help=(
            "Candidate order. 'sample' preserves first-seen order; 'plurality' "
            "sorts by self-consistency count; 'shuffle' is a seeded neutral order."
        ),
    )
    ap.add_argument("--shuffle-seed", type=int, default=0)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams  # lazy / GPU

    rows = []
    with open(args.inp, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if args.limit and len(rows) >= args.limit:
                break

    llm = LLM(model=args.model)
    sp = SamplingParams(n=args.k, temperature=args.temperature, max_tokens=args.max_tokens)
    prompts = [_prompt(_question(r), args.task_kind) for r in rows]
    outs = llm.generate(prompts, sp)

    with open(args.out, "w", encoding="utf-8") as fh:
        for row_idx, (r, out) in enumerate(zip(rows, outs)):
            grouped = OrderedDict()
            sample_answers = []
            for idx, sample in enumerate(out.outputs):
                solution = sample.text.strip()
                answer = extract_answer(solution)
                sample_answers.append(answer)
                if answer not in grouped:
                    grouped[answer] = {
                        "solution": solution,
                        "count": 0,
                        "first_index": idx,
                    }
                grouped[answer]["count"] += 1

            ordered = sorted(grouped.items(), key=lambda item: item[1]["first_index"])
            if args.order == "plurality":
                ordered = sorted(
                    grouped.items(),
                    key=lambda item: (-item[1]["count"], item[1]["first_index"]),
                )
            elif args.order == "shuffle":
                ordered = list(ordered)
                rng = random.Random(args.shuffle_seed + row_idx)
                rng.shuffle(ordered)
            cands = [answer for answer, _ in ordered]
            fh.write(json.dumps({
                "task_id": r.get("task_id", f"candidate-{row_idx:05d}"),
                "question": _question(r),
                "candidates": cands,
                "candidate_order": args.order,
                "candidate_solutions": {
                    answer: meta["solution"] for answer, meta in ordered
                },
                "candidate_counts": {
                    answer: meta["count"] for answer, meta in ordered
                },
                "candidate_first_indices": {
                    answer: meta["first_index"] for answer, meta in ordered
                },
                "candidate_sample_answers": sample_answers,
                "answer": _answer(r),
            }) + "\n")
    print(f"wrote {len(rows)} tasks -> {args.out}")


if __name__ == "__main__":
    main()
