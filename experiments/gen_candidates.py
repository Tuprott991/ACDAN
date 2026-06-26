"""Pre-generate K candidate answers per GSM8K/MATH question (preprocessing).

This is the only expensive GPU step for the math benchmarks: it samples K full
chain-of-thought solutions per question with the policy, extracts each final
answer, and writes the candidates JSONL that ``acdan.datasets.gsm8k`` consumes.
DTO + PRM then *select* among these candidates (the H=1, V=K framing).

Usage (on the VM):
    python experiments/gen_candidates.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --in data/gsm8k_test.jsonl \
        --out data/gsm8k_qwen7b_k8.jsonl --k 8

Input JSONL lines: {"question": "...", "answer": "42"}
Output JSONL lines: {"question": ..., "candidates": ["...","..."], "answer": "42"}
"""

from __future__ import annotations

import argparse
import json
import re


def extract_answer(text: str) -> str:
    m = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return m[-1] if m else text.strip()[-32:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
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
    prompts = [
        f"Solve the problem step by step and end with 'The answer is <number>'.\n\n{r['question']}"
        for r in rows
    ]
    outs = llm.generate(prompts, sp)

    with open(args.out, "w", encoding="utf-8") as fh:
        for r, out in zip(rows, outs):
            cands = sorted({extract_answer(o.text) for o in out.outputs})
            fh.write(json.dumps({
                "question": r["question"],
                "candidates": cands,
                "answer": str(r["answer"]),
            }) + "\n")
    print(f"wrote {len(rows)} tasks -> {args.out}")


if __name__ == "__main__":
    main()
