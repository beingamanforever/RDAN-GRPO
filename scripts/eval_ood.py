#!/usr/bin/env python3
"""Score checkpoints on the out-of-domain reasoning benchmarks.

RDAN trains only on instruction following, so these measure whether that training cost the
model reasoning ability it already had. A flat score here is the good outcome; the in-domain
benchmarks are where improvement is expected.

Prompt templates follow RTT's appendix so the numbers sit beside theirs. MATH-500 answers are
compared with math_verify rather than string equality, since the same value has many valid
renderings.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MATH_PROMPT = "Question: {question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
CHOICE_PROMPT = (
    "Question: {question}\nAnswer the multiple choice question. The last line of your response "
    "should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one "
    "of choices. Think step by step before answering."
)
ANSWER_PATTERN = re.compile(r"Answer:\s*\**\s*([A-J])\b", re.IGNORECASE)


def main() -> int:
    """Generate and score every model on every requested benchmark."""

    args = _parse_args()
    results: dict[str, dict[str, float]] = {}
    for entry in args.model:
        name, path = entry.split("=", 1)
        results[name] = {}
        for benchmark in args.benchmark:
            rows = BENCHMARKS[benchmark]["load"](args.data_root)
            run_dir = args.out / name / benchmark
            print(f"\n=== {name} on {benchmark}: {len(rows)} questions ===", flush=True)
            responses = _generate(Path(path), [row["prompt"] for row in rows], run_dir, args)
            score = BENCHMARKS[benchmark]["score"](rows, responses, run_dir)
            results[name][benchmark] = score
            print(f"{name}/{benchmark}: {score:.4f}", flush=True)
            _write(args.out, results)
    _report(results, args.benchmark)
    return 0


def _generate(model: Path, prompts: list[str], run_dir: Path, args: argparse.Namespace) -> list[str]:
    """Generate one response per question, caching so a rerun costs nothing."""

    run_dir.mkdir(parents=True, exist_ok=True)
    cache = run_dir / "responses.jsonl"
    if cache.exists():
        texts = [json.loads(line)["response"] for line in cache.read_text(encoding="utf-8").splitlines() if line]
        if len(texts) == len(prompts):
            print(f"reusing {cache}")
            return texts

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(str(model))
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]
    engine = LLM(
        model=str(model),
        dtype="bfloat16",
        max_model_len=args.max_new_tokens + 2048,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    outputs = engine.generate(rendered, SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens))
    texts = [output.outputs[0].text for output in outputs]

    with cache.open("w", encoding="utf-8") as handle:
        for prompt, text in zip(prompts, texts, strict=True):
            handle.write(json.dumps({"prompt": prompt, "response": text}, ensure_ascii=False) + "\n")
    truncated = sum(1 for output in outputs if output.outputs[0].finish_reason == "length")
    (run_dir / "generation.json").write_text(
        json.dumps(
            {"questions": len(prompts), "truncated": truncated, "max_new_tokens": args.max_new_tokens}, indent=2
        ),
        encoding="utf-8",
    )
    print(f"  {truncated}/{len(prompts)} truncated at {args.max_new_tokens} tokens")
    return texts


# --------------------------------------------------------------------------------------
# Loading


def _load_math500(data_root: Path) -> list[dict[str, Any]]:
    rows = []
    for line in (data_root / "math_500/test.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append({"prompt": MATH_PROMPT.format(question=row["problem"]), "answer": row["answer"]})
    return rows


def _load_mmlu_pro(data_root: Path) -> list[dict[str, Any]]:
    import pandas as pd

    frame = pd.read_parquet(data_root / "mmlu_pro/test-00000-of-00001.parquet")
    rows = []
    for _, row in frame.iterrows():
        choices = "\n".join(f"{letter}. {text}" for letter, text in zip(string.ascii_uppercase, row["options"]))
        rows.append(
            {
                "prompt": CHOICE_PROMPT.format(question=f"{row['question']}\n{choices}"),
                "answer": str(row["answer"]).strip().upper(),
            }
        )
    return rows


def _load_gpqa(data_root: Path) -> list[dict[str, Any]]:
    """Load GPQA Diamond, shuffling each question's options under a fixed seed.

    The source file always lists the correct answer first, so the options must be permuted or
    every answer is A.
    """

    import csv
    import random

    path = data_root / "gpqa/gpqa_diamond.csv"
    rows = []
    rng = random.Random(240520)
    with path.open(encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            options = [
                record["Correct Answer"],
                record["Incorrect Answer 1"],
                record["Incorrect Answer 2"],
                record["Incorrect Answer 3"],
            ]
            order = list(range(4))
            rng.shuffle(order)
            shuffled = [options[index] for index in order]
            choices = "\n".join(f"{letter}. {text}" for letter, text in zip("ABCD", shuffled))
            rows.append(
                {
                    "prompt": CHOICE_PROMPT.format(question=f"{record['Question']}\n{choices}"),
                    "answer": "ABCD"[order.index(0)],
                }
            )
    return rows


# --------------------------------------------------------------------------------------
# Scoring


def _score_choice(rows: list[dict[str, Any]], responses: list[str], run_dir: Path) -> float:
    """Read the trailing 'Answer: X' the prompt asks for, scanning from the end."""

    correct, unparsed = 0, 0
    for row, response in zip(rows, responses, strict=True):
        matches = ANSWER_PATTERN.findall(response)
        if not matches:
            unparsed += 1
            continue
        correct += matches[-1].upper() == row["answer"]
    (run_dir / "parse.json").write_text(
        json.dumps({"questions": len(rows), "unparsed": unparsed}, indent=2), encoding="utf-8"
    )
    print(f"  {unparsed}/{len(rows)} responses had no parsable answer")
    return correct / len(rows)


def _score_math(rows: list[dict[str, Any]], responses: list[str], run_dir: Path) -> float:
    """Compare final answers with math_verify, so equivalent renderings still match."""

    from math_verify import parse, verify

    correct, unparsed = 0, 0
    for row, response in zip(rows, responses, strict=True):
        predicted = parse(response)
        if not predicted:
            unparsed += 1
            continue
        try:
            correct += bool(verify(parse(f"${row['answer']}$"), predicted))
        except Exception:  # noqa: BLE001 - an unverifiable pair is simply not a match
            continue
    (run_dir / "parse.json").write_text(
        json.dumps({"questions": len(rows), "unparsed": unparsed}, indent=2), encoding="utf-8"
    )
    print(f"  {unparsed}/{len(rows)} responses had no parsable answer")
    return correct / len(rows)


BENCHMARKS: dict[str, dict[str, Any]] = {
    "math500": {"load": _load_math500, "score": _score_math},
    "gpqa": {"load": _load_gpqa, "score": _score_choice},
    "mmlu_pro": {"load": _load_mmlu_pro, "score": _score_choice},
}


def _write(out: Path, results: dict[str, dict[str, float]]) -> None:
    """Merge into the metrics file, since each model is scored by a separate process."""

    out.mkdir(parents=True, exist_ok=True)
    path = out / "ood_metrics.json"
    merged = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for model, scores in results.items():
        merged.setdefault(model, {}).update(scores)
    path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _report(results: dict[str, dict[str, float]], benchmarks: list[str]) -> None:
    print("\n" + "=" * 60)
    print(f"{'model':<14}" + "".join(f"{name:>12}" for name in benchmarks))
    for name, scores in results.items():
        print(f"{name:<14}" + "".join(f"{scores.get(b, float('nan')) * 100:>11.2f}%" for b in benchmarks))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, metavar="NAME=PATH", help="repeatable")
    parser.add_argument("--data-root", type=Path, required=True, help="directory holding the benchmark data")
    parser.add_argument("--benchmark", action="append", choices=sorted(BENCHMARKS), help="repeatable, default all")
    parser.add_argument("--out", type=Path, default=ROOT / "output/ood-eval")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    # Reasoning benchmarks need room for a chain of thought before the final answer.
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()
    if any("=" not in entry for entry in args.model):
        raise ValueError("--model takes NAME=PATH")
    args.benchmark = args.benchmark or sorted(BENCHMARKS)
    return args


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"out-of-domain evaluation failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
