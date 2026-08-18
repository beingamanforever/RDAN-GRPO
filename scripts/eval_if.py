#!/usr/bin/env python3
"""Score checkpoints on the three instruction-following benchmarks RTT reports.

IFEval, IFBench, and MulDimIF all ship deterministic scorers, so generation happens here and
scoring shells out to each benchmark's own evaluator. The numbers are therefore the
benchmarks' own rather than a reimplementation of their rules.

Held-out benchmarks answer the question training curves cannot: a rising training reward only
says the policy learned what the reward measures, while these say whether constraint following
generalized to instructions and checkers the run never saw.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Every benchmark here is scored greedily, which is each one's published protocol.
TEMPERATURE = 0.0


def main() -> int:
    """Generate and score each model on each benchmark, then write one comparison file."""

    args = _parse_args()
    results: dict[str, dict[str, dict[str, float]]] = {}
    for entry in args.model:
        name, path = entry.split("=", 1)
        results[name] = {}
        for benchmark in args.benchmark:
            spec = BENCHMARKS[benchmark]
            run_dir = args.out / name / benchmark
            prompts, rows = spec["load"](args.rtt_root)
            print(f"\n=== {name} on {benchmark}: {len(prompts)} prompts ===", flush=True)
            responses = _generate(Path(path), prompts, run_dir, args)
            results[name][benchmark] = spec["score"](args, run_dir, rows, responses)
            print(f"{name}/{benchmark}: {json.dumps(results[name][benchmark])}", flush=True)
        _write(args.out, results)
    _report(results, args.benchmark)
    return 0


def _generate(model: Path, prompts: list[str], run_dir: Path, args: argparse.Namespace) -> list[str]:
    """Generate one greedy response per prompt, caching so a rerun costs nothing."""

    run_dir.mkdir(parents=True, exist_ok=True)
    cache = run_dir / "responses.jsonl"
    if cache.exists():
        texts = [json.loads(line)["response"] for line in cache.read_text(encoding="utf-8").splitlines() if line]
        if len(texts) == len(prompts):
            print(f"reusing {cache}")
            return texts

    from transformers import AutoTokenizer
    from vllm import SamplingParams

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
    engine = _engine(model, args)
    outputs = engine.generate(rendered, SamplingParams(temperature=TEMPERATURE, max_tokens=args.max_new_tokens))
    texts = [output.outputs[0].text for output in outputs]

    with cache.open("w", encoding="utf-8") as handle:
        for prompt, text in zip(prompts, texts, strict=True):
            handle.write(json.dumps({"prompt": prompt, "response": text}, ensure_ascii=False) + "\n")
    # A truncated response fails nearly every constraint, so the truncation rate distinguishes
    # a genuine instruction-following gap from simply running out of token budget.
    truncated = sum(1 for output in outputs if output.outputs[0].finish_reason == "length")
    (run_dir / "generation.json").write_text(
        json.dumps({"prompts": len(prompts), "truncated": truncated, "max_new_tokens": args.max_new_tokens}, indent=2),
        encoding="utf-8",
    )
    print(f"  {truncated}/{len(prompts)} truncated at {args.max_new_tokens} tokens")
    return texts


_ENGINE: dict[str, Any] = {}


def _engine(model: Path, args: argparse.Namespace) -> Any:
    """Hold one engine per model, so several benchmarks reuse a single weight load."""

    from vllm import LLM

    key = str(model)
    if key not in _ENGINE:
        _ENGINE.clear()  # One 24GB card holds one model at a time.
        _ENGINE[key] = LLM(
            model=key,
            dtype="bfloat16",
            max_model_len=args.max_new_tokens + 2048,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    return _ENGINE[key]


# --------------------------------------------------------------------------------------
# Per-benchmark loading and scoring


def _load_jsonl_prompts(path: Path, field: str = "prompt") -> tuple[list[str], list[dict[str, Any]]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row[field] for row in rows], rows


def _load_ifeval(rtt: Path) -> tuple[list[str], list[dict[str, Any]]]:
    return _load_jsonl_prompts(rtt / "Benchmark/instruction_following_eval/data/input_data.jsonl")


def _load_ifbench(rtt: Path) -> tuple[list[str], list[dict[str, Any]]]:
    return _load_jsonl_prompts(rtt / "Benchmark/IFBench/data/IFBench_test.jsonl")


def _load_muldimif(rtt: Path) -> tuple[list[str], list[dict[str, Any]]]:
    rows = json.loads((rtt / "Benchmark/MulDimIF/Data/test.json").read_text(encoding="utf-8"))
    return [row["conversations"][0]["content"] for row in rows], rows


def _score_google_style(
    args: argparse.Namespace, run_dir: Path, rows: list[dict[str, Any]], responses: list[str], benchmark: str
) -> dict[str, float]:
    """Score IFEval or IFBench, which share Google's evaluator interface and output format."""

    spec = BENCHMARKS[benchmark]
    native = run_dir / "native"
    native.mkdir(exist_ok=True)
    payload = run_dir / "scored_input.jsonl"
    with payload.open("w", encoding="utf-8") as handle:
        for row, response in zip(rows, responses, strict=True):
            handle.write(json.dumps({"prompt": row["prompt"], "response": response}, ensure_ascii=False) + "\n")

    subprocess.run(
        [
            str(args.benchmark_python),
            spec["entry"],
            f"--input_data={args.rtt_root / spec['data']}",
            f"--input_response_data={payload}",
            f"--output_dir={native}",
        ],
        cwd=args.rtt_root / spec["cwd"],
        check=True,
        capture_output=True,
    )
    metrics: dict[str, float] = {}
    for mode in ("strict", "loose"):
        scored = [
            json.loads(line)
            for line in (native / f"eval_results_{mode}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        followed = [value for row in scored for value in row["follow_instruction_list"]]
        metrics[f"prompt_{mode}"] = sum(row["follow_all_instructions"] for row in scored) / len(scored)
        metrics[f"instruction_{mode}"] = sum(followed) / len(followed)
    return metrics


def _score_muldimif(
    args: argparse.Namespace, run_dir: Path, rows: list[dict[str, Any]], responses: list[str]
) -> dict[str, float]:
    """Score MulDimIF, whose evaluator reads the response as the last conversation turn."""

    payload = run_dir / "scored_input.jsonl"
    with payload.open("w", encoding="utf-8") as handle:
        for row, response in zip(rows, responses, strict=True):
            item = dict(row)
            item["conversations"] = list(row["conversations"]) + [{"role": "assistant", "content": response}]
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    save = run_dir / "muldimif_score.json"
    subprocess.run(
        [str(args.benchmark_python), "Code/evaluation/evaluation.py", f"--file_path={payload}", f"--save_path={save}"],
        cwd=args.rtt_root / "Benchmark/MulDimIF",
        check=True,
        capture_output=True,
    )
    score = json.loads(save.read_text(encoding="utf-8"))
    return {key: float(value) for key, value in _flatten(score).items() if isinstance(value, (int, float))}


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix.rstrip("."): value}
    flat: dict[str, Any] = {}
    for key, item in value.items():
        flat.update(_flatten(item, f"{prefix}{key}."))
    return flat


BENCHMARKS: dict[str, dict[str, Any]] = {
    "ifeval": {
        "load": _load_ifeval,
        "score": lambda a, d, r, x: _score_google_style(a, d, r, x, "ifeval"),
        # IFEval imports itself as a package, so it runs from Benchmark rather than from its
        # own directory the way IFBench's flat evaluator does.
        "entry": "instruction_following_eval/evaluation_main.py",
        "cwd": "Benchmark",
        "data": "Benchmark/instruction_following_eval/data/input_data.jsonl",
        "headline": "prompt_strict",
    },
    "ifbench": {
        "load": _load_ifbench,
        "score": lambda a, d, r, x: _score_google_style(a, d, r, x, "ifbench"),
        "entry": "run_eval.py",
        "cwd": "Benchmark/IFBench",
        "data": "Benchmark/IFBench/data/IFBench_test.jsonl",
        "headline": "prompt_strict",
    },
    "muldimif": {
        "load": _load_muldimif,
        "score": _score_muldimif,
        "headline": None,
    },
}


# --------------------------------------------------------------------------------------


def _write(out: Path, results: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "if_metrics.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _report(results: dict[str, dict[str, dict[str, float]]], benchmarks: list[str]) -> None:
    """Print the headline number per benchmark, so checkpoints compare at a glance."""

    print("\n" + "=" * 72)
    for benchmark in benchmarks:
        keys = sorted({key for model in results.values() for key in model.get(benchmark, {})})
        headline = BENCHMARKS[benchmark]["headline"] or (keys[0] if keys else None)
        if headline is None:
            continue
        print(f"\n{benchmark}  ({headline})")
        for name, per_benchmark in results.items():
            value = per_benchmark.get(benchmark, {}).get(headline)
            if value is not None:
                print(f"  {name:<12} {value:>8.1%}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, metavar="NAME=PATH", help="repeatable")
    parser.add_argument("--rtt-root", type=Path, required=True, help="RTT checkout holding Benchmark/")
    parser.add_argument("--benchmark-python", type=Path, required=True, help="interpreter with benchmark requirements")
    parser.add_argument("--benchmark", action="append", choices=sorted(BENCHMARKS), help="repeatable, default all")
    parser.add_argument("--out", type=Path, default=ROOT / "output/if-eval")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()
    if any("=" not in entry for entry in args.model):
        raise ValueError("--model takes NAME=PATH")
    args.benchmark = args.benchmark or sorted(BENCHMARKS)
    return args


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, ValueError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", b"")
        if detail:
            print(detail.decode()[-1500:], file=sys.stderr)
        print(f"instruction-following evaluation failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
