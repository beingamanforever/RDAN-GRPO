#!/usr/bin/env python3
"""Check the whole RDAN chain on a handful of real prompts before spending GPU hours.

Generates with vLLM on the real model, scores the rollouts through the real checkers and the
real OpenRouter judge, builds the RDAN advantage, and reports measured cost so the full run
can be budgeted. Runs in minutes and writes nothing into the training output tree.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_DATA = ROOT / "data/hybrid.jsonl"


def main() -> int:
    """Run generation, reward, and advantage over a few prompts and print the findings."""

    args = _parse_args()
    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        raise ValueError("OPENROUTER_API_KEY must be set to score soft rubrics")

    from rdan_grpo.compat import install_rtt_runtime

    install_rtt_runtime(args.rtt_root)

    rows = _load_rows(args.data, args.prompts)
    print(f"loaded {len(rows)} prompts: " + ", ".join(sorted({row["source"] for row in rows})))

    responses = _generate(args.model, rows, args.samples, args.max_new_tokens)
    scored = _score(rows, responses, args.samples)
    _report(scored, rows, args)
    return 0


def _generate(model: str, rows: list[dict[str, Any]], samples: int, max_new_tokens: int) -> list[list[str]]:
    """Sample responses per prompt with the same decoding settings training uses."""

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for row in rows
    ]
    engine = LLM(model=model, dtype="bfloat16", max_model_len=8000, gpu_memory_utilization=0.8)
    sampling = SamplingParams(
        n=samples, temperature=0.99, top_p=0.99, top_k=100, max_tokens=max_new_tokens, seed=240520
    )
    outputs = engine.generate(prompts, sampling)
    return [[completion.text for completion in output.outputs] for output in outputs]


def _score(rows: list[dict[str, Any]], responses: list[list[str]], samples: int) -> dict[str, Any]:
    """Run the real reward path and the real advantage math over the generated rollouts."""

    import torch

    from rdan_grpo.judge import JudgeRequest, OpenRouterJudge, aggregate_stats, load_judge_config
    from rdan_grpo.reward_worker import JUDGE_CONFIG, _evaluate_hard_rubrics
    from rdan_grpo.scalar import build_scalar_output

    judge = OpenRouterJudge(load_judge_config(JUDGE_CONFIG), os.environ["OPENROUTER_API_KEY"])
    evaluated = [
        _evaluate_hard_rubrics(row["prompt"], text, row["rubrics"], row["source"], row["ground_truth"])
        for row, texts in zip(rows, responses, strict=True)
        for text in texts
    ]
    flat_responses = [text for texts in responses for text in texts]
    flat_prompts = [row["prompt"] for row in rows for _ in range(samples)]

    pending = [index for index, item in enumerate(evaluated) if item["judge_rubrics"]]
    requests = [
        JudgeRequest(flat_prompts[index], flat_responses[index], tuple(evaluated[index]["judge_rubrics"]))
        for index in pending
    ]
    print(f"judging {len(requests)} of {len(evaluated)} responses concurrently")
    for index, result in zip(pending, judge.judge_batch(requests), strict=True):
        if result.valid:
            for rubric in evaluated[index]["judge_rubrics"]:
                evaluated[index]["scores"][rubric["id"] - 1] = float(result.judgments[rubric["id"]]["score"])
                evaluated[index]["evaluated"][rubric["id"] - 1] = True
        else:
            evaluated[index]["judge_failed"] = True

    scores = torch.tensor([item["scores"] for item in evaluated], dtype=torch.float32)
    active = torch.zeros_like(scores, dtype=torch.bool)
    for index, item in enumerate(evaluated):
        active[index, : item["active"]] = True
    eval_mask = torch.tensor([item["evaluated"] for item in evaluated], dtype=torch.bool) & active
    hard_mask = torch.tensor([item["hard"] for item in evaluated], dtype=torch.bool) & active
    prompt_keys = [row["id"] for row in rows for _ in range(samples)]

    output = build_scalar_output(
        "rdan",
        prompt_keys,
        torch.where(active, scores, torch.zeros_like(scores)),
        active,
        eval_mask,
        hard_mask,
        group_size=samples,
        quality_weight=1.0,
    )
    return {
        "output": output,
        "evaluated": evaluated,
        "judge_stats": aggregate_stats([judge.drain_stats()]),
        "lengths": [len(text) for text in flat_responses],
    }


def _report(scored: dict[str, Any], rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """Print reward, advantage, judge health, and the projected cost of a full run."""

    output = scored["output"]
    stats = scored["judge_stats"]
    judged = sum(1 for item in scored["evaluated"] if item["judge_rubrics"])
    responses = len(scored["evaluated"])

    print("\n--- rubric outcomes ---")
    print(f"hard pass rate          {output.diagnostics['hard_pass_rate']:.3f}")
    print(f"response valid rate     {output.diagnostics['response_valid_rate']:.3f}")
    print(f"quality eligible rate   {output.diagnostics['quality_eligible_rate']:.3f}")
    print(f"outcome reward mean     {output.selected_raw_reward.mean():.3f}")
    eligible = output.raw_quality[output.quality_eligible]
    print(f"process quality mean    {eligible.mean().item() if eligible.numel() else float('nan'):.3f}")
    print(f"mean response chars     {statistics.fmean(scored['lengths']):.0f}")

    print("\n--- advantage ---")
    print(f"outcome advantage std   {output.response_advantage.std(unbiased=False):.3f}")
    print(f"process advantage std   {output.quality_advantage.std(unbiased=False):.3f}")
    print(f"total advantage std     {output.scalar_advantage.std(unbiased=False):.3f}")
    zero_rate = float((output.scalar_advantage.abs() <= 1e-8).float().mean())
    print(f"zero advantage rate     {zero_rate:.3f}")
    if zero_rate == 1.0:
        print("WARNING: every advantage is zero, so training would produce no gradient")

    if stats:
        print("\n--- judge ---")
        print(f"failure rate            {stats['judge/failure_rate']:.3f}")
        print(f"retry rate              {stats['judge/retry_rate']:.3f}")
        print(f"latency p50 / p95       {stats['judge/latency_p50']:.1f}s / {stats['judge/latency_p95']:.1f}s")
        prompt_per_call = stats["judge/prompt_tokens"] / max(judged, 1)
        completion_per_call = stats["judge/completion_tokens"] / max(judged, 1)
        print(f"tokens per call         {prompt_per_call:.0f} in / {completion_per_call:.0f} out")
        print(f"measured cost           ${stats['judge/cost_usd']:.4f} for {judged} calls")

        judge_rate = judged / responses
        calls = args.rollout_batch_size * args.samples * judge_rate * args.steps
        print(f"\nprojected {args.steps}-step run at rollout_batch_size={args.rollout_batch_size}:")
        print(f"  judge calls           {calls:,.0f}")
        print(f"  judge cost            ${stats['judge/cost_usd'] / max(judged, 1) * calls:,.2f}")
        waves = args.rollout_batch_size * args.samples * judge_rate / (args.reward_workers * args.concurrency)
        print(f"  judging per step      ~{max(waves, 1) * stats['judge/latency_p50']:.0f}s")


def _load_rows(path: Path, count: int) -> list[dict[str, Any]]:
    """Take the first row of each source, so every checker route is exercised."""

    picked: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["source"] in picked:
                continue
            truth = row["ground_truth"]
            picked[row["source"]] = {
                "id": str(row["id"]),
                "source": row["source"],
                "prompt": row["prompt"],
                "rubrics": json.loads(row["rubrics"]) if isinstance(row["rubrics"], str) else row["rubrics"],
                "ground_truth": json.loads(truth) if isinstance(truth, str) else truth,
            }
            if len(picked) >= count:
                break
    return list(picked.values())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("RDAN_MODEL_SNAPSHOT"), help="model path or snapshot")
    parser.add_argument("--rtt-root", type=Path, help="RTT checkout supplying ROLL (or set RTT_ROOT)")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--prompts", type=int, default=5, help="distinct prompts, one per source")
    parser.add_argument("--samples", type=int, default=8, help="responses per prompt, the training group size")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=500, help="horizon to project cost over")
    parser.add_argument("--rollout-batch-size", type=int, default=64)
    parser.add_argument("--reward-workers", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=48)
    args = parser.parse_args()
    if not args.model:
        raise ValueError("--model or RDAN_MODEL_SNAPSHOT is required")
    return args


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"preflight failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
