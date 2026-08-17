#!/usr/bin/env python3
"""Turn teacher samples into SFT and DPO datasets using the training rubrics.

Every sample is scored through the same deterministic checkers and the same OpenRouter judge
the RL reward uses, so what SFT imitates and what DPO prefers are defined by the objective the
policy is later trained against. A sample ranks by hard-rubric pass first and soft-rubric
quality second: prose can never buy a place over a satisfied constraint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    """Score every teacher sample and write the SFT and preference datasets."""

    args = _parse_args()
    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        raise ValueError("OPENROUTER_API_KEY must be set to score soft rubrics")

    from rdan_grpo.compat import install_rtt_runtime
    from rdan_grpo.distillation import select_pairs, select_sft

    install_rtt_runtime(args.rtt_root)

    prompts = _load_prompts(args.data)
    samples = _load_samples(args.samples, prompts)
    print(f"scoring {len(samples)} samples over {len({s['id'] for s in samples})} prompts")

    scored = _score(samples, prompts)
    # A sample whose soft rubrics never came back has an unknown quality, not a low one, so it
    # can neither be imitated nor ranked against its siblings.
    judged = [sample for sample in scored if sample["quality_valid"] or not sample["has_soft"]]
    groups = defaultdict(list)
    for sample in judged:
        groups[sample["id"]].append(sample)

    sft = select_sft(groups, prompts, args.min_quality)
    dpo = select_pairs(groups, prompts, args.min_gap)
    _write(args.sft_out, sft)
    _write(args.dpo_out, dpo)
    _report(scored, sft, dpo, groups)
    return 0


def _score(samples: list[dict[str, Any]], prompts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach hard pass and soft quality to each sample through the real reward path."""

    import torch

    from rdan_grpo.distillation import rank
    from rdan_grpo.judge import JudgeRequest, OpenRouterJudge, load_judge_config
    from rdan_grpo.reward_worker import JUDGE_CONFIG, _evaluate_hard_rubrics
    from rdan_grpo.rewards import extract_quality

    rows = [
        _evaluate_hard_rubrics(
            prompts[sample["id"]]["prompt"],
            sample["text"],
            prompts[sample["id"]]["rubrics"],
            prompts[sample["id"]]["source"],
            prompts[sample["id"]]["ground_truth"],
        )
        for sample in samples
    ]

    judge = OpenRouterJudge(load_judge_config(JUDGE_CONFIG), os.environ["OPENROUTER_API_KEY"])
    pending = [index for index, row in enumerate(rows) if row["judge_rubrics"]]
    requests = [
        JudgeRequest(
            prompts[samples[index]["id"]]["prompt"], samples[index]["text"], tuple(rows[index]["judge_rubrics"])
        )
        for index in pending
    ]
    print(f"judging {len(requests)} of {len(rows)} samples")
    for index, result in zip(pending, judge.judge_batch(requests), strict=True):
        if not result.valid:
            continue
        for rubric in rows[index]["judge_rubrics"]:
            judgment = result.judgments.get(rubric["id"])
            if judgment is not None:
                rows[index]["scores"][rubric["id"] - 1] = float(judgment["score"])
                rows[index]["evaluated"][rubric["id"] - 1] = True

    scores = torch.tensor([row["scores"] for row in rows], dtype=torch.float32)
    active = torch.zeros_like(scores, dtype=torch.bool)
    for index, row in enumerate(rows):
        active[index, : row["active"]] = True
    evaluated = torch.tensor([row["evaluated"] for row in rows], dtype=torch.bool) & active
    hard = torch.tensor([row["hard"] for row in rows], dtype=torch.bool) & active
    quality = extract_quality(torch.where(active, scores, torch.zeros_like(scores)), active, evaluated, hard)

    print(f"judge cost ${judge.drain_stats()['cost_usd']:.2f}")
    return [
        {
            **sample,
            "hard_pass": bool(quality.hard_pass[index]),
            "quality": float(quality.quality[index]),
            "quality_valid": bool(quality.quality_valid[index]),
            "has_soft": bool(rows[index]["judge_rubrics"]),
            "rank": rank(bool(quality.hard_pass[index]), float(quality.quality[index])),
        }
        for index, sample in enumerate(samples)
    ]


def _report(
    scored: list[dict[str, Any]],
    sft: list[dict[str, Any]],
    dpo: list[dict[str, Any]],
    groups: dict[str, list[dict[str, Any]]],
) -> None:
    hard_pass = sum(1 for sample in scored if sample["hard_pass"])
    quality = [sample["quality"] for sample in scored if sample["quality_valid"]]
    dropped = sum(1 for sample in scored if sample["has_soft"] and not sample["quality_valid"])
    print(f"\nhard pass rate    {hard_pass / len(scored):.3f} over {len(scored)} samples")
    if quality:
        print(f"mean soft quality {sum(quality) / len(quality):.3f} over {len(quality)} judged samples")
    print(f"unjudged dropped  {dropped}")
    print(f"sft rows          {len(sft)} of {len(groups)} prompts")
    print(f"dpo pairs         {len(dpo)} of {len(groups)} prompts")


def _load_samples(path: Path, prompts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Read the teacher log, dropping failed calls and responses the token cap truncated."""

    samples = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            text = record.get("text", "").strip()
            if not text or record.get("finish_reason") != "stop" or record["id"] not in prompts:
                continue
            samples.append({"id": str(record["id"]), "sample": record["sample"], "text": text})
    return samples


def _load_prompts(path: Path) -> dict[str, dict[str, Any]]:
    """Index the training rows by id, decoding the fields the checkers need."""

    prompts = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rubrics, truth = row["rubrics"], row["ground_truth"]
            prompts[str(row["id"])] = {
                "prompt": row["prompt"],
                "source": row["source"],
                "rubrics": json.loads(rubrics) if isinstance(rubrics, str) else rubrics,
                "ground_truth": json.loads(truth) if isinstance(truth, str) else truth,
            }
    return prompts


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/hybrid.jsonl")
    parser.add_argument("--samples", type=Path, default=ROOT / "data/distill/teacher_samples.jsonl")
    parser.add_argument("--sft-out", type=Path, default=ROOT / "data/distill/sft.jsonl")
    parser.add_argument("--dpo-out", type=Path, default=ROOT / "data/distill/dpo.jsonl")
    parser.add_argument("--rtt-root", type=Path, help="RTT checkout supplying ROLL (or set RTT_ROOT)")
    parser.add_argument("--min-quality", type=float, default=0.7, help="soft quality floor for an SFT target")
    parser.add_argument("--min-gap", type=float, default=0.2, help="rank separation a DPO pair must show")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"build_preferences failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
