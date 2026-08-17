"""Selection rules that turn scored teacher samples into SFT targets and preference pairs.

A sample ranks by hard-rubric pass first and soft-rubric quality second, so prose can never
buy a place over a satisfied constraint. Both datasets carry the prompt-completion shape the
trainer masks on, so the loss falls only on what the teacher wrote.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# Hard pass outranks any soft quality, which lives in [0, 1].
HARD_PASS_WEIGHT = 2.0


def rank(hard_pass: bool, quality: float) -> float:
    """Score one sample on the scale the two datasets rank against."""

    return HARD_PASS_WEIGHT * float(hard_pass) + float(quality)


def select_sft(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    prompts: Mapping[str, Mapping[str, Any]],
    min_quality: float,
) -> list[dict[str, Any]]:
    """Keep the best sample per prompt, and only when it satisfies the rubrics outright."""

    rows = []
    for prompt_id, samples in groups.items():
        best = max(samples, key=lambda sample: sample["rank"])
        if not best["hard_pass"] or (best["quality_valid"] and best["quality"] < min_quality):
            continue
        rows.append(
            {
                "id": prompt_id,
                "source": prompts[prompt_id]["source"],
                "quality": round(best["quality"], 4),
                "prompt": [{"role": "user", "content": prompts[prompt_id]["prompt"]}],
                "completion": [{"role": "assistant", "content": best["text"]}],
            }
        )
    return rows


def select_pairs(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    prompts: Mapping[str, Mapping[str, Any]],
    min_gap: float,
) -> list[dict[str, Any]]:
    """Pair the best and worst sample per prompt when the rubrics separate them clearly."""

    rows = []
    for prompt_id, samples in groups.items():
        if len(samples) < 2:
            continue
        ordered = sorted(samples, key=lambda sample: sample["rank"])
        worst, best = ordered[0], ordered[-1]
        if not best["hard_pass"] or best["rank"] - worst["rank"] < min_gap:
            continue
        rows.append(
            {
                "id": prompt_id,
                "source": prompts[prompt_id]["source"],
                "gap": round(best["rank"] - worst["rank"], 4),
                "prompt": [{"role": "user", "content": prompts[prompt_id]["prompt"]}],
                "chosen": [{"role": "assistant", "content": best["text"]}],
                "rejected": [{"role": "assistant", "content": worst["text"]}],
            }
        )
    return rows
