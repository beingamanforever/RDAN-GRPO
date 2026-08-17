#!/usr/bin/env python3
"""Sample teacher responses for training prompts through OpenRouter.

Writes one JSONL row per (prompt, sample). That file is also the resume log: a rerun reads
what already landed and requests only the missing samples, so an interrupted run or an
exhausted budget costs nothing to pick up. Sampling mirrors the RL rollout settings, so the
distilled responses come from the same region of the output distribution the policy explores.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BASE_URL = "https://openrouter.ai/api/v1"
ROUTING = {"order": ["alibaba"], "allow_fallbacks": True, "data_collection": "deny"}
TEMPERATURE = 0.99
TOP_P = 0.99


class BudgetExhausted(RuntimeError):
    """Raised once measured spend reaches the requested ceiling."""


def main() -> int:
    """Generate the missing teacher samples and report what they cost."""

    args = _parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY must be set to call the teacher")

    rows = _load_prompts(args.data, args.prompts)
    pending = _pending_samples(rows, args.out, args.samples)
    print(f"{len(rows)} prompts, {len(pending)} samples to generate, budget ${args.budget:.2f}")
    if not pending:
        return 0

    writer = _Writer(args.out, args.budget)
    client = _client(api_key, args.timeout)
    with ThreadPoolExecutor(max_workers=args.concurrency, thread_name_prefix="teacher") as pool:
        pool.map(lambda task: _generate(client, writer, args, *task), pending)
    print(f"wrote {writer.written} samples for ${writer.spent:.2f}")
    return 0


def _generate(client: Any, writer: _Writer, args: argparse.Namespace, row: dict[str, Any], sample: int) -> None:
    """Request one sample and append it, or record why the teacher could not produce it."""

    try:
        writer.check_budget()
        completion = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": row["prompt"]}],
            max_tokens=args.max_tokens,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            seed=args.seed + sample,
            extra_body={"provider": ROUTING, "reasoning": {"enabled": False}},
        )
    except BudgetExhausted:
        return
    except Exception as error:  # noqa: BLE001 - a sample that never arrived is data, not a crash
        writer.append({"id": row["id"], "sample": sample, "error": type(error).__name__})
        return

    choice = completion.choices[0]
    usage = completion.usage
    writer.append(
        {
            "id": row["id"],
            "sample": sample,
            "text": choice.message.content or "",
            "finish_reason": choice.finish_reason,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "cost": float(getattr(usage, "cost", 0.0) or 0.0),
        }
    )


class _Writer:
    """Append-only sample log with a hard spend ceiling shared across worker threads."""

    def __init__(self, path: Path, budget: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self._budget = budget
        self.spent = 0.0
        self.written = 0

    def append(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._handle.flush()
            self.spent += float(record.get("cost", 0.0))
            self.written += 1

    def check_budget(self) -> None:
        with self._lock:
            if self.spent >= self._budget:
                raise BudgetExhausted(f"spend reached ${self.spent:.2f}")


def _pending_samples(rows: list[dict[str, Any]], out: Path, samples: int) -> list[tuple[dict[str, Any], int]]:
    """Return the (row, sample index) pairs that the output file does not already hold."""

    done: set[tuple[str, int]] = set()
    if out.exists():
        with out.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if "text" in record:
                    done.add((str(record["id"]), int(record["sample"])))
    return [(row, sample) for row in rows for sample in range(samples) if (row["id"], sample) not in done]


def _load_prompts(path: Path, limit: int | None) -> list[dict[str, Any]]:
    """Read the training rows, keeping the fields the scorer needs alongside the prompt."""

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if limit is not None and len(rows) >= limit:
                break
            row = json.loads(line)
            rows.append({"id": str(row["id"]), "source": row["source"], "prompt": row["prompt"]})
    return rows


def _client(api_key: str, timeout: float) -> Any:
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=BASE_URL, timeout=timeout, max_retries=5)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/hybrid.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "data/distill/teacher_samples.jsonl")
    parser.add_argument("--model", default="qwen/qwen3.7-flash")
    parser.add_argument("--prompts", type=int, help="prompt count, default every row")
    parser.add_argument("--samples", type=int, default=4, help="samples per prompt")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=240520)
    parser.add_argument("--budget", type=float, default=10.0, help="stop once measured spend reaches this")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"distill failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
