#!/usr/bin/env python3
"""Write soft rubrics for the training rows that only carry deterministic checkers.

Those rows stop teaching once every sampled response satisfies their checkers: the outcome
reward goes flat across the group, the advantage normalizes to zero, and the prompt costs a
generation that moves no weights. A judged quality dimension gives the group a tiebreaker, so
the row keeps producing gradient after its mechanical requirements are saturated.

Generated rubrics stay in their own file and carry a `generated` flag, so any result computed
on them can be separated from results on the original data.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdan_grpo.rewards import hard_mask  # noqa: E402

JUDGE_CONFIG = ROOT / "configs/judges/openrouter.json"
PROMPT_FILE = ROOT / "configs/judges/soft_rubric_prompt.txt"

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "soft_rubrics",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["rubrics"],
            "properties": {
                "rubrics": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 4,
                    "items": {"type": "string", "minLength": 20},
                }
            },
        },
    },
}


def main() -> int:
    """Generate rubrics for every selected row that does not already have them."""

    args = _parse_args()
    config = json.loads(JUDGE_CONFIG.read_text(encoding="utf-8"))
    prompt = PROMPT_FILE.read_text(encoding="utf-8")

    rows = _select_rows(args.data, args.limit, args.seed)
    done = _already_generated(args.out)
    pending = [row for row in rows if row["id"] not in done]
    print(f"{len(rows)} hard-only rows selected, {len(done)} already generated, {len(pending)} to do")
    if not pending:
        return 0

    generator = RubricGenerator(config, prompt, args.count_range, _api_key())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    written = failed = 0

    with args.out.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=min(config["max_concurrency"], len(pending))) as pool:
            for row, rubrics in zip(pending, pool.map(generator.generate, pending), strict=True):
                if not rubrics:
                    failed += 1
                    continue
                record = {
                    "id": row["id"],
                    "source": row["source"],
                    "prompt": row["prompt"],
                    "hard_rubrics": [rubric["description"] for rubric in row["rubrics"]],
                    "soft_rubrics": rubrics,
                    "generated": True,
                }
                with lock:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                written += 1

    print(f"\nwrote {written} rows to {args.out}, {failed} failed")
    print(f"cost ${generator.cost:.4f} over {generator.calls} calls")
    return 0


class RubricGenerator:
    """Ask the model for soft rubrics, retrying transport failures before giving up."""

    def __init__(self, config: dict[str, Any], prompt: str, count_range: str, api_key: str) -> None:
        from openai import OpenAI

        self.config = config
        self.prompt = prompt
        self.count_range = count_range
        self.client = OpenAI(
            api_key=api_key,
            base_url=config["base_url"],
            timeout=float(config["timeout_seconds"]),
            max_retries=config["max_attempts"],
        )
        self.cost = 0.0
        self.calls = 0
        self._lock = threading.Lock()

    def generate(self, row: dict[str, Any]) -> list[str]:
        """Return the rubrics for one row, or an empty list when the call cannot be completed."""

        hard = "\n".join(f"- {rubric['description']}" for rubric in row["rubrics"])
        content = (
            self.prompt.replace("{{instruction}}", row["prompt"])
            .replace("{{hard_rubrics}}", hard)
            .replace("{{count_range}}", self.count_range)
        )
        try:
            completion = self.client.chat.completions.create(
                model=self.config["model"],
                messages=[{"role": "user", "content": content}],
                max_tokens=self.config["max_tokens"],
                temperature=self.config["temperature"],
                seed=self.config["seed"],
                response_format=RESPONSE_FORMAT,
                extra_body={"provider": self.config["routing"], "reasoning": self.config["reasoning"]},
            )
        except Exception as error:  # noqa: BLE001 - a row that cannot be generated is skipped, not fatal
            print(f"  row {row['id']} failed: {type(error).__name__}: {error}", flush=True)
            return []

        self._record(completion)
        try:
            return json.loads(completion.choices[0].message.content)["rubrics"]
        except (json.JSONDecodeError, KeyError, TypeError, IndexError):
            print(f"  row {row['id']} returned unparsable rubrics", flush=True)
            return []

    def _record(self, completion: Any) -> None:
        usage = getattr(completion, "usage", None)
        price = self.config["price_per_million"]
        cost = getattr(usage, "cost", None) or getattr(getattr(usage, "usage", None), "cost", None)
        if cost is None:
            cost = (
                int(getattr(usage, "prompt_tokens", 0) or 0) * price["prompt"]
                + int(getattr(usage, "completion_tokens", 0) or 0) * price["completion"]
            ) / 1_000_000
        with self._lock:
            self.cost += float(cost)
            self.calls += 1


def _select_rows(data: Path, limit: int | None, seed: int) -> list[dict[str, Any]]:
    """Return the rows whose rubrics are all deterministic, sampled when a limit is given."""

    rows = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines() if line.strip()]
    hard_only = [row for row in rows if _soft_count(row) == 0]
    if limit is None or limit >= len(hard_only):
        return hard_only
    return random.Random(seed).sample(hard_only, limit)


def _soft_count(row: dict[str, Any]) -> int:
    """Count the row's judged rubrics exactly as the reward worker classifies them."""

    rubrics = row.get("rubrics") or []
    if not rubrics:
        return 0
    truth = row.get("ground_truth") or {}
    if isinstance(truth, str):
        truth = json.loads(truth)
    return sum(1 for hard in hard_mask(row["source"], truth, len(rubrics)) if not hard)


def _already_generated(out: Path) -> set[Any]:
    """Read the ids already written, so an interrupted run resumes instead of repeating."""

    if not out.exists():
        return set()
    return {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}


def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise ValueError("OPENROUTER_API_KEY is not set and is not in .env")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/hybrid.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "data/soft_rubrics_generated.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="sample this many rows, default all")
    parser.add_argument("--seed", type=int, default=240520)
    parser.add_argument("--count-range", default="2 to 4")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"soft rubric generation failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
