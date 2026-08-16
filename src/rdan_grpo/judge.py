"""Concurrent OpenRouter judge for soft rubrics.

A 500-step run issues on the order of 100k judge calls, so throughput and transport
resilience decide whether the GPUs are training or waiting. Calls for one batch run
concurrently on one pooled client; transport failures retry with jittered backoff and,
if they still fail, mark only that response's soft channel invalid rather than failing
the step.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROCESS_SCORES = (0, 0.5, 1)
SIGNED_PROCESS_SCORES = (-1.0, 0.0, 1.0)
RETRY_BASE_SECONDS = 1.0
RETRY_CAP_SECONDS = 30.0
# Status codes worth another attempt: rate limits, upstream overload, gateway errors.
RETRY_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class JudgeRequest:
    """One response and the soft rubrics to score against it."""

    instruction: str
    response: str
    rubrics: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class JudgeResult:
    """Judgments keyed by rubric id, plus what the call cost and why it failed."""

    judgments: dict[int, dict[str, Any]]
    valid: bool
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    attempts: int = 1


@dataclass
class JudgeStats:
    """Rolling per-batch judge health, reset by :meth:`OpenRouterJudge.drain_stats`."""

    calls: int = 0
    failures: int = 0
    retries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latencies: list[float] = field(default_factory=list)
    errors: dict[str, int] = field(default_factory=dict)


def load_judge_config(path: str | Path) -> dict[str, Any]:
    """Read the judge configuration and its sibling prompt template."""

    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["prompt"] = (config_path.parent / "rubric_prompt.txt").read_text(encoding="utf-8")
    return config


def signed_process_score(score: float) -> float:
    """Map a judge score in {0, 0.5, 1} onto the signed rubric scale."""

    return 2.0 * float(score) - 1.0


class OpenRouterJudge:
    """Score soft rubrics through OpenRouter with bounded concurrency and retries."""

    def __init__(self, config: Mapping[str, Any], api_key: str, client: Any | None = None) -> None:
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required")
        self.config = config
        self.prompt = config["prompt"]
        self.max_concurrency = int(config["max_concurrency"])
        self.max_attempts = int(config["max_attempts"])
        self.stats = JudgeStats()
        self._lock = threading.Lock()
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url=config["base_url"],
                timeout=float(config["timeout_seconds"]),
                # Backoff, jitter, and retry accounting are handled here so failures are
                # observable; the SDK's own retry loop would hide them.
                max_retries=0,
            )
        self.client = client

    def judge_batch(self, requests: Sequence[JudgeRequest]) -> list[JudgeResult]:
        """Judge every request concurrently and return results in request order."""

        if not requests:
            return []
        workers = min(self.max_concurrency, len(requests))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="judge") as pool:
            return list(pool.map(self.judge, requests))

    def judge(self, request: JudgeRequest) -> JudgeResult:
        """Judge one response, retrying transport failures before giving up."""

        ids = [rubric["id"] for rubric in request.rubrics]
        payload = self._build_payload(request)
        started = time.monotonic()
        error = "no_attempt"
        for attempt in range(1, self.max_attempts + 1):
            try:
                completion = self.client.chat.completions.create(**payload)
                result = self._parse(completion, ids, time.monotonic() - started, attempt)
                self._record(result)
                return result
            except Exception as exc:  # noqa: BLE001 - every transport failure is retryable data
                error = type(exc).__name__
                if attempt == self.max_attempts or not _retryable(exc):
                    break
                time.sleep(_backoff(attempt, exc))
        result = JudgeResult({}, False, error, latency_seconds=time.monotonic() - started, attempts=attempt)
        self._record(result)
        return result

    def drain_stats(self) -> dict[str, Any]:
        """Return this worker's raw counters and reset the accumulator.

        Counters rather than rates, because one training step is judged across several reward
        workers and rates cannot be averaged back together correctly.
        """

        with self._lock:
            stats, self.stats = self.stats, JudgeStats()
        price = self.config["price_per_million"]
        return {
            "calls": stats.calls,
            "failures": stats.failures,
            "retries": stats.retries,
            "prompt_tokens": stats.prompt_tokens,
            "completion_tokens": stats.completion_tokens,
            "latencies": stats.latencies,
            "errors": stats.errors,
            "cost_usd": (stats.prompt_tokens * price["prompt"] + stats.completion_tokens * price["completion"])
            / 1_000_000,
        }

    def _build_payload(self, request: JudgeRequest) -> dict[str, Any]:
        values = {
            "instruction": request.instruction,
            "response": request.response,
            "rubrics_json": json.dumps(
                [{"id": rubric["id"], "text": rubric["text"]} for rubric in request.rubrics],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        }
        content = re.sub(r"{{(instruction|response|rubrics_json)}}", lambda match: values[match.group(1)], self.prompt)
        return {
            "model": self.config["model"],
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.config["max_tokens"],
            "temperature": self.config["temperature"],
            "seed": self.config["seed"],
            "response_format": self.config["response_format"],
            # Reasoning is a provider-level control OpenRouter forwards, not an OpenAI field.
            "extra_body": {"provider": self.config["routing"], "reasoning": self.config["reasoning"]},
        }

    def _parse(self, completion: Any, ids: list[int], latency: float, attempts: int) -> JudgeResult:
        usage = getattr(completion, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        try:
            content = completion.choices[0].message.content
            rows = json.loads(content)["rubrics"]
            judgments = _validate_rows(rows, ids)
        except Exception as exc:  # noqa: BLE001 - a malformed judgment is a soft failure
            return JudgeResult(
                {}, False, f"schema_{type(exc).__name__}", prompt_tokens, completion_tokens, latency, attempts
            )
        return JudgeResult(judgments, True, None, prompt_tokens, completion_tokens, latency, attempts)

    def _record(self, result: JudgeResult) -> None:
        with self._lock:
            self.stats.calls += 1
            self.stats.retries += result.attempts - 1
            self.stats.prompt_tokens += result.prompt_tokens
            self.stats.completion_tokens += result.completion_tokens
            self.stats.latencies.append(result.latency_seconds)
            if not result.valid:
                self.stats.failures += 1
                name = result.error or "unknown"
                self.stats.errors[name] = self.stats.errors.get(name, 0) + 1


def aggregate_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Combine per-worker counters from one training step into judge health metrics."""

    totals = {name: sum(int(row.get(name, 0)) for row in rows) for name in ("calls", "failures", "retries")}
    tokens = {name: sum(int(row.get(name, 0)) for row in rows) for name in ("prompt_tokens", "completion_tokens")}
    if not totals["calls"]:
        return {}
    latencies = sorted(value for row in rows for value in row.get("latencies", ()))
    errors: dict[str, int] = {}
    for row in rows:
        for name, count in row.get("errors", {}).items():
            errors[name] = errors.get(name, 0) + int(count)
    return {
        "judge/calls": float(totals["calls"]),
        "judge/failure_rate": totals["failures"] / totals["calls"],
        "judge/retry_rate": totals["retries"] / totals["calls"],
        "judge/latency_p50": _quantile(latencies, 0.5),
        "judge/latency_p95": _quantile(latencies, 0.95),
        "judge/prompt_tokens": float(tokens["prompt_tokens"]),
        "judge/completion_tokens": float(tokens["completion_tokens"]),
        "judge/cost_usd": float(sum(row.get("cost_usd", 0.0) for row in rows)),
        **{f"judge/error_{name}": float(count) for name, count in errors.items()},
    }


def _validate_rows(rows: Any, ids: list[int]) -> dict[int, dict[str, Any]]:
    if not isinstance(rows, list) or [row.get("id") for row in rows if isinstance(row, dict)] != ids:
        raise ValueError("judge returned different rubric ids")
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        if set(row) != {"id", "score", "reason"} or row["score"] not in PROCESS_SCORES:
            raise ValueError("judge output does not match the schema")
        if not isinstance(row["reason"], str) or not row["reason"]:
            raise ValueError("judge reason is empty")
        result[row["id"]] = {"score": signed_process_score(row["score"]), "reason": row["reason"]}
    return result


def _retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status in RETRY_STATUS
    # Connection resets, timeouts, and unclassified transport errors all deserve another try.
    return not isinstance(exc, (ValueError, KeyError, TypeError))


def _backoff(attempt: int, exc: Exception) -> float:
    """Honour a server-supplied Retry-After, otherwise exponential backoff with jitter."""

    headers = getattr(getattr(exc, "response", None), "headers", None)
    retry_after = headers.get("retry-after") if isinstance(headers, Mapping) else None
    if retry_after is not None:
        try:
            return min(float(retry_after), RETRY_CAP_SECONDS)
        except (TypeError, ValueError):
            pass
    delay = min(RETRY_BASE_SECONDS * 2 ** (attempt - 1), RETRY_CAP_SECONDS)
    return delay * random.uniform(0.5, 1.5)


def _quantile(ordered: Sequence[float], quantile: float) -> float:
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
