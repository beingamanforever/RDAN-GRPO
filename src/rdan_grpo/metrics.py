"""Lossless metric records and deterministic offline training figures."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

SCHEMA_VERSION = 2
ZERO_TOLERANCE = 1e-6
IDENTITY_KEYS = {
    "project",
    "run_id",
    "method",
    "stage",
    "model_id",
    "model_revision",
    "tokenizer_revision",
    "data_revision",
    "code_revision",
    "config_sha256",
    "protocol_sha256",
    "seed",
}
RECORD_KEYS = {
    "schema_version",
    "kind",
    "identity",
    "step",
    "wall_time_s",
    "rows",
    "stats",
    "eval",
    "hparams",
    "evidence",
}
ROW_KEYS = {
    "sample_sha256",
    "group_sha256",
    "response_index",
    "group_size",
    "group_complete",
    "hard_valid",
    "correct",
    "evaluation",
    "rubrics",
    "reward",
    "advantage",
    "generation",
    "token",
}
EVALUATION_KEYS = {"evaluator_revision", "unsupported_hard", "judge_failed", "invalid_output"}
RUBRIC_KEYS = {"rubric_sha256", "rubric_type", "route", "outcome", "score", "evaluator_revision"}
REWARD_KEYS = {"aon", "csr", "signed_csr", "quality", "mix", "selected"}
ADVANTAGE_KEYS = {"aon", "csr", "mix", "rdan", "response", "quality", "token"}
GENERATION_KEYS = {"token_count", "cap_hit", "aborted", "stop_reason"}
TOKEN_KEYS = {"mask", "relevance"}
STAT_KEYS = {"entropy", "kl", "clip", "length", "throughput", "gpu_hours"}
EVIDENCE_KEYS = {"calibration_checked", "calibration_failed", "nonfinite_count"}
STOP_ORDER = (
    "format_collapse",
    "no_hard_variance",
    "zero_advantage",
    "clip_saturated",
    "calibration_invalid",
    "evaluator_invalid",
    "credit_leakage",
    "token_invalid",
    "nonfinite",
    "halted",
)
FIGURE_KEYS = {
    "rtt_fig3_sensitivity",
    "rtt_fig4_checkpoint_eval",
    "rtt_fig5_training",
    "papo_fig2_reward_hacking",
    "papo_fig4_advantage_health",
    "papo_fig8_correct_advantage",
    "papo_fig9_advantage_components",
}
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,127}\Z")
SAFE_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+/-]{0,255}\Z")
HEX_REVISION = re.compile(r"[0-9a-f]{40,64}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class MetricError(ValueError):
    """Raised when metric evidence violates the recording contract."""


def write_records(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Validate and atomically write one identity-stable JSONL run."""

    target = Path(path)
    values = list(records)
    validate_records(values)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            for record in values:
                handle.write(_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate an identity-stable JSONL run."""

    source = Path(path)
    records: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MetricError(f"{source}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise MetricError(f"{source}:{line_number}: record must be an object")
            records.append(value)
    validate_records(records)
    return records


def validate_records(records: Sequence[Mapping[str, Any]]) -> None:
    """Validate schema, ordering, identity, groups, and credit invariants."""

    if not records:
        raise MetricError("metric run must contain at least one record")
    identity: Mapping[str, Any] | None = None
    kind: str | None = None
    hparams: Mapping[str, Any] | None = None
    last_step = -1
    last_wall_time = -1.0
    for index, record in enumerate(records):
        _validate_record(record)
        current = record["identity"]
        if identity is None:
            identity = current
            kind = record["kind"]
            hparams = record["hparams"]
        elif current != identity:
            raise MetricError(f"record {index}: identity changed within run")
        elif record["kind"] != kind:
            raise MetricError(f"record {index}: kind changed within run")
        elif record["hparams"] != hparams:
            raise MetricError(f"record {index}: hparams changed within run")
        step = record["step"]
        wall_time = record["wall_time_s"]
        if step < last_step or wall_time < last_wall_time:
            raise MetricError(f"record {index}: step and wall time must be monotonic")
        last_step = step
        last_wall_time = wall_time


def derive_stop(record: Mapping[str, Any], zero: float = ZERO_TOLERANCE) -> dict[str, bool]:
    """Derive fail-closed stop indicators only from raw record evidence."""

    rows = record["rows"]
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["group_sha256"]].append(row)
    complete = [group for group in groups.values() if group[0]["group_complete"]]
    failure_keys = ("unsupported_hard", "judge_failed", "invalid_output")
    hard_variance = any(
        len(
            {
                row["correct"]
                for row in group
                if row["hard_valid"] and not any(row["evaluation"][key] for key in failure_keys)
            }
        )
        > 1
        for group in complete
    )
    evaluator_invalid = any(
        row["evaluation"][key] for row in rows for key in ("unsupported_hard", "judge_failed", "invalid_output")
    )
    credit_leakage = any(
        (
            not row["hard_valid"]
            or not row["correct"]
            or row["evaluation"]["unsupported_hard"]
            or row["evaluation"]["judge_failed"]
            or row["evaluation"]["invalid_output"]
        )
        and (abs(row["reward"]["quality"]) > zero or abs(row["advantage"]["quality"]) > zero)
        for row in rows
    )
    token_beta = float(record["hparams"].get("token_beta", 0.0))
    token_values = [
        relevance
        for row in rows
        for relevance, selected in zip(row["token"]["relevance"], row["token"]["mask"], strict=True)
        if selected
    ]
    token_missing = token_beta > zero and any(not row["token"]["relevance"] for row in rows)
    token_uniform = token_beta > zero and token_values and max(token_values) - min(token_values) <= zero
    indicators = {
        "format_collapse": bool(rows) and not any(row["hard_valid"] for row in rows),
        "no_hard_variance": record["kind"] == "train" and not hard_variance,
        "zero_advantage": record["kind"] == "train"
        and bool(rows)
        and all(abs(row["advantage"]["rdan"]) <= zero for row in rows),
        "clip_saturated": _ratio(record["stats"]["clip"]) >= 1.0 - zero,
        "calibration_invalid": record["evidence"]["calibration_checked"] == 0
        or record["evidence"]["calibration_failed"] > 0,
        "evaluator_invalid": evaluator_invalid,
        "credit_leakage": credit_leakage,
        "token_invalid": token_missing or token_uniform,
        "nonfinite": record["evidence"]["nonfinite_count"] > 0,
    }
    indicators["halted"] = any(indicators.values())
    return indicators


def plot_runs(paths: Sequence[str | Path], output_dir: str | Path, config_path: str | Path) -> dict[str, Any]:
    """Regenerate configured paper-style figures, a summary, and a manifest."""

    if not paths:
        raise MetricError("at least one metric run is required")
    config_file = Path(config_path)
    config = _load_config(config_file)
    sources = [(Path(path), load_records(path)) for path in paths]
    sources.sort(key=lambda item: _identity_sort_key(item[1][0]["identity"]))
    _validate_run_set([records for _, records in sources])
    runs = [records for _, records in sources]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    expected = _expected_outputs(config)
    _clear_outputs(output, expected)

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    settings = {
        "figure.figsize": (9.0, 2.8),
        "figure.dpi": config["dpi"],
        "font.size": 8,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "lines.linewidth": 1.2,
        "savefig.bbox": "tight",
        "svg.fonttype": "none",
        "svg.hashsalt": "rdan-grpo-metrics-v2",
    }
    with plt.rc_context(settings):
        for name, metrics in config["figures"].items():
            if name == "rtt_fig3_sensitivity":
                figure = _plot_sensitivity(runs, metrics, config["sensitivity_parameter"])
            else:
                figure = _plot_steps(runs, name, metrics, config["smoothing_window"], config["zero_tolerance"])
            for suffix in config["formats"]:
                target = output / f"{name}.{suffix}"
                metadata = {"Creator": "rdan-grpo", "Date": None} if suffix == "svg" else {"Software": "rdan-grpo"}
                figure.savefig(target, format=suffix, dpi=config["dpi"], metadata=metadata)
            plt.close(figure)

    summary = _summary(runs, config["zero_tolerance"])
    (output / "summary.json").write_text(_json(summary) + "\n", encoding="utf-8")
    inputs = [
        {
            "identity": records[0]["identity"],
            "bytes": source.stat().st_size,
            "sha256": _sha256(source.read_bytes()),
        }
        for source, records in sources
    ]
    manifest = _manifest(output, inputs, _sha256(config_file.read_bytes()), config["paper_reference"])
    (output / "sha256_manifest.json").write_text(_json(manifest) + "\n", encoding="utf-8")
    return manifest


def _validate_record(record: Mapping[str, Any]) -> None:
    _exact_keys(record, RECORD_KEYS, "record")
    if record["schema_version"] != SCHEMA_VERSION:
        raise MetricError(f"schema_version must equal {SCHEMA_VERSION}")
    if record["kind"] not in {"train", "eval", "sweep"}:
        raise MetricError("kind must be train, eval, or sweep")
    _validate_identity(record["identity"])
    _integer(record["step"], "step", minimum=0)
    _finite(record["wall_time_s"], "wall_time_s", minimum=0)
    if not isinstance(record["rows"], list):
        raise MetricError("rows must be a list")
    for row in record["rows"]:
        _validate_row(row)
    if record["kind"] == "train" and not record["rows"]:
        raise MetricError("train records require row metrics")
    _validate_groups(record["rows"])
    _validate_stats(record["stats"])
    _validate_eval(record["eval"])
    _validate_hparams(record["hparams"])
    _validate_evidence(record["evidence"])
    if record["kind"] == "eval" and not record["eval"]:
        raise MetricError("eval records require evaluation metrics")
    if record["kind"] == "sweep" and (not record["eval"] or not record["hparams"]):
        raise MetricError("sweep records require evaluation metrics and hyperparameters")


def _validate_identity(identity: Any) -> None:
    if not isinstance(identity, Mapping):
        raise MetricError("identity must be an object")
    _exact_keys(identity, IDENTITY_KEYS, "identity")
    for key in ("project", "run_id", "method", "stage"):
        _safe_string(identity[key], key, SAFE_NAME)
    _safe_string(identity["model_id"], "model_id", SAFE_MODEL)
    for key in ("model_revision", "tokenizer_revision", "data_revision", "code_revision"):
        _safe_string(identity[key], key, HEX_REVISION)
    for key in ("config_sha256", "protocol_sha256"):
        _safe_string(identity[key], key, SHA256)
    _integer(identity["seed"], "seed", minimum=0)


def _validate_row(row: Any) -> None:
    if not isinstance(row, Mapping):
        raise MetricError("row metric must be an object")
    _exact_keys(row, ROW_KEYS, "row")
    _safe_string(row["sample_sha256"], "sample_sha256", SHA256)
    _safe_string(row["group_sha256"], "group_sha256", SHA256)
    _integer(row["response_index"], "response_index", minimum=0)
    _integer(row["group_size"], "group_size", minimum=1)
    if row["response_index"] >= row["group_size"]:
        raise MetricError("response_index must be smaller than group_size")
    for name in ("group_complete", "hard_valid", "correct"):
        if type(row[name]) is not bool:
            raise MetricError(f"{name} must be a boolean")
    if not row["hard_valid"] and row["correct"]:
        raise MetricError("hard-invalid rows cannot be correct")
    _validate_evaluation(row["evaluation"])
    _validate_rubrics(row["rubrics"])
    _number_map(row["reward"], REWARD_KEYS, "reward")
    _number_map(row["advantage"], ADVANTAGE_KEYS, "advantage")
    _validate_generation(row["generation"])
    _validate_token(row["token"], row["generation"]["token_count"])
    if row["reward"]["aon"] not in {0, 1}:
        raise MetricError("reward.aon must be binary")
    for name in ("csr", "quality", "mix", "selected"):
        if not 0 <= row["reward"][name] <= 1:
            raise MetricError(f"reward.{name} must be in [0, 1]")
    if not -1 <= row["reward"]["signed_csr"] <= 1:
        raise MetricError("reward.signed_csr must be in [-1, 1]")


def _validate_evaluation(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise MetricError("evaluation must be an object")
    _exact_keys(value, EVALUATION_KEYS, "evaluation")
    _safe_string(value["evaluator_revision"], "evaluator_revision", SHA256)
    for name in EVALUATION_KEYS - {"evaluator_revision"}:
        if type(value[name]) is not bool:
            raise MetricError(f"evaluation.{name} must be a boolean")


def _validate_rubrics(rubrics: Any) -> None:
    if not isinstance(rubrics, list) or not rubrics:
        raise MetricError("rubrics must be a non-empty list")
    for rubric in rubrics:
        if not isinstance(rubric, Mapping):
            raise MetricError("rubric outcome must be an object")
        _exact_keys(rubric, RUBRIC_KEYS, "rubric")
        _safe_string(rubric["rubric_sha256"], "rubric_sha256", SHA256)
        if rubric["rubric_type"] not in {"hard", "soft"}:
            raise MetricError("rubric_type must be hard or soft")
        if rubric["route"] not in {"deterministic", "judge"}:
            raise MetricError("rubric route must be deterministic or judge")
        if rubric["outcome"] not in {"pass", "fail", "invalid", "unsupported"}:
            raise MetricError("rubric outcome is invalid")
        _finite(rubric["score"], "rubric.score", minimum=0)
        if rubric["score"] > 1:
            raise MetricError("rubric.score must be in [0, 1]")
        _safe_string(rubric["evaluator_revision"], "rubric.evaluator_revision", SHA256)


def _validate_generation(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise MetricError("generation must be an object")
    _exact_keys(value, GENERATION_KEYS, "generation")
    _integer(value["token_count"], "generation.token_count", minimum=0)
    for name in ("cap_hit", "aborted"):
        if type(value[name]) is not bool:
            raise MetricError(f"generation.{name} must be a boolean")
    _safe_string(value["stop_reason"], "generation.stop_reason", SAFE_NAME)
    if value["aborted"] and value["token_count"] != 0:
        raise MetricError("aborted generations must have zero tokens")


def _validate_token(value: Any, token_count: int) -> None:
    if not isinstance(value, Mapping):
        raise MetricError("token must be an object")
    _exact_keys(value, TOKEN_KEYS, "token")
    mask = value["mask"]
    relevance = value["relevance"]
    if not isinstance(mask, list) or not isinstance(relevance, list) or len(mask) != len(relevance):
        raise MetricError("token mask and relevance must be aligned lists")
    if mask and len(mask) != token_count:
        raise MetricError("token arrays must align with generation.token_count")
    if any(type(selected) is not bool for selected in mask):
        raise MetricError("token.mask values must be booleans")
    for score in relevance:
        _finite(score, "token.relevance", minimum=0)
        if score > 1:
            raise MetricError("token.relevance must be in [0, 1]")


def _validate_groups(rows: Sequence[Mapping[str, Any]]) -> None:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["group_sha256"]].append(row)
    for group in groups.values():
        size = group[0]["group_size"]
        complete = group[0]["group_complete"]
        if any(row["group_size"] != size or row["group_complete"] is not complete for row in group):
            raise MetricError("group completeness metadata changed within a group")
        indexes = [row["response_index"] for row in group]
        if len(indexes) != len(set(indexes)):
            raise MetricError("response_index must be unique within a group")
        if complete and len(group) != size:
            raise MetricError("complete group row count must equal group_size")
        if not complete and len(group) >= size:
            raise MetricError("incomplete group row count must be smaller than group_size")


def _validate_stats(stats: Any) -> None:
    if not isinstance(stats, Mapping):
        raise MetricError("stats must be an object")
    _exact_keys(stats, STAT_KEYS, "stats")
    for name, ratio in stats.items():
        _validate_ratio(ratio, f"stats.{name}")
    for name in ("entropy", "clip", "length", "throughput", "gpu_hours"):
        if stats[name]["numerator"] < 0:
            raise MetricError(f"stats.{name}.numerator must be non-negative")
    if stats["clip"]["numerator"] > stats["clip"]["denominator"]:
        raise MetricError("stats.clip must be a ratio in [0, 1]")


def _validate_eval(metrics: Any) -> None:
    if not isinstance(metrics, Mapping):
        raise MetricError("eval must be an object")
    for name, ratio in metrics.items():
        _safe_string(name, "eval metric", SAFE_NAME)
        _validate_ratio(ratio, f"eval.{name}")
        if ratio["numerator"] < 0 or ratio["numerator"] > ratio["denominator"]:
            raise MetricError(f"eval.{name} must be a ratio in [0, 1]")


def _validate_hparams(hparams: Any) -> None:
    if not isinstance(hparams, Mapping):
        raise MetricError("hparams must be an object")
    for name, value in hparams.items():
        _safe_string(name, "hyperparameter", SAFE_NAME)
        _finite(value, f"hparams.{name}")


def _validate_evidence(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise MetricError("evidence must be an object")
    _exact_keys(value, EVIDENCE_KEYS, "evidence")
    for name in EVIDENCE_KEYS:
        _integer(value[name], f"evidence.{name}", minimum=0)
    if value["calibration_failed"] > value["calibration_checked"]:
        raise MetricError("calibration failures cannot exceed checked records")


def _validate_ratio(value: Any, name: str) -> None:
    if not isinstance(value, Mapping):
        raise MetricError(f"{name} must be an object")
    _exact_keys(value, {"numerator", "denominator"}, name)
    _finite(value["numerator"], f"{name}.numerator")
    _finite(value["denominator"], f"{name}.denominator", minimum=0, exclusive=True)


def _number_map(value: Any, keys: set[str], name: str) -> None:
    if not isinstance(value, Mapping):
        raise MetricError(f"{name} must be an object")
    _exact_keys(value, keys, name)
    for key, number in value.items():
        _finite(number, f"{name}.{key}")


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise MetricError(
            f"{name} keys differ: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _safe_string(value: Any, name: str, pattern: re.Pattern[str]) -> None:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise MetricError(f"{name} has an unsafe or invalid value")


def _integer(value: Any, name: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MetricError(f"{name} must be an integer >= {minimum}")


def _finite(value: Any, name: str, *, minimum: float | None = None, exclusive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise MetricError(f"{name} must be finite")
    result = float(value)
    if minimum is not None and (result <= minimum if exclusive else result < minimum):
        operator = ">" if exclusive else ">="
        raise MetricError(f"{name} must be {operator} {minimum}")
    return result


def _load_config(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "smoothing_window",
        "zero_tolerance",
        "formats",
        "dpi",
        "eval_metrics",
        "sensitivity_parameter",
        "figures",
        "stop_indicators",
        "paper_reference",
    }
    if not isinstance(value, dict):
        raise MetricError("logging config must be an object")
    _exact_keys(value, required, "logging config")
    if value["schema_version"] != SCHEMA_VERSION:
        raise MetricError("logging schema version mismatch")
    _integer(value["smoothing_window"], "smoothing_window", minimum=1)
    if value["zero_tolerance"] != ZERO_TOLERANCE:
        raise MetricError(f"zero_tolerance must equal {ZERO_TOLERANCE}")
    _integer(value["dpi"], "dpi", minimum=72)
    if value["formats"] != ["svg", "png"]:
        raise MetricError("formats must be ['svg', 'png']")
    if value["stop_indicators"] != list(STOP_ORDER):
        raise MetricError("stop indicators differ from metric schema")
    if not isinstance(value["figures"], Mapping):
        raise MetricError("figures must be an object")
    _exact_keys(value["figures"], FIGURE_KEYS, "figures")
    if not isinstance(value["eval_metrics"], list) or not value["eval_metrics"]:
        raise MetricError("eval_metrics must be a non-empty list")
    for metric in value["eval_metrics"]:
        _safe_string(metric, "eval metric", SAFE_NAME)
    for figure, metrics in value["figures"].items():
        if not isinstance(metrics, list) or not metrics:
            raise MetricError(f"figures.{figure} must be a non-empty list")
        for metric in metrics:
            if not isinstance(metric, str):
                raise MetricError(f"figures.{figure} metrics must be strings")
            if metric.startswith("eval/") and metric.removeprefix("eval/") not in value["eval_metrics"]:
                raise MetricError(f"figures.{figure} contains an unregistered evaluation metric")
    if value["figures"]["rtt_fig3_sensitivity"] != ["macro_instruction_following"]:
        raise MetricError("RTT Figure 3 must be one instruction-following macro average")
    reference = value["paper_reference"]
    if not isinstance(reference, Mapping):
        raise MetricError("paper_reference must be an object")
    _exact_keys(reference, {"papo_fig9_process_active_final", "papo_fig9_process_active_peak"}, "paper_reference")
    if reference["papo_fig9_process_active_final"] != 0.4844:
        raise MetricError("PAPO Figure 9 released history must end at 0.4844")
    if reference["papo_fig9_process_active_peak"] != 0.5234:
        raise MetricError("PAPO Figure 9 released history peak must equal 0.5234")
    _safe_string(value["sensitivity_parameter"], "sensitivity_parameter", SAFE_NAME)
    return value


def _expected_outputs(config: Mapping[str, Any]) -> set[str]:
    names = {"summary.json", "sha256_manifest.json"}
    names.update(f"{figure}.{suffix}" for figure in config["figures"] for suffix in config["formats"])
    return names


def _clear_outputs(output: Path, expected: set[str]) -> None:
    unexpected = sorted(path.name for path in output.iterdir() if path.name not in expected)
    if unexpected:
        raise MetricError(f"output directory contains unexpected files: {unexpected}")
    for name in expected:
        path = output / name
        if path.is_dir():
            raise MetricError(f"output target is a directory: {name}")
        path.unlink(missing_ok=True)


def _validate_run_set(runs: Sequence[Sequence[Mapping[str, Any]]]) -> None:
    identities = [run[0]["identity"] for run in runs]
    immutable = [_identity_sort_key(identity) for identity in identities]
    if len(immutable) != len(set(immutable)):
        raise MetricError("duplicate immutable run identity")
    seen: set[tuple[tuple[Any, ...], int]] = set()
    for identity in identities:
        key = (_aggregate_key(identity), identity["seed"])
        if key in seen:
            raise MetricError("aggregate contains a duplicate seed")
        seen.add(key)


def _plot_steps(
    runs: Sequence[Sequence[Mapping[str, Any]]], name: str, metrics: Sequence[str], window: int, zero: float
):
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, len(metrics), squeeze=False, figsize=(max(3.0 * len(metrics), 6.0), 2.8))
    for metric_index, metric in enumerate(metrics):
        axis = axes[0][metric_index]
        grouped: dict[tuple[Any, ...], list[tuple[int, list[tuple[int, float]]]]] = defaultdict(list)
        labels: dict[tuple[Any, ...], str] = {}
        for run in runs:
            identity = run[0]["identity"]
            points = [(record["step"], _metric(record, metric, zero)) for record in run]
            points = [(step, value) for step, value in points if value is not None]
            if not points:
                continue
            key = _aggregate_key(identity)
            grouped[key].append((identity["seed"], points))
            labels[key] = _series_name(identity, run[0]["hparams"])
        color_map = plt.get_cmap("tab10")
        for group_index, (key, traces) in enumerate(sorted(grouped.items())):
            color = color_map(group_index % 10)
            label = labels[key]
            for seed, points in traces:
                axis.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    color=color,
                    alpha=0.22,
                    label=f"{label} s{seed} raw",
                )
            if len(traces) < 2:
                continue
            seeds = [seed for seed, _ in traces]
            if len(seeds) != len(set(seeds)):
                raise MetricError("aggregate contains a duplicate seed")
            common = sorted(set.intersection(*(set(step for step, _ in points) for _, points in traces)))
            means: list[float] = []
            deviations: list[float] = []
            for step in common:
                values = [dict(points)[step] for _, points in traces]
                means.append(fmean(values))
                deviations.append(stdev(values))
            axis.plot(common, means, color=color, linewidth=1.8, label=f"{label} mean raw")
            axis.plot(
                common,
                _smooth(means, window),
                color=color,
                linestyle="--",
                linewidth=1.8,
                label=f"{label} mean smoothed",
            )
            axis.fill_between(
                common,
                [mean - deviation for mean, deviation in zip(means, deviations, strict=True)],
                [mean + deviation for mean, deviation in zip(means, deviations, strict=True)],
                color=color,
                alpha=0.12,
                label=f"{label} raw sample SD",
            )
        axis.set_title(metric)
        axis.set_xlabel("optimizer step")
        if not axis.lines:
            raise MetricError(f"{name}: no data for {metric}")
    _finish_figure(figure, axes[0], name)
    return figure


def _plot_sensitivity(runs: Sequence[Sequence[Mapping[str, Any]]], metrics: Sequence[str], parameter: str):
    from matplotlib import pyplot as plt

    metric = metrics[0]
    figure, axis = plt.subplots(1, 1, figsize=(6.0, 2.8))
    points: dict[tuple[Any, ...], dict[float, list[tuple[int, str, float]]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        latest = run[-1]
        if parameter not in latest["hparams"]:
            continue
        value = _metric(latest, metric, ZERO_TOLERANCE)
        if value is None:
            continue
        identity = latest["identity"]
        sweep_key = _sweep_key(identity, latest["hparams"], parameter)
        points[sweep_key][float(latest["hparams"][parameter])].append(
            (identity["seed"], identity["config_sha256"], value)
        )
    for sweep_key, by_parameter in sorted(points.items()):
        xs: list[float] = []
        means: list[float] = []
        deviations: list[float] = []
        label = _sweep_label(sweep_key)
        for x, values in sorted(by_parameter.items()):
            seeds = [seed for seed, _, _ in values]
            configs = {config for _, config, _ in values}
            if len(seeds) != len(set(seeds)):
                raise MetricError("sensitivity point contains a duplicate seed")
            if len(configs) != 1:
                raise MetricError("sensitivity point mixes config identities")
            raw = [value for _, _, value in values]
            for seed, _, value in values:
                point_label = f"{label} seed {seed} raw"
                if x == 0:
                    point_label += " response-only beta=0"
                axis.scatter([x], [value], alpha=0.3, label=point_label)
            xs.append(x)
            means.append(fmean(raw))
            deviations.append(stdev(raw) if len(raw) > 1 else 0.0)
        axis.plot(xs, means, marker="o", linewidth=1.8, label=f"{label} mean raw")
        if any(deviations):
            axis.fill_between(
                xs,
                [mean - deviation for mean, deviation in zip(means, deviations, strict=True)],
                [mean + deviation for mean, deviation in zip(means, deviations, strict=True)],
                alpha=0.12,
                label=f"{label} raw sample SD",
            )
    axis.set_title("IFEval + IFBench + MulDimIF macro average")
    axis.set_xlabel(parameter)
    axis.set_ylabel("macro accuracy")
    if not axis.lines:
        raise MetricError("rtt_fig3_sensitivity: no data for macro_instruction_following")
    _finish_figure(figure, [axis], "rtt_fig3_sensitivity")
    return figure


def _finish_figure(figure: Any, axes: Sequence[Any], name: str) -> None:
    handles: list[Any] = []
    labels: list[str] = []
    for axis in axes:
        found_handles, found_labels = axis.get_legend_handles_labels()
        for handle, label in zip(found_handles, found_labels, strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    figure.suptitle(name)
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(4, max(1, len(labels))),
        fontsize=6,
    )
    figure.subplots_adjust(bottom=0.38, top=0.80, wspace=0.32)


def _metric(record: Mapping[str, Any], name: str, zero: float) -> float | None:
    if name == "macro_instruction_following":
        values = [record["eval"].get(metric) for metric in ("ifeval", "ifbench", "muldimif")]
        return fmean(_ratio(value) for value in values if value) if all(values) else None
    if name.startswith("eval/"):
        ratio = record["eval"].get(name.removeprefix("eval/"))
        return _ratio(ratio) if ratio else None
    if name in STAT_KEYS:
        return _ratio(record["stats"][name])
    rows = record["rows"]
    if not rows:
        return None
    failure_keys = EVALUATION_KEYS - {"evaluator_revision"}
    if name.startswith("reward_"):
        return fmean(row["reward"][name.removeprefix("reward_")] for row in rows)
    advantages = [row["advantage"]["rdan"] for row in rows]
    if name == "rollout_accuracy":
        valid = [row for row in rows if row["hard_valid"] and not any(row["evaluation"][key] for key in failure_keys)]
        return fmean(float(row["correct"]) for row in valid) if valid else None
    if name == "adv_zero_ratio":
        return fmean(float(abs(value) <= zero) for value in advantages)
    if name == "adv_positive_ratio":
        return fmean(float(value > zero) for value in advantages)
    if name == "adv_std":
        return _sample_sd(advantages)
    if name in {"correct_adv_min", "correct_adv_std", "correct_adv_mean", "wrong_adv_mean"}:
        selected_correct = not name.startswith("wrong")
        selected = [row["advantage"]["rdan"] for row in rows if row["correct"] is selected_correct]
        if not selected:
            return None
        if name.endswith("min"):
            return min(selected)
        if name.endswith("std"):
            return _sample_sd(selected)
        return fmean(selected)
    if name == "outcome_adv_std":
        return _sample_sd([row["advantage"]["response"] for row in rows])
    if name == "process_adv_nonzero_std":
        return _sample_sd([row["advantage"]["quality"] for row in rows if abs(row["advantage"]["quality"]) > zero])
    if name == "process_active_group_ratio":
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[row["group_sha256"]].append(row)
        complete = [group for group in groups.values() if group[0]["group_complete"]]
        if not complete:
            return None
        active = 0
        for group in complete:
            correct = [
                row["advantage"]["quality"]
                for row in group
                if row["correct"] and row["hard_valid"] and not any(row["evaluation"][key] for key in failure_keys)
            ]
            active += len(correct) >= 2 and max(correct) - min(correct) > zero
        return active / len(complete)
    raise MetricError(f"unknown plotted metric: {name}")


def _ratio(value: Mapping[str, float]) -> float:
    return value["numerator"] / value["denominator"]


def _sample_sd(values: Sequence[float]) -> float | None:
    return stdev(values) if len(values) > 1 else None


def _smooth(values: Sequence[float], window: int) -> list[float]:
    return [fmean(values[max(0, index - window + 1) : index + 1]) for index in range(len(values))]


def _identity_sort_key(identity: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(identity[key] for key in sorted(IDENTITY_KEYS))


def _aggregate_key(identity: Mapping[str, Any]) -> tuple[Any, ...]:
    keys = sorted(IDENTITY_KEYS - {"run_id", "seed"})
    return tuple(identity[key] for key in keys)


def _aggregate_label(key: tuple[Any, ...]) -> str:
    keys = sorted(IDENTITY_KEYS - {"run_id", "seed"})
    identity = dict(zip(keys, key, strict=True))
    return f"{identity['method']} {identity['stage']} cfg={identity['config_sha256'][:6]}"


def _series_name(identity: Mapping[str, Any], hparams: Mapping[str, float]) -> str:
    name = _aggregate_label(_aggregate_key(identity))
    if not hparams:
        return name
    values = ",".join(f"{key}={hparams[key]:g}" for key in sorted(hparams))
    return f"{name}[{values}]"


def _sweep_key(identity: Mapping[str, Any], hparams: Mapping[str, float], parameter: str) -> tuple[Any, ...]:
    identity_keys = sorted(IDENTITY_KEYS - {"run_id", "seed", "config_sha256"})
    fixed_hparams = tuple((key, hparams[key]) for key in sorted(hparams) if key != parameter)
    return tuple(identity[key] for key in identity_keys) + (fixed_hparams,)


def _sweep_label(key: tuple[Any, ...]) -> str:
    identity_keys = sorted(IDENTITY_KEYS - {"run_id", "seed", "config_sha256"})
    identity = dict(zip(identity_keys, key[:-1], strict=True))
    return f"{identity['method']} {identity['stage']}"


def _summary(runs: Sequence[Sequence[Mapping[str, Any]]], zero: float) -> dict[str, Any]:
    summaries = []
    for run in runs:
        latest = run[-1]
        summaries.append(
            {
                "identity": latest["identity"],
                "records": len(run),
                "last_step": latest["step"],
                "last_metrics": {
                    name: _metric(latest, name, zero)
                    for name in (
                        "reward_selected",
                        "rollout_accuracy",
                        "adv_zero_ratio",
                        "entropy",
                        "kl",
                        "clip",
                        "length",
                        "process_active_group_ratio",
                    )
                    if latest["rows"]
                },
                "eval": {name: _ratio(value) for name, value in sorted(latest["eval"].items())},
                "stop_events": [
                    {
                        "step": record["step"],
                        "indicators": [key for key in STOP_ORDER if derive_stop(record, zero)[key]],
                    }
                    for record in run
                    if derive_stop(record, zero)["halted"]
                ],
            }
        )
    return {"schema_version": SCHEMA_VERSION, "runs": summaries}


def _manifest(
    output: Path, inputs: Sequence[Mapping[str, Any]], config_sha256: str, paper_reference: Mapping[str, Any]
) -> dict[str, Any]:
    files = []
    for path in sorted(output.iterdir(), key=lambda value: value.name):
        if path.is_file() and path.name != "sha256_manifest.json":
            data = path.read_bytes()
            files.append({"path": path.name, "bytes": len(data), "sha256": _sha256(data)})
    return {
        "algorithm": "sha256",
        "schema_version": SCHEMA_VERSION,
        "config_sha256": config_sha256,
        "inputs": list(inputs),
        "paper_reference": dict(paper_reference),
        "files": files,
    }


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
