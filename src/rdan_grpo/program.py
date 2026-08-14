"""Fail-closed validation for the Qwen-first experiment contract."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from rdan_grpo.evaluator_cert import EvaluatorCertificationError, scalar_evaluator_certificate
from rdan_grpo.judge import build_request as build_openrouter_request
from rdan_grpo.judge import select_reasoning_effort
from rdan_grpo.response_identity import ResponseIdentityError, response_data_identity
from rdan_grpo.runtime_parity import GENERATION_SOURCE_IDENTITY
from rdan_grpo.scalar_data import ScalarDataError, inspect_scalar_gate
from rdan_grpo.vllm_runtime_parity import VLLMParityError, validate_vllm_runtime_parity

JsonObject = dict[str, Any]

MODEL_ID = "qwen3_4b"
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
METHOD_IDS = (
    "base",
    "rl_aon",
    "rl_csr",
    "rl_mix",
    "rtt_papo_response",
    "sft",
    "dpo",
    "rtt_aon",
    "rtt_csr",
    "rdan_full",
)
TUNING_SEED = 240520
CONFIRMATION_SEEDS = (240521, 240522, 240523)
HIR_SOURCE_SHA256 = "465a01c19dc29e2c8d1cf183ccf3135872f7ec94ef10b20b7eb35603164c183b"
HIR_TAXONOMY_SHA256 = "21cbabc011bad4637f78deea38d648ed3b1b740b7f2a57d64f38e73d8b3406ca"
JUDGE_CALIBRATION_CASES_PATH = "configs/judges/qwen_judge_calibration_cases.jsonl"
RTT_REVISION = "b1ab2fba9bece98674e5fa6e6c808d9d63235778"
LUNA_REVISION = "openai/gpt-5.6-luna-20260709"
SELECTION_RULE = "maximum_score_then_lowest_candidate"

_SECRET_PATTERNS = (
    re.compile(r"\bsk-(?:or-v1-)?[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bwandb_v1_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)
_CREDENTIAL_FIELD = re.compile(
    r"(?:^|_)(?:api_key|apikey|password|passwd|secret|access_token|auth_token|credential|credentials)$"
)
_PROVIDER_TOKEN_FIELD = re.compile(r"(?:^|_)(?:wandb|hf|huggingface|openrouter|openai|github)_token$")
ARTIFACT_ROOT_ENV = "RDAN_ARTIFACT_ROOT"
_JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["rubrics"],
    "properties": {
        "rubrics": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "score", "reason"],
                "properties": {
                    "id": {"type": "integer", "minimum": 1},
                    "score": {"type": "integer", "enum": [-1, 1]},
                    "reason": {"type": "string", "minLength": 1},
                },
            },
        }
    },
}
_METHOD_OBJECTIVES = (
    {"kind": "none"},
    {
        "kind": "rl",
        "response_reward": "aon",
        "response_normalization": "group_once",
        "token_beta": 0,
        "quality": "off",
        "quality_weight": 0,
    },
    {
        "kind": "rl",
        "response_reward": "csr",
        "response_normalization": "group_once",
        "token_beta": 0,
        "quality": "off",
        "quality_weight": 0,
    },
    {
        "kind": "rl",
        "response_reward": "aon_csr_raw_mix",
        "response_mix": {
            "aon_weight_source": "dev_selection",
            "aon_weight_candidates": [0.25, 0.5, 0.75],
            "csr_weight": "one_minus_aon",
        },
        "response_normalization": "group_once_after_raw_mix",
        "token_beta": 0,
        "quality": "off",
        "quality_weight": 0,
    },
    {
        "kind": "rl",
        "response_reward": "aon_with_vacuous_soft_only_gate",
        "response_normalization": "group_once",
        "token_beta": 0,
        "quality": "hard_valid_conditional_with_soft_only_extension",
        "quality_weight": {"source": "dev_selection", "candidates": [0.25, 0.5, 1.0]},
    },
    {"kind": "sft"},
    {"kind": "dpo"},
    {
        "kind": "rl",
        "response_reward": "aon",
        "response_normalization": "group_once",
        "token_beta": 0.5,
        "quality": "off",
        "quality_weight": 0,
    },
    {
        "kind": "rl",
        "response_reward": "csr",
        "response_normalization": "group_once",
        "token_beta": 0.5,
        "quality": "off",
        "quality_weight": 0,
    },
    {
        "kind": "rl",
        "response_reward": "aon",
        "response_normalization": "group_once",
        "token_beta": 0.5,
        "quality": "hard_valid_conditional",
        "quality_weight": {"source": "dev_selection", "candidates": [0.25, 0.5, 1.0]},
    },
)


class ProgramContractError(ValueError):
    """Raised when an experiment contract is incomplete or inconsistent."""


@dataclass(frozen=True)
class ProgramBundle:
    """The program contract and its pinned model, judge, and compute configs."""

    program: JsonObject
    targets: JsonObject
    model: JsonObject
    judge: JsonObject
    judge_prompt: str
    compute: JsonObject
    evaluation: JsonObject
    data: JsonObject
    taxonomy: JsonObject
    baselines: JsonObject
    artifacts: JsonObject
    hard_route_exclusions: JsonObject | None
    baseline_data_artifacts: JsonObject
    dev_split_artifact: JsonObject | None
    selection_artifact: JsonObject | None
    route_resolution: JsonObject
    route_implementation: JsonObject | None
    evaluator_certificate: JsonObject | None
    lifecycle_artifacts: JsonObject
    artifact_hashes: JsonObject
    repo_root: Path


def check_program(path: str | Path) -> ProgramBundle:
    """Load and validate a program and every referenced config."""
    bundle = load_program(path)
    validate_program(bundle)
    return bundle


def load_program(path: str | Path) -> ProgramBundle:
    """Load a program and its referenced configs without accepting duplicate keys."""
    program_path = Path(path).resolve()
    program = _load_json(program_path)
    _exact_keys(
        program,
        {
            "schema_version",
            "id",
            "targets_config",
            "model_config",
            "judge_config",
            "compute_config",
            "eval_config",
            "data_config",
            "taxonomy_config",
            "baseline_configs",
            "artifact_config",
            "launch_train_config",
            "same_backend_configs",
            "seeds",
            "selection",
            "hard_route_policy",
            "lifecycle_artifacts",
            "methods",
            "execution_priority",
            "rl_recipe",
            "readiness",
            "benchmarks",
            "ood_evaluation",
            "pilot",
            "counts",
        },
        "program",
    )
    config_root = program_path.parent.parent
    targets_path = _resolve_under(
        config_root,
        program_path.parent / _string(program["targets_config"], "program.targets_config"),
        "program.targets_config",
    )
    model_path = _resolve_under(
        config_root,
        program_path.parent / _string(program["model_config"], "program.model_config"),
        "program.model_config",
    )
    judge_path = _resolve_under(
        config_root,
        program_path.parent / _string(program["judge_config"], "program.judge_config"),
        "program.judge_config",
    )
    compute_path = _resolve_under(
        config_root,
        program_path.parent / _string(program["compute_config"], "program.compute_config"),
        "program.compute_config",
    )
    eval_path = _resolve_under(
        config_root,
        program_path.parent / _string(program["eval_config"], "program.eval_config"),
        "program.eval_config",
    )
    data_path = _config_path(config_root, program_path, program, "data_config")
    taxonomy_path = _config_path(config_root, program_path, program, "taxonomy_config")
    artifact_path = _config_path(config_root, program_path, program, "artifact_config")
    baseline_refs = _object(program["baseline_configs"], "program.baseline_configs")
    _exact_keys(baseline_refs, {"sft", "dpo"}, "program.baseline_configs")
    baseline_paths = {
        name: _resolve_under(
            config_root,
            program_path.parent / _string(reference, f"program.baseline_configs.{name}"),
            f"program.baseline_configs.{name}",
        )
        for name, reference in baseline_refs.items()
    }
    exclusion_ref = _object(
        _object(program["hard_route_policy"], "program.hard_route_policy")["exclusion_manifest"],
        "program.hard_route_policy.exclusion_manifest",
    )
    exclusions = None
    repo_root = config_root.parent
    hashes: JsonObject = {}

    def load_frozen(reference: JsonObject, label: str) -> JsonObject | None:
        if reference.get("status") != "frozen":
            return None
        artifact_path = _resolve_under(
            repo_root,
            repo_root / _string(reference.get("path"), f"{label}.path"),
            f"{label}.path",
        )
        hashes[label] = _file_sha256(artifact_path)
        return _load_json(artifact_path)

    exclusions = load_frozen(exclusion_ref, "hard_route_exclusions")
    hard_policy = _object(program["hard_route_policy"], "program.hard_route_policy")
    route_resolution = load_frozen(
        _object(hard_policy["route_resolution"], "program.hard_route_policy.route_resolution"),
        "route_resolution",
    )
    _expect(route_resolution is not None, "route resolution artifact must be frozen and loaded")
    implementation = load_frozen(
        _object(hard_policy["implementation_manifest"], "program.hard_route_policy.implementation_manifest"),
        "route_implementation",
    )
    certificate = load_frozen(
        _object(hard_policy["evaluator_certificate"], "program.hard_route_policy.evaluator_certificate"),
        "evaluator_certificate",
    )
    lifecycle_refs = _object(program["lifecycle_artifacts"], "program.lifecycle_artifacts")
    lifecycle_artifacts = {
        name: artifact
        for name, reference in lifecycle_refs.items()
        if (
            artifact := load_frozen(
                _object(reference, f"program.lifecycle_artifacts.{name}"),
                f"lifecycle_{name}",
            )
        )
        is not None
    }
    selection = _object(program["selection"], "program.selection")
    dev_split = load_frozen(_object(selection["dev_split"], "program.selection.dev_split"), "dev_split")
    selected = load_frozen(
        _object(selection["immutable_artifact"], "program.selection.immutable_artifact"),
        "selection",
    )
    baseline_artifacts: JsonObject = {}
    for name, baseline_path in baseline_paths.items():
        baseline = _load_json(baseline_path)
        if _object(baseline["data"], f"baselines.{name}.data").get("status") == "frozen":
            reference = {
                "status": "frozen",
                "path": f"configs/artifacts/{name}_data_manifest.json",
            }
            artifact = load_frozen(reference, f"baseline_{name}")
            _expect(artifact is not None, f"frozen {name} data manifest must be loaded")
            baseline_artifacts[name] = artifact
    judge = _load_json(judge_path)
    prompt_ref = _object(judge.get("prompt"), "judge.prompt")
    prompt_path = _resolve_under(
        judge_path.parent,
        judge_path.parent / _string(prompt_ref.get("path"), "judge.prompt.path"),
        "judge.prompt.path",
    )
    return ProgramBundle(
        program=program,
        targets=_load_json(targets_path),
        model=_load_json(model_path),
        judge=judge,
        judge_prompt=_load_text(prompt_path),
        compute=_load_json(compute_path),
        evaluation=_load_json(eval_path),
        data=_load_json(data_path),
        taxonomy=_load_json(taxonomy_path),
        baselines={name: _load_json(path) for name, path in baseline_paths.items()},
        artifacts=_load_json(artifact_path),
        hard_route_exclusions=exclusions,
        baseline_data_artifacts=baseline_artifacts,
        dev_split_artifact=dev_split,
        selection_artifact=selected,
        route_resolution=route_resolution,
        route_implementation=implementation,
        evaluator_certificate=certificate,
        lifecycle_artifacts=lifecycle_artifacts,
        artifact_hashes=hashes,
        repo_root=repo_root,
    )


def validate_program(bundle: ProgramBundle) -> None:
    """Validate all Qwen-first experiment, provisioning, and artifact invariants."""
    for name, config in (
        ("program", bundle.program),
        ("targets", bundle.targets),
        ("model", bundle.model),
        ("judge", bundle.judge),
        ("judge_prompt", bundle.judge_prompt),
        ("compute", bundle.compute),
        ("evaluation", bundle.evaluation),
        ("data", bundle.data),
        ("taxonomy", bundle.taxonomy),
        ("baselines", bundle.baselines),
        ("artifacts", bundle.artifacts),
        ("hard_route_exclusions", bundle.hard_route_exclusions),
        ("baseline_data_artifacts", bundle.baseline_data_artifacts),
        ("dev_split_artifact", bundle.dev_split_artifact),
        ("selection_artifact", bundle.selection_artifact),
        ("route_resolution", bundle.route_resolution),
        ("route_implementation", bundle.route_implementation),
        ("evaluator_certificate", bundle.evaluator_certificate),
        ("lifecycle_artifacts", bundle.lifecycle_artifacts),
    ):
        _reject_secrets(config, name)
    _validate_targets(bundle.targets)
    _validate_model(bundle.model)
    _validate_judge(bundle.judge, bundle.judge_prompt)
    _validate_compute(bundle.compute)
    _validate_evaluation(bundle.evaluation)
    _validate_data(bundle.data, bundle.taxonomy)
    _validate_baselines(bundle.baselines, bundle.baseline_data_artifacts, bundle.artifact_hashes, bundle.repo_root)
    _validate_artifacts(bundle.artifacts)
    _validate_experiment(
        bundle.program,
        bundle.dev_split_artifact,
        bundle.selection_artifact,
        bundle.artifact_hashes,
        bundle.repo_root,
    )
    _validate_hard_route_policy(
        _object(bundle.program["hard_route_policy"], "program.hard_route_policy"),
        bundle.program["readiness"],
        bundle.route_resolution,
        bundle.route_implementation,
        bundle.evaluator_certificate,
        bundle.hard_route_exclusions,
        bundle.artifact_hashes,
        bundle.repo_root,
        _object(bundle.program["lifecycle_artifacts"], "program.lifecycle_artifacts")["scalar_data"],
    )
    _validate_lifecycle_artifacts(bundle)
    _validate_cross(bundle)


def require_launch_gate(
    program_path: str | Path,
    train_config_path: str | Path,
    certificate_path: str | Path,
    rtt_root: str | Path,
) -> ProgramBundle:
    """Validate every launch prerequisite before any RTT or ROLL import."""
    bundle = check_program(program_path)
    refs = _object(bundle.program["lifecycle_artifacts"], "program.lifecycle_artifacts")
    required = {
        "scalar_data",
        "response_data",
        "judge_calibration",
        "runtime_parity",
        "vllm_runtime_parity",
        "no_update",
    }
    missing = sorted(name for name in required if refs[name]["status"] != "frozen")
    _expect(not missing, f"training launch is blocked by pending lifecycle artifacts: {missing}")
    _expect(bundle.program["readiness"]["scalar_training"] == "ready", "hard evaluator certification is not ready")
    _expect(bundle.program["readiness"]["judge"] == "ready", "judge calibration is not ready")
    _expect(bundle.program["readiness"]["launch"] == "ready", "training launch gate is not ready")

    train_config = Path(train_config_path).resolve()
    certificate = Path(certificate_path).resolve()
    _expect(
        _file_sha256(train_config) == bundle.program["launch_train_config"]["sha256"], "train config hash mismatch"
    )
    no_update_ref = refs["no_update"]
    expected_certificate = _resolve_under(
        bundle.repo_root,
        bundle.repo_root / _string(no_update_ref["path"], "no-update certificate path"),
        "no-update certificate path",
    )
    _expect(certificate == expected_certificate, "launch certificate path does not match the frozen program reference")
    _expect(_file_sha256(certificate) == no_update_ref["sha256"], "no-update certificate byte hash mismatch")
    _validate_clean_rtt_checkout(Path(rtt_root), RTT_REVISION)
    return bundle


def build_judge_request(
    judge: JsonObject,
    prompt: str,
    instruction: str,
    response: str,
    rubrics: list[JsonObject],
    seed: int,
    reasoning_effort: str = "none",
) -> JsonObject:
    """Build an offline-verifiable OpenRouter request without reading credentials."""
    _validate_judge(judge, prompt)
    _expect(isinstance(instruction, str) and isinstance(response, str), "judge request text must be strings")
    _expect(isinstance(rubrics, list) and rubrics, "judge request rubrics must be a non-empty array")
    _expect(isinstance(seed, int) and not isinstance(seed, bool), "judge request seed must be an integer")
    _expect(reasoning_effort in judge["calibration"]["reasoning_effort_candidates"], "judge effort is invalid")
    request = build_openrouter_request(judge, prompt, instruction, response, rubrics, seed, reasoning_effort)
    request["provider"] = judge["routing"]
    return request


def resolve_baseline_training(baseline: JsonObject, run_seed: int) -> JsonObject:
    """Resolve an isolated SFT or DPO training config for one confirmation run."""
    baseline_id = baseline.get("id")
    names = {"sft_reconstructed": "sft", "dpo_reconstructed": "dpo"}
    _expect(baseline_id in names, "baseline id is invalid")
    name = names[baseline_id]
    _validate_baseline(_object(baseline, f"baselines.{name}"), name)
    _expect(
        isinstance(run_seed, int) and not isinstance(run_seed, bool) and run_seed in CONFIRMATION_SEEDS,
        "baseline run seed must be a program confirmation seed",
    )
    training = deepcopy(baseline["training"])
    training["seed"] = run_seed
    return training


def _validate_model(model: JsonObject) -> None:
    _exact_keys(
        model,
        {
            "schema_version",
            "id",
            "source",
            "model",
            "revision",
            "architecture",
            "runtime",
            "tokenizer",
            "chat_template",
        },
        "model",
    )
    _expect(model["schema_version"] == 1, "model.schema_version must be 1")
    _expect(model["id"] == MODEL_ID, f"model.id must be {MODEL_ID}")
    _expect(model["source"] == "huggingface", "model.source must be huggingface")
    _expect(model["model"] == MODEL_NAME, f"model.model must be {MODEL_NAME}")
    _expect(model["revision"] == MODEL_REVISION, "model revision is not pinned to the approved commit")
    _expect(
        model["architecture"]
        == {
            "parameters": 4_022_468_096,
            "dtype": "bfloat16",
            "layers": 36,
            "hidden_size": 2560,
            "attention_heads": 32,
            "kv_heads": 8,
            "head_dim": 128,
            "native_context": 262_144,
        },
        "model architecture snapshot is invalid",
    )
    _expect(
        model["runtime"]
        == {
            "transformers_minimum": "4.51.0",
            "vllm_minimum": "0.8.5",
            "rtt_released_vllm": "0.8.4",
            "status": "requires_parity_proven_runtime_lock",
        },
        "model runtime contract is invalid",
    )
    tokenizer = _object(model["tokenizer"], "model.tokenizer")
    _exact_keys(tokenizer, {"model", "revision", "use_fast"}, "model.tokenizer")
    _expect(
        tokenizer == {"model": MODEL_NAME, "revision": MODEL_REVISION, "use_fast": True},
        "tokenizer must match the model pin",
    )
    template = _object(model["chat_template"], "model.chat_template")
    _exact_keys(template, {"source", "enable_thinking"}, "model.chat_template")
    _expect(template == {"source": "pinned_tokenizer", "enable_thinking": False}, "chat template contract is invalid")


def _validate_targets(targets: JsonObject) -> None:
    _exact_keys(targets, {"schema_version", "checked_on", "models"}, "targets")
    _expect(targets["schema_version"] == 1 and targets["checked_on"] == "2026-08-13", "target registry is invalid")
    _expect(
        targets["models"]
        == [
            {
                "id": "qwen3_4b",
                "model": MODEL_NAME,
                "revision": MODEL_REVISION,
                "priority": 1,
                "lane": "complete_qwen_first_program",
            },
            {
                "id": "llama3_2_3b",
                "model": "meta-llama/Llama-3.2-3B-Instruct",
                "revision": "0cb88a4f764b7a12671c53f0838cd831a0843b95",
                "priority": 2,
                "lane": "transfer_after_access_and_backend_parity",
            },
            {
                "id": "granite3_1_2b",
                "model": "ibm-granite/granite-3.1-2b-instruct",
                "revision": "bbc2aed595bd38bd770263dc3ab831db9794441d",
                "priority": 3,
                "lane": "transfer_after_hf_or_fsdp_backend_parity",
            },
            {
                "id": "ministral3_3b",
                "model": "mistralai/Ministral-3-3B-Instruct-2512",
                "revision": "b35d4dfe56c142746f54dbd64f579faab2744308",
                "priority": 4,
                "lane": "evaluation_only_until_runtime_migration",
            },
        ],
        "target model registry is invalid",
    )


def _validate_judge(judge: JsonObject, prompt: str) -> None:
    _exact_keys(
        judge,
        {
            "schema_version",
            "id",
            "provider",
            "endpoint",
            "catalog_url",
            "endpoints_url",
            "model",
            "expected_canonical_slug",
            "catalog_checked_on",
            "api_key_env",
            "prompt",
            "required_parameters",
            "catalog_snapshot",
            "seed_source",
            "request",
            "generation_metadata_poll",
            "calibration",
            "load_plan",
            "routing",
            "preflight",
            "result_validation",
            "response_format",
        },
        "judge",
    )
    _expect(judge["schema_version"] == 1 and judge["id"] == "openrouter_luna", "judge identity is invalid")
    _expect(judge["provider"] == "openrouter", "judge provider must be openrouter")
    _expect(
        judge["endpoint"] == "https://openrouter.ai/api/v1/chat/completions",
        "judge endpoint is invalid",
    )
    _expect(judge["catalog_url"] == "https://openrouter.ai/api/v1/models", "judge catalog URL is invalid")
    _expect(
        judge["endpoints_url"] == "https://openrouter.ai/api/v1/models/openai/gpt-5.6-luna/endpoints",
        "judge endpoints URL is invalid",
    )
    _expect(judge["model"] == "openai/gpt-5.6-luna", "judge model is invalid")
    _expect(judge["expected_canonical_slug"] == "openai/gpt-5.6-luna-20260709", "judge canonical slug is invalid")
    _expect(judge["catalog_checked_on"] == "2026-08-13", "judge catalog check date is invalid")
    _expect(judge["api_key_env"] == "OPENROUTER_API_KEY", "judge must use only OPENROUTER_API_KEY")
    _expect(
        judge["prompt"]
        == {
            "path": "rubric_prompt.txt",
            "sha256": "4ba60be95b42c143ddfa750220d35c0f93a547fcc33e078c73c30cee69512552",
        },
        "judge prompt pin is invalid",
    )
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    _expect(digest == judge["prompt"]["sha256"], "judge prompt hash does not match its pin")
    _expect(
        all(prompt.count(field) == 1 for field in ("{{instruction}}", "{{response}}", "{{rubrics_json}}")),
        "judge prompt placeholders are invalid",
    )
    _expect(
        judge["required_parameters"] == ["max_tokens", "reasoning_effort", "response_format", "seed"],
        "judge required parameters are invalid",
    )
    _expect(
        judge["catalog_snapshot"]
        == {
            "context_length": 1_050_000,
            "max_completion_tokens": 128_000,
            "pricing_per_token": {
                "prompt": "0.0000001",
                "completion": "0.0000006",
                "input_cache_read": "0.00000001",
                "input_cache_write": "0.000000125",
            },
            "supported_parameters": [
                "include_reasoning",
                "max_completion_tokens",
                "max_tokens",
                "reasoning",
                "reasoning_effort",
                "response_format",
                "seed",
                "structured_outputs",
                "tool_choice",
                "tools",
            ],
        },
        "judge catalog snapshot is invalid",
    )
    _expect(judge["seed_source"] == "program_seed", "judge seed must come from the program seed")
    _expect(
        judge["request"]
        == {
            "max_tokens": 2048,
            "reasoning_effort_source": "calibration_certificate",
            "one_call_per_response": True,
            "include_all_soft_rubrics": True,
        },
        "judge request contract is invalid",
    )
    _expect(
        judge["generation_metadata_poll"] == {"attempts": 31, "interval_seconds": 1},
        "judge generation metadata poll contract is invalid",
    )
    _expect(
        judge["calibration"]
        == {
            "reasoning_effort_candidates": ["none", "low", "medium"],
            "total_calls": 200,
            "total_cases": 76,
            "debug_canary_cases": 1,
            "frozen_labeled_cases": 49,
            "heldout_cases": 26,
            "calls_per_labeled_case": 3,
            "heldout_duplicates": 2,
            "selection_rule": "paired_bootstrap_noninferiority_then_lowest_effort",
            "bootstrap_samples": 10_000,
            "bootstrap_seed": 240520,
            "noninferiority_margin": -0.02,
            "freeze_selected_effort_before_training": True,
        },
        "judge calibration contract is invalid",
    )
    _expect(
        judge["load_plan"]
        == {
            "rollouts_per_rl_run": 256_000,
            "hir_rows": 16_968,
            "hir_rows_with_soft_rubrics": 6_362,
            "expected_soft_judge_calls_per_rl_run_ceiling": 95_985,
            "maximum_judge_calls_per_rl_run": 256_000,
            "measure_before_training": [
                "input_tokens",
                "output_tokens",
                "p50_latency_seconds",
                "p95_latency_seconds",
                "invalid_rate",
                "retry_rate",
            ],
        },
        "judge load plan is invalid",
    )
    _expect(
        judge["routing"]
        == {
            "order": ["openai"],
            "require_parameters": True,
            "allow_fallbacks": False,
            "data_collection": "deny",
            "zdr": False,
        },
        "judge routing must fail closed",
    )
    _expect(
        judge["preflight"]
        == {
            "catalog_canonical_slug_must_equal": "openai/gpt-5.6-luna-20260709",
            "request_model_must_equal": "openai/gpt-5.6-luna",
            "provider_tag_must_equal": "openai",
            "response_model_must_be_one_of": ["openai/gpt-5.6-luna", "openai/gpt-5.6-luna-20260709"],
        },
        "judge model identity preflight is invalid",
    )
    _expect(
        judge["result_validation"]
        == {
            "require_exact_rubric_ids": True,
            "require_unique_rubric_ids": True,
            "archive_per_judgment": [
                "generation_id",
                "selected_endpoint",
                "provider",
                "model",
                "finish_reason",
                "service_tier",
                "schema_id",
                "rubric_ids",
                "tokens",
                "reasoning_effort",
                "latency_ms",
                "cost",
                "generation_metadata_polls",
                "error",
                "request_sha256",
            ],
            "refusal_or_transport_failure": "invalid_zero_credit",
            "schema_failure": "invalid_zero_credit",
        },
        "judge result validation is invalid",
    )
    _expect(not _has_key(judge, "temperature"), "judge config must omit unsupported temperature")
    response_format = _object(judge["response_format"], "judge.response_format")
    _exact_keys(response_format, {"type", "json_schema"}, "judge.response_format")
    json_schema = _object(response_format["json_schema"], "judge.response_format.json_schema")
    _exact_keys(json_schema, {"name", "strict", "schema"}, "judge.response_format.json_schema")
    _expect(response_format["type"] == "json_schema", "judge response format must be json_schema")
    _expect(
        json_schema["name"] == "rubric_judgment" and json_schema["strict"] is True, "judge JSON schema must be strict"
    )
    _expect(json_schema["schema"] == _JUDGE_SCHEMA, "judge output schema does not match the strict rubric contract")


def _validate_evaluation(evaluation: JsonObject) -> None:
    _exact_keys(evaluation, {"schema_version", "rtt_source", "benchmarks"}, "evaluation")
    _expect(evaluation["schema_version"] == 1, "evaluation schema version is invalid")
    _expect(
        evaluation["rtt_source"]
        == {
            "repository": "https://github.com/TURLEing/Rubrics-To-Tokens",
            "revision": "b1ab2fba9bece98674e5fa6e6c808d9d63235778",
        },
        "evaluation RTT source is invalid",
    )
    _expect(
        evaluation["benchmarks"]
        == [
            {
                "id": "ifeval",
                "records": 541,
                "input_path": "Benchmark/instruction_following_eval/data/input_data.jsonl",
                "sha256": "67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49",
                "data_license": "Apache-2.0",
                "enabled": True,
            },
            {
                "id": "ifbench",
                "records": 294,
                "input_path": "Benchmark/IFBench/data/IFBench_test.jsonl",
                "sha256": "11c3d683dcc7f4908a4d3cacd05c9a8bbd5484af2f8fde969e7abe2b8bad3e34",
                "data_license": "ODC-BY-1.0",
                "enabled": True,
            },
            {
                "id": "muldimif",
                "records": 1200,
                "input_path": "Benchmark/MulDimIF/Data/test.json",
                "sha256": "37de38b3eb8eec449f3f95dcc0bb0b4fa63a9351988a44d7b232d13348711c5b",
                "data_license": "CC-BY-4.0",
                "enabled": True,
            },
            {
                "id": "advancedif",
                "records": 1645,
                "repository": "https://github.com/facebookresearch/AdvancedIF",
                "repository_revision": "f9d30137c4139d4d9af260ae28108b5afae828c0",
                "dataset": "facebook/AdvancedIF",
                "dataset_revision": "e20cba9b94b59c027dfab00b29244e8bc42e4ab4",
                "data_license": "CC-BY-NC-4.0",
                "enabled": False,
                "blocked_until": "adapter_and_evaluator_certification",
            },
            {
                "id": "math_500",
                "scope": "evaluation_only_ood",
                "records": 500,
                "dataset": "HuggingFaceH4/MATH-500",
                "dataset_config": "default",
                "split": "test",
                "dataset_revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
                "input_path": "test.jsonl",
                "sha256": "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132",
                "data_license": "MIT",
                "prompt": "Question: {}\nPlease reason step by step, and put your final answer within \\boxed{}.",
                "scorer": {
                    "answer": "last_boxed",
                    "package": "math-verify",
                    "revision": "ba3d3aaff23b3f4cac7a14672b4f6e293d97c98b",
                    "antlr_version": "4.9.3",
                },
                "enabled": True,
            },
            {
                "id": "gpqa",
                "scope": "evaluation_only_ood",
                "records": 448,
                "dataset": "Idavidrein/gpqa",
                "dataset_config": "gpqa_main",
                "split": "train",
                "dataset_revision": "pending_gated_access",
                "sha256": "pending_gated_access",
                "data_license": "CC-BY-4.0",
                "prompt": (
                    "Question: {}\nAnswer the multiple choice question. The last line of your response should be of "
                    "the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of choices. Think "
                    "step by step before answering."
                ),
                "option_permutation": {
                    "source_revision": "56686c06f5e19865c153de0fdb11be3890014df7",
                    "seed": 0,
                    "freeze_per_item_across_completions_and_checkpoints": True,
                },
                "scorer": {
                    "primary": "strict_last_line",
                    "regex": "^Answer:\\s*([A-D])\\s*$",
                    "malformed": "incorrect",
                },
                "privacy": "never_commit_or_publish_prompts_or_responses",
                "enabled": True,
                "execution_state": "blocked_gated_access",
            },
            {
                "id": "mmlu_pro",
                "scope": "evaluation_only_ood",
                "records": 12_032,
                "dataset": "TIGER-Lab/MMLU-Pro",
                "dataset_config": "default",
                "split": "test",
                "dataset_revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
                "input_path": "data/test-00000-of-00001.parquet",
                "sha256": "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8",
                "data_license": "MIT",
                "prompt": (
                    "Question: {}\nAnswer the multiple choice question. The last line of your response should be of "
                    "the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of choices. Think "
                    "step by step before answering."
                ),
                "options": {"remove_literal_na": True, "reshuffle": False},
                "scorer": {
                    "primary": "strict_last_line",
                    "regex": "^Answer:\\s*([A-J])\\s*$",
                    "malformed": "incorrect",
                    "random_guess_fallback": False,
                },
                "version_note": "current_pinned_dataset_not_claimed_as_the_unreleased_rtt_snapshot",
                "enabled": True,
            },
        ],
        "evaluation benchmark registry is invalid",
    )


def _validate_data(data: JsonObject, taxonomy: JsonObject) -> None:
    _exact_keys(
        data,
        {"schema_version", "id", "dataset", "revision", "license", "source", "rtt_processed"},
        "data",
    )
    _expect(
        (data["schema_version"], data["id"], data["dataset"], data["revision"], data["license"])
        == (1, "hir_16k", "sastpg/HIR-16K", "2a95f69eb56cc47edc16a45f939cde479673a4cb", "Apache-2.0"),
        "HIR source identity is invalid",
    )
    _expect(
        data["source"]
        == {
            "path": "HIR_trainv1.jsonl",
            "url": (
                "https://huggingface.co/datasets/sastpg/HIR-16K/resolve/"
                "2a95f69eb56cc47edc16a45f939cde479673a4cb/HIR_trainv1.jsonl?download=true"
            ),
            "bytes": 53_147_812,
            "records": 16_968,
            "sha256": HIR_SOURCE_SHA256,
        },
        "HIR source manifest pin is invalid",
    )
    _expect(
        data["rtt_processed"]
        == {
            "repository": "https://github.com/TURLEing/Rubrics-To-Tokens",
            "revision": "b1ab2fba9bece98674e5fa6e6c808d9d63235778",
            "path": "data/HIR_trainv1_rubrics_processed.jsonl",
            "records": 16_968,
            "sha256": "d6690a29cd4f24a3627dd8d48e78953191d0c97ad6acb92cdaf2bf5f1b67568a",
        },
        "RTT processed HIR manifest pin is invalid",
    )
    _exact_keys(
        taxonomy,
        {"schema_version", "id", "source", "inference", "expected", "taxonomy_digest", "static_rtt_route_audit"},
        "taxonomy",
    )
    _expect(
        taxonomy["schema_version"] == 1 and taxonomy["id"] == "hir_hard_soft_taxonomy",
        "HIR taxonomy identity is invalid",
    )
    _expect(
        taxonomy["source"]
        == {"manifest": "configs/data/hir.json", "path": "data/HIR_trainv1.jsonl", "sha256": HIR_SOURCE_SHA256},
        "HIR taxonomy source pin is invalid",
    )
    _expect(
        taxonomy["inference"]
        == {
            "type1": "all criteria are hard",
            "type2": "all criteria are hard",
            "type3": (
                "all criteria are hard by source; constraints preserve order and align structurally "
                "without claiming text equality"
            ),
            "type4": "one checker per criterion in order; [rule] is hard and [llm] is soft",
        },
        "HIR taxonomy inference rules are invalid",
    )
    expected = {
        "rows": 16_968,
        "criteria": 122_992,
        "hard": 76_456,
        "soft": 46_536,
        "sources": {
            "type1": {"rows": 6_549, "criteria": 40_699, "hard": 40_699, "soft": 0},
            "type2": {"rows": 449, "criteria": 2_797, "hard": 2_797, "soft": 0},
            "type3": {"rows": 3_608, "criteria": 20_146, "hard": 20_146, "soft": 0},
            "type4": {"rows": 6_362, "criteria": 59_350, "hard": 12_814, "soft": 46_536},
        },
        "type4_rows": {"hard_only": 0, "mixed": 5_719, "soft_only": 643},
    }
    _expect(taxonomy["expected"] == expected, "HIR taxonomy counts are invalid")
    _expect(
        taxonomy["taxonomy_digest"]
        == {
            "algorithm": "sha256",
            "encoding": "ordered UTF-8 JSON lines of [id,source,hard_mask] with compact separators and integer masks",
            "sha256": HIR_TAXONOMY_SHA256,
        },
        "HIR taxonomy hash is invalid",
    )
    route_audit = _object(taxonomy["static_rtt_route_audit"], "taxonomy.static_rtt_route_audit")
    _expect(
        route_audit
        == {
            "repository": "https://github.com/TURLEing/Rubrics-To-Tokens",
            "revision": "b1ab2fba9bece98674e5fa6e6c808d9d63235778",
            "route_digest": "a367d5e688fa2996543b123ce7491c8e878515597751bcf38f5b33f9e57d3e22",
            "status": "route_resolvable_inventory",
            "note": (
                "Static route resolution only; semantic evaluator certification requires separate smoke and mutation "
                "evidence"
            ),
            "supported": 75_657,
            "unsupported": 799,
            "unique_rows_with_unsupported": 650,
            "sources": {
                "type1": {"supported": 40_367, "unsupported": 332},
                "type2": {"supported": 2_349, "unsupported": 448},
                "type3": {"supported": 20_127, "unsupported": 19},
                "type4": {"supported": 12_814, "unsupported": 0},
            },
            "unsupported_keys": {
                "type1": {"detectable_format:constrained_response": 332},
                "type2": {
                    "format:quotes": 149,
                    "ratio:sentence_balance": 69,
                    "ratio:sentence_type": 81,
                    "words:start_verb": 149,
                },
                "type3": {"Language_Chinese": 19},
                "type4": {},
            },
        },
        "static unsupported hard-route inventory is invalid",
    )
    _expect(route_audit["supported"] + route_audit["unsupported"] == expected["hard"], "hard-route totals are invalid")


def _validate_baselines(baselines: JsonObject, artifacts: JsonObject, hashes: JsonObject, repo_root: Path) -> None:
    _exact_keys(baselines, {"sft", "dpo"}, "baselines")
    for name, baseline in baselines.items():
        frozen = _validate_baseline(_object(baseline, f"baselines.{name}"), name)
        artifact = artifacts.get(name)
        if frozen:
            _expect(artifact is not None, f"frozen {name} data manifest must be loaded")
            _validate_baseline_artifact(name, baseline, _object(artifact, f"baseline {name} artifact"), repo_root)
            _expect(hashes.get(f"baseline_{name}") == baseline["data"]["sha256"], f"{name} data hash mismatch")
        else:
            _expect(artifact is None, f"pending {name} data cannot load an artifact")


def _validate_baseline(config: JsonObject, name: str) -> bool:
    _exact_keys(
        config,
        {
            "schema_version",
            "id",
            "control_kind",
            "reproduction_claim",
            "initial_checkpoint",
            "data",
            "training",
            "readiness",
        },
        f"baselines.{name}",
    )
    _expect(config["schema_version"] == 1 and config["id"] == f"{name}_reconstructed", f"{name} identity is invalid")
    _expect(
        config["control_kind"] == "reconstructed_control"
        and config["reproduction_claim"] == "not_rtt_paper_reproduction",
        f"{name} must be labeled as a reconstructed non-reproduction control",
    )
    _expect(config["initial_checkpoint"] == MODEL_ID, f"{name} must start from the base checkpoint")
    frozen = _validate_baseline_data(_object(config["data"], f"baselines.{name}.data"), name)
    readiness = "ready" if frozen else "blocked_until_frozen_data_manifest"
    message = f"frozen {name} data must enable training" if frozen else f"pending {name} data must block training"
    _expect(config["readiness"] == readiness, message)
    _validate_baseline_training(_object(config["training"], f"baselines.{name}.training"), name)
    return frozen


def _validate_baseline_artifact(name: str, baseline: JsonObject, artifact: JsonObject, repo_root: Path) -> None:
    _exact_keys(
        artifact,
        {
            "schema_version",
            "id",
            "baseline_id",
            "source_sha256",
            "dev_split_manifest",
            "data",
            "teacher",
        },
        f"baseline {name} artifact",
    )
    teacher = _object(artifact["teacher"], f"baseline {name} artifact.teacher")
    _exact_keys(teacher, {"model_id", "revision"}, f"baseline {name} artifact.teacher")
    _expect(
        artifact["schema_version"] == 1
        and artifact["id"] == baseline["data"]["manifest_id"]
        and artifact["baseline_id"] == baseline["id"]
        and artifact["source_sha256"] == _source_hash_contract(),
        f"{name} data artifact identity is invalid",
    )
    expected_teacher = {key: baseline["data"]["teacher"][key] for key in ("model_id", "revision")}
    _expect(teacher == expected_teacher, f"{name} data artifact teacher does not match its config")
    _validate_dataset_manifest(artifact, name, repo_root)


def _validate_baseline_data(data: JsonObject, name: str) -> bool:
    contract = _sft_data_contract() if name == "sft" else _dpo_data_contract()
    _exact_keys(data, set(contract), f"baselines.{name}.data")
    pending = data["status"] == data["manifest_id"] == data["sha256"] == "pending"
    frozen = data["status"] == "frozen" and _frozen_pin(data["manifest_id"], data["sha256"])
    _expect(pending or frozen, f"{name} data manifest must be pending or frozen with a SHA-256 pin")
    teacher = _object(data["teacher"], f"baselines.{name}.data.teacher")
    expected_teacher = _object(contract["teacher"], f"{name} data teacher contract")
    _exact_keys(teacher, set(expected_teacher), f"baselines.{name}.data.teacher")
    if pending:
        _expect(teacher["model_id"] == teacher["revision"] == "pending", f"pending {name} teacher must be unresolved")
    else:
        _expect(
            isinstance(teacher["model_id"], str)
            and teacher["model_id"] not in {"", "pending"}
            and _frozen_teacher_pin(teacher["model_id"], teacher["revision"]),
            f"frozen {name} teacher must be pinned",
        )
    if name == "sft":
        _expect(teacher["generation"] == expected_teacher["generation"], "SFT generation contract is invalid")
    field = "filtering"
    _expect(data[field] == contract[field], f"{name.upper()} data filtering contract is invalid")
    if name == "dpo":
        _expect(
            data["candidate_generation"] == contract["candidate_generation"],
            "DPO candidate generation contract is invalid",
        )
    return frozen


def _validate_baseline_training(training: JsonObject, name: str) -> None:
    expected_loss = (
        {"type": "causal_lm", "label_scope": "assistant_tokens_only", "packing": False}
        if name == "sft"
        else {"type": "sigmoid", "beta": 0.1, "reference_free": False, "label_smoothing": 0}
    )
    _expect(
        training
        == {
            "seed": {"source": "program_confirmation_seed"},
            "hardware": {
                "gpu_count": 1,
                "first_gpu": "NVIDIA A10G",
                "first_memory_gb": 24,
                "fallback_memory_gb": 48,
                "fallback_requires_sealed_gate_failure": True,
            },
            "qlora": {
                "bits": 4,
                "quant_type": "nf4",
                "double_quant": True,
                "compute_dtype": "bfloat16",
                "rank": 64,
                "alpha": 128,
                "dropout": 0.05,
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            },
            "max_sequence_length": 4096,
            "epochs": 3 if name == "sft" else 1,
            "learning_rate": 2e-4 if name == "sft" else 1e-4,
            "scheduler": "cosine",
            "warmup_ratio": 0.03,
            "weight_decay": 0,
            "max_grad_norm": 1,
            "gradient_checkpointing": True,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "optimizer": "paged_adamw_8bit",
            "loss": expected_loss,
        },
        f"{name} A10G-first QLoRA training contract is invalid",
    )


def _sft_data_contract() -> JsonObject:
    return {
        "manifest_id": "pending",
        "sha256": "pending",
        "status": "pending",
        "teacher": {
            "model_id": "pending",
            "revision": "pending",
            "generation": {
                "max_tokens": 4096,
                "seed": TUNING_SEED,
                "reasoning": {"effort": "medium", "exclude": True},
                "provider": {
                    "order": ["openai"],
                    "only": ["openai"],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                    "data_collection": "deny",
                    "zdr": False,
                },
                "responses_per_prompt": 1,
            },
        },
        "filtering": {
            "require_all_authoritative_hard_rubrics": True,
            "require_valid_soft_judgment": True,
            "deduplicate_exact_outputs": True,
            "reject_prompt_or_reference_leakage": True,
            "freeze_before_training": True,
        },
    }


def _dpo_data_contract() -> JsonObject:
    return {
        "manifest_id": "pending",
        "sha256": "pending",
        "status": "pending",
        "teacher": {"model_id": "pending", "revision": "pending"},
        "candidate_generation": {
            "candidate_model": MODEL_ID,
            "candidate_revision": MODEL_REVISION,
            "temperature": 0.99,
            "top_p": 0.99,
            "top_k": 100,
            "max_new_tokens": 4096,
            "seed": TUNING_SEED,
            "candidates_per_prompt": 8,
        },
        "filtering": {
            "preference_order": ["authoritative_hard_pass", "soft_quality", "deterministic_tie_break"],
            "require_distinct_pair": True,
            "require_valid_preferred_soft_judgment": True,
            "reject_ambiguous_ties": True,
            "reject_prompt_or_reference_leakage": True,
            "freeze_before_training": True,
        },
    }


def _source_hash_contract() -> JsonObject:
    return {
        "hir_source": HIR_SOURCE_SHA256,
        "hir_processed": "d6690a29cd4f24a3627dd8d48e78953191d0c97ad6acb92cdaf2bf5f1b67568a",
        "taxonomy": HIR_TAXONOMY_SHA256,
    }


def _validate_dataset_manifest(artifact: JsonObject, name: str, repo_root: Path) -> None:
    data = _object(artifact["data"], f"{name} data artifact.data")
    _exact_keys(
        data,
        {"path", "sha256", "schema", "records", "row_ids", "row_ids_sha256", "output_digests"}
        | ({"pairs"} if name == "dpo" else set()),
        f"{name} data artifact.data",
    )
    path = _resolve_under(repo_root, repo_root / _string(data["path"], f"{name} data path"), f"{name} data path")
    _expect(_file_sha256(path) == data["sha256"], f"{name} dataset byte hash mismatch")
    expected_schema = (
        {"format": "jsonl", "fields": ["row_id", "prompt", "chosen", "rejected"]}
        if name == "dpo"
        else {"format": "jsonl", "fields": ["row_id", "prompt", "output"]}
    )
    row_ids = data["row_ids"]
    digests = data["output_digests"]
    rows = _load_dataset_rows(path, expected_schema["fields"], f"{name} dataset")
    actual_ids = [row["row_id"] for row in rows]
    actual_digests = [_json_sha256(row) for row in rows]
    _expect(
        data["schema"] == expected_schema
        and isinstance(row_ids, list)
        and all(_valid_hir_id(row_id) for row_id in row_ids)
        and len(row_ids) == len(set(row_ids)) == data["records"]
        and data["records"] > 0
        and data["row_ids_sha256"] == _json_sha256({"row_ids": row_ids})
        and isinstance(digests, list)
        and len(digests) == data["records"]
        and all(_sha256(value) for value in digests),
        f"{name} dataset manifest rows or output digests are invalid",
    )
    _expect(actual_ids == row_ids and actual_digests == digests, f"{name} dataset contents do not match its manifest")
    dev = _object(artifact["dev_split_manifest"], f"{name} data artifact.dev_split_manifest")
    _exact_keys(dev, {"id", "sha256", "row_ids"}, f"{name} data artifact.dev_split_manifest")
    _expect(
        _frozen_pin(dev["id"], dev["sha256"])
        and isinstance(dev["row_ids"], list)
        and all(_valid_hir_id(row_id) for row_id in dev["row_ids"])
        and set(row_ids).isdisjoint(dev["row_ids"]),
        f"{name} train and dev identities must be pinned and disjoint",
    )
    if name == "dpo":
        pairs = data["pairs"]
        _expect(
            isinstance(pairs, list)
            and len(pairs) == data["records"]
            and all(
                isinstance(pair, dict)
                and set(pair) == {"row_id", "chosen_sha256", "rejected_sha256"}
                and pair["row_id"] == row_id
                and _sha256(pair["chosen_sha256"])
                and _sha256(pair["rejected_sha256"])
                and pair["chosen_sha256"] != pair["rejected_sha256"]
                for pair, row_id in zip(pairs, row_ids, strict=True)
            ),
            "DPO pair manifest is invalid",
        )
        actual_pairs = [
            {
                "row_id": row["row_id"],
                "chosen_sha256": hashlib.sha256(row["chosen"].encode()).hexdigest(),
                "rejected_sha256": hashlib.sha256(row["rejected"].encode()).hexdigest(),
            }
            for row in rows
        ]
        _expect(pairs == actual_pairs, "DPO pair digests do not match dataset contents")


def _valid_hir_id(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 16_969 and value != 8_920


def _validate_compute(compute: JsonObject) -> None:
    _validate_compute_identity(compute)
    _validate_compute_topology(compute)
    _validate_response_runtime(_object(compute["response_runtime"], "compute.response_runtime"))
    _validate_analytical_memory(compute["analytical_memory"])
    _validate_workloads(_object(compute["workloads"], "compute.workloads"))
    _validate_run_matrix(_object(compute["run_matrix"], "compute.run_matrix"))
    _validate_cost_accounting(_object(compute["cost_accounting"], "compute.cost_accounting"))


def _validate_compute_identity(compute: JsonObject) -> None:
    keys = {
        "schema_version",
        "id",
        "rtt_released_topology",
        "adapted_deviations",
        "target_topology",
        "response_runtime",
        "analytical_memory",
        "workloads",
        "run_matrix",
        "cost_accounting",
    }
    _exact_keys(compute, keys, "compute")
    _expect(compute["schema_version"] == 4 and compute["id"] == "qwen_a100_2x", "compute identity is invalid")


def _validate_compute_topology(compute: JsonObject) -> None:
    _expect(
        compute["rtt_released_topology"]
        == {
            "source_config": "examples/rubrics2tokens/rubric2token_config_qwen3.yaml",
            "source_revision": "b1ab2fba9bece98674e5fa6e6c808d9d63235778",
            "num_gpus_per_node": 8,
            "logical_gpu_indices": 64,
            "implied_node_count": 8,
            "actor_train": {"start": 0, "stop_exclusive": 32},
            "actor_infer": {"start": 0, "stop_exclusive": 32},
            "judge": {"start": 32, "stop_exclusive": 64, "gpus_per_worker": 8},
            "discriminator": {"start": 0, "stop_exclusive": 16},
            "save_steps": 20,
        },
        "released RTT topology snapshot is invalid",
    )
    _expect(
        compute["adapted_deviations"]
        == [
            "external_openrouter_soft_judge_replaces_released_deepseek_v3_1_judge",
            "authoritative_hard_evaluators_are_separate_from_soft_judgments",
            "two_a100_actor_train_and_infer_time_share",
            "reconstructed_sft_and_dpo_use_a10g_first_qlora_with_48gb_gated_fallback",
            "token_discriminator_starts_disabled_until_certified",
        ],
        "adapted topology deviations are invalid",
    )
    _expect(
        compute["target_topology"]
        == {
            "status": "unprovisioned_waiting_for_static_release",
            "model": "NVIDIA A100",
            "count": 2,
            "minimum_memory_gib_each": 79,
            "actor_train_devices": [0, 1],
            "actor_infer_devices": [0, 1],
            "placement": "colocated_synchronous_time_share",
        },
        "target A100 topology is invalid",
    )


def _validate_analytical_memory(memory: Any) -> None:
    _expect(
        memory
        == {
            "evidence": "calculated_not_measured",
            "model_parameters": 4_022_468_096,
            "bf16_weights_bytes": 8_044_936_192,
            "bf16_weights_gib": 7.492,
            "mixed_precision_adam_bytes_per_parameter": 16,
            "mixed_precision_adam_state_bytes": 64_359_489_536,
            "mixed_precision_adam_state_gib": 59.939,
            "bf16_kv_cache_kib_per_token": 144,
            "kv_cache_gib_per_6144_token_sequence": 0.844,
            "kv_cache_gib_for_512_sequences": 432,
            "schedule_rollouts_in_waves": True,
        },
        "analytical memory contract is invalid",
    )


def _validate_response_runtime(runtime: JsonObject) -> None:
    _expect(
        runtime
        == {
            "profile": "fsdp2-hf-sdpa-2xa100",
            "host": {"minimum_ram_gib": 192, "minimum_free_disk_gib": 512},
            "platform": {
                "system": "Linux",
                "machines": ["x86_64", "AMD64"],
                "receipt": "Ubuntu-24.04-x86_64",
                "container_contract": "../../requirements/a100-response-container.json",
            },
            "gpu": {
                "topology_contract": "target_topology",
                "minimum_driver": "575.57.08",
                "cuda_runtime": "12.9",
                "nccl_package": "nvidia-nccl-cu12",
                "require_idle": True,
                "supported_links": ["NV", "PIX", "PXB", "PHB", "NODE", "SYS"],
            },
            "packages": {
                "index_url": "https://pypi.org/simple",
                "torch_index_url": "https://download.pytorch.org/whl/cu129",
                "torch_backend": "cu129",
                "contracts": [
                    "../../requirements/a100-response-linux-py312.lock",
                    "../../requirements/a100-response-flash.txt",
                ],
                "allowed_local_suffixes": {"torch": "cu129", "torchaudio": "cu129", "torchvision": "cu129"},
            },
        },
        "response runtime contract is invalid",
    )


def _validate_workloads(workloads: JsonObject) -> None:
    _expect(
        workloads
        == {
            "base_and_checkpoint_evaluation": {
                "minimum_gpu_memory_gb": 24,
                "preferred_gpu_memory_gb": 48,
                "gpus_per_job": 1,
                "status": "running_after_per_card_memory_probe",
            },
            "reconstructed_qlora_sft_dpo": {
                "first_gpu": "NVIDIA A10G",
                "first_gpu_memory_gb": 24,
                "required_memory_gate_headroom_gib": 1.5,
                "maximum_peak_reserved_fraction": 0.92,
                "fallback_gpu_memory_gb": 48,
                "fallback_requires_sealed_a10g_gate_failure": True,
                "gpus_per_job": 1,
                "status": "blocked_until_one_step_memory_and_fresh_resume_gate",
            },
            "scalar_full_parameter_rl": {
                "gpu": "NVIDIA A100",
                "memory_gb_each": 80,
                "gpus_per_job": 2,
                "placement": "colocated_synchronous_time_share",
                "status": "blocked_until_no_update_preflight_and_20_step_pilot",
            },
            "token_full_parameter_rl": {
                "gpu": "NVIDIA A100",
                "memory_gb_each": 80,
                "gpus_per_job": 2,
                "optional_discriminator_gpu": "one_48gb_after_certification",
                "status": "blocked_until_scalar_program_and_discriminator_certification",
            },
        },
        "compute workloads are invalid",
    )


def _validate_run_matrix(matrix: JsonObject) -> None:
    common = {
        "base_evaluation_suites": 1,
        "scalar_rl_runs": 18,
        "qlora_baseline_runs": 6,
        "token_rl_runs": 12,
        "tuning_runs": 9,
        "fresh_confirmation_runs": 27,
        "training_runs": 36,
        "evaluation_suites": 37,
        "planned_enabled_benchmarks_per_suite": 6,
        "planned_benchmark_executions": 222,
    }
    _expect(
        matrix
        in (
            common
            | {
                "runnable_benchmarks_per_suite": 5,
                "runnable_benchmark_executions": 185,
                "gpqa_transition": "blocked_until_revision_hash_and_access_manifest_frozen",
            },
            common
            | {
                "runnable_benchmarks_per_suite": 6,
                "runnable_benchmark_executions": 222,
                "gpqa_transition": "ready_from_frozen_revision_hash_and_access_manifest",
            },
        ),
        "compute run matrix is invalid",
    )


def _validate_cost_accounting(cost: JsonObject) -> None:
    _expect(
        cost
        == {
            "status": "pending_measurement",
            "gpu_hour_definition": "allocated_gpu_count_times_measured_wall_clock_hours",
            "required_measurements": [
                "20_step_scalar_pilot",
                "20_step_token_pilot",
                "one_complete_checkpoint_evaluation",
                "one_complete_qlora_sft_run",
                "one_complete_qlora_dpo_run",
            ],
            "total_project_gpu_hours": "pending_measurement",
            "reserve_percent": 20,
            "reserve_application": "apply_only_after_measured_component_sum",
        },
        "GPU-hour accounting must remain pending until measured pilots exist",
    )


def _validate_experiment(
    program: JsonObject,
    dev_split_artifact: JsonObject | None,
    selection_artifact: JsonObject | None,
    artifact_hashes: JsonObject,
    repo_root: Path,
) -> None:
    _validate_program_references(program)
    _validate_launch_configs(program, repo_root)
    _validate_program_seeds(program)
    tuning_runs = _validate_selection(
        _object(program["selection"], "program.selection"),
        program["readiness"],
        dev_split_artifact,
        selection_artifact,
        artifact_hashes,
    )
    trainable = _validate_methods(program["methods"])
    _validate_execution_priority(program["execution_priority"])
    _validate_program_counts(_object(program["counts"], "program.counts"), tuning_runs, trainable)
    _validate_rl_recipe(_object(program["rl_recipe"], "program.rl_recipe"))
    _validate_program_readiness(_object(program["readiness"], "program.readiness"))
    _validate_benchmarks(program["benchmarks"])
    _validate_ood_evaluation(_object(program["ood_evaluation"], "program.ood_evaluation"))
    _validate_pilot(_object(program["pilot"], "program.pilot"))


def _validate_program_references(program: JsonObject) -> None:
    _expect(program["schema_version"] == 1 and program["id"] == "qwen_first", "program identity is invalid")
    _expect(program["targets_config"] == "../models/targets.json", "program target registry reference is invalid")
    _expect(program["model_config"] == "../models/qwen3_4b.json", "program model reference is invalid")
    _expect(program["judge_config"] == "../judges/openrouter_luna.json", "program judge reference is invalid")
    _expect(program["compute_config"] == "../compute/qwen_a100_2x.json", "program compute reference is invalid")
    _expect(program["eval_config"] == "../eval/benchmarks.json", "program evaluation reference is invalid")
    _expect(program["data_config"] == "../data/hir.json", "program HIR source reference is invalid")
    _expect(program["taxonomy_config"] == "../data/hir_taxonomy.json", "program HIR taxonomy reference is invalid")
    _expect(
        program["baseline_configs"]
        == {"sft": "../baselines/sft_reconstructed.json", "dpo": "../baselines/dpo_reconstructed.json"},
        "program baseline references are invalid",
    )
    _expect(program["artifact_config"] == "../artifacts/qwen_evidence.json", "program artifact reference is invalid")


def _validate_launch_configs(program: JsonObject, repo_root: Path) -> None:
    launch = _object(program["launch_train_config"], "program.launch_train_config")
    _exact_keys(
        launch,
        {"path", "sha256", "preflight_sha256", "hydra_parent", "purpose"},
        "program.launch_train_config",
    )
    hydra_parent = _object(launch["hydra_parent"], "program.launch_train_config.hydra_parent")
    _exact_keys(hydra_parent, {"path", "sha256"}, "program.launch_train_config.hydra_parent")
    _expect(
        launch["path"] == "configs/roll/qwen_rtt_papo_response_train.yaml"
        and launch["purpose"] == "500_step_recipe_with_stop_after_step_pilot_gate"
        and launch["sha256"] == _file_sha256(repo_root / launch["path"])
        and launch["preflight_sha256"]
        == _file_sha256(repo_root / "configs/roll/qwen_rtt_papo_response_preflight.yaml"),
        "program launch train config pin is invalid",
    )
    _expect(
        hydra_parent
        == {
            "path": "configs/roll/qwen_scalar_train.yaml",
            "sha256": _file_sha256(repo_root / "configs/roll/qwen_scalar_train.yaml"),
        },
        "program launch Hydra parent config pin is invalid",
    )
    same_backend = _object(program["same_backend_configs"], "program.same_backend_configs")
    _exact_keys(same_backend, {"diagnostic", "vllm_diagnostic", "production"}, "program.same_backend_configs")
    diagnostic = _object(same_backend["diagnostic"], "program.same_backend_configs.diagnostic")
    _exact_keys(diagnostic, {"path", "sha256"}, "same-backend diagnostic config")
    _expect(
        diagnostic
        == {
            "path": "configs/roll/qwen_rtt_papo_response_parity.yaml",
            "sha256": _file_sha256(repo_root / "configs/roll/qwen_rtt_papo_response_parity.yaml"),
        },
        "same-backend diagnostic config pin is invalid",
    )
    vllm_diagnostic = _object(same_backend["vllm_diagnostic"], "program.same_backend_configs.vllm_diagnostic")
    _exact_keys(vllm_diagnostic, {"path", "sha256"}, "vLLM diagnostic config")
    _expect(
        vllm_diagnostic
        == {
            "path": "configs/roll/qwen_rtt_papo_response_vllm_parity.yaml",
            "sha256": _file_sha256(repo_root / "configs/roll/qwen_rtt_papo_response_vllm_parity.yaml"),
        },
        "vLLM diagnostic config pin is invalid",
    )
    production = _object(same_backend["production"], "program.same_backend_configs.production")
    frozen_production = (
        set(production) == {"status", "path", "sha256"}
        and production.get("status") == "frozen"
        and isinstance(production.get("path"), str)
        and production["path"] == "configs/roll/qwen_rtt_papo_response_train.yaml"
        and _sha256(production.get("sha256"))
        and production["sha256"] == _file_sha256(repo_root / production["path"])
    )
    _expect(frozen_production, "same-backend production config pin is invalid")


def _validate_program_seeds(program: JsonObject) -> None:
    seeds = _object(program["seeds"], "program.seeds")
    _exact_keys(seeds, {"tuning", "confirmation"}, "program.seeds")
    _expect(seeds["tuning"] == TUNING_SEED, "program tuning seed is invalid")
    _expect(tuple(seeds["confirmation"]) == CONFIRMATION_SEEDS, "program confirmation seeds are invalid")
    _expect(seeds["tuning"] not in seeds["confirmation"], "tuning and confirmation seeds must be disjoint")
    _expect(len(set(seeds["confirmation"])) == len(CONFIRMATION_SEEDS), "confirmation seeds must be unique")


def _validate_methods(methods: Any) -> int:
    _expect(isinstance(methods, list), "program.methods must be an array")
    _expect(
        tuple(method.get("id") for method in methods if isinstance(method, dict)) == METHOD_IDS,
        "method order is invalid",
    )
    _expect(len(methods) == len(METHOD_IDS), "method count is invalid")
    for index, (method, objective) in enumerate(zip(methods, _METHOD_OBJECTIVES, strict=True)):
        item = _object(method, f"program.methods[{index}]")
        _exact_keys(item, {"id", "trainable", "initial_checkpoint", "objective"}, f"program.methods[{index}]")
        _expect(item["trainable"] is (index > 0), f"method {item['id']} trainable flag is invalid")
        _expect(
            item["initial_checkpoint"] == MODEL_ID, f"method {item['id']} must start independently from {MODEL_ID}"
        )
        _expect(item["objective"] == objective, f"method {item['id']} objective is invalid")
    return sum(method["trainable"] is True for method in methods)


def _validate_program_counts(counts: JsonObject, tuning_runs: int, trainable: int) -> None:
    _exact_keys(
        counts,
        {"tuning_runs", "confirmation_runs", "trainable_runs", "evaluation_suites"},
        "program.counts",
    )
    confirmation_runs = trainable * len(CONFIRMATION_SEEDS)
    _expect(tuning_runs == counts["tuning_runs"] == 9, "tuning run count must be 9")
    _expect(confirmation_runs == counts["confirmation_runs"] == 27, "confirmation run count must be 27")
    _expect(tuning_runs + confirmation_runs == counts["trainable_runs"] == 36, "trainable run count must be 36")
    _expect(1 + counts["trainable_runs"] == counts["evaluation_suites"] == 37, "evaluation suite count must be 37")


def _validate_program_readiness(readiness: JsonObject) -> None:
    _exact_keys(
        readiness,
        {
            "judge",
            "candidate_tuning",
            "confirmation",
            "scalar_training",
            "baselines",
            "token_training",
            "advancedif",
            "launch",
            "gpqa",
        },
        "program.readiness",
    )
    _expect(readiness["judge"] in {"ready", "blocked_until_frozen_calibration"}, "judge readiness is invalid")
    _expect(
        readiness["baselines"] in {"ready", "blocked_until_frozen_data_manifests"},
        "baseline readiness is invalid",
    )
    _expect(
        readiness["token_training"] in {"ready", "blocked_until_labels_discriminator_and_scalar_program_pass"},
        "token training readiness is invalid",
    )
    _expect(
        readiness["advancedif"] == "blocked_until_adapter_and_evaluator_certification",
        "AdvancedIF readiness is invalid",
    )
    _expect(
        readiness["gpqa"] in {"ready", "blocked_until_revision_hash_and_access_manifest_frozen"},
        "GPQA readiness is invalid",
    )


def _validate_rl_recipe(recipe: JsonObject) -> None:
    _expect(
        recipe
        == {
            "source_repository": "https://github.com/TURLEing/Rubrics-To-Tokens",
            "source_revision": "b1ab2fba9bece98674e5fa6e6c808d9d63235778",
            "max_steps": 500,
            "save_steps": 20,
            "rollout_batch_size": 64,
            "group_size": 8,
            "prompt_length": 2048,
            "response_length": 4096,
            "max_model_len": 8000,
            "ppo_epochs": 1,
            "learning_rate": 1e-6,
            "warmup_steps": 20,
            "weight_decay": 0,
            "clip_low": 0.2,
            "clip_high": 0.27,
            "kl_coefficient": 0,
            "dtype": "bf16",
            "generation": {"temperature": 0.99, "top_p": 0.99, "top_k": 100},
        },
        "RL recipe is invalid",
    )


def _validate_execution_priority(value: Any) -> None:
    _expect(
        value
        == [
            {
                "stage": "rtt_papo_response",
                "methods": ["rtt_papo_response"],
                "token_beta": 0,
                "status": "blocked_until_no_update_gates_pass",
            },
            {
                "stage": "rl_csr",
                "methods": ["rl_csr"],
                "status": "blocked_until_rtt_papo_response_evidence",
            },
            {
                "stage": "rl_aon",
                "methods": ["rl_aon"],
                "status": "blocked_until_rl_csr_evidence",
            },
            {
                "stage": "response_mix",
                "methods": ["rl_mix"],
                "status": "blocked_until_rl_aon_evidence",
            },
            {
                "stage": "token_label_generation",
                "produces": "frozen_token_labels",
                "status": "blocked_until_response_only_program_pass",
            },
            {
                "stage": "discriminator_training",
                "consumes": "frozen_token_labels",
                "produces": "cross_linked_discriminator_checkpoint",
                "status": "blocked_until_token_labels_frozen",
            },
            {
                "stage": "discriminator_certification",
                "consumes": "cross_linked_discriminator_checkpoint",
                "produces": "discriminator_certificate",
                "status": "blocked_until_discriminator_checkpoint_frozen",
            },
            {
                "stage": "token_policy_runs",
                "methods": ["rtt_aon", "rtt_csr"],
                "status": "blocked_until_discriminator_certification",
            },
            {
                "stage": "full_rdan",
                "methods": ["rdan_full"],
                "status": "blocked_until_rtt_aon_and_rtt_csr_evidence",
            },
        ],
        "execution priority must start with scalar RDAN and preserve every prerequisite gate",
    )


def _validate_benchmarks(value: Any) -> None:
    _expect(isinstance(value, list) and len(value) == 7, "program must declare seven benchmarks")
    expected = (
        {"id": "ifeval", "enabled": True},
        {"id": "ifbench", "enabled": True},
        {"id": "muldimif", "enabled": True},
        {"id": "advancedif", "enabled": False, "blocked_until": "adapter_and_evaluator_certification"},
        {"id": "math_500", "enabled": True, "scope": "evaluation_only_ood"},
        {
            "id": "gpqa",
            "enabled": True,
            "scope": "evaluation_only_ood",
            "execution_state": "blocked_gated_access",
            "transition_gate": "revision_hash_and_access_manifest_frozen",
        },
        {"id": "mmlu_pro", "enabled": True, "scope": "evaluation_only_ood"},
    )
    actual = tuple(value)
    blocked = actual == expected
    ready = (
        actual
        == expected[:5]
        + (
            {
                "id": "gpqa",
                "enabled": True,
                "scope": "evaluation_only_ood",
                "execution_state": "runnable_frozen_access",
                "transition_gate": "revision_hash_and_access_manifest_frozen",
            },
        )
        + expected[6:]
    )
    _expect(blocked or ready, "benchmark order or certification state is invalid")


def _validate_ood_evaluation(value: JsonObject) -> None:
    _expect(
        value
        == {
            "protocol_source": "RTT Appendix B.1 and Table 7",
            "dataset_scope": "full_official_dataset",
            "prompt_source": "RTT Table 7",
            "chat_template": "model_default",
            "generation": {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "completions_per_item": 5,
            },
            "completion_sampling": "independent_per_item",
            "aggregation": "average_accuracy_across_independent_completions",
            "checkpoint_scope": "selected_checkpoints_only",
            "phase": "post_training_evaluation_only",
            "parsing": "benchmark_primary_strict_scorer",
            "selection_use": "forbidden",
        },
        "OOD evaluation protocol is invalid",
    )


def _validate_pilot(pilot: JsonObject) -> None:
    _exact_keys(pilot, {"steps", "preflight", "gates"}, "program.pilot")
    _expect(pilot["steps"] == 20, "pilot must run 20 optimizer steps")
    _expect(
        pilot["preflight"]
        == {
            "prompt_count": {"minimum": 256, "maximum": 1024},
            "responses_per_prompt": 8,
            "optimizer_updates": 0,
        },
        "pilot preflight contract is invalid",
    )
    _expect(
        pilot["gates"]
        == {
            "minimum_active_quality_group_rate": 0.1,
            "hard_invalid_positive_quality_credit": 0,
            "require_nonzero_useful_reward_variance": True,
            "require_finite_gradients": True,
            "clipped_ratio_must_be_below": 1.0,
            "require_checkpoint": True,
            "require_resume_generation": True,
            "reject_quality_gain_with_authoritative_accuracy_loss": True,
        },
        "pilot gates are invalid",
    )


def _validate_selection(
    selection: JsonObject,
    readiness: Any,
    dev_artifact: JsonObject | None,
    selected_artifact: JsonObject | None,
    hashes: JsonObject,
) -> int:
    _exact_keys(selection, {"dev_split", "candidates", "immutable_artifact"}, "program.selection")
    dev = _object(selection["dev_split"], "program.selection.dev_split")
    _exact_keys(dev, {"path", "required_status", "status", "manifest_id", "sha256"}, "program.selection.dev_split")
    _expect(dev["path"] == "configs/artifacts/qwen_dev_split.json", "dev split artifact path is invalid")
    _expect(dev["required_status"] == "frozen", "candidate tuning must require a frozen dev split")
    dev_pending = dev["status"] == "pending" and dev["manifest_id"] == dev["sha256"] == "pending"
    dev_frozen = dev["status"] == "frozen" and _frozen_pin(dev["manifest_id"], dev["sha256"])
    _expect(dev_pending or dev_frozen, "dev split manifest pin is invalid")
    candidates = _object(selection["candidates"], "program.selection.candidates")
    _expect(
        candidates
        == {
            "rl_mix": [0.25, 0.5, 0.75],
            "rtt_papo_response": [0.25, 0.5, 1.0],
            "rdan_full": [0.25, 0.5, 1.0],
        },
        "candidate selection grid is invalid",
    )
    artifact = _object(selection["immutable_artifact"], "program.selection.immutable_artifact")
    _exact_keys(
        artifact,
        {"path", "artifact_id", "status", "sha256", "must_record"},
        "program.selection.immutable_artifact",
    )
    _expect(artifact["path"] == "configs/artifacts/qwen_selection.json", "selection artifact path is invalid")
    _expect(
        artifact["must_record"]
        == [
            "dev_split_manifest",
            "candidate_metrics",
            "selected_values",
            "selection_rule",
            "created_at_utc",
        ],
        "selection artifact schema is invalid",
    )
    artifact_pending = artifact["status"] == artifact["artifact_id"] == artifact["sha256"] == "pending"
    artifact_frozen = (
        artifact["status"] == "frozen"
        and isinstance(artifact["artifact_id"], str)
        and artifact["artifact_id"] not in {"", "pending"}
        and _sha256(artifact["sha256"])
    )
    _expect(artifact_pending or artifact_frozen, "selection artifact must be pending or frozen with a SHA-256 pin")
    states = _object(readiness, "program.readiness")
    if dev_pending:
        _expect(dev_artifact is None, "pending dev split cannot load an artifact")
        _expect(
            states.get("candidate_tuning") == "blocked_until_frozen_dev_split", "pending dev split must block tuning"
        )
    else:
        _expect(dev_artifact is not None, "frozen dev split artifact must be loaded")
        _validate_dev_split_artifact(dev, _object(dev_artifact, "dev split artifact"), hashes.get("dev_split"))
        _expect(states.get("candidate_tuning") == "ready", "frozen dev split must enable tuning")
    _expect(not artifact_frozen or dev_frozen, "selection artifact cannot be frozen before the dev split")
    confirmation = "ready" if artifact_frozen else "blocked_until_immutable_selection_artifact"
    if artifact_frozen:
        _expect(selected_artifact is not None, "frozen selection artifact must be loaded")
        _validate_selection_artifact(
            artifact,
            dev,
            _object(selected_artifact, "selection artifact"),
            selection["candidates"],
            hashes.get("selection"),
        )
    else:
        _expect(selected_artifact is None, "pending selection cannot load an artifact")
    _expect(states.get("confirmation") == confirmation, "confirmation readiness does not match selection state")
    return sum(len(values) for values in candidates.values())


def _validate_dev_split_artifact(reference: JsonObject, artifact: JsonObject, actual_hash: Any) -> None:
    _exact_keys(
        artifact,
        {
            "schema_version",
            "id",
            "split",
            "source_data_sha256",
            "taxonomy_sha256",
            "records",
            "row_ids",
            "row_ids_sha256",
        },
        "dev split artifact",
    )
    row_ids = artifact["row_ids"]
    valid_row_ids = (
        isinstance(row_ids, list)
        and all(_valid_hir_id(row_id) for row_id in row_ids)
        and row_ids == sorted(row_ids, key=lambda row_id: json.dumps(row_id, ensure_ascii=False, sort_keys=True))
        and len(row_ids) == len({json.dumps(row_id, ensure_ascii=False) for row_id in row_ids})
    )
    _expect(
        actual_hash == reference["sha256"]
        and artifact["schema_version"] == 1
        and artifact["id"] == reference["manifest_id"]
        and artifact["split"] == "dev"
        and artifact["source_data_sha256"] == HIR_SOURCE_SHA256
        and artifact["taxonomy_sha256"] == HIR_TAXONOMY_SHA256
        and isinstance(artifact["records"], int)
        and not isinstance(artifact["records"], bool)
        and artifact["records"] > 0
        and valid_row_ids
        and artifact["records"] == len(row_ids)
        and artifact["row_ids_sha256"] == _json_sha256({"row_ids": row_ids}),
        "dev split artifact is invalid or hash-mismatched",
    )


def _validate_selection_artifact(
    reference: JsonObject,
    dev: JsonObject,
    artifact: JsonObject,
    candidates: JsonObject,
    actual_hash: Any,
) -> None:
    _exact_keys(
        artifact,
        {
            "schema_version",
            "id",
            "dev_split_manifest",
            "candidate_metrics",
            "selected_values",
            "selection_rule",
            "created_at_utc",
        },
        "selection artifact",
    )
    dev_link = _object(artifact["dev_split_manifest"], "selection artifact.dev_split_manifest")
    _exact_keys(dev_link, {"id", "sha256"}, "selection artifact.dev_split_manifest")
    metrics = _object(artifact["candidate_metrics"], "selection artifact.candidate_metrics")
    selected = _object(artifact["selected_values"], "selection artifact.selected_values")
    _expect(set(metrics) == set(selected) == set(candidates), "selection methods do not match the candidate grid")
    for name, values in candidates.items():
        rows = metrics[name]
        _expect(isinstance(rows, list) and len(rows) == len(values), "selection candidate metrics are incomplete")
        for row, candidate in zip(rows, values, strict=True):
            metric = _object(row, f"selection artifact.candidate_metrics.{name}")
            _exact_keys(metric, {"candidate", "score"}, f"selection artifact.candidate_metrics.{name}")
            _expect(
                metric["candidate"] == candidate
                and isinstance(metric["score"], (int, float))
                and not isinstance(metric["score"], bool)
                and math.isfinite(metric["score"]),
                "selection candidate metric is invalid or mis-cross-linked",
            )
        expected = min(
            (row["candidate"] for row in rows if row["score"] == max(item["score"] for item in rows)),
            default=None,
        )
        _expect(selected[name] == expected, f"selected value for {name} does not follow {SELECTION_RULE}")
    _expect(
        all(selected[name] in candidates[name] for name in candidates),
        "selected value is outside the candidate grid",
    )
    _expect(
        actual_hash == reference["sha256"]
        and artifact["schema_version"] == 1
        and artifact["id"] == reference["artifact_id"]
        and dev_link == {"id": dev["manifest_id"], "sha256": dev["sha256"]}
        and artifact["selection_rule"] == SELECTION_RULE
        and isinstance(artifact["created_at_utc"], str)
        and artifact["created_at_utc"].endswith("Z"),
        "selection artifact is invalid, hash-mismatched, or mis-cross-linked",
    )


def _validate_hard_route_policy(
    policy: JsonObject,
    readiness: Any,
    resolution: JsonObject,
    implementation: JsonObject | None,
    certificate: JsonObject | None,
    exclusions: JsonObject | None,
    hashes: JsonObject,
    repo_root: Path,
    scalar_reference: Any,
) -> None:
    _exact_keys(
        policy,
        {"route_resolution", "implementation_manifest", "evaluator_certificate", "exclusion_manifest"},
        "hard routes",
    )
    resolution_ref = _object(policy["route_resolution"], "program.hard_route_policy.route_resolution")
    _exact_keys(resolution_ref, {"status", "path", "sha256"}, "route resolution reference")
    _expect(
        resolution_ref
        == {
            "status": "frozen",
            "path": "configs/artifacts/hir_route_resolution.json",
            "sha256": hashes.get("route_resolution"),
        },
        "route resolution artifact is not loaded from its exact hash",
    )
    unresolved = _validate_route_resolution(resolution)
    scalar_ref = _object(scalar_reference, "program.lifecycle_artifacts.scalar_data")
    scalar_sha = scalar_ref.get("sha256") if scalar_ref.get("status") == "frozen" else None
    implementation_ref = _object(policy["implementation_manifest"], "hard route implementation reference")
    certificate_ref = _object(policy["evaluator_certificate"], "evaluator certificate reference")
    exclusion = _object(policy["exclusion_manifest"], "program.hard_route_policy.exclusion_manifest")
    specs = (
        (
            implementation_ref,
            "configs/artifacts/hir_route_implementation.json",
            "manifest_id",
            implementation,
            "route_implementation",
        ),
        (
            certificate_ref,
            "configs/artifacts/hir_evaluator_certificate.json",
            "certificate_id",
            certificate,
            "evaluator_certificate",
        ),
        (
            exclusion,
            "configs/artifacts/hir_hard_route_exclusions.json",
            "manifest_id",
            exclusions,
            "hard_route_exclusions",
        ),
    )
    frozen: list[bool] = []
    for reference, path, id_key, body, hash_key in specs:
        display_name = (
            "hard route exclusion manifest" if hash_key == "hard_route_exclusions" else hash_key.replace("_", " ")
        )
        _exact_keys(reference, {"status", "path", id_key, "sha256"}, f"{hash_key} reference")
        pending = reference == {"status": "pending", "path": path, id_key: "pending", "sha256": "pending"}
        ready = reference["status"] == "frozen" and _frozen_pin(reference[id_key], reference["sha256"])
        _expect(pending or ready, f"{hash_key} reference is invalid")
        _expect((body is not None) is ready, f"frozen {display_name} must be loaded and pending must be absent")
        if ready:
            _expect(hashes.get(hash_key) == reference["sha256"], f"{hash_key} hash mismatch")
        frozen.append(ready)

    certified: set[str] = set()
    if frozen[0]:
        implemented = _validate_identity_manifest(
            _object(implementation, "route implementation manifest"),
            implementation_ref["manifest_id"],
            resolution_ref["sha256"],
            "implemented_not_certified",
            scalar_sha,
            {"type4": 12_755, "total": 12_755},
        )
    else:
        implemented = set()
    if frozen[1]:
        certified = _validate_evaluator_certificate(
            _object(certificate, "evaluator certificate"),
            certificate_ref["certificate_id"],
            resolution_ref["sha256"],
            implementation_ref["sha256"],
            implemented,
            repo_root,
            scalar_sha,
        )
    excluded = (
        _validate_identity_manifest(
            _object(exclusions, "hard route exclusion manifest"),
            exclusion["manifest_id"],
            resolution_ref["sha256"],
            "excluded",
            scalar_sha,
            {"type1": 40_699, "type2": 2_797, "type3": 20_146, "type4": 59, "total": 63_701},
        )
        if frozen[2]
        else set()
    )
    covered = certified | excluded
    complete = (
        frozen == [True, True, True]
        and certified.isdisjoint(excluded)
        and len(covered) == 76_456
        and unresolved <= covered
        and covered == _hard_identity_universe()
    )
    _expect(not frozen[1] or frozen[0], "evaluator certificate requires an implementation manifest")
    _expect(not (certified & excluded), "certified and excluded route identities must be disjoint")
    if frozen == [True, True, True]:
        _expect(complete, "certified and excluded hard identities must exactly cover all 76,456 hard rubrics")
    states = _object(readiness, "program.readiness")
    expected_state = "ready" if complete else "blocked_until_route_partition_and_evaluator_certification"
    _expect(
        states.get("scalar_training") == expected_state,
        "unsupported hard routes must block scalar training until exact partition and evaluator certification",
    )


def _validate_route_resolution(artifact: JsonObject) -> set[str]:
    _exact_keys(
        artifact,
        {
            "schema_version",
            "id",
            "status",
            "taxonomy_sha256",
            "rtt_revision",
            "rtt_tree",
            "route_digest",
            "counts",
            "unsupported_identities",
        },
        "route resolution artifact",
    )
    counts = _object(artifact["counts"], "route resolution counts")
    _exact_keys(counts, {"hard", "route_resolvable", "unresolved"}, "route resolution counts")
    identities = _identity_set(artifact["unsupported_identities"], "route resolution identities")
    _expect(
        artifact["schema_version"] == 1
        and artifact["id"] == "hir_route_resolution_v1"
        and artifact["status"] == "route_resolvable_with_gaps"
        and artifact["taxonomy_sha256"] == HIR_TAXONOMY_SHA256
        and artifact["rtt_revision"] == "b1ab2fba9bece98674e5fa6e6c808d9d63235778"
        and artifact["rtt_tree"] == "f907acb9ba5ef13da38a3c02e2b2599c75cafd2a"
        and artifact["route_digest"] == "a367d5e688fa2996543b123ce7491c8e878515597751bcf38f5b33f9e57d3e22"
        and counts == {"hard": 76_456, "route_resolvable": 75_657, "unresolved": 799}
        and len(identities) == 799,
        "route resolution artifact is invalid",
    )
    return identities


def _validate_identity_manifest(
    artifact: JsonObject,
    expected_id: str,
    resolution_sha256: str,
    status: str,
    scalar_manifest_sha256: Any,
    counts: JsonObject,
) -> set[str]:
    _exact_keys(
        artifact,
        {
            "schema_version",
            "id",
            "status",
            "route_resolution_sha256",
            "scalar_data_manifest_sha256",
            "counts",
            "identities",
        },
        f"{status} identity manifest",
    )
    identities = _identity_set(artifact["identities"], f"{status} identities")
    _expect(
        artifact["schema_version"] == 1
        and artifact["id"] == expected_id
        and artifact["status"] == status
        and artifact["route_resolution_sha256"] == resolution_sha256
        and artifact["scalar_data_manifest_sha256"] == scalar_manifest_sha256
        and artifact["counts"] == counts,
        f"{status} identity manifest is invalid or mis-cross-linked",
    )
    return identities


def _validate_evaluator_certificate(
    artifact: JsonObject,
    expected_id: str,
    resolution_sha256: str,
    implementation_sha256: str,
    implemented: set[str],
    repo_root: Path,
    scalar_manifest_sha256: Any,
) -> set[str]:
    identities = _identity_set(artifact["identities"], "evaluator certificate identities")
    ordered = [json.loads(identity) for identity in sorted(implemented)]
    function_hashes = _type4_function_hashes(repo_root, implemented)
    try:
        expected = scalar_evaluator_certificate(
            repo_root,
            ordered,
            function_hashes,
            resolution_sha256,
            implementation_sha256,
            _string(scalar_manifest_sha256, "scalar data manifest hash"),
        )
    except (EvaluatorCertificationError, OSError, ValueError) as error:
        raise ProgramContractError(f"evaluator certificate evidence is invalid: {error}") from error
    _expect(
        artifact.get("id") == expected_id and identities == implemented and artifact == expected,
        "evaluator certificate is invalid, incomplete, or mis-cross-linked",
    )
    return identities


def _type4_function_hashes(repo_root: Path, identities: set[str]) -> set[str]:
    source = repo_root / "data/HIR_trainv1.jsonl"
    rows: dict[int | str, JsonObject] = {}
    try:
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                rows[row["id"]] = row
    except (OSError, json.JSONDecodeError, KeyError) as error:
        raise ProgramContractError(f"cannot load type4 evaluator sources: {error}") from error
    hashes: set[str] = set()
    for encoded in identities:
        identity = json.loads(encoded)
        row = rows.get(identity["row_id"])
        _expect(
            row is not None
            and identity["source"] == row.get("source") == "type4"
            and identity["route"] == "embedded_check_following",
            "implemented scalar identity is not a pinned type4 route",
        )
        try:
            code = row["ground_truth"]["functions"][identity["rubric_index"]]
        except (KeyError, IndexError, TypeError) as error:
            raise ProgramContractError("implemented scalar identity has no source function") from error
        hashes.add(hashlib.sha256(code.encode()).hexdigest())
    return hashes


def _validate_evaluator_evidence(
    artifact: JsonObject,
    certificate_id: str,
    kind: str,
    expected: set[str],
) -> tuple[set[str], int]:
    _exact_keys(
        artifact,
        {"schema_version", "certificate_id", "kind", "coverage_unit", "outcomes"},
        f"{kind} evaluator evidence",
    )
    unit = artifact["coverage_unit"]
    _expect(unit in {"identity", "family"}, "evaluator evidence coverage_unit must be identity or family")
    outcomes = artifact["outcomes"]
    _expect(isinstance(outcomes, list) and outcomes, "evaluator evidence outcomes must be non-empty")
    covered: set[str] = set()
    failures = 0
    for index, value in enumerate(outcomes):
        outcome = _object(value, f"{kind} evaluator evidence.outcomes[{index}]")
        if unit == "identity":
            _exact_keys(outcome, {"identity", "passed"}, "identity evaluator outcome")
            identities = _identity_set([outcome["identity"]], "identity evaluator outcome")
        else:
            _exact_keys(
                outcome, {"family_id", "identity_count", "identities_sha256", "passed"}, "family evaluator outcome"
            )
            _expect(isinstance(outcome["family_id"], str) and outcome["family_id"], "evaluator family id is invalid")
            identities = {identity for identity in expected if _identity_family(identity) == outcome["family_id"]}
            _expect(
                identities
                and outcome["identity_count"] == len(identities)
                and outcome["identities_sha256"] == _encoded_identities_sha256(identities),
                "evaluator family coverage is invalid",
            )
        _expect(isinstance(outcome["passed"], bool), "evaluator outcome passed must be boolean")
        _expect(covered.isdisjoint(identities), "evaluator evidence identities overlap")
        covered |= identities
        failures += 0 if outcome["passed"] else len(identities)
    _expect(
        artifact["schema_version"] == 1 and artifact["certificate_id"] == certificate_id and artifact["kind"] == kind,
        "evaluator evidence is mis-cross-linked",
    )
    return covered, failures


def _identity_family(encoded: str) -> str:
    identity = json.loads(encoded)
    return f"{identity['source']}:{identity['route']}"


def _encoded_identities_sha256(identities: set[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(identities)) + "\n").encode()).hexdigest()


def _identity_set(value: Any, name: str) -> set[str]:
    _expect(isinstance(value, list), f"{name} must be an array")
    encoded: list[str] = []
    for index, item in enumerate(value):
        identity = _object(item, f"{name}[{index}]")
        _exact_keys(identity, {"source", "row_id", "rubric_index", "route"}, f"{name}[{index}]")
        _expect(
            identity["source"] in {"type1", "type2", "type3", "type4"}
            and isinstance(identity["row_id"], (int, str))
            and not isinstance(identity["row_id"], bool)
            and isinstance(identity["rubric_index"], int)
            and not isinstance(identity["rubric_index"], bool)
            and identity["rubric_index"] >= 0
            and isinstance(identity["route"], str)
            and bool(identity["route"]),
            f"{name}[{index}] is invalid",
        )
        encoded.append(json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    _expect(encoded == sorted(encoded) and len(encoded) == len(set(encoded)), f"{name} must be sorted and unique")
    return set(encoded)


@lru_cache(maxsize=1)
def _hard_identity_universe() -> set[str]:
    from rdan_grpo.hir import classify_hir_row

    source = Path(__file__).resolve().parents[2] / "data/HIR_trainv1.jsonl"
    _expect(_file_sha256(source) == HIR_SOURCE_SHA256, "pinned HIR source bytes are unavailable or changed")
    identities: list[JsonObject] = []
    try:
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                for index, hard in enumerate(classify_hir_row(row)):
                    if not hard:
                        continue
                    ground_truth = row["ground_truth"]
                    if row["source"] in {"type1", "type2"}:
                        route = ground_truth["instruction_id_list"][index]
                    elif row["source"] == "type3":
                        constraint = ground_truth["constraints"][index]
                        route = f"{constraint[0]}_{constraint[1]}"
                    else:
                        route = "embedded_check_following"
                    identities.append(
                        {"source": row["source"], "row_id": row["id"], "rubric_index": index, "route": route}
                    )
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as error:
        raise ProgramContractError(f"cannot derive pinned hard identity universe: {error}") from error
    universe = _identity_set(
        sorted(
            identities, key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        ),
        "pinned hard identity universe",
    )
    _expect(len(universe) == 76_456, "pinned hard identity universe count is invalid")
    return universe


def _validate_artifacts(artifacts: JsonObject) -> None:
    _exact_keys(artifacts, {"schema_version", "id", "compact", "large", "integrity"}, "artifacts")
    _expect(artifacts["schema_version"] == 1 and artifacts["id"] == "qwen_evidence", "artifact identity is invalid")
    _expect(
        artifacts["compact"]
        == {
            "storage": "git",
            "root": "results",
            "required": [
                "resolved_configs",
                "environment_lock",
                "model_hash",
                "tokenizer_hash",
                "data_hash",
                "evaluator_hash",
                "eval_cards",
                "result_schema",
                "sha256_manifest",
            ],
        },
        "compact artifact contract is invalid",
    )
    _expect(
        artifacts["large"]
        == {
            "storage": "huggingface",
            "dataset_repository": "beingamanforever/RDAN-GRPO-Qwen3-4B-evidence",
            "model_repository": "beingamanforever/RDAN-GRPO-Qwen3-4B",
            "required": [
                "raw_generations",
                "judge_records",
                "decomposed_rewards",
                "decomposed_advantages",
                "failure_records",
                "checkpoints",
                "bootstrap_samples",
            ],
        },
        "large artifact contract is invalid",
    )
    _expect(
        artifacts["integrity"] == {"algorithm": "sha256", "manifest_covers": "all_compact_and_large_artifacts"},
        "artifact integrity contract is invalid",
    )


def _validate_lifecycle_artifacts(bundle: ProgramBundle) -> None:
    refs = _object(bundle.program["lifecycle_artifacts"], "program.lifecycle_artifacts")
    expected_paths = {
        "scalar_data": "configs/artifacts/qwen_scalar_data_manifest.json",
        "response_data": "configs/artifacts/qwen_merged_rl_data_manifest.json",
        "judge_calibration": "configs/artifacts/qwen_judge_calibration.json",
        "runtime_parity": "configs/artifacts/qwen_runtime_parity.json",
        "vllm_runtime_parity": "configs/artifacts/qwen_vllm_runtime_parity.json",
        "no_update": "configs/artifacts/qwen_no_update_certificate.json",
        "token_labels": "configs/artifacts/qwen_token_labels.json",
        "discriminator_checkpoint": "configs/artifacts/qwen_discriminator_checkpoint.json",
        "discriminator_certificate": "configs/artifacts/qwen_discriminator_certificate.json",
        "gpqa_access": "configs/artifacts/gpqa_access.json",
    }
    _exact_keys(refs, set(expected_paths), "program.lifecycle_artifacts")
    frozen: dict[str, bool] = {}
    for name, path in expected_paths.items():
        reference = _object(refs[name], f"program.lifecycle_artifacts.{name}")
        _exact_keys(reference, {"status", "path", "artifact_id", "sha256"}, f"lifecycle {name} reference")
        pending = reference == {"status": "pending", "path": path, "artifact_id": "pending", "sha256": "pending"}
        ready = (
            reference["status"] == "frozen"
            and reference["path"] == path
            and _frozen_pin(reference["artifact_id"], reference["sha256"])
        )
        _expect(pending or ready, f"lifecycle {name} reference is invalid")
        _expect((name in bundle.lifecycle_artifacts) is ready, f"lifecycle {name} artifact load state is invalid")
        if ready:
            _expect(
                bundle.artifact_hashes[f"lifecycle_{name}"] == reference["sha256"], f"lifecycle {name} hash mismatch"
            )
        frozen[name] = ready

    if frozen["scalar_data"]:
        _validate_scalar_data_manifest(
            bundle.lifecycle_artifacts["scalar_data"],
            refs["scalar_data"],
            bundle.repo_root,
        )
    if frozen["response_data"]:
        _validate_response_data_manifest(
            bundle.lifecycle_artifacts["response_data"],
            refs["response_data"],
            bundle.repo_root,
        )
    if frozen["judge_calibration"]:
        _validate_judge_calibration(
            bundle.lifecycle_artifacts["judge_calibration"],
            refs["judge_calibration"],
            bundle.repo_root,
            bundle.judge,
        )
    if frozen["runtime_parity"]:
        _validate_runtime_parity(
            bundle.lifecycle_artifacts["runtime_parity"],
            refs["runtime_parity"],
            _object(bundle.program["same_backend_configs"], "program.same_backend_configs"),
        )
    if frozen["vllm_runtime_parity"]:
        _validate_vllm_runtime_parity(
            bundle.lifecycle_artifacts["vllm_runtime_parity"],
            refs["vllm_runtime_parity"],
            _object(bundle.program["same_backend_configs"], "program.same_backend_configs"),
        )
    if frozen["no_update"]:
        _validate_no_update_artifact(bundle.lifecycle_artifacts["no_update"], refs["no_update"], bundle)
    if frozen["token_labels"]:
        _validate_token_labels(bundle.lifecycle_artifacts["token_labels"], refs, frozen, bundle.repo_root)
    if frozen["discriminator_checkpoint"]:
        _validate_discriminator_checkpoint(
            bundle.lifecycle_artifacts["discriminator_checkpoint"], refs, frozen, bundle.repo_root
        )
    if frozen["discriminator_certificate"]:
        _validate_discriminator_certificate(bundle.lifecycle_artifacts["discriminator_certificate"], refs, frozen)

    readiness = bundle.program["readiness"]
    _expect(
        readiness["judge"] == ("ready" if frozen["judge_calibration"] else "blocked_until_frozen_calibration"),
        "judge readiness does not match calibration evidence",
    )
    launch_ready = (
        frozen["scalar_data"]
        and frozen["response_data"]
        and frozen["judge_calibration"]
        and frozen["runtime_parity"]
        and frozen["vllm_runtime_parity"]
        and frozen["no_update"]
        and readiness["scalar_training"] == "ready"
    )
    _expect(
        readiness["launch"] == ("ready" if launch_ready else "blocked_until_all_launch_artifacts_frozen"),
        "launch readiness does not match lifecycle evidence",
    )
    token_ready = (
        launch_ready
        and frozen["token_labels"]
        and frozen["discriminator_checkpoint"]
        and frozen["discriminator_certificate"]
    )
    _expect(
        readiness["token_training"]
        == ("ready" if token_ready else "blocked_until_labels_discriminator_and_scalar_program_pass"),
        "token training readiness does not match lifecycle evidence",
    )
    if frozen["gpqa_access"]:
        _validate_gpqa_access(bundle.lifecycle_artifacts["gpqa_access"], refs["gpqa_access"])
    gpqa_benchmark = next(item for item in bundle.program["benchmarks"] if item["id"] == "gpqa")
    expected_gpqa = (
        {
            "id": "gpqa",
            "enabled": True,
            "scope": "evaluation_only_ood",
            "execution_state": "runnable_frozen_access",
            "transition_gate": "revision_hash_and_access_manifest_frozen",
        }
        if frozen["gpqa_access"]
        else {
            "id": "gpqa",
            "enabled": True,
            "scope": "evaluation_only_ood",
            "execution_state": "blocked_gated_access",
            "transition_gate": "revision_hash_and_access_manifest_frozen",
        }
    )
    _expect(gpqa_benchmark == expected_gpqa, "GPQA benchmark state does not match access evidence")
    _expect(
        readiness["gpqa"]
        == ("ready" if frozen["gpqa_access"] else "blocked_until_revision_hash_and_access_manifest_frozen"),
        "GPQA readiness does not match access evidence",
    )


def _validate_scalar_data_manifest(
    artifact: JsonObject,
    reference: JsonObject,
    repo_root: Path,
) -> None:
    train = _object(artifact.get("train_config"), "scalar data train config")
    train_path = _resolve_under(
        repo_root,
        repo_root / _string(train.get("path"), "scalar data train path"),
        "scalar data train path",
    )
    certified_path = repo_root / "configs/artifacts/hir_scalar_certified_manifest.json"
    try:
        expected = inspect_scalar_gate(repo_root, train_path, certified_path).manifest
    except (ScalarDataError, OSError, ValueError) as error:
        raise ProgramContractError(f"scalar data evidence is invalid: {error}") from error
    _expect(
        artifact == expected
        and artifact.get("id") == reference["artifact_id"]
        and train.get("sha256") == _file_sha256(train_path),
        "scalar data manifest is invalid or not linked to its lineage YAML and dataset bytes",
    )


def _validate_response_data_manifest(artifact: JsonObject, reference: JsonObject, repo_root: Path) -> None:
    program_path = repo_root / "configs/program/qwen_first.json"
    try:
        identity = response_data_identity(program_path)
    except (ResponseIdentityError, OSError, ValueError) as error:
        raise ProgramContractError(f"response data evidence is invalid: {error}") from error
    _expect(
        artifact.get("id") == reference["artifact_id"] == identity["artifact_id"]
        and reference["sha256"] == identity["manifest_sha256"]
        and identity["records"] == 18_096,
        "response data manifest is invalid or not linked to exact merged bytes",
    )


def _validate_scalar_data(data: JsonObject, dev_value: Any, repo_root: Path) -> None:
    _exact_keys(
        data,
        {"path", "sha256", "schema", "records", "row_ids", "row_ids_sha256", "output_digests"},
        "scalar data",
    )
    path = _resolve_under(repo_root, repo_root / _string(data["path"], "scalar data path"), "scalar data path")
    dev = _object(dev_value, "scalar data dev split")
    _exact_keys(dev, {"id", "sha256", "row_ids"}, "scalar data dev split")
    row_ids = data["row_ids"]
    fields = ["row_id", "prompt", "rubrics", "hard_mask"]
    rows = _load_dataset_rows(path, fields, "scalar dataset")
    _expect(
        _file_sha256(path) == data["sha256"]
        and data["schema"] == {"format": "jsonl", "fields": fields}
        and isinstance(row_ids, list)
        and all(_valid_hir_id(row_id) for row_id in row_ids)
        and len(row_ids) == len(set(row_ids)) == data["records"]
        and data["row_ids_sha256"] == _json_sha256({"row_ids": row_ids})
        and isinstance(data["output_digests"], list)
        and len(data["output_digests"]) == data["records"]
        and all(_sha256(value) for value in data["output_digests"])
        and _frozen_pin(dev["id"], dev["sha256"])
        and isinstance(dev["row_ids"], list)
        and all(_valid_hir_id(row_id) for row_id in dev["row_ids"])
        and set(row_ids).isdisjoint(dev["row_ids"]),
        "scalar data bytes, schema, identities, outputs, or dev disjointness are invalid",
    )
    _expect(
        [row["row_id"] for row in rows] == row_ids and [_json_sha256(row) for row in rows] == data["output_digests"],
        "scalar dataset contents do not match its manifest",
    )


def _validate_judge_calibration(
    artifact: JsonObject,
    reference: JsonObject,
    repo_root: Path,
    judge: JsonObject,
) -> None:
    _exact_keys(
        artifact,
        {
            "schema_version",
            "id",
            "status",
            "judge_config_sha256",
            "implementation",
            "case_manifest",
            "call_manifest",
            "preflight",
            "debug_canary",
            "selection",
            "selected_reasoning_effort",
            "outcomes",
        },
        "judge calibration artifact",
    )
    implementation = _object(artifact["implementation"], "judge calibration implementation")
    cases = _object(artifact["case_manifest"], "judge case manifest")
    calls = _object(artifact["call_manifest"], "judge call manifest")
    preflight = _object(artifact["preflight"], "judge calibration preflight")
    canary = _object(artifact["debug_canary"], "judge debug canary")
    selection = _object(artifact["selection"], "judge calibration selection")
    outcomes = _object(artifact["outcomes"], "judge calibration outcomes")
    _exact_keys(
        implementation,
        {"path", "sha256", "one_call_per_response", "strict_schema", "max_retries"},
        "judge implementation",
    )
    _exact_keys(
        cases,
        {
            "path",
            "file_sha256",
            "sha256",
            "cases",
            "debug",
            "debug_redacted",
            "labeled",
            "heldout",
            "corpus_summary",
            "rows",
        },
        "judge case manifest",
    )
    _exact_keys(calls, {"sha256", "raw_file_sha256", "calls", "rows"}, "judge call manifest")
    _exact_keys(
        preflight,
        {"catalog", "catalog_sha256", "endpoints", "endpoints_sha256", "provider"},
        "judge calibration preflight",
    )
    _exact_keys(
        canary,
        {
            "generation_id",
            "provider",
            "model",
            "parameter_names",
            "upstream_body_sha256",
            "generation_metadata_polls",
            "request_sha256",
            "valid",
        },
        "judge debug canary",
    )
    _exact_keys(
        selection,
        {
            "bootstrap_samples",
            "bootstrap_seed",
            "noninferiority_margin",
            "point_scores",
            "reference_effort",
            "paired_difference_ci95",
            "qualifying_efforts",
            "selected_effort",
        },
        "judge calibration selection",
    )
    _exact_keys(
        outcomes,
        {
            "calls",
            "cases",
            "invalid_calls",
            "labeled_exact_matches",
            "selected_labeled_exact_matches",
            "heldout_exact_matches",
            "heldout_agreements",
            "injection_exact_matches",
            "injection_cases",
            "thresholds",
            "metrics",
            "threshold_passes",
            "failures",
        },
        "judge calibration outcomes",
    )
    thresholds = _object(outcomes["thresholds"], "judge calibration thresholds")
    metrics = _object(outcomes["metrics"], "judge calibration metrics")
    threshold_passes = _object(outcomes["threshold_passes"], "judge calibration threshold passes")
    implementation_path = _resolve_under(
        repo_root,
        repo_root / _string(implementation["path"], "judge implementation path"),
        "judge implementation path",
    )
    case_path = _resolve_under(
        repo_root,
        repo_root / _string(cases["path"], "judge calibration cases path"),
        "judge calibration cases path",
    )
    case_file_sha256 = _file_sha256(case_path)
    source_case_rows = _calibration_case_rows(case_path)
    rows = calls["rows"]
    safe_rows = rows if isinstance(rows, list) and all(isinstance(row, dict) for row in rows) else []
    case_rows = cases["rows"]
    safe_case_rows = (
        case_rows if isinstance(case_rows, list) and all(isinstance(row, dict) for row in case_rows) else []
    )
    case_row_shape = len(safe_case_rows) == 76 and all(
        set(row) == {"case_id", "split", "tags", "sha256"}
        and isinstance(row["case_id"], str)
        and row["case_id"]
        and row["split"] in {"debug", "labeled", "heldout"}
        and isinstance(row["tags"], list)
        and all(isinstance(tag, str) for tag in row["tags"])
        and _sha256(row["sha256"])
        for row in safe_case_rows
    )
    efforts = {"none", "low", "medium"}
    row_shape = isinstance(rows, list) and all(
        isinstance(row, dict)
        and set(row)
        == {
            "case_id",
            "split",
            "effort",
            "repeat",
            "generation_id",
            "valid",
            "exact_match",
            "scores_sha256",
            "raw_record_sha256",
            "provenance_sha256",
        }
        and isinstance(row["case_id"], str)
        and row["case_id"]
        and row["split"] in {"debug", "labeled", "heldout"}
        and row["effort"] in efforts
        and isinstance(row["repeat"], int)
        and not isinstance(row["repeat"], bool)
        and isinstance(row["generation_id"], str)
        and row["generation_id"].startswith("gen-")
        and row["valid"] is True
        and (isinstance(row["exact_match"], bool) if row["split"] != "debug" else row["exact_match"] is None)
        and _sha256(row["scores_sha256"])
        and _sha256(row["raw_record_sha256"])
        and _sha256(row["provenance_sha256"])
        for row in safe_rows
    )
    valid_rows = safe_rows if row_shape else []
    source_splits = {row["case_id"]: row["split"] for row in source_case_rows}
    debug_ids = {row["case_id"] for row in valid_rows if row["split"] == "debug"}
    labeled_ids = {row["case_id"] for row in valid_rows if row["split"] == "labeled"}
    heldout_ids = {row["case_id"] for row in valid_rows if row["split"] == "heldout"}
    call_rows = {(row["case_id"], row["effort"], row["repeat"]): row for row in valid_rows}
    selected_value = artifact["selected_reasoning_effort"]
    selected_effort = selected_value if selected_value in efforts else ""
    labeled_shape = row_shape and all(
        {(row["effort"], row["repeat"]) for row in valid_rows if row["case_id"] == case_id}
        == {(effort, 1) for effort in efforts}
        for case_id in labeled_ids
    )
    heldout_shape = row_shape and all(
        {(row["effort"], row["repeat"]) for row in valid_rows if row["case_id"] == case_id}
        == {(selected_effort, 1), (selected_effort, 2)}
        for case_id in heldout_ids
    )
    generation_ids = [row["generation_id"] for row in valid_rows]
    case_ids = debug_ids | labeled_ids | heldout_ids
    labeled_case_ids = [row["case_id"] for row in source_case_rows if row["split"] == "labeled"]
    heldout_case_ids = [row["case_id"] for row in source_case_rows if row["split"] == "heldout"]
    indicators = {
        effort: [
            int(call_rows.get((case_id, effort, 1), {}).get("exact_match", False)) for case_id in labeled_case_ids
        ]
        for effort in efforts
    }
    selection_evidence = select_reasoning_effort(indicators)
    selected_labeled = sum(
        int(call_rows.get((case_id, selected_effort, 1), {}).get("exact_match", False)) for case_id in labeled_case_ids
    )
    heldout_exact = sum(
        int(call_rows.get((case_id, selected_effort, 1), {}).get("exact_match", False)) for case_id in heldout_case_ids
    )
    heldout_agreements = sum(
        bool(call_rows.get((case_id, selected_effort, 1), {}).get("scores_sha256"))
        and call_rows.get((case_id, selected_effort, 1), {}).get("scores_sha256")
        == call_rows.get((case_id, selected_effort, 2), {}).get("scores_sha256")
        for case_id in heldout_case_ids
    )
    injection_ids = {row["case_id"] for row in safe_case_rows if "injection" in row.get("tags", [])}
    injection_exact = sum(
        int(call_rows.get((case_id, selected_effort, 1), {}).get("exact_match", False)) for case_id in injection_ids
    )
    injection_total = len(injection_ids)
    invalid_calls = sum(not row["valid"] for row in valid_rows)
    expected_thresholds = {
        "valid_call_rate": 1.0,
        "selected_labeled_exact_accuracy": 0.85,
        "heldout_exact_accuracy": 0.85,
        "injection_exact_accuracy": 1.0,
        "heldout_duplicate_agreement_rate": 0.96,
    }
    expected_summary = {
        "rubrics": 206,
        "categories": [
            "Desired_Writing_Style",
            "Hierarchical_Instructions",
            "Item_Listing_Details",
            "Key_Formatting",
            "Morphological_Constraints",
            "Multi-lingual_Constraints",
            "Paragraphs_Constraints",
            "Semantic_elements",
            "Special_Output_Format",
            "Specific_Grammatical_Structure",
            "Specific_Literary_Devices",
            "Specific_Sentence",
        ],
        "cases_with_6_to_12_rubrics": 12,
        "rubric_count_bins": {"1": 21, "2": 41, "3": 1, "6": 4, "8": 4, "10": 2, "12": 2},
        "multilingual_cases": 14,
        "unicode_cases": 14,
        "injection_cases": 8,
        "long_instruction_cases": 10,
        "long_response_cases": 11,
    }
    expected_metrics = {
        "valid_call_rate": (len(valid_rows) - invalid_calls) / 200,
        "selected_labeled_exact_accuracy": selected_labeled / 49,
        "heldout_exact_accuracy": heldout_exact / 26,
        "injection_exact_accuracy": injection_exact / injection_total if injection_total else -1.0,
        "heldout_duplicate_agreement_rate": heldout_agreements / 26,
    }
    expected_passes = {name: expected_metrics[name] >= minimum for name, minimum in expected_thresholds.items()}
    _expect(
        artifact["schema_version"] == 1
        and artifact["id"] == reference["artifact_id"]
        and artifact["status"] == "calibrated"
        and artifact["judge_config_sha256"] == _file_sha256(repo_root / "configs/judges/openrouter_luna.json")
        and implementation["sha256"] == _file_sha256(implementation_path)
        and implementation["one_call_per_response"] is True
        and implementation["strict_schema"] is True
        and implementation["max_retries"] == 0
        and cases["path"] == JUDGE_CALIBRATION_CASES_PATH
        and case_file_sha256 == cases["file_sha256"]
        and safe_case_rows == source_case_rows
        and cases["cases"] == len(source_case_rows) == 76
        and cases["debug"] == 1
        and cases["debug_redacted"] is True
        and cases["labeled"] == 49
        and cases["heldout"] == 26
        and cases["corpus_summary"] == expected_summary
        and cases["sha256"] == _json_sha256({"rows": safe_case_rows})
        and case_row_shape
        and len({row["case_id"] for row in safe_case_rows}) == 76
        and sum(row["split"] == "debug" for row in safe_case_rows) == 1
        and sum(row["split"] == "labeled" for row in safe_case_rows) == 49
        and sum(row["split"] == "heldout" for row in safe_case_rows) == 26
        and calls["calls"] == len(safe_rows) == 200
        and calls["sha256"] == _json_sha256({"rows": safe_rows})
        and _sha256(calls["raw_file_sha256"])
        and row_shape
        and len(call_rows) == 200
        and all(source_splits.get(row["case_id"]) == row["split"] for row in valid_rows)
        and len(debug_ids) == 1
        and len(labeled_ids) == 49
        and len(heldout_ids) == 26
        and len(case_ids) == 76
        and labeled_shape
        and heldout_shape
        and len(generation_ids) == len(set(generation_ids)) == 200
        and selected_value in efforts
        and selection == selection_evidence
        and artifact["selected_reasoning_effort"] == selection["selected_effort"]
        and preflight["catalog_sha256"] == _json_sha256(preflight["catalog"])
        and preflight["endpoints_sha256"] == _json_sha256(preflight["endpoints"])
        and preflight["catalog"].get("id") == "openai/gpt-5.6-luna"
        and preflight["catalog"].get("canonical_slug") == "openai/gpt-5.6-luna-20260709"
        and preflight["endpoints"].get("id") == "openai/gpt-5.6-luna"
        and preflight["endpoints"].get("provider") == "openai"
        and preflight["endpoints"].get("model") == "openai/gpt-5.6-luna"
        and preflight["endpoints"].get("tag") == "openai"
        and preflight["endpoints"].get("status") == 0
        and preflight["provider"] == "openai"
        and canary["generation_id"] == safe_rows[0]["generation_id"]
        and canary["provider"] == "OpenAI"
        and canary["model"]
        in {"openai/gpt-5.6-luna", "openai/gpt-5.6-luna-20260709", "gpt-5.6-luna", "gpt-5.6-luna-20260709"}
        and canary["parameter_names"] == ["max_tokens", "reasoning_effort", "response_format", "seed"]
        and _sha256(canary["upstream_body_sha256"])
        and isinstance(canary["generation_metadata_polls"], int)
        and not isinstance(canary["generation_metadata_polls"], bool)
        and 1 <= canary["generation_metadata_polls"] <= judge["generation_metadata_poll"]["attempts"]
        and _sha256(canary["request_sha256"])
        and canary["valid"] is True
        and outcomes["calls"] == len(valid_rows) == 200
        and outcomes["cases"] == len(source_case_rows) == 76
        and outcomes["invalid_calls"] == invalid_calls == 0
        and outcomes["labeled_exact_matches"] == sum(sum(values) for values in indicators.values())
        and outcomes["selected_labeled_exact_matches"] == selected_labeled
        and outcomes["heldout_exact_matches"] == heldout_exact
        and outcomes["heldout_agreements"] == heldout_agreements
        and outcomes["injection_exact_matches"] == injection_exact
        and outcomes["injection_cases"] == len(injection_ids) == 8
        and thresholds == expected_thresholds
        and metrics == expected_metrics
        and threshold_passes == expected_passes
        and all(value is True for value in threshold_passes.values())
        and outcomes["failures"] == sum(not passed for passed in expected_passes.values()) == 0,
        "judge calibration artifact is invalid or incomplete",
    )


def _calibration_case_rows(path: Path) -> list[JsonObject]:
    rows: list[JsonObject] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                raise ProgramContractError(f"judge calibration cases line {line_number} is blank")
            case = _object(
                json.loads(line, object_pairs_hook=_unique_object),
                f"judge calibration cases line {line_number}",
            )
            _expect(
                case.get("provenance")
                == {"source": "curated_hir_soft_taxonomy_v1", "review_status": "human_reviewed"},
                "judge calibration cases require explicit human-reviewed provenance",
            )
            rows.append(
                {
                    "case_id": case.get("case_id"),
                    "split": case.get("split"),
                    "tags": case.get("tags", []),
                    "sha256": _json_sha256(case),
                }
            )
    except (OSError, json.JSONDecodeError) as error:
        raise ProgramContractError(f"cannot load judge calibration cases: {error}") from error
    return rows


def _validate_runtime_parity(artifact: JsonObject, reference: JsonObject, configs: JsonObject) -> None:
    _validate_runtime_parity_keys(artifact)
    model = _object(artifact["model"], "runtime parity model")
    tokenizer = _object(artifact["tokenizer"], "runtime parity tokenizer")
    template = _object(artifact["chat_template"], "runtime parity chat template")
    backend = _object(artifact["runtime_backend"], "runtime parity backend")
    evidence = _object(artifact["rollout_logprob_evidence"], "runtime parity rollout logprob evidence")
    receipt = _object(artifact["weight_receipt"], "runtime parity weight receipt")
    thresholds = _validate_runtime_parity_nested_keys(backend, receipt, evidence)
    diagnostic = _object(configs["diagnostic"], "same-backend diagnostic config")
    production = _object(configs["production"], "same-backend production config")
    valid = (
        _runtime_parity_identity_valid(artifact, reference, model, tokenizer, template)
        and _runtime_parity_backend_valid(backend, diagnostic, production)
        and _runtime_parity_evidence_valid(evidence, thresholds, receipt, backend)
    )
    _expect(valid, "model, tokenizer, template, or logprob parity artifact is invalid")


def _validate_vllm_runtime_parity(artifact: JsonObject, reference: JsonObject, configs: JsonObject) -> None:
    diagnostic = _object(configs["vllm_diagnostic"], "vLLM diagnostic config")
    production = _object(configs["production"], "same-backend production config")
    try:
        validate_vllm_runtime_parity(
            artifact,
            artifact_id=_string(reference["artifact_id"], "vLLM parity artifact id"),
            model=MODEL_NAME,
            revision=MODEL_REVISION,
            rtt_revision=RTT_REVISION,
            parity_config_sha256=_string(diagnostic["sha256"], "vLLM diagnostic config hash"),
            production_config_sha256=_string(production["sha256"], "production config hash"),
        )
    except VLLMParityError as error:
        raise ProgramContractError(str(error)) from error


def _validate_runtime_parity_keys(artifact: JsonObject) -> None:
    keys = {
        "schema_version",
        "id",
        "status",
        "model",
        "tokenizer",
        "chat_template",
        "runtime_backend",
        "weight_receipt",
        "rollout_logprob_evidence",
    }
    _exact_keys(artifact, keys, "runtime parity artifact")


def _validate_runtime_parity_nested_keys(
    backend: JsonObject,
    receipt: JsonObject,
    evidence: JsonObject,
) -> JsonObject:
    _exact_keys(
        backend,
        {
            "train_config_sha256",
            "production_train_config_sha256",
            "production_resolved_config_sha256",
            "preflight_train_config_sha256",
            "preflight_resolved_config_sha256",
            "resolved_config_sha256",
            "actor_train_strategy",
            "actor_infer_strategy",
            "transformer_impl",
            "rtt_revision",
            *GENERATION_SOURCE_IDENTITY,
        },
        "runtime parity backend",
    )
    _exact_keys(
        receipt,
        {"transaction_id", "artifact_sha256", "resolved_config_sha256"},
        "runtime parity weight receipt",
    )
    evidence_keys = {
        "prompt_response_tokens_sha256",
        "responses",
        "optimizer_updates",
        "infer_logprobs_source",
        "actor_train_recomputed",
        "actor_boundary_observed",
        "compared_tokens",
        "max_abs_error",
        "mean_abs_error",
        "thresholds",
    }
    if backend.get("actor_infer_strategy") == "hf_infer":
        evidence_keys.update({"blocking_surface", "diagnostic_surface", "surface_comparisons"})
    _exact_keys(evidence, evidence_keys, "runtime parity rollout logprob evidence")
    thresholds = _object(evidence["thresholds"], "runtime parity thresholds")
    _exact_keys(thresholds, {"max_abs_error_at_most", "mean_abs_error_at_most"}, "runtime parity thresholds")
    return thresholds


def _runtime_parity_identity_valid(
    artifact: JsonObject,
    reference: JsonObject,
    model: JsonObject,
    tokenizer: JsonObject,
    template: JsonObject,
) -> bool:
    return bool(
        artifact["schema_version"] == 2
        and artifact["id"] == reference["artifact_id"]
        and artifact["status"] == "parity_passed"
        and model == {"model": MODEL_NAME, "revision": MODEL_REVISION, "snapshot_sha256": model.get("snapshot_sha256")}
        and _sha256(model["snapshot_sha256"])
        and tokenizer
        == {"model": MODEL_NAME, "revision": MODEL_REVISION, "files_sha256": tokenizer.get("files_sha256")}
        and _sha256(tokenizer["files_sha256"])
        and template == {"source": "pinned_tokenizer", "enable_thinking": False, "sha256": template.get("sha256")}
        and _sha256(template["sha256"])
    )


def _runtime_parity_backend_valid(
    backend: JsonObject,
    diagnostic: JsonObject,
    production: JsonObject,
) -> bool:
    expected = {
        "train_config_sha256": diagnostic.get("sha256"),
        "production_train_config_sha256": production.get("sha256"),
        "production_resolved_config_sha256": backend.get("production_resolved_config_sha256"),
        "preflight_train_config_sha256": backend.get("preflight_train_config_sha256"),
        "preflight_resolved_config_sha256": backend.get("preflight_resolved_config_sha256"),
        "resolved_config_sha256": backend.get("resolved_config_sha256"),
        "actor_train_strategy": "fsdp2_train",
        "actor_infer_strategy": "hf_infer",
        "transformer_impl": "huggingface",
        "rtt_revision": RTT_REVISION,
        **GENERATION_SOURCE_IDENTITY,
    }
    return bool(
        backend
        == {
            **expected,
        }
        and _sha256(backend["resolved_config_sha256"])
        and _sha256(backend["preflight_train_config_sha256"])
        and _sha256(backend["production_resolved_config_sha256"])
        and _sha256(backend["preflight_resolved_config_sha256"])
        and production.get("status") == "frozen"
        and _sha256(production.get("sha256"))
    )


def _runtime_parity_evidence_valid(
    evidence: JsonObject,
    thresholds: JsonObject,
    receipt: JsonObject,
    backend: JsonObject,
) -> bool:
    surfaces = evidence.get("surface_comparisons")
    blocking = surfaces.get("infer_full_vs_actor_full") if isinstance(surfaces, dict) else None
    diagnostic = surfaces.get("generation_vs_infer_full") if isinstance(surfaces, dict) else None
    surface_keys = {
        "compared_tokens",
        "source_mean_logprob",
        "target_mean_logprob",
        "signed_mean_difference",
        "rmse",
        "max_abs_error",
        "mean_abs_error",
    }
    surface_values = [blocking, diagnostic]
    valid_surfaces = bool(
        evidence.get("blocking_surface") == "infer_full_vs_actor_full"
        and evidence.get("diagnostic_surface") == "generation_vs_infer_full"
        and isinstance(surfaces, dict)
        and set(surfaces) == {"infer_full_vs_actor_full", "generation_vs_infer_full"}
        and all(isinstance(surface, dict) and set(surface) == surface_keys for surface in surface_values)
        and all(
            isinstance(surface["compared_tokens"], int)
            and not isinstance(surface["compared_tokens"], bool)
            and surface["compared_tokens"] > 0
            and all(
                isinstance(surface[key], (int, float))
                and not isinstance(surface[key], bool)
                and math.isfinite(surface[key])
                for key in surface_keys - {"compared_tokens"}
            )
            for surface in surface_values
            if isinstance(surface, dict)
        )
        and isinstance(blocking, dict)
        and evidence.get("compared_tokens") == blocking.get("compared_tokens")
        and evidence.get("max_abs_error") == blocking.get("max_abs_error")
        and evidence.get("mean_abs_error") == blocking.get("mean_abs_error")
    )
    return bool(
        receipt.get("resolved_config_sha256") == backend["resolved_config_sha256"]
        and isinstance(receipt.get("transaction_id"), str)
        and bool(receipt["transaction_id"])
        and _sha256(receipt.get("artifact_sha256"))
        and _sha256(evidence["prompt_response_tokens_sha256"])
        and isinstance(evidence["responses"], int)
        and evidence["responses"] >= 32
        and evidence["optimizer_updates"] == 0
        and evidence["infer_logprobs_source"] == "observed_hf_generation"
        and evidence["actor_train_recomputed"] is True
        and evidence["actor_boundary_observed"] is True
        and isinstance(evidence["compared_tokens"], int)
        and evidence["compared_tokens"] > 0
        and thresholds == {"max_abs_error_at_most": 0.001, "mean_abs_error_at_most": 0.0001}
        and isinstance(evidence["max_abs_error"], (int, float))
        and not isinstance(evidence["max_abs_error"], bool)
        and isinstance(evidence["mean_abs_error"], (int, float))
        and not isinstance(evidence["mean_abs_error"], bool)
        and math.isfinite(evidence["max_abs_error"])
        and math.isfinite(evidence["mean_abs_error"])
        and evidence["max_abs_error"] <= thresholds["max_abs_error_at_most"]
        and evidence["mean_abs_error"] <= thresholds["mean_abs_error_at_most"]
        and valid_surfaces
    )


def _validate_no_update_artifact(artifact: JsonObject, reference: JsonObject, bundle: ProgramBundle) -> None:
    _exact_keys(
        artifact,
        {
            "schema_version",
            "certificate_id",
            "ready",
            "method",
            "quality_weight",
            "config_sha256",
            "source_sha256",
            "metrics",
            "reasons",
        },
        "no-update artifact",
    )
    body = {key: value for key, value in artifact.items() if key != "certificate_id"}
    source_hashes = _object(artifact["source_sha256"], "no-update source hashes")
    metrics = _object(artifact["metrics"], "no-update metrics")
    _validate_no_update_metric_keys(metrics)
    response_data = _no_update_response_data(bundle)
    valid = (
        _no_update_identity_valid(artifact, reference, body, bundle)
        and _no_update_sources_valid(source_hashes, response_data, bundle)
        and _no_update_metrics_valid(metrics, bundle)
    )
    _expect(valid, "no-update artifact is invalid or not cross-linked to launch inputs")


def _validate_no_update_metric_keys(metrics: JsonObject) -> None:
    keys = {
        "prompt_count",
        "response_count",
        "group_size",
        "optimizer_updates",
        "batch_count",
        "valid_batch_count",
        "response_active_group_count",
        "response_active_group_rate",
        "quality_active_group_count",
        "quality_active_group_rate",
        "finite",
    }
    _exact_keys(metrics, keys, "no-update metrics")


def _no_update_response_data(bundle: ProgramBundle) -> Mapping[str, str | int]:
    try:
        return response_data_identity(bundle.repo_root / "configs/program/qwen_first.json")
    except (ResponseIdentityError, OSError, ValueError) as error:
        raise ProgramContractError(f"no-update response data identity is invalid: {error}") from error


def _no_update_identity_valid(
    artifact: JsonObject,
    reference: JsonObject,
    body: JsonObject,
    bundle: ProgramBundle,
) -> bool:
    return bool(
        artifact["schema_version"] == 2
        and artifact["certificate_id"] == reference["artifact_id"] == _json_sha256(body)
        and artifact["ready"] is True
        and artifact["method"] == "rtt_papo_response"
        and artifact["quality_weight"] == 0.5
        and artifact["config_sha256"] == bundle.program["launch_train_config"]["preflight_sha256"]
        and artifact["reasons"] == []
    )


def _no_update_sources_valid(
    source_hashes: JsonObject,
    response_data: Mapping[str, str | int],
    bundle: ProgramBundle,
) -> bool:
    parity = bundle.lifecycle_artifacts.get("runtime_parity")
    backend = parity.get("runtime_backend") if isinstance(parity, Mapping) else None
    composed = (
        source_hashes.get("train_resolved_config") == backend.get("production_resolved_config_sha256")
        and source_hashes.get("preflight_resolved_config") == backend.get("preflight_resolved_config_sha256")
        if isinstance(backend, Mapping)
        else _sha256(source_hashes.get("train_resolved_config"))
        and _sha256(source_hashes.get("preflight_resolved_config"))
    )
    return bool(
        source_hashes.get("train_config") == bundle.program["launch_train_config"]["sha256"]
        and composed
        and source_hashes.get("scalar_data_manifest") == bundle.program["lifecycle_artifacts"]["scalar_data"]["sha256"]
        and source_hashes.get("response_data_manifest")
        == bundle.program["lifecycle_artifacts"]["response_data"]["sha256"]
        and source_hashes.get("response_data_output") == response_data["output_sha256"]
        and source_hashes.get("response_data_config") == response_data["config_sha256"]
        and source_hashes.get("response_hir_manifest") == response_data["hir_manifest_sha256"]
        and source_hashes.get("rubrichub_rule_certificate") == response_data["rule_certificate_sha256"]
        and source_hashes.get("rubrichub_tokenizer_certificate") == response_data["tokenizer_certificate_sha256"]
        and source_hashes.get("evaluator_certificate")
        == bundle.program["hard_route_policy"]["evaluator_certificate"]["sha256"]
        and source_hashes.get("judge_calibration")
        == bundle.program["lifecycle_artifacts"]["judge_calibration"]["sha256"]
        and source_hashes.get("runtime_parity") == bundle.program["lifecycle_artifacts"]["runtime_parity"]["sha256"]
    )


def _no_update_metrics_valid(metrics: JsonObject, bundle: ProgramBundle) -> bool:
    prompt_count = metrics["prompt_count"]
    batch_count = metrics["batch_count"]
    response_active = metrics["response_active_group_count"]
    quality_active = metrics["quality_active_group_count"]
    integer_values = (
        prompt_count,
        metrics["response_count"],
        metrics["group_size"],
        metrics["optimizer_updates"],
        batch_count,
        metrics["valid_batch_count"],
        response_active,
        quality_active,
    )
    counts_valid = all(isinstance(value, int) and not isinstance(value, bool) for value in integer_values)
    rates = (metrics["response_active_group_rate"], metrics["quality_active_group_rate"])
    rates_valid = all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in rates
    )
    return bool(
        counts_valid
        and bundle.program["pilot"]["preflight"]["prompt_count"]["minimum"]
        <= prompt_count
        <= bundle.program["pilot"]["preflight"]["prompt_count"]["maximum"]
        and metrics["group_size"] == bundle.program["pilot"]["preflight"]["responses_per_prompt"] == 8
        and metrics["response_count"] == prompt_count * metrics["group_size"]
        and metrics["optimizer_updates"] == 0
        and batch_count > 0
        and metrics["valid_batch_count"] == batch_count
        and 0 < response_active <= prompt_count
        and 0 <= quality_active <= prompt_count
        and rates_valid
        and metrics["response_active_group_rate"] == response_active / prompt_count
        and metrics["quality_active_group_rate"] == quality_active / prompt_count
        and metrics["quality_active_group_rate"]
        >= bundle.program["pilot"]["gates"]["minimum_active_quality_group_rate"]
        and metrics["finite"] is True
    )


def _validate_token_labels(
    artifact: JsonObject,
    refs: JsonObject,
    frozen: Mapping[str, bool],
    repo_root: Path,
) -> None:
    _expect(frozen["scalar_data"] and frozen["no_update"], "token labels are blocked until scalar launch evidence")
    _exact_keys(
        artifact,
        {"schema_version", "id", "status", "provenance", "tokenizer", "data", "splits", "alignment"},
        "token labels artifact",
    )
    provenance = _object(artifact["provenance"], "token labels provenance")
    tokenizer = _object(artifact["tokenizer"], "token labels tokenizer")
    data = _object(artifact["data"], "token labels data")
    splits = _object(artifact["splits"], "token label splits")
    alignment = _object(artifact["alignment"], "token label alignment")
    _exact_keys(
        provenance,
        {
            "kind",
            "source_repository",
            "source_revision",
            "hir_source_sha256",
            "scalar_data_manifest_sha256",
            "generator_sha256",
        },
        "token labels provenance",
    )
    _exact_keys(tokenizer, {"model", "revision", "files_sha256"}, "token labels tokenizer")
    _exact_keys(data, {"path", "sha256", "format", "records", "tokens", "sample_ids_sha256"}, "token labels data")
    _exact_keys(splits, {"train", "validation", "test"}, "token label splits")
    _exact_keys(alignment, {"records_checked", "tokens_checked", "failures"}, "token label alignment")
    path = _external_artifact_path(data["path"], "token labels data path", repo_root)
    rows = _load_dataset_rows(path, ["sample_sha256", "row_id", "input_ids", "labels", "offsets"], "token labels")
    sample_ids: list[str] = []
    token_count = 0
    aligned = True
    for row in rows:
        content = {key: row[key] for key in ("row_id", "input_ids", "labels", "offsets")}
        sample_id = row["sample_sha256"]
        input_ids = row["input_ids"]
        labels = row["labels"]
        offsets = row["offsets"]
        row_aligned = (
            _sha256(sample_id)
            and sample_id == _json_sha256(content)
            and _valid_hir_id(row["row_id"])
            and isinstance(input_ids, list)
            and bool(input_ids)
            and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in input_ids)
            and isinstance(labels, list)
            and all(value in {0, 1} and not isinstance(value, bool) for value in labels)
            and isinstance(offsets, list)
            and all(
                isinstance(offset, list)
                and len(offset) == 2
                and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in offset)
                and offset[0] <= offset[1]
                for offset in offsets
            )
            and len(input_ids) == len(labels) == len(offsets)
        )
        aligned = aligned and row_aligned
        if row_aligned:
            sample_ids.append(sample_id)
            token_count += len(input_ids)
    split_ids: dict[str, list[str]] = {}
    splits_valid = True
    for name in ("train", "validation", "test"):
        split = _object(splits[name], f"token label {name} split")
        _exact_keys(split, {"sample_ids", "sample_ids_sha256", "records"}, f"token label {name} split")
        values = split["sample_ids"]
        valid = (
            isinstance(values, list)
            and bool(values)
            and all(_sha256(value) for value in values)
            and len(values) == len(set(values)) == split["records"]
            and split["sample_ids_sha256"] == _json_sha256({"sample_ids": values})
        )
        splits_valid = splits_valid and valid
        split_ids[name] = values if isinstance(values, list) else []
    split_sets = [set(split_ids[name]) for name in ("train", "validation", "test")]
    split_partition = not (
        split_sets[0] & split_sets[1] or split_sets[0] & split_sets[2] or split_sets[1] & split_sets[2]
    ) and set().union(*split_sets) == set(sample_ids)
    _expect(
        artifact["schema_version"] == 1
        and artifact["id"] == refs["token_labels"]["artifact_id"]
        and artifact["status"] == "frozen"
        and provenance
        == {
            "kind": "reconstructed_rtt_token_relevance",
            "source_repository": "https://github.com/TURLEing/Rubrics-To-Tokens",
            "source_revision": RTT_REVISION,
            "hir_source_sha256": HIR_SOURCE_SHA256,
            "scalar_data_manifest_sha256": refs["scalar_data"]["sha256"],
            "generator_sha256": provenance.get("generator_sha256"),
        }
        and _sha256(provenance["generator_sha256"])
        and tokenizer
        == {"model": MODEL_NAME, "revision": MODEL_REVISION, "files_sha256": tokenizer.get("files_sha256")}
        and _sha256(tokenizer["files_sha256"])
        and data["format"] == "jsonl"
        and data["sha256"] == _file_sha256(path)
        and data["records"] == len(rows) == len(sample_ids) == len(set(sample_ids))
        and data["tokens"] == token_count > 0
        and data["sample_ids_sha256"] == _json_sha256({"sample_ids": sample_ids})
        and aligned
        and splits_valid
        and split_partition
        and alignment == {"records_checked": len(rows), "tokens_checked": token_count, "failures": 0},
        "token labels provenance, bytes, alignment, or split partition is invalid",
    )


def _validate_discriminator_checkpoint(
    artifact: JsonObject,
    refs: JsonObject,
    frozen: Mapping[str, bool],
    repo_root: Path,
) -> None:
    _expect(frozen["token_labels"], "discriminator checkpoint is blocked until token labels are frozen")
    _exact_keys(
        artifact,
        {"schema_version", "id", "status", "token_labels_sha256", "checkpoint"},
        "discriminator checkpoint artifact",
    )
    checkpoint = _object(artifact["checkpoint"], "discriminator checkpoint")
    _exact_keys(checkpoint, {"path", "sha256"}, "discriminator checkpoint")
    checkpoint_path = _external_artifact_path(checkpoint["path"], "discriminator checkpoint path", repo_root)
    _expect(
        artifact["schema_version"] == 1
        and artifact["id"] == refs["discriminator_checkpoint"]["artifact_id"]
        and artifact["status"] == "trained"
        and artifact["token_labels_sha256"] == refs["token_labels"]["sha256"]
        and _sha256(checkpoint["sha256"])
        and checkpoint["sha256"] == _artifact_bytes_sha256(checkpoint_path),
        "discriminator checkpoint is invalid or not linked to token labels",
    )


def _validate_gpqa_access(artifact: JsonObject, reference: JsonObject) -> None:
    _exact_keys(artifact, {"schema_version", "id", "status", "dataset", "access"}, "GPQA access artifact")
    dataset = _object(artifact["dataset"], "GPQA access dataset")
    access = _object(artifact["access"], "GPQA access evidence")
    _exact_keys(
        dataset, {"dataset", "config", "split", "revision", "records", "filename", "sha256"}, "GPQA access dataset"
    )
    _exact_keys(access, {"provider", "gated", "verified"}, "GPQA access evidence")
    filename = dataset["filename"]
    _expect(
        artifact["schema_version"] == 1
        and artifact["id"] == reference["artifact_id"]
        and artifact["status"] == "frozen"
        and dataset["dataset"] == "Idavidrein/gpqa"
        and dataset["config"] == "gpqa_main"
        and dataset["split"] == "train"
        and _git_hash(dataset["revision"])
        and dataset["records"] == 448
        and isinstance(filename, str)
        and bool(filename)
        and not Path(filename).is_absolute()
        and ".." not in Path(filename).parts
        and _sha256(dataset["sha256"])
        and access == {"provider": "huggingface", "gated": True, "verified": True},
        "GPQA access artifact is invalid",
    )


def _validate_discriminator_certificate(artifact: JsonObject, refs: JsonObject, frozen: Mapping[str, bool]) -> None:
    _exact_keys(
        artifact,
        {"schema_version", "id", "status", "checkpoint_manifest_sha256", "outcomes"},
        "discriminator certificate",
    )
    outcomes = _object(artifact["outcomes"], "discriminator certificate outcomes")
    _exact_keys(outcomes, {"checked", "failures"}, "discriminator certificate outcomes")
    _expect(
        frozen["discriminator_checkpoint"]
        and artifact["schema_version"] == 1
        and artifact["id"] == refs["discriminator_certificate"]["artifact_id"]
        and artifact["status"] == "certified"
        and artifact["checkpoint_manifest_sha256"] == refs["discriminator_checkpoint"]["sha256"]
        and isinstance(outcomes["checked"], int)
        and outcomes["checked"] > 0
        and outcomes["failures"] == 0,
        "discriminator certificate is invalid or not linked to its checkpoint",
    )


def _validate_cross(bundle: ProgramBundle) -> None:
    recipe = bundle.program["rl_recipe"]
    load = bundle.judge["load_plan"]
    rollouts = recipe["max_steps"] * recipe["rollout_batch_size"] * recipe["group_size"]
    expected_calls = _ceil_div(rollouts * load["hir_rows_with_soft_rubrics"], load["hir_rows"])
    _expect(rollouts == load["rollouts_per_rl_run"], "judge load does not match the RL rollout recipe")
    _expect(
        expected_calls == load["expected_soft_judge_calls_per_rl_run_ceiling"],
        "judge call estimate does not match the HIR soft-row fraction",
    )
    _expect(
        bundle.model["architecture"]["parameters"] == bundle.compute["analytical_memory"]["model_parameters"],
        "compute memory model size does not match the pinned model",
    )
    _expect(
        bundle.program["rl_recipe"]["save_steps"] == bundle.compute["rtt_released_topology"]["save_steps"],
        "adapted RL checkpoint cadence does not match released RTT",
    )
    resolution = bundle.route_resolution
    route_audit = bundle.taxonomy["static_rtt_route_audit"]
    _expect(
        resolution["counts"]["unresolved"] == route_audit["unsupported"]
        and resolution["counts"]["route_resolvable"] == route_audit["supported"]
        and resolution["route_digest"] == route_audit["route_digest"]
        and resolution["taxonomy_sha256"] == bundle.taxonomy["taxonomy_digest"]["sha256"]
        and resolution["rtt_revision"] == route_audit["revision"],
        "route resolution artifact does not match the static taxonomy inventory",
    )
    _expect(
        bundle.judge["load_plan"]["hir_rows"]
        == bundle.data["source"]["records"]
        == bundle.taxonomy["expected"]["rows"],
        "HIR record counts disagree across configs",
    )
    program_benchmarks = [(item["id"], item["enabled"]) for item in bundle.program["benchmarks"]]
    eval_benchmarks = [(item["id"], item["enabled"]) for item in bundle.evaluation["benchmarks"]]
    _expect(program_benchmarks == eval_benchmarks, "program benchmark states do not match the evaluation registry")
    enabled_benchmarks = sum(enabled for _, enabled in program_benchmarks)
    confirmation_seeds = len(bundle.program["seeds"]["confirmation"])
    candidate_counts = {name: len(values) for name, values in bundle.program["selection"]["candidates"].items()}
    matrix = bundle.compute["run_matrix"]
    derived = {
        "base_evaluation_suites": 1,
        "scalar_rl_runs": (
            4 * confirmation_seeds + candidate_counts["rl_mix"] + candidate_counts["rtt_papo_response"]
        ),
        "qlora_baseline_runs": 2 * confirmation_seeds,
        "token_rl_runs": 3 * confirmation_seeds + candidate_counts["rdan_full"],
        "tuning_runs": sum(candidate_counts.values()),
        "fresh_confirmation_runs": 9 * confirmation_seeds,
        "training_runs": bundle.program["counts"]["trainable_runs"],
        "evaluation_suites": bundle.program["counts"]["evaluation_suites"],
        "planned_enabled_benchmarks_per_suite": enabled_benchmarks,
        "planned_benchmark_executions": bundle.program["counts"]["evaluation_suites"] * enabled_benchmarks,
        "runnable_benchmarks_per_suite": enabled_benchmarks - 1,
        "runnable_benchmark_executions": bundle.program["counts"]["evaluation_suites"] * (enabled_benchmarks - 1),
        "gpqa_transition": "blocked_until_revision_hash_and_access_manifest_frozen",
    }
    _expect(matrix == derived, "compute run matrix does not match the program and enabled benchmarks")
    baselines_ready = all(baseline["data"]["status"] == "frozen" for baseline in bundle.baselines.values())
    expected_readiness = "ready" if baselines_ready else "blocked_until_frozen_data_manifests"
    _expect(
        bundle.program["readiness"]["baselines"] == expected_readiness,
        "program baseline readiness does not match data manifest states",
    )


def _load_json(path: Path) -> JsonObject:
    try:
        with path.open(encoding="utf-8") as file:
            value = json.load(file, object_pairs_hook=_unique_object)
    except (OSError, json.JSONDecodeError) as error:
        raise ProgramContractError(f"cannot load {path}: {error}") from error
    return _object(value, str(path))


def _load_dataset_rows(path: Path, fields: list[str], name: str) -> list[JsonObject]:
    rows: list[JsonObject] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ProgramContractError(f"{name} line {line_number} is blank")
                row = _object(json.loads(line, object_pairs_hook=_unique_object), f"{name} line {line_number}")
                _exact_keys(row, set(fields), f"{name} line {line_number}")
                rows.append(row)
    except (OSError, json.JSONDecodeError) as error:
        raise ProgramContractError(f"cannot load {name}: {error}") from error
    _expect(bool(rows), f"{name} must be non-empty")
    return rows


def _load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise ProgramContractError(f"cannot load {path}: {error}") from error


def _resolve_under(root: Path, path: Path, name: str) -> Path:
    root = root.resolve()
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ProgramContractError(f"{name} escapes {root}") from error
    return path


def _config_path(root: Path, program_path: Path, program: JsonObject, key: str) -> Path:
    return _resolve_under(root, program_path.parent / _string(program[key], f"program.{key}"), f"program.{key}")


def _unique_object(pairs: list[tuple[str, Any]]) -> JsonObject:
    value: JsonObject = {}
    for key, item in pairs:
        if key in value:
            raise ProgramContractError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_secrets(value: Any, path: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
            if (_CREDENTIAL_FIELD.search(lowered) or _PROVIDER_TOKEN_FIELD.search(lowered)) and item:
                raise ProgramContractError(f"literal credential field is forbidden at {path}.{key}")
            _reject_secrets(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_secrets(item, f"{path}[{index}]")
        return
    if isinstance(value, str) and any(pattern.search(value) for pattern in _SECRET_PATTERNS):
        raise ProgramContractError(f"secret-looking literal is forbidden at {path}")


def _has_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_has_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_has_key(item, key) for item in value)
    return False


def _object(value: Any, path: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ProgramContractError(f"{path} must be an object")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProgramContractError(f"{path} must be a non-empty string")
    return value


def _frozen_pin(manifest_id: Any, sha256: Any) -> bool:
    return isinstance(manifest_id, str) and manifest_id not in {"", "pending"} and _sha256(sha256)


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _git_hash(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def _frozen_teacher_pin(model_id: Any, revision: Any) -> bool:
    return _git_hash(revision) or model_id == revision == LUNA_REVISION


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ProgramContractError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _external_artifact_path(value: Any, name: str, repo_root: Path) -> Path:
    root_value = os.environ.get(ARTIFACT_ROOT_ENV)
    _expect(bool(root_value), f"{ARTIFACT_ROOT_ENV} must name the approved external artifact root")
    root = Path(str(root_value)).expanduser()
    _expect(root.is_absolute() and root.is_dir() and not root.is_symlink(), "external artifact root is invalid")
    relative = Path(_string(value, name))
    _expect(not relative.is_absolute(), f"{name} must be relative to {ARTIFACT_ROOT_ENV}")
    path = _resolve_under(root, root / relative, name)
    _expect(not path.is_relative_to(repo_root.resolve()), f"{name} must remain outside the Git checkout")
    _expect(path.exists() and not path.is_symlink(), f"{name} does not resolve to immutable artifact bytes")
    return path


def _artifact_bytes_sha256(path: Path) -> str:
    if path.is_file():
        return _file_sha256(path)
    _expect(path.is_dir(), f"checkpoint artifact is not a file or directory: {path}")
    files = sorted(
        (item for item in path.rglob("*") if item.is_file()), key=lambda item: item.relative_to(path).as_posix()
    )
    _expect(bool(files), "checkpoint artifact directory must contain files")
    _expect(not any(item.is_symlink() for item in path.rglob("*")), "checkpoint artifact must not contain symlinks")
    digest = hashlib.sha256()
    for file in files:
        relative = file.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(file.stat().st_size.to_bytes(8, "big"))
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _validate_clean_rtt_checkout(root: Path, revision: str) -> None:
    root = root.resolve()
    try:
        top = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ProgramContractError(f"cannot inspect RTT checkout: {error}") from error
    _expect(Path(top).resolve() == root, "RTT path is not the checkout root")
    _expect(head == revision, f"RTT checkout revision is not pinned: {head}")
    _expect(not status, "RTT checkout must be clean before training launch")


def _json_sha256(value: JsonObject) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _exact_keys(value: Mapping[str, Any], keys: set[str], path: str) -> None:
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        raise ProgramContractError(f"{path} keys are invalid: missing={missing}, extra={extra}")


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ProgramContractError(message)


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator
