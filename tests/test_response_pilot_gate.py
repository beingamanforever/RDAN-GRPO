from __future__ import annotations

import hashlib
import importlib.util
import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from rdan_grpo.fsdp_hf_receipt import (
    MODEL,
    MODEL_REVISION,
    RTT_REVISION,
    FSDPHFStreamReceipt,
    FSDPHFTransaction,
)
from rdan_grpo.response_pilot_lifecycle import (
    LifecycleCertificateError,
    _complete_response_run,
    _receipt_matches_identity,
    _validate_step_records,
    issue_lifecycle_certificate,
    validate_lifecycle_certificate,
)
from rdan_grpo.roll_response_checkpoint import (
    ArtifactIdentity,
    CheckpointIdentity,
    CheckpointState,
    create_checkpoint_stage,
    promote_checkpoint,
)
from rdan_grpo.roll_response_config import UPDATES_PER_STEP
from rdan_grpo.roll_response_receipt import build_response_receipt
from rdan_grpo.runtime_parity import GENERATION_SOURCE_IDENTITY

ROOT = Path(__file__).resolve().parents[1]
MODEL_IDENTITY = {
    "model": MODEL,
    "revision": MODEL_REVISION,
    "snapshot_sha256": "1" * 64,
    "tokenizer_files_sha256": "2" * 64,
    "chat_template_sha256": "3" * 64,
}
RUNTIME_IDENTITY = {
    "resolved_config_sha256": "a" * 64,
    "production_train_config_sha256": "4" * 64,
    "response_data_manifest_sha256": "c" * 64,
    "response_data_output_sha256": "6" * 64,
    "rtt_revision": RTT_REVISION,
    **GENERATION_SOURCE_IDENTITY,
}


def _module():
    path = ROOT / "scripts/run_response_train.py"
    spec = importlib.util.spec_from_file_location("test_response_pilot_gate_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _identity(stage: str) -> CheckpointIdentity:
    return CheckpointIdentity(
        planned_horizon=500,
        method="rtt_papo_response",
        method_weight=0.5,
        resolved_config_sha256="a" * 64,
        certificate=ArtifactIdentity("preflight", "b" * 64),
        data=ArtifactIdentity("response-data", "c" * 64),
        revisions={"code": "d" * 40, "rtt": "e" * 40, "model": "f" * 40},
        base_checkpoint_sha256="1" * 64,
        wandb={
            "entity": "RDAN-GRPO",
            "project": "rdan-grpo-qwen3-4b",
            "run_id": f"run-{stage}",
            "name": f"qwen-rtt-papo-response-{stage}-s240520",
            "group": "qwen-rtt-papo-response",
        },
    )


def _identities() -> dict[str, CheckpointIdentity]:
    return {stage: _identity(stage) for stage in ("recovery", "pilot", "train")}


def _worker_receipt(side: str, rank: int, transaction: str, weight: int) -> dict[str, Any]:
    receipt = FSDPHFStreamReceipt(
        FSDPHFTransaction(transaction, rank, rank),
        side,
        accelerator_name="NVIDIA A100-SXM4-80GB",
    )
    values = [("model.weight", torch.tensor([weight], dtype=torch.bfloat16))]
    if side == "actor":
        receipt.open_actor_stream(torch.bfloat16)
        list(receipt.wrap_actor_batches([values]))
    else:
        receipt.finish_infer(values)
    return receipt.snapshot()


def _receipt(phase: str, pipeline_step: int, updates: int, weight: int) -> dict[str, Any]:
    transaction = f"transaction-{phase}-{pipeline_step}-{weight}"
    counters = [
        {
            "rank": rank,
            "optimizer_steps": updates,
            "scheduler_steps": updates,
            "finite_steps": updates,
            "skipped_optimizer_steps": 0,
        }
        for rank in range(2)
    ]
    return build_response_receipt(
        [_worker_receipt("actor", rank, transaction, weight) for rank in range(2)],
        [_worker_receipt("infer", rank, transaction, weight) for rank in range(2)],
        phase=phase,
        pipeline_step=pipeline_step,
        actor_counters=counters,
        resolved_config_sha256="a" * 64,
        runtime_identity=RUNTIME_IDENTITY,
        model_identity=MODEL_IDENTITY,
        method="rtt_papo_response",
        fixed_weight=0.5,
    )


def _body(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_body(value))


def _response(prefix: str, step: int, index: int) -> dict[str, Any]:
    rubric = {"id": 1, "category": "", "description": "Be correct.", "weight": 1}
    return {
        "response_index": index,
        "prompt_key": f"prompt-{step}",
        "generation_id": f"{prefix}-generation-{step}-{index}",
        "prompt": "Follow every rubric.",
        "source": "test",
        "ground_truth": {},
        "rubrics": [rubric],
        "response_tokens": [step + index],
        "response_text": f"answer {step}-{index}",
        "response_length": 1,
        "reward": {
            "raw_aon": 1.0,
            "raw_csr": 1.0,
            "raw_signed_csr": 1.0,
            "selected_reward": float(index),
            "response_advantage": 1.0,
            "raw_quality": float(index),
            "quality_eligible": True,
            "quality_advantage": 1.0,
            "scalar_advantage": 1.0,
            "response_valid": True,
        },
        "rubric_outcomes": {
            "scores": [1.0],
            "rubric_mask": [True],
            "eval_mask": [True],
            "hard_mask": [False],
            "evidence": [_evidence(rubric, 1.0)],
        },
        "failures": {"judge_failed": False, "unsupported_hard": False},
    }


def _evidence(rubric: dict[str, Any], score: float) -> dict[str, Any]:
    return {
        "rubric_index": rubric["id"] - 1,
        "rubric_id": rubric["id"],
        "rubric_description_sha256": hashlib.sha256(rubric["description"].encode()).hexdigest(),
        "score": score,
        "evaluator_route": "code_type1",
        "reason": "ok",
        "generation_id": None,
        "request_sha256": None,
        "judge_provenance": None,
        "judge_failed": False,
        "evaluator_failed": False,
        "reward_lane": "authoritative_strict",
        "judge_role": None,
        "fallback_reason": None,
    }


def _group(step: int) -> dict[str, Any]:
    return {
        "group_index": 0,
        "prompt_key": f"prompt-{step}",
        "selected_rewards": [0.0, 1.0],
        "selected_reward_variance": 0.25,
        "quality_eligible_count": 2,
        "conditional_quality_variance": 0.25,
    }


def _artifact(
    root: Path,
    checkpoint: Path,
    prefix: str,
    step: int,
    initial: dict[str, Any],
    post: dict[str, Any],
    *,
    final: bool,
    bad_receipt: bool = False,
    generation_ids: list[str] | None = None,
    response_rows: list[dict[str, Any]] | None = None,
) -> Path:
    path = root / f"step-{step:06d}"
    path.mkdir(parents=True)
    response_rows = (
        deepcopy(response_rows)
        if response_rows is not None
        else [_response(prefix, step, index) for index in range(2)]
    )
    if generation_ids is not None:
        for row, generation_id in zip(response_rows, generation_ids, strict=True):
            row["generation_id"] = generation_id
    responses = b"".join(_body(row) for row in response_rows)
    group_row = _group(step)
    group_row["prompt_key"] = response_rows[0]["prompt_key"]
    group = _body(group_row)
    (path / "responses.jsonl").write_bytes(responses)
    (path / "groups.jsonl").write_bytes(group)
    _write(path / "metrics.json", {"actor/clipfrac": 0.1})
    _write(
        path / "diagnostics.json",
        {
            "group_count": 1,
            "response_active_group_count": 1,
            "response_active_group_rate": 1.0,
            "quality_active_group_count": 1,
            "quality_active_group_rate": 1.0,
            "selected_reward_variance_mean": 0.25,
        },
    )
    if bad_receipt:
        initial = {"status": "receipt_passed"}
    _write(path / "receipts/initial.json", initial)
    _write(path / "receipts/post-update.json", post)
    files = sorted(child for child in path.rglob("*") if child.is_file())
    inventory = [
        {
            "path": child.relative_to(path).as_posix(),
            "size": child.stat().st_size,
            "sha256": __import__("hashlib").sha256(child.read_bytes()).hexdigest(),
        }
        for child in files
    ]
    _write(
        path / "manifest.json",
        {
            "schema_version": 1,
            "status": "sealed",
            "step": step,
            "checkpoint": {
                "path": str(checkpoint),
                "status": "pending_local_promotion" if final else "not_scheduled",
            },
            "inventory": inventory,
        },
    )
    return path


def _checkpoint(root: Path, identity: CheckpointIdentity, step: int, artifact: Path) -> Path:
    stage = create_checkpoint_stage(root, step)
    files = {
        "actor/rank-0.distcp": b"model-0",
        "actor/rank-1.distcp": b"model-1",
        "rng/rng_state_driver.pth": b"rng",
        "rng/infer-rank-0.pt": b"infer-0",
        "rng/infer-rank-1.pt": b"infer-1",
        "scheduler/state.json": _body({"dataset_iter_count": step}),
        "receipts/initial.json": (artifact / "receipts/initial.json").read_bytes(),
        "receipts/post-update.json": (artifact / "receipts/post-update.json").read_bytes(),
        "metrics/step.json": _body({"actor/clipfrac": 0.1}),
        "artifacts/step.json": _body(
            {
                "path": str(artifact),
                "manifest_sha256": __import__("hashlib").sha256((artifact / "manifest.json").read_bytes()).hexdigest(),
            }
        ),
    }
    for relative, body in files.items():
        path = stage / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    updates = step * UPDATES_PER_STEP
    state = CheckpointState(
        completed_step=step,
        optimizer_counters={0: updates, 1: updates},
        scheduler_counters={0: updates, 1: updates},
        scheduler_state={"dataset_iter_count": step},
        rng_artifacts={
            "driver": "rng/rng_state_driver.pth",
            "infer_0": "rng/infer-rank-0.pt",
            "infer_1": "rng/infer-rank-1.pt",
        },
        metrics={"actor/clipfrac": 0.1},
        peak_memory={0: 70_000_000_000, 1: 71_000_000_000},
        reward_variance=0.25,
        group_diagnostics={
            "group_count": 10,
            "response_active_group_count": 8,
            "response_active_group_rate": 0.8,
            "quality_active_group_count": 2,
            "quality_active_group_rate": 0.2,
            "selected_reward_variance_mean": 0.25,
        },
        clipping_fraction=0.1,
        receipt_links={"initial": "receipts/initial.json", "post_update": "receipts/post-update.json"},
    )
    return promote_checkpoint(stage, identity=identity, state=state, artifacts=list(files))


def _run(
    root: Path,
    identity: CheckpointIdentity,
    prefix: str,
    steps: range,
    *,
    resumed_from: dict[str, Any] | None = None,
    bad_receipt: bool = False,
    generation_ids: list[str] | None = None,
    response_rows: list[dict[str, Any]] | None = None,
) -> tuple[Path, Path]:
    artifact_root = root / "artifacts"
    checkpoint_root = root / "checkpoints"
    artifact_root.mkdir(parents=True)
    first = steps.start
    if resumed_from is None:
        initial = _receipt("initial", 0, 0, 0)
    else:
        initial = _receipt("resume_initial", first - 1, (first - 1) * UPDATES_PER_STEP, first - 1)
        assert initial["actor_receipts"][0]["manifest_sha256"] == resumed_from["actor_receipts"][0]["manifest_sha256"]
    final_artifact = None
    for step in steps:
        post = _receipt("post_update", step, step * UPDATES_PER_STEP, step)
        final_artifact = _artifact(
            artifact_root,
            checkpoint_root / f"step-{steps.stop - 1:06d}",
            prefix,
            step,
            initial,
            post,
            final=step == steps.stop - 1,
            bad_receipt=bad_receipt and step == first,
            generation_ids=generation_ids if step == first else None,
            response_rows=response_rows if step == first else None,
        )
        initial = post
    assert final_artifact is not None
    return _checkpoint(checkpoint_root, identity, steps.stop - 1, final_artifact), artifact_root


def _sequence(tmp_path: Path) -> tuple[dict[str, CheckpointIdentity], dict[str, Path]]:
    identities = _identities()
    recovery_root = tmp_path / "recovery"
    recovery_1, recovery_artifacts = _run(recovery_root, identities["recovery"], "recovery", range(1, 2))
    cert_1 = issue_lifecycle_certificate(
        recovery_root / "recovery-step-1.json",
        stage="recovery_step_1",
        outcome=_outcome(identities["recovery"], recovery_1, recovery_artifacts),
    )
    post_1 = json.loads((recovery_artifacts / "step-000001/receipts/post-update.json").read_bytes())
    recovery_2, _ = _run(
        recovery_root / "resumed",
        identities["recovery"],
        "resumed",
        range(2, 3),
        resumed_from=post_1,
    )
    cert_2 = issue_lifecycle_certificate(
        recovery_root / "recovery-step-2.json",
        stage="recovery_step_2",
        outcome=_outcome(
            identities["recovery"],
            recovery_2,
            recovery_root / "resumed/artifacts",
            predecessor=cert_1,
            resume=recovery_1,
        ),
    )
    pilot_root = tmp_path / "pilot"
    pilot_20, pilot_artifacts = _run(pilot_root, identities["pilot"], "pilot", range(1, 21))
    pilot_cert = issue_lifecycle_certificate(
        pilot_root / "pilot-step-20.json",
        stage="pilot_step_20",
        outcome=_outcome(identities["pilot"], pilot_20, pilot_artifacts, predecessor=cert_2),
    )
    return identities, {
        "recovery_1": recovery_1,
        "recovery_2": recovery_2,
        "cert_1": cert_1,
        "cert_2": cert_2,
        "pilot_20": pilot_20,
        "pilot_cert": pilot_cert,
    }


def _args(stage: str, stop: int | None) -> SimpleNamespace:
    return SimpleNamespace(stage=stage, stop_after_step=stop)


def _outcome(
    identity: CheckpointIdentity,
    checkpoint: Path,
    artifacts: Path,
    *,
    predecessor: Path | None = None,
    resume: Path | None = None,
):
    return _complete_response_run(
        identity=identity,
        runtime_identity=RUNTIME_IDENTITY,
        model_identity=MODEL_IDENTITY,
        checkpoints=[checkpoint],
        artifact_root=artifacts,
        predecessor=predecessor,
        resume_checkpoint=resume,
    )


def _gate(
    module: Any,
    identities: dict[str, CheckpointIdentity],
    stage: str,
    stop: int | None,
    *,
    resume: Path | None = None,
    recovery: Path | None = None,
    pilot: Path | None = None,
) -> None:
    module._require_pilot_sequence(
        _args(stage, stop),
        identities=identities,
        runtime_identity=RUNTIME_IDENTITY,
        model_identity=MODEL_IDENTITY,
        resume=resume,
        recovery_evidence=recovery,
        pilot_evidence=pilot,
    )


def test_exact_runner_certified_recovery_pilot_train_sequence(tmp_path: Path) -> None:
    module = _module()
    identities, evidence = _sequence(tmp_path)

    _gate(module, identities, "recovery", 1)
    _gate(module, identities, "recovery", 2, resume=evidence["recovery_1"], recovery=evidence["cert_1"])
    _gate(module, identities, "pilot", 20, recovery=evidence["cert_2"])
    _gate(module, identities, "train", None, pilot=evidence["pilot_cert"])

    pilot = validate_lifecycle_certificate(
        evidence["pilot_cert"],
        expected_stage="pilot_step_20",
        expected_identity=identities["pilot"],
        expected_runtime_identity=RUNTIME_IDENTITY,
        expected_model_identity=MODEL_IDENTITY,
    )
    assert [item["step"] for item in pilot["step_artifacts"]] == list(range(1, 21))
    assert pilot["runtime_identity"] == RUNTIME_IDENTITY
    assert pilot["model_identity"] == MODEL_IDENTITY
    assert (
        pilot["predecessor"]["checkpoint_manifest_sha256"]
        == json.loads(evidence["cert_2"].read_text(encoding="utf-8"))["checkpoint"]["manifest_sha256"]
    )


def test_inference_boundary_ids_issue_step_one_certificate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workers_path = ROOT / "tests/test_roll_response_workers_e2e.py"
    spec = importlib.util.spec_from_file_location("test_generation_id_boundary", workers_path)
    assert spec is not None and spec.loader is not None
    workers_test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workers_test)
    workers = workers_test._load_workers(monkeypatch)
    worker = workers.ResponseInferWorker()
    worker.worker_config = SimpleNamespace(infer_batch_size=1)
    worker.rank_info = SimpleNamespace(dp_rank=0)
    worker.calls = []
    output = worker.generate(workers_test.FakeData(torch.tensor([10]), {"global_step": 1}))

    identity = _identity("recovery")
    checkpoint, artifacts = _run(
        tmp_path / "generated",
        identity,
        "generated",
        range(1, 2),
        generation_ids=output.non_tensor_batch["generation_id"].tolist(),
    )
    certificate = issue_lifecycle_certificate(
        tmp_path / "generated/step-1.json",
        stage="recovery_step_1",
        outcome=_outcome(identity, checkpoint, artifacts),
    )

    assert certificate.is_file()


def test_real_frozen_hir_row_records_issue_lifecycle_certificate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pipeline_path = ROOT / "tests/test_roll_response_pipeline_e2e.py"
    spec = importlib.util.spec_from_file_location("test_real_hir_artifact_boundary", pipeline_path)
    assert spec is not None and spec.loader is not None
    pipeline_test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline_test)
    pipeline = pipeline_test._load_pipeline(monkeypatch)
    row = json.loads((ROOT / "data/HIR_trainv1_rubrics_processed.jsonl").open(encoding="utf-8").readline())
    rubric_count = len(row["rubrics"])
    scores = torch.ones((2, rubric_count))

    class Batch:
        batch = {
            "input_ids": torch.tensor([[10, 11], [10, 12]]),
            "response_mask": torch.tensor([[False, True], [False, True]]),
            **{
                name: torch.tensor([0.0, 1.0])
                for name in (
                    "rdan_raw_aon",
                    "rdan_raw_csr",
                    "rdan_raw_signed_csr",
                    "rdan_selected_reward",
                    "rdan_response_advantage",
                    "rdan_raw_quality",
                    "rdan_quality_advantage",
                    "rdan_scalar_advantage",
                )
            },
            "rdan_quality_eligible": torch.ones(2, dtype=torch.bool),
            "rdan_response_valid": torch.ones(2, dtype=torch.bool),
            "rdan_scores": scores,
            "rdan_rubric_mask": torch.ones((2, rubric_count), dtype=torch.bool),
            "rdan_eval_mask": torch.ones((2, rubric_count), dtype=torch.bool),
            "rdan_hard_mask": torch.ones((2, rubric_count), dtype=torch.bool),
            "rdan_judge_failed": torch.zeros(2, dtype=torch.bool),
            "rdan_unsupported_hard": torch.zeros(2, dtype=torch.bool),
        }
        evidence = [[_evidence(rubric, 1.0) for rubric in row["rubrics"]] for _ in range(2)]
        non_tensor_batch = {
            "prompt": [row["prompt"]] * 2,
            "rubrics": [row["rubrics"]] * 2,
            "source": [row["source"]] * 2,
            "ground_truth": [json.dumps(row["ground_truth"])] * 2,
            "rdan_prompt_key": [str(row["id"])] * 2,
            "rdan_rubric_evidence": evidence,
            "generation_id": ["gen-000001-r0-000000000000", "gen-000001-r0-000000000001"],
        }

        def __len__(self) -> int:
            return 2

    records = pipeline._response_records(Batch(), SimpleNamespace(decode=lambda tokens, **_: "valid response"))
    assert records[0]["ground_truth"] == row["ground_truth"]
    assert records[0]["rubrics"] == row["rubrics"]
    identity = _identity("recovery")
    checkpoint, artifacts = _run(tmp_path / "real-hir", identity, "real-hir", range(1, 2), response_rows=records)
    certificate = issue_lifecycle_certificate(
        tmp_path / "real-hir/step-1.json",
        stage="recovery_step_1",
        outcome=_outcome(identity, checkpoint, artifacts),
    )
    validate_lifecycle_certificate(
        certificate,
        expected_stage="recovery_step_1",
        expected_identity=identity,
        expected_runtime_identity=RUNTIME_IDENTITY,
        expected_model_identity=MODEL_IDENTITY,
    )


def test_raw_or_synthetic_checkpoint_never_authorizes_next_stage(tmp_path: Path) -> None:
    module = _module()
    identities, evidence = _sequence(tmp_path)

    with pytest.raises(LifecycleCertificateError, match="certificate"):
        _gate(module, identities, "pilot", 20, recovery=evidence["recovery_2"])
    with pytest.raises(LifecycleCertificateError, match="certificate"):
        _gate(module, identities, "train", None, pilot=evidence["pilot_20"])

    plausible = tmp_path / "caller-written.json"
    _write(
        plausible,
        {
            "schema_version": 1,
            "status": "lifecycle_passed",
            "stage": "pilot_step_20",
            "checkpoint": {"path": str(evidence["pilot_20"])},
        },
    )
    with pytest.raises(LifecycleCertificateError, match="schema"):
        _gate(module, identities, "train", None, pilot=plausible)

    (evidence["pilot_20"] / "actor/rank-0.distcp").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="corrupt"):
        _gate(module, identities, "train", None, pilot=evidence["pilot_cert"])

    bad_root = tmp_path / "synthetic"
    checkpoint, artifacts = _run(
        bad_root,
        identities["recovery"],
        "synthetic",
        range(1, 2),
        bad_receipt=True,
    )
    with pytest.raises(LifecycleCertificateError, match="receipt schema"):
        issue_lifecycle_certificate(
            bad_root / "plausible.json",
            stage="recovery_step_1",
            outcome=_outcome(identities["recovery"], checkpoint, artifacts),
        )


def test_predecessor_substitution_and_tamper_fail_closed(tmp_path: Path) -> None:
    identities, evidence = _sequence(tmp_path / "first")
    other_identities, other = _sequence(tmp_path / "other")

    with pytest.raises(LifecycleCertificateError, match="not the certified predecessor"):
        issue_lifecycle_certificate(
            tmp_path / "substituted.json",
            stage="recovery_step_2",
            outcome=_outcome(
                identities["recovery"],
                evidence["recovery_2"],
                tmp_path / "first/recovery/resumed/artifacts",
                predecessor=other["cert_1"],
                resume=evidence["recovery_1"],
            ),
        )

    predecessor = evidence["cert_2"]
    payload = json.loads(predecessor.read_bytes())
    payload["checkpoint"]["manifest_sha256"] = "0" * 64
    predecessor.write_bytes(_body(payload))
    with pytest.raises(LifecycleCertificateError, match="digest"):
        validate_lifecycle_certificate(
            evidence["pilot_cert"],
            expected_stage="pilot_step_20",
            expected_identity=identities["pilot"],
            expected_runtime_identity=RUNTIME_IDENTITY,
            expected_model_identity=MODEL_IDENTITY,
        )
    assert asdict(other_identities["recovery"]) == asdict(identities["recovery"])


def test_receipt_and_step_artifact_schema_rejection(tmp_path: Path) -> None:
    identities, evidence = _sequence(tmp_path)
    response = tmp_path / "pilot/artifacts/step-000010/responses.jsonl"
    response.write_text("{}\n", encoding="utf-8")
    with pytest.raises(LifecycleCertificateError, match="inventory digest|artifact row schema"):
        validate_lifecycle_certificate(
            evidence["pilot_cert"],
            expected_stage="pilot_step_20",
            expected_identity=identities["pilot"],
            expected_runtime_identity=RUNTIME_IDENTITY,
            expected_model_identity=MODEL_IDENTITY,
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("response_index",), True),
        (("response_tokens",), []),
        (("response_tokens",), [True]),
        (("response_tokens",), "redacted"),
        (("response_length",), True),
        (("response_length",), 2),
        (("prompt",), 1),
        (("source",), None),
        (("response_text",), []),
        (("ground_truth",), []),
        (("ground_truth",), {"bad": float("nan")}),
        (("rubrics",), [{"id": 1, "description": "missing schema"}]),
        (("rubrics", 0, "weight"), True),
        (("reward", "selected_reward"), float("inf")),
        (("reward", "quality_eligible"), 1),
        (("failures", "judge_failed"), 0),
        (("rubric_outcomes", "scores"), [True]),
        (("rubric_outcomes", "eval_mask"), []),
        (("rubric_outcomes", "evidence"), []),
        (("rubric_outcomes", "evidence"), [{"rubric_id": 1}]),
        (("rubric_outcomes", "evidence", 0, "score"), float("nan")),
        (("rubric_outcomes", "evidence", 0, "judge_failed"), 0),
    ],
)
def test_response_leaf_mutations_fail_closed(path: tuple[Any, ...], value: Any) -> None:
    responses = [_response("mutation", 1, index) for index in range(2)]
    target = responses[0]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(LifecycleCertificateError, match="response"):
        _validate_step_records(responses, [_group(1)])


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("group_index", True),
        ("prompt_key", True),
        ("selected_rewards", [0.0, float("nan")]),
        ("selected_rewards", [0.0]),
        ("selected_reward_variance", float("inf")),
        ("selected_reward_variance", -1.0),
        ("quality_eligible_count", True),
        ("quality_eligible_count", -1),
        ("conditional_quality_variance", -1.0),
    ],
)
def test_group_leaf_mutations_fail_closed(key: str, value: Any) -> None:
    group = _group(1)
    group[key] = value
    with pytest.raises(LifecycleCertificateError, match="group"):
        _validate_step_records([_response("mutation", 1, index) for index in range(2)], [group])


@pytest.mark.parametrize(
    ("identity_name", "key"),
    [("runtime", "production_train_config_sha256"), ("model", "tokenizer_files_sha256")],
)
def test_receipt_requires_exact_complete_runtime_and_model_identity(identity_name: str, key: str) -> None:
    receipt = _receipt("initial", 0, 0, 0)
    receipt[identity_name][key] = "9" * 64
    with pytest.raises(LifecycleCertificateError, match="lifecycle identity"):
        _receipt_matches_identity(receipt, _identity("recovery"), RUNTIME_IDENTITY, MODEL_IDENTITY)


def test_lifecycle_identities_share_inputs_but_not_stage_run_identity() -> None:
    module = _module()
    identities = _identities()
    module._validate_stage_identities(identities)

    identities["pilot"] = replace(identities["pilot"], data=ArtifactIdentity("other-data", "9" * 64))
    with pytest.raises(ValueError, match="immutable identities differ"):
        module._validate_stage_identities(identities)

    identities = _identities()
    identities["pilot"] = replace(identities["pilot"], wandb=identities["recovery"].wandb)
    with pytest.raises(ValueError, match="must be distinct"):
        module._validate_stage_identities(identities)


def test_certificate_is_issued_only_after_pipeline_run_returns() -> None:
    source = (ROOT / "scripts/run_response_train.py").read_text(encoding="utf-8")
    main = source[source.index("def main()") : source.index("def _launch_paths(")]
    assert main.index("completed = _run_pipeline(") < main.index("_print_outcome(args, completed, run_dir)")
    assert "return pipeline.run()" in source
    assert "lifecycle = _issue_lifecycle_outcome(" in source
    assert "_complete_response_run" not in source
    prepare = source[source.index("def _prepare_lifecycle(") : source.index("def _stage_tracking(")]
    assert "_require_pilot_sequence(" in prepare
    assert source.index("def _prepare_lifecycle(") < source.index("def _run_pipeline(")


def test_generic_issuer_rejects_non_runner_outcome(tmp_path: Path) -> None:
    with pytest.raises(LifecycleCertificateError, match="completed runner outcome"):
        issue_lifecycle_certificate(tmp_path / "forged.json", stage="recovery_step_1", outcome=object())  # type: ignore[arg-type]


def test_completed_runner_outcome_is_single_use(tmp_path: Path) -> None:
    identities, evidence = _sequence(tmp_path)
    outcome = _outcome(identities["recovery"], evidence["recovery_1"], tmp_path / "recovery/artifacts")
    issue_lifecycle_certificate(tmp_path / "first.json", stage="recovery_step_1", outcome=outcome)
    with pytest.raises(LifecycleCertificateError, match="already consumed"):
        issue_lifecycle_certificate(tmp_path / "second.json", stage="recovery_step_1", outcome=outcome)
