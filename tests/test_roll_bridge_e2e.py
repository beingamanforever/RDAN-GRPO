from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rdan_grpo.advantages import group_advantages
from rdan_grpo.response_sampling import PREFLIGHT_SOURCES, balanced_preflight_indices
from rdan_grpo.roll_bridge import (
    assess_scalar_batch,
    attach_roll_reward_fields,
    build_preflight_certificate,
    inject_roll_advantages,
    make_roll_compute_advantage,
    require_train_certificate,
    write_certificate,
)
from rdan_grpo.roll_scalar import ScalarMethod


def _batch(
    prompt_count: int = 256,
    *,
    response_variance: bool = True,
    quality_active_prompts: int = 256,
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    group_size = 8
    keys = [f"p{prompt:04d}" for prompt in range(prompt_count) for _ in range(group_size)]
    scores = torch.ones((prompt_count * group_size, 2), dtype=torch.float32)
    for prompt in range(prompt_count):
        start = prompt * group_size
        if response_variance:
            scores[start, 0] = -1
        if prompt < quality_active_prompts:
            scores[start + 1 : start + 4, 1] = -1
    rubric_mask = torch.ones_like(scores, dtype=torch.bool)
    eval_mask = torch.ones_like(scores, dtype=torch.bool)
    hard_mask = torch.zeros_like(scores, dtype=torch.bool)
    hard_mask[:, 0] = True
    return keys, scores, rubric_mask, eval_mask, hard_mask


def test_no_update_preflight_covers_full_merged_inventory() -> None:
    inventory = {
        "type1": 6549,
        "type2": 449,
        "type3": 3603,
        "type4": 6361,
        "rubrichub_instruction_following": 1134,
    }
    sources = [source for source in PREFLIGHT_SOURCES for _ in range(inventory[source])]
    assert len(sources) == 18_096

    first = balanced_preflight_indices(sources, 256)
    second = balanced_preflight_indices(sources, 256)

    assert first == second
    assert len(first) == len(set(first)) == 256
    selected = [sources[index] for index in first]
    assert {source: selected.count(source) for source in PREFLIGHT_SOURCES} == {
        "type1": 52,
        "type2": 51,
        "type3": 51,
        "type4": 51,
        "rubrichub_instruction_following": 51,
    }
    with pytest.raises(ValueError, match="every frozen source"):
        balanced_preflight_indices([PREFLIGHT_SOURCES[0]] * 256, 256)
    with pytest.raises(ValueError, match="unexpected source"):
        balanced_preflight_indices(sources + ["unknown"], 256)
    with pytest.raises(ValueError, match="cover every frozen source"):
        balanced_preflight_indices(sources, 4)


def _assessment(**kwargs: object):
    keys, scores, rubric_mask, eval_mask, hard_mask = _batch(**kwargs)
    return assess_scalar_batch(keys, scores, rubric_mask, eval_mask, hard_mask, quality_weight=0.5)


def _certificate(assessment, optimizer_updates: int = 0):
    digest = "a" * 64
    return build_preflight_certificate(
        [assessment],
        config_sha256=digest,
        source_sha256={"rows": "b" * 64},
        optimizer_updates=optimizer_updates,
        quality_weight=0.5,
    )


@dataclass
class RollLikeBatch:
    batch: dict[str, torch.Tensor]
    non_tensor_batch: dict[str, object] | None = None


def _install_real_roll_compat() -> None:
    from rdan_grpo.roll_compat import install_rtt_compat

    root = os.environ.get("RTT_ROOT")
    if not root:
        pytest.skip("RTT_ROOT is required for the pinned ROLL boundary tests")
    install_rtt_compat(Path(root))


def test_reward_boundary_preserves_hard_soft_provenance() -> None:
    data = RollLikeBatch(
        batch={},
        non_tensor_batch={
            "id": [7, 7],
            "source": ["type1", "type4"],
            "ground_truth": [{}, {"checker": ["[rule] exact", "[llm] quality"]}],
            "prompt": ["redacted", "redacted"],
        },
    )
    output = RollLikeBatch(batch={"rubric_scores_list": torch.tensor([[1.0, -1.0, -100.0], [1.0, 0.5, -100.0]])})
    infos = [
        {"method": "code_type1"},
        {"method": "llm_judge", "llm_response": "{NO}"},
        {"method": "code_type4_rule"},
        {"method": "llm_judge", "llm_response": "{YES}"},
    ]
    returned = attach_roll_reward_fields(data, output, infos)
    assert returned is output
    assert output.batch["rdan_rubric_mask"].tolist() == [[True, True, False], [True, True, False]]
    assert output.batch["rdan_hard_mask"].tolist() == [[True, False, False], [True, False, False]]
    assert output.batch["rdan_eval_mask"].tolist() == [[True, True, False], [True, True, False]]
    assert output.batch["rdan_unsupported_hard"].tolist() == [True, False]
    assert output.batch["rdan_judge_failed"].tolist() == [False, False]
    assert data.non_tensor_batch["rdan_prompt_key"] == [7, 7]


@pytest.mark.skipif(importlib.util.find_spec("roll") is None, reason="pinned RTT ROLL is not installed")
def test_pinned_roll_adapter_installs_at_real_pipeline_boundary() -> None:
    _install_real_roll_compat()
    from roll.pipeline.rlvr import rubircs_pipeline

    certificate = _certificate(_assessment())
    adapter = __import__("rdan_grpo.roll_bridge", fromlist=["install_roll_adapter"]).install_roll_adapter(certificate)
    assert rubircs_pipeline.compute_advantage is adapter
    assert adapter.__name__ == "compute_rdan_scalar_advantage"


@pytest.mark.skipif(importlib.util.find_spec("roll") is None, reason="pinned RTT ROLL is not installed")
def test_live_preflight_rejects_any_training_pipeline_before_construction() -> None:
    _install_real_roll_compat()
    from rdan_grpo.roll_live import run_live_preflight

    class TrainingPipeline:
        constructed = False

        def __init__(self, config) -> None:
            self.constructed = True

    with pytest.raises(TypeError, match="ScalarPreflightPipeline"):
        run_live_preflight(TrainingPipeline, object())
    assert not TrainingPipeline.constructed


@pytest.mark.skipif(importlib.util.find_spec("roll") is None, reason="pinned RTT ROLL is not installed")
def test_live_preflight_rejects_training_config_before_pipeline_construction() -> None:
    _install_real_roll_compat()
    from types import SimpleNamespace

    from rdan_grpo.roll_live import ScalarPreflightPipeline, run_live_preflight

    class ProbePipeline(ScalarPreflightPipeline):
        constructed = False

        def __init__(self, config) -> None:
            self.constructed = True

    config = SimpleNamespace(actor_train=SimpleNamespace(device_mapping=[0]), max_steps=1)
    with pytest.raises(ValueError, match="device_mapping"):
        run_live_preflight(ProbePipeline, config)
    assert not ProbePipeline.constructed


def test_no_update_preflight_certificate_and_roll_injection_end_to_end(tmp_path) -> None:
    assessment = _assessment()
    certificate = _certificate(assessment)
    assert assessment.batch_valid
    assert certificate.ready
    assert certificate.metrics["prompt_count"] == 256
    assert certificate.metrics["group_size"] == 8
    assert certificate.metrics["optimizer_updates"] == 0
    assert certificate.metrics["quality_active_group_rate"] == 1.0
    assert not assessment.output.hard_pass[0]
    assert assessment.output.quality_advantage[0] == 0
    assert assessment.output.scalar_advantage[0] == assessment.output.response_advantage[0]

    path = tmp_path / "certificate.json"
    write_certificate(certificate, path)
    loaded = require_train_certificate(
        path,
        config_sha256="a" * 64,
        source_sha256={"rows": "b" * 64},
        quality_weight=0.5,
    )
    assert loaded["certificate_id"] == certificate.certificate_id
    assert loaded["quality_weight"] == 0.5
    with pytest.raises(ValueError, match="quality weight"):
        require_train_certificate(path, quality_weight=0.4)
    with pytest.raises(ValueError, match="source hashes"):
        require_train_certificate(path, source_sha256={"train_config": "c" * 64})
    with pytest.raises(FileExistsError):
        write_certificate(certificate, path)

    response_mask = torch.tensor([[True, True, False]]).expand(2048, -1).clone()
    data = RollLikeBatch(batch={"response_mask": response_mask})
    returned = inject_roll_advantages(data, assessment.output, response_mask, certificate)
    assert returned is data
    expected = assessment.output.scalar_advantage.unsqueeze(-1) * response_mask
    assert torch.equal(data.batch["advantages"], expected)
    assert torch.equal(data.batch["raw_advantages"], expected)
    assert data.batch["advantages"].data_ptr() != data.batch["raw_advantages"].data_ptr()


def test_launch_rejects_quality_weight_mismatch_before_rtt_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    certificate_path = tmp_path / "certificate.json"
    write_certificate(_certificate(_assessment()), certificate_path)
    spec = importlib.util.spec_from_file_location("run_roll_preflight_quality", Path("scripts/run_roll_preflight.py"))
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setenv("RTT_ROOT", str(tmp_path / "uninspected-rtt"))
    monkeypatch.setattr(
        script,
        "check_program",
        lambda *args, **kwargs: pytest.fail("program was inspected"),
        raising=False,
    )
    args = SimpleNamespace(
        certificate=certificate_path,
        config=tmp_path / "preflight.yaml",
        train_config=tmp_path / "train.yaml",
        program=tmp_path / "program.json",
        group_size=8,
        method="rdan_scalar",
        quality_weight=0.4,
        mix_weight=None,
        output=tmp_path / "rows.jsonl",
        restricted_output=tmp_path / "raw.jsonl",
    )

    with pytest.raises(ValueError, match="quality method"):
        script._live_rollout(args)


def test_real_roll_compute_boundary_replaces_normalization_before_train_step() -> None:
    keys, scores, rubric_mask, eval_mask, hard_mask = _batch()
    assessment = _assessment()
    certificate = _certificate(assessment)
    response_mask = torch.tensor([[True, True, False]]).expand(2048, -1).clone()
    data = RollLikeBatch(
        batch={
            "rdan_scores": scores,
            "rdan_rubric_mask": rubric_mask,
            "rdan_eval_mask": eval_mask,
            "rdan_hard_mask": hard_mask,
            "final_response_mask": response_mask,
        },
        non_tensor_batch={"rdan_prompt_key": keys},
    )
    boundary = make_roll_compute_advantage(certificate)
    returned = boundary(
        data,
        gamma=1.0,
        lambd=1.0,
        adv_estimator="grpo",
        whiten_advantages=True,
        whiten_rewards=True,
        response_mask=response_mask,
    )
    assert returned is data
    expected = assessment.output.scalar_advantage.unsqueeze(-1) * response_mask
    assert torch.equal(data.batch["advantages"], expected)
    assert torch.equal(data.batch["raw_advantages"], expected)
    assert torch.equal(data.batch["returns"], expected)


@pytest.mark.parametrize(
    ("method", "quality_weight", "mix_weight"),
    [
        ("rdan_scalar", 0.5, None),
        ("rtt_papo_response", 0.5, None),
        ("rl_csr", None, None),
        ("rl_aon", None, None),
        ("rl_mix", None, 0.25),
    ],
)
def test_all_response_methods_cross_exact_roll_boundary(
    method: ScalarMethod,
    quality_weight: float | None,
    mix_weight: float | None,
    tmp_path: Path,
) -> None:
    keys = ["prompt"] * 8
    scores = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
        ]
    )
    rubric_mask = torch.ones_like(scores, dtype=torch.bool)
    eval_mask = torch.ones_like(scores, dtype=torch.bool)
    hard_mask = torch.zeros_like(scores, dtype=torch.bool)
    hard_mask[:, 0] = True
    assessment = assess_scalar_batch(
        keys,
        scores,
        rubric_mask,
        eval_mask,
        hard_mask,
        method=method,
        quality_weight=quality_weight,
        mix_weight=mix_weight,
    )
    certificate = build_preflight_certificate(
        [assessment],
        method=method,
        config_sha256="a" * 64,
        source_sha256={"rows": "b" * 64},
        optimizer_updates=0,
        quality_weight=quality_weight,
        mix_weight=mix_weight,
        min_prompts=1,
        max_prompts=1,
        min_quality_active_rate=0,
    )
    payload = certificate.as_dict()
    assert ("quality_weight" in payload) == (method in {"rdan_scalar", "rtt_papo_response"})
    assert ("mix_weight" in payload) == (method == "rl_mix")
    response_mask = torch.tensor([[True, True, False]]).expand(8, -1).clone()
    data = RollLikeBatch(
        batch={
            "rdan_scores": scores,
            "rdan_rubric_mask": rubric_mask,
            "rdan_eval_mask": eval_mask,
            "rdan_hard_mask": hard_mask,
            "final_response_mask": response_mask,
        },
        non_tensor_batch={"rdan_prompt_key": keys},
    )

    returned = make_roll_compute_advantage(
        certificate,
        method=method,
        quality_weight=quality_weight,
        mix_weight=mix_weight,
    )(data)

    unit = (scores + 1) / 2
    all_aon = (scores == 1).all(dim=-1).float()
    all_csr = unit.mean(dim=-1)
    hard_aon = (scores[:, 0] == 1).float()
    if method in {"rdan_scalar", "rtt_papo_response"}:
        raw = hard_aon
    elif method == "rl_aon":
        raw = all_aon
    elif method == "rl_csr":
        raw = all_csr
    else:
        raw = 0.25 * all_aon + 0.75 * all_csr
    response_expected = (raw - raw.mean()) / (raw.std() + 1e-6)
    expected = response_expected
    if method in {"rdan_scalar", "rtt_papo_response"}:
        quality = unit[:, 1:].mean(dim=-1)
        eligible = hard_aon.bool()
        selected = quality[eligible]
        quality_expected = torch.zeros_like(quality)
        quality_expected[eligible] = (selected - selected.mean()) / (selected.std() + 1e-6)
        expected = expected + 0.5 * quality_expected
    expected = expected.unsqueeze(-1) * response_mask
    assert returned is data
    assert torch.allclose(data.batch["advantages"], expected, atol=2e-6)
    assert torch.equal(data.batch["advantages"], data.batch["raw_advantages"])
    assert torch.equal(data.batch["advantages"], data.batch["returns"])
    evidence = {
        "rdan_raw_aon": assessment.output.raw_aon,
        "rdan_raw_csr": assessment.output.raw_csr,
        "rdan_raw_signed_csr": assessment.output.raw_signed_csr,
        "rdan_selected_reward": assessment.output.selected_raw_reward,
        "rdan_response_advantage": assessment.output.response_advantage,
        "rdan_raw_quality": assessment.output.raw_quality,
        "rdan_quality_eligible": assessment.output.quality_eligible,
        "rdan_quality_advantage": assessment.output.quality_advantage,
        "rdan_scalar_advantage": assessment.output.scalar_advantage,
        "rdan_response_valid": assessment.output.response_valid,
    }
    assert all(torch.equal(data.batch[name], value) for name, value in evidence.items())

    failed = RollLikeBatch(
        batch={
            **data.batch,
            "rdan_eval_mask": eval_mask.clone(),
            "rdan_judge_failed": torch.tensor([True] + [False] * 7),
        },
        non_tensor_batch=data.non_tensor_batch,
    )
    failed.batch["rdan_eval_mask"][0, 1] = False
    with pytest.raises(ValueError, match="judge_failure"):
        make_roll_compute_advantage(
            certificate,
            method=method,
            quality_weight=quality_weight,
            mix_weight=mix_weight,
        )(failed)

    unsupported = RollLikeBatch(
        batch={
            **data.batch,
            "rdan_unsupported_hard": torch.tensor([True] + [False] * 7),
        },
        non_tensor_batch=data.non_tensor_batch,
    )
    with pytest.raises(ValueError, match="unsupported_hard_route"):
        make_roll_compute_advantage(
            certificate,
            method=method,
            quality_weight=quality_weight,
            mix_weight=mix_weight,
        )(unsupported)

    mismatch = "rl_aon" if method != "rl_aon" else "rl_csr"
    with pytest.raises(ValueError, match="method"):
        make_roll_compute_advantage(certificate, method=mismatch)
    if method in {"rdan_scalar", "rtt_papo_response"}:
        with pytest.raises(ValueError, match="quality weight"):
            make_roll_compute_advantage(certificate, method=method, quality_weight=0.25)
    if method == "rl_mix":
        with pytest.raises(ValueError, match="mix weight"):
            make_roll_compute_advantage(certificate, method=method, mix_weight=0.5)

    tampered = dict(payload)
    tampered["mix_weight" if method in {"rdan_scalar", "rtt_papo_response"} else "quality_weight"] = None
    body = {key: value for key, value in tampered.items() if key != "certificate_id"}
    tampered["certificate_id"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    path = tmp_path / f"{method}.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        require_train_certificate(
            path,
            method=method,
            quality_weight=quality_weight,
            mix_weight=mix_weight,
        )
    with pytest.raises(ValueError, match="ready"):
        make_roll_compute_advantage(
            tampered,
            method=method,
            quality_weight=quality_weight,
            mix_weight=mix_weight,
        )


@pytest.mark.parametrize(
    ("assessment", "reason"),
    [
        (_assessment(response_variance=False), "zero_response_variance"),
        (_assessment(quality_active_prompts=25), "low_quality_active_group_rate"),
    ],
)
def test_aggregate_readiness_rejects_weak_signal(assessment, reason: str) -> None:
    certificate = _certificate(assessment)
    assert assessment.batch_valid
    assert not certificate.ready
    assert reason in certificate.reasons


def test_certificate_rejects_quality_weight_different_from_assessment() -> None:
    with pytest.raises(ValueError, match="assessed batches"):
        build_preflight_certificate(
            [_assessment()],
            config_sha256="a" * 64,
            source_sha256={"rows": "b" * 64},
            optimizer_updates=0,
            quality_weight=0.25,
        )


def test_soft_only_rows_are_batch_invalid_and_never_receive_quality_credit() -> None:
    keys, scores, rubric_mask, eval_mask, hard_mask = _batch()
    hard_mask[:8] = False
    assessment = assess_scalar_batch(keys, scores, rubric_mask, eval_mask, hard_mask)
    assert not assessment.batch_valid
    assert "soft_only_response" in assessment.reasons
    assert not bool(assessment.output.hard_pass[:8].any())
    assert not bool(assessment.output.quality_eligible[:8].any())
    assert not bool(assessment.output.quality_advantage[:8].any())
    assert not _certificate(assessment).ready


def test_hybrid_preflight_accepts_soft_only_group_with_independent_quality_signal() -> None:
    keys, scores, rubric_mask, eval_mask, hard_mask = _batch(prompt_count=2)
    hard_mask[:8] = False
    assessment = assess_scalar_batch(
        keys,
        scores,
        rubric_mask,
        eval_mask,
        hard_mask,
        method="rtt_papo_response",
        quality_weight=0.5,
    )
    certificate = build_preflight_certificate(
        [assessment],
        method="rtt_papo_response",
        config_sha256="a" * 64,
        source_sha256={"rows": "b" * 64},
        optimizer_updates=0,
        quality_weight=0.5,
        min_prompts=2,
        max_prompts=2,
    )

    assert assessment.batch_valid
    assert assessment.response_active_groups == 1
    assert assessment.quality_active_groups == 2
    assert assessment.output.raw_aon[:8].tolist() == [1.0] * 8
    assert not bool(assessment.output.response_advantage[:8].any())
    assert bool(assessment.output.quality_advantage[:8].abs().gt(0).any())
    assert certificate.ready
    assert certificate.metrics["quality_active_group_rate"] == 1.0
    assert certificate.as_dict()["quality_weight"] == 0.5


def test_hybrid_soft_only_judge_failure_fails_bridge_closed() -> None:
    keys, scores, rubric_mask, eval_mask, hard_mask = _batch(prompt_count=2)
    hard_mask[:8] = False
    eval_mask[0, 1] = False
    judge_failed = torch.zeros(len(keys), dtype=torch.bool)
    judge_failed[0] = True

    assessment = assess_scalar_batch(
        keys,
        scores,
        rubric_mask,
        eval_mask,
        hard_mask,
        method="rtt_papo_response",
        judge_failed=judge_failed,
        quality_weight=0.5,
    )

    assert not assessment.batch_valid
    assert "judge_failure" in assessment.reasons
    assert "invalid_evaluator_output" in assessment.reasons
    assert not assessment.output.response_valid[0]
    assert not assessment.output.quality_eligible[0]
    assert assessment.output.scalar_advantage[0] == 0


@pytest.mark.parametrize("failure", ["unsupported", "judge"])
def test_unsupported_routes_and_judge_failures_fail_closed(failure: str) -> None:
    keys, scores, rubric_mask, eval_mask, hard_mask = _batch()
    unsupported = torch.zeros(len(keys), dtype=torch.bool)
    judge_failed = torch.zeros(len(keys), dtype=torch.bool)
    if failure == "unsupported":
        unsupported[0] = True
    else:
        judge_failed[1] = True
        eval_mask[1, 1] = False
    assessment = assess_scalar_batch(
        keys,
        scores,
        rubric_mask,
        eval_mask,
        hard_mask,
        unsupported_hard=unsupported,
        judge_failed=judge_failed,
    )
    assert not assessment.batch_valid
    if failure == "judge":
        assert not assessment.output.response_valid[1]
        assert not assessment.output.quality_eligible[1]
        assert assessment.output.scalar_advantage[1] == 0
    assert not _certificate(assessment).ready


def test_hard_evaluator_failure_is_excluded_from_bridge_group_normalization() -> None:
    keys, scores, rubric_mask, eval_mask, hard_mask = _batch()
    eval_mask[1, 0] = False
    assessment = assess_scalar_batch(keys, scores, rubric_mask, eval_mask, hard_mask)

    assert not assessment.output.response_valid[1]
    assert assessment.output.response_advantage[1] == 0
    assert assessment.output.scalar_advantage[1] == 0
    assert torch.equal(
        assessment.output.response_advantage,
        group_advantages(
            assessment.output.selected_raw_reward,
            8,
            valid=assessment.output.response_valid,
        ),
    )
    assert not assessment.output.training_ready
    assert not assessment.batch_valid
    assert "invalid_evaluator_output" in assessment.reasons
    assert not _certificate(assessment).ready


def test_invalid_grouping_and_optimizer_update_fail() -> None:
    keys, scores, rubric_mask, eval_mask, hard_mask = _batch()
    keys[1] = "wrong"
    with pytest.raises(ValueError, match="contiguous and complete"):
        assess_scalar_batch(keys, scores, rubric_mask, eval_mask, hard_mask)

    certificate = _certificate(_assessment(), optimizer_updates=1)
    assert not certificate.ready
    assert "optimizer_update_observed" in certificate.reasons


def test_certificate_digest_detects_tampering(tmp_path) -> None:
    certificate = _certificate(_assessment())
    payload = certificate.as_dict()
    body = {key: value for key, value in payload.items() if key != "certificate_id"}
    assert (
        certificate.certificate_id
        == hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        ).hexdigest()
    )
    payload["metrics"]["optimizer_updates"] = 3
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        require_train_certificate(path)


def test_legacy_certificate_without_bound_quality_weight_is_rejected(tmp_path: Path) -> None:
    payload = _certificate(_assessment()).as_dict()
    payload["schema_version"] = 1
    payload.pop("quality_weight")
    body = {key: value for key, value in payload.items() if key != "certificate_id"}
    payload["certificate_id"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="not ready"):
        require_train_certificate(path)
