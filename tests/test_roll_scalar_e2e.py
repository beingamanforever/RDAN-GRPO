from __future__ import annotations

from types import MappingProxyType

import pytest
import torch

from rdan_grpo.advantages import group_advantages, quality_advantages
from rdan_grpo.roll_scalar import build_scalar_output, validate_groups


def _batch() -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_keys = ["alpha"] * 4 + ["beta"] * 4
    scores = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
    )
    rubric_mask = torch.ones_like(scores, dtype=torch.bool)
    eval_mask = torch.ones_like(scores, dtype=torch.bool)
    hard_mask = torch.tensor([[True, False, False]]).expand_as(scores)
    return prompt_keys, scores, rubric_mask, eval_mask, hard_mask


def test_all_scalar_methods_end_to_end_without_mutating_inputs() -> None:
    prompt_keys, scores, rubric_mask, eval_mask, hard_mask = _batch()
    snapshots = (list(prompt_keys), scores.clone(), rubric_mask.clone(), eval_mask.clone(), hard_mask.clone())
    outputs = {
        method: build_scalar_output(
            method,
            prompt_keys,
            scores,
            rubric_mask,
            eval_mask,
            hard_mask,
            group_size=4,
            mix_weight=0.25 if method == "rl_mix" else None,
            quality_weight=0.5 if method == "rdan_scalar" else None,
        )
        for method in ("rl_aon", "rl_csr", "rl_mix", "rdan_scalar")
    }

    aon = outputs["rl_aon"]
    assert aon.raw_aon.tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    assert aon.raw_csr.tolist() == pytest.approx([1.0, 1 / 3, 2 / 3, 2 / 3, 2 / 3, 1.0, 1 / 3, 1 / 3])
    assert aon.raw_signed_csr.tolist() == pytest.approx([1.0, -1 / 3, 1 / 3, 1 / 3, 1 / 3, 1.0, -1 / 3, -1 / 3])
    assert aon.hard_pass.tolist() == [True, True, False, True, True, True, True, False]
    assert aon.raw_quality.tolist() == pytest.approx([1.0, 0.0, 1.0, 0.5, 0.5, 1.0, 0.0, 0.5])
    assert bool(aon.quality_valid.all())
    assert aon.quality_eligible.tolist() == [True, True, False, True, True, True, True, False]

    assert torch.equal(aon.selected_raw_reward, aon.raw_aon)
    assert torch.equal(outputs["rl_csr"].selected_raw_reward, aon.raw_csr)
    mixed = 0.25 * aon.raw_aon + 0.75 * aon.raw_csr
    assert torch.allclose(outputs["rl_mix"].selected_raw_reward, mixed)
    assert torch.allclose(outputs["rl_mix"].response_advantage, group_advantages(mixed, 4))
    assert not torch.allclose(
        outputs["rl_mix"].response_advantage,
        0.25 * group_advantages(aon.raw_aon, 4) + 0.75 * group_advantages(aon.raw_csr, 4),
    )

    expected_quality = quality_advantages(aon.raw_quality, aon.quality_eligible, 4)
    assert torch.equal(outputs["rdan_scalar"].quality_advantage, expected_quality)
    assert torch.allclose(
        outputs["rdan_scalar"].scalar_advantage,
        outputs["rdan_scalar"].response_advantage + 0.5 * expected_quality,
    )
    assert all(output.training_ready for output in outputs.values())
    assert all(bool(torch.isfinite(output.scalar_advantage).all()) for output in outputs.values())
    assert isinstance(aon.diagnostics, MappingProxyType)
    assert aon.diagnostics["response_valid_rate"] == 1.0

    assert prompt_keys == snapshots[0]
    for current, snapshot in zip((scores, rubric_mask, eval_mask, hard_mask), snapshots[1:], strict=True):
        assert torch.equal(current, snapshot)


@pytest.mark.parametrize(
    ("prompt_keys", "message"),
    [
        (["a"] * 3, "divisible"),
        (["a", "b", "a", "a"], "contiguous and complete"),
        (["a", "a", "b", "b", "a", "a"], "more than one group"),
    ],
)
def test_validate_groups_rejects_incomplete_noncontiguous_and_repeated_groups(
    prompt_keys: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_groups(prompt_keys, 2)


@pytest.mark.parametrize(
    ("method", "weights"),
    [
        ("rl_aon", {}),
        ("rl_csr", {}),
        ("rl_mix", {"mix_weight": 0.25}),
        ("rdan_scalar", {"quality_weight": 0.5}),
    ],
)
def test_invalid_evaluator_fails_closed_without_scalar_credit(method: str, weights: dict[str, float]) -> None:
    prompt_keys, scores, rubric_mask, eval_mask, hard_mask = _batch()
    eval_mask[2, 1] = False
    output = build_scalar_output(
        method,
        prompt_keys,
        scores,
        rubric_mask,
        eval_mask,
        hard_mask,
        group_size=4,
        **weights,
    )

    assert not output.response_valid[2]
    assert output.response_advantage[2] == 0
    assert output.scalar_advantage[2] == 0
    assert torch.equal(
        output.response_advantage,
        group_advantages(output.selected_raw_reward, 4, valid=output.response_valid),
    )
    assert not output.training_ready
    assert output.diagnostics["invalid_response_count"] == 1
    assert output.diagnostics["finite"] is True
    assert bool(torch.isfinite(output.selected_raw_reward).all())
    assert bool(torch.isfinite(output.scalar_advantage).all())


def test_hybrid_soft_only_group_has_vacuous_response_gate_and_normalized_quality() -> None:
    keys = ["soft"] * 4 + ["mixed"] * 4
    scores = torch.tensor(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
        ]
    )
    rubric_mask = torch.ones_like(scores, dtype=torch.bool)
    eval_mask = torch.ones_like(scores, dtype=torch.bool)
    hard_mask = torch.zeros_like(scores, dtype=torch.bool)
    hard_mask[4:, 0] = True

    output = build_scalar_output(
        "rtt_papo_response",
        keys,
        scores,
        rubric_mask,
        eval_mask,
        hard_mask,
        group_size=4,
        quality_weight=0.5,
    )

    assert output.raw_aon[:4].tolist() == [1.0] * 4
    assert output.hard_pass[:4].tolist() == [True] * 4
    assert output.quality_eligible[:4].tolist() == [True] * 4
    assert output.diagnostics["hard_pass_rate"] == pytest.approx(7 / 8)
    assert output.diagnostics["quality_eligible_rate"] == pytest.approx(7 / 8)
    assert torch.equal(output.response_advantage[:4], torch.zeros(4))
    expected_soft_quality = quality_advantages(output.raw_quality[:4], torch.ones(4, dtype=torch.bool), 4)
    assert torch.allclose(output.quality_advantage[:4], expected_soft_quality)
    assert bool(output.quality_advantage[:4].abs().gt(0).any())
    assert output.raw_aon[4:].tolist() == [1.0, 1.0, 0.0, 1.0]
    assert output.quality_eligible[4:].tolist() == [True, True, False, True]
    assert output.training_ready

    strict = build_scalar_output(
        "rdan_scalar",
        keys,
        scores,
        rubric_mask,
        eval_mask,
        hard_mask,
        group_size=4,
        quality_weight=0.5,
    )
    assert not bool(strict.response_valid[:4].any())
    assert not bool(strict.quality_eligible[:4].any())
    assert not strict.training_ready


def test_hybrid_judge_failure_stays_fail_closed() -> None:
    keys = ["soft"] * 4
    scores = torch.tensor([[-1.0], [1.0], [-1.0], [1.0]])
    rubric_mask = torch.ones_like(scores, dtype=torch.bool)
    eval_mask = torch.ones_like(scores, dtype=torch.bool)
    eval_mask[1, 0] = False
    hard_mask = torch.zeros_like(scores, dtype=torch.bool)

    output = build_scalar_output(
        "rtt_papo_response",
        keys,
        scores,
        rubric_mask,
        eval_mask,
        hard_mask,
        group_size=4,
        quality_weight=0.5,
    )

    assert output.raw_aon.tolist() == [1.0] * 4
    assert not output.response_valid[1]
    assert not output.quality_eligible[1]
    assert output.response_advantage[1] == 0
    assert output.quality_advantage[1] == 0
    assert output.scalar_advantage[1] == 0
    assert not output.training_ready


@pytest.mark.parametrize(("name", "value"), [("mix_weight", float("nan")), ("quality_weight", float("inf"))])
def test_nonfinite_weights_fail(name: str, value: float) -> None:
    prompt_keys, scores, rubric_mask, eval_mask, hard_mask = _batch()
    kwargs = {name: value}
    method = "rl_mix" if name == "mix_weight" else "rdan_scalar"
    with pytest.raises(ValueError, match="must be finite"):
        build_scalar_output(
            method,
            prompt_keys,
            scores,
            rubric_mask,
            eval_mask,
            hard_mask,
            group_size=4,
            **kwargs,
        )
