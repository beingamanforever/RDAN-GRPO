import math

import pytest
import torch

from rdan_grpo import (
    clipped_actor_loss,
    compose_advantages,
    extract_quality,
    group_advantages,
    quality_advantages,
    score_rubrics,
    token_advantages,
)


def test_rdan_pipeline_end_to_end() -> None:
    scores = torch.tensor(
        [
            [1.0, 1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0, float("nan")],
            [1.0, 1.0, 0.0, 1.0],
            [-1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0, 1.0],
            [-1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0, -1.0],
        ]
    )
    rubric_mask = torch.ones_like(scores, dtype=torch.bool)
    rubric_mask[2, 3] = False
    eval_mask = torch.ones_like(rubric_mask)
    eval_mask[3, 3] = False
    hard_mask = torch.tensor([[True, True, False, False]]).expand_as(scores)

    rewards = score_rubrics(scores, rubric_mask, eval_mask)
    quality = extract_quality(scores, rubric_mask, eval_mask, hard_mask)

    assert rewards.valid[:4].tolist() == [True, True, True, False]
    assert bool(rewards.valid[4:].all())
    assert rewards.aon[:4].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert rewards.csr[0].item() == pytest.approx(0.75)
    assert rewards.signed_csr[0].item() == pytest.approx(0.5)
    assert rewards.csr[3].item() == rewards.signed_csr[3].item() == 0.0
    assert quality.hard_pass[:8].tolist() == [True, True, False, False, True, False, False, False]
    assert quality.quality[:4].tolist() == [0.5, 1.0, 1.0, 0.0]
    assert not quality.valid[3] and quality.quality[3] == 0

    nonbinary = scores.clone()
    nonbinary[0, 0] = 0.5
    strict_rewards = score_rubrics(nonbinary, rubric_mask, eval_mask)
    strict_quality = extract_quality(nonbinary, rubric_mask, eval_mask, hard_mask)
    assert strict_rewards.aon[0] == 0
    assert not strict_quality.valid[0] and not strict_quality.hard_pass[0]

    response = group_advantages(rewards.csr, group_size=4, valid=rewards.valid)
    assert response[:3].mean().item() == pytest.approx(0.0, abs=1e-6)
    assert response[:3].std().item() == pytest.approx(1.0, abs=1e-5)
    assert response[3].item() == 0.0

    token_mask = torch.tensor(
        [[True, True, True, True, False]] * 4
        + [[True, True, True, False, False]] * 4
        + [[True, True, True, True, True]] * 4
    )
    relevance = torch.arange(1, 12 * 4 * 5 + 1, dtype=torch.float32).reshape(12, 4, 5)
    clean_scores = torch.nan_to_num(scores).unsqueeze(-1)
    token = token_advantages(relevance * clean_scores, rubric_mask, token_mask, rewards.valid)
    assert token.shape == token_mask.shape
    assert bool((token[~token_mask] == 0).all())
    assert bool((token[3] == 0).all())

    token_example = torch.tensor([[[1.0, 2.0, 3.0, 99.0], [2.0, 4.0, 6.0, 99.0], [8.0, 8.0, 8.0, 8.0]]])
    example = token_advantages(
        token_example,
        torch.tensor([[True, True, False]]),
        torch.tensor([[True, True, True, False]]),
        torch.tensor([True]),
    )
    expected = torch.tensor([[-math.sqrt(1.5), 0.0, math.sqrt(1.5), 0.0]])
    assert torch.allclose(example, expected, atol=2e-6)

    q_off = quality_advantages(quality.quality, quality.hard_pass, group_size=4, mode="off")
    q_raw = quality_advantages(quality.quality, quality.hard_pass, group_size=4, mode="raw")
    q_full = quality_advantages(quality.quality, quality.hard_pass, group_size=4, mode="full_group")
    q_subset = quality_advantages(quality.quality, quality.hard_pass, group_size=4, mode="hard_valid")
    assert bool((q_off == 0).all())
    assert torch.equal(q_raw, quality.quality)
    assert torch.allclose(q_subset[:2], torch.tensor([-math.sqrt(0.5), math.sqrt(0.5)]), atol=3e-6)
    assert bool((q_subset[4:8] == 0).all())
    assert bool((q_subset[8:] == 0).all())
    assert bool((q_full[8:] == 0).all())
    assert quality_advantages(torch.tensor([0.2]), torch.tensor([True]), 1, mode="full_group").item() == 0
    assert q_raw[3] == 0 and q_full[3] <= 0 and q_subset[3] == 0

    rtt = (response.unsqueeze(-1) + 0.3 * token) * token_mask
    off = compose_advantages(response, token, q_off, token_mask, beta=0.3, quality_weight=0.7)
    advantages = compose_advantages(response, token, q_subset, token_mask, beta=0.3, quality_weight=0.7)
    assert torch.equal(off, rtt)
    assert bool(torch.isfinite(advantages).all())

    old_log_probs = torch.zeros_like(advantages)
    log_probs = torch.nn.Parameter(torch.full_like(advantages, 0.3))
    optimizer = torch.optim.SGD([log_probs], lr=0.05)
    before = log_probs.detach().clone()
    loss = clipped_actor_loss(log_probs, old_log_probs, advantages, token_mask, clip_low=0.2, clip_high=0.27)
    optimizer.zero_grad()
    loss.backward()
    assert torch.isfinite(loss)
    assert log_probs.grad is not None and bool(torch.isfinite(log_probs.grad).all())
    optimizer.step()
    assert not torch.equal(log_probs.detach(), before)

    asymmetric = clipped_actor_loss(
        torch.tensor([[math.log(1.25), math.log(1.25)], [0.0, 0.0]]),
        torch.zeros(2, 2),
        torch.tensor([[1.0, 1.0], [0.0, 9.0]]),
        torch.tensor([[True, True], [True, False]]),
        clip_low=0.2,
        clip_high=0.27,
    )
    assert asymmetric.item() == pytest.approx(-0.625)

    with pytest.raises(ValueError, match="importance ratios"):
        clipped_actor_loss(
            torch.tensor([[1000.0]], requires_grad=True),
            torch.zeros(1, 1),
            torch.ones(1, 1),
            torch.ones(1, 1, dtype=torch.bool),
            clip_low=0.2,
            clip_high=0.27,
        )

    padded_log_probs = torch.tensor([[0.0, 1000.0]], requires_grad=True)
    padded_loss = clipped_actor_loss(
        padded_log_probs,
        torch.tensor([[0.0, float("nan")]]),
        torch.tensor([[1.0, float("nan")]]),
        torch.tensor([[True, False]]),
        clip_low=0.2,
        clip_high=0.27,
    )
    padded_loss.backward()
    assert padded_log_probs.grad is not None
    assert torch.equal(padded_log_probs.grad, torch.tensor([[-1.0, 0.0]]))
