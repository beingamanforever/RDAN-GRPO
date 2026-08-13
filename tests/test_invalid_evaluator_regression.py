import pytest
import torch

from rdan_grpo.advantages import group_advantages
from rdan_grpo.loss import clipped_actor_loss


def test_invalid_evaluator_reward_has_no_policy_signal() -> None:
    rewards = torch.tensor([1.0, 1.0, 1.0, 0.0])
    valid = torch.tensor([True, True, True, False])

    advantages = group_advantages(rewards, group_size=4, valid=valid)

    assert torch.equal(advantages[valid], torch.zeros(3))
    assert advantages[valid].mean().item() == 0.0
    assert advantages[~valid].item() == 0.0

    log_probs = torch.nn.Parameter(torch.zeros(4, 1))
    loss = clipped_actor_loss(
        log_probs,
        torch.zeros_like(log_probs),
        advantages.unsqueeze(-1),
        torch.ones_like(log_probs, dtype=torch.bool),
        clip_low=0.2,
    )
    loss.backward()

    assert log_probs.grad is not None
    assert log_probs.grad[~valid].item() == 0.0


def test_group_advantages_centers_and_scales_only_valid_rewards() -> None:
    rewards = torch.tensor([0.0, 1.0, 2.0, 100.0])
    valid = torch.tensor([True, True, True, False])

    advantages = group_advantages(rewards, group_size=4, valid=valid)

    assert advantages[valid].mean().item() == pytest.approx(0.0, abs=1e-7)
    assert advantages[valid].std().item() == pytest.approx(1.0, abs=2e-6)
    assert advantages[~valid].item() == 0.0


@pytest.mark.parametrize(
    "valid",
    [torch.ones(4), torch.ones(2, dtype=torch.bool), torch.ones((2, 2), dtype=torch.bool)],
)
def test_group_advantages_rejects_invalid_validity_masks(valid: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="valid must be boolean and match rewards"):
        group_advantages(torch.ones(4), group_size=4, valid=valid)


def test_group_advantages_rejects_invalid_group_structure() -> None:
    with pytest.raises(ValueError, match="group_size must be positive and divide the response count"):
        group_advantages(torch.ones(4), group_size=3, valid=torch.ones(4, dtype=torch.bool))
