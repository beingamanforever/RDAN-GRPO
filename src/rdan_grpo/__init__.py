"""Compact reward, advantage, and loss functions for RDAN-GRPO."""

from rdan_grpo.advantages import group_advantages, quality_advantages
from rdan_grpo.loss import clipped_actor_loss
from rdan_grpo.rewards import QualityScores, RubricRewards, extract_quality, score_rubrics
from rdan_grpo.bridge import (
    BatchAssessment,
    assess_scalar_batch,
    inject_roll_advantages,
    install_roll_adapter,
    make_roll_compute_advantage,
)

__all__ = [
    "BatchAssessment",
    "QualityScores",
    "RubricRewards",
    "assess_scalar_batch",
    "clipped_actor_loss",
    "extract_quality",
    "group_advantages",
    "inject_roll_advantages",
    "install_roll_adapter",
    "make_roll_compute_advantage",
    "quality_advantages",
    "score_rubrics",
]
