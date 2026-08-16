"""RDAN-GRPO: rubric decoupled advantage normalization for instruction following RL."""

from rdan_grpo.advantages import group_advantages
from rdan_grpo.bridge import assess_batch, inject_advantages, make_advantage_fn
from rdan_grpo.loss import clipped_actor_loss
from rdan_grpo.rewards import QualityScores, RubricRewards, extract_quality, score_rubrics
from rdan_grpo.scalar import ScalarMethod, ScalarOutput, build_scalar_output

__all__ = [
    "QualityScores",
    "RubricRewards",
    "ScalarMethod",
    "ScalarOutput",
    "assess_batch",
    "build_scalar_output",
    "clipped_actor_loss",
    "extract_quality",
    "group_advantages",
    "inject_advantages",
    "make_advantage_fn",
    "score_rubrics",
]
