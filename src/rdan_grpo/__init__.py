"""Compact reward, advantage, and loss functions for RDAN-GRPO."""

from rdan_grpo.advantages import compose_advantages, group_advantages, quality_advantages, token_advantages
from rdan_grpo.loss import clipped_actor_loss
from rdan_grpo.program import (
    ProgramBundle,
    ProgramContractError,
    build_judge_request,
    check_program,
    load_program,
    validate_program,
)
from rdan_grpo.rewards import QualityScores, RubricRewards, extract_quality, score_rubrics
from rdan_grpo.roll_bridge import (
    BatchAssessment,
    PreflightCertificate,
    assess_scalar_batch,
    build_preflight_certificate,
    inject_roll_advantages,
    install_roll_adapter,
    make_roll_compute_advantage,
    require_train_certificate,
)

__all__ = [
    "QualityScores",
    "RubricRewards",
    "ProgramBundle",
    "ProgramContractError",
    "BatchAssessment",
    "PreflightCertificate",
    "assess_scalar_batch",
    "build_preflight_certificate",
    "build_judge_request",
    "check_program",
    "clipped_actor_loss",
    "compose_advantages",
    "extract_quality",
    "group_advantages",
    "inject_roll_advantages",
    "install_roll_adapter",
    "load_program",
    "make_roll_compute_advantage",
    "quality_advantages",
    "require_train_certificate",
    "score_rubrics",
    "token_advantages",
    "validate_program",
]
