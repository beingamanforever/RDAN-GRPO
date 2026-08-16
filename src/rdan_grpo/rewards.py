"""Rubric reward construction and hard-soft channel extraction."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RubricRewards:
    """Response rewards and a mask identifying complete evaluator outputs."""

    aon: Tensor
    csr: Tensor
    valid: Tensor


@dataclass(frozen=True)
class QualityScores:
    """Independent hard and soft validity, soft quality, and conditional eligibility."""

    hard_pass: Tensor
    quality: Tensor
    quality_valid: Tensor
    eligible: Tensor


def score_rubrics(scores: Tensor, rubric_mask: Tensor, eval_mask: Tensor) -> RubricRewards:
    """Compute all-or-nothing and mean-satisfaction rewards over the active rubrics.

    Active scores must be finite and in the signed interval ``[-1, 1]``.
    A response with no active rubric or any unevaluated active rubric scores zero and is invalid.
    """

    _check_score_inputs(scores, rubric_mask, eval_mask)
    valid = _response_valid(scores, rubric_mask, eval_mask)
    count = rubric_mask.sum(dim=-1).clamp_min(1)
    clean = torch.where(rubric_mask & eval_mask & torch.isfinite(scores), scores, 0.0)
    unit = torch.where(rubric_mask, (clean + 1.0) / 2.0, 0.0)
    csr = unit.sum(dim=-1) / count
    aon = ((scores == 1) | ~rubric_mask).all(dim=-1).to(scores.dtype)
    keep = valid.to(scores.dtype)
    return RubricRewards(aon=aon * keep, csr=csr * keep, valid=valid)


def extract_quality(scores: Tensor, rubric_mask: Tensor, eval_mask: Tensor, hard_mask: Tensor) -> QualityScores:
    """Extract hard-rubric pass and mean soft-rubric quality on independent validity.

    Hard evaluator failure affects only ``hard_pass``; judge failure affects only ``quality_valid``.
    A response is quality-eligible only when it passed every hard rubric and its soft judgment is intact.
    """

    _check_score_inputs(scores, rubric_mask, eval_mask)
    if hard_mask.dtype != torch.bool or hard_mask.shape != scores.shape:
        raise ValueError("hard_mask must be boolean and match scores")

    hard = rubric_mask & hard_mask
    soft = rubric_mask & ~hard_mask
    hard_valid = ((eval_mask & ((scores == -1) | (scores == 1))) | ~hard).all(dim=-1)
    hard_pass = hard_valid & ((scores == 1) | ~hard).all(dim=-1)
    quality_valid = soft.any(dim=-1) & (
        (eval_mask & torch.isfinite(scores) & (scores >= -1) & (scores <= 1)) | ~soft
    ).all(dim=-1)
    count = soft.sum(dim=-1).clamp_min(1)
    unit = torch.where(soft & torch.isfinite(scores), (scores + 1.0) / 2.0, 0.0)
    quality = torch.where(quality_valid, unit.sum(dim=-1) / count, 0.0)
    return QualityScores(
        hard_pass=hard_pass,
        quality=quality,
        quality_valid=quality_valid,
        eligible=hard_pass & quality_valid,
    )


def _check_score_inputs(scores: Tensor, rubric_mask: Tensor, eval_mask: Tensor) -> None:
    if not scores.is_floating_point() or scores.ndim != 2:
        raise ValueError("scores must be a floating tensor with shape [responses, rubrics]")
    if rubric_mask.dtype != torch.bool or rubric_mask.shape != scores.shape:
        raise ValueError("rubric_mask must be boolean and match scores")
    if eval_mask.dtype != torch.bool or eval_mask.shape != scores.shape:
        raise ValueError("eval_mask must be boolean and match scores")


def _response_valid(scores: Tensor, rubric_mask: Tensor, eval_mask: Tensor) -> Tensor:
    in_range = torch.isfinite(scores) & (scores >= -1) & (scores <= 1)
    return rubric_mask.any(dim=-1) & (eval_mask | ~rubric_mask).all(dim=-1) & (in_range | ~rubric_mask).all(dim=-1)
