from __future__ import annotations

from typing import Any

import numpy as np

from .config import ScoringConfig
from .types import PromptSet


def _click_containment(mask: np.ndarray, prompts: PromptSet) -> float:
    score = 1.0
    for x, y in prompts.positive:
        if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]) or not mask[y, x]:
            score -= 0.5
    for x, y in prompts.negative:
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
            score -= 0.3
    return float(max(0.0, min(1.0, score)))


def _roi_consistency(mask: np.ndarray, roi: tuple[int, int, int, int] | None) -> float:
    if roi is None:
        return 1.0
    x1, y1, x2, y2 = roi
    roi_mask = np.zeros_like(mask, dtype=bool)
    roi_mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = True
    overlap = float(np.logical_and(mask, roi_mask).sum())
    mask_area = float(mask.sum())
    if mask_area == 0:
        return 0.0
    return overlap / mask_area


def _prompt_span_coverage(features: dict[str, Any], prompts: PromptSet, config: ScoringConfig) -> float:
    if len(prompts.positive) < 2:
        return 1.0
    anchor = features.get("anchor_endpoint")
    tip = features.get("tip_endpoint")
    if anchor is None or tip is None:
        return 0.0

    axis = np.array([tip[0] - anchor[0], tip[1] - anchor[1]], dtype=np.float32)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-6:
        return 0.0
    direction = axis / axis_norm
    prompt_projections = [float(np.dot(np.array(point, dtype=np.float32), direction)) for point in prompts.positive]
    prompt_span = max(prompt_projections) - min(prompt_projections)
    average_thickness = float(features.get("average_thickness_px", 0.0))
    desired_span = prompt_span + config.target_prompt_margin_thickness * max(average_thickness, 1.0)
    if desired_span <= 1e-6:
        return 1.0
    mask_span = axis_norm
    return float(max(0.0, min(1.0, mask_span / desired_span)))


def score_candidate(
    mask: np.ndarray,
    sam_score: float,
    features: dict[str, Any],
    prompts: PromptSet,
    roi: tuple[int, int, int, int] | None,
    config: ScoringConfig,
) -> tuple[float, dict[str, float]]:
    elongation_term = min(
        1.0,
        float(features.get("elongation", 0.0)) / max(config.target_elongation, 1e-6),
    )
    average_thickness = float(features.get("average_thickness_px", 0.0))
    width_std = float(features.get("width_std_px", 0.0))
    width_consistency = 1.0 - min(1.0, width_std / max(average_thickness, 1.0))
    click_term = _click_containment(mask, prompts)
    roi_term = _roi_consistency(mask, roi)
    centerline_quality = min(
        1.0,
        float(features.get("centerline_length_to_thickness_ratio", 0.0))
        / max(config.target_centerline_ratio, 1e-6),
    )
    prompt_span_term = _prompt_span_coverage(features, prompts, config)
    blob_penalty = min(1.0, max(0.0, 1.0 - elongation_term))
    branch_penalty = min(1.0, float(features.get("skeleton_branch_point_count", 0.0)) / 3.0)
    endpoint_penalty = min(
        1.0,
        abs(float(features.get("skeleton_endpoint_count", 0.0)) - float(config.target_endpoint_count))
        / max(float(config.target_endpoint_count), 1.0),
    )

    terms = {
        "sam_score": float(sam_score),
        "elongation": elongation_term,
        "width_consistency": width_consistency,
        "click_containment": click_term,
        "roi_consistency": roi_term,
        "centerline_quality": centerline_quality,
        "prompt_span_coverage": prompt_span_term,
        "blob_penalty": blob_penalty,
        "branch_penalty": branch_penalty,
        "endpoint_penalty": endpoint_penalty,
    }
    combined = (
        config.sam_score_weight * terms["sam_score"]
        + config.elongation_weight * terms["elongation"]
        + config.width_consistency_weight * terms["width_consistency"]
        + config.click_containment_weight * terms["click_containment"]
        + config.roi_consistency_weight * terms["roi_consistency"]
        + config.centerline_quality_weight * terms["centerline_quality"]
        + config.prompt_span_weight * terms["prompt_span_coverage"]
        - config.blob_penalty_weight * terms["blob_penalty"]
        - config.branch_penalty_weight * terms["branch_penalty"]
        - config.endpoint_penalty_weight * terms["endpoint_penalty"]
    )
    return float(combined), terms
