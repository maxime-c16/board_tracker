from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import remove_small_holes

from .config import PostprocessConfig
from .types import PromptSet


def keep_largest_component(mask: np.ndarray, min_area: int) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask_u8, dtype=bool)
    best_idx = 0
    best_area = 0
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area >= min_area and area > best_area:
            best_idx = idx
            best_area = area
    if best_idx == 0:
        return np.zeros_like(mask_u8, dtype=bool)
    return labels == best_idx


def cleanup_mask(mask: np.ndarray, config: PostprocessConfig) -> tuple[np.ndarray, dict[str, int]]:
    binary = mask.astype(bool)
    largest = keep_largest_component(binary, config.min_component_area)
    if not largest.any():
        return largest, {"component_area": 0, "hole_fill_area": config.hole_fill_area}

    filled = remove_small_holes(largest, area_threshold=config.hole_fill_area)
    kernel_size = max(1, int(config.smoothing_kernel))
    open_size = max(1, int(config.opening_kernel))
    smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    cleaned = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_OPEN, open_kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, smooth_kernel)
    cleaned = ndimage.binary_fill_holes(cleaned > 0)
    cleaned = keep_largest_component(cleaned, config.min_component_area)
    metrics = {
        "component_area": int(cleaned.sum()),
        "hole_fill_area": config.hole_fill_area,
        "smoothing_kernel": kernel_size,
    }
    return cleaned.astype(bool), metrics


def _fit_axis(
    features: dict[str, object],
    prompts: PromptSet,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    centerline_points = [tuple(point) for point in features.get("centerline_points", [])]
    fit_points = centerline_points + list(prompts.positive)
    if len(fit_points) < 2:
        ys, xs = np.where(mask)
        if len(xs) < 2:
            return None
        step = max(1, len(xs) // 400)
        fit_points = [(int(x), int(y)) for x, y in zip(xs[::step], ys[::step])]
    data = np.array(fit_points, dtype=np.float32)
    vx, vy, x0, y0 = cv2.fitLine(data, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    direction = np.array([float(vx), float(vy)], dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return None
    direction /= norm
    if len(prompts.positive) >= 2:
        prompt_dir = np.array(
            [
                float(prompts.positive[-1][0] - prompts.positive[0][0]),
                float(prompts.positive[-1][1] - prompts.positive[0][1]),
            ],
            dtype=np.float32,
        )
        if float(np.dot(prompt_dir, direction)) < 0:
            direction *= -1.0
    origin = np.array([float(x0), float(y0)], dtype=np.float32)
    return origin, direction


def rectify_board_mask(
    mask: np.ndarray,
    features: dict[str, object],
    prompts: PromptSet,
    roi: tuple[int, int, int, int] | None,
    config: PostprocessConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    metrics: dict[str, object] = {"applied": False}
    if not config.enable_board_rectification:
        return mask, metrics
    if not bool(features.get("valid")):
        return mask, metrics
    if float(features.get("elongation", 0.0)) < config.rectification_min_elongation:
        return mask, metrics

    fit = _fit_axis(features, prompts, mask)
    if fit is None:
        return mask, metrics
    origin, direction = fit
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)

    ys, xs = np.where(mask)
    if len(xs) < 2:
        return mask, metrics
    mask_points = np.column_stack((xs, ys)).astype(np.float32)
    mask_proj = (mask_points - origin) @ direction
    prompt_points = np.array(prompts.positive, dtype=np.float32) if prompts.positive else np.empty((0, 2), dtype=np.float32)
    prompt_proj = (prompt_points - origin) @ direction if len(prompt_points) else np.empty((0,), dtype=np.float32)

    thickness = float(features.get("average_thickness_px", 0.0))
    if thickness <= 1.0:
        return mask, metrics
    half_width = max(2.0, 0.5 * thickness * float(config.rectification_width_scale))
    margin = max(2.0, thickness * float(config.rectification_prompt_margin_scale))

    min_t = float(mask_proj.min())
    max_t = float(mask_proj.max())
    if len(prompt_proj):
        min_t = min(min_t, float(prompt_proj.min()) - margin)
        max_t = max(max_t, float(prompt_proj.max()) + margin)

    offsets = np.abs((mask_points - origin) @ normal)
    span = max_t - min_t
    sample_window = max(2.0, min(span * 0.18, thickness * 4.0))
    start_band = offsets[mask_proj <= (min_t + sample_window)]
    end_band = offsets[mask_proj >= (max_t - sample_window)]

    def _band_half_width(values: np.ndarray) -> float:
        if len(values) == 0:
            return half_width
        estimated = float(np.percentile(values, 92))
        return float(np.clip(estimated, half_width * 0.7, half_width * 1.7))

    start_half_width = _band_half_width(start_band)
    end_half_width = _band_half_width(end_band)

    start = origin + direction * min_t
    end = origin + direction * max_t
    corners = np.array(
        [
            start + normal * start_half_width,
            start - normal * start_half_width,
            end - normal * end_half_width,
            end + normal * end_half_width,
        ],
        dtype=np.float32,
    )

    rectified = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillConvexPoly(rectified, np.round(corners).astype(np.int32), 1)
    cv2.circle(rectified, tuple(np.round(start).astype(int)), int(round(start_half_width)), 1, thickness=-1)
    cv2.circle(rectified, tuple(np.round(end).astype(int)), int(round(end_half_width)), 1, thickness=-1)
    rectified = rectified.astype(bool)
    if roi is not None:
        x1, y1, x2, y2 = roi
        roi_mask = np.zeros_like(rectified, dtype=bool)
        roi_mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = True
        rectified = np.logical_and(rectified, roi_mask)

    rectified = keep_largest_component(rectified, config.min_component_area)
    if not rectified.any():
        return mask, metrics

    metrics.update(
        {
            "applied": True,
            "half_width_px": half_width,
            "start_half_width_px": start_half_width,
            "end_half_width_px": end_half_width,
            "rectified_area": int(rectified.sum()),
        }
    )
    return rectified, metrics


def postprocess_mask(mask: np.ndarray, config: PostprocessConfig) -> tuple[np.ndarray, dict[str, int]]:
    return cleanup_mask(mask, config)
