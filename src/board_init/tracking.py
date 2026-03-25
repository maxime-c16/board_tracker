from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .io import ensure_dir, write_json


TRACK_ORDER = ("anchor", "s25", "s50", "s75", "s90", "tip")
STATION_FRACTIONS = {"anchor": 0.0, "s25": 0.25, "s50": 0.5, "s75": 0.75, "s90": 0.9, "tip": 1.0}
TRUSTED_STATES = {"seed", "tracked", "recovered"}


@dataclass
class TrackingConfig:
    win_size: tuple[int, int] = (31, 31)
    max_level: int = 3
    max_error: float = 25.0
    fb_max_error_px: float = 2.5
    quality_min_points: int = 3
    save_overlay_video: bool = True
    max_side: int | None = None
    roi_padding_px: int = 72
    roi_normal_half_width_px: float = 28.0
    min_edge_points: int = 30
    min_support_coverage: float = 0.45
    min_station_support_score: float = 0.18
    min_mean_station_support_score: float = 0.28
    max_mean_line_distance_px: float = 8.5
    max_centerline_curvature_px: float = 12.0
    max_angle_delta_deg: float = 9.0
    max_length_ratio_delta: float = 0.18
    max_centroid_drift_norm: float = 0.32
    anchor_max_drift_norm: float = 0.10
    s25_max_drift_norm: float = 0.12
    s50_max_drift_norm: float = 0.18
    s75_max_drift_norm: float = 0.25
    s90_max_drift_norm: float = 0.30
    tip_max_drift_norm: float = 0.35


def load_tracking_points(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fit_partial_affine(
    previous_points: dict[str, list[int] | None],
    current_points: dict[str, list[int] | None],
) -> np.ndarray | None:
    src: list[list[float]] = []
    dst: list[list[float]] = []
    for name in TRACK_ORDER:
        prev_point = previous_points.get(name)
        curr_point = current_points.get(name)
        if prev_point is None or curr_point is None:
            continue
        src.append([float(prev_point[0]), float(prev_point[1])])
        dst.append([float(curr_point[0]), float(curr_point[1])])
    if len(src) < 2:
        return None
    matrix, _ = cv2.estimateAffinePartial2D(np.array(src, dtype=np.float32), np.array(dst, dtype=np.float32))
    return matrix


def _reference_segment_length(points: dict[str, list[int] | None]) -> float:
    anchor = points.get("anchor")
    tip = points.get("tip")
    if anchor is None or tip is None:
        return 0.0
    return float(np.linalg.norm(np.array(tip, dtype=np.float32) - np.array(anchor, dtype=np.float32)))


def _max_normalized_drift(name: str, config: TrackingConfig) -> float:
    return {
        "anchor": config.anchor_max_drift_norm,
        "s25": config.s25_max_drift_norm,
        "s50": config.s50_max_drift_norm,
        "s75": config.s75_max_drift_norm,
        "s90": config.s90_max_drift_norm,
        "tip": config.tip_max_drift_norm,
    }[name]


def _within_motion_limit(
    name: str,
    point: list[int] | None,
    reference_points: dict[str, list[int] | None],
    reference_length: float,
    config: TrackingConfig,
) -> bool:
    reference_point = reference_points.get(name)
    if point is None or reference_point is None or reference_length <= 1e-6:
        return point is not None
    displacement = float(
        np.linalg.norm(np.array(point, dtype=np.float32) - np.array(reference_point, dtype=np.float32))
    )
    return displacement <= _max_normalized_drift(name, config) * reference_length


def _apply_affine(point: list[int] | None, matrix: np.ndarray | None) -> list[int] | None:
    if point is None or matrix is None:
        return None
    x, y = point
    transformed = matrix @ np.array([float(x), float(y), 1.0], dtype=np.float32)
    return [int(round(float(transformed[0]))), int(round(float(transformed[1])))]


def _in_frame(point: list[int] | None, frame_shape: tuple[int, int, int]) -> bool:
    if point is None:
        return False
    h, w = frame_shape[:2]
    return 0 <= point[0] < w and 0 <= point[1] < h


def _line_angle_deg(anchor: np.ndarray, tip: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(float(tip[1] - anchor[1]), float(tip[0] - anchor[0]))))


def _angle_delta_deg(a: float, b: float) -> float:
    delta = (a - b + 180.0) % 360.0 - 180.0
    return abs(delta)


def _build_points_from_segment(anchor: np.ndarray, tip: np.ndarray) -> dict[str, list[int]]:
    points: dict[str, list[int]] = {}
    for name in TRACK_ORDER:
        s = STATION_FRACTIONS[name]
        point = anchor + s * (tip - anchor)
        points[name] = [int(round(float(point[0]))), int(round(float(point[1])))]
    return points


def _point_dict_to_arrays(points: dict[str, list[int] | None]) -> tuple[np.ndarray | None, np.ndarray | None]:
    anchor = points.get("anchor")
    tip = points.get("tip")
    if anchor is None or tip is None:
        return None, None
    return np.array(anchor, dtype=np.float32), np.array(tip, dtype=np.float32)


def _crop_bounds(points: dict[str, list[int] | None], frame_shape: tuple[int, int], padding: int) -> tuple[int, int, int, int]:
    available = np.array([point for point in points.values() if point is not None], dtype=np.int32)
    if len(available) == 0:
        return 0, 0, frame_shape[1], frame_shape[0]
    x1 = max(0, int(np.min(available[:, 0])) - padding)
    x2 = min(frame_shape[1], int(np.max(available[:, 0])) + padding + 1)
    y1 = max(0, int(np.min(available[:, 1])) - padding)
    y2 = min(frame_shape[0], int(np.max(available[:, 1])) + padding + 1)
    return x1, y1, x2, y2


def _sample_line_support(
    grad_mag: np.ndarray,
    station: np.ndarray,
    axis: np.ndarray,
    roi_offset: tuple[int, int],
    radius: int,
    scale: float,
) -> float:
    normal = np.array([-axis[1], axis[0]], dtype=np.float32)
    max_value = 0.0
    for offset in range(-radius, radius + 1):
        sample = station + offset * normal
        x = int(round(float(sample[0] - roi_offset[0])))
        y = int(round(float(sample[1] - roi_offset[1])))
        if 0 <= x < grad_mag.shape[1] and 0 <= y < grad_mag.shape[0]:
            max_value = max(max_value, float(grad_mag[y, x]))
    if scale <= 1e-6:
        return 0.0
    return float(np.clip(max_value / scale, 0.0, 1.0))


def _complete_predicted_geometry(
    current_points: dict[str, list[int] | None],
    tracked_points: dict[str, list[int] | None],
    frame_shape: tuple[int, int, int],
    reference_points: dict[str, list[int] | None],
    reference_length: float,
    config: TrackingConfig,
) -> dict[str, list[int] | None]:
    completed = {name: tracked_points.get(name) for name in TRACK_ORDER}
    transform = _fit_partial_affine(current_points, tracked_points)
    if sum(1 for name in TRACK_ORDER if tracked_points.get(name) is not None) >= config.quality_min_points:
        for name in TRACK_ORDER:
            if completed.get(name) is None and current_points.get(name) is not None:
                predicted = _apply_affine(current_points[name], transform)
                if _in_frame(predicted, frame_shape) and _within_motion_limit(
                    name,
                    predicted,
                    reference_points,
                    reference_length,
                    config,
                ):
                    completed[name] = predicted
    anchor, tip = _point_dict_to_arrays(completed)
    if anchor is not None and tip is not None:
        built = _build_points_from_segment(anchor, tip)
        for name in TRACK_ORDER:
            candidate = built[name]
            if _in_frame(candidate, frame_shape) and _within_motion_limit(
                name,
                candidate,
                reference_points,
                reference_length,
                config,
            ):
                completed[name] = candidate
    return completed


def _calc_flow_prediction(
    previous_gray: np.ndarray,
    gray: np.ndarray,
    current_points: dict[str, list[int] | None],
    frame_shape: tuple[int, int, int],
    reference_points: dict[str, list[int] | None],
    reference_length: float,
    config: TrackingConfig,
) -> dict[str, Any]:
    names = [name for name in TRACK_ORDER if current_points.get(name) is not None and _in_frame(current_points.get(name), frame_shape)]
    if not names:
        return {
            "tracked_points": {name: None for name in TRACK_ORDER},
            "predicted_points": {name: None for name in TRACK_ORDER},
            "tracked_point_count": 0,
            "fb_error_median_px": None,
            "fb_error_max_px": None,
        }

    previous_points = np.array([current_points[name] for name in names], dtype=np.float32).reshape(-1, 1, 2)
    lk_params = dict(
        winSize=config.win_size,
        maxLevel=config.max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )
    next_points, status, err = cv2.calcOpticalFlowPyrLK(previous_gray, gray, previous_points, None, **lk_params)
    back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, previous_gray, next_points, None, **lk_params)

    tracked_points: dict[str, list[int] | None] = {name: None for name in TRACK_ORDER}
    fb_errors: list[float] = []
    for idx, name in enumerate(names):
        ok = bool(status[idx][0]) and bool(back_status[idx][0])
        error = float(err[idx][0]) if err is not None else 0.0
        fb_error = float(np.linalg.norm(back_points[idx][0] - previous_points[idx][0]))
        if ok and error <= config.max_error and fb_error <= config.fb_max_error_px:
            point = [int(round(float(next_points[idx][0][0]))), int(round(float(next_points[idx][0][1])))]
            if _in_frame(point, frame_shape) and _within_motion_limit(name, point, reference_points, reference_length, config):
                tracked_points[name] = point
                fb_errors.append(fb_error)

    predicted_points = _complete_predicted_geometry(
        current_points=current_points,
        tracked_points=tracked_points,
        frame_shape=frame_shape,
        reference_points=reference_points,
        reference_length=reference_length,
        config=config,
    )
    return {
        "tracked_points": tracked_points,
        "predicted_points": predicted_points,
        "tracked_point_count": sum(1 for name in TRACK_ORDER if tracked_points.get(name) is not None),
        "fb_error_median_px": float(np.median(fb_errors)) if fb_errors else None,
        "fb_error_max_px": float(np.max(fb_errors)) if fb_errors else None,
    }


def _detect_board_geometry(
    gray: np.ndarray,
    guide_points: dict[str, list[int] | None],
    trusted_points: dict[str, list[int] | None],
    config: TrackingConfig,
) -> dict[str, Any] | None:
    guide_anchor, guide_tip = _point_dict_to_arrays(guide_points)
    trusted_anchor, trusted_tip = _point_dict_to_arrays(trusted_points)
    if guide_anchor is None or guide_tip is None or trusted_anchor is None or trusted_tip is None:
        return None

    x1, y1, x2, y2 = _crop_bounds(guide_points, gray.shape, config.roi_padding_px)
    roi_gray = gray[y1:y2, x1:x2]
    if roi_gray.size == 0:
        return None

    blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    grad_scale = float(np.percentile(grad_mag, 95)) if np.any(grad_mag > 0) else 1.0
    edges = cv2.Canny(blur, 40, 120)

    guide_axis = guide_tip - guide_anchor
    length = float(np.linalg.norm(guide_axis))
    if length <= 1e-6:
        return None
    guide_axis /= length
    guide_normal = np.array([-guide_axis[1], guide_axis[0]], dtype=np.float32)

    edge_ys, edge_xs = np.where(edges > 0)
    if len(edge_xs) == 0:
        return None
    edge_points = np.column_stack((edge_xs + x1, edge_ys + y1)).astype(np.float32)
    projections = (edge_points - guide_anchor) @ guide_axis
    normals = np.abs((edge_points - guide_anchor) @ guide_normal)
    grad_vectors = np.column_stack((grad_x[edge_ys, edge_xs], grad_y[edge_ys, edge_xs])).astype(np.float32)
    grad_norm = np.linalg.norm(grad_vectors, axis=1)
    orientation = np.zeros(len(edge_points), dtype=np.float32)
    valid_grad = grad_norm > 1e-6
    orientation[valid_grad] = np.abs((grad_vectors[valid_grad] / grad_norm[valid_grad, None]) @ guide_normal)

    mask = (
        (projections >= -config.roi_padding_px * 0.5)
        & (projections <= length + config.roi_padding_px * 0.5)
        & (normals <= config.roi_normal_half_width_px)
        & (orientation >= 0.45)
    )
    support_points = edge_points[mask]
    if len(support_points) < config.min_edge_points:
        return None

    fit = cv2.fitLine(support_points.reshape(-1, 1, 2), cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    axis = np.array([float(fit[0]), float(fit[1])], dtype=np.float32)
    axis /= max(float(np.linalg.norm(axis)), 1e-6)
    if float(axis @ guide_axis) < 0.0:
        axis *= -1.0
    origin = np.array([float(fit[2]), float(fit[3])], dtype=np.float32)
    normal = np.array([-axis[1], axis[0]], dtype=np.float32)

    inlier_proj = (support_points - origin) @ axis
    inlier_signed_dist = (support_points - origin) @ normal
    inlier_dist = np.abs(inlier_signed_dist)
    inlier_mask = inlier_dist <= config.roi_normal_half_width_px
    inlier_points = support_points[inlier_mask]
    inlier_proj = inlier_proj[inlier_mask]
    inlier_signed_dist = inlier_signed_dist[inlier_mask]
    if len(inlier_points) < config.min_edge_points:
        return None

    start_proj = float(np.percentile(inlier_proj, 2))
    end_proj = float(np.percentile(inlier_proj, 98))
    if end_proj - start_proj <= 8.0:
        return None

    bin_edges = np.linspace(start_proj, end_proj, 14)
    center_points: list[np.ndarray] = []
    center_residuals: list[float] = []
    widths: list[float] = []
    occupied_bins = 0
    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
        bin_mask = (inlier_proj >= left) & (inlier_proj < right)
        if int(np.count_nonzero(bin_mask)) < 6:
            continue
        occupied_bins += 1
        offsets = inlier_signed_dist[bin_mask]
        lower = float(np.percentile(offsets, 10))
        upper = float(np.percentile(offsets, 90))
        center_offset = (lower + upper) / 2.0
        center_proj = (left + right) / 2.0
        center_point = origin + center_proj * axis + center_offset * normal
        center_points.append(center_point.astype(np.float32))
        center_residuals.append(abs(center_offset))
        widths.append(upper - lower)
    if len(center_points) < 4:
        return None

    center_fit = cv2.fitLine(np.array(center_points, dtype=np.float32).reshape(-1, 1, 2), cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    axis = np.array([float(center_fit[0]), float(center_fit[1])], dtype=np.float32)
    axis /= max(float(np.linalg.norm(axis)), 1e-6)
    if float(axis @ guide_axis) < 0.0:
        axis *= -1.0
    origin = np.array([float(center_fit[2]), float(center_fit[3])], dtype=np.float32)
    normal = np.array([-axis[1], axis[0]], dtype=np.float32)
    center_points_arr = np.array(center_points, dtype=np.float32)
    center_proj = (center_points_arr - origin) @ axis
    center_dist = np.abs((center_points_arr - origin) @ normal)

    trusted_length = float(np.linalg.norm(trusted_tip - trusted_anchor))
    center_center = np.mean(center_points_arr, axis=0)
    half_length = trusted_length / 2.0
    anchor = center_center - half_length * axis
    tip = center_center + half_length * axis
    if float((tip - anchor) @ (trusted_tip - trusted_anchor)) < 0.0:
        anchor, tip = tip, anchor

    fitted_length = float(np.linalg.norm(tip - anchor))
    stations = _build_points_from_segment(anchor, tip)
    station_arrays = {name: np.array(point, dtype=np.float32) for name, point in stations.items()}

    support_coverage = occupied_bins / max(len(bin_edges) - 1, 1)
    centerline_curvature = float(np.max(center_dist)) if len(center_dist) else 0.0

    station_support_scores = {
        name: _sample_line_support(
            grad_mag=grad_mag,
            station=station_arrays[name],
            axis=axis,
            roi_offset=(x1, y1),
            radius=max(4, int(round(config.roi_normal_half_width_px / 2.0))),
            scale=grad_scale,
        )
        for name in TRACK_ORDER
    }
    mean_station_support = float(np.mean(list(station_support_scores.values())))
    min_station_support = float(np.min(list(station_support_scores.values())))

    return {
        "points": stations,
        "anchor": [int(round(float(anchor[0]))), int(round(float(anchor[1])))],
        "tip": [int(round(float(tip[0]))), int(round(float(tip[1])))],
        "quality": {
            "roi_bounds": [x1, y1, x2, y2],
            "inlier_edge_count": int(len(inlier_points)),
            "support_coverage": float(support_coverage),
            "mean_line_distance_px": float(np.mean(center_dist)),
            "centerline_curvature_px": float(centerline_curvature),
            "board_length_px": fitted_length,
            "board_angle_deg": _line_angle_deg(anchor, tip),
            "board_thickness_px": float(np.median(widths)) if widths else None,
            "station_support_scores": station_support_scores,
            "mean_station_support_score": mean_station_support,
            "min_station_support_score": min_station_support,
        },
    }


def _evaluate_frame_quality(
    candidate: dict[str, Any] | None,
    last_trusted_points: dict[str, list[int] | None],
    flow_metrics: dict[str, Any],
    reference_points: dict[str, list[int] | None],
    reference_length: float,
    config: TrackingConfig,
) -> dict[str, Any]:
    if candidate is None:
        return {
            "passed": False,
            "reasons": ["no board-supported geometry fit in local ROI"],
            "confidence": 0.0,
        }

    quality = dict(candidate["quality"])
    trusted_anchor, trusted_tip = _point_dict_to_arrays(last_trusted_points)
    candidate_anchor = np.array(candidate["points"]["anchor"], dtype=np.float32)
    candidate_tip = np.array(candidate["points"]["tip"], dtype=np.float32)
    trusted_length = _reference_segment_length(last_trusted_points)
    candidate_length = float(np.linalg.norm(candidate_tip - candidate_anchor))
    trusted_angle = _line_angle_deg(trusted_anchor, trusted_tip) if trusted_anchor is not None and trusted_tip is not None else quality["board_angle_deg"]
    angle_delta = _angle_delta_deg(quality["board_angle_deg"], trusted_angle)
    length_ratio_delta = abs(candidate_length - trusted_length) / max(trusted_length, 1e-6)
    quality["length_ratio_delta"] = float(length_ratio_delta)
    quality["angle_delta_deg"] = float(angle_delta)
    quality["fb_error_median_px"] = flow_metrics["fb_error_median_px"]
    quality["fb_error_max_px"] = flow_metrics["fb_error_max_px"]
    quality["flow_tracked_point_count"] = flow_metrics["tracked_point_count"]

    reasons: list[str] = []
    if quality["support_coverage"] < config.min_support_coverage:
        reasons.append("edge support coverage is too sparse along the board")
    if quality["mean_line_distance_px"] > config.max_mean_line_distance_px:
        reasons.append("edge support is too far from a single board line")
    if quality["centerline_curvature_px"] > config.max_centerline_curvature_px:
        reasons.append("support deviates too much from a straight board centerline")
    if quality["min_station_support_score"] < config.min_station_support_score:
        reasons.append("at least one station lacks local image support")
    if quality["mean_station_support_score"] < config.min_mean_station_support_score:
        reasons.append("overall station image support is too weak")
    if quality["inlier_edge_count"] < config.min_edge_points:
        reasons.append("not enough edge inliers support the board fit")
    if angle_delta > config.max_angle_delta_deg:
        reasons.append("board angle continuity broke relative to the last trusted frame")
    if length_ratio_delta > config.max_length_ratio_delta:
        reasons.append("board length changed implausibly relative to the last trusted frame")
    reference_anchor, reference_tip = _point_dict_to_arrays(reference_points)
    if reference_anchor is not None and reference_tip is not None:
        reference_centroid = (reference_anchor + reference_tip) / 2.0
        candidate_centroid = (candidate_anchor + candidate_tip) / 2.0
        centroid_drift_norm = float(np.linalg.norm(candidate_centroid - reference_centroid) / max(reference_length, 1e-6))
        quality["centroid_drift_norm"] = centroid_drift_norm
        if centroid_drift_norm > config.max_centroid_drift_norm:
            reasons.append("board centroid drifted beyond the allowed seed geometry bound")

    quality["passed"] = not reasons
    quality["reasons"] = reasons

    support_term = min(1.0, quality["mean_station_support_score"])
    coverage_term = min(1.0, quality["support_coverage"])
    inlier_term = min(1.0, quality["inlier_edge_count"] / max(config.min_edge_points * 2, 1))
    line_term = max(0.0, 1.0 - quality["mean_line_distance_px"] / max(config.max_mean_line_distance_px, 1e-6))
    curvature_term = max(0.0, 1.0 - quality["centerline_curvature_px"] / max(config.max_centerline_curvature_px, 1e-6))
    angle_term = max(0.0, 1.0 - angle_delta / max(config.max_angle_delta_deg, 1e-6))
    length_term = max(0.0, 1.0 - length_ratio_delta / max(config.max_length_ratio_delta, 1e-6))
    quality["confidence"] = float(
        np.clip(
            0.24 * support_term
            + 0.16 * coverage_term
            + 0.14 * inlier_term
            + 0.16 * line_term
            + 0.12 * curvature_term
            + 0.10 * angle_term
            + 0.08 * length_term,
            0.0,
            1.0,
        )
    )
    return quality


def _refresh_observation_state(
    tracking_points: dict[str, list[int] | None],
    frame_state: str,
    frame_shape: tuple[int, int, int],
    border_margin: int = 2,
) -> dict[str, Any]:
    def describe(point: list[int] | None) -> dict[str, Any]:
        if point is None:
            return {
                "point": None,
                "in_frame": False,
                "near_image_border": False,
                "observability": "missing",
            }
        h, w = frame_shape[:2]
        x, y = point
        in_frame = 0 <= x < w and 0 <= y < h
        near_border = in_frame and (x <= border_margin or x >= w - 1 - border_margin or y <= border_margin or y >= h - 1 - border_margin)
        observability = frame_state
        if not in_frame:
            observability = "missing"
        elif near_border:
            observability = "truncated_or_edge"
        return {
            "point": point,
            "in_frame": in_frame,
            "near_image_border": near_border,
            "observability": observability,
        }

    stations: list[dict[str, Any]] = []
    partial = False
    for name in TRACK_ORDER:
        point = tracking_points.get(name)
        state = describe(point)
        state["name"] = name
        state["s"] = STATION_FRACTIONS[name]
        stations.append(state)
        partial = partial or state["near_image_border"] or not state["in_frame"]
    anchor_state = next(item for item in stations if item["name"] == "anchor")
    tip_state = next(item for item in stations if item["name"] == "tip")
    return {
        "anchor": {k: anchor_state[k] for k in ("point", "in_frame", "near_image_border", "observability")},
        "tip": {k: tip_state[k] for k in ("point", "in_frame", "near_image_border", "observability")},
        "stations": stations,
        "partial_board_observation": partial,
    }


def _build_tracking_record(
    frame_index: int,
    timestamp_sec: float,
    tracking_points: dict[str, list[int] | None],
    point_statuses: dict[str, str],
    frame_shape: tuple[int, int, int],
    tracking_state: str,
    frame_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    anchor = tracking_points.get("anchor")
    tip = tracking_points.get("tip")
    axis_vector = None
    if anchor is not None and tip is not None:
        axis_vector = [tip[0] - anchor[0], tip[1] - anchor[1]]
    visible_records = []
    for name in TRACK_ORDER:
        visible_records.append(
            {
                "name": name,
                "s": STATION_FRACTIONS[name],
                "point": tracking_points.get(name),
                "status": point_statuses.get(name, tracking_state),
            }
        )
    return {
        "frame_index": frame_index,
        "timestamp_sec": timestamp_sec,
        "visible_segment_state": {
            "segment_start_point": anchor,
            "segment_end_point": tip,
            "axis_vector": axis_vector,
            "tracking_points": tracking_points,
            "tracking_point_records": visible_records,
            "tracking_state": tracking_state,
            "frame_quality": frame_quality or {},
            "observation_state": _refresh_observation_state(tracking_points, tracking_state, frame_shape),
        },
    }


def _compute_resize_scale(frame_shape: tuple[int, int, int], max_side: int | None) -> float:
    if max_side is None:
        return 1.0
    height, width = frame_shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return 1.0
    return float(max_side) / float(longest)


def _resize_frame(frame_bgr: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-9:
        return frame_bgr
    return cv2.resize(
        frame_bgr,
        (int(round(frame_bgr.shape[1] * scale)), int(round(frame_bgr.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _scale_tracking_points(points: dict[str, list[int] | None], scale: float) -> dict[str, list[int] | None]:
    if abs(scale - 1.0) < 1e-9:
        return {name: (list(point) if point is not None else None) for name, point in points.items()}
    return {
        name: ([int(round(point[0] * scale)), int(round(point[1] * scale))] if point is not None else None)
        for name, point in points.items()
    }


def _unscale_record(record: dict[str, Any], scale: float) -> dict[str, Any]:
    if abs(scale - 1.0) < 1e-9:
        return record

    def unscale_point(point: list[int] | None) -> list[int] | None:
        if point is None:
            return None
        return [int(round(point[0] / scale)), int(round(point[1] / scale))]

    visible = dict(record["visible_segment_state"])
    visible["segment_start_point"] = unscale_point(visible.get("segment_start_point"))
    visible["segment_end_point"] = unscale_point(visible.get("segment_end_point"))
    visible["tracking_points"] = {name: unscale_point(point) for name, point in visible.get("tracking_points", {}).items()}
    visible["tracking_point_records"] = [
        {**item, "point": unscale_point(item.get("point"))}
        for item in visible.get("tracking_point_records", [])
    ]
    frame_quality = dict(visible.get("frame_quality", {}))
    for key in ("roi_bounds",):
        if key in frame_quality:
            x1, y1, x2, y2 = frame_quality[key]
            frame_quality[key] = [
                int(round(x1 / scale)),
                int(round(y1 / scale)),
                int(round(x2 / scale)),
                int(round(y2 / scale)),
            ]
    for key in ("board_length_px", "board_thickness_px", "mean_line_distance_px", "centerline_curvature_px", "fb_error_median_px", "fb_error_max_px"):
        if frame_quality.get(key) is not None:
            frame_quality[key] = float(frame_quality[key] / scale)
    visible["frame_quality"] = frame_quality
    obs = dict(visible.get("observation_state", {}))
    for key in ("anchor", "tip"):
        if obs.get(key):
            obs[key] = {**obs[key], "point": unscale_point(obs[key].get("point"))}
    if obs.get("stations") is not None:
        obs["stations"] = [{**item, "point": unscale_point(item.get("point"))} for item in obs["stations"]]
    visible["observation_state"] = obs
    if visible.get("axis_vector") is not None:
        visible["axis_vector"] = [int(round(visible["axis_vector"][0] / scale)), int(round(visible["axis_vector"][1] / scale))]
    return {**record, "visible_segment_state": visible}


def _draw_tracking_overlay(frame_bgr: np.ndarray, record: dict[str, Any]) -> np.ndarray:
    overlay = frame_bgr.copy()
    visible = record["visible_segment_state"]
    tracking_points = visible["tracking_points"]
    statuses = {item["name"]: item["status"] for item in visible["tracking_point_records"]}
    state = visible.get("tracking_state", "tracked")
    frame_quality = visible.get("frame_quality", {})

    line_colors = {
        "seed": (255, 200, 0),
        "tracked": (0, 255, 0),
        "recovered": (255, 200, 0),
        "predicted": (0, 165, 255),
        "invalid": (0, 0, 255),
    }
    point_colors = {
        "seed": (255, 200, 0),
        "tracked": (0, 255, 0),
        "recovered": (255, 200, 0),
        "predicted": (0, 165, 255),
        "invalid": (0, 0, 255),
        "lost": (0, 0, 255),
    }

    ordered_points: list[tuple[int, int]] = []
    for name in TRACK_ORDER:
        point = tracking_points.get(name)
        if point is None:
            continue
        ordered_points.append((point[0], point[1]))
        status = statuses.get(name, state)
        cv2.circle(overlay, (point[0], point[1]), 6, point_colors.get(status, (255, 255, 0)), thickness=-1)
        cv2.putText(
            overlay,
            name,
            (point[0] + 6, point[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            point_colors.get(status, (255, 255, 0)),
            1,
            cv2.LINE_AA,
        )
    if len(ordered_points) >= 2:
        cv2.polylines(overlay, [np.array(ordered_points, dtype=np.int32).reshape(-1, 1, 2)], False, line_colors.get(state, (255, 255, 0)), 2)
    quality_text = f"{state} conf={frame_quality.get('confidence', 0.0):.2f}"
    cv2.putText(overlay, quality_text, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, line_colors.get(state, (255, 255, 0)), 2, cv2.LINE_AA)
    if frame_quality.get("passed") is False and frame_quality.get("reasons"):
        cv2.putText(
            overlay,
            frame_quality["reasons"][0][:80],
            (18, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return overlay


def track_points_in_frames(
    frames_bgr: list[np.ndarray],
    initial_tracking_points: dict[str, list[int] | None],
    fps: float,
    config: TrackingConfig | None = None,
) -> list[dict[str, Any]]:
    if not frames_bgr:
        return []
    config = config or TrackingConfig()
    reference_points = {
        name: (list(initial_tracking_points.get(name)) if initial_tracking_points.get(name) is not None else None)
        for name in TRACK_ORDER
    }
    reference_length = _reference_segment_length(reference_points)
    previous_gray = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2GRAY)
    current_points = {name: initial_tracking_points.get(name) for name in TRACK_ORDER}
    last_trusted_points = {name: initial_tracking_points.get(name) for name in TRACK_ORDER}
    tracker_points = {name: initial_tracking_points.get(name) for name in TRACK_ORDER}
    previous_state = "seed"
    results = [
        _build_tracking_record(
            frame_index=0,
            timestamp_sec=0.0,
            tracking_points=current_points,
            point_statuses={name: "seed" for name in TRACK_ORDER if current_points.get(name) is not None},
            frame_shape=frames_bgr[0].shape,
            tracking_state="seed",
            frame_quality={"passed": True, "confidence": 1.0, "reasons": []},
        )
    ]

    for frame_index in range(1, len(frames_bgr)):
        frame_bgr = frames_bgr[frame_index]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        guide_points = tracker_points if any(tracker_points.get(name) is not None for name in TRACK_ORDER) else last_trusted_points
        flow = _calc_flow_prediction(
            previous_gray=previous_gray,
            gray=gray,
            current_points=guide_points,
            frame_shape=frame_bgr.shape,
            reference_points=reference_points,
            reference_length=reference_length,
            config=config,
        )
        predicted_points = flow["predicted_points"]
        candidate = _detect_board_geometry(gray=gray, guide_points=last_trusted_points, trusted_points=last_trusted_points, config=config)
        frame_quality = _evaluate_frame_quality(
            candidate=candidate,
            last_trusted_points=last_trusted_points,
            flow_metrics=flow,
            reference_points=reference_points,
            reference_length=reference_length,
            config=config,
        )

        if candidate is not None and frame_quality["passed"]:
            current_points = {name: candidate["points"][name] for name in TRACK_ORDER}
            tracker_points = current_points
            last_trusted_points = current_points
            current_state = "tracked" if flow["tracked_point_count"] >= config.quality_min_points else "recovered"
            point_statuses = {name: current_state for name in TRACK_ORDER}
        elif any(predicted_points.get(name) is not None for name in TRACK_ORDER):
            current_points = predicted_points
            tracker_points = predicted_points
            current_state = "predicted"
            point_statuses = {name: ("predicted" if predicted_points.get(name) is not None else "lost") for name in TRACK_ORDER}
            frame_quality = {
                **frame_quality,
                "passed": False,
                "reasons": frame_quality.get("reasons", []) or ["falling back to motion-only prediction without board support"],
                "confidence": min(0.4, float(frame_quality.get("confidence", 0.0))),
            }
        else:
            current_points = {name: None for name in TRACK_ORDER}
            tracker_points = last_trusted_points if previous_state in TRUSTED_STATES else guide_points
            current_state = "invalid"
            point_statuses = {name: "invalid" for name in TRACK_ORDER}
            frame_quality = {
                **frame_quality,
                "passed": False,
                "reasons": frame_quality.get("reasons", []) or ["board geometry and prediction both failed"],
                "confidence": 0.0,
            }

        results.append(
            _build_tracking_record(
                frame_index=frame_index,
                timestamp_sec=float(frame_index / fps) if fps > 0 else 0.0,
                tracking_points=current_points,
                point_statuses=point_statuses,
                frame_shape=frame_bgr.shape,
                tracking_state=current_state,
                frame_quality=frame_quality,
            )
        )
        previous_gray = gray
        previous_state = current_state
    return results


def track_video(
    video_path: str | Path,
    tracking_points_payload: dict[str, Any],
    output_dir: str | Path,
    start_frame: int = 0,
    end_frame: int | None = None,
    config: TrackingConfig | None = None,
) -> dict[str, Any]:
    config = config or TrackingConfig()
    output_path = ensure_dir(output_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame = max(0, start_frame)
    end_frame = total_frames - 1 if end_frame is None else min(end_frame, total_frames - 1)
    if start_frame > end_frame:
        raise ValueError("start_frame must be <= end_frame")

    frames_bgr: list[np.ndarray] = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(start_frame, end_frame + 1):
        ok, frame = cap.read()
        if not ok:
            break
        frames_bgr.append(frame)
    cap.release()
    if not frames_bgr:
        raise RuntimeError("No frames were read from the requested video range.")

    scale = _compute_resize_scale(frames_bgr[0].shape, config.max_side)
    working_frames = [_resize_frame(frame, scale) for frame in frames_bgr]
    original_tracking_points = tracking_points_payload.get("tracking_points", {})
    tracking_points = _scale_tracking_points(original_tracking_points, scale)
    results = track_points_in_frames(working_frames, tracking_points, fps=fps, config=config)
    results = [_unscale_record(record, scale) for record in results]

    jsonl_path = output_path / "tracked_points.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in results:
            handle.write(json.dumps(record) + "\n")

    csv_path = output_path / "tracked_points.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "timestamp_sec", "tracking_state", "name", "s", "x", "y", "status"])
        for record in results:
            for item in record["visible_segment_state"]["tracking_point_records"]:
                point = item["point"] or [None, None]
                writer.writerow(
                    [
                        record["frame_index"],
                        record["timestamp_sec"],
                        record["visible_segment_state"].get("tracking_state", "tracked"),
                        item["name"],
                        item["s"],
                        point[0],
                        point[1],
                        item["status"],
                    ]
                )

    quality_rows = []
    for record in results:
        visible = record["visible_segment_state"]
        quality = visible.get("frame_quality", {})
        quality_rows.append(
            {
                "frame_index": int(record["frame_index"]),
                "timestamp_sec": float(record["timestamp_sec"]),
                "tracking_state": visible.get("tracking_state", "tracked"),
                "quality_passed": bool(quality.get("passed", False)),
                "confidence": float(quality.get("confidence", 0.0)),
                "board_length_px": quality.get("board_length_px"),
                "board_thickness_px": quality.get("board_thickness_px"),
                "board_angle_deg": quality.get("board_angle_deg"),
                "angle_delta_deg": quality.get("angle_delta_deg"),
                "length_ratio_delta": quality.get("length_ratio_delta"),
                "centroid_drift_norm": quality.get("centroid_drift_norm"),
                "centerline_curvature_px": quality.get("centerline_curvature_px"),
                "mean_line_distance_px": quality.get("mean_line_distance_px"),
                "inlier_edge_count": quality.get("inlier_edge_count"),
                "support_coverage": quality.get("support_coverage"),
                "min_station_support_score": quality.get("min_station_support_score"),
                "mean_station_support_score": quality.get("mean_station_support_score"),
                "flow_tracked_point_count": quality.get("flow_tracked_point_count"),
                "fb_error_median_px": quality.get("fb_error_median_px"),
                "fb_error_max_px": quality.get("fb_error_max_px"),
                "reasons": "; ".join(quality.get("reasons", [])),
            }
        )

    write_json(output_path / "tracking_quality.json", quality_rows)
    with (output_path / "tracking_quality.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(quality_rows[0].keys()))
        writer.writeheader()
        writer.writerows(quality_rows)

    if config.save_overlay_video:
        first_frame = frames_bgr[0]
        overlay_path = output_path / "tracking_overlay.mp4"
        writer = cv2.VideoWriter(
            str(overlay_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps if fps > 0 else 30.0,
            (first_frame.shape[1], first_frame.shape[0]),
        )
        for frame_bgr, record in zip(frames_bgr, results):
            writer.write(_draw_tracking_overlay(frame_bgr, record))
        writer.release()

    state_counts: dict[str, int] = {}
    for record in results:
        state = record["visible_segment_state"].get("tracking_state", "tracked")
        state_counts[state] = state_counts.get(state, 0) + 1

    summary = {
        "video_path": str(video_path),
        "start_frame": start_frame,
        "end_frame": start_frame + len(frames_bgr) - 1,
        "fps": fps,
        "frame_count": len(results),
        "tracking_resize_scale": scale,
        "initial_tracking_points": original_tracking_points,
        "working_tracking_points": tracking_points,
        "tracking_state_counts": state_counts,
        "quality_artifacts": {
            "json": "tracking_quality.json",
            "csv": "tracking_quality.csv",
        },
    }
    write_json(output_path / "tracking_summary.json", summary)
    return summary
