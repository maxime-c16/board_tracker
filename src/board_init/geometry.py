from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


TRACKING_STATIONS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)


def _ordered_skeleton_points(skeleton: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere(skeleton)
    if len(coords) == 0:
        return []
    if len(coords) == 1:
        y, x = coords[0]
        return [(int(x), int(y))]

    coord_set = {tuple(coord) for coord in coords}

    def neighbors(node: tuple[int, int]) -> list[tuple[int, int]]:
        y, x = node
        result: list[tuple[int, int]] = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                candidate = (y + dy, x + dx)
                if candidate in coord_set:
                    result.append(candidate)
        return result

    endpoints = [node for node in coord_set if len(neighbors(node)) <= 1]
    start = endpoints[0] if endpoints else tuple(coords[0])
    ordered: list[tuple[int, int]] = []
    visited: set[tuple[int, int]] = set()
    current = start
    previous: tuple[int, int] | None = None
    while current not in visited:
        visited.add(current)
        ordered.append((int(current[1]), int(current[0])))
        nxt = [node for node in neighbors(current) if node != previous and node not in visited]
        if not nxt:
            break
        if len(nxt) == 1:
            previous, current = current, nxt[0]
            continue
        current_xy = np.array([current[1], current[0]], dtype=np.float32)
        previous_xy = current_xy if previous is None else np.array([previous[1], previous[0]], dtype=np.float32)
        direction = current_xy - previous_xy
        best = max(
            nxt,
            key=lambda node: float(
                np.dot(
                    direction,
                    np.array([node[1], node[0]], dtype=np.float32) - current_xy,
                )
            ),
        )
        previous, current = current, best
    return ordered


def _skeleton_graph_metrics(skeleton: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(skeleton)
    if len(coords) == 0:
        return 0, 0

    coord_set = {tuple(coord) for coord in coords}
    endpoint_count = 0
    branch_count = 0
    for y, x in coord_set:
        degree = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                if (y + dy, x + dx) in coord_set:
                    degree += 1
        if degree <= 1:
            endpoint_count += 1
        elif degree >= 3:
            branch_count += 1
    return endpoint_count, branch_count


def _polyline_length(points: list[tuple[int, int]]) -> float:
    length = 0.0
    for idx in range(1, len(points)):
        p0 = np.array(points[idx - 1], dtype=np.float32)
        p1 = np.array(points[idx], dtype=np.float32)
        length += float(np.linalg.norm(p1 - p0))
    return length


def _point_visibility(point: tuple[int, int] | None, image_shape: tuple[int, int], border_margin: int = 2) -> dict[str, Any]:
    height, width = image_shape
    if point is None:
        return {
            "point": None,
            "in_frame": False,
            "near_image_border": False,
            "observability": "missing",
        }
    x, y = point
    in_frame = 0 <= x < width and 0 <= y < height
    near_border = in_frame and (
        x <= border_margin or x >= width - 1 - border_margin or y <= border_margin or y >= height - 1 - border_margin
    )
    observability = "truncated_or_edge" if near_border else "visible"
    return {
        "point": [int(x), int(y)],
        "in_frame": in_frame,
        "near_image_border": near_border,
        "observability": observability if in_frame else "missing",
    }


def _select_anchor_tip(
    centerline_points: list[tuple[int, int]],
    mask_points: np.ndarray,
    primary_axis: np.ndarray,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    if len(centerline_points) >= 2:
        line_arr = np.array(centerline_points, dtype=np.float32)
        line_center = np.mean(line_arr, axis=0)
        proj = (line_arr - line_center) @ primary_axis
        anchor = tuple(int(v) for v in line_arr[int(np.argmin(proj))])
        tip = tuple(int(v) for v in line_arr[int(np.argmax(proj))])
        return anchor, tip

    if len(mask_points) == 0:
        return None, None
    mean = np.mean(mask_points, axis=0)
    proj = (mask_points - mean) @ primary_axis
    anchor_idx = int(np.argmin(proj))
    tip_idx = int(np.argmax(proj))
    anchor = (int(mask_points[anchor_idx][0]), int(mask_points[anchor_idx][1]))
    tip = (int(mask_points[tip_idx][0]), int(mask_points[tip_idx][1]))
    return anchor, tip


def _sample_centerline_stations(
    centerline_points: list[tuple[int, int]],
    stations: tuple[float, ...] = TRACKING_STATIONS,
) -> tuple[dict[str, list[int] | None], list[dict[str, Any]]]:
    if not centerline_points:
        station_payload = {
            "anchor": None,
            "s25": None,
            "s50": None,
            "s75": None,
            "s90": None,
            "tip": None,
        }
        return station_payload, []

    cumulative = [0.0]
    for idx in range(1, len(centerline_points)):
        p0 = np.array(centerline_points[idx - 1], dtype=np.float32)
        p1 = np.array(centerline_points[idx], dtype=np.float32)
        cumulative.append(cumulative[-1] + float(np.linalg.norm(p1 - p0)))
    total_length = cumulative[-1]

    if total_length <= 1e-6:
        single = [int(centerline_points[0][0]), int(centerline_points[0][1])]
        station_payload = {
            "anchor": single,
            "s25": single,
            "s50": single,
            "s75": single,
            "s90": single,
            "tip": single,
        }
        station_records = [{"name": "anchor", "s": 0.0, "point": single}, {"name": "tip", "s": 1.0, "point": single}]
        return station_payload, station_records

    station_names = {0.0: "anchor", 0.25: "s25", 0.5: "s50", 0.75: "s75", 0.9: "s90", 1.0: "tip"}
    payload: dict[str, list[int] | None] = {}
    records: list[dict[str, Any]] = []
    for station in stations:
        target = station * total_length
        idx = int(np.searchsorted(cumulative, target, side="left"))
        idx = min(max(idx, 1), len(centerline_points) - 1)
        prev_len = cumulative[idx - 1]
        next_len = cumulative[idx]
        p0 = np.array(centerline_points[idx - 1], dtype=np.float32)
        p1 = np.array(centerline_points[idx], dtype=np.float32)
        alpha = 0.0 if next_len <= prev_len else (target - prev_len) / (next_len - prev_len)
        point = p0 + alpha * (p1 - p0)
        point_list = [int(round(point[0])), int(round(point[1]))]
        name = station_names.get(station, f"s{int(round(station * 100))}")
        payload[name] = point_list
        records.append({"name": name, "s": station, "point": point_list})
    return payload, records


def extract_geometry(mask: np.ndarray) -> dict[str, Any]:
    binary = mask.astype(bool)
    area = int(binary.sum())
    image_shape = mask.shape[:2]
    if area == 0:
        return {
            "valid": False,
            "mask_area": 0,
            "bbox": None,
            "rotated_bbox": None,
            "principal_axis_angle_deg": None,
            "centerline_points": [],
            "anchor_endpoint": None,
            "tip_endpoint": None,
            "length_estimate_px": 0.0,
            "average_thickness_px": 0.0,
            "width_std_px": 0.0,
            "elongation": 0.0,
            "centerline_quality": 0.0,
            "skeleton_endpoint_count": 0,
            "skeleton_branch_point_count": 0,
            "tracking_points": {},
            "tracking_point_records": [],
            "observation_state": {
                "anchor": _point_visibility(None, image_shape),
                "tip": _point_visibility(None, image_shape),
                "stations": [],
                "partial_board_observation": False,
            },
        }

    ys, xs = np.where(binary)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    points = np.column_stack((xs, ys)).astype(np.float32)
    mean = np.mean(points, axis=0)
    centered = points - mean
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    primary = eigenvectors[:, 0]
    angle = math.degrees(math.atan2(float(primary[1]), float(primary[0])))
    elongation = float(eigenvalues[0] / max(eigenvalues[1], 1e-6))

    contour_points = np.column_stack((xs, ys)).astype(np.float32).reshape(-1, 1, 2)
    rect = cv2.minAreaRect(contour_points)
    box = cv2.boxPoints(rect)
    box = np.round(box).astype(int).tolist()

    distance_map = ndimage.distance_transform_edt(binary)
    skeleton = skeletonize(binary)
    centerline_points = _ordered_skeleton_points(skeleton)
    endpoint_count, branch_count = _skeleton_graph_metrics(skeleton)
    length = _polyline_length(centerline_points)

    thickness_samples = [
        float(distance_map[y, x] * 2.0)
        for x, y in centerline_points
        if 0 <= y < distance_map.shape[0] and 0 <= x < distance_map.shape[1]
    ]
    average_thickness = float(np.mean(thickness_samples)) if thickness_samples else 0.0
    width_std = float(np.std(thickness_samples)) if thickness_samples else 0.0

    anchor_endpoint, tip_endpoint = _select_anchor_tip(centerline_points, points, primary)
    centerline_length_to_thickness_ratio = float(length / max(average_thickness, 1.0))
    tracking_points, tracking_point_records = _sample_centerline_stations(centerline_points)

    anchor_visibility = _point_visibility(anchor_endpoint, image_shape)
    tip_visibility = _point_visibility(tip_endpoint, image_shape)
    station_visibility = [
        {
            "name": record["name"],
            "s": record["s"],
            **_point_visibility(tuple(record["point"]), image_shape),
        }
        for record in tracking_point_records
    ]
    partial_board_observation = bool(anchor_visibility["near_image_border"] or tip_visibility["near_image_border"])

    axis_vector = None
    if anchor_endpoint is not None and tip_endpoint is not None:
        axis_vector = [int(tip_endpoint[0] - anchor_endpoint[0]), int(tip_endpoint[1] - anchor_endpoint[1])]

    return {
        "valid": True,
        "mask_area": area,
        "bbox": [x1, y1, x2, y2],
        "rotated_bbox": box,
        "principal_axis_angle_deg": angle,
        "centerline_points": [list(point) for point in centerline_points],
        "anchor_endpoint": list(anchor_endpoint) if anchor_endpoint is not None else None,
        "tip_endpoint": list(tip_endpoint) if tip_endpoint is not None else None,
        "axis_vector": axis_vector,
        "length_estimate_px": length,
        "average_thickness_px": average_thickness,
        "width_std_px": width_std,
        "elongation": elongation,
        "centerline_length_to_thickness_ratio": centerline_length_to_thickness_ratio,
        "skeleton_endpoint_count": endpoint_count,
        "skeleton_branch_point_count": branch_count,
        "tracking_points": tracking_points,
        "tracking_point_records": tracking_point_records,
        "observation_state": {
            "anchor": anchor_visibility,
            "tip": tip_visibility,
            "stations": station_visibility,
            "partial_board_observation": partial_board_observation,
        },
    }
