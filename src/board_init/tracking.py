from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .io import ensure_dir, write_json


TRACK_ORDER = ("anchor", "s25", "s50", "s75", "s90", "tip")


@dataclass
class TrackingConfig:
    win_size: tuple[int, int] = (31, 31)
    max_level: int = 3
    max_error: float = 25.0
    quality_min_points: int = 3
    save_overlay_video: bool = True
    max_side: int | None = None


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


def _refresh_observation_state(
    tracking_points: dict[str, list[int] | None],
    statuses: dict[str, str],
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
        observability = statuses.get("current", "tracked")
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
        if name not in ("anchor", "tip"):
            state["s"] = {"s25": 0.25, "s50": 0.5, "s75": 0.75, "s90": 0.9}[name]
        else:
            state["s"] = 0.0 if name == "anchor" else 1.0
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
    statuses: dict[str, str],
    frame_shape: tuple[int, int, int],
) -> dict[str, Any]:
    anchor = tracking_points.get("anchor")
    tip = tracking_points.get("tip")
    axis_vector = None
    if anchor is not None and tip is not None:
        axis_vector = [tip[0] - anchor[0], tip[1] - anchor[1]]
    visible_records = []
    station_s = {"anchor": 0.0, "s25": 0.25, "s50": 0.5, "s75": 0.75, "s90": 0.9, "tip": 1.0}
    for name in TRACK_ORDER:
        visible_records.append(
            {
                "name": name,
                "s": station_s[name],
                "point": tracking_points.get(name),
                "status": statuses.get(name, "missing"),
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
            "observation_state": _refresh_observation_state(tracking_points, {"current": "tracked"}, frame_shape),
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
    return cv2.resize(frame_bgr, (int(round(frame_bgr.shape[1] * scale)), int(round(frame_bgr.shape[0] * scale))), interpolation=cv2.INTER_AREA)


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


def _draw_tracking_overlay(frame_bgr: np.ndarray, tracking_points: dict[str, list[int] | None], statuses: dict[str, str]) -> np.ndarray:
    overlay = frame_bgr.copy()
    ordered_points: list[tuple[int, int]] = []
    colors = {"tracked": (0, 255, 0), "predicted": (0, 200, 255), "lost": (0, 0, 255), "seed": (255, 200, 0)}
    for name in TRACK_ORDER:
        point = tracking_points.get(name)
        if point is None:
            continue
        ordered_points.append((point[0], point[1]))
        status = statuses.get(name, "tracked")
        cv2.circle(overlay, (point[0], point[1]), 6, colors.get(status, (255, 255, 0)), thickness=-1)
        cv2.putText(overlay, name, (point[0] + 6, point[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors.get(status, (255, 255, 0)), 1, cv2.LINE_AA)
    if len(ordered_points) >= 2:
        cv2.polylines(overlay, [np.array(ordered_points, dtype=np.int32).reshape(-1, 1, 2)], False, (255, 255, 0), 2)
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
    previous_gray = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2GRAY)
    current_points = {name: initial_tracking_points.get(name) for name in TRACK_ORDER}
    results = [
        _build_tracking_record(
            frame_index=0,
            timestamp_sec=0.0,
            tracking_points=current_points,
            statuses={name: "seed" for name in TRACK_ORDER if current_points.get(name) is not None},
            frame_shape=frames_bgr[0].shape,
        )
    ]

    lk_params = dict(winSize=config.win_size, maxLevel=config.max_level, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    for frame_index in range(1, len(frames_bgr)):
        frame_bgr = frames_bgr[frame_index]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        names = [name for name in TRACK_ORDER if current_points.get(name) is not None and _in_frame(current_points.get(name), frame_bgr.shape)]
        previous_points = np.array([current_points[name] for name in names], dtype=np.float32).reshape(-1, 1, 2) if names else np.empty((0, 1, 2), dtype=np.float32)
        tracked_points: dict[str, list[int] | None] = {name: None for name in TRACK_ORDER}
        statuses: dict[str, str] = {}

        if len(previous_points):
            next_points, status, err = cv2.calcOpticalFlowPyrLK(previous_gray, gray, previous_points, None, **lk_params)
            for idx, name in enumerate(names):
                ok = bool(status[idx][0])
                error = float(err[idx][0]) if err is not None else 0.0
                if ok and error <= config.max_error:
                    point = [int(round(float(next_points[idx][0][0]))), int(round(float(next_points[idx][0][1])))]
                    if _in_frame(point, frame_bgr.shape):
                        tracked_points[name] = point
                        statuses[name] = "tracked"

        transform = _fit_partial_affine(current_points, tracked_points)
        if sum(1 for name in TRACK_ORDER if tracked_points.get(name) is not None) >= config.quality_min_points:
            for name in TRACK_ORDER:
                if tracked_points.get(name) is None and current_points.get(name) is not None:
                    predicted = _apply_affine(current_points[name], transform)
                    if _in_frame(predicted, frame_bgr.shape):
                        tracked_points[name] = predicted
                        statuses[name] = "predicted"

        for name in TRACK_ORDER:
            if tracked_points.get(name) is None:
                statuses[name] = "lost"

        current_points = tracked_points
        results.append(
            _build_tracking_record(
                frame_index=frame_index,
                timestamp_sec=float(frame_index / fps) if fps > 0 else 0.0,
                tracking_points=current_points,
                statuses=statuses,
                frame_shape=frame_bgr.shape,
            )
        )
        previous_gray = gray
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
        writer.writerow(["frame_index", "timestamp_sec", "name", "s", "x", "y", "status"])
        for record in results:
            for item in record["visible_segment_state"]["tracking_point_records"]:
                point = item["point"] or [None, None]
                writer.writerow([record["frame_index"], record["timestamp_sec"], item["name"], item["s"], point[0], point[1], item["status"]])

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
            overlay = _draw_tracking_overlay(
                frame_bgr,
                record["visible_segment_state"]["tracking_points"],
                {item["name"]: item["status"] for item in record["visible_segment_state"]["tracking_point_records"]},
            )
            writer.write(overlay)
        writer.release()

    summary = {
        "video_path": str(video_path),
        "start_frame": start_frame,
        "end_frame": start_frame + len(frames_bgr) - 1,
        "fps": fps,
        "frame_count": len(results),
        "tracking_resize_scale": scale,
        "initial_tracking_points": original_tracking_points,
        "working_tracking_points": tracking_points,
    }
    write_json(output_path / "tracking_summary.json", summary)
    return summary
