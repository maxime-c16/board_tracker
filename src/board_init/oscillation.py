from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
from scipy.signal import find_peaks, medfilt, savgol_filter

from .io import ensure_dir, write_json
from .tracking import TRACK_ORDER

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


VALID_TRACKING_STATES = {"tracked", "recovered"}


def load_tracking_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _station_points(record: dict[str, Any]) -> dict[str, list[int] | None]:
    return {item["name"]: item["point"] for item in record["visible_segment_state"]["tracking_point_records"]}


def _validated_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for record in records:
        visible = record.get("visible_segment_state", {})
        tracking_state = visible.get("tracking_state", "tracked")
        frame_quality = visible.get("frame_quality", {})
        if tracking_state not in VALID_TRACKING_STATES:
            continue
        if frame_quality and frame_quality.get("passed") is False:
            continue
        pts = _station_points(record)
        if any(pts.get(key) is None for key in ("anchor", "s50", "s90", "tip")):
            continue
        valid.append(record)
    return valid


def _extract_series(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    frames: list[int] = []
    timestamps: list[float] = []
    tip_rel_y: list[float] = []
    s90_rel_y: list[float] = []
    chord_dev: list[float] = []
    confidence: list[float] = []

    for rec in records:
        pts = _station_points(rec)
        anchor = np.array(pts["anchor"], dtype=np.float32)
        s50 = np.array(pts["s50"], dtype=np.float32)
        s90 = np.array(pts["s90"], dtype=np.float32)
        tip = np.array(pts["tip"], dtype=np.float32)

        chord = tip - anchor
        denom = float(np.linalg.norm(chord))
        dist = 0.0 if denom < 1e-6 else float(abs(np.cross(chord, s90 - anchor)) / denom)

        frames.append(int(rec["frame_index"]))
        timestamps.append(float(rec["timestamp_sec"]))
        tip_rel_y.append(float(tip[1] - anchor[1]))
        s90_rel_y.append(float(s90[1] - s50[1]))
        chord_dev.append(dist)
        confidence.append(float(rec["visible_segment_state"].get("frame_quality", {}).get("confidence", 1.0)))

    return {
        "frame_index": np.array(frames, dtype=int),
        "timestamp_sec": np.array(timestamps, dtype=float),
        "tip_rel_y": np.array(tip_rel_y, dtype=float),
        "s90_rel_y": np.array(s90_rel_y, dtype=float),
        "s90_chord_dev": np.array(chord_dev, dtype=float),
        "frame_confidence": np.array(confidence, dtype=float),
    }


def _extract_tracking_geometry(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    frames: list[int] = []
    anchor_x: list[float] = []
    anchor_y: list[float] = []
    tip_x: list[float] = []
    tip_y: list[float] = []
    centroid_x: list[float] = []
    centroid_y: list[float] = []
    segment_length: list[float] = []
    angle_deg: list[float] = []

    for rec in records:
        pts = _station_points(rec)
        anchor = np.array(pts["anchor"], dtype=np.float32)
        tip = np.array(pts["tip"], dtype=np.float32)
        chord = tip - anchor
        frames.append(int(rec["frame_index"]))
        anchor_x.append(float(anchor[0]))
        anchor_y.append(float(anchor[1]))
        tip_x.append(float(tip[0]))
        tip_y.append(float(tip[1]))
        centroid_x.append(float((anchor[0] + tip[0]) / 2.0))
        centroid_y.append(float((anchor[1] + tip[1]) / 2.0))
        segment_length.append(float(np.linalg.norm(chord)))
        angle_deg.append(float(np.degrees(np.arctan2(chord[1], chord[0]))))

    return {
        "frame_index": np.array(frames, dtype=int),
        "anchor_x": np.array(anchor_x, dtype=float),
        "anchor_y": np.array(anchor_y, dtype=float),
        "tip_x": np.array(tip_x, dtype=float),
        "tip_y": np.array(tip_y, dtype=float),
        "centroid_x": np.array(centroid_x, dtype=float),
        "centroid_y": np.array(centroid_y, dtype=float),
        "segment_length": np.array(segment_length, dtype=float),
        "segment_angle_deg": np.array(angle_deg, dtype=float),
    }


def _normalize(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    if std < 1e-6:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / std


def _rolling_rms(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return np.array([], dtype=float)
    window = max(1, min(window, len(values)))
    kernel = np.ones(window, dtype=float) / float(window)
    return np.sqrt(np.convolve(values**2, kernel, mode="same"))


def _support_threshold(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("inf")
    return max(0.75, float(np.percentile(np.abs(values), 65)))


def _extrema_support(
    idx: int,
    sign: int,
    combined: np.ndarray,
    tip_osc: np.ndarray,
    s90_osc: np.ndarray,
    chord_osc: np.ndarray,
    amplitude_threshold: float,
) -> bool:
    if abs(float(combined[idx])) < amplitude_threshold:
        return False
    component_values = [float(tip_osc[idx]), float(s90_osc[idx]), float(chord_osc[idx])]
    strong_components = sum(1 for value in component_values if sign * value > 0.35)
    return strong_components >= 2


def _filter_extrema(
    candidate_indices: np.ndarray,
    sign: int,
    combined: np.ndarray,
    tip_osc: np.ndarray,
    s90_osc: np.ndarray,
    chord_osc: np.ndarray,
    amplitude_threshold: float,
) -> list[int]:
    kept: list[int] = []
    for idx in candidate_indices.tolist():
        if _extrema_support(idx, sign, combined, tip_osc, s90_osc, chord_osc, amplitude_threshold):
            kept.append(int(idx))
    return kept


def _validate_tracking_geometry(geometry: dict[str, np.ndarray]) -> dict[str, Any]:
    length = geometry["segment_length"]
    median_length = float(np.median(length))
    if median_length < 1e-6:
        return {
            "passed": False,
            "reasons": ["segment length collapsed"],
            "metrics": {},
        }

    anchor = np.column_stack((geometry["anchor_x"], geometry["anchor_y"]))
    centroid = np.column_stack((geometry["centroid_x"], geometry["centroid_y"]))
    anchor_disp = np.linalg.norm(anchor - anchor[0], axis=1) / median_length
    centroid_disp = np.linalg.norm(centroid - centroid[0], axis=1) / median_length
    length_cv = float(np.std(length) / median_length)
    angle_span = float(np.ptp(geometry["segment_angle_deg"]))

    metrics = {
        "median_segment_length_px": median_length,
        "anchor_rms_drift_norm": float(np.sqrt(np.mean(anchor_disp**2))),
        "anchor_max_drift_norm": float(np.max(anchor_disp)),
        "centroid_rms_drift_norm": float(np.sqrt(np.mean(centroid_disp**2))),
        "centroid_max_drift_norm": float(np.max(centroid_disp)),
        "segment_length_cv": length_cv,
        "segment_angle_span_deg": angle_span,
    }

    reasons: list[str] = []
    if metrics["anchor_rms_drift_norm"] > 0.35:
        reasons.append("anchor drift too large for a board-like oscillation track")
    if metrics["centroid_rms_drift_norm"] > 0.75:
        reasons.append("segment centroid drifts too far, likely tracking the diver instead of the board")
    if metrics["segment_length_cv"] > 0.25:
        reasons.append("segment length varies too much for a stable visible board segment")
    if metrics["segment_angle_span_deg"] > 120.0:
        reasons.append("segment angle span is implausibly large for stable board tracking")

    return {
        "passed": not reasons,
        "reasons": reasons,
        "metrics": metrics,
    }


def _interpolate_short_gaps(
    frame_index: np.ndarray,
    timestamps: np.ndarray,
    values: np.ndarray,
    max_gap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = int(frame_index[0])
    end = int(frame_index[-1])
    dense_frames = np.arange(start, end + 1, dtype=int)
    dense_times = np.interp(dense_frames, frame_index, timestamps)
    dense_values = np.full(len(dense_frames), np.nan, dtype=float)
    positions = {int(frame): idx for idx, frame in enumerate(dense_frames.tolist())}
    for frame, value in zip(frame_index.tolist(), values.tolist()):
        dense_values[positions[int(frame)]] = float(value)

    observed = np.isfinite(dense_values)
    idx = 0
    while idx < len(dense_values):
        if observed[idx]:
            idx += 1
            continue
        gap_start = idx
        while idx < len(dense_values) and not observed[idx]:
            idx += 1
        gap_end = idx
        gap_len = gap_end - gap_start
        if gap_start == 0 or gap_end == len(dense_values) or gap_len > max_gap:
            continue
        left = gap_start - 1
        right = gap_end
        interp_frames = dense_frames[gap_start:gap_end]
        dense_values[gap_start:gap_end] = np.interp(
            interp_frames,
            [dense_frames[left], dense_frames[right]],
            [dense_values[left], dense_values[right]],
        )
    valid = np.isfinite(dense_values)
    return dense_frames[valid], dense_times[valid], dense_values[valid]


def _odd_window(length: int, preferred: int, minimum: int) -> int:
    window = min(length if length % 2 == 1 else length - 1, preferred)
    window = max(minimum, window)
    if window % 2 == 0:
        window -= 1
    return max(3, min(window, length if length % 2 == 1 else length - 1))


def _robust_trend(values: np.ndarray, fps: float) -> np.ndarray:
    if len(values) < 7:
        return np.full_like(values, float(np.mean(values)))
    median_window = _odd_window(len(values), preferred=max(7, int(round(fps * 0.2))), minimum=5)
    smooth_window = _odd_window(len(values), preferred=max(31, int(round(fps * 1.75))), minimum=7)
    median = medfilt(values, kernel_size=median_window)
    return savgol_filter(median, window_length=smooth_window, polyorder=3, mode="interp")


def _build_contact_sheet(output_path: Path, chosen_frames: list[int]) -> None:
    images = []
    for frame_idx in chosen_frames:
        path = output_path / f"sample_frame_{frame_idx:04d}.png"
        if path.exists():
            image = cv2.imread(str(path))
            if image is not None:
                label = f"frame {frame_idx}"
                cv2.putText(image, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                images.append(image)
    if not images:
        return

    max_h = max(image.shape[0] for image in images)
    max_w = max(image.shape[1] for image in images)
    tiles = []
    for image in images:
        canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
        canvas[: image.shape[0], : image.shape[1]] = image
        tiles.append(canvas)
    rows = []
    for idx in range(0, len(tiles), 2):
        row = tiles[idx : idx + 2]
        if len(row) == 1:
            row.append(np.zeros_like(row[0]))
        rows.append(cv2.hconcat(row))
    sheet = rows[0] if len(rows) == 1 else cv2.vconcat(rows)
    cv2.imwrite(str(output_path / "contact_sheet.png"), sheet)


def analyze_tracking_oscillation(records: list[dict[str, Any]], fps: float) -> dict[str, Any]:
    valid_records = _validated_records(records)
    if len(valid_records) < 30:
        raise RuntimeError("Not enough validated tracking frames to analyze oscillation.")

    geometry = _extract_tracking_geometry(valid_records)
    validation = _validate_tracking_geometry(geometry)
    if not validation["passed"]:
        reason_text = "; ".join(validation["reasons"])
        raise RuntimeError(f"Tracking validation failed before oscillation analysis: {reason_text}.")

    raw = _extract_series(valid_records)
    dense_frames, dense_times, tip_rel_y = _interpolate_short_gaps(raw["frame_index"], raw["timestamp_sec"], raw["tip_rel_y"], max_gap=4)
    dense_frames_2, dense_times_2, s90_rel_y = _interpolate_short_gaps(raw["frame_index"], raw["timestamp_sec"], raw["s90_rel_y"], max_gap=4)
    dense_frames_3, dense_times_3, chord_dev = _interpolate_short_gaps(raw["frame_index"], raw["timestamp_sec"], raw["s90_chord_dev"], max_gap=4)
    dense_frames_4, _, frame_confidence = _interpolate_short_gaps(raw["frame_index"], raw["timestamp_sec"], raw["frame_confidence"], max_gap=4)

    common_frames = sorted(set(dense_frames.tolist()) & set(dense_frames_2.tolist()) & set(dense_frames_3.tolist()) & set(dense_frames_4.tolist()))
    if len(common_frames) < 30:
        raise RuntimeError("Not enough short-gap-cleaned frames to analyze oscillation.")

    frame_to_idx = {int(frame): idx for idx, frame in enumerate(common_frames)}
    def subset(source_frames: np.ndarray, values: np.ndarray) -> np.ndarray:
        lookup = {int(frame): float(value) for frame, value in zip(source_frames.tolist(), values.tolist())}
        return np.array([lookup[frame] for frame in common_frames], dtype=float)

    series = {
        "frame_index": np.array(common_frames, dtype=int),
        "timestamp_sec": subset(dense_frames, dense_times),
        "tip_rel_y": subset(dense_frames, tip_rel_y),
        "s90_rel_y": subset(dense_frames_2, s90_rel_y),
        "s90_chord_dev": subset(dense_frames_3, chord_dev),
        "frame_confidence": subset(dense_frames_4, frame_confidence),
    }

    tip_trend = _robust_trend(series["tip_rel_y"], fps=fps)
    s90_trend = _robust_trend(series["s90_rel_y"], fps=fps)
    chord_trend = _robust_trend(series["s90_chord_dev"], fps=fps)

    tip_osc = series["tip_rel_y"] - tip_trend
    s90_osc = series["s90_rel_y"] - s90_trend
    chord_osc = series["s90_chord_dev"] - chord_trend
    combined = 0.45 * _normalize(tip_osc) + 0.4 * _normalize(s90_osc) + 0.15 * _normalize(chord_osc)
    combined *= np.clip(series["frame_confidence"], 0.35, 1.0)
    energy = _rolling_rms(combined, window=max(15, int(round(fps * 0.5))))

    peak_distance = max(8, int(round(fps * 0.2)))
    rolling_window = max(20, int(round(fps * 2.0)))
    best_start = 0
    best_std = -1.0
    for idx in range(0, len(combined) - rolling_window + 1):
        score = float(np.mean(energy[idx : idx + rolling_window]))
        if score > best_std:
            best_start = idx
            best_std = score
    best_end = best_start + rolling_window - 1

    window_indices = np.arange(best_start, best_end + 1, dtype=int)
    amplitude_threshold = _support_threshold(combined[window_indices])

    peak_candidates, _ = find_peaks(combined, distance=peak_distance, prominence=max(0.35, amplitude_threshold * 0.35))
    trough_candidates, _ = find_peaks(-combined, distance=peak_distance, prominence=max(0.35, amplitude_threshold * 0.35))
    peaks = _filter_extrema(peak_candidates, 1, combined, tip_osc, s90_osc, chord_osc, amplitude_threshold)
    troughs = _filter_extrema(trough_candidates, -1, combined, tip_osc, s90_osc, chord_osc, amplitude_threshold)

    window_index_set = set(window_indices.tolist())
    window_peaks = [int(series["frame_index"][idx]) for idx in peaks if idx in window_index_set]
    window_troughs = [int(series["frame_index"][idx]) for idx in troughs if idx in window_index_set]

    if not window_peaks or not window_troughs:
        raise RuntimeError("Oscillation validation failed: no robust peak/trough pair was found inside the active motion window.")

    validated_frame_indices = [int(record["frame_index"]) for record in valid_records]
    interpolated_frames = [frame for frame in common_frames if frame not in set(validated_frame_indices)]

    return {
        "series": {
            "frame_index": series["frame_index"].tolist(),
            "timestamp_sec": series["timestamp_sec"].tolist(),
            "tip_rel_y": series["tip_rel_y"].tolist(),
            "s90_rel_y": series["s90_rel_y"].tolist(),
            "s90_chord_dev": series["s90_chord_dev"].tolist(),
            "frame_confidence": series["frame_confidence"].tolist(),
            "tip_rel_y_trend": tip_trend.tolist(),
            "s90_rel_y_trend": s90_trend.tolist(),
            "s90_chord_dev_trend": chord_trend.tolist(),
            "tip_rel_y_detrended": tip_osc.tolist(),
            "s90_rel_y_detrended": s90_osc.tolist(),
            "s90_chord_dev_detrended": chord_osc.tolist(),
            "combined_oscillation_index": combined.tolist(),
            "combined_energy_envelope": energy.tolist(),
        },
        "summary": {
            "valid_frame_count": int(len(valid_records)),
            "clean_frame_count": int(len(series["frame_index"])),
            "interpolated_short_gap_frames": interpolated_frames,
            "validated_frame_index_range": [int(validated_frame_indices[0]), int(validated_frame_indices[-1])],
            "tracking_validation": validation,
            "dominant_window": {
                "start_frame": int(series["frame_index"][best_start]),
                "end_frame": int(series["frame_index"][best_end]),
                "energy_score": best_std,
            },
            "combined_peak_frames": [int(series["frame_index"][idx]) for idx in peaks],
            "combined_trough_frames": [int(series["frame_index"][idx]) for idx in troughs],
            "dominant_window_peak_frames": window_peaks,
            "dominant_window_trough_frames": window_troughs,
            "dominant_window_peak_values": [float(combined[np.where(series["frame_index"] == frame)[0][0]]) for frame in window_peaks],
            "dominant_window_trough_values": [float(combined[np.where(series["frame_index"] == frame)[0][0]]) for frame in window_troughs],
            "peak_selection_threshold": amplitude_threshold,
        },
    }


def save_oscillation_artifacts(
    output_dir: str | Path,
    video_path: str | Path,
    tracking_jsonl_path: str | Path,
    analysis: dict[str, Any],
    sample_frames: int = 3,
) -> None:
    output_path = ensure_dir(output_dir)
    write_json(output_path / "oscillation_summary.json", analysis["summary"])
    write_json(output_path / "oscillation_series.json", analysis["series"])

    frames = np.array(analysis["series"]["frame_index"])
    times = np.array(analysis["series"]["timestamp_sec"])
    combined = np.array(analysis["series"]["combined_oscillation_index"])
    tip_osc = np.array(analysis["series"]["tip_rel_y_detrended"])
    s90_osc = np.array(analysis["series"]["s90_rel_y_detrended"])
    energy = np.array(analysis["series"]["combined_energy_envelope"])
    confidence = np.array(analysis["series"]["frame_confidence"])
    window = analysis["summary"]["dominant_window"]

    start_frame = window["start_frame"]
    end_frame = window["end_frame"]
    start_time = float(times[np.where(frames == start_frame)[0][0]]) if start_frame in frames else float(times[0])
    end_time = float(times[np.where(frames == end_frame)[0][0]]) if end_frame in frames else float(times[-1])

    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    axes[0].plot(times, tip_osc, label="tip rel y detrended")
    axes[0].plot(times, s90_osc, label="s90 rel y detrended")
    axes[0].axvspan(start_time, end_time, color="orange", alpha=0.2)
    axes[0].legend(loc="upper right")
    axes[0].set_ylabel("Pixels")

    axes[1].plot(times, combined, color="black", label="combined oscillation index")
    axes[1].axvspan(start_time, end_time, color="orange", alpha=0.2)
    axes[1].legend(loc="upper right")
    axes[1].set_ylabel("Z-score")

    axes[2].plot(times, energy, color="tab:blue", label="energy envelope")
    axes[2].axvspan(start_time, end_time, color="orange", alpha=0.2)
    axes[2].legend(loc="upper right")
    axes[2].set_ylabel("Energy")

    axes[3].plot(times, confidence, color="tab:green", label="frame confidence")
    axes[3].axvspan(start_time, end_time, color="orange", alpha=0.2)
    axes[3].legend(loc="upper right")
    axes[3].set_ylabel("Confidence")
    axes[3].set_xlabel("Time (s)")

    plt.tight_layout()
    fig.savefig(output_path / "oscillation_plot.png", dpi=150)
    plt.close(fig)

    records = load_tracking_records(tracking_jsonl_path)
    record_by_frame = {int(record["frame_index"]): record for record in records}
    chosen_frames = []
    for source in (analysis["summary"]["dominant_window_peak_frames"], analysis["summary"]["dominant_window_trough_frames"]):
        for frame_idx in source[:sample_frames]:
            if frame_idx not in chosen_frames:
                chosen_frames.append(frame_idx)
    cap = cv2.VideoCapture(str(video_path))
    for frame_idx in chosen_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        record = record_by_frame.get(frame_idx)
        if record is None:
            continue
        visible = record["visible_segment_state"]
        state = visible.get("tracking_state", "tracked")
        state_colors = {
            "tracked": (0, 255, 0),
            "recovered": (255, 200, 0),
            "predicted": (0, 165, 255),
            "invalid": (0, 0, 255),
        }
        for item in visible["tracking_point_records"]:
            point = item["point"]
            if point is None:
                continue
            color = state_colors.get(item["status"], state_colors.get(state, (255, 255, 255)))
            cv2.circle(frame, tuple(point), 6, color, -1)
            cv2.putText(frame, item["name"], (point[0] + 6, point[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"{state} conf={visible.get('frame_quality', {}).get('confidence', 0.0):.2f}",
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            state_colors.get(state, (255, 255, 255)),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(output_path / f"sample_frame_{frame_idx:04d}.png"), frame)
    cap.release()
    _build_contact_sheet(output_path, chosen_frames)
