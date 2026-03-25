from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
from scipy.signal import find_peaks, savgol_filter

from .io import ensure_dir, write_json
from .tracking import TRACK_ORDER

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_tracking_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _extract_series(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    frames: list[int] = []
    timestamps: list[float] = []
    tip_rel_y: list[float] = []
    s90_rel_y: list[float] = []
    chord_dev: list[float] = []

    for rec in records:
        pts = {item["name"]: item["point"] for item in rec["visible_segment_state"]["tracking_point_records"]}
        if any(pts.get(key) is None for key in ("anchor", "s50", "s90", "tip")):
            continue
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

    return {
        "frame_index": np.array(frames, dtype=int),
        "timestamp_sec": np.array(timestamps, dtype=float),
        "tip_rel_y": np.array(tip_rel_y, dtype=float),
        "s90_rel_y": np.array(s90_rel_y, dtype=float),
        "s90_chord_dev": np.array(chord_dev, dtype=float),
    }


def _normalize(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    if std < 1e-6:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / std


def analyze_tracking_oscillation(records: list[dict[str, Any]], fps: float) -> dict[str, Any]:
    series = _extract_series(records)
    if len(series["frame_index"]) < 30:
        raise RuntimeError("Not enough valid tracking frames to analyze oscillation.")

    window = min(len(series["frame_index"]) // 2 * 2 - 1, 181)
    window = max(31, window if window % 2 == 1 else window - 1)

    tip_trend = savgol_filter(series["tip_rel_y"], window_length=window, polyorder=3, mode="interp")
    s90_trend = savgol_filter(series["s90_rel_y"], window_length=window, polyorder=3, mode="interp")
    chord_trend = savgol_filter(series["s90_chord_dev"], window_length=window, polyorder=3, mode="interp")

    tip_osc = series["tip_rel_y"] - tip_trend
    s90_osc = series["s90_rel_y"] - s90_trend
    chord_osc = series["s90_chord_dev"] - chord_trend
    combined = 0.45 * _normalize(tip_osc) + 0.4 * _normalize(s90_osc) + 0.15 * _normalize(chord_osc)

    peak_distance = max(8, int(round(fps * 0.2)))
    peaks, _ = find_peaks(combined, distance=peak_distance, prominence=0.5)
    troughs, _ = find_peaks(-combined, distance=peak_distance, prominence=0.5)

    rolling_window = max(20, int(round(fps * 2.0)))
    best_start = 0
    best_std = -1.0
    for idx in range(0, len(combined) - rolling_window + 1):
        score = float(np.std(combined[idx:idx + rolling_window]))
        if score > best_std:
            best_start = idx
            best_std = score
    best_end = best_start + rolling_window - 1

    window_mask = (series["frame_index"] >= series["frame_index"][best_start]) & (series["frame_index"] <= series["frame_index"][best_end])
    window_indices = np.where(window_mask)[0]
    window_peaks = [int(series["frame_index"][idx]) for idx in peaks if idx in window_indices]
    window_troughs = [int(series["frame_index"][idx]) for idx in troughs if idx in window_indices]

    return {
        "series": {
            "frame_index": series["frame_index"].tolist(),
            "timestamp_sec": series["timestamp_sec"].tolist(),
            "tip_rel_y": series["tip_rel_y"].tolist(),
            "s90_rel_y": series["s90_rel_y"].tolist(),
            "s90_chord_dev": series["s90_chord_dev"].tolist(),
            "tip_rel_y_detrended": tip_osc.tolist(),
            "s90_rel_y_detrended": s90_osc.tolist(),
            "s90_chord_dev_detrended": chord_osc.tolist(),
            "combined_oscillation_index": combined.tolist(),
        },
        "summary": {
            "valid_frame_count": int(len(series["frame_index"])),
            "dominant_window": {
                "start_frame": int(series["frame_index"][best_start]),
                "end_frame": int(series["frame_index"][best_end]),
                "std_score": best_std,
            },
            "combined_peak_frames": [int(series["frame_index"][idx]) for idx in peaks],
            "combined_trough_frames": [int(series["frame_index"][idx]) for idx in troughs],
            "dominant_window_peak_frames": window_peaks,
            "dominant_window_trough_frames": window_troughs,
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
    window = analysis["summary"]["dominant_window"]

    start_frame = window["start_frame"]
    end_frame = window["end_frame"]
    start_time = float(times[np.where(frames == start_frame)[0][0]]) if start_frame in frames else float(times[0])
    end_time = float(times[np.where(frames == end_frame)[0][0]]) if end_frame in frames else float(times[-1])

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    axes[0].plot(times, tip_osc, label="tip rel y detrended")
    axes[0].plot(times, s90_osc, label="s90 rel y detrended")
    axes[0].axvspan(start_time, end_time, color="orange", alpha=0.2)
    axes[0].legend(loc="upper right")
    axes[0].set_ylabel("Pixels")

    axes[1].plot(times, combined, color="black", label="combined oscillation index")
    axes[1].axvspan(start_time, end_time, color="orange", alpha=0.2)
    axes[1].legend(loc="upper right")
    axes[1].set_ylabel("Z-score")
    axes[1].set_xlabel("Time (s)")

    axes[2].plot(frames, combined, color="tab:blue")
    axes[2].set_xlabel("Frame")
    axes[2].set_ylabel("Combined Index")
    axes[2].axvspan(start_frame, end_frame, color="orange", alpha=0.2)
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
        for item in record["visible_segment_state"]["tracking_point_records"]:
            point = item["point"]
            if point is None:
                continue
            color = (0, 255, 0) if item["status"] == "tracked" else ((0, 200, 255) if item["status"] == "predicted" else (0, 0, 255))
            cv2.circle(frame, tuple(point), 6, color, -1)
            cv2.putText(frame, item["name"], (point[0] + 6, point[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.imwrite(str(output_path / f"sample_frame_{frame_idx:04d}.png"), frame)
    cap.release()
