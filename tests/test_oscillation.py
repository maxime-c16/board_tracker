import math

import pytest

from board_init.oscillation import analyze_tracking_oscillation


def _make_record(frame_index: int, anchor: tuple[float, float], tip: tuple[float, float], s50: tuple[float, float], s90: tuple[float, float]) -> dict:
    def pt(value: tuple[float, float]) -> list[int]:
        return [int(round(value[0])), int(round(value[1]))]

    tracking_point_records = [
        {"name": "anchor", "s": 0.0, "point": pt(anchor), "status": "tracked"},
        {"name": "s25", "s": 0.25, "point": pt(((3 * anchor[0] + tip[0]) / 4.0, (3 * anchor[1] + tip[1]) / 4.0)), "status": "tracked"},
        {"name": "s50", "s": 0.5, "point": pt(s50), "status": "tracked"},
        {"name": "s75", "s": 0.75, "point": pt(((anchor[0] + 3 * tip[0]) / 4.0, (anchor[1] + 3 * tip[1]) / 4.0)), "status": "tracked"},
        {"name": "s90", "s": 0.9, "point": pt(s90), "status": "tracked"},
        {"name": "tip", "s": 1.0, "point": pt(tip), "status": "tracked"},
    ]
    return {
        "frame_index": frame_index,
        "timestamp_sec": frame_index / 60.0,
        "visible_segment_state": {
            "tracking_point_records": tracking_point_records,
        },
    }


def test_analyze_tracking_oscillation_accepts_stable_board_motion() -> None:
    records = []
    for frame_index in range(180):
        phase = frame_index / 9.0
        anchor = (100.0, 200.0 + 1.5 * math.sin(phase * 0.2))
        tip = (300.0, 200.0 + 16.0 * math.sin(phase))
        s50 = (200.0, 200.0 + 8.0 * math.sin(phase))
        s90 = (280.0, 200.0 + 13.0 * math.sin(phase))
        records.append(_make_record(frame_index, anchor, tip, s50, s90))

    analysis = analyze_tracking_oscillation(records, fps=60.0)

    assert analysis["summary"]["valid_frame_count"] == 180
    assert analysis["summary"]["tracking_validation"]["passed"] is True
    assert analysis["summary"]["dominant_window_peak_frames"]
    assert analysis["summary"]["dominant_window_trough_frames"]


def test_analyze_tracking_oscillation_rejects_track_drifting_with_subject() -> None:
    records = []
    for frame_index in range(180):
        dx = 3.0 * frame_index
        dy = 1.5 * frame_index
        anchor = (100.0 + dx, 200.0 + dy)
        tip = (220.0 + dx, 200.0 + dy)
        s50 = (160.0 + dx, 200.0 + dy)
        s90 = (208.0 + dx, 200.0 + dy)
        records.append(_make_record(frame_index, anchor, tip, s50, s90))

    with pytest.raises(RuntimeError, match="Tracking validation failed"):
        analyze_tracking_oscillation(records, fps=60.0)
