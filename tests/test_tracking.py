import numpy as np

from board_init.tracking import TrackingConfig, track_points_in_frames


def test_track_points_in_frames_tracks_simple_translation() -> None:
    frames = []
    for dx in (0, 3, 6):
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        frame[38:42, 20 + dx:100 + dx] = 255
        frames.append(frame)

    initial_points = {
        "anchor": [20, 40],
        "s25": [40, 40],
        "s50": [60, 40],
        "s75": [80, 40],
        "s90": [92, 40],
        "tip": [99, 40],
    }
    results = track_points_in_frames(
        frames,
        initial_points,
        fps=60.0,
        config=TrackingConfig(
            save_overlay_video=False,
            anchor_max_drift_norm=0.2,
            s25_max_drift_norm=0.3,
            s50_max_drift_norm=0.45,
            s75_max_drift_norm=0.55,
            s90_max_drift_norm=0.65,
            tip_max_drift_norm=0.8,
        ),
    )
    assert len(results) == 3
    last = results[-1]["visible_segment_state"]["tracking_points"]
    assert last["anchor"][0] >= 24
    assert last["s50"][0] >= 64


def test_track_points_in_frames_rejects_large_drift_from_seed_geometry() -> None:
    frames = []
    for dx in (0, 40, 80):
        frame = np.zeros((120, 220, 3), dtype=np.uint8)
        frame[58:62, 20 + dx:100 + dx] = 255
        frames.append(frame)

    initial_points = {
        "anchor": [20, 60],
        "s25": [40, 60],
        "s50": [60, 60],
        "s75": [80, 60],
        "s90": [92, 60],
        "tip": [99, 60],
    }

    results = track_points_in_frames(frames, initial_points, fps=60.0, config=TrackingConfig(save_overlay_video=False))

    last = results[-1]["visible_segment_state"]["tracking_points"]
    assert last["anchor"] is None
    assert last["s50"] is None
