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
    results = track_points_in_frames(frames, initial_points, fps=60.0, config=TrackingConfig(save_overlay_video=False))
    assert len(results) == 3
    last = results[-1]["visible_segment_state"]["tracking_points"]
    assert last["anchor"][0] >= 24
    assert last["s50"][0] >= 64
