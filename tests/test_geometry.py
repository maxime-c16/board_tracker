import numpy as np

from board_init.geometry import extract_geometry


def test_extract_geometry_for_thin_board() -> None:
    mask = np.zeros((120, 200), dtype=bool)
    mask[55:65, 20:180] = True
    geometry = extract_geometry(mask)
    assert geometry["valid"] is True
    assert geometry["length_estimate_px"] > 100
    assert geometry["average_thickness_px"] > 5
    assert geometry["elongation"] > 10
    assert geometry["skeleton_branch_point_count"] == 0
    assert geometry["tracking_points"]["anchor"] is not None
    assert geometry["tracking_points"]["tip"] is not None
    assert geometry["tracking_points"]["s50"] is not None
    assert geometry["observation_state"]["partial_board_observation"] is False


def test_extract_geometry_marks_border_truncation() -> None:
    mask = np.zeros((60, 100), dtype=bool)
    mask[26:34, 0:80] = True
    geometry = extract_geometry(mask)
    assert geometry["observation_state"]["partial_board_observation"] is True
    assert geometry["observation_state"]["anchor"]["near_image_border"] is True
