import numpy as np

from board_init.config import PostprocessConfig
from board_init.geometry import extract_geometry
from board_init.postprocess import cleanup_mask, rectify_board_mask
from board_init.types import PromptSet


def test_cleanup_mask_removes_small_components() -> None:
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:25, 5:10] = True
    mask[30:32, 30:32] = True
    cleaned, metrics = cleanup_mask(mask, PostprocessConfig(min_component_area=20, hole_fill_area=4))
    assert cleaned.sum() > 0
    assert not cleaned[30:32, 30:32].any()
    assert metrics["component_area"] == int(cleaned.sum())


def test_rectify_board_mask_builds_single_strip() -> None:
    mask = np.zeros((120, 220), dtype=bool)
    mask[56:64, 20:200] = True
    mask[60:88, 120:128] = True
    prompts = PromptSet(positive=[(30, 60), (190, 60)], negative=[])

    refined, metrics = rectify_board_mask(
        mask,
        extract_geometry(mask),
        prompts,
        roi=None,
        config=PostprocessConfig(),
    )

    refined_geometry = extract_geometry(refined)
    assert metrics["applied"] is True
    assert refined_geometry["skeleton_branch_point_count"] == 0
    assert refined_geometry["bbox"][0] <= 30
    assert refined_geometry["bbox"][2] >= 190
