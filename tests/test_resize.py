import json

import numpy as np

from board_init.config import AppConfig
from board_init.pipeline import run_pipeline
from board_init.resize import build_resize_metadata, prompts_to_working, resize_image_for_sam, scale_point_to_original
from board_init.types import PromptSet


class FakePredictor:
    def __init__(self) -> None:
        self.image_shape: tuple[int, int, int] | None = None
        self.last_prompts: PromptSet | None = None

    def set_image(self, image_rgb: np.ndarray) -> None:
        self.image_shape = image_rgb.shape

    def predict(self, prompts: PromptSet, multimask_output: bool = True) -> tuple[np.ndarray, np.ndarray]:
        del multimask_output
        self.last_prompts = prompts
        assert self.image_shape is not None
        height, width = self.image_shape[:2]
        mask = np.zeros((height, width), dtype=bool)
        mask[max(0, height // 2 - 4):min(height, height // 2 + 4), 5:max(6, width - 5)] = True
        return np.stack([mask]), np.array([0.93], dtype=float)


def test_resize_metadata_and_prompt_mapping_round_trip() -> None:
    metadata = build_resize_metadata((100, 200, 3), max_side=100)
    prompts = PromptSet(positive=[(40, 20)], negative=[(60, 40)])
    working_prompts = prompts_to_working(prompts, metadata)
    assert metadata.resize_scale == 0.5
    assert working_prompts.positive == [(20, 10)]
    assert working_prompts.negative == [(30, 20)]
    assert scale_point_to_original((20, 10), metadata) == (40, 20)


def test_run_pipeline_exports_original_space_metadata(tmp_path) -> None:
    image_rgb = np.zeros((100, 200, 3), dtype=np.uint8)
    metadata = build_resize_metadata(image_rgb.shape, max_side=100)
    working_image = resize_image_for_sam(image_rgb, metadata)
    predictor = FakePredictor()
    prompts = PromptSet(positive=[(40, 50)], negative=[])

    artifacts = run_pipeline(
        image_rgb=image_rgb,
        working_image_rgb=working_image,
        image_path="frame.png",
        prompts=prompts,
        predictor=predictor,
        config=AppConfig(),
        output_dir=tmp_path,
        resize_metadata=metadata,
    )

    assert predictor.image_shape == (50, 100, 3)
    assert predictor.last_prompts is not None
    assert predictor.last_prompts.positive == [(20, 25)]
    assert artifacts.summary["resize_metadata"]["original_image_size"] == {"width": 200, "height": 100}
    assert artifacts.summary["resize_metadata"]["sam_working_image_size"] == {"width": 100, "height": 50}
    assert artifacts.summary["final_geometry"]["bbox"] is not None

    best_features = json.loads((tmp_path / "best_features.json").read_text(encoding="utf-8"))
    assert best_features["geometry_coordinate_space"] == "original_image_pixels"
    assert best_features["resize_metadata"]["resize_applied"] is True
    tracking_points = json.loads((tmp_path / "tracking_points.json").read_text(encoding="utf-8"))
    assert tracking_points["tracking_points"]["s50"] is not None
    assert "observation_state" in tracking_points
    best_mask = (tmp_path / "best_mask.png").read_bytes()
    assert best_mask


def test_run_pipeline_uses_roi_crop_for_predictor(tmp_path) -> None:
    image_rgb = np.zeros((100, 200, 3), dtype=np.uint8)
    metadata = build_resize_metadata(image_rgb.shape, max_side=100)
    working_image = resize_image_for_sam(image_rgb, metadata)
    predictor = FakePredictor()
    prompts = PromptSet(positive=[(40, 50)], negative=[])

    artifacts = run_pipeline(
        image_rgb=image_rgb,
        working_image_rgb=working_image,
        image_path="frame.png",
        prompts=prompts,
        predictor=predictor,
        config=AppConfig(),
        output_dir=tmp_path,
        resize_metadata=metadata,
        roi=(20, 30, 100, 90),
    )

    assert predictor.image_shape == (30, 40, 3)
    assert predictor.last_prompts is not None
    assert predictor.last_prompts.positive == [(10, 10)]
    assert artifacts.summary["roi"] == [20, 30, 100, 90]
    assert artifacts.summary["working_roi"] == [10, 15, 50, 45]
