from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


Point = tuple[int, int]
BBox = tuple[int, int, int, int]


@dataclass
class PromptSet:
    positive: list[Point] = field(default_factory=list)
    negative: list[Point] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "positive": [list(point) for point in self.positive],
            "negative": [list(point) for point in self.negative],
        }


@dataclass
class CandidateResult:
    index: int
    raw_mask: np.ndarray
    processed_mask: np.ndarray
    working_raw_mask: np.ndarray
    working_processed_mask: np.ndarray
    sam_score: float
    features: dict[str, Any]
    working_features: dict[str, Any]
    scoring_terms: dict[str, float]
    combined_score: float


@dataclass(frozen=True)
class ResizeMetadata:
    original_width: int
    original_height: int
    working_width: int
    working_height: int
    resize_scale: float
    max_side: int | None
    prompt_coordinate_space: str = "original_image_pixels"
    sam_prompt_coordinate_space: str = "sam_working_image_pixels"
    exported_geometry_coordinate_space: str = "original_image_pixels"

    @property
    def applied(self) -> bool:
        return (self.original_width, self.original_height) != (self.working_width, self.working_height)

    def to_json(self) -> dict[str, Any]:
        return {
            "original_image_size": {
                "width": self.original_width,
                "height": self.original_height,
            },
            "sam_working_image_size": {
                "width": self.working_width,
                "height": self.working_height,
            },
            "resize_scale": self.resize_scale,
            "max_side": self.max_side,
            "resize_applied": self.applied,
            "prompt_coordinate_space": self.prompt_coordinate_space,
            "sam_prompt_coordinate_space": self.sam_prompt_coordinate_space,
            "exported_geometry_coordinate_space": self.exported_geometry_coordinate_space,
        }


@dataclass
class RunArtifacts:
    output_dir: Path
    selected_index: int
    candidates: list[CandidateResult]
    summary: dict[str, Any]
