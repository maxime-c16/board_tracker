from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PostprocessConfig:
    min_component_area: int = 400
    hole_fill_area: int = 250
    smoothing_kernel: int = 5
    opening_kernel: int = 3
    crop_pad: int = 80
    enable_board_rectification: bool = True
    rectification_width_scale: float = 1.0
    rectification_prompt_margin_scale: float = 1.5
    rectification_min_elongation: float = 8.0


@dataclass
class ScoringConfig:
    sam_score_weight: float = 0.35
    elongation_weight: float = 0.2
    width_consistency_weight: float = 0.12
    click_containment_weight: float = 0.12
    roi_consistency_weight: float = 0.08
    centerline_quality_weight: float = 0.13
    prompt_span_weight: float = 0.12
    blob_penalty_weight: float = 0.1
    branch_penalty_weight: float = 0.16
    endpoint_penalty_weight: float = 0.08
    min_click_distance_from_edge: float = 0.0
    target_elongation: float = 20.0
    target_centerline_ratio: float = 25.0
    target_endpoint_count: int = 2
    target_prompt_margin_thickness: float = 4.0


@dataclass
class RuntimeConfig:
    multimask_output: bool = True
    save_all_candidates: bool = True
    auto_roi_scale: float = 1.8
    debug: bool = False


@dataclass
class VisualizationConfig:
    overlay_alpha: float = 0.45
    point_radius: int = 6


@dataclass
class AppConfig:
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass(instance: Any, updates: dict[str, Any]) -> Any:
    for key, value in updates.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path | None = None) -> AppConfig:
    config = AppConfig()
    if path is None:
        return config
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return _merge_dataclass(config, raw)
