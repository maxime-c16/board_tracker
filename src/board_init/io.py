from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .types import PromptSet


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_image(path: str | Path, image_rgb: np.ndarray) -> None:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image_bgr)


def save_mask_png(path: str | Path, mask: np.ndarray) -> None:
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))


def load_prompts_json(path: str | Path) -> tuple[PromptSet, int | None]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    coordinate_space = payload.get("prompt_coordinate_space", "original_image_pixels")
    if coordinate_space != "original_image_pixels":
        raise ValueError(
            "Prompt JSON must use original-image pixel coordinates. "
            f"Received prompt_coordinate_space={coordinate_space!r}."
        )
    prompts = PromptSet(
        positive=[tuple(point) for point in payload.get("positive", [])],
        negative=[tuple(point) for point in payload.get("negative", [])],
    )
    select_mask_index = payload.get("select_mask_index")
    return prompts, select_mask_index


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def write_json(path: str | Path, payload: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
