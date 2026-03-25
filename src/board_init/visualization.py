from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .types import PromptSet


def overlay_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.45,
) -> np.ndarray:
    overlay = image_rgb.copy()
    mask_bool = mask.astype(bool)
    color_arr = np.array(color, dtype=np.uint8)
    overlay[mask_bool] = (
        image_rgb[mask_bool] * (1.0 - alpha) + color_arr * alpha
    ).astype(np.uint8)
    return overlay


def draw_prompts(
    image_rgb: np.ndarray,
    prompts: PromptSet,
    radius: int = 6,
) -> np.ndarray:
    canvas = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    for x, y in prompts.positive:
        cv2.circle(canvas, (x, y), radius, (0, 255, 0), thickness=-1)
    for x, y in prompts.negative:
        cv2.circle(canvas, (x, y), radius, (0, 0, 255), thickness=-1)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def draw_centerline(
    image_rgb: np.ndarray,
    centerline_points: list[tuple[int, int]],
    anchor: tuple[int, int] | None,
    tip: tuple[int, int] | None,
) -> np.ndarray:
    canvas = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    if len(centerline_points) >= 2:
        pts = np.array(centerline_points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=False, color=(255, 255, 0), thickness=2)
    if anchor is not None:
        cv2.circle(canvas, anchor, 7, (255, 0, 0), thickness=-1)
    if tip is not None:
        cv2.circle(canvas, tip, 7, (0, 165, 255), thickness=-1)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def save_overlay(path: str | Path, image_rgb: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
