from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .types import BBox, Point, PromptSet, ResizeMetadata


def build_resize_metadata(
    image_shape: tuple[int, int, int] | tuple[int, int],
    max_side: int | None,
) -> ResizeMetadata:
    height, width = image_shape[:2]
    if max_side is None or max(height, width) <= max_side:
        return ResizeMetadata(
            original_width=width,
            original_height=height,
            working_width=width,
            working_height=height,
            resize_scale=1.0,
            max_side=max_side,
        )

    scale = float(max_side) / float(max(height, width))
    working_width = max(1, int(round(width * scale)))
    working_height = max(1, int(round(height * scale)))
    return ResizeMetadata(
        original_width=width,
        original_height=height,
        working_width=working_width,
        working_height=working_height,
        resize_scale=scale,
        max_side=max_side,
    )


def resize_image_for_sam(image_rgb: np.ndarray, metadata: ResizeMetadata) -> np.ndarray:
    if not metadata.applied:
        return image_rgb.copy()
    return cv2.resize(
        image_rgb,
        (metadata.working_width, metadata.working_height),
        interpolation=cv2.INTER_AREA,
    )


def scale_point_to_working(point: Point, metadata: ResizeMetadata) -> Point:
    if not metadata.applied:
        return point
    scale = metadata.resize_scale
    x = min(metadata.working_width - 1, max(0, int(round(point[0] * scale))))
    y = min(metadata.working_height - 1, max(0, int(round(point[1] * scale))))
    return x, y


def scale_point_to_original(point: Point, metadata: ResizeMetadata) -> Point:
    if not metadata.applied:
        return point
    scale = 1.0 / metadata.resize_scale
    x = min(metadata.original_width - 1, max(0, int(round(point[0] * scale))))
    y = min(metadata.original_height - 1, max(0, int(round(point[1] * scale))))
    return x, y


def prompts_to_working(prompts: PromptSet, metadata: ResizeMetadata) -> PromptSet:
    if not metadata.applied:
        return PromptSet(positive=list(prompts.positive), negative=list(prompts.negative))
    return PromptSet(
        positive=[scale_point_to_working(point, metadata) for point in prompts.positive],
        negative=[scale_point_to_working(point, metadata) for point in prompts.negative],
    )


def roi_to_working(roi: BBox | None, metadata: ResizeMetadata) -> BBox | None:
    if roi is None or not metadata.applied:
        return roi
    x1, y1 = scale_point_to_working((roi[0], roi[1]), metadata)
    x2, y2 = scale_point_to_working((roi[2], roi[3]), metadata)
    return x1, y1, x2, y2


def resize_mask_to_original(mask: np.ndarray, metadata: ResizeMetadata) -> np.ndarray:
    if not metadata.applied:
        return mask.astype(bool)
    resized = cv2.resize(
        mask.astype(np.uint8),
        (metadata.original_width, metadata.original_height),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)


def clip_roi_to_image(roi: BBox, image_shape: tuple[int, int, int] | tuple[int, int]) -> BBox:
    height, width = image_shape[:2]
    x1 = min(width - 1, max(0, int(roi[0])))
    y1 = min(height - 1, max(0, int(roi[1])))
    x2 = min(width, max(x1 + 1, int(roi[2])))
    y2 = min(height, max(y1 + 1, int(roi[3])))
    return x1, y1, x2, y2


def crop_image(image_rgb: np.ndarray, roi: BBox | None) -> np.ndarray:
    if roi is None:
        return image_rgb
    x1, y1, x2, y2 = clip_roi_to_image(roi, image_rgb.shape)
    return image_rgb[y1:y2, x1:x2].copy()


def shift_prompts_to_crop(prompts: PromptSet, roi: BBox | None) -> PromptSet:
    if roi is None:
        return PromptSet(positive=list(prompts.positive), negative=list(prompts.negative))
    x1, y1, _, _ = roi
    return PromptSet(
        positive=[(x - x1, y - y1) for x, y in prompts.positive],
        negative=[(x - x1, y - y1) for x, y in prompts.negative],
    )


def embed_crop_mask(mask: np.ndarray, full_shape: tuple[int, int], roi: BBox | None) -> np.ndarray:
    if roi is None:
        return mask.astype(bool)
    x1, y1, x2, y2 = roi
    canvas = np.zeros(full_shape, dtype=bool)
    canvas[y1:y2, x1:x2] = mask.astype(bool)
    return canvas


def display_point_to_original(
    point: Point,
    metadata: ResizeMetadata,
    view_origin_working: Point = (0, 0),
) -> Point:
    working_point = (point[0] + view_origin_working[0], point[1] + view_origin_working[1])
    return scale_point_to_original(working_point, metadata)


def original_point_to_display(
    point: Point,
    metadata: ResizeMetadata,
    view_origin_working: Point = (0, 0),
) -> Point:
    working_point = scale_point_to_working(point, metadata)
    return working_point[0] - view_origin_working[0], working_point[1] - view_origin_working[1]


def _scale_length(value: float | int | None, metadata: ResizeMetadata) -> float | int | None:
    if value is None or not metadata.applied:
        return value
    return float(value) / metadata.resize_scale


def scale_geometry_to_original(features: dict[str, Any], metadata: ResizeMetadata) -> dict[str, Any]:
    if not metadata.applied:
        return dict(features)

    scaled = dict(features)
    if scaled.get("bbox") is not None:
        x1, y1 = scale_point_to_original((scaled["bbox"][0], scaled["bbox"][1]), metadata)
        x2, y2 = scale_point_to_original((scaled["bbox"][2], scaled["bbox"][3]), metadata)
        scaled["bbox"] = [x1, y1, x2, y2]
    if scaled.get("rotated_bbox") is not None:
        scaled["rotated_bbox"] = [
            list(scale_point_to_original((point[0], point[1]), metadata))
            for point in scaled["rotated_bbox"]
        ]
    if scaled.get("centerline_points") is not None:
        scaled["centerline_points"] = [
            list(scale_point_to_original((point[0], point[1]), metadata))
            for point in scaled["centerline_points"]
        ]
    if scaled.get("tracking_points") is not None:
        scaled["tracking_points"] = {
            name: (
                list(scale_point_to_original((point[0], point[1]), metadata))
                if point is not None
                else None
            )
            for name, point in scaled["tracking_points"].items()
        }
    if scaled.get("tracking_point_records") is not None:
        scaled["tracking_point_records"] = [
            {
                **record,
                "point": (
                    list(scale_point_to_original((record["point"][0], record["point"][1]), metadata))
                    if record.get("point") is not None
                    else None
                ),
            }
            for record in scaled["tracking_point_records"]
        ]
    if scaled.get("anchor_endpoint") is not None:
        scaled["anchor_endpoint"] = list(
            scale_point_to_original((scaled["anchor_endpoint"][0], scaled["anchor_endpoint"][1]), metadata)
        )
    if scaled.get("tip_endpoint") is not None:
        scaled["tip_endpoint"] = list(
            scale_point_to_original((scaled["tip_endpoint"][0], scaled["tip_endpoint"][1]), metadata)
        )
    if scaled.get("observation_state") is not None:
        observation_state = dict(scaled["observation_state"])
        for key in ("anchor", "tip"):
            if observation_state.get(key) and observation_state[key].get("point") is not None:
                point = observation_state[key]["point"]
                observation_state[key] = {
                    **observation_state[key],
                    "point": list(scale_point_to_original((point[0], point[1]), metadata)),
                }
        if observation_state.get("stations") is not None:
            observation_state["stations"] = [
                {
                    **station,
                    "point": (
                        list(scale_point_to_original((station["point"][0], station["point"][1]), metadata))
                        if station.get("point") is not None
                        else None
                    ),
                }
                for station in observation_state["stations"]
            ]
        scaled["observation_state"] = observation_state

    if scaled.get("mask_area") is not None:
        scaled["mask_area"] = int(round(float(scaled["mask_area"]) / (metadata.resize_scale ** 2)))
    for key in ("length_estimate_px", "average_thickness_px", "width_std_px"):
        if key in scaled:
            scaled[key] = _scale_length(scaled[key], metadata)
    return scaled
