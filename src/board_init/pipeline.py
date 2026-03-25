from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .config import AppConfig
from .geometry import extract_geometry
from .io import ensure_dir, save_image, save_mask_png, write_json
from .postprocess import postprocess_mask, rectify_board_mask
from .resize import (
    crop_image,
    embed_crop_mask,
    prompts_to_working,
    resize_mask_to_original,
    roi_to_working,
    scale_geometry_to_original,
    shift_prompts_to_crop,
)
from .scoring import score_candidate
from .types import CandidateResult, PromptSet, ResizeMetadata, RunArtifacts
from .visualization import draw_centerline, draw_prompts, overlay_mask, save_overlay

LOGGER = logging.getLogger(__name__)


def compute_roi(
    image_shape: tuple[int, int, int],
    prompts: PromptSet,
    roi: tuple[int, int, int, int] | None,
    crop_pad: int,
) -> tuple[int, int, int, int] | None:
    if roi is not None:
        return roi
    if not prompts.positive:
        return None
    xs = [point[0] for point in prompts.positive]
    ys = [point[1] for point in prompts.positive]
    x_center = int(np.mean(xs))
    y_center = int(np.mean(ys))
    h, w = image_shape[:2]
    x1 = max(0, x_center - crop_pad)
    y1 = max(0, y_center - crop_pad)
    x2 = min(w, x_center + crop_pad)
    y2 = min(h, y_center + crop_pad)
    return (x1, y1, x2, y2)


def encode_rle(mask: np.ndarray) -> dict[str, Any]:
    flat = mask.astype(np.uint8).ravel(order="F")
    counts: list[int] = []
    current = 0
    run = 0
    for value in flat:
        if int(value) == current:
            run += 1
        else:
            counts.append(run)
            run = 1
            current = int(value)
    counts.append(run)
    return {"size": list(mask.shape), "counts": counts}


def export_coco_annotation(image_path: str, mask: np.ndarray, category_id: int = 1) -> dict[str, Any]:
    ys, xs = np.where(mask)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return {
        "images": [{"id": 1, "file_name": image_path, "width": int(mask.shape[1]), "height": int(mask.shape[0])}],
        "annotations": [{
            "id": 1,
            "image_id": 1,
            "category_id": category_id,
            "iscrowd": 1,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "area": int(mask.sum()),
            "segmentation": encode_rle(mask),
        }],
        "categories": [{"id": category_id, "name": "springboard"}],
    }


def save_candidate_artifacts(
    output_dir: Path,
    image_rgb: np.ndarray,
    prompts: PromptSet,
    candidate: CandidateResult,
    overlay_alpha: float,
    debug: bool,
    resize_metadata: ResizeMetadata,
) -> None:
    prefix = output_dir / f"candidate_{candidate.index}"
    save_mask_png(prefix.with_name(f"{prefix.name}_mask.png"), candidate.processed_mask)
    overlay = draw_prompts(overlay_mask(image_rgb, candidate.processed_mask, alpha=overlay_alpha), prompts)
    overlay = draw_centerline(
        overlay,
        [tuple(point) for point in candidate.features.get("centerline_points", [])],
        tuple(candidate.features["anchor_endpoint"]) if candidate.features.get("anchor_endpoint") else None,
        tuple(candidate.features["tip_endpoint"]) if candidate.features.get("tip_endpoint") else None,
    )
    save_overlay(prefix.with_name(f"{prefix.name}_overlay.png"), overlay)
    feature_payload = {
        "resize_metadata": resize_metadata.to_json(),
        "prompt_coordinate_space": resize_metadata.prompt_coordinate_space,
        "geometry_coordinate_space": resize_metadata.exported_geometry_coordinate_space,
        "features": candidate.features,
        "working_resolution_features": candidate.working_features,
        "scoring_terms": candidate.scoring_terms,
        "combined_score": candidate.combined_score,
        "sam_score": candidate.sam_score,
    }
    write_json(prefix.with_name(f"{prefix.name}_features.json"), feature_payload)
    if debug:
        save_mask_png(prefix.with_name(f"{prefix.name}_raw_mask.png"), candidate.raw_mask)


def run_pipeline(
    image_rgb: np.ndarray,
    working_image_rgb: np.ndarray,
    image_path: str,
    prompts: PromptSet,
    predictor: Any,
    config: AppConfig,
    output_dir: Path,
    resize_metadata: ResizeMetadata,
    model_type: str | None = None,
    checkpoint_path: str | None = None,
    device: str | None = None,
    roi: tuple[int, int, int, int] | None = None,
    select_mask_index: int | None = None,
    export_rle: bool = False,
    export_coco_json: bool = False,
) -> RunArtifacts:
    ensure_dir(output_dir)
    save_image(output_dir / "input_copy.png", image_rgb)
    write_json(
        output_dir / "prompts.json",
        {
            **prompts.to_json(),
            **resize_metadata.to_json(),
            "prompt_coordinate_space": resize_metadata.prompt_coordinate_space,
        },
    )

    working_prompts = prompts_to_working(prompts, resize_metadata)
    working_roi = roi_to_working(roi, resize_metadata)
    predictor_image = crop_image(working_image_rgb, working_roi)
    predictor_prompts = shift_prompts_to_crop(working_prompts, working_roi)

    predictor.set_image(predictor_image)
    masks, sam_scores = predictor.predict(predictor_prompts, multimask_output=config.runtime.multimask_output)
    effective_roi = compute_roi(working_image_rgb.shape, working_prompts, working_roi, config.postprocess.crop_pad)
    candidates: list[CandidateResult] = []
    for index, (mask, sam_score) in enumerate(zip(masks, sam_scores)):
        full_mask_working = embed_crop_mask(mask, working_image_rgb.shape[:2], working_roi)
        processed_mask_working, post_metrics = postprocess_mask(full_mask_working, config.postprocess)
        working_features = extract_geometry(processed_mask_working)
        rectified_mask_working, rectification_metrics = rectify_board_mask(
            processed_mask_working,
            working_features,
            working_prompts,
            effective_roi,
            config.postprocess,
        )
        if rectification_metrics.get("applied"):
            processed_mask_working = rectified_mask_working
            working_features = extract_geometry(processed_mask_working)
        post_metrics = {**post_metrics, "rectification": rectification_metrics}
        working_features["postprocess"] = dict(post_metrics)
        processed_mask_original = resize_mask_to_original(processed_mask_working, resize_metadata)
        raw_mask_original = resize_mask_to_original(full_mask_working, resize_metadata)
        features = scale_geometry_to_original(working_features, resize_metadata)
        features["postprocess"] = dict(post_metrics)
        score, scoring_terms = score_candidate(
            processed_mask_working,
            sam_score=float(sam_score),
            features=working_features,
            prompts=working_prompts,
            roi=effective_roi,
            config=config.scoring,
        )
        candidates.append(
            CandidateResult(
                index=index,
                raw_mask=raw_mask_original,
                processed_mask=processed_mask_original,
                working_raw_mask=full_mask_working,
                working_processed_mask=processed_mask_working,
                sam_score=float(sam_score),
                features=features,
                working_features=working_features,
                scoring_terms=scoring_terms,
                combined_score=score,
            )
        )

    ranked = sorted(candidates, key=lambda item: item.combined_score, reverse=True)
    selected = select_mask_index if select_mask_index is not None else ranked[0].index
    selected_candidate = next(candidate for candidate in candidates if candidate.index == selected)

    if config.runtime.save_all_candidates:
        for candidate in candidates:
            save_candidate_artifacts(
                output_dir=output_dir,
                image_rgb=image_rgb,
                prompts=prompts,
                candidate=candidate,
                overlay_alpha=config.visualization.overlay_alpha,
                debug=config.runtime.debug,
                resize_metadata=resize_metadata,
            )

    best_overlay = draw_prompts(
        overlay_mask(image_rgb, selected_candidate.processed_mask, alpha=config.visualization.overlay_alpha),
        prompts,
    )
    best_overlay = draw_centerline(
        best_overlay,
        [tuple(point) for point in selected_candidate.features.get("centerline_points", [])],
        tuple(selected_candidate.features["anchor_endpoint"]) if selected_candidate.features.get("anchor_endpoint") else None,
        tuple(selected_candidate.features["tip_endpoint"]) if selected_candidate.features.get("tip_endpoint") else None,
    )
    save_mask_png(output_dir / "best_mask.png", selected_candidate.processed_mask)
    save_overlay(output_dir / "best_overlay.png", best_overlay)
    write_json(
        output_dir / "best_features.json",
        {
            "resize_metadata": resize_metadata.to_json(),
            "geometry_coordinate_space": resize_metadata.exported_geometry_coordinate_space,
            "features": selected_candidate.features,
            "working_resolution_features": selected_candidate.working_features,
        },
    )
    write_json(
        output_dir / "centerline.json",
        {
            "resize_metadata": resize_metadata.to_json(),
            "geometry_coordinate_space": resize_metadata.exported_geometry_coordinate_space,
            "centerline_points": selected_candidate.features.get("centerline_points", []),
            "anchor_endpoint": selected_candidate.features.get("anchor_endpoint"),
            "tip_endpoint": selected_candidate.features.get("tip_endpoint"),
        },
    )
    write_json(
        output_dir / "tracking_points.json",
        {
            "resize_metadata": resize_metadata.to_json(),
            "geometry_coordinate_space": resize_metadata.exported_geometry_coordinate_space,
            "anchor_point": selected_candidate.features.get("anchor_endpoint"),
            "tip_point": selected_candidate.features.get("tip_endpoint"),
            "axis_vector": selected_candidate.features.get("axis_vector"),
            "board_length_px": selected_candidate.features.get("length_estimate_px"),
            "tracking_points": selected_candidate.features.get("tracking_points", {}),
            "tracking_point_records": selected_candidate.features.get("tracking_point_records", []),
            "observation_state": selected_candidate.features.get("observation_state", {}),
        },
    )
    if export_rle:
        write_json(output_dir / "best_mask_rle.json", encode_rle(selected_candidate.processed_mask))
    if export_coco_json:
        write_json(
            output_dir / "best_mask_coco.json",
            export_coco_annotation(image_path=image_path, mask=selected_candidate.processed_mask),
        )

    summary = {
        "image_path": image_path,
        "model_type": model_type,
        "checkpoint_path": checkpoint_path,
        "device": device,
        "resize_metadata": resize_metadata.to_json(),
        "prompts": prompts.to_json(),
        "candidate_ranking": [
            {"index": candidate.index, "combined_score": candidate.combined_score}
            for candidate in ranked
        ],
        "selected_candidate": selected_candidate.index,
        "roi": list(roi) if roi else None,
        "working_roi": list(effective_roi) if effective_roi else None,
        "per_candidate_features": [
            {
                "index": candidate.index,
                "features": candidate.features,
                "working_resolution_features": candidate.working_features,
                "scoring_terms": candidate.scoring_terms,
                "combined_score": candidate.combined_score,
                "sam_score": candidate.sam_score,
            }
            for candidate in candidates
        ],
        "final_geometry": selected_candidate.features,
        "working_final_geometry": selected_candidate.working_features,
        "tracking_points": selected_candidate.features.get("tracking_points", {}),
        "observation_state": selected_candidate.features.get("observation_state", {}),
    }
    write_json(output_dir / "run_summary.json", summary)
    return RunArtifacts(output_dir=output_dir, selected_index=selected_candidate.index, candidates=candidates, summary=summary)
