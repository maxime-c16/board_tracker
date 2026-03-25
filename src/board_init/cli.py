from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Sequence

from .config import load_config
from .headless import prompts_from_cli
from .interactive import collect_prompts_gui, collect_roi_gui, detect_gui_support, select_candidate_gui
from .io import ensure_dir, load_image, load_prompts_json, write_json
from .pipeline import run_pipeline
from .resize import build_resize_metadata, crop_image, resize_image_for_sam, roi_to_working
from .sam_wrapper import SamModelWrapper, resolve_device
from .types import PromptSet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline springboard board initialization with Meta SAM.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--checkpoint", required=True, help="Path to the Meta SAM checkpoint.")
    parser.add_argument("--model-type", required=True, choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--gui", action="store_true", help="Force native OpenCV GUI mode.")
    parser.add_argument("--headless", action="store_true", help="Force headless mode.")
    parser.add_argument("--positive", action="append", default=[], help="Positive click as x,y in original-image pixels. Repeatable.")
    parser.add_argument("--negative", action="append", default=[], help="Negative click as x,y in original-image pixels. Repeatable.")
    parser.add_argument("--prompts-json", help="Path to a JSON file containing positive/negative prompts.")
    parser.add_argument("--select-mask-index", type=int, help="Finalize this candidate index after ranking.")
    parser.add_argument("--output-dir", help="Output directory. Defaults to outputs/<timestamp>_<image_stem>.")
    parser.add_argument("--roi", help="Optional ROI as x1,y1,x2,y2 in original-image pixels.")
    parser.add_argument("--config", help="Path to YAML config.")
    parser.add_argument("--save-all-candidates", action="store_true", help="Override config to save all candidates.")
    parser.add_argument("--export-rle", action="store_true", help="Export final mask as COCO-style RLE JSON.")
    parser.add_argument("--export-coco-json", action="store_true", help="Export final mask as minimal COCO JSON.")
    parser.add_argument("--crop-pad", type=int, help="Override auto ROI crop padding.")
    parser.add_argument("--max-side", type=int, help="Resize the longest image side to this many pixels before SAM inference.")
    parser.add_argument("--debug", action="store_true", help="Save extra debug artifacts.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser


def parse_roi(value: str | None) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    x1, y1, x2, y2 = [int(item) for item in value.split(",")]
    return x1, y1, x2, y2


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


def build_output_dir(image_path: Path, requested: str | None) -> Path:
    if requested:
        return ensure_dir(requested)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(Path("outputs") / f"{timestamp}_{image_path.stem}")


def collect_prompts(args: argparse.Namespace) -> tuple[PromptSet, int | None, str]:
    prompts = prompts_from_cli(args.positive, args.negative)
    select_mask_index = args.select_mask_index
    source = "cli"
    if args.prompts_json:
        json_prompts, json_selection = load_prompts_json(args.prompts_json)
        if json_prompts.positive or json_prompts.negative:
            prompts = json_prompts
            source = "json"
        if json_selection is not None and select_mask_index is None:
            select_mask_index = json_selection
    return prompts, select_mask_index, source


def _format_prompt_args(points: list[tuple[int, int]], flag: str) -> str:
    if not points:
        return ""
    return " ".join(f"{flag} {x},{y}" for x, y in points)


def log_prompt_repro(
    logger: logging.Logger,
    prompts: PromptSet,
    image_path: Path,
    roi: tuple[int, int, int, int] | None = None,
) -> None:
    positive_args = _format_prompt_args(prompts.positive, "--positive")
    negative_args = _format_prompt_args(prompts.negative, "--negative")
    logger.info("Positive prompts (original-image pixels): %s", prompts.positive)
    logger.info("Negative prompts (original-image pixels): %s", prompts.negative)
    command_parts = [
        "python -m board_init",
        f"--image {image_path}",
    ]
    if roi is not None:
        command_parts.append(f"--roi {roi[0]},{roi[1]},{roi[2]},{roi[3]}")
    if positive_args:
        command_parts.append(positive_args)
    if negative_args:
        command_parts.append(negative_args)
    logger.info("Headless prompt replay: %s", " ".join(command_parts))


def log_roi_repro(logger: logging.Logger, roi: tuple[int, int, int, int] | None) -> None:
    if roi is None:
        return
    logger.info("ROI (original-image pixels): %s", roi)
    logger.info("Headless ROI replay: --roi %d,%d,%d,%d", roi[0], roi[1], roi[2], roi[3])


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    logger = logging.getLogger("board_init")

    if args.gui and args.headless:
        parser.error("Choose only one of --gui or --headless.")
    if args.max_side is not None and args.max_side <= 0:
        parser.error("--max-side must be a positive integer.")

    config = load_config(args.config or "configs/default.yaml")
    if args.save_all_candidates:
        config.runtime.save_all_candidates = True
    if args.debug:
        config.runtime.debug = True
    if args.crop_pad is not None:
        config.postprocess.crop_pad = args.crop_pad

    image_path = Path(args.image)
    image_rgb = load_image(image_path)
    resize_metadata = build_resize_metadata(image_rgb.shape, args.max_side)
    working_image_rgb = resize_image_for_sam(image_rgb, resize_metadata)

    output_dir = build_output_dir(image_path, args.output_dir)
    roi = parse_roi(args.roi)
    prompts, select_mask_index, prompt_source = collect_prompts(args)
    gui_diagnostics = detect_gui_support()

    logger.info("GUI backend available=%s", gui_diagnostics.backend_available)
    logger.info("GUI diagnostics=%s", gui_diagnostics.diagnostic_message)
    logger.info("Resolved device=%s", resolve_device(args.device))
    logger.info(
        "Image size original=%sx%s working=%sx%s resize_scale=%.6f",
        resize_metadata.original_width,
        resize_metadata.original_height,
        resize_metadata.working_width,
        resize_metadata.working_height,
        resize_metadata.resize_scale,
    )

    mode = "headless"
    if args.gui:
        mode = "gui"
    elif not args.headless and gui_diagnostics.backend_available and gui_diagnostics.local_session_likely and not prompts.positive:
        mode = "gui"

    if mode == "gui" and not gui_diagnostics.backend_available:
        logger.warning("GUI requested but unavailable: %s", gui_diagnostics.diagnostic_message)
        logger.warning("Falling back to headless mode. Use --positive x,y or --prompts-json FILE.")
        mode = "headless"

    if mode == "gui" and not prompts.positive:
        try:
            if roi is None:
                roi = collect_roi_gui(working_image_rgb, resize_metadata=resize_metadata)
            working_roi = roi_to_working(roi, resize_metadata)
            prompt_view = crop_image(working_image_rgb, working_roi)
            prompt_view_origin = (working_roi[0], working_roi[1]) if working_roi is not None else (0, 0)
            selection = collect_prompts_gui(
                prompt_view,
                resize_metadata=resize_metadata,
                view_origin_working=prompt_view_origin,
            )
        except RuntimeError as exc:
            logger.warning("GUI prompt collection failed: %s", exc)
            logger.warning("Falling back to headless mode. Supply --positive x,y or --prompts-json FILE.")
            mode = "headless"
        else:
            prompts = selection.prompts
            prompt_source = "gui"

    if mode == "headless" and not prompts.positive:
        parser.error(
            "No positive prompts available. Use --positive x,y, --prompts-json FILE, or --gui when local OpenCV GUI is available."
        )

    log_roi_repro(logger, roi)
    log_prompt_repro(logger, prompts, image_path, roi=roi)

    predictor = SamModelWrapper(
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        device=args.device,
    )
    artifacts = run_pipeline(
        image_rgb=image_rgb,
        working_image_rgb=working_image_rgb,
        image_path=str(image_path),
        prompts=prompts,
        predictor=predictor,
        config=config,
        output_dir=output_dir,
        resize_metadata=resize_metadata,
        model_type=args.model_type,
        checkpoint_path=str(Path(args.checkpoint)),
        device=resolve_device(args.device),
        roi=roi,
        select_mask_index=select_mask_index,
        export_rle=args.export_rle,
        export_coco_json=args.export_coco_json,
    )

    if mode == "gui" and select_mask_index is None:
        try:
            selected_idx = select_candidate_gui(
                image_rgb=image_rgb,
                prompts=prompts,
                candidate_masks=[candidate.processed_mask for candidate in artifacts.candidates],
            )
        except RuntimeError as exc:
            logger.warning("GUI candidate review failed: %s", exc)
        else:
            rerun_dir = output_dir / "selection_finalize"
            run_pipeline(
                image_rgb=image_rgb,
                working_image_rgb=working_image_rgb,
                image_path=str(image_path),
                prompts=prompts,
                predictor=predictor,
                config=config,
                output_dir=rerun_dir,
                resize_metadata=resize_metadata,
                model_type=args.model_type,
                checkpoint_path=str(Path(args.checkpoint)),
                device=resolve_device(args.device),
                roi=roi,
                select_mask_index=selected_idx,
                export_rle=args.export_rle,
                export_coco_json=args.export_coco_json,
            )
            artifacts.selected_index = selected_idx
            logger.info("Finalized candidate %d in %s", selected_idx, rerun_dir)

    write_json(
        output_dir / "session_metadata.json",
        {
            "mode": mode,
            "prompt_source": prompt_source,
            "device": resolve_device(args.device),
            "model_type": args.model_type,
            "checkpoint": str(Path(args.checkpoint)),
            "resize_metadata": resize_metadata.to_json(),
            "gui": {
                "backend_available": gui_diagnostics.backend_available,
                "backend_name": gui_diagnostics.backend_name,
                "platform_name": gui_diagnostics.platform_name,
                "display_env": gui_diagnostics.display_env,
                "ssh_connection": gui_diagnostics.ssh_connection,
                "local_session_likely": gui_diagnostics.local_session_likely,
                "diagnostic_message": gui_diagnostics.diagnostic_message,
            },
        },
    )
    logger.info("Selected candidate index: %d", artifacts.selected_index)
    logger.info("Artifacts written to %s", output_dir)
    return 0
