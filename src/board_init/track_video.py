from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from .tracking import TrackingConfig, load_tracking_points, track_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track the visible springboard segment through a video.")
    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument("--init-tracking-json", required=True, help="Path to tracking_points.json from board initialization.")
    parser.add_argument("--output-dir", required=True, help="Directory for tracking artifacts.")
    parser.add_argument("--start-frame", type=int, default=0, help="Inclusive start frame.")
    parser.add_argument("--end-frame", type=int, help="Inclusive end frame.")
    parser.add_argument("--max-side", type=int, help="Resize the longest video side to this many pixels for tracking.")
    parser.add_argument("--max-error", type=float, default=25.0, help="Maximum LK optical flow error before a point is rejected.")
    parser.add_argument("--no-overlay-video", action="store_true", help="Skip writing tracking_overlay.mp4.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    payload = load_tracking_points(args.init_tracking_json)
    config = TrackingConfig(max_error=args.max_error, save_overlay_video=not args.no_overlay_video, max_side=args.max_side)
    summary = track_video(
        video_path=Path(args.video),
        tracking_points_payload=payload,
        output_dir=Path(args.output_dir),
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        config=config,
    )
    logging.getLogger("board_init.track_video").info("Tracked %d frames into %s", summary["frame_count"], args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
