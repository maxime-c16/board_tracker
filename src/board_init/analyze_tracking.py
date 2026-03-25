from __future__ import annotations

import argparse
import logging
from typing import Sequence

from .oscillation import analyze_tracking_oscillation, load_tracking_records, save_oscillation_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze visible-segment board tracking and extract oscillation patterns.")
    parser.add_argument("--tracking-jsonl", required=True, help="Path to tracked_points.jsonl.")
    parser.add_argument("--video", required=True, help="Source video path for validation frame export.")
    parser.add_argument("--output-dir", required=True, help="Output directory for oscillation artifacts.")
    parser.add_argument("--fps", type=float, default=59.81308411214953, help="Video frame rate.")
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
    records = load_tracking_records(args.tracking_jsonl)
    analysis = analyze_tracking_oscillation(records, fps=args.fps)
    save_oscillation_artifacts(args.output_dir, args.video, args.tracking_jsonl, analysis)
    logging.getLogger("board_init.analyze_tracking").info("Oscillation artifacts written to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
