# Handoff

This file is for restarting work quickly in a fresh Codex session after cloning the repo.

## Repo

- Project root: `/Volumes/DiveRecorderGPT/cv_diving`
- Remote: `git@github.com:maxime-c16/board_tracker.git`
- Primary branch used in this work: `master`

## Current Goal

The project is no longer centered on Debian/X11. The primary path is now:

1. local macOS terminal
2. native OpenCV GUI for board initialization
3. offline classic Meta SAM inference
4. export a board-state representation suitable for downstream tracking
5. track the visible board segment through video
6. extract an oscillation pattern over time

The working assumption for most data is:

- no visible fulcrum
- no visible true anchor
- only the visible board segment is available

So the tracking model is centered on visible shape, visible endpoints, centerline stations, and oscillation, not on full physical-board reconstruction from a single frame.

## Environment

Use the repo venv:

```bash
cd /Volumes/DiveRecorderGPT/cv_diving
source .venv/bin/activate
```

If the venv needs to be recreated:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install torch torchvision
python -m pip install -e .
```

Checkpoint path used during development:

```text
/Volumes/DiveRecorderGPT/cv_diving/models/sam/sam_vit_b_01ec64.pth
```

## Important Files

- CLI init path: [src/board_init/cli.py](/Volumes/DiveRecorderGPT/cv_diving/src/board_init/cli.py)
- GUI interaction: [src/board_init/interactive.py](/Volumes/DiveRecorderGPT/cv_diving/src/board_init/interactive.py)
- SAM wrapper: [src/board_init/sam_wrapper.py](/Volumes/DiveRecorderGPT/cv_diving/src/board_init/sam_wrapper.py)
- Pipeline: [src/board_init/pipeline.py](/Volumes/DiveRecorderGPT/cv_diving/src/board_init/pipeline.py)
- Geometry/model export: [src/board_init/geometry.py](/Volumes/DiveRecorderGPT/cv_diving/src/board_init/geometry.py)
- Scoring: [src/board_init/scoring.py](/Volumes/DiveRecorderGPT/cv_diving/src/board_init/scoring.py)
- Rectification/postprocess: [src/board_init/postprocess.py](/Volumes/DiveRecorderGPT/cv_diving/src/board_init/postprocess.py)
- Video tracking: [src/board_init/track_video.py](/Volumes/DiveRecorderGPT/cv_diving/src/board_init/track_video.py)
- Oscillation analysis: [src/board_init/analyze_tracking.py](/Volumes/DiveRecorderGPT/cv_diving/src/board_init/analyze_tracking.py)
- Config: [configs/default.yaml](/Volumes/DiveRecorderGPT/cv_diving/configs/default.yaml)
- Main docs: [README.md](/Volumes/DiveRecorderGPT/cv_diving/README.md)

## What Has Already Been Implemented

- macOS-native OpenCV GUI path is the primary happy path
- headless mode remains supported
- GUI capability diagnostics for local macOS sessions
- built-in resize support with `--max-side`
- explicit original-image vs SAM-working-image coordinate metadata
- ROI-first prompting flow in GUI mode
- ROI-aware cropped SAM inference
- candidate replay logging for GUI-collected prompts
- board-specific scoring improvements to penalize merged/branching masks
- geometric rectification of the selected board mask
- tracking-point export derived from the visible board segment
- video tracking over time
- oscillation extraction and validation-frame export

## Current Modeling Approach

Do not assume the true anchor or fulcrum is visible.

The tracked object model is a visible-segment board state:

- visible proximal endpoint
- visible distal endpoint
- centerline
- normalized centerline stations
- board axis
- visible segment length

This is derived from the final rectified initialization mask and exported as tracking metadata.

Downstream, the most useful points are:

- `s50`
- `s75`
- `s90`
- visible proximal endpoint
- visible distal endpoint when still in frame

The mask itself is a measurement source. The actual state to track is the extracted centerline/station geometry.

## Known Good Initialization Example

The following prompt set produced a good initialization for `IMG_8684.PNG`:

```text
ROI: (13, 1899, 521, 2033)
positive: (118, 1964), (355, 1982)
negative: (157, 2004)
```

Equivalent command:

```bash
python -m board_init \
  --image IMG_8684.PNG \
  --checkpoint /Volumes/DiveRecorderGPT/cv_diving/models/sam/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --headless \
  --roi 13,1899,521,2033 \
  --positive 118,1964 \
  --positive 355,1982 \
  --negative 157,2004 \
  --max-side 2048 \
  --device auto \
  --output-dir outputs/debug_user_best_prompt
```

Important caveat:

- the exported field currently called `anchor_point` may represent the visible proximal endpoint, not the physical fulcrum-side anchor, when the true anchor is out of frame

That semantic distinction matters if the next session changes naming or downstream assumptions.

## Known Good Video Tracking Flow

### 1. Initialize from an image or frame

Generate `tracking_points.json` from `board_init`.

### 2. Track through video

```bash
python -m board_init.track_video \
  --video IMG_8432.MOV \
  --init-tracking-json outputs/video_init_frame0_try1/tracking_points.json \
  --output-dir outputs/video_track_try1_full \
  --max-side 1280 \
  --verbose
```

Outputs include:

- `tracked_points.jsonl`
- `tracked_points.csv`
- `tracking_overlay.mp4` unless disabled
- `tracking_summary.json`

### 3. Analyze oscillation

```bash
python -m board_init.analyze_tracking \
  --tracking-jsonl outputs/video_track_try1_full/tracked_points.jsonl \
  --video IMG_8432.MOV \
  --output-dir outputs/video_track_try1_full/oscillation \
  --verbose
```

Outputs include:

- `oscillation_summary.json`
- `oscillation_series.json`
- `oscillation_plot.png`
- sample validation frames near peaks/troughs

## Current Oscillation Result

Artifacts:

- [outputs/video_track_try1_full/oscillation/oscillation_summary.json](/Volumes/DiveRecorderGPT/cv_diving/outputs/video_track_try1_full/oscillation/oscillation_summary.json)
- [outputs/video_track_try1_full/oscillation/oscillation_series.json](/Volumes/DiveRecorderGPT/cv_diving/outputs/video_track_try1_full/oscillation/oscillation_series.json)
- [outputs/video_track_try1_full/oscillation/oscillation_plot.png](/Volumes/DiveRecorderGPT/cv_diving/outputs/video_track_try1_full/oscillation/oscillation_plot.png)

Summary:

- valid tracked frames: `1117`
- strongest oscillation window: frame `783` to frame `902`
- dominant-window peaks: `808, 824, 875, 894`
- dominant-window troughs: `788, 812, 854, 882`

Validation frames exported:

- [sample_frame_0788.png](/Volumes/DiveRecorderGPT/cv_diving/outputs/video_track_try1_full/oscillation/sample_frame_0788.png)
- [sample_frame_0808.png](/Volumes/DiveRecorderGPT/cv_diving/outputs/video_track_try1_full/oscillation/sample_frame_0808.png)
- [sample_frame_0812.png](/Volumes/DiveRecorderGPT/cv_diving/outputs/video_track_try1_full/oscillation/sample_frame_0812.png)
- [sample_frame_0824.png](/Volumes/DiveRecorderGPT/cv_diving/outputs/video_track_try1_full/oscillation/sample_frame_0824.png)
- [sample_frame_0854.png](/Volumes/DiveRecorderGPT/cv_diving/outputs/video_track_try1_full/oscillation/sample_frame_0854.png)
- [sample_frame_0875.png](/Volumes/DiveRecorderGPT/cv_diving/outputs/video_track_try1_full/oscillation/sample_frame_0875.png)

These were used as a qualitative cross-check that the tracker follows a visible oscillation pattern rather than random drift.

## Validation Commands

Before continuing work, rerun:

```bash
./.venv/bin/python -m pytest
./.venv/bin/python -m board_init.track_video --help
./.venv/bin/python -m board_init.analyze_tracking --help
```

Expected state at the time of writing:

- tests passing: `17 passed`

## Git Notes

Current pushed commit before this handoff update:

- `b1416ed` `Build board init and visible segment tracking workflow`

The repo now includes:

- `.gitignore`
- `outputs/.gitkeep`
- `models/sam/.gitkeep`

and ignores:

- `.venv/`
- generated `outputs/*`
- raw media files
- local checkpoints under `models/sam/*`

## Best Next Technical Steps

If continuing development, the highest-value next steps are:

1. rename semantic exports so `anchor_point` is not confused with the true physical anchor when the fulcrum is out of frame
2. export a clearer `visible_segment_state.json`
3. add confidence values per exported tracking point
4. improve re-detection when a point exits frame or is briefly occluded
5. fit a low-order board shape model over normalized stations so oscillation is represented as deformation, not just point motion
6. add quantitative validation against manually reviewed sample frames

## Notes For A New Codex Session

Start by reading:

1. [README.md](/Volumes/DiveRecorderGPT/cv_diving/README.md)
2. [HANDOFF.md](/Volumes/DiveRecorderGPT/cv_diving/HANDOFF.md)
3. [src/board_init/track_video.py](/Volumes/DiveRecorderGPT/cv_diving/src/board_init/track_video.py)
4. [src/board_init/analyze_tracking.py](/Volumes/DiveRecorderGPT/cv_diving/src/board_init/analyze_tracking.py)

Then inspect these output artifacts:

1. [outputs/video_track_try1_full/tracking_summary.json](/Volumes/DiveRecorderGPT/cv_diving/outputs/video_track_try1_full/tracking_summary.json)
2. [outputs/video_track_try1_full/oscillation/oscillation_summary.json](/Volumes/DiveRecorderGPT/cv_diving/outputs/video_track_try1_full/oscillation/oscillation_summary.json)
3. [outputs/video_track_try1_full/oscillation/oscillation_plot.png](/Volumes/DiveRecorderGPT/cv_diving/outputs/video_track_try1_full/oscillation/oscillation_plot.png)

Do not assume the hidden anchor is recoverable from a single frame. The current code is intentionally built around visible-segment tracking.
