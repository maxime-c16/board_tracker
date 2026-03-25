# Handoff

This file is for restarting work quickly in a fresh Codex session after cloning the repo.

## Repo

- Project root: `/home/macauchy/board_tracker`
- Remote: `git@github.com:maxime-c16/board_tracker.git`
- Primary branch used in this work: `master`

## Current Goal

The project is no longer centered on Debian/X11. The historical primary path was:

1. local terminal session
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

The current active engineering goal is to replace the old sparse LK-point tracker with a board-locked tracker that:

- searches inside a local ROI around the last trusted board position
- fits a board centerline / long thin segment each frame from image support
- exports explicit tracking states: `tracked`, `recovered`, `predicted`, `invalid`
- writes per-frame quality metrics
- feeds oscillation only from validated board-supported frames

## Environment

Use the repo venv:

```bash
cd /home/macauchy/board_tracker
source .venv/bin/activate
```

If the venv needs to be recreated:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-dev.txt
python -m pip install torch torchvision
python -m pip install -e .
```

On this machine:

- OS/session assumption: Linux, not macOS
- safest validation path: headless CLI and tests
- GUI should be treated as optional and only used if OpenCV has a working local display backend

macOS reproduction note:

- this branch was also validated on `/Volumes/DiveRecorderGPT/cv_diving`
- the tracker and oscillation pipeline run on macOS
- for long runs on macOS, prefer `--no-overlay-video`
- the board-locked tracker completed cleanly on macOS without overlay video
- overlay-video finalization for long runs may stall after writing the `.mp4`

Checkpoint path used during development:

```text
/home/macauchy/board_tracker/models/sam/sam_vit_b_01ec64.pth
```

## Important Files

- CLI init path: [src/board_init/cli.py](/home/macauchy/board_tracker/src/board_init/cli.py)
- GUI interaction: [src/board_init/interactive.py](/home/macauchy/board_tracker/src/board_init/interactive.py)
- SAM wrapper: [src/board_init/sam_wrapper.py](/home/macauchy/board_tracker/src/board_init/sam_wrapper.py)
- Pipeline: [src/board_init/pipeline.py](/home/macauchy/board_tracker/src/board_init/pipeline.py)
- Geometry/model export: [src/board_init/geometry.py](/home/macauchy/board_tracker/src/board_init/geometry.py)
- Scoring: [src/board_init/scoring.py](/home/macauchy/board_tracker/src/board_init/scoring.py)
- Rectification/postprocess: [src/board_init/postprocess.py](/home/macauchy/board_tracker/src/board_init/postprocess.py)
- Video tracking: [src/board_init/track_video.py](/home/macauchy/board_tracker/src/board_init/track_video.py)
- Oscillation analysis: [src/board_init/analyze_tracking.py](/home/macauchy/board_tracker/src/board_init/analyze_tracking.py)
- Config: [configs/default.yaml](/home/macauchy/board_tracker/configs/default.yaml)
- Main docs: [README.md](/home/macauchy/board_tracker/README.md)

## What Has Already Been Implemented

- native OpenCV GUI remains supported when a local backend is available
- headless mode remains supported
- GUI capability diagnostics for local sessions
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
- board-locked tracking prototype in [src/board_init/tracking.py](/home/macauchy/board_tracker/src/board_init/tracking.py)
- per-frame quality export in `tracking_quality.json` and `tracking_quality.csv`
- confidence-colored overlay support in tracking records / overlay rendering
- oscillation filtering to trusted tracking states only in [src/board_init/oscillation.py](/home/macauchy/board_tracker/src/board_init/oscillation.py)
- short-gap interpolation and robust detrending for oscillation analysis
- validation contact sheet generation for oscillation extrema

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
  --checkpoint /home/macauchy/board_tracker/models/sam/sam_vit_b_01ec64.pth \
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
- `tracking_quality.json`
- `tracking_quality.csv`
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
- `contact_sheet.png`

## Current Replay Baseline

The most useful initialization on this machine is the replayed prompt set for frame 0 of `IMG_8432.MOV`:

```text
ROI: (0, 1458, 448, 1609)
positive: (320, 1557), (101, 1540)
negative: (142, 1575)
```

Equivalent headless command:

```bash
./.venv/bin/python -m board_init \
  --image outputs/video_frame0.png \
  --checkpoint /home/macauchy/board_tracker/models/sam/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --headless \
  --roi 0,1458,448,1609 \
  --positive 320,1557 \
  --positive 101,1540 \
  --negative 142,1575 \
  --max-side 1080 \
  --device auto \
  --output-dir outputs/video_init_frame0_gui_replay \
  --verbose
```

That initialization writes:

- [outputs/video_init_frame0_gui_replay/best_overlay.png](/home/macauchy/board_tracker/outputs/video_init_frame0_gui_replay/best_overlay.png)
- [outputs/video_init_frame0_gui_replay/tracking_points.json](/home/macauchy/board_tracker/outputs/video_init_frame0_gui_replay/tracking_points.json)

macOS reproduction of that baseline:

```bash
./.venv/bin/python -m board_init \
  --image outputs/video_frame0.png \
  --checkpoint /Volumes/DiveRecorderGPT/cv_diving/models/sam/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --headless \
  --roi 0,1458,448,1609 \
  --positive 320,1557 \
  --positive 101,1540 \
  --negative 142,1575 \
  --max-side 1080 \
  --device auto \
  --output-dir outputs/video_init_frame0_gui_replay_mac \
  --verbose
```

Artifacts written on macOS:

- [outputs/video_init_frame0_gui_replay_mac/best_overlay.png](/Volumes/DiveRecorderGPT/cv_diving/outputs/video_init_frame0_gui_replay_mac/best_overlay.png)
- [outputs/video_init_frame0_gui_replay_mac/tracking_points.json](/Volumes/DiveRecorderGPT/cv_diving/outputs/video_init_frame0_gui_replay_mac/tracking_points.json)

## Current Tracking Rework Status

The old production path was sparse LK tracking plus affine fill-in. That is no longer the intended direction.

The current `tracking.py` now does this per frame:

1. optical-flow prediction for continuity
2. board search in a local ROI around the last trusted board position
3. edge-supported long-thin board fit
4. stable-length segment projection back to `anchor`, `s25`, `s50`, `s75`, `s90`, `tip`
5. per-frame quality scoring and state assignment

Tracking states now mean:

- `tracked`: board fit passed and had enough direct flow support
- `recovered`: board fit passed but direct flow support was weaker
- `predicted`: motion-only fallback, not trusted for oscillation
- `invalid`: no acceptable board-supported geometry and no safe fallback

Quality artifacts now include:

## macOS Reproduction Status

The Linux branch was checked on macOS against the local `IMG_8432.MOV`.

Successful macOS tracking command:

```bash
./.venv/bin/python -m board_init.track_video \
  --video IMG_8432.MOV \
  --init-tracking-json outputs/video_init_frame0_gui_replay_mac/tracking_points.json \
  --output-dir outputs/mac_branch_repro_track_replay \
  --max-side 1080 \
  --no-overlay-video \
  --verbose
```

That run completed with:

- `1117` tracked frames total
- tracking summary written to [outputs/mac_branch_repro_track_replay/tracking_summary.json](/Volumes/DiveRecorderGPT/cv_diving/outputs/mac_branch_repro_track_replay/tracking_summary.json)

State counts on macOS for that replay baseline:

- `seed`: `1`
- `tracked`: `46`
- `recovered`: `63`
- `predicted`: `603`
- `invalid`: `404`

Oscillation analysis command on macOS:

```bash
./.venv/bin/python -m board_init.analyze_tracking \
  --tracking-jsonl outputs/mac_branch_repro_track_replay/tracked_points.jsonl \
  --video IMG_8432.MOV \
  --output-dir outputs/mac_branch_repro_track_replay/oscillation \
  --verbose
```

macOS oscillation artifacts:

- [outputs/mac_branch_repro_track_replay/oscillation/oscillation_summary.json](/Volumes/DiveRecorderGPT/cv_diving/outputs/mac_branch_repro_track_replay/oscillation/oscillation_summary.json)
- [outputs/mac_branch_repro_track_replay/oscillation/oscillation_plot.png](/Volumes/DiveRecorderGPT/cv_diving/outputs/mac_branch_repro_track_replay/oscillation/oscillation_plot.png)
- [outputs/mac_branch_repro_track_replay/oscillation/contact_sheet.png](/Volumes/DiveRecorderGPT/cv_diving/outputs/mac_branch_repro_track_replay/oscillation/contact_sheet.png)

Current macOS oscillation summary for this branch:

- valid frames accepted by oscillation analysis: `109`
- dominant window: frame `20` to frame `1003`
- dominant-window peaks: `90, 106, 802`
- dominant-window troughs: `81, 289`

Interpretation:

- the branch reproduces on macOS in the sense that the board-locked tracker and oscillation analysis both run successfully
- the current branch baseline is still not strong enough to be called a robust production oscillation extraction on this video
- most frames are still landing in `predicted` or `invalid`, so the next technical work should focus on board-support recovery quality, not on platform portability

- board length / angle continuity
- centroid drift vs seed geometry
- centerline deviation
- edge inlier count
- support coverage along the board
- per-station image support score
- forward/backward flow consistency

## Latest Probe Result

The best quick validation so far is a replay-based probe on frames `0..60`:

```bash
./.venv/bin/python -m board_init.track_video \
  --video IMG_8432.MOV \
  --init-tracking-json outputs/video_init_frame0_gui_replay/tracking_points.json \
  --output-dir outputs/video_track_gui_replay_locked_probe9 \
  --max-side 1280 \
  --start-frame 0 \
  --end-frame 60 \
  --no-overlay-video \
  --verbose
```

Observed state counts from [outputs/video_track_gui_replay_locked_probe9/tracking_summary.json](/home/macauchy/board_tracker/outputs/video_track_gui_replay_locked_probe9/tracking_summary.json):

- `seed`: `1`
- `tracked`: `53`
- `predicted`: `7`

This is materially better than the previous prototype, which quickly collapsed into mostly `predicted`.

Important caveat:

- the full `0..1116` rerun was interrupted to switch into branch/doc prep work
- the full updated oscillation result is therefore not finalized yet on the new tracker branch state
- the next critical validation step is a full replay-based rerun into a fresh output directory, then visual inspection of the oscillation contact sheet

## Prior Oscillation Result

The older oscillation artifacts below were generated before the current board-locked rework and should be treated as historical only, not as the final trusted result for the new implementation:

Artifacts:

- [outputs/video_track_try1_full/oscillation/oscillation_summary.json](/home/macauchy/board_tracker/outputs/video_track_try1_full/oscillation/oscillation_summary.json)
- [outputs/video_track_try1_full/oscillation/oscillation_series.json](/home/macauchy/board_tracker/outputs/video_track_try1_full/oscillation/oscillation_series.json)
- [outputs/video_track_try1_full/oscillation/oscillation_plot.png](/home/macauchy/board_tracker/outputs/video_track_try1_full/oscillation/oscillation_plot.png)

Summary:

- valid tracked frames: `1117`
- strongest oscillation window: frame `783` to frame `902`
- dominant-window peaks: `808, 824, 875, 894`
- dominant-window troughs: `788, 812, 854, 882`

Validation frames exported:

- [sample_frame_0788.png](/home/macauchy/board_tracker/outputs/video_track_try1_full/oscillation/sample_frame_0788.png)
- [sample_frame_0808.png](/home/macauchy/board_tracker/outputs/video_track_try1_full/oscillation/sample_frame_0808.png)
- [sample_frame_0812.png](/home/macauchy/board_tracker/outputs/video_track_try1_full/oscillation/sample_frame_0812.png)
- [sample_frame_0824.png](/home/macauchy/board_tracker/outputs/video_track_try1_full/oscillation/sample_frame_0824.png)
- [sample_frame_0854.png](/home/macauchy/board_tracker/outputs/video_track_try1_full/oscillation/sample_frame_0854.png)
- [sample_frame_0875.png](/home/macauchy/board_tracker/outputs/video_track_try1_full/oscillation/sample_frame_0875.png)

These were used as a qualitative cross-check that the tracker follows a visible oscillation pattern rather than random drift.

## Validation Commands

Before continuing work, rerun:

```bash
./.venv/bin/python -m pytest
./.venv/bin/python -m board_init.track_video --help
./.venv/bin/python -m board_init.analyze_tracking --help
```

Expected state at the time of writing:

- tests passing: `20 passed`

## Git Notes

Current pushed commit before this handoff update:

- `b1416ed` `Build board init and visible segment tracking workflow`

Current working branch during the rework:

- `master`

Recommended next git step before pushing:

```bash
git switch -c feat/board-locked-tracking
git add HANDOFF.md README.md src/board_init/tracking.py src/board_init/oscillation.py tests/test_tracking.py tests/test_oscillation.py scripts/bootstrap.sh scripts/bootstrap_macos.sh
git commit -m "Implement board-locked tracking quality pipeline"
git push -u origin feat/board-locked-tracking
```

Note:

- there are also untracked local files `cmd` and `out` in the worktree at the time of this handoff
- inspect them before staging anything broad like `git add .`

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

1. run the full replay-based rerun with the new tracker into a fresh output directory
2. inspect `tracking_quality.json`, selected validation frames, and the oscillation `contact_sheet.png`
3. verify that tracked stations remain on the board during the dominant oscillation window
4. tune ROI search / support scoring if the full run still falls back into too many `predicted` or `invalid` frames
5. only after visual validation, treat the new oscillation graph as trustworthy
6. optionally rename semantic exports so `anchor_point` is not confused with the physical fulcrum-side anchor when that anchor is out of frame

## Notes For A New Codex Session

Start by reading:

1. [README.md](/home/macauchy/board_tracker/README.md)
2. [HANDOFF.md](/home/macauchy/board_tracker/HANDOFF.md)
3. [src/board_init/track_video.py](/home/macauchy/board_tracker/src/board_init/track_video.py)
4. [src/board_init/analyze_tracking.py](/home/macauchy/board_tracker/src/board_init/analyze_tracking.py)

Then inspect these output artifacts:

1. [outputs/video_track_try1_full/tracking_summary.json](/home/macauchy/board_tracker/outputs/video_track_try1_full/tracking_summary.json)
2. [outputs/video_track_try1_full/oscillation/oscillation_summary.json](/home/macauchy/board_tracker/outputs/video_track_try1_full/oscillation/oscillation_summary.json)
3. [outputs/video_track_try1_full/oscillation/oscillation_plot.png](/home/macauchy/board_tracker/outputs/video_track_try1_full/oscillation/oscillation_plot.png)

Do not assume the hidden anchor is recoverable from a single frame. The current code is intentionally built around visible-segment tracking.
