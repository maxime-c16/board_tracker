# Board Init

Offline springboard-board initialization and mask extraction for diving analysis using classic Meta Segment Anything with the official `segment-anything` package and `SamPredictor`.

The original primary workflow was local macOS use with a native OpenCV window. On this machine, assume a Linux environment under `/home/macauchy/board_tracker`; headless mode is the safest default unless you have a working local OpenCV GUI session.

## Project Layout

```text
.
├── configs/
├── examples/
├── models/
├── outputs/
├── scripts/
├── src/
└── tests/
```

## Setup

Create and activate a local virtual environment:

```bash
cd /home/macauchy/board_tracker
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

Helper script on this machine:

```bash
./scripts/bootstrap.sh
source .venv/bin/activate
```

The macOS helper script, `./scripts/bootstrap_macos.sh`, remains available for local Cocoa-backed setups.

## PyTorch

`segment-anything` needs PyTorch, but the right wheel depends on your machine.

Linux CPU example:

```bash
python -m pip install torch torchvision
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Linux CUDA example:

```bash
python -m pip install torch torchvision
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

macOS Apple Silicon / Metal (`mps`) example:

```bash
python -m pip install torch torchvision
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

Intel Mac CPU example:

```bash
python -m pip install torch torchvision
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The CLI accepts `--device auto|cpu|cuda|mps`. `auto` prefers `cuda`, then `mps`, then `cpu`.

## SAM Checkpoint

Recommended local checkpoint path:

```text
/home/macauchy/board_tracker/models/sam/sam_vit_b_01ec64.pth
```

The expected local command path is therefore:

```bash
python -m board_init \
  --image /path/to/image.jpg \
  --checkpoint /home/macauchy/board_tracker/models/sam/sam_vit_b_01ec64.pth \
  --model-type vit_b
```

## Coordinate Conventions

The project now uses explicit coordinate spaces:

- Prompt input from CLI or JSON is always `original_image_pixels`.
- GUI clicks are collected on the working SAM image, then mapped back to `original_image_pixels`.
- SAM inference runs in `sam_working_image_pixels`.
- Exported geometry, saved overlays, saved masks, and centerline JSON use `original_image_pixels`.

When `--max-side` is set, outputs include:

- `original_image_size`
- `sam_working_image_size`
- `resize_scale`
- `prompt_coordinate_space`
- `sam_prompt_coordinate_space`
- `exported_geometry_coordinate_space`

These fields are written into `prompts.json`, `best_features.json`, `candidate_*_features.json`, `centerline.json`, `run_summary.json`, and `session_metadata.json`.

## Local GUI Workflow

Native OpenCV GUI mode when a local display backend is available:

```bash
python -m board_init \
  --image /path/to/image.jpg \
  --checkpoint /home/macauchy/board_tracker/models/sam/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --device auto \
  --gui \
  --max-side 2048 \
  --verbose
```

Interactive controls:

- ROI window first when `--roi` is not supplied in GUI mode:
  - drag left mouse to draw ROI
  - `Enter` accepts ROI
  - `s` skips ROI cropping
  - `c` clears ROI
  - `q` cancels
- Left click: add positive point
- Right click: add negative point
- `c`: clear prompts
- `Enter` or `r`: run prediction
- Candidate review: `n` next, `p` previous, `Enter` or `a` accept, `q` cancel

If `--gui` is omitted, the tool may still choose GUI mode automatically when it detects a likely local GUI session. On this Linux machine, do not assume that path is available unless OpenCV has a usable backend and your session has display access.

If you accept an ROI in GUI mode, SAM inference runs only on that cropped working region and masks are mapped back to the original image for export.

## Headless Workflow

Headless CLI prompts:

```bash
python -m board_init \
  --image /path/to/image.jpg \
  --checkpoint /home/macauchy/board_tracker/models/sam/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --headless \
  --max-side 2048 \
  --positive 840,420 \
  --negative 770,520 \
  --negative 900,300
```

Headless JSON prompts:

```bash
python -m board_init \
  --image /path/to/image.jpg \
  --checkpoint /home/macauchy/board_tracker/models/sam/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --headless \
  --prompts-json examples/prompts.json \
  --max-side 2048
```

Finalize a reviewed candidate:

```bash
python -m board_init \
  --image /path/to/image.jpg \
  --checkpoint /home/macauchy/board_tracker/models/sam/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --headless \
  --prompts-json examples/prompts.json \
  --select-mask-index 1 \
  --max-side 2048
```

## Resize Support

`--max-side` downscales only for SAM inference. The project keeps the original image in memory and maps results back to the original resolution for export.

Example:

```bash
python -m board_init \
  --image /path/to/very_large_photo.jpg \
  --checkpoint /home/macauchy/board_tracker/models/sam/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --gui \
  --max-side 1600
```

This reduces SAM compute cost while preserving original-size exported masks and geometry.

## Outputs

Each run writes to:

```text
outputs/<timestamp>_<image_stem>/
```

Core artifacts:

- `input_copy.png`
- `prompts.json`
- `candidate_<n>_mask.png`
- `candidate_<n>_overlay.png`
- `candidate_<n>_features.json`
- `best_mask.png`
- `best_overlay.png`
- `best_features.json`
- `centerline.json`
- `run_summary.json`
- `session_metadata.json`

Optional artifacts:

- `best_mask_rle.json`
- `best_mask_coco.json`
- `candidate_<n>_raw_mask.png` when `--debug` is enabled

These remain suitable for downstream tracking initialization:

- final board mask
- candidate overlays
- candidate feature JSON
- selected feature JSON
- centerline export
- run summary
- resize metadata for original-to-working coordinate mapping

## Video Tracking

Track the visible board segment through a video using an existing `tracking_points.json` initialization:

```bash
python -m board_init.track_video \
  --video IMG_8432.MOV \
  --init-tracking-json outputs/video_init_frame0_try1/tracking_points.json \
  --output-dir outputs/video_track_try1_full \
  --max-side 720
```

On macOS, for long runs, prefer:

```bash
python -m board_init.track_video \
  --video IMG_8432.MOV \
  --init-tracking-json outputs/video_init_frame0_try1/tracking_points.json \
  --output-dir outputs/video_track_try1_full \
  --max-side 720 \
  --no-overlay-video
```

The board-locked tracker itself is validated on macOS, but `tracking_overlay.mp4` finalization can stall on long runs. `--no-overlay-video` is the reliable path for full-video tracking on the Mac.

This writes:

- `tracked_points.jsonl`
- `tracked_points.csv`
- `tracking_summary.json`
- optionally `tracking_overlay.mp4`

Analyze the tracked motion to extract an oscillation pattern:

```bash
python -m board_init.analyze_tracking \
  --tracking-jsonl outputs/video_track_try1_full/tracked_points.jsonl \
  --video IMG_8432.MOV \
  --output-dir outputs/video_track_try1_full/oscillation
```

This writes:

- `oscillation_summary.json`
- `oscillation_series.json`
- `oscillation_plot.png`
- sample peak/trough validation frames

## Troubleshooting OpenCV GUI

If GUI mode fails, check the diagnostic line printed by the CLI. On this Linux machine, headless mode is expected unless you have a working local display session.

Common cases:

- OpenCV built without a usable GUI backend: reinstall `opencv-python` inside the venv.
- Terminal launched from a remote SSH session or without `DISPLAY`/Wayland access: auto-GUI will not be assumed.
- PyTorch installed without GPU support: use `--device cpu` or reinstall PyTorch for your platform.

Quick checks:

```bash
python -c "import cv2; print(getattr(cv2, 'currentUIFramework', lambda: 'n/a')())"
python -c "import cv2; print('GUI:' in cv2.getBuildInformation())"
python -c "import torch; print(torch.cuda.is_available(), getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())"
```

If GUI is unavailable, run headless with `--positive` / `--negative` or `--prompts-json`.

## Testing

Run the test suite:

```bash
pytest
```

Check CLI help:

```bash
python -m board_init --help
```
