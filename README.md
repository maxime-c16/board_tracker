# Board Init

Offline springboard-board initialization and mask extraction for diving analysis using classic Meta Segment Anything with the official `segment-anything` package and `SamPredictor`.

The primary workflow is now local macOS use with a native OpenCV window. Headless mode remains supported for scripted or remote runs.

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

## macOS Setup

Create and activate a local virtual environment:

```bash
cd /Volumes/DiveRecorderGPT/cv_diving
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
```

Helper script:

```bash
./scripts/bootstrap_macos.sh
source .venv/bin/activate
```

If you prefer the existing generic bootstrap script, `./scripts/bootstrap.sh` still works.

## PyTorch On Mac

`segment-anything` needs PyTorch, but the right wheel depends on your machine.

Apple Silicon / Metal (`mps`) example:

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
/Volumes/DiveRecorderGPT/cv_diving/models/sam/sam_vit_b_01ec64.pth
```

The expected local command path is therefore:

```bash
python -m board_init \
  --image /path/to/image.jpg \
  --checkpoint /Volumes/DiveRecorderGPT/cv_diving/models/sam/sam_vit_b_01ec64.pth \
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

Native OpenCV GUI mode on macOS:

```bash
python -m board_init \
  --image /path/to/image.jpg \
  --checkpoint /Volumes/DiveRecorderGPT/cv_diving/models/sam/sam_vit_b_01ec64.pth \
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

If `--gui` is omitted, the tool will still choose GUI mode automatically on a likely local macOS session when no positive prompts were supplied.

If you accept an ROI in GUI mode, SAM inference runs only on that cropped working region and masks are mapped back to the original image for export.

## Headless Workflow

Headless CLI prompts:

```bash
python -m board_init \
  --image /path/to/image.jpg \
  --checkpoint /Volumes/DiveRecorderGPT/cv_diving/models/sam/sam_vit_b_01ec64.pth \
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
  --checkpoint /Volumes/DiveRecorderGPT/cv_diving/models/sam/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --headless \
  --prompts-json examples/prompts.json \
  --max-side 2048
```

Finalize a reviewed candidate:

```bash
python -m board_init \
  --image /path/to/image.jpg \
  --checkpoint /Volumes/DiveRecorderGPT/cv_diving/models/sam/sam_vit_b_01ec64.pth \
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
  --checkpoint /Volumes/DiveRecorderGPT/cv_diving/models/sam/sam_vit_b_01ec64.pth \
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

## Troubleshooting macOS OpenCV GUI

If GUI mode fails, check the diagnostic line printed by the CLI.

Common cases:

- OpenCV built without Cocoa/Qt support: reinstall `opencv-python` inside the venv.
- Terminal launched from a remote SSH session: local macOS auto-GUI will not be assumed.
- PyTorch installed without MPS support: use `--device cpu` or reinstall PyTorch.

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
