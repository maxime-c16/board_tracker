from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .types import PromptSet

LOGGER = logging.getLogger(__name__)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


class SamModelWrapper:
    def __init__(self, checkpoint: str | Path, model_type: str, device: str = "auto") -> None:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.device = resolve_device(device)

        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            missing_name = getattr(exc, "name", None)
            if missing_name == "torch":
                raise RuntimeError(
                    "PyTorch is not installed in the active environment. "
                    "Install torch/torchvision in this venv, then retry."
                ) from exc
            if missing_name == "segment_anything":
                raise RuntimeError(
                    "segment-anything is not installed in the active environment. "
                    "Install requirements.txt first."
                ) from exc
            raise RuntimeError(
                f"Failed to import SAM dependencies ({missing_name or 'unknown module'}). "
                "Check that torch and segment-anything are installed in this venv."
            ) from exc

        sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)
        LOGGER.info("Loaded SAM model %s on %s", model_type, self.device)

    def set_image(self, image_rgb: np.ndarray) -> None:
        self.predictor.set_image(image_rgb)

    def predict(
        self,
        prompts: PromptSet,
        multimask_output: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not prompts.positive and not prompts.negative:
            raise ValueError("At least one prompt point is required.")
        points = np.array(prompts.positive + prompts.negative, dtype=np.float32)
        labels = np.array(
            [1] * len(prompts.positive) + [0] * len(prompts.negative),
            dtype=np.int32,
        )
        masks, scores, _ = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=multimask_output,
        )
        return masks.astype(bool), scores.astype(float)
