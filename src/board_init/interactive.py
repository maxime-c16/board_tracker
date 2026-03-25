from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass, field

import cv2
import numpy as np

from .resize import display_point_to_original, original_point_to_display
from .types import BBox, PromptSet, ResizeMetadata
from .visualization import draw_prompts, overlay_mask

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GuiDiagnostics:
    backend_available: bool
    backend_name: str | None
    platform_name: str
    display_env: str | None
    ssh_connection: bool
    local_session_likely: bool
    diagnostic_message: str


@dataclass
class InteractiveSelection:
    prompts: PromptSet = field(default_factory=PromptSet)
    selected_candidate: int | None = None
    window_opened: bool = False


def _opencv_backend_name() -> str | None:
    current_framework = getattr(cv2, "currentUIFramework", None)
    if callable(current_framework):
        try:
            backend = current_framework()
        except Exception:
            backend = ""
        if backend:
            return str(backend)
    try:
        info = cv2.getBuildInformation()
    except Exception:
        return None

    for line in info.splitlines():
        stripped = line.strip()
        if stripped.startswith("GUI:"):
            backend = stripped.split(":", maxsplit=1)[1].strip()
            return backend or None
    return None


def opencv_gui_available() -> bool:
    backend = (_opencv_backend_name() or "").upper()
    if not backend or backend == "NONE":
        return False
    return any(token in backend for token in ("COCOA", "QT", "GTK", "WIN32"))


def detect_gui_support() -> GuiDiagnostics:
    backend_name = _opencv_backend_name()
    backend_available = opencv_gui_available()
    display_env = os.environ.get("DISPLAY") or None
    ssh_connection = bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT"))
    platform_name = platform.system()
    local_session_likely = platform_name == "Darwin" and not ssh_connection

    if backend_available and local_session_likely:
        message = f"OpenCV GUI backend {backend_name or 'unknown'} looks usable for a local macOS window session."
    elif backend_available and display_env:
        message = f"OpenCV GUI backend {backend_name or 'unknown'} is present and DISPLAY is set."
    elif backend_available:
        message = (
            f"OpenCV GUI backend {backend_name or 'unknown'} is present, but no local macOS session or DISPLAY "
            "was detected."
        )
    else:
        message = "OpenCV was built without a usable GUI backend. Reinstall with Cocoa/Qt support for local windows."

    return GuiDiagnostics(
        backend_available=backend_available,
        backend_name=backend_name,
        platform_name=platform_name,
        display_env=display_env,
        ssh_connection=ssh_connection,
        local_session_likely=local_session_likely,
        diagnostic_message=message,
    )


def _validate_gui_or_raise() -> GuiDiagnostics:
    diagnostics = detect_gui_support()
    if diagnostics.backend_available and (diagnostics.local_session_likely or diagnostics.display_env):
        return diagnostics
    raise RuntimeError(diagnostics.diagnostic_message)


def _draw_roi(image_rgb: np.ndarray, roi: BBox | None) -> np.ndarray:
    canvas = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    if roi is not None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 200, 0), thickness=2)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def collect_roi_gui(
    image_rgb: np.ndarray,
    resize_metadata: ResizeMetadata,
    window_name: str = "board_init_roi",
) -> BBox | None:
    diagnostics = _validate_gui_or_raise()
    LOGGER.info("GUI diagnostics: %s", diagnostics.diagnostic_message)
    LOGGER.info("ROI controls: drag left mouse to draw ROI, Enter to accept, s to skip, c to clear, q to cancel.")

    drag_start: tuple[int, int] | None = None
    current_roi_working: BBox | None = None

    def callback(event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        nonlocal drag_start, current_roi_working
        if event == cv2.EVENT_LBUTTONDOWN:
            drag_start = (x, y)
            current_roi_working = (x, y, x + 1, y + 1)
        elif event == cv2.EVENT_MOUSEMOVE and drag_start is not None:
            x1 = min(drag_start[0], x)
            y1 = min(drag_start[1], y)
            x2 = max(drag_start[0], x)
            y2 = max(drag_start[1], y)
            current_roi_working = (x1, y1, max(x1 + 1, x2), max(y1 + 1, y2))
        elif event == cv2.EVENT_LBUTTONUP and drag_start is not None:
            x1 = min(drag_start[0], x)
            y1 = min(drag_start[1], y)
            x2 = max(drag_start[0], x)
            y2 = max(drag_start[1], y)
            current_roi_working = (x1, y1, max(x1 + 1, x2), max(y1 + 1, y2))
            drag_start = None

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, callback)
    try:
        while True:
            display_frame = cv2.cvtColor(_draw_roi(image_rgb, current_roi_working), cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                raise KeyboardInterrupt("Interactive ROI selection cancelled.")
            if key == ord("c"):
                current_roi_working = None
            if key == ord("s"):
                return None
            if key in (13, 10, ord("r")):
                break
    finally:
        cv2.destroyWindow(window_name)

    if current_roi_working is None:
        return None
    x1, y1 = display_point_to_original((current_roi_working[0], current_roi_working[1]), resize_metadata)
    x2, y2 = display_point_to_original((current_roi_working[2], current_roi_working[3]), resize_metadata)
    return min(x1, x2), min(y1, y2), max(x1 + 1, x2), max(y1 + 1, y2)


def collect_prompts_gui(
    image_rgb: np.ndarray,
    resize_metadata: ResizeMetadata,
    view_origin_working: tuple[int, int] = (0, 0),
    window_name: str = "board_init",
) -> InteractiveSelection:
    diagnostics = _validate_gui_or_raise()
    LOGGER.info("GUI diagnostics: %s", diagnostics.diagnostic_message)
    LOGGER.info("Prompt controls: left=positive, right=negative, Enter/r=run, c=clear, q=cancel.")

    state = InteractiveSelection()

    def callback(event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        point = display_point_to_original((x, y), resize_metadata, view_origin_working=view_origin_working)
        if event == cv2.EVENT_LBUTTONDOWN:
            state.prompts.positive.append(point)
        elif event == cv2.EVENT_RBUTTONDOWN:
            state.prompts.negative.append(point)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    state.window_opened = True
    cv2.setMouseCallback(window_name, callback)

    try:
        while True:
            display_prompts = PromptSet(
                positive=[
                    original_point_to_display(point, resize_metadata, view_origin_working=view_origin_working)
                    for point in state.prompts.positive
                ],
                negative=[
                    original_point_to_display(point, resize_metadata, view_origin_working=view_origin_working)
                    for point in state.prompts.negative
                ],
            )
            display_frame = cv2.cvtColor(draw_prompts(image_rgb, display_prompts), cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                raise KeyboardInterrupt("Interactive prompt collection cancelled.")
            if key == ord("c"):
                state.prompts = PromptSet()
            if key in (13, 10, ord("r")):
                break
    finally:
        cv2.destroyWindow(window_name)

    return state


def select_candidate_gui(
    image_rgb: np.ndarray,
    prompts: PromptSet,
    candidate_masks: list[np.ndarray],
    window_name: str = "board_init_results",
) -> int:
    if not candidate_masks:
        raise ValueError("No candidate masks available.")
    diagnostics = _validate_gui_or_raise()
    LOGGER.info("GUI diagnostics: %s", diagnostics.diagnostic_message)

    current = 0
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        while True:
            frame = draw_prompts(overlay_mask(image_rgb, candidate_masks[current]), prompts)
            cv2.imshow(window_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("n"), 83):
                current = (current + 1) % len(candidate_masks)
            elif key in (ord("p"), 81):
                current = (current - 1) % len(candidate_masks)
            elif key in (13, 10, ord("a")):
                return current
            elif key == ord("q"):
                raise KeyboardInterrupt("Candidate selection cancelled.")
    finally:
        cv2.destroyWindow(window_name)


def collect_prompts_x11(image_rgb: np.ndarray, window_name: str = "board_init") -> InteractiveSelection:
    metadata = ResizeMetadata(
        original_width=image_rgb.shape[1],
        original_height=image_rgb.shape[0],
        working_width=image_rgb.shape[1],
        working_height=image_rgb.shape[0],
        resize_scale=1.0,
        max_side=None,
    )
    return collect_prompts_gui(image_rgb=image_rgb, resize_metadata=metadata, window_name=window_name)


def select_candidate_x11(
    image_rgb: np.ndarray,
    prompts: PromptSet,
    candidate_masks: list[np.ndarray],
    window_name: str = "board_init_results",
) -> int:
    return select_candidate_gui(
        image_rgb=image_rgb,
        prompts=prompts,
        candidate_masks=candidate_masks,
        window_name=window_name,
    )
