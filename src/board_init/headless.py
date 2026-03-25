from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .types import PromptSet


def parse_point(value: str) -> tuple[int, int]:
    x_str, y_str = value.split(",", maxsplit=1)
    return int(x_str), int(y_str)


def prompts_from_cli(
    positive: Iterable[str] | None,
    negative: Iterable[str] | None,
) -> PromptSet:
    return PromptSet(
        positive=[parse_point(item) for item in (positive or [])],
        negative=[parse_point(item) for item in (negative or [])],
    )


def prompts_to_payload(prompts: PromptSet, select_mask_index: int | None = None) -> dict[str, object]:
    payload: dict[str, object] = prompts.to_json()
    if select_mask_index is not None:
        payload["select_mask_index"] = select_mask_index
    return payload
