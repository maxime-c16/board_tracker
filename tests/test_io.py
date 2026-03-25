import json
from pathlib import Path

import pytest

from board_init.io import load_prompts_json, write_json


def test_json_export_structure(tmp_path: Path) -> None:
    payload = {
        "image_path": "frame.png",
        "candidate_ranking": [{"index": 0, "combined_score": 0.9}],
        "selected_candidate": 0,
    }
    target = tmp_path / "run_summary.json"
    write_json(target, payload)
    assert target.exists()
    data = target.read_text(encoding="utf-8")
    assert '"selected_candidate": 0' in data


def test_load_prompts_json_rejects_non_original_coordinate_space(tmp_path: Path) -> None:
    target = tmp_path / "prompts.json"
    target.write_text(
        json.dumps(
            {
                "positive": [[10, 20]],
                "prompt_coordinate_space": "sam_working_image_pixels",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="original-image pixel coordinates"):
        load_prompts_json(target)
