import numpy as np

from board_init.config import ScoringConfig
from board_init.geometry import extract_geometry
from board_init.scoring import score_candidate
from board_init.types import PromptSet


def test_scoring_prefers_elongated_mask() -> None:
    thin = np.zeros((120, 200), dtype=bool)
    thin[55:65, 20:180] = True
    blob = np.zeros((120, 200), dtype=bool)
    blob[35:95, 70:130] = True

    prompts = PromptSet(positive=[(40, 60)], negative=[])
    thin_score, _ = score_candidate(thin, 0.8, extract_geometry(thin), prompts, None, ScoringConfig())
    blob_score, _ = score_candidate(blob, 0.8, extract_geometry(blob), prompts, None, ScoringConfig())
    assert thin_score > blob_score


def test_scoring_penalizes_branching_mask() -> None:
    straight = np.zeros((160, 220), dtype=bool)
    straight[76:84, 20:200] = True

    branching = straight.copy()
    branching[40:120, 100:108] = True

    prompts = PromptSet(positive=[(40, 80), (180, 80)], negative=[])
    straight_score, straight_terms = score_candidate(
        straight, 0.8, extract_geometry(straight), prompts, None, ScoringConfig()
    )
    branching_score, branching_terms = score_candidate(
        branching, 0.8, extract_geometry(branching), prompts, None, ScoringConfig()
    )

    assert straight_terms["branch_penalty"] < branching_terms["branch_penalty"]
    assert straight_score > branching_score


def test_scoring_prefers_mask_that_extends_beyond_prompt_span() -> None:
    short_mask = np.zeros((120, 220), dtype=bool)
    short_mask[56:64, 70:128] = True

    long_mask = np.zeros((120, 220), dtype=bool)
    long_mask[56:64, 20:200] = True

    prompts = PromptSet(positive=[(80, 60), (120, 60)], negative=[])
    short_score, short_terms = score_candidate(
        short_mask, 0.8, extract_geometry(short_mask), prompts, None, ScoringConfig()
    )
    long_score, long_terms = score_candidate(
        long_mask, 0.8, extract_geometry(long_mask), prompts, None, ScoringConfig()
    )

    assert short_terms["prompt_span_coverage"] < long_terms["prompt_span_coverage"]
    assert short_score < long_score
