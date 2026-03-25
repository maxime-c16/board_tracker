from board_init.headless import parse_point, prompts_from_cli, prompts_to_payload


def test_parse_point() -> None:
    assert parse_point("10,20") == (10, 20)


def test_prompts_from_cli() -> None:
    prompts = prompts_from_cli(["1,2", "3,4"], ["5,6"])
    assert prompts.positive == [(1, 2), (3, 4)]
    assert prompts.negative == [(5, 6)]


def test_prompts_to_payload() -> None:
    prompts = prompts_from_cli(["7,8"], [])
    payload = prompts_to_payload(prompts, select_mask_index=2)
    assert payload["positive"] == [[7, 8]]
    assert payload["select_mask_index"] == 2
