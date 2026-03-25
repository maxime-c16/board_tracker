from board_init.cli import build_parser


def test_cli_help_includes_max_side() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert "--max-side" in help_text
    assert "--device {auto,cpu,cuda,mps}" in help_text
