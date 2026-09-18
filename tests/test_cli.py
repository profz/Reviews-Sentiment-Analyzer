"""Unit tests for the CLI module.
"""

from sentiment_analyzer.cli import format_output, parse_args


class TestCli:
    """Tests for CLI arguments and output formatting."""

    def test_format_output(self):
        formatted = format_output("Positive", 98.64)
        assert formatted == "Positive (98.6%)"

    def test_parse_args_single_text(self):
        args = parse_args(["Great", "product!"])
        assert args.text == ["Great", "product!"]
        assert not args.train
        assert not args.evaluate

    def test_parse_args_train_flag(self):
        args = parse_args(["--train"])
        assert args.train is True
        assert not args.text

    def test_parse_args_evaluate_flag(self):
        args = parse_args(["--evaluate"])
        assert args.evaluate is True
