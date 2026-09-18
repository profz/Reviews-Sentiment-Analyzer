"""Unit tests for the preprocessor and rating mapping module.
"""

import pytest
from sentiment_analyzer.config import LABEL_NEGATIVE, LABEL_NEUTRAL, LABEL_POSITIVE
from sentiment_analyzer.preprocessor import clean_text, map_rating


class TestMapRating:
    """Tests for numerical rating to sentiment label mapping."""

    def test_positive_ratings(self):
        assert map_rating(5) == LABEL_POSITIVE
        assert map_rating(4) == LABEL_POSITIVE
        assert map_rating(4.5) == LABEL_POSITIVE
        assert map_rating("5") == LABEL_POSITIVE

    def test_neutral_ratings(self):
        assert map_rating(3) == LABEL_NEUTRAL
        assert map_rating(3.0) == LABEL_NEUTRAL
        assert map_rating("3") == LABEL_NEUTRAL

    def test_negative_ratings(self):
        assert map_rating(2) == LABEL_NEGATIVE
        assert map_rating(1) == LABEL_NEGATIVE
        assert map_rating(1.5) == LABEL_NEGATIVE
        assert map_rating("1") == LABEL_NEGATIVE

    def test_invalid_ratings_raise_error(self):
        with pytest.raises(ValueError):
            map_rating("invalid_rating")
        with pytest.raises(ValueError):
            map_rating(None)


class TestCleanText:
    """Tests for text normalization and sanitization."""

    def test_strip_whitespace(self):
        assert clean_text("  hello world  ") == "hello world"

    def test_collapse_multiple_spaces(self):
        assert clean_text("Great   product,   loved    it!") == "Great product, loved it!"

    def test_collapse_newlines_and_tabs(self):
        assert clean_text("Line 1\nLine 2\t\nLine 3") == "Line 1 Line 2 Line 3"

    def test_none_input_returns_empty_string(self):
        assert clean_text(None) == ""

    def test_numeric_input_converted_to_string(self):
        assert clean_text(12345) == "12345"

    def test_empty_string(self):
        assert clean_text("   ") == ""
