"""Text preprocessing and label mapping module.

Provides cleaning functions and rating-to-sentiment transformation.
"""

import re
from sentiment_analyzer.config import (
    LABEL_NEGATIVE,
    LABEL_NEUTRAL,
    LABEL_POSITIVE,
    RATING_NEGATIVE_MAX,
    RATING_POSITIVE_MIN,
)


def map_rating(rating) -> str:
    """Map a numerical star rating (1-5) to a 3-tier categorical sentiment label.

    Args:
        rating (int, float, or str): Numerical rating between 1 and 5.

    Returns:
        str: 'Negative', 'Neutral', or 'Positive'.

    Raises:
        ValueError: If rating cannot be parsed into a numeric value.
    """
    try:
        val = float(rating)
    except (ValueError, TypeError) as err:
        raise ValueError(f"Invalid rating value: {rating}. Must be numeric.") from err

    if val >= RATING_POSITIVE_MIN:
        return LABEL_POSITIVE
    if val <= RATING_NEGATIVE_MAX:
        return LABEL_NEGATIVE
    return LABEL_NEUTRAL


def clean_text(text: str) -> str:
    """Normalize and clean raw review text.

    - Strips leading and trailing whitespace.
    - Replaces consecutive whitespace characters with a single space.
    - Handles null / None values gracefully by returning an empty string.

    Args:
        text (str or None): Raw review text.

    Returns:
        str: Cleaned text string.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # Replace multiple whitespace/newlines with single space
    cleaned = re.sub(r"\s+", " ", text.strip())
    return cleaned
