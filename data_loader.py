"""Data loading and dataset management module.

Handles CSV data ingestion, file discovery, validation, and stratified sampling.
"""

import os
import pandas as pd
from sentiment_analyzer.config import (
    FULL_DATA_FILE,
    SAMPLE_DATA_FILE,
)
from sentiment_analyzer.preprocessor import clean_text, map_rating


def get_data_file() -> str:
    """Locate the best available dataset file.

    Returns:
        str: Path to reviews_0-250.csv if present, otherwise reviews_sample.csv.

    Raises:
        FileNotFoundError: If neither dataset file can be found.
    """
    if os.path.exists(FULL_DATA_FILE):
        return FULL_DATA_FILE
    if os.path.exists(SAMPLE_DATA_FILE):
        return SAMPLE_DATA_FILE
    raise FileNotFoundError(
        f"Neither '{FULL_DATA_FILE}' nor '{SAMPLE_DATA_FILE}' could be found."
    )


def load_dataset(csv_path: str = None, nrows: int = None) -> pd.DataFrame:
    """Load and preprocess reviews dataset from a CSV file.

    Args:
        csv_path (str, optional): Path to the CSV file. Defaults to auto-discovered path.
        nrows (int, optional): Maximum number of rows to load. Defaults to None (all rows).

    Returns:
        pd.DataFrame: Processed dataframe with columns ['rating', 'review_text', 'sentiment'].

    Raises:
        FileNotFoundError: If the specified CSV file does not exist.
        ValueError: If required columns ('rating', 'review_text') are missing.
    """
    if csv_path is None:
        csv_path = get_data_file()

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    # Read required columns
    df = pd.read_csv(csv_path, usecols=["rating", "review_text"], nrows=nrows)

    # Validate columns
    required_cols = {"rating", "review_text"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    # Drop nulls and empty texts
    df = df.dropna(subset=["rating", "review_text"]).copy()
    df["review_text"] = df["review_text"].apply(clean_text)
    df = df[df["review_text"].str.len() > 0]

    # Map ratings to sentiment labels
    df["sentiment"] = df["rating"].apply(map_rating)

    return df.reset_index(drop=True)
