"""Unit tests for the data loading module.
"""

import os
import pytest
from sentiment_analyzer.config import VALID_SENTIMENTS
from sentiment_analyzer.data_loader import get_data_file, load_dataset


class TestDataLoader:
    """Tests for dataset discovery and loading."""

    def test_get_data_file_exists(self):
        path = get_data_file()
        assert os.path.exists(path)
        assert path.endswith(".csv")

    def test_load_dataset_sample(self):
        df = load_dataset(nrows=50)
        assert not df.empty
        assert "rating" in df.columns
        assert "review_text" in df.columns
        assert "sentiment" in df.columns
        assert set(df["sentiment"].unique()).issubset(set(VALID_SENTIMENTS))
        assert len(df) <= 50

    def test_load_dataset_invalid_path_raises_error(self):
        with pytest.raises(FileNotFoundError):
            load_dataset(csv_path="non_existent_file.csv")
