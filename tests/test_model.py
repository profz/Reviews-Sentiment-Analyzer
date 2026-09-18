"""Unit tests for model pipeline construction and persistence.
"""

from sklearn.pipeline import Pipeline
from sentiment_analyzer.config import VALID_SENTIMENTS
from sentiment_analyzer.model import build_pipeline, load_model


class TestModelPipeline:
    """Tests for model construction and loading."""

    def test_build_pipeline_structure(self):
        pipeline = build_pipeline(max_features=100)
        assert isinstance(pipeline, Pipeline)
        assert "tfidf" in pipeline.named_steps
        assert "clf" in pipeline.named_steps

    def test_load_existing_model(self):
        pipeline = load_model()
        assert isinstance(pipeline, Pipeline)
        assert hasattr(pipeline, "classes_")
        for sentiment in VALID_SENTIMENTS:
            assert sentiment in pipeline.classes_
