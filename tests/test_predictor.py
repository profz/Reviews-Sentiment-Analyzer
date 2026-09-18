"""Unit tests for inference and prediction engine.
"""

from sentiment_analyzer.config import VALID_SENTIMENTS
from sentiment_analyzer.model import load_model
from sentiment_analyzer.predictor import SentimentPredictor, predict


class TestSentimentPredictor:
    """Tests for SentimentPredictor class and inference functions."""

    @classmethod
    def setup_class(cls):
        cls.model = load_model()
        cls.predictor = SentimentPredictor(pipeline=cls.model)

    def test_predict_positive_review(self):
        sentiment, conf = self.predictor.predict("This product exceeded all my expectations, absolutely love it!")
        assert sentiment == "Positive"
        assert 0.0 <= conf <= 100.0

    def test_predict_negative_review(self):
        sentiment, conf = self.predictor.predict("Broke after two days, completely useless.")
        assert sentiment == "Negative"
        assert 0.0 <= conf <= 100.0

    def test_predict_empty_text_defaults_to_neutral(self):
        sentiment, conf = self.predictor.predict("")
        assert sentiment == "Neutral"
        assert conf == 50.0

    def test_predict_whitespace_only(self):
        sentiment, conf = self.predictor.predict("     ")
        assert sentiment == "Neutral"
        assert conf == 50.0

    def test_predict_detailed(self):
        res = self.predictor.predict_detailed("Amazing customer support and fast delivery.")
        assert "sentiment" in res
        assert "confidence" in res
        assert "probabilities" in res
        assert res["sentiment"] in VALID_SENTIMENTS
        probs = res["probabilities"]
        assert len(probs) == 3
        assert pytest_approx_sum(list(probs.values()), 100.0)

    def test_predict_batch(self):
        texts = [
            "Superb build quality!",
            "Terrible experience, would not buy again.",
            "Average quality, nothing special.",
        ]
        results = self.predictor.predict_batch(texts)
        assert len(results) == 3
        for sentiment, conf in results:
            assert sentiment in VALID_SENTIMENTS
            assert 0.0 <= conf <= 100.0

    def test_legacy_predict_function(self):
        sentiment, conf = predict(self.model, "Outstanding performance!")
        assert sentiment in VALID_SENTIMENTS
        assert 0.0 <= conf <= 100.0


def pytest_approx_sum(values, expected, tol=1.0):
    return abs(sum(values) - expected) < tol
