"""Inference and sentiment prediction engine.

Provides class and utility functions for single and batch text sentiment classification.
"""

from typing import Dict, List, Tuple
from sentiment_analyzer.config import DEFAULT_MODEL_FILE, LABEL_NEUTRAL
from sentiment_analyzer.model import load_model
from sentiment_analyzer.preprocessor import clean_text


class SentimentPredictor:
    """Predictor class encapsulating inference logic with error handling."""

    def __init__(self, model_path: str = DEFAULT_MODEL_FILE, pipeline=None):
        """Initialize predictor with either a loaded pipeline or path to joblib file.

        Args:
            model_path (str): Path to serialized joblib file.
            pipeline (Pipeline, optional): Pre-loaded scikit-learn pipeline.
        """
        if pipeline is not None:
            self.pipeline = pipeline
        else:
            self.pipeline = load_model(model_path)

    def predict(self, text: str) -> Tuple[str, float]:
        """Classify a single text string into sentiment category and confidence score.

        Args:
            text (str): Input text string.

        Returns:
            tuple: (sentiment_label, confidence_percentage)
        """
        cleaned = clean_text(text)
        if not cleaned:
            return LABEL_NEUTRAL, 50.0

        pred = self.pipeline.predict([cleaned])[0]
        prob = float(max(self.pipeline.predict_proba([cleaned])[0]) * 100)
        return str(pred), prob

    def predict_detailed(self, text: str) -> Dict[str, any]:
        """Provide detailed prediction including class probability distributions.

        Args:
            text (str): Input text string.

        Returns:
            dict: {
                'text': original text,
                'sentiment': predicted label,
                'confidence': top confidence %,
                'probabilities': {class_label: float %}
            }
        """
        cleaned = clean_text(text)
        if not cleaned:
            return {
                "text": text,
                "sentiment": LABEL_NEUTRAL,
                "confidence": 50.0,
                "probabilities": {cls: 33.33 for cls in self.pipeline.classes_},
            }

        probs = self.pipeline.predict_proba([cleaned])[0]
        classes = self.pipeline.classes_
        prob_dict = {cls: float(prob * 100) for cls, prob in zip(classes, probs)}
        top_sentiment = str(self.pipeline.predict([cleaned])[0])
        top_confidence = float(max(probs) * 100)

        return {
            "text": text,
            "sentiment": top_sentiment,
            "confidence": top_confidence,
            "probabilities": prob_dict,
        }

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Perform batch prediction over a list of texts.

        Args:
            texts (list): List of input text strings.

        Returns:
            list: List of tuples (sentiment_label, confidence_percentage).
        """
        return [self.predict(t) for t in texts]


def predict(model, text: str) -> Tuple[str, float]:
    """Top-level functional interface matching legacy predict signature.

    Args:
        model: Trained pipeline model.
        text (str): Input text review.

    Returns:
        tuple: (sentiment_label, confidence_percentage)
    """
    cleaned = clean_text(text)
    if not cleaned:
        return LABEL_NEUTRAL, 50.0

    pred = model.predict([cleaned])[0]
    prob = float(max(model.predict_proba([cleaned])[0]) * 100)
    return str(pred), prob
