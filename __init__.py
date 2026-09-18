"""Sentiment Analyzer Package.

An end-to-end NLP sentiment classification library for customer review analysis.
"""

from sentiment_analyzer.config import (
    DEFAULT_MODEL_FILE,
    FULL_DATA_FILE,
    SAMPLE_DATA_FILE,
    VALID_SENTIMENTS,
)
from sentiment_analyzer.data_loader import get_data_file, load_dataset
from sentiment_analyzer.evaluator import evaluate_pipeline, run_train_test_evaluation
from sentiment_analyzer.model import build_pipeline, load_model, save_model, train_model
from sentiment_analyzer.predictor import SentimentPredictor, predict
from sentiment_analyzer.preprocessor import clean_text, map_rating

__version__ = "1.0.0"
__all__ = [
    "predict",
    "load_model",
    "save_model",
    "train_model",
    "build_pipeline",
    "evaluate_pipeline",
    "run_train_test_evaluation",
    "SentimentPredictor",
    "clean_text",
    "map_rating",
    "load_dataset",
    "get_data_file",
    "DEFAULT_MODEL_FILE",
    "SAMPLE_DATA_FILE",
    "FULL_DATA_FILE",
    "VALID_SENTIMENTS",
]
