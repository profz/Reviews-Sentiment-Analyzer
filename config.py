"""Configuration module for Sentiment Analyzer.

Defines default hyperparameters, file paths, and label mappings.
"""

import os
from pathlib import Path

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR

# Data file paths
DEFAULT_MODEL_FILE = os.path.join(BASE_DIR, "sentiment_model.joblib")
FULL_DATA_FILE = os.path.join(BASE_DIR, "reviews_0-250.csv")
SAMPLE_DATA_FILE = os.path.join(BASE_DIR, "reviews_sample.csv")

# Text vectorization hyperparameters
TFIDF_MAX_FEATURES = 25000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_SUBLINEAR_TF = True

# Classification hyperparameters
CLF_MAX_ITER = 200
CLF_SOLVER = "lbfgs"
CLF_MULTI_CLASS = "multinomial"
RANDOM_STATE = 42

# Label categories
LABEL_POSITIVE = "Positive"
LABEL_NEUTRAL = "Neutral"
LABEL_NEGATIVE = "Negative"
VALID_SENTIMENTS = [LABEL_NEGATIVE, LABEL_NEUTRAL, LABEL_POSITIVE]

# Rating thresholds
RATING_POSITIVE_MIN = 4
RATING_NEGATIVE_MAX = 2
