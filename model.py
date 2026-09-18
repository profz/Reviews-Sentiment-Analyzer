"""Model pipeline definition, training, and persistence module.

Manages scikit-learn Pipeline construction, training, serialization, and deserialization.
"""

import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sentiment_analyzer.config import (
    CLF_MAX_ITER,
    CLF_SOLVER,
    DEFAULT_MODEL_FILE,
    RANDOM_STATE,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_SUBLINEAR_TF,
)
from sentiment_analyzer.data_loader import get_data_file, load_dataset


def build_pipeline(
    max_features: int = TFIDF_MAX_FEATURES,
    ngram_range: tuple = TFIDF_NGRAM_RANGE,
    sublinear_tf: bool = TFIDF_SUBLINEAR_TF,
    max_iter: int = CLF_MAX_ITER,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Construct an NLP text classification pipeline.

    Combines TF-IDF feature extraction with multi-class Logistic Regression.

    Args:
        max_features (int): Max vocabulary size for TF-IDF.
        ngram_range (tuple): Lower and upper boundary of n-values for n-grams.
        sublinear_tf (bool): Apply sublinear scaling to term frequency.
        max_iter (int): Maximum solver iterations for Logistic Regression.
        random_state (int): Random seed for reproducible solver initialization.

    Returns:
        Pipeline: Unfitted scikit-learn Pipeline instance.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
    )
    classifier = LogisticRegression(
        max_iter=max_iter,
        solver=CLF_SOLVER,
        random_state=random_state,
    )
    return Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier),
    ])


def save_model(pipeline: Pipeline, model_path: str = DEFAULT_MODEL_FILE) -> str:
    """Serialize and compress the trained pipeline to disk.

    Args:
        pipeline (Pipeline): Trained scikit-learn pipeline.
        model_path (str): Destination file path.

    Returns:
        str: Absolute path of saved model file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    joblib.dump(pipeline, model_path, compress=3)
    return model_path


def load_model(model_path: str = DEFAULT_MODEL_FILE) -> Pipeline:
    """Load a serialized model from disk, or train one if not found.

    Args:
        model_path (str): Path to the saved joblib model file.

    Returns:
        Pipeline: Loaded or freshly trained scikit-learn Pipeline.
    """
    if not os.path.exists(model_path):
        return train_model(csv_path=get_data_file(), model_path=model_path)
    return joblib.load(model_path)


def train_model(
    csv_path: str = None,
    model_path: str = DEFAULT_MODEL_FILE,
    sample_size: int = 100000,
) -> Pipeline:
    """Train the sentiment pipeline on review data and persist weights.

    Args:
        csv_path (str, optional): Path to input dataset CSV.
        model_path (str): Path to write serialized model artifact.
        sample_size (int): Max number of samples to train on.

    Returns:
        Pipeline: Trained scikit-learn Pipeline.
    """
    df = load_dataset(csv_path=csv_path, nrows=sample_size)
    pipeline = build_pipeline()
    pipeline.fit(df["review_text"], df["sentiment"])
    save_model(pipeline, model_path=model_path)
    return pipeline
