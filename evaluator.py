"""Model evaluation module.

Provides performance metrics calculation, confusion matrix generation,
and classification reports.
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sentiment_analyzer.config import RANDOM_STATE, VALID_SENTIMENTS
from sentiment_analyzer.data_loader import load_dataset
from sentiment_analyzer.model import build_pipeline


def evaluate_pipeline(
    pipeline,
    x_test: List[str],
    y_test: List[str],
    labels: List[str] = None,
) -> Dict[str, Any]:
    """Evaluate a trained model against test data.

    Args:
        pipeline: Trained classification pipeline.
        x_test: Collection of test text strings.
        y_test: True sentiment labels for test set.
        labels: Ordered list of labels (default: VALID_SENTIMENTS).

    Returns:
        dict: Evaluation metrics including accuracy, precision, recall, F1,
              confusion matrix, and detailed classification report dictionary.
    """
    if labels is None:
        labels = VALID_SENTIMENTS

    y_pred = pipeline.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, labels=labels, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, labels=labels, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, labels=labels, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, labels=labels, average="weighted", zero_division=0)

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report_dict = classification_report(
        y_test, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    return {
        "accuracy": float(acc),
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "classification_report": report_dict,
        "test_samples_count": len(y_test),
    }


def run_train_test_evaluation(
    csv_path: str = None,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Perform an 80/20 train-test split evaluation on dataset.

    Args:
        csv_path: Optional path to dataset CSV.
        test_size: Proportion of dataset to include in test split.
        random_state: Seed for reproducible splitting.

    Returns:
        dict: Complete evaluation metrics on the held-out test split.
    """
    df = load_dataset(csv_path=csv_path)
    x = df["review_text"].tolist()
    y = df["sentiment"].tolist()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    return evaluate_pipeline(pipeline, x_test, y_test)
