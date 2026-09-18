"""NLP Sentiment Analyzer CLI and Legacy Entry Point.

This module provides the primary command-line interface and legacy API compatibility
for sentiment classification of customer reviews.
"""

import sys
import os
import argparse
from sentiment_analyzer.config import (
    DEFAULT_MODEL_FILE as MODEL_FILE,
    FULL_DATA_FILE as DATA_FILE,
    SAMPLE_DATA_FILE,
)
from sentiment_analyzer.data_loader import get_data_file
from sentiment_analyzer.model import load_model, train_model
from sentiment_analyzer.predictor import predict


def interactive_mode(model):
    """Run interactive CLI prompt."""
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in ("exit", "quit", "q"):
            break
        sentiment, conf = predict(model, text)
        print(f"{sentiment} ({conf:.1f}%)")


def main():
    """Main CLI execution flow."""
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("text", nargs="*", help=argparse.SUPPRESS)
    parser.add_argument("--train", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.train:
        train_model()
        return

    model = load_model()

    if args.text:
        query = " ".join(args.text).strip()
        sentiment, conf = predict(model, query)
        print(f"{sentiment} ({conf:.1f}%)")
    else:
        interactive_mode(model)


if __name__ == "__main__":
    main()
