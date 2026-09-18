"""Command-Line Interface (CLI) module for Sentiment Analyzer.

Provides interactive REPL mode, single review evaluation, and training invocation.
"""

import argparse
import sys
from sentiment_analyzer.evaluator import run_train_test_evaluation
from sentiment_analyzer.model import load_model, train_model
from sentiment_analyzer.predictor import SentimentPredictor, predict


def format_output(sentiment: str, confidence: float) -> str:
    """Format the sentiment classification output string.

    Args:
        sentiment (str): Predicted sentiment label.
        confidence (float): Confidence score (0-100).

    Returns:
        str: Formatted string (e.g., 'Positive (99.8%)').
    """
    return f"{sentiment} ({confidence:.1f}%)"


def interactive_mode(model=None):
    """Run interactive REPL loop accepting user input until exit command.

    Args:
        model: Trained pipeline model. If None, loads default model.
    """
    if model is None:
        model = load_model()

    print("--- NLP Sentiment Analyzer (Interactive Mode) ---")
    print("Type your review and press Enter. Type 'exit', 'quit', or 'q' to stop.\n")

    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break

        if not text:
            continue
        if text.lower() in ("exit", "quit", "q"):
            print("Exiting interactive mode.")
            break

        sentiment, conf = predict(model, text)
        print(format_output(sentiment, conf))


def parse_args(args=None):
    """Parse command line arguments.

    Args:
        args: List of arguments (defaults to sys.argv[1:]).

    Returns:
        argparse.Namespace: Parsed argument values.
    """
    parser = argparse.ArgumentParser(
        prog="sentiment",
        description="NLP-based sentiment analysis on customer reviews.",
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Text string(s) to analyze. If omitted, starts interactive mode.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Trigger model training on customer reviews dataset.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run model evaluation on test split and display classification report.",
    )
    return parser.parse_args(args)


def main(args=None):
    """Main CLI entry point function.

    Args:
        args: Command line argument list.
    """
    parsed = parse_args(args)

    if parsed.train:
        print("Starting model training pipeline...")
        train_model()
        print("Training complete. Model saved successfully.")
        return

    if parsed.evaluate:
        print("Evaluating model performance on test split (80/20)...")
        results = run_train_test_evaluation()
        print(f"Accuracy:         {results['accuracy'] * 100:.2f}%")
        print(f"Macro Precision:  {results['macro_precision']:.4f}")
        print(f"Macro Recall:     {results['macro_recall']:.4f}")
        print(f"Macro F1-Score:   {results['macro_f1']:.4f}")
        print(f"Weighted F1:      {results['weighted_f1']:.4f}")
        print("\nClassification Report:")
        rep = results["classification_report"]
        print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
        for label in results["labels"]:
            metrics = rep[label]
            print(
                f"{label:<12} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                f"{metrics['f1-score']:<10.4f} {int(metrics['support']):<8}"
            )
        return

    model = load_model()

    if parsed.text:
        query = " ".join(parsed.text).strip()
        sentiment, conf = predict(model, query)
        print(format_output(sentiment, conf))
    else:
        interactive_mode(model)


if __name__ == "__main__":
    main()
