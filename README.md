# NLP Sentiment Analyzer

<html>
  <h2 align="center">
    <img src="pngfind.com-mca-logo-png-6131138(1).png" width="250"/>
  </h2>
</html>

An end-to-end, modular, and lightweight Natural Language Processing (NLP) tool trained on customer reviews. Classifies text into **Positive**, **Neutral**, or **Negative** sentiment categories alongside calibrated confidence percentages. Built to adhere strictly to production software engineering and academic project guidelines.

---

## 1. Project Overview

Customer reviews on e-commerce platforms are voluminous, unstructured, and noisy. This project provides a robust, high-performance machine learning pipeline and Command-Line Interface (CLI) application for automated sentiment classification. 

The system leverages:
- **Sublinear TF-IDF Vectorization** with unigram and bigram tokenization (25,000 features).
- **Multinomial Logistic Regression** with L2 regularization and softmax probability calibration.
- **Modular Architecture** decoupling data loading, preprocessing, model serialization, inference, and CLI presentation into dedicated modules.
- **Dual Inference Modalities**: Instant single-shot command-line argument evaluation and an interactive Read-Eval-Print-Loop (REPL).

---

## 2. Key Features

- **3-Tier Categorical Sentiment Classification**: Accurately classifies reviews as Positive, Neutral, or Negative.
- **Calibrated Confidence Scores**: Outputs probabilistic certainty percentages for each prediction.
- **Low-Latency Inference**: Vectorized feature extraction delivering `<50 ms` prediction time per text query.
- **Self-Healing Model Discovery**: Automatically locates pre-trained serialized weights; if missing, automatically triggers training on the dataset.
- **Robust Error Handling**: Gracefully sanitizes inputs, handles empty strings, non-string types, and missing files.
- **Interactive REPL & CLI Scripting**: Direct single-string evaluation or interactive session mode.
- **Comprehensive Test Suite**: Fully covered by 26 automated unit and integration tests using `pytest`.

---

## 3. Project Structure

```text
.
├── sentiment_analyzer/              # Core modular package
│   ├── __init__.py                  # Package exports & public API
│   ├── config.py                    # Constants, hyperparams, file paths
│   ├── preprocessor.py              # Text normalization & rating mapping
│   ├── data_loader.py               # Dataset discovery & ingestion
│   ├── model.py                     # Pipeline construction & persistence
│   ├── evaluator.py                 # Performance metrics & reports
│   ├── predictor.py                 # SentimentPredictor class & inference
│   └── cli.py                       # CLI parsing & interactive REPL
├── tests/                           # Automated test suite (26 tests)
│   ├── __init__.py
│   ├── test_preprocessor.py         # Tests for text cleaning & mapping
│   ├── test_data_loader.py          # Tests for dataset loading
│   ├── test_model.py                # Tests for pipeline & serialization
│   ├── test_predictor.py            # Tests for single/batch inference
│   └── test_cli.py                  # Tests for CLI argument parsing
├── sentiment.py                     # Root CLI entry point (backward compatible)
├── statement.md                     # Problem statement & scope specification
├── guidlines.md                     # Academic project requirements & guidelines
├── PROJECT_REPORT.md                # Comprehensive 15-section project report
├── PROJECT_REPORT.docx              # Formatted Word report document
├── PROJECT_REPORT.pdf               # Submission-ready PDF report
├── requirements.txt                 # Project dependencies
├── reviews_sample.csv               # Balanced 7,500-review sample dataset
└── sentiment_model.joblib           # Serialized pre-trained model artifact
```

---

## 4. Technologies & Tools Used

- **Language:** Python 3.8+
- **Machine Learning & NLP:** `scikit-learn` (TF-IDF Vectorizer, Logistic Regression, Pipeline, Model Selection, Metrics)
- **Data Manipulation:** `pandas`, `numpy`
- **Model Persistence:** `joblib` (Level 3 compression)
- **Testing Framework:** `pytest`
- **Documentation & Conversion:** `pandoc`, `libreoffice`

---

## 5. Installation & Setup

### Prerequisites
- Python 3.8 or higher installed on your system.

### 1. Clone the Repository
```bash
git clone https://github.com/profz/NLP-sentiment-analysis-.git
cd NLP-sentiment-analysis-
```

### 2. Create and Activate a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 6. Usage Guide

### 6.1 Single Review Analysis (CLI)
Pass the review text directly as command-line arguments:
```bash
python sentiment.py "This product exceeded all my expectations, absolutely love it!"
```
**Output:**
```text
Positive (99.8%)
```

```bash
python sentiment.py "Broke after two days, completely useless and poor quality."
```
**Output:**
```text
Negative (98.6%)
```

```bash
python sentiment.py "It works okay, nothing particularly great or bad."
```
**Output:**
```text
Neutral (72.4%)
```

### 6.2 Interactive Mode (REPL)
Launch the interactive terminal interface by running without arguments:
```bash
python sentiment.py
```
**Interactive Session:**
```text
> Excellent quality and fast shipping!
Positive (99.7%)
> It is okay, nothing special.
Neutral (89.4%)
> Terrible battery life and customer service refused refund.
Negative (98.2%)
> exit
```

### 6.3 Retraining the Model
To re-train the model pipeline on `reviews_sample.csv` (or full corpus if present):
```bash
python sentiment.py --train
```

### 6.4 Model Evaluation Mode
To evaluate model performance on an 80/20 train-test split and generate a full metrics breakdown:
```bash
python -m sentiment_analyzer.cli --evaluate
```
**Output:**
```text
Evaluating model performance on test split (80/20)...
Accuracy:         73.10%
Macro Precision:  0.7320
Macro Recall:     0.7310
Macro F1-Score:   0.7315
Weighted F1:      0.7314

Classification Report:
Class        Precision  Recall     F1-Score   Support 
Negative     0.7031     0.7214     0.7122     499     
Neutral      0.6406     0.6380     0.6393     500     
Positive     0.8525     0.8337     0.8430     499     
```

### 6.5 Python API Integration
You can import the module directly into any Python workflow:
```python
from sentiment_analyzer import SentimentPredictor

predictor = SentimentPredictor()

# Single prediction
sentiment, confidence = predictor.predict("Best purchase I have made this year!")
print(f"Sentiment: {sentiment} ({confidence:.1f}%)")

# Detailed prediction with class probabilities
details = predictor.predict_detailed("Decent product for the price.")
print(details)
# {'text': 'Decent product for the price.', 'sentiment': 'Neutral', 'confidence': 68.2, 'probabilities': {'Negative': 12.1, 'Neutral': 68.2, 'Positive': 19.7}}

# Batch prediction
batch = predictor.predict_batch([
    "Loved it!",
    "Hated it!",
    "It is alright."
])
print(batch)
```

---

## 7. Instructions for Testing

The project includes 26 automated unit and validation tests verifying text cleaning, label mapping, dataset integrity, pipeline creation, inference edge cases, and CLI functionality.

Run the test suite using `pytest`:
```bash
pytest
```
Or run with verbose output:
```bash
pytest -v
```

**Expected Output:**
```text
============================= test session starts ==============================
collected 26 items

tests/test_cli.py::TestCli::test_format_output PASSED                     [  3%]
tests/test_cli.py::TestCli::test_parse_args_single_text PASSED            [  7%]
tests/test_cli.py::TestCli::test_parse_args_train_flag PASSED             [ 11%]
tests/test_cli.py::TestCli::test_parse_args_evaluate_flag PASSED          [ 15%]
tests/test_data_loader.py::TestDataLoader::test_get_data_file_exists PASSED [ 19%]
tests/test_data_loader.py::TestDataLoader::test_load_dataset_sample PASSED [ 23%]
tests/test_data_loader.py::TestDataLoader::test_load_dataset_invalid_path_raises_error PASSED [ 26%]
tests/test_model.py::TestModelPipeline::test_build_pipeline_structure PASSED [ 30%]
tests/test_model.py::TestModelPipeline::test_load_existing_model PASSED   [ 34%]
tests/test_predictor.py::TestSentimentPredictor::test_predict_positive_review PASSED [ 38%]
tests/test_predictor.py::TestSentimentPredictor::test_predict_negative_review PASSED [ 42%]
tests/test_predictor.py::TestSentimentPredictor::test_predict_empty_text_defaults_to_neutral PASSED [ 46%]
tests/test_predictor.py::TestSentimentPredictor::test_predict_whitespace_only PASSED [ 50%]
tests/test_predictor.py::TestSentimentPredictor::test_predict_detailed PASSED [ 53%]
tests/test_predictor.py::TestSentimentPredictor::test_predict_batch PASSED [ 57%]
tests/test_predictor.py::TestSentimentPredictor::test_legacy_predict_function PASSED [ 61%]
tests/test_preprocessor.py::TestMapRating::test_positive_ratings PASSED   [ 65%]
tests/test_preprocessor.py::TestMapRating::test_neutral_ratings PASSED    [ 69%]
tests/test_preprocessor.py::TestMapRating::test_negative_ratings PASSED   [ 73%]
tests/test_preprocessor.py::TestMapRating::test_invalid_ratings_raise_error PASSED [ 76%]
tests/test_preprocessor.py::TestCleanText::test_strip_whitespace PASSED   [ 80%]
tests/test_preprocessor.py::TestCleanText::test_collapse_multiple_spaces PASSED [ 84%]
tests/test_preprocessor.py::TestCleanText::test_collapse_newlines_and_tabs PASSED [ 88%]
tests/test_preprocessor.py::TestCleanText::test_none_input_returns_empty_string PASSED [ 92%]
tests/test_preprocessor.py::TestCleanText::test_numeric_input_converted_to_string PASSED [ 96%]
tests/test_preprocessor.py::TestCleanText::test_empty_string PASSED       [100%]

============================== 26 passed in 2.18s ==============================
```

---

## 8. Dataset Information

- `reviews_sample.csv`: A balanced, stratified sample dataset containing 7,500 reviews (2,500 Positive, 2,500 Neutral, 2,500 Negative; ~3.4 MB) bundled directly in the repository for immediate training and test verification.
- `reviews_0-250.csv`: The complete raw corpus containing over 602,000 reviews (~282 MB), excluded from version control via `.gitignore` in accordance with repository size best practices.

---

## 9. Performance Metrics

Evaluated on a held-out test split (1,498 reviews) from the balanced corpus:

| Sentiment Class | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **Negative** | 0.7031 | 0.7214 | 0.7122 | 499 |
| **Neutral** | 0.6406 | 0.6380 | 0.6393 | 500 |
| **Positive** | 0.8525 | 0.8337 | 0.8430 | 499 |
| **Overall Accuracy** | **73.10%** | | | **1,498** |
| **Macro Average** | **0.7320** | **0.7310** | **0.7315** | **1,498** |
| **Weighted Average** | **0.7320** | **0.7310** | **0.7314** | **1,498** |

---

## 10. Submission Documents

In accordance with project guidelines:
- [statement.md](statement.md): Formal problem statement, project scope, target users, and features.
- [PROJECT_REPORT.md](PROJECT_REPORT.md): Complete 15-section project report including UML and architecture diagrams.
- `PROJECT_REPORT.pdf`: Exported PDF report for submission portal upload.
- `PROJECT_REPORT.docx`: Formatted Word document version.
