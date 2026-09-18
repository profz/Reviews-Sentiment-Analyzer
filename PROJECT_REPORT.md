---
title: "NLP-Based Sentiment Analysis on Customer Reviews"
subtitle: "VITyarthi - Build Your Own Project Evaluation Report"
author: "Kowshik Raj R"
date: "September 2026"
geometry: margin=1in
fontsize: 11pt
---

# 1. Cover Page

**Course Name:** Fundamentals of Artificial Intelligence and Machine Learning  
**Project Title:** End-to-End NLP Sentiment Analysis System on Multi-Tier Customer Feedback  
**Student Name / Roll Number:** Kowshik Raj R 25MIM10158 
**Course Code / Evaluation:** VITyarthi - Build Your Own Project  
**Submission Date:** September 18, 2026  
**Repository URL:** https://github.com/profz/NLP-sentiment-analysis-  
**Target Domain:** Natural Language Processing (NLP), Text Classification, E-Commerce Analytics  

---

# 2. Introduction

Customer opinions and feedback are central to the operational and strategic decision-making processes of modern e-commerce enterprises, service providers, and digital platforms. Every day, platforms such as Amazon, Flipkart, and Yelp receive hundreds of thousands of customer reviews detailing experiences with products, logistics, customer service, and software interfaces. 

Sentiment Analysis—often referred to as opinion mining—is a sub-discipline of Natural Language Processing (NLP) and Artificial Intelligence (AI) tasked with computationally identifying and categorizing subjective opinions expressed in unstructured natural text. 

The objective of this project is to architect, develop, rigorously test, and package an end-to-end, high-performance sentiment analysis system. Unlike primitive script-based prototypes, this solution is engineered as a production-grade Python package with a decoupled modular architecture, strict data validation, calibrated probability scoring, sub-50 ms inference latency, and an interactive command-line interface (CLI). By classifying customer feedback into **Positive**, **Neutral**, and **Negative** sentiment classes, the system empowers stakeholders to automate feedback triage, rapidly detect product defects, and track consumer satisfaction at scale.

---

# 3. Problem Statement

Modern e-commerce platforms suffer from substantial data scale and distribution anomalies that render manual and naive automated analysis impractical:

1. **Volume and Unstructured Noise:** With millions of reviews posted continuously, manual human moderation is economically non-viable and slow. Furthermore, customer reviews are rife with colloquialisms, irregular capitalization, punctuation, and shorthand.
2. **Subjectivity & Human Inconsistency:** Different human reviewers apply varying standards of evaluation. An automated model provides deterministic, consistent scoring across all data streams.
3. **Severe Class Imbalance:** Real-world review platforms exhibit severe class imbalance, where over 70% of reviews are 5-star ratings. A model trained naively on raw e-commerce distributions develops a heavy bias toward positive sentiment, failing to capture critical negative complaints that require immediate remediation.
4. **Production Latency & Cost Constraints:** Heavy Deep Learning or Large Language Model (LLM) architectures introduce steep computational overhead (>500 ms latency per inference, costly GPU infrastructure). A production triage system requires a lightweight, resource-efficient model providing instant inference (<50 ms) on standard CPU hardware.

---

# 4. Functional Requirements

In accordance with the system specification, the project implements three primary functional modules:

### Module 1: Data Ingestion & Preprocessing (`sentiment_analyzer.data_loader`, `preprocessor`)
- **FR-1.1:** Discover and ingest review data from either full-scale (`reviews_0-250.csv`) or stratified sample (`reviews_sample.csv`) data sources.
- **FR-1.2:** Validate schema integrity, verifying mandatory presence of numerical `rating` and textual `review_text` features.
- **FR-1.3:** Sanitize text by stripping anomalous whitespace, trimming outer tokens, and handling null/empty entries gracefully.
- **FR-1.4:** Map discrete 1–5 star ratings into the 3-tier sentiment taxonomy:
  - Ratings 4 and 5 $\rightarrow$ **Positive**
  - Rating 3 $\rightarrow$ **Neutral**
  - Ratings 1 and 2 $\rightarrow$ **Negative**

### Module 2: Model Training & Evaluation (`sentiment_analyzer.model`, `evaluator`)
- **FR-2.1:** Construct an end-to-end vectorization and classification pipeline utilizing sublinear TF-IDF (1-2 n-grams, 25,000 features) and Multinomial Logistic Regression.
- **FR-2.2:** Provide stratified train-test splitting (80/20 ratio) for unbiased generalization measurement.
- **FR-2.3:** Calculate comprehensive evaluation metrics: Accuracy, Macro Precision, Macro Recall, Macro F1-score, Weighted F1-score, and Confusion Matrix.
- **FR-2.4:** Serialize trained pipelines to disk with `joblib` compression (level 3) to produce compact model artifacts (<1 MB).

### Module 3: Inference Engine & CLI Interface (`sentiment_analyzer.predictor`, `cli`)
- **FR-3.1:** Execute single-review classification via command-line arguments, returning predicted class and percentage confidence.
- **FR-3.2:** Provide a persistent interactive REPL (Read-Eval-Print-Loop) terminal interface allowing iterative evaluation.
- **FR-3.3:** Deliver detailed probability distributions across all three classes via programmatic API.
- **FR-3.4:** Implement automatic model recovery: if serialized model weights are missing upon invocation, trigger training automatically without crashing.

---

# 5. Non-Functional Requirements

To ensure software excellence, four key non-functional requirements are specified and strictly enforced:

1. **Performance & Latency:**
   - Single-text inference latency must remain strictly under 50 milliseconds on standard CPU hardware.
   - Vectorization and classification must leverage sparse matrix operations for minimal computation time.
2. **Usability & Clean Interaction:**
   - The CLI must provide intuitive argument parsing, clear command flags (`--train`, `--evaluate`), helpful syntax error messaging, and an accessible interactive prompt.
3. **Reliability & Error Handling:**
   - The system must not crash on malformed inputs (e.g., empty strings, whitespace-only strings, non-ASCII characters, out-of-vocabulary terms, or non-string types).
   - Incomplete or corrupted data files must raise clear, descriptive exceptions.
4. **Maintainability & Modularity:**
   - The codebase must be partitioned into decoupled, single-responsibility modules under a clean Python package structure (`sentiment_analyzer`).
   - Clean docstrings, strict type annotations, and 100% test coverage across core functionality must be maintained.
5. **Resource Efficiency:**
   - The serialized model artifact must not exceed 2 MB in storage footprint (achieved: 833 KB).
   - Operational runtime RAM consumption must remain below 150 MB during inference.

---

# 6. System Architecture

The system follows a modular 3-tier architecture separating the Presentation Layer (CLI), Business Logic / ML Engine Layer, and the Data & Persistence Layer.

```
+-----------------------------------------------------------------------+
|                         PRESENTATION LAYER                            |
|  +---------------------------+       +-----------------------------+  |
|  |    Single-Shot CLI        |       |    Interactive REPL Loop    |  |
|  |  (python sentiment.py "…")|       |    (python sentiment.py)    |  |
|  +-------------+-------------+       +--------------+--------------+  |
+----------------|------------------------------------|-----------------+
                 |                                    |
+----------------v------------------------------------v-----------------+
|                      BUSINESS LOGIC & ML LAYER                        |
|                                                                       |
|  +------------------------+          +-----------------------------+  |
|  |  SentimentPredictor    | <------+ |  Trained Model Pipeline     |  |
|  |  (Inference & Probs)   |          |  (TF-IDF + Softmax LogReg)  |  |
|  +-----------+------------+          +--------------^--------------+  |
|              |                                      |                 |
|  +-----------v------------+          +--------------+--------------+  |
|  |  Text Preprocessor     |          |  Model Trainer & Evaluator  |  |
|  |  (Clean & Normalize)   |          |  (80/20 Train-Test Split)   |  |
|  +------------------------+          +--------------^--------------+  |
+-----------------------------------------------------|-----------------+
                                                      |
+-----------------------------------------------------|-----------------+
|                       DATA & STORAGE LAYER          |                 |
|  +----------------------------+      +--------------+--------------+  |
|  |  Raw / Sample Review CSVs  |      |  Serialized Weights File    |  |
|  |  (reviews_sample.csv)      |      |  (sentiment_model.joblib)   |  |
|  +----------------------------+      +-----------------------------+  |
+-----------------------------------------------------------------------+
```

---

# 7. Design Diagrams

### 7.1 Use Case Diagram

```
                 +---------------------------------------------+
                 |            Sentiment Analysis System        |
                 |                                             |
                 |   +-------------------------------------+   |
                 |   |  UC-1: Predict Single Text Review   |   |
                 |   +------------------^------------------+   |
                 |                      |                      |
                 |   +------------------+------------------+   |
                 |   |  UC-2: Interactive REPL Session     |   |
   +--------+    |   +------------------^------------------+   |
   |  User  |--->|                      |                      |
   +--------+    |   +------------------+------------------+   |
                 |   |  UC-3: View Confidence Metrics      |   |
                 |   +-------------------------------------+   |
                 |                                             |
                 |   +-------------------------------------+   |
                 |   |  UC-4: Train / Retrain Model        |   |
                 |   +------------------^------------------+   |
   +--------+    |                      |                      |
   | Admin/ |--->|   +------------------+------------------+   |
   | Dev    |    |   |  UC-5: Run Train/Test Evaluation    |   |
   +--------+    |   +-------------------------------------+   |
                 +---------------------------------------------+
```

### 7.2 Process Workflow Diagram

```
 [User Invokes CLI]
         │
         ▼
 [Model Exists on Disk?] ──(No)──► [Load CSV Dataset]
         │                               │
       (Yes)                             ▼
         │                     [Preprocess & Map Ratings]
         │                               │
         │                               ▼
         │                     [Fit TF-IDF + Logistic Regression]
         │                               │
         │                               ▼
         │                     [Save sentiment_model.joblib]
         │                               │
         ├───────────────────────────────┘
         ▼
 [Load Pipeline into Memory]
         │
 ┌───────┴────────────────────────┐
 │ Single Query                   │ Interactive REPL Loop
 ▼                                ▼
[Sanitize Input Text]            [Prompt User for String "> "]
 │                                │
 ▼                                ▼
[Transform Text via TF-IDF]      [User types "exit"?] ──(Yes)──► [Terminate]
 │                                │ (No)
 ▼                                ▼
[Predict Log-Odds & Softmax]     [Sanitize & Vectorize Input]
 │                                │
 ▼                                ▼
[Format Class & Confidence %]    [Compute Class & Confidence %]
 │                                │
 ▼                                ▼
[Print to stdout: e.g. Pos 99%]  [Print Result to stdout]
                                  │
                                  └─► (Loop back to Prompt)
```

### 7.3 Sequence Diagram

```
User               CLI / main           Predictor          ModelPipeline
 │                     │                    │                    │
 │─── Run Command ────►│                    │                    │
 │                     │─── load_model() ──►│                    │
 │                     │                    │── joblib.load() ──►│
 │                     │                    │◄── pipeline obj ───│
 │                     │◄── predictor ready ┤                    │
 │                     │                    │                    │
 │─── Query Text ─────►│                    │                    │
 │                     │─── predict(text) ─►│                    │
 │                     │                    │── clean_text()     │
 │                     │                    │── pipeline.predict ──►│
 │                     │                    │── predict_proba ──────►│
 │                     │                    │◄── probs / label ─────│
 │                     │◄── (label, conf) ──│                    │
 │◄── "Positive (99%)" ┤                    │                    │
```

### 7.4 Class / Component Diagram

```
+-------------------------------------------------------------------------+
|                          sentiment_analyzer                             |
+-------------------------------------------------------------------------+
|  class SentimentPredictor:                                              |
|    - pipeline: Pipeline                                                 |
|    + __init__(model_path: str, pipeline: Pipeline = None)               |
|    + predict(text: str) -> Tuple[str, float]                            |
|    + predict_detailed(text: str) -> Dict[str, Any]                      |
|    + predict_batch(texts: List[str]) -> List[Tuple[str, float]]         |
+-------------------------------------------------------------------------+
                                    │
                                    │ uses
                                    ▼
+-------------------------------------------------------------------------+
|  module model:                                                          |
|    + build_pipeline(max_features, ngram_range, max_iter) -> Pipeline    |
|    + save_model(pipeline: Pipeline, model_path: str) -> str             |
|    + load_model(model_path: str) -> Pipeline                            |
|    + train_model(csv_path: str, model_path: str, sample_size) -> Pipe   |
+-------------------------------------------------------------------------+
         │                                               │
         │ uses                                          │ evaluates
         ▼                                               ▼
+-----------------------------+         +---------------------------------+
|  module preprocessor:       |         |  module evaluator:              |
|    + clean_text(str) -> str |         |    + evaluate_pipeline(...)     |
|    + map_rating(num) -> str |         |    + run_train_test_eval(...)   |
+-----------------------------+         +---------------------------------+
         │
         │ uses
         ▼
+-------------------------------------+
|  module data_loader:                |
|    + get_data_file() -> str         |
|    + load_dataset(path, nrows) -> df|
+-------------------------------------+
```

### 7.5 Database / Storage Design (Artifact Schema)

The storage design consists of tabular review records and compressed binary model serialization:

**Tabular Review Schema (`reviews_sample.csv`):**

| Column Name | Data Type | Constraint | Description |
|:---|:---:|:---:|:---|
| `rating` | Integer (1–5) | NOT NULL, 1 to 5 | User-assigned product satisfaction score |
| `review_text` | String (Text) | NOT NULL, Non-empty | Unstructured customer opinion text |
| `sentiment` (derived) | String | NOT NULL, Enum | Mapped class: `Negative`, `Neutral`, `Positive` |

**Model Artifact Schema (`sentiment_model.joblib`):**

| Component | Serialized Object | Key Parameters |
|:---|:---|:---|
| Step 1: Feature Extractor | `sklearn.feature_extraction.text.TfidfVectorizer` | `ngram_range=(1,2)`, `max_features=25000`, `sublinear_tf=True` |
| Step 2: Classifier | `sklearn.linear_model.LogisticRegression` | `solver='lbfgs'`, `multi_class='multinomial'`, `max_iter=200`, `C=1.0` |
| Serialization Format | `joblib` binary file | `compress=3`, size = 833 KB |

---

# 8. Design Decisions & Rationale

1. **TF-IDF + Multinomial Logistic Regression vs. Deep Learning (BERT/LSTM):**
   - *Decision:* Select TF-IDF with (1, 2) n-grams combined with Logistic Regression.
   - *Rationale:* For standard text classification of product reviews, linear classifiers on sparse n-gram spaces provide competitive accuracy (~73–75% balanced 3-class) with sub-millisecond CPU inference latency and an 833 KB memory footprint. Deep transformers (e.g. BERT) require GPU infrastructure, 400+ MB model storage, and >200 ms latency per query, which violates our edge/embedded operational constraint.
2. **3-Tier Classification Derived from Star Ratings:**
   - *Decision:* Aggregate 5-star ratings into Negative (1-2), Neutral (3), and Positive (4-5).
   - *Rationale:* Binary classification (Positive vs. Negative) ignores moderate, ambivalent, or mixed feedback. Real-world business operations specifically need to isolate ambivalent (Neutral) reviews to identify opportunities for incremental product improvement.
3. **Sublinear Term Frequency Scaling (`sublinear_tf=True`):**
   - *Decision:* Apply logarithmic term frequency scaling: $1 + \log(\text{tf})$.
   - *Rationale:* Customers writing lengthy reviews frequently repeat common sentiment words (e.g., "good", "great", "bad"). Linear frequency scaling unfairly skews feature weights towards long reviews. Sublinear scaling dampens this effect.
4. **Balanced Stratified Sampling:**
   - *Decision:* Construct a balanced 7,500-sample dataset (2,500 per class) for training and evaluation.
   - *Rationale:* In raw e-commerce data, ~75% of reviews are 5-star. Training on raw distributions yields artificially high accuracy (75%) by predicting "Positive" unconditionally. Balanced sampling forces the model to learn distinct discriminating features across all polarities.

---

# 9. Implementation Details

The implementation comprises 8 dedicated Python modules and 5 test suites structured under clean namespaces:

- **`sentiment_analyzer.config`:** Holds immutable hyperparameter definitions (`TFIDF_MAX_FEATURES = 25000`, `CLF_MAX_ITER = 200`), rating thresholds, and absolute path resolvers.
- **`sentiment_analyzer.preprocessor`:** Encapsulates `clean_text` (regex-driven whitespace normalization, non-destructive punctuation handling) and `map_rating` with strict numeric boundary validation.
- **`sentiment_analyzer.data_loader`:** Resolves data paths dynamically (`reviews_0-250.csv` or `reviews_sample.csv`), verifies CSV header schemas, scrubs missing data, and generates normalized dataframes.
- **`sentiment_analyzer.model`:** Builds the two-stage scikit-learn `Pipeline`, exposes `train_model`, and implements `save_model` / `load_model` utilizing `joblib` level-3 compression.
- **`sentiment_analyzer.evaluator`:** Executes stratified 80/20 train-test splits, calculates macro and weighted classification metrics, and outputs confusion matrix breakdowns.
- **`sentiment_analyzer.predictor`:** Implements the `SentimentPredictor` class providing single prediction, probability distribution extraction, and batch evaluation.
- **`sentiment_analyzer.cli`:** Parses CLI arguments (`--train`, `--evaluate`, positional text), formats output percentages, and drives the interactive REPL loop.
- **`sentiment.py`:** Root entry point providing seamless backward compatibility for existing command-line workflows and API imports.

---

# 10. Screenshots & Results

### 10.1 Experimental Classification Metrics

The model was evaluated using an 80/20 stratified split on the balanced dataset (6,002 train samples, 1,498 test samples):

| Sentiment Class | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **Negative** | 0.7031 | 0.7214 | 0.7122 | 499 |
| **Neutral** | 0.6406 | 0.6380 | 0.6393 | 500 |
| **Positive** | 0.8525 | 0.8337 | 0.8430 | 499 |
| **Overall Accuracy** | **73.10%** | | | **1,498** |
| **Macro Average** | **0.7320** | **0.7310** | **0.7315** | **1,498** |
| **Weighted Average** | **0.7320** | **0.7310** | **0.7314** | **1,498** |

### 10.2 Confusion Matrix Analysis

```
                    Predicted Negative    Predicted Neutral    Predicted Positive
 Actual Negative           360                   117                   22
 Actual Neutral            131                   319                   50
 Actual Positive            21                    62                  416
```

- **Cross-Polarity Accuracy:** The cross-polarity error rate is exceptionally low: only 22 negative samples were misclassified as positive, and only 21 positive samples were misclassified as negative (error rate < 2.9%).
- **Neutral Boundary Dispersion:** As expected in natural language, neutral sentiment exhibits greater semantic overlap with both mild praise and mild complaints.

### 10.3 Terminal Execution Samples

**Single Argument Classification:**
```text
$ python sentiment.py "This product exceeded all my expectations, absolutely love it!"
Positive (99.8%)

$ python sentiment.py "Broke after two days, completely useless."
Negative (98.6%)

$ python sentiment.py "It works okay, nothing special about it."
Neutral (72.4%)
```

**Interactive REPL Execution:**
```text
$ python sentiment.py
--- NLP Sentiment Analyzer (Interactive Mode) ---
Type your review and press Enter. Type 'exit', 'quit', or 'q' to stop.

> The display is vibrant and battery lasts 2 full days!
Positive (98.4%)
> Customer service never responded to my inquiry.
Negative (96.1%)
> Average product, met expectations but nothing more.
Neutral (81.3%)
> exit
Session ended.
```

---

# 11. Testing Approach

Quality assurance is enforced through an automated test suite executed via `pytest`. The suite spans 26 unit and integration test cases covering:

1. **Preprocessing Verification (`test_preprocessor.py`):**
   - Correct mapping across boundary ratings ($1 \rightarrow$ Neg, $2 \rightarrow$ Neg, $3 \rightarrow$ Neu, $4 \rightarrow$ Pos, $5 \rightarrow$ Pos).
   - Validation that non-numeric inputs raise `ValueError`.
   - String normalization, multiple whitespace collapsing, and null safety.
2. **Data Ingestion Verification (`test_data_loader.py`):**
   - Dataset existence, column schema enforcement, and missing file exception raising.
3. **Pipeline & Serialization Verification (`test_model.py`):**
   - Pipeline structure verification (`tfidf` and `clf` named steps).
   - Serialization round-trip and validation that class labels match `VALID_SENTIMENTS`.
4. **Inference & Edge Case Verification (`test_predictor.py`):**
   - Prediction correctness on positive and negative benchmark sentences.
   - Graceful handling of empty strings and whitespace-only strings (defaults safely to Neutral with 50% confidence).
   - Class probability summation to 100%.
   - Batch inference consistency.
5. **CLI Verification (`test_cli.py`):**
   - Argument parsing for query text, `--train`, and `--evaluate` flags.
   - Output string formatting precision.

**Test Run Execution Result:**
```text
============================= test session starts ==============================
platform linux -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/shin/vitya
collected 26 items

tests/test_cli.py ....                                                   [ 15%]
tests/test_data_loader.py ...                                            [ 26%]
tests/test_model.py ..                                                   [ 34%]
tests/test_predictor.py .......                                          [ 61%]
tests/test_preprocessor.py ..........                                    [100%]

============================== 26 passed in 2.18s ==============================
```

---

# 12. Challenges Faced

1. **Extreme Class Skew in Raw Review Corpora:**
   - In raw customer feedback, positive reviews outnumber negative and neutral reviews by a factor of 4:1. Unbalanced training resulted in models achieving high apparent accuracy while having poor recall on negative reviews.
   - *Resolution:* Curated a balanced 7,500-sample dataset using stratified sampling across all three categories.
2. **Ambiguity of Neutral Sentiment:**
   - Neutral customer reviews frequently contain both positive remarks and criticisms (e.g., "Great sound quality but terrible battery life").
   - *Resolution:* Integrated bigrams into the TF-IDF vocabulary (`ngram_range=(1, 2)`), enabling the model to capture contrastive phrases and negations.
3. **Git File Size Constraints:**
   - The full corpus (`reviews_0-250.csv`) exceeds 282 MB, far surpassing GitHub's 100 MB hard limit.
   - *Resolution:* Configured `.gitignore` to exclude large CSVs while bundling the high-quality 7,500-row sample dataset (`reviews_sample.csv`, 3.4 MB) for 100% out-of-the-box reproducibility.

---

# 13. Learnings & Key Takeaways

1. **Feature Engineering vs. Model Complexity:**
   - Carefully tuned TF-IDF vectorization with sublinear scaling and bigrams paired with Logistic Regression provides performance that approaches complex neural networks on standardized review classification tasks, at a fraction of the computational and memory footprint.
2. **Importance of Calibrated Confidence Probabilities:**
   - Binary predictions without confidence metrics are insufficient for automated customer triage. Softmax probability calibration enables downstream workflows to flag low-confidence predictions for human verification.
3. **Clean Software Engineering in Applied AI:**
   - Machine learning code frequently degenerates into monolithic scripts. Transitioning the system to a structured Python package (`sentiment_analyzer`) with automated test suites substantially improved maintainability and reliability.

---

# 14. Future Enhancements

1. **Aspect-Based Sentiment Analysis (ABSA):**
   - Expand the pipeline to extract sentiment toward specific product dimensions (e.g., "Build Quality", "Battery Life", "Customer Support") rather than assigning a single document-level label.
2. **Transformer Fine-Tuning:**
   - Evaluate fine-tuned lightweight transformer backbones (such as DistilBERT or MobileBERT) using ONNX Runtime for edge acceleration.
3. **Containerized REST / FastAPI Microservice:**
   - Package the inference engine into a Docker container exposing a RESTful JSON endpoint for real-time web application integration.
4. **Multilingual Sentiment Classification:**
   - Incorporate multilingual embeddings to process customer reviews written in non-English languages.

---

# 15. References

1. Jurafsky, D., & Martin, J. H. (2024). *Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition* (3rd ed. draft). Pearson.
2. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.
3. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
4. Pang, B., & Lee, L. (2008). *Opinion Mining and Sentiment Analysis*. Foundations and Trends in Information Retrieval, 2(1–2), 1-135.
5. McKinney, W. (2010). *Data Structures for Statistical Computing in Python*. Proceedings of the 9th Python in Science Conference, 51-56.
