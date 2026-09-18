# Project Problem Statement & Scope

## 1. Problem Statement

Modern e-commerce ecosystems, review aggregators, and digital service platforms generate millions of unstructured, text-based customer reviews daily. While this feedback holds critical intelligence regarding product quality, customer satisfaction, and operational defects, businesses face severe bottlenecks in extracting actionable insights:

1. **Information Overload & Scale:** Manual inspection and categorization of hundreds of thousands of customer reviews is economically infeasible and operationally slow.
2. **Subjectivity & Human Inconsistency:** Manual human moderation introduces cognitive fatigue and personal bias, leading to inconsistent sentiment classification across products and categories.
3. **Severe Class Imbalance:** Real-world e-commerce platforms suffer from acute rating skew (typically >70% 4-star and 5-star reviews), causing naive automated models to overpredict positive sentiment and miss critical negative signals.
4. **Latency & Production Constraints:** Heavy neural network and large language model (LLM) architectures often incur prohibitive latency (>500ms) and infrastructure costs for real-time customer feedback triage and batch analytics pipelines.

There is a compelling need for a lightweight, deterministic, high-throughput, and robust Natural Language Processing (NLP) system capable of accurately classifying customer sentiment into discrete categories with calibrated confidence scores.

---

## 2. Scope of the Project

The scope of this project encompasses the design, implementation, evaluation, and packaging of an end-to-end sentiment classification pipeline:

- **In-Scope:**
  - **Data Ingestion & Cleaning:** Ingestion of raw customer review datasets (supporting both full multi-gigabyte corpora and stratified evaluation samples), text sanitization, whitespace normalization, and handling of null/empty text.
  - **Categorical Transformation:** Transformation of 1-to-5 star discrete customer ratings into a calibrated 3-tier sentiment taxonomy:
    - **Positive:** Ratings 4 and 5
    - **Neutral:** Rating 3
    - **Negative:** Ratings 1 and 2
  - **Feature Engineering:** Extraction of unigram and bigram contextual tokens using sublinear Term Frequency-Inverse Document Frequency (`TfidfVectorizer`) bounded to 25,000 top features.
  - **Classification Model:** Training a multi-class multinomial Logistic Regression classifier with L2 regularization and softmax probability outputs.
  - **Inference Interfaces:** Providing dual interaction modalities:
    - Single-review command-line argument inference.
    - An interactive Read-Eval-Print-Loop (REPL) terminal interface.
    - Programmatic Python API (`SentimentPredictor`, `predict`, `load_model`).
  - **Model Persistence & Packaging:** Efficient serialization of model pipelines using `joblib` compression (<1 MB footprint) for sub-second cold starts.
  - **Verification & Testing:** Automated unit and integration testing suite utilizing `pytest`.

- **Out-of-Scope (Future Work):**
  - Real-time web scraping / crawling of live e-commerce sites.
  - Multilingual sentiment classification (currently English review focused).
  - Heavy GPU-dependent transformer fine-tuning (e.g., RoBERTa/LLaMA), which exceeds edge and low-compute deployment parameters.

---

## 3. Target Users

The system is designed to serve multiple stakeholder personas across technical and business domains:

1. **E-Commerce & Product Managers:**
   - Rapidly gauge consumer reception of newly launched products.
   - Aggregate sentiment across product versions and identify recurring defects or customer grievances.
2. **Customer Experience & Support Teams:**
   - Prioritize urgent customer service tickets by triaging negative feedback in real time.
   - Detect severe complaints before public escalation on social media.
3. **Data Scientists & Machine Learning Engineers:**
   - Integrate a lightweight, reproducible sentiment inference module into larger analytics pipelines, ETL jobs, or microservice architectures.
4. **Academic & Research Evaluators:**
   - Assess NLP text categorization techniques, linear classifiers on sparse high-dimensional token spaces, and handling class-imbalanced corpora.

---

## 4. High-Level Features

| Feature Module | Description | Technical Implementation |
| :--- | :--- | :--- |
| **Data Ingestion & Preprocessing** | Loads raw review CSVs, performs text cleaning, sanitizes inputs, and maps ratings to sentiment labels. | `sentiment_analyzer.data_loader`, `sentiment_analyzer.preprocessor` |
| **TF-IDF Feature Extraction** | Extracts unigram and bigram n-grams, applies sublinear TF scaling, and manages a 25,000-word vocabulary. | `sklearn.feature_extraction.text.TfidfVectorizer` |
| **Multiclass Classifier** | Classifies sentiment into Positive, Neutral, or Negative with calibrated softmax confidence scores. | `sklearn.linear_model.LogisticRegression` |
| **Model Persistence & Auto-Discovery** | Automatically discovers serialized weights; triggers training automatically if weights are missing. | `sentiment_analyzer.model`, `joblib` |
| **Interactive Terminal REPL** | Terminal loop allowing users to test reviews dynamically with instant feedback. | `sentiment_analyzer.cli.interactive_mode` |
| **CLI & Batch Inference** | One-line terminal commands for single review classification and automated script pipelines. | `sentiment.py`, `sentiment_analyzer.cli` |
| **Automated Testing Suite** | Comprehensive unit and integration test coverage for all components and edge cases. | `pytest` test suite in `tests/` |
