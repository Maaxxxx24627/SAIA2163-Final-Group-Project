# Pipeline Notebook

**File:** `pipeline.ipynb`

## What it does

This notebook trains and evaluates four machine learning models for sentiment classification of Malaysian public opinion on government policies. It loads 7,383 Gemini-labeled training rows, preprocesses the text, extracts features using TF-IDF and Bag of Words, and compares Naive Bayes vs SVM performance against a human-labeled gold standard.

### Steps

1. **Load Data** — 7,383 train rows from `malaysian_sentiment_labeled.csv`, 402 test rows from `gold_labeled.csv`
2. **Preprocessing** — lowercasing, URL/markdown/emoji stripping, HTML entity cleanup, short row filtering
3. **Feature Extraction** — TF-IDF and CountVectorizer (unigrams+bigrams, 30k features, English stopwords)
4. **Model Training** — MultinomialNB and LinearSVC, each with both TF-IDF and BoW features
5. **Evaluation** — accuracy, Cohen's Kappa, per-class F1, confusion matrices
6. **Save Best** — best model by Kappa saved to `models/best_model.pkl`

## What it tells us

- Which feature representation (TF-IDF vs BoW) works better for Manglish social media text
- Whether SVM outperforms Naive Bayes on this imbalanced dataset
- Per-class performance — how well each model detects rare positive comments vs majority neutral
- Kappa scores corrected for chance agreement with human labels
- Confusion matrices showing where each model misclassifies
- Final pickled model for deployment in the Streamlit app