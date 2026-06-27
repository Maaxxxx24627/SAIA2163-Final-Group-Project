# Models

This folder contains two sets of classical ML sentiment classification models,
plus transformer-based models in the `transformers/` subdirectory.

> **Note:** The `classical_ml/` folder contains the best-performing **classical
> machine learning models** specifically. It is **not** the overall best model
> for the project. The overall best model project-wide is the fine-tuned
> XLM-RoBERTa transformer ensemble, which is documented separately in the
> `transformers/` section below and is not stored in this repository (it is
> loaded from Hugging Face Hub or local checkpoint).

## classical_ml/

These are the three models to use. They are the best-performing baseline models,
evaluated against a 402-row human-labeled gold test set.

- naive_bayes_tfidf.pkl
  Naive Bayes with TF-IDF features. Accuracy 64.2 percent, Cohen's Kappa 0.373.
  Use with naive_bayes_tfidf_vectorizer.pkl.

- svm_bow.pkl
  Support Vector Machine with Bag of Words features. Accuracy 68.4 percent,
  Cohen's Kappa 0.366. Highest accuracy of the three models.
  Use with svm_bow_vectorizer.pkl.

- logistic_regression_tfidf.pkl
  Logistic Regression with TF-IDF features. Accuracy 66.9 percent,
  Cohen's Kappa 0.365. Most balanced performance across classes.
  Use with logistic_regression_tfidf_vectorizer.pkl.

Each model must be loaded together with its matching vectorizer file. Do not mix
a model with a vectorizer from a different model, since vocabulary and feature
settings differ between them.

## classical_ml_tuned_experiments/

These are hyperparameter-tuned versions of the same three models, produced using
GridSearchCV with cross-validation on the training data. They are kept only as
documented evidence for the technical report.

These models scored worse than the baseline models above on every evaluation
metric (accuracy, Cohen's Kappa, and F1-score for all three sentiment classes).
This happened because the training data is approximately 4 percent positive
sentiment, while the gold test set is approximately 9 percent positive sentiment.
Hyperparameters selected to perform well on training data did not transfer well
to the differently distributed test set.

Do not use these models in the Streamlit application or for any final reporting
of model performance. They exist only to document the tuning experiment.


## transformers/

This folder contains transformer-based sentiment classification models, in
addition to the classical ML models above.

- bestfinedtuned_trans_xmlr/
  XLM-RoBERTa (xlm-roberta-base), fine-tuned on the training dataset for 6 epochs.
  This is the best-performing model in the entire project across all approaches
  (classical and transformer). Accuracy 68.9 percent, Cohen's Kappa 0.390.
  Load with AutoModelForSequenceClassification.from_pretrained() and
  AutoTokenizer.from_pretrained(), both pointed at this folder. Labels are
  0 equals negative, 1 equals neutral, 2 equals positive.

- xlmr_finetuned_checkpoints/
  Leftover training checkpoint files from the Hugging Face Trainer. Not needed
  for inference or deployment. Safe to delete.

A hyperparameter sweep was conducted on training epochs (3, 6, 6.5, 7, and 9
epochs). Accuracy and Kappa peaked sharply at 6 epochs and declined at every
epoch count tested beyond that, indicating overfitting past this point. All
five results are recorded in results/model_results.csv for reference.

## Zero-shot transformer

A zero-shot baseline was also evaluated using cardiffnlp/twitter-xlm-roberta-base-sentiment,
a model pretrained for multilingual social media sentiment with no fine-tuning
on this project's data. It requires no local model files and is loaded directly
from the Hugging Face Hub at runtime using the transformers pipeline function.
It scored 48.3 percent accuracy and Cohen's Kappa of 0.169 on the gold test set,
substantially below the fine-tuned model, demonstrating the value of domain-specific
fine-tuning over generic pretrained sentiment models for this task.


## Full results

See results/model_results.csv for the complete comparison table across all eight
model and configuration combinations, including the five baseline combinations
and the three tuned combinations, with a status column indicating which models
are classical_ml, baseline_not_selected, or tuned_underperformed.