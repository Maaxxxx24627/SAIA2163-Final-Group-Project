# Models

This folder contains two sets of classical ML sentiment classification models.

## final_best/

These are the three models to use. They are the best-performing baseline models,
evaluated against a 402-row human-labeled gold test set.

- rakyat_speaks_ml_NB_TFIDF_FINAL_BEST.pkl
  Naive Bayes with TF-IDF features. Accuracy 64.2 percent, Cohen's Kappa 0.373.
  Use with rakyat_speaks_ml_NB_TFIDF_FINAL_BEST_vectorizer.pkl.

- rakyat_speaks_ml_SVM_BoW_FINAL_BEST.pkl
  Support Vector Machine with Bag of Words features. Accuracy 68.4 percent,
  Cohen's Kappa 0.366. Highest accuracy of the three models.
  Use with rakyat_speaks_ml_SVM_BoW_FINAL_BEST_vectorizer.pkl.

- rakyat_speaks_ml_LogReg_TFIDF_FINAL_BEST.pkl
  Logistic Regression with TF-IDF features. Accuracy 66.9 percent,
  Cohen's Kappa 0.365. Most balanced performance across classes.
  Use with rakyat_speaks_ml_LogReg_TFIDF_FINAL_BEST_vectorizer.pkl.

Each model must be loaded together with its matching vectorizer file. Do not mix
a model with a vectorizer from a different model, since vocabulary and feature
settings differ between them.

## experiments_tuned_underperformed/

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

## Full results

See results/model_results.csv for the complete comparison table across all eight
model and configuration combinations, including the five baseline combinations
and the three tuned combinations, with a status column indicating which models
are final_best, baseline_not_selected, or tuned_underperformed.