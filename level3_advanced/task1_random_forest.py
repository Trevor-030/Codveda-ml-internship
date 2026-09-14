"""
Codveda ML Internship - Level 3, Task 1
Build a Random Forest Classifier
Dataset: BigML Telecom Churn (same pre-split 80/20 as Level 2 Task 1)

Objectives covered:
  1. Train a Random Forest and tune hyperparameters (n_estimators, max_depth)
  2. Evaluate using cross-validation and classification metrics
  3. Perform feature importance analysis
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# -----------------------------------------------------------------------
# 1. Load and preprocess (same approach as Level 2 Task 1)
# -----------------------------------------------------------------------
def preprocess(df):
    df = df.copy()
    df = df.drop(columns=["State", "Area code"])
    df["International plan"] = (df["International plan"] == "Yes").astype(int)
    df["Voice mail plan"] = (df["Voice mail plan"] == "Yes").astype(int)
    df["Churn"] = df["Churn"].astype(int)
    return df

train_df = preprocess(pd.read_csv("..\\data\\churn-bigml-80.csv"))
test_df = preprocess(pd.read_csv("..\\data\\churn-bigml-20.csv"))

X_train = train_df.drop(columns=["Churn"])
y_train = train_df["Churn"]
X_test = test_df.drop(columns=["Churn"])
y_test = test_df["Churn"]

print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# Note: Random Forest, like a single decision tree, splits on raw
# thresholds - no feature scaling needed here, unlike logistic regression.

# -----------------------------------------------------------------------
# 2. Tune hyperparameters with cross-validation (grid search)
# -----------------------------------------------------------------------
# GridSearchCV tries every combination below, and for each one runs
# 5-fold cross-validation on the TRAINING data: split train into 5 parts,
# train on 4, validate on the 5th, rotate 5 times, average the score.
# This picks hyperparameters using only training data - the test set
# stays completely untouched until final evaluation.
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight="balanced"),
    param_grid,
    cv=5,
    scoring="f1",  # optimize for F1, not accuracy, given the class imbalance
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

print("\nHyperparameter search results:")
results_df = pd.DataFrame(grid_search.cv_results_)[
    ["param_n_estimators", "param_max_depth", "mean_test_score"]
].sort_values("mean_test_score", ascending=False)
print(results_df.to_string(index=False))

print(f"\nBest params: {grid_search.best_params_}")
print(f"Best cross-validated F1: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_

# -----------------------------------------------------------------------
# 3. Cross-validation score for the final chosen model (extra confirmation)
# -----------------------------------------------------------------------
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="f1")
print(f"\n5-fold CV F1 scores: {np.round(cv_scores, 4)}")
print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# -----------------------------------------------------------------------
# 4. Evaluate on the held-out test set
# -----------------------------------------------------------------------
y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n--- Test set evaluation ---")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nConfusion matrix (rows=actual, cols=predicted) [0=stayed, 1=churned]:")
print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                    index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"]))

# -----------------------------------------------------------------------
# 5. Feature importance analysis
# -----------------------------------------------------------------------
importances = pd.Series(best_model.feature_importances_, index=X_train.columns) \
    .sort_values(ascending=False)

print("\nTop 10 most important features:")
print(importances.head(10).round(4))

importances.to_csv("task1_rf_feature_importance.csv", header=["importance"])
pd.DataFrame({"actual": y_test.values, "predicted": y_pred}).to_csv(
    "task1_rf_predictions.csv", index=False
)
print("\nSaved: task1_rf_feature_importance.csv, task1_rf_predictions.csv")
