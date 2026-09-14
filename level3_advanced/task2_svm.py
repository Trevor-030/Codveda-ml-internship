"""
Codveda ML Internship - Level 3, Task 2
Support Vector Machine (SVM) for Classification
Dataset: BigML Telecom Churn (same binary churn problem as Tasks before)

Objectives covered:
  1. Train an SVM model on a labeled dataset
  2. Use different kernels (linear, RBF) and compare performance
  3. Visualize the decision boundary
  4. Evaluate using accuracy, precision, recall, and AUC
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# -----------------------------------------------------------------------
# 1. Load and preprocess (same as previous churn tasks)
# -----------------------------------------------------------------------
def preprocess(df):
    df = df.copy()
    df = df.drop(columns=["State", "Area code"])
    df["International plan"] = (df["International plan"] == "Yes").astype(int)
    df["Voice mail plan"] = (df["Voice mail plan"] == "Yes").astype(int)
    df["Churn"] = df["Churn"].astype(int)
    return df

train_df = preprocess(pd.read_csv("churn-bigml-80.csv"))
test_df = preprocess(pd.read_csv("churn-bigml-20.csv"))

X_train = train_df.drop(columns=["Churn"])
y_train = train_df["Churn"]
X_test = test_df.drop(columns=["Churn"])
y_test = test_df["Churn"]

# SVM finds the widest possible margin between classes using distances,
# so - like KNN and logistic regression - features must be scaled or
# large-range features (minutes) would swamp small-range ones (0/1 flags).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------------------
# 2. Train SVM with different kernels and compare
# -----------------------------------------------------------------------
# class_weight="balanced" - as with Random Forest - tells the SVM to
# penalize mistakes on the minority (churn) class more heavily, since
# only ~14.6% of customers actually churn.
kernels = ["linear", "rbf"]
kernel_results = {}

print("Comparing SVM kernels:")
print(f"{'kernel':>8} | {'accuracy':>8} | {'precision':>9} | {'recall':>6} | {'f1':>6} | {'auc':>6}")

for kernel in kernels:
    svm = SVC(kernel=kernel, probability=True, class_weight="balanced", random_state=42)
    svm.fit(X_train_scaled, y_train)
    preds = svm.predict(X_test_scaled)
    proba = svm.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)

    kernel_results[kernel] = {"model": svm, "accuracy": acc, "precision": prec,
                               "recall": rec, "f1": f1, "auc": auc}
    print(f"{kernel:>8} | {acc:>8.4f} | {prec:>9.4f} | {rec:>6.4f} | {f1:>6.4f} | {auc:>6.4f}")

best_kernel = max(kernel_results, key=lambda k: kernel_results[k]["f1"])
print(f"\nBest kernel by F1-score: {best_kernel}")

# -----------------------------------------------------------------------
# 3. Visualize the decision boundary
# -----------------------------------------------------------------------
# A real decision boundary lives in 17-dimensional space and can't be
# drawn. To visualize the CONCEPT, we retrain small 2D-only SVMs (one
# per kernel) using just the two strongest churn predictors identified
# by Random Forest: Customer service calls and Total day minutes.
feat_x, feat_y = "Customer service calls", "Total day minutes"
X_train_2d = X_train_scaled[:, [X_train.columns.get_loc(feat_x), X_train.columns.get_loc(feat_y)]]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
xx, yy = np.meshgrid(
    np.linspace(X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1, 300),
    np.linspace(X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1, 300),
)

for ax, kernel in zip(axes, kernels):
    svm_2d = SVC(kernel=kernel, class_weight="balanced", random_state=42)
    svm_2d.fit(X_train_2d, y_train)

    Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
    ax.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, cmap="coolwarm",
               edgecolors="k", s=15, alpha=0.7)
    ax.set_title(f"SVM decision boundary - {kernel} kernel")
    ax.set_xlabel(f"{feat_x} (scaled)")
    ax.set_ylabel(f"{feat_y} (scaled)")

plt.tight_layout()
plt.savefig("task2_svm_decision_boundary.png", dpi=150)
print("Saved decision boundary plot to task2_svm_decision_boundary.png")
print("(Note: this 2-feature view is for illustration only - the models")
print(" compared above used all features, not just these two.)")

# -----------------------------------------------------------------------
# 4. Final evaluation summary for the best full-feature model
# -----------------------------------------------------------------------
best_model = kernel_results[best_kernel]["model"]
y_pred = best_model.predict(X_test_scaled)

print(f"\n--- Final evaluation: SVM ({best_kernel} kernel), all features ---")
for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
    print(f"{metric.capitalize()}: {kernel_results[best_kernel][metric]:.4f}")

pd.DataFrame({"actual": y_test.values, "predicted": y_pred}).to_csv(
    "task2_svm_predictions.csv", index=False
)
print("\nSaved predictions to task2_svm_predictions.csv")
