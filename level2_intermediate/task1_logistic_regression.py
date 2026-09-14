"""
Codveda ML Internship - Level 2, Task 1
Logistic Regression for Binary Classification
Dataset: BigML Telecom Churn (pre-split 80/20 by the data provider)

Objectives covered:
  1. Load and preprocess the dataset
  2. Train a logistic regression model
  3. Interpret model coefficients and the odds ratio
  4. Evaluate using accuracy, precision, recall, and the ROC curve
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display available in this environment
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score
)

# -----------------------------------------------------------------------
# 1. Load the pre-split train/test data
# -----------------------------------------------------------------------
train_df = pd.read_csv("..\\data\\churn-bigml-80.csv")
test_df = pd.read_csv("..\\data\\churn-bigml-20.csv")
print(f"Train: {train_df.shape[0]} customers | Test: {test_df.shape[0]} customers")
print(f"Churn rate in train: {train_df['Churn'].mean():.1%}\n")

# -----------------------------------------------------------------------
# 2. Preprocess
# -----------------------------------------------------------------------
def preprocess(df):
    df = df.copy()
    # 'State' has 51 distinct values with weak individual signal for churn,
    # and 'Area code' is just a phone-routing artifact, not a real feature -
    # both are dropped to keep the model compact and interpretable.
    df = df.drop(columns=["State", "Area code"])

    # Binary Yes/No columns -> 1/0
    df["International plan"] = (df["International plan"] == "Yes").astype(int)
    df["Voice mail plan"] = (df["Voice mail plan"] == "Yes").astype(int)

    # Target: True/False -> 1/0
    df["Churn"] = df["Churn"].astype(int)
    return df

train_df = preprocess(train_df)
test_df = preprocess(test_df)

X_train = train_df.drop(columns=["Churn"])
y_train = train_df["Churn"]
X_test = test_df.drop(columns=["Churn"])
y_test = test_df["Churn"]

# Scale numeric features - logistic regression's optimizer converges
# faster and more reliably when features are on comparable scales.
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# -----------------------------------------------------------------------
# 3. Train logistic regression
# -----------------------------------------------------------------------
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# -----------------------------------------------------------------------
# 4. Interpret coefficients and odds ratios
# -----------------------------------------------------------------------
# Logistic regression predicts log-odds of churn as a linear combination
# of features. Exponentiating a coefficient gives the ODDS RATIO: how
# much the odds of churning get multiplied by, per 1-unit (1 std-dev,
# since features are scaled) increase in that feature.
coef_table = pd.DataFrame({
    "feature": X_train.columns,
    "coefficient": model.coef_[0],
    "odds_ratio": np.exp(model.coef_[0])
}).sort_values("coefficient", key=abs, ascending=False)

print("Top features by influence on churn (odds ratio > 1 = increases churn odds):")
print(coef_table.head(8).round(3).to_string(index=False))

# -----------------------------------------------------------------------
# 5. Evaluate
# -----------------------------------------------------------------------
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]  # probability of churn

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"\nAccuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC AUC:   {auc:.4f}")

print("\nConfusion matrix (rows=actual, cols=predicted) [0=stayed, 1=churned]:")
print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                    index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"]))

# -----------------------------------------------------------------------
# 6. Plot the ROC curve
# -----------------------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.3f})", color="darkorange")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Churn Prediction (Logistic Regression)")
plt.legend()
plt.tight_layout()
plt.savefig("task1_roc_curve.png", dpi=150)
print("\nSaved ROC curve to task1_roc_curve.png")

# Save predictions
results = X_test.copy()
results["actual_churn"] = y_test.values
results["predicted_churn"] = y_pred
results["churn_probability"] = y_proba.round(4)
results.to_csv("task1_logreg_predictions.csv", index=False)
print("Saved predictions to task1_logreg_predictions.csv")
