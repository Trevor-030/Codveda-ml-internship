"""
Codveda ML Internship - Level 1, Task 3
Implement K-Nearest Neighbors (KNN) Classifier
Dataset: iris.csv (150 flowers, 4 measurements, 3 species)

Objectives covered:
  1. Train a KNN model on a labeled dataset
  2. Evaluate performance using accuracy, confusion matrix, precision/recall
  3. Use different values of K and compare results
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------------------------------------------------
# 1. Load the data
# -----------------------------------------------------------------------
df = pd.read_csv("iris.csv")
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print("Species counts:\n", df["species"].value_counts(), "\n")

# -----------------------------------------------------------------------
# 2. Split features/target, then train/test
# -----------------------------------------------------------------------
X = df.drop(columns=["species"])
y = df["species"]

# stratify=y keeps the same species proportions in both train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------------------------------------------
# 3. Scale features
# -----------------------------------------------------------------------
# KNN measures distance between points, so features must be on the same
# scale - otherwise a feature with naturally larger numbers (e.g. petal
# length in cm) would dominate the distance calculation over one with
# smaller numbers, even if both matter equally for classification.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------------------
# 4. Train KNN with different values of K and compare
# -----------------------------------------------------------------------
k_values = [1, 3, 5, 7, 9, 11, 15]
results = []

print("Comparing different values of K:")
print(f"{'K':>3} | {'Accuracy':>8}")
print("-" * 16)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    preds = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    results.append((k, acc))
    print(f"{k:>3} | {acc:>8.4f}")

best_k, best_acc = max(results, key=lambda r: r[1])
print(f"\nBest K on this test split: K={best_k} (accuracy={best_acc:.4f})")

# -----------------------------------------------------------------------
# 5. Detailed evaluation for the best K
# -----------------------------------------------------------------------
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train_scaled, y_train)
y_pred = final_model.predict(X_test_scaled)

print(f"\n--- Detailed evaluation for K={best_k} ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")

print("Confusion matrix (rows=actual, cols=predicted):")
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print(cm_df, "\n")

print("Precision / recall / F1 per class:")
print(classification_report(y_test, y_pred))

# Save the K comparison and final predictions
pd.DataFrame(results, columns=["K", "accuracy"]).to_csv("task3_k_comparison.csv", index=False)
pd.DataFrame({"actual": y_test.values, "predicted": y_pred}).to_csv("task3_predictions.csv", index=False)
print("Saved: task3_k_comparison.csv, task3_predictions.csv")
