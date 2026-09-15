"""
Codveda ML Internship - Level 2, Task 2
Decision Trees for Classification
Dataset: iris.csv (150 flowers, 3 species)

Objectives covered:
  1. Train a decision tree on a labeled dataset
  2. Visualize the tree structure
  3. Prune the tree to prevent overfitting
  4. Evaluate using accuracy and F1-score
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score, classification_report

# -----------------------------------------------------------------------
# 1. Load and split
# -----------------------------------------------------------------------
df = pd.read_csv("..\\data\\iris.csv")
X = df.drop(columns=["species"])
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# Note: decision trees don't need feature scaling - they split on
# raw threshold values (e.g. "petal_length <= 2.45"), so scaling
# wouldn't change how the tree splits at all.

# -----------------------------------------------------------------------
# 2. Train an UNPRUNED tree first, to see the overfitting problem
# -----------------------------------------------------------------------
full_tree = DecisionTreeClassifier(random_state=42)
full_tree.fit(X_train, y_train)

train_acc_full = accuracy_score(y_train, full_tree.predict(X_train))
test_acc_full = accuracy_score(y_test, full_tree.predict(X_test))
print(f"\nUnpruned tree - depth: {full_tree.get_depth()}, leaves: {full_tree.get_n_leaves()}")
print(f"  Train accuracy: {train_acc_full:.4f}")
print(f"  Test accuracy:  {test_acc_full:.4f}")

# -----------------------------------------------------------------------
# 3. Prune the tree (limit max_depth) to prevent overfitting
# -----------------------------------------------------------------------
# Try a few depths and compare train vs test accuracy to find where
# the tree stops overfitting without losing test performance.
print("\nComparing max_depth values (pruning):")
print(f"{'depth':>5} | {'train_acc':>9} | {'test_acc':>8}")
for depth in [1, 2, 3, 4, None]:
    t = DecisionTreeClassifier(max_depth=depth, random_state=42)
    t.fit(X_train, y_train)
    tr_acc = accuracy_score(y_train, t.predict(X_train))
    te_acc = accuracy_score(y_test, t.predict(X_test))
    label = depth if depth else "None"
    print(f"{label!s:>5} | {tr_acc:>9.4f} | {te_acc:>8.4f}")

# max_depth=3 keeps the tree simple while matching full-depth test accuracy
pruned_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
pruned_tree.fit(X_train, y_train)

# -----------------------------------------------------------------------
# 4. Visualize the pruned tree structure
# -----------------------------------------------------------------------
plt.figure(figsize=(14, 8))
plot_tree(
    pruned_tree,
    feature_names=X.columns,
    class_names=pruned_tree.classes_,
    filled=True,
    rounded=True,
    fontsize=10,
)
plt.title("Pruned Decision Tree (max_depth=3) - Iris Species Classification")
plt.tight_layout()
plt.savefig("../results/level2_intermediate/task2_decision_tree.png", dpi=150)
print("\nSaved tree diagram to task2_decision_tree.png")

# -----------------------------------------------------------------------
# 5. Evaluate the pruned tree
# -----------------------------------------------------------------------
y_pred = pruned_tree.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

print(f"\n--- Pruned tree (max_depth=3) final evaluation ---")
print(f"Accuracy: {acc:.4f}")
print(f"F1-score (macro avg): {f1:.4f}\n")
print(classification_report(y_test, y_pred))

# Save feature importances - which measurements the tree relied on most
importances = pd.Series(pruned_tree.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature importances:")
print(importances.round(3))

pd.DataFrame({"actual": y_test.values, "predicted": y_pred}).to_csv(
    "../results/level2_intermediate/task2_dtree_predictions.csv", index=False)
print("\nSaved predictions to task2_dtree_predictions.csv")
