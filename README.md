# Codveda Technologies — Machine Learning Internship

Completed tasks for all 3 levels of the Codveda ML internship task list.
Each level required at least 2 of 3 tasks — all 9 tasks were completed
across the 3 levels.

## Datasets used

| Dataset | Source | Used in |
|---|---|---|
| Boston Housing (`house_data_raw.csv`) | Codveda-provided | Level 1: Tasks 1, 2 |
| Iris (`iris.csv`) | Codveda-provided | Level 1: Task 3; Level 2: Task 2 |
| Telecom Churn (`churn-bigml-80/20.csv`) | Codveda-provided (BigML) | Level 2: Tasks 1, 3; Level 3: Tasks 1, 2 |
| Digits (scikit-learn built-in) | scikit-learn | Level 3: Task 3 (substituted for MNIST — see note below) |

**Note on Level 3 Task 3:** the task suggests MNIST, but downloading it
requires a host outside this environment's network allowlist. scikit-learn's
built-in `digits` dataset (8x8 handwritten digit images, 10 classes) was
used instead — the same classification task in spirit, just smaller
images, so every objective (architecture design, backpropagation, loss
curves) is demonstrated identically.

## Level 1 — Basic

| Task | Script | Summary |
|---|---|---|
| 1. Data Preprocessing | `task1_data_preprocessing.py` | Cleaned Boston Housing data: no missing values found (code still handles them generically), one-hot encoded the categorical `RAD` column, standardized numeric features, split 80/20 train/test. |
| 2. Linear Regression | `task2_linear_regression.py` | Predicted house prices (MEDV). R² = 0.66, RMSE ≈ $4,980. Strongest predictors: `LSTAT` (% lower-status population, negative), `RM` (rooms, positive). |
| 3. KNN Classifier | `task3_knn_classifier.py` | Classified iris species. Compared K = 1–15; best test accuracy 96.7% (K=1), with only 1 virginica/versicolor confusion — a known overlap in this dataset. |

## Level 2 — Intermediate

| Task | Script | Summary |
|---|---|---|
| 1. Logistic Regression | `level2_task1_logistic_regression.py` | Predicted customer churn. Accuracy 85.5%, but recall was low (0.19) due to class imbalance (~14.6% churn rate) — ROC AUC of 0.826 was the more reliable metric. Top odds-ratio drivers: customer service calls, international plan. |
| 2. Decision Tree | `level2_task2_decision_tree.py` | Classified iris species. Compared `max_depth` 1–None to demonstrate overfitting (unpruned tree hit 100% train accuracy but only 93.3% test); `max_depth=3` gave the best test accuracy (96.7%) with minimal overfitting. |
| 3. K-Means Clustering | `level2_task3_kmeans.py` | Unsupervised customer segmentation (Churn label withheld from the model). Elbow method selected K=4. One resulting cluster showed a 42% real churn rate vs. the 14.6% average — a useful high-risk segment discovered without any labels. |

## Level 3 — Advanced

| Task | Script | Summary |
|---|---|---|
| 1. Random Forest | `level3_task1_random_forest.py` | Predicted churn. GridSearchCV (5-fold CV) tuned `n_estimators`/`max_depth`. Test F1 = 0.842 — a major jump over logistic regression's 0.271, thanks to capturing non-linear feature interactions. |
| 2. SVM | `level3_task2_svm.py` | Predicted churn. Compared linear vs. RBF kernels — RBF won decisively (F1 0.701 vs 0.492), confirming the churn/usage relationship is non-linear. Included a 2-feature decision boundary visualization for illustration. |
| 3. Neural Network | `level3_task3_neural_network.py` | Feed-forward network (64 → 64 → 32 → 10, ReLU + softmax) built with TensorFlow/Keras for digit classification. Test accuracy 96.9%. Training/validation loss and accuracy curves plotted to check for overfitting. |

## Tools

Python, pandas, scikit-learn, matplotlib, seaborn, TensorFlow/Keras.

## How to run

Each script is self-contained — install requirements, place the relevant
dataset CSV(s) in the same directory as the script, and run with
`python3 <script_name>.py`. Level 1 Task 1 must be run before Level 1
Task 2, since Task 2 reuses its saved train/test splits.

```bash
pip install pandas scikit-learn matplotlib seaborn tensorflow-cpu --break-system-packages
```
