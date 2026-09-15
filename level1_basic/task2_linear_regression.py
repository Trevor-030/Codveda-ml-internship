"""
Codveda ML Internship - Level 1, Task 2
Build a Simple Linear Regression Model
Dataset: house Prediction Data Set.csv (Boston Housing), pre-processed in Task 1

Objectives covered:
  1. Load a dataset and preprocess it        -> reuses Task 1's saved splits
  2. Train a linear regression model         -> sklearn LinearRegression
  3. Interpret the model coefficients
  4. Evaluate using R-squared and MSE
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------------------------------------------------------
# 1. Load the already-preprocessed data from Task 1
# -----------------------------------------------------------------------
X_train = pd.read_csv("../results/level1_basic/X_train.csv")
X_test = pd.read_csv("../results/level1_basic/X_test.csv")
y_train = pd.read_csv("../results/level1_basic/y_train.csv").squeeze("columns")  # read as Series, not 1-col DataFrame
y_test = pd.read_csv("../results/level1_basic/y_test.csv").squeeze("columns")

print(f"Training on {X_train.shape[0]} houses, {X_train.shape[1]} features")
print(f"Testing on {X_test.shape[0]} houses\n")

# -----------------------------------------------------------------------
# 2. Train the linear regression model
# -----------------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------------------------------------------------
# 3. Interpret the model coefficients
# -----------------------------------------------------------------------
# Each coefficient tells us: holding every other feature fixed, how much
# does the predicted price (MEDV, in $1000s) change for a 1-unit increase
# in that (standardized) feature?
coefficients = pd.Series(model.coef_, index=X_train.columns).sort_values(key=abs, ascending=False)

print("Model intercept (baseline predicted price):", round(model.intercept_, 3))
print("\nTop 10 most influential features (by absolute coefficient size):")
print(coefficients.head(10).round(3))

# -----------------------------------------------------------------------
# 4. Evaluate the model
# -----------------------------------------------------------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f"\nR-squared (R^2):        {r2:.4f}")
print(f"Mean Squared Error:     {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}  (in $1000s, same units as MEDV)")

# Save predictions alongside actual values for inspection
results = pd.DataFrame({"actual_MEDV": y_test, "predicted_MEDV": y_pred})
results["error"] = results["actual_MEDV"] - results["predicted_MEDV"]
results.to_csv("task2_predictions.csv", index=False)
print("\nSaved predictions to task2_predictions.csv")
