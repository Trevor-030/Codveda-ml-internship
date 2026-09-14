"""
Codveda ML Internship - Level 1, Task 1
Data Preprocessing for Machine Learning
Dataset: house Prediction Data Set.csv (Boston Housing dataset, 506 rows, 14 features)

Objectives covered:
  1. Handle missing data
  2. Encode categorical variables (one-hot encoding)
  3. Normalize/standardize numerical features
  4. Split into training and testing sets
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------
# 1. Load the data
# -----------------------------------------------------------------------
# The raw file has no header row and is whitespace-delimited (not comma),
# so we supply the standard Boston Housing column names ourselves.
COLUMN_NAMES = [
    "CRIM",    # per-capita crime rate by town
    "ZN",      # proportion of residential land zoned for large lots
    "INDUS",   # proportion of non-retail business acres per town
    "CHAS",    # Charles River dummy variable (1 if tract bounds river, else 0)
    "NOX",     # nitric oxide concentration
    "RM",      # average number of rooms per dwelling
    "AGE",     # proportion of owner-occupied units built before 1940
    "DIS",     # weighted distance to employment centres
    "RAD",     # index of accessibility to radial highways (categorical)
    "TAX",     # property-tax rate
    "PTRATIO", # pupil-teacher ratio by town
    "B",       # proportion of Black residents (1000(Bk - 0.63)^2 formula)
    "LSTAT",   # % lower status of the population
    "MEDV",    # median value of owner-occupied homes in $1000s (target)
]

df = pd.read_csv("..\\data\\house_data_raw.csv", sep=r"\s+", names=COLUMN_NAMES)
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# -----------------------------------------------------------------------
# 2. Handle missing data
# -----------------------------------------------------------------------
missing_before = df.isnull().sum().sum()
print(f"Missing values before cleaning: {missing_before}")

# This dataset happens to be complete, but the pipeline below is written
# to handle missing data generically: numeric columns are filled with
# their median (robust to outliers), so it still works if new data
# arrives with gaps.
for col in df.columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

print(f"Missing values after cleaning: {df.isnull().sum().sum()}")

# -----------------------------------------------------------------------
# 3. Encode categorical variables
# -----------------------------------------------------------------------
# CHAS is already a binary 0/1 flag, so no encoding is needed.
# RAD is a categorical index (9 distinct highway-access zones), so we
# one-hot encode it rather than treat it as an ordinary continuous number.
df["RAD"] = df["RAD"].astype("category")
df = pd.get_dummies(df, columns=["RAD"], prefix="RAD", drop_first=True)

print(f"Shape after one-hot encoding RAD: {df.shape}")

# -----------------------------------------------------------------------
# 4. Split into features/target, then train/test sets
# -----------------------------------------------------------------------
X = df.drop(columns=["MEDV"])
y = df["MEDV"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")

# -----------------------------------------------------------------------
# 5. Normalize / standardize numerical features
# -----------------------------------------------------------------------
# Fit the scaler on the TRAINING data only, then apply it to both train
# and test sets, to avoid leaking test-set statistics into training.
numeric_cols = [
    "CRIM", "ZN", "INDUS", "NOX", "RM", "AGE", "DIS", "TAX", "PTRATIO", "B", "LSTAT"
]

scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

print("\nSample of processed training features:")
print(X_train.head())

# -----------------------------------------------------------------------
# 6. Save the processed splits
# -----------------------------------------------------------------------
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\nSaved: X_train.csv, X_test.csv, y_train.csv, y_test.csv")
