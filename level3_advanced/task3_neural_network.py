"""
Codveda ML Internship - Level 3, Task 3
Neural Networks with TensorFlow/Keras
Dataset: scikit-learn's built-in "digits" dataset (1797 handwritten digits,
8x8 grayscale images, 10 classes: 0-9)

Note: the task suggests MNIST (28x28 images), but that dataset requires
downloading from a host outside this environment's network allowlist.
The scikit-learn digits dataset is the same task in spirit - classify
handwritten digits 0-9 - just smaller and bundled locally, so every
objective below (architecture, backprop, accuracy, loss curves) is
demonstrated identically.

Objectives covered:
  1. Load and preprocess the dataset
  2. Design a neural network architecture (input, hidden, output layers)
  3. Train the model using backpropagation
  4. Evaluate using accuracy and visualize training/validation loss
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)
np.random.seed(42)

# -----------------------------------------------------------------------
# 1. Load and preprocess
# -----------------------------------------------------------------------
digits = load_digits()
X, y = digits.data, digits.target   # X: (1797, 64) flattened 8x8 images, y: digit 0-9
print(f"Dataset: {X.shape[0]} images, {X.shape[1]} pixels each, {len(set(y))} classes")

# Pixel values range 0-16 in this dataset; scale to 0-1 so inputs are
# small and consistent, which helps the network train faster and more
# stably (large raw pixel values can cause unstable gradients).
X = X / 16.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Carve a validation set out of training data, to monitor for
# overfitting DURING training (separate from the final test set).
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)
print(f"Train: {X_train.shape[0]} | Validation: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# -----------------------------------------------------------------------
# 2. Design the network architecture
# -----------------------------------------------------------------------
# Input layer: 64 neurons (one per pixel)
# Hidden layer 1: 64 neurons, ReLU activation (learns feature combinations)
# Hidden layer 2: 32 neurons, ReLU (refines those into higher-level patterns)
# Output layer: 10 neurons, softmax (one probability per digit, summing to 1)
model = keras.Sequential([
    layers.Input(shape=(64,)),
    layers.Dense(64, activation="relu", name="hidden_1"),
    layers.Dense(32, activation="relu", name="hidden_2"),
    layers.Dense(10, activation="softmax", name="output"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",  # y is plain integers (0-9), not one-hot
    metrics=["accuracy"],
)

model.summary()

# -----------------------------------------------------------------------
# 3. Train using backpropagation
# -----------------------------------------------------------------------
# model.fit() runs backpropagation under the hood: for each batch, it
# computes predictions, measures the loss, then propagates the error
# backward through the layers to update every weight via gradient descent
# (here, the "adam" variant of it).
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=16,
    verbose=0,
)
print("Training complete.")

# -----------------------------------------------------------------------
# 4. Evaluate and visualize training/validation loss
# -----------------------------------------------------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal test accuracy: {test_acc:.4f}")
print(f"Final test loss:     {test_loss:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(history.history["loss"], label="Training loss")
axes[0].plot(history.history["val_loss"], label="Validation loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training vs Validation Loss")
axes[0].legend()

axes[1].plot(history.history["accuracy"], label="Training accuracy")
axes[1].plot(history.history["val_accuracy"], label="Validation accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Training vs Validation Accuracy")
axes[1].legend()

plt.tight_layout()
plt.savefig("../results/level3_advanced/task3_nn_training_curves.png", dpi=150)
print("Saved training curves to task3_nn_training_curves.png")

# Save a few sample predictions for a sanity check
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)
import pandas as pd
pd.DataFrame({"actual_digit": y_test, "predicted_digit": y_pred}).to_csv(
    "../results/level3_advanced/task3_nn_predictions.csv", index=False
)
print("Saved predictions to task3_nn_predictions.csv")
