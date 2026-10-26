"""
Breast Cancer Diagnosis with Logistic Regression from Scratch.

Classifies breast tumours as malignant or benign from 30 measurements of cell
nuclei (radius, texture, perimeter, area, smoothness... as mean, standard
error and "worst" values), using the Breast Cancer Wisconsin (Diagnostic)
dataset. Logistic Regression is implemented with NumPy and trained by batch
gradient descent; scikit-learn's LogisticRegression is trained alongside it
as a reference to check the implementation.

For learning purposes only: not a diagnostic tool.
"""

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class PipelineConfig:
    csv_path: str = "data.csv"
    target_column: str = "diagnosis"
    label_map: dict = field(default_factory=lambda: {"M": 1, "B": 0})   # 1 = malignant
    drop_columns: list = field(default_factory=lambda: ["id", "Unnamed: 32"])
    test_size: float = 0.15
    random_state: int = 42
    stratify: bool = True
    learning_rate: float = 1.0
    num_iterations: int = 5000
    log_every: int = 500
    show_plots: bool = True


# --------------------------------------------------------------------------- #
# Logistic Regression from scratch
# --------------------------------------------------------------------------- #
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegressionGD:
    """Binary logistic regression trained with batch gradient descent.

    Rows are samples and columns are features (the usual scikit-learn layout).
    """

    def __init__(self, learning_rate: float = 1.0, num_iterations: int = 5000,
                 log_every: int = 500, random_state: int = 42):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.log_every = log_every
        self.random_state = random_state

    def _cost(self, y: np.ndarray, p: np.ndarray) -> float:
        p = np.clip(p, 1e-12, 1 - 1e-12)          # avoid log(0)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.random_state)
        self.w_ = rng.normal(0, 0.01, X.shape[1])
        self.b_ = 0.0
        self.costs_ = []
        m = X.shape[0]

        for i in range(self.num_iterations):
            p = sigmoid(X @ self.w_ + self.b_)            # forward pass
            error = p - y
            grad_w = X.T @ error / m                       # backward pass
            grad_b = error.mean()
            self.w_ -= self.learning_rate * grad_w
            self.b_ -= self.learning_rate * grad_b

            cost = self._cost(y, p)
            self.costs_.append(cost)
            if self.log_every and i % self.log_every == 0:
                print(f"  Cost after iteration {i:>5}: {cost:.4f}")
        print(f"  Final cost after {self.num_iterations} iterations: {self.costs_[-1]:.4f}")
        return self

    def predict_proba(self, X) -> np.ndarray:
        return sigmoid(np.asarray(X, dtype=float) @ self.w_ + self.b_)

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


# --------------------------------------------------------------------------- #
# Loading and preparation
# --------------------------------------------------------------------------- #
def load_data(config: PipelineConfig):
    df = pd.read_csv(config.csv_path)
    print(f"Loaded {len(df)} tumours, {df.shape[1]} columns from {config.csv_path}")
    print(df.head())

    df = df.drop(columns=[c for c in config.drop_columns if c in df.columns])
    y = df[config.target_column].map(config.label_map).astype(int)
    X = df.drop(columns=[config.target_column])
    print(f"Features: {X.shape[1]} | Malignant: {y.mean():.1%} | Benign: {1 - y.mean():.1%}")
    return X, y


def split_data(X, y, config: PipelineConfig):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state,
        stratify=y if config.stratify else None,
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def min_max_scale(X_train, X_test):
    """Scale to 0-1 using the training set's min and max only."""
    lo, hi = X_train.min(), X_train.max()
    span = (hi - lo).replace(0, 1)
    return (X_train - lo) / span, (X_test - lo) / span


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(models: dict, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        rows.append({
            "Model": name,
            "Train accuracy (%)": 100 * accuracy_score(y_train, model.predict(X_train)),
            "Test accuracy (%)": 100 * accuracy_score(y_test, model.predict(X_test)),
            "Malignant recall": recall_score(y_test, model.predict(X_test)),
            "Test ROC AUC": roc_auc_score(y_test, _proba(model, X_test)),
        })
    results = pd.DataFrame(rows)
    print("\nModel comparison:")
    print(results.round(4).to_string(index=False))
    print(f"Baseline accuracy (always predict benign): {100 * (1 - y_test.mean()):.2f}%")
    return results


def _proba(model, X) -> np.ndarray:
    p = model.predict_proba(X)
    return p[:, 1] if p.ndim == 2 else p


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_cost_curve(costs: list) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(costs)
    plt.title("Training cost (binary cross-entropy) over gradient-descent iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.tight_layout()


def plot_confusion_matrix(y_test, y_pred) -> None:
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Purples",
                xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
    plt.title("Confusion Matrix (from-scratch model)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(config: PipelineConfig = PipelineConfig()) -> pd.DataFrame:
    X, y = load_data(config)
    X_train, X_test, y_train, y_test = split_data(X, y, config)
    X_train, X_test = min_max_scale(X_train, X_test)

    print(f"\nTraining from-scratch Logistic Regression "
          f"(learning rate {config.learning_rate}, {config.num_iterations} iterations):")
    scratch = LogisticRegressionGD(config.learning_rate, config.num_iterations,
                                   config.log_every, config.random_state).fit(X_train, y_train)
    reference = LogisticRegression(max_iter=5000).fit(X_train, y_train)

    results = evaluate({"From scratch (NumPy)": scratch, "scikit-learn": reference},
                       X_train, X_test, y_train, y_test)

    y_pred = scratch.predict(X_test)
    print("\nClassification report (from-scratch model):")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

    plot_cost_curve(scratch.costs_)
    plot_confusion_matrix(y_test, y_pred)

    if config.show_plots:
        plt.show()
    return results


if __name__ == "__main__":
    run_pipeline(PipelineConfig())