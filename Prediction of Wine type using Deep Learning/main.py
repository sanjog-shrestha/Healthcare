"""
Wine Type Classification with a Neural Network.

Predicts whether a wine is red or white from 11 physicochemical measurements
(acidity, sugar, sulphur dioxide, density, pH, alcohol...) plus its quality
score, using the UCI Wine Quality dataset. Standardizes the features, trains a
small Keras feed-forward network with early stopping, evaluates it on a
held-out test set, and compares it with a Logistic Regression baseline.
"""

from dataclasses import dataclass, field

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras import layers
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality"


@dataclass
class PipelineConfig:
    red_path: str = f"{UCI_BASE}/winequality-red.csv"      # URL or local file
    white_path: str = f"{UCI_BASE}/winequality-white.csv"
    separator: str = ";"
    target_column: str = "type"                             # 1 = red, 0 = white
    test_size: float = 0.34
    random_state: int = 45
    hidden_units: list = field(default_factory=lambda: [12, 9])
    epochs: int = 50
    batch_size: int = 32
    validation_split: float = 0.1                           # taken from the training set
    early_stopping_patience: int = 5
    show_plots: bool = True


# --------------------------------------------------------------------------- #
# Loading and preparation
# --------------------------------------------------------------------------- #
def load_data(config: PipelineConfig) -> pd.DataFrame:
    red = pd.read_csv(config.red_path, sep=config.separator).assign(**{config.target_column: 1})
    white = pd.read_csv(config.white_path, sep=config.separator).assign(**{config.target_column: 0})
    wines = pd.concat([red, white], ignore_index=True).dropna()
    print(f"Loaded {len(red)} red and {len(white)} white wines ({len(wines)} total, "
          f"{wines.shape[1] - 1} features)")
    return wines


def split_data(wines: pd.DataFrame, config: PipelineConfig):
    X = wines.drop(columns=[config.target_column])
    y = wines[config.target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state, stratify=y
    )
    print(f"Train: {len(X_train)} wines | Test: {len(X_test)} wines")
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """Standardize features using statistics from the training set only."""
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def build_network(n_features: int, config: PipelineConfig) -> keras.Model:
    model = keras.Sequential([keras.Input(shape=(n_features,))])
    for units in config.hidden_units:
        model.add(layers.Dense(units, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def train_network(model: keras.Model, X_train, y_train, config: PipelineConfig):
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=config.early_stopping_patience, restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train.to_numpy(),
        epochs=config.epochs, batch_size=config.batch_size,
        validation_split=config.validation_split, callbacks=[early_stop], verbose=0,
    )
    print(f"\nNeural network trained for {len(history.history['loss'])} epochs "
          f"(early stopping, patience {config.early_stopping_patience})")
    return history


def train_baseline(X_train, y_train, config: PipelineConfig) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, random_state=config.random_state)
    model.fit(X_train, y_train)
    return model


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(network: keras.Model, baseline: LogisticRegression, X_test, y_test) -> tuple:
    nn_pred = (network.predict(X_test, verbose=0).ravel() >= 0.5).astype(int)
    lr_pred = baseline.predict(X_test)

    results = pd.DataFrame([
        {"Model": "Neural network", "Test accuracy": accuracy_score(y_test, nn_pred)},
        {"Model": "Logistic Regression", "Test accuracy": accuracy_score(y_test, lr_pred)},
        {"Model": "Always predict 'white'", "Test accuracy": 1 - y_test.mean()},
    ])
    print("\nTest-set accuracy:")
    print(results.round(4).to_string(index=False))

    print("\nClassification report (neural network):")
    print(classification_report(y_test, nn_pred, target_names=["White", "Red"]))
    return nn_pred, results


def show_sample_predictions(nn_pred, y_test, n: int = 12) -> None:
    print(f"First {n} test predictions:")
    for pred, actual in zip(nn_pred[:n], y_test.to_numpy()[:n]):
        label = lambda v: "Red wine" if v == 1 else "White wine"
        mark = "correct" if pred == actual else "WRONG"
        print(f"  Predicted: {label(pred):<10}  Actual: {label(actual):<10}  ({mark})")


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_alcohol_by_type(wines: pd.DataFrame, config: PipelineConfig) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    ax[0].hist(wines.loc[wines[config.target_column] == 1, "alcohol"], bins=10,
               facecolor="red", alpha=0.5)
    ax[1].hist(wines.loc[wines[config.target_column] == 0, "alcohol"], bins=10,
               facecolor="white", edgecolor="black", lw=0.5, alpha=0.5)
    ax[0].set_title("Alcohol Content in Red Wine")
    ax[1].set_title("Alcohol Content in White Wine")
    for a in ax:
        a.set_xlabel("Alcohol in % Vol")
    ax[0].set_ylabel("Frequency")
    fig.suptitle("Distribution of Alcohol by Wine Type")
    plt.tight_layout()


def plot_training_history(history) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ["loss", "accuracy"]):
        ax.plot(history.history[metric], label="train")
        ax.plot(history.history[f"val_{metric}"], label="validation")
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.tight_layout()


def plot_confusion_matrix(y_test, y_pred) -> None:
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Reds",
                xticklabels=["White", "Red"], yticklabels=["White", "Red"])
    plt.title("Confusion Matrix (neural network)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(config: PipelineConfig = PipelineConfig()) -> pd.DataFrame:
    keras.utils.set_random_seed(config.random_state)

    wines = load_data(config)
    plot_alcohol_by_type(wines, config)

    X_train, X_test, y_train, y_test = split_data(wines, config)
    X_train_s, X_test_s = scale_features(X_train, X_test)

    network = build_network(X_train_s.shape[1], config)
    history = train_network(network, X_train_s, y_train, config)
    baseline = train_baseline(X_train_s, y_train, config)

    nn_pred, results = evaluate(network, baseline, X_test_s, y_test)
    show_sample_predictions(nn_pred, y_test)

    plot_training_history(history)
    plot_confusion_matrix(y_test, nn_pred)

    if config.show_plots:
        plt.show()
    return results


if __name__ == "__main__":
    run_pipeline(PipelineConfig())