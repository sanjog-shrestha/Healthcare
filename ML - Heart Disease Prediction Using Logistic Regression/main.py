"""
Heart Disease Risk Prediction (Framingham Heart Study).

Predicts whether a patient will develop coronary heart disease (CHD) within
ten years from six risk factors: age, sex, cigarettes per day, total
cholesterol, systolic blood pressure and glucose. Uses Logistic Regression
inside a scaling pipeline fitted on training data only, and compares a
standard model with a class-balanced one, because only about one patient in
seven develops CHD.

For learning purposes only: not a medical risk calculator.
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class PipelineConfig:
    csv_path: str = "framingham.csv"
    target_column: str = "TenYearCHD"
    rename_columns: dict = field(default_factory=lambda: {"male": "Sex_male"})
    feature_columns: list = field(default_factory=lambda: [
        "age", "Sex_male", "cigsPerDay", "totChol", "sysBP", "glucose",
    ])
    # True = drop rows missing ANY column (original behaviour); False = only rows
    # missing one of the features actually used, which keeps more patients.
    drop_na_all_columns: bool = False
    test_size: float = 0.3
    random_state: int = 4
    stratify: bool = True
    show_plots: bool = True


# --------------------------------------------------------------------------- #
# Loading and preparation
# --------------------------------------------------------------------------- #
def load_data(config: PipelineConfig) -> pd.DataFrame:
    df = pd.read_csv(config.csv_path).rename(columns=config.rename_columns)
    print(f"Loaded {len(df)} patients, {df.shape[1]} columns from {config.csv_path}")
    print("\nMissing values per column:")
    print(df.isnull().sum()[lambda s: s > 0].to_string())
    return df


def drop_missing(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    subset = None if config.drop_na_all_columns else config.feature_columns + [config.target_column]
    cleaned = df.dropna(subset=subset)
    print(f"\nDropped {len(df) - len(cleaned)} rows with missing values -> {len(cleaned)} patients")
    rate = cleaned[config.target_column].mean()
    print(f"Patients who developed CHD within 10 years: {rate:.1%}")
    return cleaned


def split_data(df: pd.DataFrame, config: PipelineConfig):
    X = df[config.feature_columns]
    y = df[config.target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state,
        stratify=y if config.stratify else None,
    )
    print(f"Train set: {X_train.shape} | Test set: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------- #
# Modelling and evaluation
# --------------------------------------------------------------------------- #
def get_models() -> dict:
    """Scaling lives inside each pipeline, so it is learned from training data only."""
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression()),
        ]),
        "Logistic Regression (balanced)": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced")),
        ]),
    }


def evaluate_models(models: dict, X_train, X_test, y_train, y_test) -> tuple:
    rows, predictions = [], {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        predictions[name] = y_pred
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "ROC AUC": roc_auc_score(y_test, y_prob),
            "CHD recall": recall_score(y_test, y_pred),
        })
    results = pd.DataFrame(rows)
    print("\nTest-set comparison:")
    print(results.round(4).to_string(index=False))
    print(f"Baseline accuracy (always predict 'no CHD'): {1 - y_test.mean():.4f}")

    for name, y_pred in predictions.items():
        print(f"\nClassification report: {name}")
        print(classification_report(y_test, y_pred, target_names=["No CHD", "CHD"], zero_division=0))
    return results, predictions


def odds_ratios(pipe: Pipeline, feature_names) -> pd.Series:
    """exp(coefficient): how the odds of CHD change per one standard deviation increase."""
    coefs = pipe.named_steps["model"].coef_[0]
    ratios = pd.Series(np.exp(coefs), index=feature_names).sort_values(ascending=False)
    print("\nOdds ratio per one-standard-deviation increase (standard model):")
    print(ratios.round(3).to_string())
    return ratios


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_target_counts(df: pd.DataFrame, config: PipelineConfig) -> None:
    plt.figure(figsize=(7, 5))
    sns.countplot(x=config.target_column, data=df, hue=config.target_column,
                  palette="BuGn_r", legend=False)
    plt.title("Ten-year CHD outcome (0 = no, 1 = yes)")
    plt.tight_layout()


def plot_features_by_outcome(df: pd.DataFrame, config: PipelineConfig) -> None:
    """Box plots of each numeric risk factor for patients with and without CHD."""
    numeric = [c for c in config.feature_columns if df[c].nunique() > 2]
    fig, axes = plt.subplots(1, len(numeric), figsize=(4 * len(numeric), 4))
    for ax, col in zip(axes, numeric):
        sns.boxplot(x=config.target_column, y=col, data=df, ax=ax)
        ax.set_title(col)
    plt.tight_layout()


def plot_confusion_matrix(y_test, y_pred, title: str) -> None:
    cm = confusion_matrix(y_test, y_pred)
    conf_matrix = pd.DataFrame(cm, columns=["Predicted: 0", "Predicted: 1"],
                               index=["Actual: 0", "Actual: 1"])
    plt.figure(figsize=(8, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens")
    plt.title(title)
    plt.tight_layout()


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(config: PipelineConfig = PipelineConfig()) -> pd.DataFrame:
    df = load_data(config)
    df = drop_missing(df, config)

    plot_target_counts(df, config)
    plot_features_by_outcome(df, config)

    X_train, X_test, y_train, y_test = split_data(df, config)
    models = get_models()
    results, predictions = evaluate_models(models, X_train, X_test, y_train, y_test)
    odds_ratios(models["Logistic Regression"], config.feature_columns)

    for name, y_pred in predictions.items():
        plot_confusion_matrix(y_test, y_pred, f"Confusion matrix: {name}")

    if config.show_plots:
        plt.show()
    return results


if __name__ == "__main__":
    run_pipeline(PipelineConfig())