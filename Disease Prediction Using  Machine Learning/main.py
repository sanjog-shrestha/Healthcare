"""
Disease Prediction from Symptoms.

Predicts a disease from a set of binary symptom indicators (1 = present,
0 = absent). Holds out a stratified test set, compares Decision Tree, Random
Forest, SVM and Naive Bayes with cross-validation (oversampling applied inside
each fold), trains the final models on a class-balanced training set,
evaluates each model and a majority-vote ensemble on the untouched test set,
and exposes predict_disease() for new symptom lists.

For learning purposes only: not a medical diagnosis tool.
"""

from collections import Counter
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class PipelineConfig:
    csv_path: str = "improved_disease_dataset.csv"
    target_column: str = "disease"
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    oversample: bool = True               # balance classes in the training data only
    ensemble_members: tuple = ("Random Forest", "Naive Bayes", "SVM")
    show_plots: bool = True


# --------------------------------------------------------------------------- #
# Loading and preparation
# --------------------------------------------------------------------------- #
def load_data(config: PipelineConfig):
    df = pd.read_csv(config.csv_path)
    X = df.drop(columns=[config.target_column])
    # Any text columns (e.g. gender) become numbers; missing symptoms count as absent.
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.fillna(0)

    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(df[config.target_column]), index=df.index)
    print(f"Loaded {len(df)} records: {X.shape[1]} symptoms, {len(encoder.classes_)} diseases")
    return X, y, encoder


def split_data(X, y, config: PipelineConfig):
    """Stratified hold-out set: never oversampled, never seen during training."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state, stratify=y
    )
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    return X_train, X_test, y_train, y_test


def oversample(X, y, config: PipelineConfig):
    if not config.oversample:
        return X, y
    X_res, y_res = RandomOverSampler(random_state=config.random_state).fit_resample(X, y)
    print(f"Oversampled training set: {len(X)} -> {len(X_res)} rows")
    return X_res, y_res


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def get_models(config: PipelineConfig) -> dict:
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=config.random_state),
        "Random Forest": RandomForestClassifier(random_state=config.random_state),
        "SVM": SVC(random_state=config.random_state),
        "Naive Bayes": GaussianNB(),
    }


def cross_validate_models(X_train, y_train, config: PipelineConfig) -> pd.Series:
    """Oversampling sits inside the pipeline, so it only ever sees each fold's training part."""
    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
    scores = {}
    print(f"\n{config.cv_folds}-fold cross-validation on the training set:")
    for name, model in get_models(config).items():
        steps = [("oversample", RandomOverSampler(random_state=config.random_state))] if config.oversample else []
        pipe = ImbPipeline(steps + [("model", model)])
        fold_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        scores[name] = fold_scores.mean()
        print(f"  {name:<14} mean accuracy {fold_scores.mean():.4f}  (folds: {np.round(fold_scores, 3)})")
    return pd.Series(scores).sort_values(ascending=False)


def train_final_models(X_train, y_train, config: PipelineConfig) -> dict:
    models = get_models(config)
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def majority_vote(predictions: list, tie_breaker: int):
    """Most common prediction; if every model disagrees, use the tie-breaker model's answer."""
    winner, count = Counter(predictions).most_common(1)[0]
    return winner if count > 1 else predictions[tie_breaker]


class Ensemble:
    """Majority vote across several fitted models, with an explicit tie-break rule."""

    def __init__(self, models: dict, members: tuple, cv_scores: pd.Series):
        self.members = [m for m in members if m in models]
        self.models = models
        best = max(self.members, key=lambda m: cv_scores.get(m, 0))
        self.tie_breaker = self.members.index(best)
        self.tie_breaker_name = best

    def predict(self, X) -> np.ndarray:
        all_preds = np.column_stack([self.models[m].predict(X) for m in self.members])
        return np.array([majority_vote(list(row), self.tie_breaker) for row in all_preds])


def evaluate_on_test(models: dict, ensemble: Ensemble, X_test, y_test) -> tuple:
    rows, predictions = [], {}
    candidates = {**models, "Ensemble (majority vote)": ensemble}
    for name, model in candidates.items():
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        rows.append({
            "Model": name,
            "Test accuracy (%)": 100 * accuracy_score(y_test, y_pred),
            "Macro F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        })
    results = pd.DataFrame(rows).sort_values("Test accuracy (%)", ascending=False).reset_index(drop=True)
    print(f"\nHeld-out test set (ensemble ties go to {ensemble.tie_breaker_name}):")
    print(results.round(4).to_string(index=False))
    return results, predictions


# --------------------------------------------------------------------------- #
# Prediction for new patients
# --------------------------------------------------------------------------- #
def predict_disease(symptoms, models: dict, ensemble: Ensemble, encoder: LabelEncoder,
                    feature_names) -> dict:
    """symptoms: comma-separated string or list, e.g. "skin_rash, fever, headache"."""
    if isinstance(symptoms, str):
        symptoms = symptoms.split(",")
    wanted = {s.strip().lower().replace(" ", "_") for s in symptoms if s.strip()}
    lookup = {name.lower(): name for name in feature_names}

    unknown = sorted(s for s in wanted if s not in lookup)
    if unknown:
        print(f"Warning: unknown symptoms ignored: {unknown}")

    row = pd.DataFrame([[0] * len(feature_names)], columns=feature_names)
    for s in wanted:
        if s in lookup:
            row.loc[0, lookup[s]] = 1

    result = {f"{name} Prediction": encoder.classes_[models[name].predict(row)[0]]
              for name in ensemble.members}
    result["Final Prediction"] = encoder.classes_[ensemble.predict(row)[0]]
    return result


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_class_distribution(y, encoder: LabelEncoder, title: str) -> None:
    counts = pd.Series(encoder.inverse_transform(y)).value_counts()
    plt.figure(figsize=(18, 8))
    sns.barplot(x=counts.index, y=counts.values)
    plt.title(title)
    plt.ylabel("Records")
    plt.xticks(rotation=90)
    plt.tight_layout()


def plot_confusion_matrix(y_test, y_pred, title: str) -> None:
    plt.figure(figsize=(12, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=False, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted disease (index)")
    plt.ylabel("True disease (index)")
    plt.tight_layout()


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(config: PipelineConfig = PipelineConfig()) -> dict:
    X, y, encoder = load_data(config)
    plot_class_distribution(y, encoder, "Disease class distribution (all data)")

    X_train, X_test, y_train, y_test = split_data(X, y, config)
    cv_scores = cross_validate_models(X_train, y_train, config)

    X_bal, y_bal = oversample(X_train, y_train, config)
    models = train_final_models(X_bal, y_bal, config)
    ensemble = Ensemble(models, config.ensemble_members, cv_scores)

    results, predictions = evaluate_on_test(models, ensemble, X_test, y_test)
    best = results.loc[0, "Model"]
    plot_confusion_matrix(y_test, predictions[best], f"Confusion matrix on the test set: {best}")

    example = "skin_rash,fever,headache"
    print(f"\nExample prediction for '{example}':")
    print(predict_disease(example, models, ensemble, encoder, X.columns))

    if config.show_plots:
        plt.show()
    return {"models": models, "ensemble": ensemble, "encoder": encoder,
            "feature_names": X.columns, "cv_scores": cv_scores, "test_results": results}


if __name__ == "__main__":
    run_pipeline(PipelineConfig())