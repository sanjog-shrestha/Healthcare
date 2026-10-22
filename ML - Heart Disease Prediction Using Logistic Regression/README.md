# Heart Disease Risk Prediction (Framingham)

Predicts whether a patient will develop **coronary heart disease (CHD) within ten years** from six risk factors: age, sex, cigarettes per day, total cholesterol, systolic blood pressure and glucose. It uses Logistic Regression on data from the [Framingham Heart Study](https://www.framinghamheartstudy.org/), and compares a standard model with a class-balanced one.

> **Not medical advice.** This is a learning project. Its predictions are not a clinical risk score and must not be used to make health decisions. Anyone concerned about heart health should talk to a qualified doctor.

## What it does

1. Loads the Framingham data (`framingham.csv`, about 4,200 patients) and reports missing values.
2. Renames `male` to `Sex_male` and drops rows missing any of the **features actually used** (or every column, if configured).
3. Prints the share of patients who developed CHD within ten years, about 15% in the real dataset.
4. Plots the outcome counts, and box plots comparing each risk factor for patients with and without CHD.
5. Splits the data into 70% training and 30% test, **stratified** so both keep the same CHD rate.
6. Trains two Logistic Regression models, each inside a pipeline that standardizes the features using training data only:
   - a **standard** model;
   - a **class-balanced** model (`class_weight="balanced"`) that gives the rarer CHD cases more weight.
7. Compares them on accuracy, **ROC AUC** and **CHD recall** against an "always predict no CHD" baseline, with a full classification report for each.
8. Prints **odds ratios** that show how strongly each risk factor is linked to CHD, and plots a confusion matrix for each model.

## Concepts covered

- **Logistic Regression** — predicts the *probability* of an outcome (here, developing CHD) from a weighted sum of the features passed through an S-shaped curve. A patient is classified as at risk when that probability exceeds 0.5.
- **Odds ratios** — exponentiating a coefficient gives an odds ratio. Because the features are standardized, an odds ratio of 2.0 for `age` means "one standard deviation older roughly doubles the odds of CHD, other factors held fixed". This makes logistic regression easy to interpret.
- **Scaling inside a pipeline** — the scaler learns its mean and standard deviation from the training data only. Fitting it on the full dataset before splitting (as the original script did) lets information about the test patients leak into training.
- **Handling missing values deliberately** — dropping rows missing *any* column throws away patients whose only gap is in a column the model never uses (education, BMI, heart rate…). Dropping only rows missing the features in use keeps more data.
- **Class imbalance** — only about one patient in seven develops CHD. A model can reach high accuracy just by predicting "no CHD" for nearly everyone, while missing most of the patients who actually get sick.
- **Why accuracy misleads here** — the script prints the "always predict no CHD" baseline next to the model's accuracy. The metrics that matter are:
  - **CHD recall** — the share of real CHD cases the model catches;
  - **ROC AUC** — how well the model ranks high-risk patients above low-risk ones, where 0.5 is random and 1.0 is perfect.
- **Class weighting** — `class_weight="balanced"` makes each missed CHD case count as much as several false alarms during training. Recall goes up a lot, at the cost of more false positives and lower accuracy. ROC AUC stays about the same, because the ranking barely changes; only the cut-off moves.

## Project structure

```
main.py      # full pipeline, organized into functions (config, loading,
              # missing-value handling, stratified split, scaling pipelines,
              # standard vs. balanced Logistic Regression, odds ratios, plotting)
README.md    # this file
```

The code is organized into small, named functions driven by a `PipelineConfig` dataclass: `load_data`, `drop_missing`, `split_data`, `get_models`, `evaluate_models`, `odds_ratios`, `plot_target_counts`, `plot_features_by_outcome`, `plot_confusion_matrix`. `run_pipeline()` runs them in order.

Settings such as the feature list, how missing values are handled, the test size and the random seed live in `PipelineConfig`, so you can experiment without touching the pipeline code. For example, to add BMI and heart rate as features:

```python
from main import PipelineConfig, run_pipeline

run_pipeline(PipelineConfig(feature_columns=[
    "age", "Sex_male", "cigsPerDay", "totChol", "sysBP", "glucose", "BMI", "heartRate",
]))
```

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Install with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## How to run

Download `framingham.csv` (for example from the [Kaggle Framingham dataset](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset)), place it in the same directory as `main.py`, then:

```bash
python main.py
```

## Sample output

Running the script prints the missing-value counts, the CHD rate, the train/test sizes, a comparison table of the two models with the baseline, a classification report for each and the odds ratios, then displays the charts.

**Risk factors by ten-year CHD outcome**

![Risk factors by outcome](features_by_outcome.png)

**Confusion matrix (balanced model)**

![Confusion matrix](confusion_matrix.png)

**Test-set comparison**

| Model | Accuracy | ROC AUC | CHD recall |
|-------|---------:|--------:|-----------:|
| Logistic Regression | _run to fill in_ | _run to fill in_ | _run to fill in_ |
| Logistic Regression (balanced) | _run to fill in_ | _run to fill in_ | _run to fill in_ |
| Baseline (always "no CHD") | _run to fill in_ | – | 0 |

## A note on the results

The standard model usually reports an accuracy of around 85%, which sounds impressive until you compare it with the baseline: predicting "no CHD" for everyone scores almost exactly the same, because about 85% of patients never develop CHD. The classification report tells the real story. The standard model catches only a small fraction of the patients who go on to develop heart disease.

The balanced model trades some accuracy for a large jump in CHD recall. In a screening setting, missing a patient at risk is usually worse than a false alarm that leads to a follow-up check, so the balanced model is often the more useful one despite its lower accuracy. ROC AUC is nearly identical for both, which shows that class weighting mainly moves the decision threshold rather than making the model rank patients better.

The odds ratios typically show **age** and **systolic blood pressure** as the strongest risk factors, followed by sex and smoking. That matches what is known from the Framingham study itself. Natural next steps would be:

- adding more risk factors (BMI, blood pressure medication, diabetes, prevalent hypertension);
- tuning the decision threshold instead of using 0.5;
- imputing missing values instead of dropping rows;
- comparing against tree-based models.

## References

- [Framingham Heart Study](https://www.framinghamheartstudy.org/)
- [scikit-learn — LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [scikit-learn — Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [GeeksforGeeks — Machine Learning Projects](https://www.geeksforgeeks.org/machine-learning/machine-learning-projects/), used as a general reference/inspiration while working on this project.

---
*This is a personal learning project. It is not a medical device and must not be used for health decisions.*
