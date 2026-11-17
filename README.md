# Spaceship Titanic: binary classification

Predicting which passengers survived the Spaceship Titanic anomaly, the introductory Kaggle competition. Nearly 13,000 passengers, a skewed target, a lot of missing values in the spending columns, and a leaderboard that punishes you for not handling them carefully.

The point of this repository is not the leaderboard position. It is the model comparison: nine classifiers, same features, same split, so the differences you see are differences between models rather than between experiments.

## Dataset

Kaggle's Spaceship Titanic. Each row is a passenger with home planet, cabin, age, and spending on five onboard amenities, and the target is whether they were transported. Missing spending values are informative rather than random, which is the central feature-engineering decision in this competition.

## What is in here

| Path | What it does |
|---|---|
| `Spaceship_titanic.ipynb` | Loading, feature engineering, training and comparison of the models |
| `models/*.pkl` | The trained models, one pickle each |
| `Kaggle_submission.csv` | Predictions in submission format |
| `sample_submission.csv` | The competition's expected format |

The trained models are AdaBoost, decision tree, gradient boosting, k-nearest neighbours, LightGBM, a multi-layer perceptron, random forest, support vector classifier and XGBoost.

## Requirements

```bash
pip install numpy pandas scikit-learn lightgbm xgboost
```

The notebook also needs Jupyter.

## Notes

- The competition ships separate train and test files from Kaggle; `sample_submission.csv` is included here so the output format is clear without downloading anything.
- On tabular data this small, gradient-boosted trees usually beat a neural network without any tuning. Running the comparison once and watching that happen is more convincing than reading it.
- To run it against the real leaderboard you will need your own Kaggle API credentials, which belong outside the repository.
