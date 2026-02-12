#  Titanic Survival Prediction – Machine Learning Classification

## Project Overview

This project builds and compares multiple machine learning classification models to predict passenger survival on the Titanic.

The goal is to classify whether a passenger survived (`1`) or did not survive (`0`) using demographic and ticket-related features.

The best-performing model was selected using cross-validation and then applied to an unseen test dataset to generate final predictions.

---

## Dataset

Two datasets were used:

- **train.csv** → Includes features and the target variable (`Survived`)
- **test.csv** → Includes features only (used for final prediction)

### Target Variable
Survived
0 = Did Not Survive
1 = Survived


### Main Features Used
- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked

---

## Data Preprocessing

The following preprocessing steps were applied:

- Removed irrelevant columns:
  - `PassengerId`
  - `Name`
  - `Ticket`
  - `Cabin`
- Handled missing values:
  - Replaced missing `Age` values with mean
  - Replaced missing `Fare` values with mean
- Converted categorical variables:
  - `Sex`
  - `Embarked`
- Ensured identical preprocessing for both training and test datasets

---

## Models Trained & Compared

The following classification models were evaluated:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Extra Trees Classifier
- LightGBM Classifier
- XGBoost Classifier (Best Model)

### Evaluation Metrics

- Accuracy Score
- 5-Fold Cross-Validation (CV)

---

## Best Model: XGBoost

XGBoost achieved the highest cross-validation performance and was selected as the final model.

Final steps:
- Retrained on the full training dataset
- Generated predictions for the test dataset
- Built final submission DataFrame

---

## Libraries Used

```python
pandas
numpy
scikit-learn
xgboost
lightgbm


