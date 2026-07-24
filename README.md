# Titanic Survival Prediction — Classification Model Comparison

A machine learning project comparing five classification algorithms to predict passenger survival on the Titanic, using the classic Titanic dataset.

## Overview

This project walks through a full classification workflow: data cleaning, encoding, feature scaling, and training/evaluating multiple models to compare their performance on the same task.

## Dataset

The Titanic dataset from `seaborn`'s built-in datasets (891 passengers, 15 original features). Target variable: `survived` (0 = did not survive, 1 = survived).

## Workflow

1. **Data Cleaning**
   - Dropped redundant/leaky columns: `deck`, `embark_town`, `alive`, `who`, `adult_male`, `class`
   - Filled missing `age` values with the column mean
   - Dropped rows with missing `embarked`

2. **Encoding**
   - Label-encoded categorical columns: `sex`, `embarked`

3. **Train/Test Split**
   - 80/20 split, `random_state=42`

4. **Feature Scaling**
   - `StandardScaler` applied for distance-based models (KNN, Decision Tree, SVM)

5. **Models Trained**

   | Model                 | Accuracy |
   |------------------------|:--------:|
   | Support Vector Machine (RBF) | **0.826** |
   | Logistic Regression    | 0.803    |
   | K-Nearest Neighbors    | 0.775    |
   | Naive Bayes            | 0.775    |
   | Decision Tree          | 0.770    |

   Each model is evaluated with accuracy score, confusion matrix, and a full classification report (precision/recall/F1).

## Results

The SVM (RBF kernel) performed best on this split, followed closely by Logistic Regression. This is a reasonable outcome given the dataset's moderate size and the non-linear decision boundary an RBF kernel can capture.

## Tech Stack

- Python 3
- pandas, NumPy
- seaborn, matplotlib
- scikit-learn

## Project Structure

```
├── titanic_classification.ipynb   # Main notebook: EDA, preprocessing, model training & evaluation
├── requirements.txt               # Python dependencies
└── README.md
```

## How to Run

```bash
git clone https://github.com/<your-username>/titanic-classification.git
cd titanic-classification
pip install -r requirements.txt
jupyter notebook titanic_classification.ipynb
```

## Possible Next Steps

- Hyperparameter tuning (`GridSearchCV`) for each model
- Cross-validation instead of a single train/test split
- Feature engineering (e.g. extracting titles from names, family size from `sibsp`/`parch`)
- Wrap the best model in a simple API (FastAPI/Flask) for deployment

## Author

Parth Vaghasiya
