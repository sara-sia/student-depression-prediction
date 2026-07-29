"""
Student Depression Risk — Training Script
==========================================
Trains a scikit-learn Pipeline (preprocessing + classifier) on the
Student Depression dataset and saves it as a single artifact so that
inference code never has to re-implement encoding logic by hand.

Run:
    python train.py
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)

RANDOM_STATE = 42
DATA_PATH = "Student Depression Dataset.csv"
MODEL_PATH = "student_depression_model.pkl"

TARGET = "Depression"

# Columns that are not real predictive signal (row identifier).
DROP_COLS = ["id"]

CATEGORICAL_COLS = [
    "Gender", "City", "Profession", "Sleep Duration", "Dietary Habits",
    "Degree", "Have you ever had suicidal thoughts ?",
    "Family History of Mental Illness",
]
NUMERIC_COLS = [
    "Age", "Academic Pressure", "Work Pressure", "CGPA",
    "Study Satisfaction", "Job Satisfaction", "Work/Study Hours",
    "Financial Stress",
]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.drop_duplicates()
    # Financial Stress has a few missing values in the raw data.
    df["Financial Stress"] = pd.to_numeric(df["Financial Stress"], errors="coerce")
    df["Financial Stress"] = df["Financial Stress"].fillna(df["Financial Stress"].median())
    return df


def build_preprocessor() -> ColumnTransformer:
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("cat", categorical_pipe, CATEGORICAL_COLS),
        ("num", numeric_pipe, NUMERIC_COLS),
    ])


def main():
    df = load_data(DATA_PATH)
    print(f"Rows after cleaning: {len(df)}")
    print("Class balance:\n", df[TARGET].value_counts(normalize=True).round(3))

    X = df[CATEGORICAL_COLS + NUMERIC_COLS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=8),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    results = {}
    for name, clf in candidates.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1")
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]

        results[name] = {
            "pipeline": pipe,
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(),
            "test_accuracy": accuracy_score(y_test, preds),
            "test_f1": f1_score(y_test, preds),
            "test_roc_auc": roc_auc_score(y_test, proba),
        }
        print(f"\n=== {name} ===")
        print(f"5-fold CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"Test accuracy: {results[name]['test_accuracy']:.4f}")
        print(f"Test F1: {results[name]['test_f1']:.4f}")
        print(f"Test ROC-AUC: {results[name]['test_roc_auc']:.4f}")

    best_name = max(results, key=lambda n: results[n]["cv_f1_mean"])
    best_pipe = results[best_name]["pipeline"]
    print(f"\nBest model by CV F1: {best_name}")

    final_preds = best_pipe.predict(X_test)
    print("\nConfusion matrix (test set):")
    print(confusion_matrix(y_test, final_preds))
    print("\nClassification report (test set):")
    print(classification_report(y_test, final_preds))

    joblib.dump({
        "pipeline": best_pipe,
        "model_name": best_name,
        "categorical_cols": CATEGORICAL_COLS,
        "numeric_cols": NUMERIC_COLS,
        "metrics": {k: v for k, v in results[best_name].items() if k != "pipeline"},
    }, MODEL_PATH)
    print(f"\nSaved pipeline + metadata to {MODEL_PATH}")


if __name__ == "__main__":
    main()
