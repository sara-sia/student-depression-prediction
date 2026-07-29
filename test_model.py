"""
Sanity tests for the saved model pipeline.
Run: python test_model.py
"""

import joblib
import pandas as pd

MODEL_PATH = "student_depression_model.pkl"

BASE_ROW = {
    "Gender": "Male",
    "City": "Kalyan",
    "Profession": "Student",
    "Sleep Duration": "5-6 hours",
    "Dietary Habits": "Moderate",
    "Degree": "B.Tech",
    "Have you ever had suicidal thoughts ?": "No",
    "Family History of Mental Illness": "No",
    "Age": 21,
    "Academic Pressure": 3,
    "Work Pressure": 0,
    "CGPA": 7.0,
    "Study Satisfaction": 3,
    "Job Satisfaction": 0,
    "Work/Study Hours": 6,
    "Financial Stress": 3,
}


def test_artifact_loads():
    artifact = joblib.load(MODEL_PATH)
    assert "pipeline" in artifact
    print("PASS: artifact loads and contains a pipeline")


def test_predicts_without_crashing():
    artifact = joblib.load(MODEL_PATH)
    pipe = artifact["pipeline"]
    row = pd.DataFrame([BASE_ROW])
    pred = pipe.predict(row)
    proba = pipe.predict_proba(row)
    assert pred.shape == (1,)
    assert proba.shape == (1, 2)
    print("PASS: predict/predict_proba run without error")


def test_handles_unseen_categories():
    artifact = joblib.load(MODEL_PATH)
    pipe = artifact["pipeline"]
    row = dict(BASE_ROW)
    row["City"] = "A City Not In Training Data"
    row["Degree"] = "A Degree Not In Training Data"
    pipe.predict(pd.DataFrame([row]))
    print("PASS: unseen categorical values do not crash prediction")


def test_high_risk_vs_low_risk_direction():
    artifact = joblib.load(MODEL_PATH)
    pipe = artifact["pipeline"]

    high = dict(BASE_ROW)
    high.update({
        "Academic Pressure": 5, "Financial Stress": 5, "Work/Study Hours": 12,
        "Study Satisfaction": 0, "Have you ever had suicidal thoughts ?": "Yes",
    })
    low = dict(BASE_ROW)
    low.update({
        "Academic Pressure": 0, "Financial Stress": 1, "Work/Study Hours": 2,
        "Study Satisfaction": 5, "Have you ever had suicidal thoughts ?": "No",
    })

    p_high = pipe.predict_proba(pd.DataFrame([high]))[0, 1]
    p_low = pipe.predict_proba(pd.DataFrame([low]))[0, 1]

    assert p_high > p_low, f"Expected higher-risk profile ({p_high:.3f}) > lower-risk profile ({p_low:.3f})"
    print(f"PASS: risk direction is sane (high-risk={p_high:.3f} > low-risk={p_low:.3f})")


def test_missing_optional_numeric_is_handled():
    # Simulate a row with a NaN in a numeric column, mimicking real-world
    # missing input the way "Financial Stress" had 3 nulls in the raw data.
    artifact = joblib.load(MODEL_PATH)
    pipe = artifact["pipeline"]
    row = dict(BASE_ROW)
    row["Financial Stress"] = None
    pipe.predict(pd.DataFrame([row]))
    print("PASS: missing numeric value is imputed instead of crashing")


if __name__ == "__main__":
    test_artifact_loads()
    test_predicts_without_crashing()
    test_handles_unseen_categories()
    test_high_risk_vs_low_risk_direction()
    test_missing_optional_numeric_is_handled()
    print("\nAll tests passed.")
