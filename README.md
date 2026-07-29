# Student Depression Risk Screener

An educational ML project that predicts depression risk from the
[Student Depression dataset](Student%20Depression%20Dataset.csv) and
serves predictions through a Streamlit app.

> **Not a diagnostic tool.** This is a portfolio/learning project. It must
> not be used to make real decisions about someone's mental health.

## What changed from the original version

The original notebook trained `RandomForestClassifier` on **all 17 raw
columns** — including `id` and several text columns that were
label-encoded in a throwaway loop (the encoders were never saved). The
Streamlit apps (`app.py` / `student.py`) then tried to predict from just
4 different, differently-named columns. Every prediction crashed:

```
ValueError: The feature names should match those that were passed during fit.
Feature names seen at fit time, yet now missing:
- City
- Degree
- Dietary Habits
...
```

Fixes made:

| Issue | Fix |
|---|---|
| `id` used as a training feature | Dropped — it's a row index, not signal |
| Encoders fit-and-discarded, so the app couldn't reproduce them | Replaced with a single `sklearn.Pipeline` (`ColumnTransformer` + model) saved as one artifact — preprocessing travels with the model |
| App only sent 4 of 17 fields, wrong names | App now collects every field the model actually uses, with the exact training-time names |
| Ordinal fields (sleep, diet) label-encoded alphabetically | Switched to `OneHotEncoder`, which doesn't impose a false numeric order |
| No handling of missing values / unseen categories at inference | `SimpleImputer` in the pipeline + `OneHotEncoder(handle_unknown="ignore")` |
| No cross-validation or model comparison | `train.py` compares Logistic Regression, Decision Tree, and Random Forest with 5-fold CV F1, picks the best, and reports accuracy/F1/ROC-AUC on a held-out test set |
| Dead code (a second unused `model` variable) | Removed |
| Incomplete `requirements.txt` | Filled in and version-pinned |
| No tests | Added `test_model.py` (5 automated checks, see below) |

## Results

Best model: **Logistic Regression** (selected by 5-fold CV F1 on the
training split; Random Forest was close behind).

- Test accuracy: **0.844**
- Test F1: **0.868**
- Test ROC-AUC: **0.918**

Class balance is ~59% depressed / 41% not — mild imbalance, not severe
enough to require resampling, but worth knowing when reading accuracy.

## Files

- `train.py` — cleans the data, builds the pipeline, compares 3 models,
  saves the best one (with metadata) to `student_depression_model.pkl`
- `app.py` — Streamlit UI, loads the saved pipeline, form matches the
  real feature set exactly
- `test_model.py` — automated sanity tests (load, predict, unseen
  categories, missing values, risk-direction check)
- `requirements.txt` — pinned dependencies
- `Student Depression Dataset.csv` — training data

## Running it

```bash
pip install -r requirements.txt

# 1. Train (regenerates student_depression_model.pkl)
python train.py

# 2. Test the artifact
python test_model.py

# 3. Launch the app
streamlit run app.py
```

## Known limitations (worth stating if you publish this)

- The dataset is self-reported survey data from Indian students — the
  model will not generalize to other populations without retraining.
- `Profession` is ~99.9% "Student" in this dataset, so it contributes
  little signal; it's kept in the UI for completeness but don't expect
  it to move predictions much.
- This is a binary risk classifier trained on a public dataset, not a
  validated clinical instrument. Treat outputs as illustrative only.
