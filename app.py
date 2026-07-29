"""
Student Depression Risk — Streamlit App
=========================================
Loads the trained sklearn Pipeline (preprocessing baked in) and exposes
a form matching every feature the model was trained on, so there is no
mismatch between what the UI collects and what the model expects.
"""

import pandas as pd
import joblib
import streamlit as st

MODEL_PATH = "student_depression_model.pkl"


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


artifact = load_artifact()
pipeline = artifact["pipeline"]
model_name = artifact["model_name"]

st.set_page_config(page_title="Student Depression Risk Screener", page_icon="🧠")
st.title("🧠 Student Depression Risk Screener")
st.caption(f"Model: {model_name} · trained on the Student Depression dataset")

st.info(
    "This is an educational ML demo, **not a diagnostic tool**. It cannot "
    "replace a conversation with a mental health professional. If you or "
    "someone you know is struggling, please reach out to a counselor, "
    "doctor, or a local crisis line."
)

with st.form("input_form"):
    st.subheader("About you")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=15, max_value=60, value=20)
        city = st.text_input("City", value="Kalyan")
        profession = st.selectbox(
            "Profession",
            ["Student", "Architect", "Teacher", "Digital Marketer",
             "Content Writer", "Chef", "Doctor", "Pharmacist",
             "Civil Engineer", "UX/UI Designer", "Educational Consultant",
             "Manager", "Lawyer", "Entrepreneur"],
        )
        degree = st.selectbox(
            "Degree",
            ["Class 12", "B.Ed", "B.Com", "B.Arch", "BCA", "MSc", "B.Tech",
             "MCA", "M.Tech", "BHM", "BSc", "M.Ed", "B.Pharm", "M.Com",
             "MBBS", "Other"],
        )
    with col2:
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.01)
        sleep_duration = st.selectbox(
            "Sleep Duration",
            ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours", "Others"],
        )
        dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy", "Others"])
        suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["No", "Yes"])
        family_history = st.selectbox("Family History of Mental Illness", ["No", "Yes"])

    st.subheader("Pressures & satisfaction (0 = none, 5 = highest)")
    col3, col4 = st.columns(2)
    with col3:
        academic_pressure = st.slider("Academic Pressure", 0, 5, 3)
        work_pressure = st.slider("Work Pressure", 0, 5, 0)
        study_satisfaction = st.slider("Study Satisfaction", 0, 5, 3)
    with col4:
        job_satisfaction = st.slider("Job Satisfaction", 0, 4, 0)
        financial_stress = st.slider("Financial Stress", 1, 5, 3)
        work_study_hours = st.slider("Work/Study Hours per day", 0, 12, 6)

    submitted = st.form_submit_button("Predict")

if submitted:
    row = pd.DataFrame([{
        "Gender": gender,
        "City": city,
        "Profession": profession,
        "Sleep Duration": sleep_duration,
        "Dietary Habits": dietary_habits,
        "Degree": degree,
        "Have you ever had suicidal thoughts ?": suicidal_thoughts,
        "Family History of Mental Illness": family_history,
        "Age": age,
        "Academic Pressure": academic_pressure,
        "Work Pressure": work_pressure,
        "CGPA": cgpa,
        "Study Satisfaction": study_satisfaction,
        "Job Satisfaction": job_satisfaction,
        "Work/Study Hours": work_study_hours,
        "Financial Stress": financial_stress,
    }])

    proba = pipeline.predict_proba(row)[0, 1]
    prediction = pipeline.predict(row)[0]

    st.divider()
    st.subheader("Result")
    st.progress(min(max(proba, 0.0), 1.0))
    if prediction == 1:
        st.error(f"Higher predicted risk of depression (model confidence: {proba:.0%})")
    else:
        st.success(f"Lower predicted risk of depression (model confidence: {1 - proba:.0%})")

    st.caption(
        "Confidence is the model's estimated probability, not a clinical "
        "measure. Use this as a conversation starter, not a verdict."
    )
