import os
import pandas as pd
import joblib
import streamlit as st


# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Student Depression Prediction",
    page_icon="🧠",
    layout="wide"
)


# -------------------------------------------------
# Model Path
# -------------------------------------------------
MODEL_PATH = "student_depression_model.pkl"


# -------------------------------------------------
# Load Model
# -------------------------------------------------
@st.cache_resource
def load_artifact():
    if not os.path.exists(MODEL_PATH):
        return None

    return joblib.load(MODEL_PATH)


artifact = load_artifact()


# -------------------------------------------------
# Check Model
# -------------------------------------------------
if artifact is None:
    st.error(
        f"Model file '{MODEL_PATH}' was not found. "
        "Please place the .pkl file in the same folder as app.py."
    )
    st.stop()


# The training code should save:
# {
#     "pipeline": trained_pipeline,
#     "model_name": model_name
# }

if isinstance(artifact, dict) and "pipeline" in artifact:
    pipeline = artifact["pipeline"]
    model_name = artifact.get("model_name", "Trained Model")
else:
    # If the .pkl contains only the trained pipeline
    pipeline = artifact
    model_name = "Trained Model"


# -------------------------------------------------
# Title
# -------------------------------------------------
st.title("🧠 Student Depression Prediction Analysis")

st.write(
    "Enter the student's information below to estimate the "
    "predicted depression risk using the trained machine learning model."
)

st.info(
    "⚠️ This application is an educational machine learning project "
    "and is not a medical or clinical diagnostic tool."
)

st.caption(f"Model used: {model_name}")


# -------------------------------------------------
# Input Form
# -------------------------------------------------
with st.form("prediction_form"):

    st.header("👤 Student Information")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        age = st.number_input(
            "Age",
            min_value=15,
            max_value=60,
            value=20,
            step=1
        )

        city = st.text_input(
            "City",
            value="Kalyan"
        )

        profession = st.selectbox(
            "Profession",
            [
                "Student",
                "Architect",
                "Teacher",
                "Digital Marketer",
                "Content Writer",
                "Chef",
                "Doctor",
                "Pharmacist",
                "Civil Engineer",
                "UX/UI Designer",
                "Educational Consultant",
                "Manager",
                "Lawyer",
                "Entrepreneur"
            ]
        )

        degree = st.selectbox(
            "Degree",
            [
                "Class 12",
                "B.Ed",
                "B.Com",
                "B.Arch",
                "BCA",
                "MSc",
                "B.Tech",
                "MCA",
                "M.Tech",
                "BHM",
                "BSc",
                "M.Ed",
                "B.Pharm",
                "M.Com",
                "MBBS",
                "Other"
            ]
        )

    with col2:

        cgpa = st.number_input(
            "CGPA",
            min_value=0.0,
            max_value=10.0,
            value=7.0,
            step=0.01
        )

        sleep_duration = st.selectbox(
            "Sleep Duration",
            [
                "Less than 5 hours",
                "5-6 hours",
                "7-8 hours",
                "More than 8 hours",
                "Others"
            ]
        )

        dietary_habits = st.selectbox(
            "Dietary Habits",
            [
                "Healthy",
                "Moderate",
                "Unhealthy",
                "Others"
            ]
        )

        suicidal_thoughts = st.selectbox(
            "Have you ever had suicidal thoughts?",
            ["No", "Yes"]
        )

        family_history = st.selectbox(
            "Family History of Mental Illness",
            ["No", "Yes"]
        )


    # -------------------------------------------------
    # Academic / Work Information
    # -------------------------------------------------

    st.header("📊 Academic, Work & Financial Factors")

    col3, col4 = st.columns(2)

    with col3:

        academic_pressure = st.slider(
            "Academic Pressure",
            min_value=0,
            max_value=5,
            value=3
        )

        work_pressure = st.slider(
            "Work Pressure",
            min_value=0,
            max_value=5,
            value=0
        )

        study_satisfaction = st.slider(
            "Study Satisfaction",
            min_value=0,
            max_value=5,
            value=3
        )

    with col4:

        job_satisfaction = st.slider(
            "Job Satisfaction",
            min_value=0,
            max_value=4,
            value=0
        )

        financial_stress = st.slider(
            "Financial Stress",
            min_value=1,
            max_value=5,
            value=3
        )

        work_study_hours = st.slider(
            "Work/Study Hours per day",
            min_value=0,
            max_value=12,
            value=6
        )


    # -------------------------------------------------
    # Predict Button
    # -------------------------------------------------

    submitted = st.form_submit_button(
        "🔍 Predict Depression Risk",
        use_container_width=True
    )


# -------------------------------------------------
# Prediction
# -------------------------------------------------

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
        "Financial Stress": financial_stress
    }])

    try:

        prediction = pipeline.predict(row)[0]

        # Get probability if the model supports it
        if hasattr(pipeline, "predict_proba"):

            probabilities = pipeline.predict_proba(row)[0]

            # Probability of class 1
            if len(probabilities) > 1:
                proba = probabilities[1]
            else:
                proba = probabilities[0]

        else:
            proba = None


        # -------------------------------------------------
        # Result
        # -------------------------------------------------

        st.divider()

        st.header("📋 Prediction Result")

        if prediction == 1:

            st.warning(
                "The model predicts a higher likelihood of depression."
            )

        else:

            st.success(
                "The model predicts a lower likelihood of depression."
            )


        # -------------------------------------------------
        # Probability
        # -------------------------------------------------

        if proba is not None:

            st.subheader("Prediction Probability")

            st.progress(float(proba))

            col5, col6 = st.columns(2)

            with col5:
                st.metric(
                    "Predicted Risk",
                    f"{proba:.1%}"
                )

            with col6:
                st.metric(
                    "Other Class Probability",
                    f"{1 - proba:.1%}"
                )


        # -------------------------------------------------
        # Input Summary
        # -------------------------------------------------

        st.subheader("Student Information")

        display_data = pd.DataFrame({
            "Feature": [
                "Gender",
                "Age",
                "City",
                "Profession",
                "Degree",
                "CGPA",
                "Sleep Duration",
                "Dietary Habits",
                "Suicidal Thoughts",
                "Family History",
                "Academic Pressure",
                "Work Pressure",
                "Study Satisfaction",
                "Job Satisfaction",
                "Work/Study Hours",
                "Financial Stress"
            ],

            "Value": [
                gender,
                age,
                city,
                profession,
                degree,
                cgpa,
                sleep_duration,
                dietary_habits,
                suicidal_thoughts,
                family_history,
                academic_pressure,
                work_pressure,
                study_satisfaction,
                job_satisfaction,
                work_study_hours,
                financial_stress
            ]
        })

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )


    except Exception as e:

        st.error("Prediction could not be completed.")

        st.write("Please check that the input columns and values match "
                 "the features used when the model was trained.")

        st.exception(e)
