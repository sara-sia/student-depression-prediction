import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("student_depression_model.pkl")

st.title("Student Depression Prediction")

age = st.number_input("Age", 15, 60, 20)
academic_pressure = st.slider("Academic Pressure", 1, 5, 3)
cgpa = st.number_input("CGPA", 0.0, 10.0, 7.0)
sleep_duration = st.slider("Sleep Duration (Hours)", 1, 12, 6)

if st.button("Predict"):

    user_data = pd.DataFrame({
        "Age": [age],
        "Academic Pressure": [academic_pressure],
        "CGPA": [cgpa],
        "Sleep Duration": [sleep_duration]
    })

    prediction = model.predict(user_data)[0]

    if prediction == 1:
        st.error("High Risk of Depression")
    else:
        st.success("Low Risk of Depression")