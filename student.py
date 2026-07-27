import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load('student_depression_model.pkl')

st.title("Student Depression Prediction System")

# Create input fields matching your model's feature order and names
age = st.number_input("Age", min_value=15, max_value=60, value=20)
academic_pressure = st.slider("Academic Pressure (1-5)", 1, 5, 3)
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.0)
sleep_duration = st.slider("Sleep Duration (Hours)", 1, 12, 6)

if st.button("Predict"):
    # Combine inputs into DataFrame matching exact column names used in training
    user_data = pd.DataFrame([{
        'age': age,
        'academic_pressure': academic_pressure,
        'cgpa': cgpa,
        'sleep_duration': sleep_duration
    }])
    
    prediction = model.predict(user_data)
    
    if prediction[0] == 1:
        st.error("High Risk of Depression detected.")
    else:
        st.success("Low Risk of Depression detected.")