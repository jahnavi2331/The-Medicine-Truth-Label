import streamlit as st
import pandas as pd
import joblib

model = joblib.load("adherence_model.pkl")
features = joblib.load("features.pkl")

st.title("Patient Adherence Risk Prediction App")

input_data = []
for feature in features:
    value = st.number_input(feature, value=0.0)
    input_data.append(value)

if st.button("Predict"):
    input_df = pd.DataFrame([input_data], columns=features)
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Adherence Risk Score: {round(prediction, 2)}")
