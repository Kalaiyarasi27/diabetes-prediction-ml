import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Diabetes Prediction", page_icon="💉", layout="centered")

st.markdown("""
    <h1 style='text-align:center;color:#ff4d4d;'>💉 Diabetes Prediction App</h1>
    <p style='text-align:center;'>Enter your health details below to predict diabetes.</p>
""", unsafe_allow_html=True)

model = joblib.load("model/diabetes_model.pkl")
scaler = joblib.load("model/scaler.pkl")

col1, col2 = st.columns(2)

with col1:
    Pregnancies = st.number_input("Pregnancies", 0, 20)
    Glucose = st.number_input("Glucose", 0, 300)
    BloodPressure = st.number_input("Blood Pressure", 0, 200)
    SkinThickness = st.number_input("Skin Thickness", 0, 100)

with col2:
    Insulin = st.number_input("Insulin", 0, 900)
    BMI = st.number_input("BMI", 0.0, 70.0)
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
    Age = st.number_input("Age", 1, 120)

if st.button("🔍 Predict"):
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                            Insulin, BMI, DiabetesPedigreeFunction, Age]])

    scaled = scaler.transform(input_data)
    prediction = model.predict(scaled)[0]
    
    if prediction == 1:
        st.error("⚠️ **High chance of Diabetes! Consult a doctor.**")
    else:
        st.success("✅ **No Diabetes – You seem safe!**")
