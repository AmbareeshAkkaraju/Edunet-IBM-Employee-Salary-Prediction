import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Set page config
st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="centered"
)

# Custom CSS for better visuals
st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    h1 {color: #2E86C1;}
    .stButton button {background-color: #2E86C1; color: white;}
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown("<h1 style='text-align: center;'>💼 Employee Salary Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict if an employee earns more than $50K/year using a trained ML model.</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📌 About this App")
    st.markdown("""
    - 🎯 **Predict income range**: >50K or <=50K  
    - 🧠 Model: **Random Forest**  
    - 📊 Dataset: **UCI Adult Income**  
    - 🔢 Categorical features encoded numerically  
    """)
    st.info("Fill the form to get salary prediction ⬇️")

# Two-column layout for form
st.subheader("🔍 Enter Employee Information")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("🧓 Age", 18, 75, 30)
    workclass = st.selectbox("🏢 Workclass (0–7)", [0, 1, 2, 3, 4, 5, 6, 7])
    fnlwgt = st.number_input("⚖️ Final Weight (fnlwgt)", value=100000)
    educational_num = st.slider("🎓 Education Number (1–16)", 1, 16, 10)
    marital_status = st.selectbox("💍 Marital Status (0–6)", [0, 1, 2, 3, 4, 5, 6])
    occupation = st.selectbox("💼 Occupation (0–12)", list(range(13)))

with col2:
    relationship = st.selectbox("👨‍👩‍👧 Relationship (0–5)", [0, 1, 2, 3, 4, 5])
    race = st.selectbox("🧑 Race (0–4)", [0, 1, 2, 3, 4])
    gender = st.selectbox("⚧ Gender (0: Female, 1: Male)", [0, 1])
    capital_gain = st.number_input("📈 Capital Gain", value=0)
    capital_loss = st.number_input("📉 Capital Loss", value=0)
    hours_per_week = st.slider("⏱ Hours per Week", 1, 99, 40)
    native_country = st.selectbox("🌍 Native Country (0–10)", list(range(11)))

# Predict button
if st.button("🚀 Predict Salary Range"):
    with st.spinner("🔍 Analyzing and predicting..."):
        input_data = pd.DataFrame({
            'age': [age],
            'workclass': [workclass],
            'fnlwgt': [fnlwgt],
            'educational-num': [educational_num],
            'marital-status': [marital_status],
            'occupation': [occupation],
            'relationship': [relationship],
            'race': [race],
            'gender': [gender],
            'capital-gain': [capital_gain],
            'capital-loss': [capital_loss],
            'hours-per-week': [hours_per_week],
            'native-country': [native_country]
        })

        prediction = model.predict(input_data)
        result = ">50K" if prediction[0] == 1 else "<=50K"
        st.success(f"✅ Predicted Income Range: **{result}**")

# Footer
st.markdown("---")
st.caption("📘 Project by Edunet IBM SkillBuild | 2025 ©")
