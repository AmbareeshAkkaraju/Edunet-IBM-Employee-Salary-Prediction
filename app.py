import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Page configuration
st.set_page_config(page_title="Employee Salary Prediction", layout="centered")

# Main Title
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>💼 Employee Salary Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 16px;'>Predict whether an employee earns >50K/year based on demographic data using a trained ML model.</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar Info
with st.sidebar:
    st.header("🛠 About This App")
    st.markdown("""
    This application uses a **Random Forest** model trained on the **UCI Adult Income dataset**  
    to predict income range of employees based on:
    - Age, Workclass, Education, etc.
    - Capital gain/loss
    - Hours worked per week
    """)
    st.info("Prediction output: **>50K** or **<=50K** income")

# Dictionaries for label encoding
workclass_options = {
    "Private": 0, "Self-emp-not-inc": 1, "Self-emp-inc": 2,
    "Federal-gov": 3, "Local-gov": 4, "State-gov": 5,
    "Without-pay": 6, "Others": 7
}
marital_status_options = {
    "Never-married": 0, "Married-civ-spouse": 1, "Divorced": 2,
    "Separated": 3, "Widowed": 4, "Married-spouse-absent": 5, "Others": 6
}
occupation_options = {
    "Tech-support": 0, "Craft-repair": 1, "Other-service": 2, "Sales": 3,
    "Exec-managerial": 4, "Prof-specialty": 5, "Handlers-cleaners": 6,
    "Machine-op-inspct": 7, "Adm-clerical": 8, "Farming-fishing": 9,
    "Transport-moving": 10, "Priv-house-serv": 11, "Others": 12
}
relationship_options = {
    "Wife": 0, "Own-child": 1, "Husband": 2, "Not-in-family": 3,
    "Other-relative": 4, "Unmarried": 5
}
race_options = {
    "White": 0, "Asian-Pac-Islander": 1, "Amer-Indian-Eskimo": 2,
    "Other": 3, "Black": 4
}
gender_options = {"Female": 0, "Male": 1}
native_country_options = {
    "United-States": 0, "Cambodia": 1, "England": 2, "Puerto-Rico": 3,
    "Canada": 4, "Germany": 5, "India": 6, "Japan": 7, "China": 8,
    "Others": 9, "Philippines": 10
}

# User Inputs in 2 Columns
st.header("🔍 Enter Employee Information")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 75, 30)
    workclass = st.selectbox("Workclass", list(workclass_options.keys()))
    fnlwgt = st.number_input("Final Weight (fnlwgt)", value=100000)
    educational_num = st.slider("Education Number", 1, 16, 10)
    marital_status = st.selectbox("Marital Status", list(marital_status_options.keys()))
    occupation = st.selectbox("Occupation", list(occupation_options.keys()))

with col2:
    relationship = st.selectbox("Relationship", list(relationship_options.keys()))
    race = st.selectbox("Race", list(race_options.keys()))
    gender = st.selectbox("Gender", list(gender_options.keys()))
    capital_gain = st.number_input("Capital Gain", value=0)
    capital_loss = st.number_input("Capital Loss", value=0)
    hours_per_week = st.slider("Hours per Week", 1, 99, 40)
    native_country = st.selectbox("Native Country", list(native_country_options.keys()))

# Predict Button
if st.button("🎯 Predict Salary Range"):
    with st.spinner("Analyzing data and predicting..."):
        input_data = pd.DataFrame({
            'age': [age],
            'workclass': [workclass_options[workclass]],
            'fnlwgt': [fnlwgt],
            'educational-num': [educational_num],
            'marital-status': [marital_status_options[marital_status]],
            'occupation': [occupation_options[occupation]],
            'relationship': [relationship_options[relationship]],
            'race': [race_options[race]],
            'gender': [gender_options[gender]],
            'capital-gain': [capital_gain],
            'capital-loss': [capital_loss],
            'hours-per-week': [hours_per_week],
            'native-country': [native_country_options[native_country]]
        })

        prediction = model.predict(input_data)
        result = ">50K" if prediction[0] == 1 else "<=50K"
        st.success(f"✅ Predicted Income Range: **{result}**")

# Footer
st.markdown("---")
st.caption("© 2025 | Developed as part of Edunet IBM Skill Build Internship Project")
