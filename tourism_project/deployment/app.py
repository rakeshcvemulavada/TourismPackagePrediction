# ==========================================
# Streamlit Interactive Web Application
# ==========================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configure Streamlit page
st.set_page_config(page_title="Wellness Tourism Predictor", layout="centered")

# Load the serialised pipeline model
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.0.joblib")
model = joblib.load(model_path)

st.title("🌲 Wellness Tourism Package Purchase Predictor")
st.write("""
Use this tool to predict the likelihood of a customer subscribing to our Wellness Tourism package.
""")

st.markdown("### Customer Profile Inputs")

# Collect User inputs via Streamlit interactive widgets
col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", min_value=18, max_value=100, value=30)
    TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    Occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business", "Large Business"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    NumberOfPersonsVisiting = st.number_input("Number of Persons Accompanying", min_value=1, max_value=10, value=2)
    PreferredPropertyStar = st.selectbox("Preferred Hotel Star Rating", [1, 2, 3, 4, 5])

with col2:
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    NumberOfTrips = st.number_input("Number of Annual Trips", min_value=0, max_value=50, value=2)
    Passport = st.selectbox("Has Valid Passport?", ["Yes", "No"])
    OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
    NumberOfChildrenVisiting = st.number_input("Number of Children Accompanying", min_value=0, max_value=5, value=0)
    Designation = st.selectbox("Current Designation", ["Executive", "Manager", "Senior Manager", "VP"])
    MonthlyIncome = st.number_input("Monthly Income (INR)", min_value=5000, max_value=500000, value=25000)

st.markdown("### Interaction & Sales Pitch Details")
col3, col4 = st.columns(2)

with col3:
    PitchSatisfactionScore = st.slider("Pitch Satisfaction Score (1-5)", min_value=1, max_value=5, value=3)
    ProductPitched = st.selectbox("Product Type Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe"])

with col4:
    NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=20, value=2)
    DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=120, value=15)

# Construct input data dictionary matching the structure required by the preprocessing steps
input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonsVisiting,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "ProductPitched": ProductPitched,
    "NumberOfFollowups": NumberOfFollowups,
    "DurationOfPitch": DurationOfPitch
}])

# Set prediction probability classification threshold
classification_threshold = 0.5

if st.button("💡 Run Prediction"):
    prediction_probability = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_probability >= classification_threshold).astype(int)

    st.markdown("--- ")
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"🎯 **The customer is likely to purchase the package.** (Probability: {prediction_probability:.2f})")
    else:
        st.error(f"❌ **The customer is unlikely to purchase the package.** (Probability: {prediction_probability:.2f})")
