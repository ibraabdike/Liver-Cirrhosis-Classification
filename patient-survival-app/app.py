import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the trained model
model = joblib.load("best_model1.pkl")

# Load the preprocessor 
preprocessor = joblib.load("preprocessor1.pkl")

# Define the app title and layout
st.title("Cirrhosis Patient Survival Prediction App")


# Input fields for the features 
n_days = st.number_input("Number of Days (N_Days)", min_value=0, value=0, step=1)
drug = st.selectbox("Type of Drug", ["D-penicillamine", "Placebo"])
age = st.number_input("Age (in days)", min_value=0, max_value=12000, value=3650, step=1)
sex = st.selectbox("Sex", ["M", "F"])
ascites = st.selectbox("Ascites (Presence of Ascites)", ["N", "Y"])
hepatomegaly = st.selectbox("Hepatomegaly (Presence of Hepatomegaly)", ["N", "Y"])
spiders = st.selectbox("Spiders (Presence of Spiders)", ["N", "Y"])
edema = st.selectbox("Edema (Presence of Edema)", ["N", "S", "Y"])
bilirubin = st.number_input("Serum Bilirubin (mg/dl)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, max_value=1000, value=200, step=10)
albumin = st.number_input("Albumin (gm/dl)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
copper = st.number_input("Urine Copper (ug/day)", min_value=0, max_value=500, value=100, step=10)
alk_phos = st.number_input("Alkaline Phosphatase (U/liter)", min_value=0.0, max_value=1000.0, value=80.0, step=5.0)
sgot = st.number_input("SGOT (U/ml)", min_value=0.0, max_value=500.0, value=50.0, step=5.0)
tryglicerides = st.number_input("Triglycerides (mg/dl)", min_value=0, max_value=1000, value=150, step=10)
platelets = st.number_input("Platelets (per cubic ml/1000)", min_value=0, max_value=1000, value=250, step=10)
prothrombin = st.number_input("Prothrombin Time (s)", min_value=0.0, max_value=100.0, value=12.0, step=0.1)
stage = st.selectbox("Histologic Stage of Disease", ["1", "2", "3", "4"])

# Button for making predictions
if st.button("Predict"):
    # Process input values into a DataFrame for model prediction
    input_data = pd.DataFrame(
        {
            "N_Days": [n_days],
            "Drug": [drug],
            "Age": [age],
            "Sex": [sex],
            "Ascites": [ascites],
            "Hepatomegaly": [hepatomegaly],
            "Spiders": [spiders],
            "Edema": [edema],
            "Bilirubin": [bilirubin],
            "Cholesterol": [cholesterol],
            "Albumin": [albumin],
            "Copper": [copper],
            "Alk_Phos": [alk_phos],
            "SGOT": [sgot],
            "Tryglicerides": [tryglicerides],
            "Platelets": [platelets],
            "Prothrombin": [prothrombin],
            "Stage": [stage]
        }
    )

    
    # Preprocess the input data using the preprocessor (ColumnTransformer with OneHotEncoder)
    input_data_encoded = preprocessor.transform(input_data)

    # Make the prediction
    prediction = model.predict(input_data_encoded)

    # Map the prediction back to the corresponding status
    status_map = {0: "C", 1: "CL", 2: "D"}  # Map the integer predictions to the status labels
    predicted_status = status_map.get(prediction[0], "Unknown")

    # Display the prediction
    if predicted_status == "C":
        st.success("The patient is censored (C).")
    elif predicted_status == "CL":
        st.success("The patient is censored due to liver transplant (CL).")
    elif predicted_status == "D":
        st.success("The patient will pass away (D).")
    else:
        st.error("An unknown error occurred.")
