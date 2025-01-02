import streamlit as st
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')
import os
from PIL import Image
import requests

# Page Settings
st.set_page_config(page_title="Ad Click Prediction App", layout="centered", page_icon="📈")

# App Header
st.image("https://github.com/SeifEldeen259/Customer_predection/blob/main/Customer_predection/ads_logo.png?raw=true", caption="Ad Click Prediction",use_column_width=True)
st.markdown("# 📊 Ad Click Prediction")
st.text("Predict whether a user will click on the next ad based on their data.")

# Function to encode age into age groups
def findAgeGroup(Age):
    if Age >= 19 and Age < 35:
        return 1
    elif Age >= 35 and Age < 50:
        return 2
    else:
        return 3

# Sidebar Inputs
st.sidebar.image('https://github.com/SeifEldeen259/Customer_predection/blob/main/Customer_predection/ads.png?raw=true')
st.sidebar.header("User Inputs")
daily_time_spent_on_site = st.sidebar.slider('Daily Time Spent on Site (minutes):', 0.0, 500.0, step=0.1, value=60.0)
age = st.sidebar.slider('Age:', 10, 100, step=1, value=30, help="Enter the user's age in years.")
area_income = st.sidebar.number_input('Area Income (average income of the user’s area):', 0.0, 2000000.0, step=100.0, value=50000.0)
daily_internet_usage = st.sidebar.slider('Daily Internet Usage (minutes):', 0.0, 1000.0, step=0.1, value=200.0)
gender = st.sidebar.radio('Gender:', ['Male', 'Female'])

model_url = "https://github.com/SeifEldeen259/Customer_predection/raw/refs/heads/main/Customer_predection/ad_model.pkl"
model_local_path = "ad_model.pkl"

# Download the model if it doesn't exist locally
if not os.path.exists(model_local_path):
    try:
        st.info("Downloading the model...")
        response = requests.get(model_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        with open(model_local_path, "wb") as f:
            f.write(response.content)
        st.success("Model downloaded successfully!")
    except Exception as e:
        st.error(f"Failed to download the model: {e}")
        model = None
else:
    st.info("Model file already exists locally.")

# Load the model
model = None
if os.path.exists(model_local_path):
    try:
        model = joblib.load(model_local_path)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading the model: {e}")

# Input data
daily_time_spent_on_site = st.number_input("Daily Time Spent on Site", value=68.5)
encoded_age_group = st.number_input("Encoded Age Group", value=1)
input_data = np.array([[daily_time_spent_on_site, encoded_age_group]])

# Make predictions
if st.button("Predict"):
    if model is not None:
        try:
            prediction = model.predict(input_data)
            st.success(f"Prediction: {prediction[0]}")  # Assuming binary classification
            if prediction[0] == 1:
                st.write("The user is predicted to click the ad.")
            else:
                st.write("The user is not predicted to click the ad.")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.error("Model is not defined. Cannot make predictions.")
