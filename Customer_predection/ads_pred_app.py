import streamlit as st
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')
import os
import streamlit as st

image_path = "ads_logo.png"
if os.path.exists(image_path):
    st.image(image_path, caption="Ad Click Prediction", use_column_width=True)
else:
    st.image("https://example.com/path-to-image/ads_logo.png", caption="Ad Click Prediction", use_column_width=True)


# Page Settings
st.set_page_config(page_title="Ad Click Prediction App", layout="centered", page_icon="📈")

# App Header
image_path = "ads_logo.png"
if os.path.exists(image_path):
    st.image(image_path, caption="Ad Click Prediction", use_container_width=True)
else:
    st.image("https://media.istockphoto.com/id/1699842032/photo/online-shopping-e-commerce-concept-woman-using-mobile-phone-and-laptop-computer-for-online.webp?a=1&b=1&s=612x612&w=0&k=20&c=M_f7m5KkrCB6Tj7R6ShPOqWYWIuZqUCQTPgWPzjsJpI=.png", caption="Ad Click Prediction", use_container_width=True)
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
st.sidebar.image("ads.png")
st.sidebar.header("User Inputs")
daily_time_spent_on_site = st.sidebar.slider('Daily Time Spent on Site (minutes):', 0.0, 500.0, step=0.1, value=60.0)
age = st.sidebar.slider('Age:', 10, 100, step=1, value=30, help="Enter the user's age in years.")
area_income = st.sidebar.number_input('Area Income (average income of the user’s area):', 0.0, 2000000.0, step=100.0, value=50000.0)
daily_internet_usage = st.sidebar.slider('Daily Internet Usage (minutes):', 0.0, 1000.0, step=0.1, value=200.0)
gender = st.sidebar.radio('Gender:', ['Male', 'Female'])

# Predict Button
if st.sidebar.button("Predict"):
    # Encode gender as binary
    gender_binary = 1 if gender == "Male" else 0

    # Encode age into an age group
    encoded_age_group = findAgeGroup(age)

    # Load the trained model
    model = joblib.load('ad_click_model.pkl')

    # Create input data for prediction
    input_data = np.array([[daily_time_spent_on_site, encoded_age_group, area_income, daily_internet_usage, gender_binary]])

    # Make prediction
    prediction = model.predict(input_data)

    # Display result
    if prediction[0] == 1:
        st.success(f"The user is likely to click on the next ad. 🎯")
    else:
        st.error(f"The user is unlikely to click on the next ad. ❌")
