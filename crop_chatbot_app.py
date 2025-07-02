import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
from keras.models import load_model

# Load encoders and model
season_encoder = pickle.load(open("season_encoder.pkl", "rb"))
district_encoder = pickle.load(open("district_encoder.pkl", "rb"))
crop_encoder = pickle.load(open("crop_encoder.pkl", "rb"))
model = load_model("my_safe_model.keras")

# Title
st.title("🌾 Groundwater & Crop Prediction Chatbot")

# Chatbot-like interface
user_input = st.text_input("👤 Ask me something like:", "What is the groundwater level of Tirupati in May 2025?")

def parse_question(question):
    question = question.lower()
    try:
        # Extract district
        districts = district_encoder.classes_
        district = next(d for d in districts if d.lower() in question)

        # Extract month and year
        months = {
            'january':1, 'february':2, 'march':3, 'april':4, 'may':5, 'june':6,
            'july':7, 'august':8, 'september':9, 'october':10, 'november':11, 'december':12
        }
        month = next((months[m] for m in months if m in question), datetime.datetime.now().month)
        year = int(next((w for w in question.split() if w.isdigit() and len(w) == 4), datetime.datetime.now().year))

        # Define rules for seasons
        if month in [6, 7, 8, 9]: season = "kharif"
        elif month in [10, 11]: season = "rabi"
        elif month in [12, 1, 2]: season = "winter"
        else: season = "summer"

        return district, season, year, month
    except Exception as e:
        st.error(f"❌ Couldn't understand the question properly: {e}")
        return None, None, None, None

def predict_groundwater(district, season, year, month):
    try:
        # Create dummy inputs (replace with actual inputs if available)
        temp = 28.0
        humidity = 65.0
        ph = 6.5
        groundwater_level = 10.0  # dummy input

        # Encode categorical
        district_encoded = district_encoder.transform([district])[0]
        season_encoded = season_encoder.transform([season])[0]

        # Input format
        X = pd.DataFrame([[temp, humidity, ph, groundwater_level, season_encoded, district_encoded, year, month]],
                         columns=['Temperature', 'Humidity', 'ph', 'Ground Water Level', 'Season',
                                  'District', 'Year', 'Month'])

        # Predict
        y_pred = model.predict(X)
        groundwater_level_pred = float(y_pred[0][0])

        # Dummy crop recommendation logic
        crop_idx = int(np.round(groundwater_level_pred)) % len(crop_encoder.classes_)
        recommended_crop = crop_encoder.inverse_transform([crop_idx])[0]

        return groundwater_level_pred, recommended_crop
    except Exception as e:
        st.error(f"Oops something went wrong while predicting: {e}")
        return None, None

# Handle user question
if st.button("💬 Ask"):
    district, season, year, month = parse_question(user_input)
    if district and season:
        groundwater, crop = predict_groundwater(district, seaso
                                                n, year, month)
        if groundwater and crop:
            st.success(f"📍 Predicted Groundwater Level in **{district}** for **{month}-{year}** is **{groundwater:.2f} meters** 💧")
            st.success(f"🌾 Recommended Crop: **{crop}**")

