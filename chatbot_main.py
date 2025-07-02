import streamlit as st
import pickle
import os
from PIL import Image

# Set page configuration
st.set_page_config(page_title="KrishiBot", layout="wide")

# Set background image if it exists
bg_path = "C:/Users/user/Desktop/colab/background.jpg.png"
if os.path.exists(bg_path):
    st.image(Image.open(bg_path), use_column_width=True)

# App Title
st.title("🤖 KrishiBot - AI Assistant for Groundwater & Crop Guidance")
st.subheader("Ask your farming-related questions below 👇")

# Load models and encoders with error handling
def load_pickle(file_path, description):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.warning(f"{description} file not found: {file_path}")
        return None

# Paths to your models
gwl_model = load_pickle("gwl_prediction_model.pkl", "Groundwater Level Model")
crop_model = load_pickle("crop_recommendation_model.pkl", "Crop Recommendation Model")
le_district = load_pickle("label_encoder_district.pkl", "District Label Encoder")
le_month = load_pickle("label_encoder_month.pkl", "Month Label Encoder")
le_season = load_pickle("label_encoder_season.pkl", "Season Label Encoder")
le_crop = load_pickle("label_encoder_crop.pkl", "Crop Label Encoder")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_input = st.chat_input("💬 Type your question here")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Basic intent recognition
    lower_input = user_input.lower()
    if "groundwater" in lower_input or "water level" in lower_input:
        intent = "groundwater_prediction"
    elif "crop" in lower_input or "recommend" in lower_input:
        intent = "crop_recommendation"
    else:
        intent = "general"

    response = "I'm not sure how to respond. Please ask about groundwater or crop recommendation."

    if intent == "groundwater_prediction":
        # Sample hardcoded values (replace with NLP extraction if needed)
        district = "tirupati"
        month = "july"
        season = "kharif"
        year = 2025

        if all([gwl_model, le_district, le_month, le_season]):
            try:
                input_data = [[
                    le_district.transform([district])[0],
                    le_month.transform([month])[0],
                    le_season.transform([season])[0],
                    year
                ]]
                prediction = gwl_model.predict(input_data)[0]
                response = f"📉 Predicted Groundwater Level in *{district.title()}* during *{month.title()} {year}* is *{prediction:.2f} m*."
            except Exception as e:
                response = f"⚠ Error in groundwater prediction: {e}"
        else:
            response = "⚠ Required files for groundwater prediction are missing."

    elif intent == "crop_recommendation":
        district = "tirupati"
        month = "july"
        season = "kharif"
        year = 2025
        gwl_value = 10.0  # Example

        if all([crop_model, le_district, le_month, le_season, le_crop]):
            try:
                input_data = [[
                    le_district.transform([district])[0],
                    le_month.transform([month])[0],
                    le_season.transform([season])[0],
                    year,
                    gwl_value
                ]]
                prediction = crop_model.predict(input_data)[0]
                crop_name = le_crop.inverse_transform([prediction])[0]
                response = f"🌾 Recommended Crop for *{district.title()}* in *{month.title()} {year}* is *{crop_name.title()}*."
            except Exception as e:
                response = f"⚠ Error in crop recommendation: {e}"
        else:
            response = "⚠ Required files for crop recommendation are missing."

    # Add response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# Display chat
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])
