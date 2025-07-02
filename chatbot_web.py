import streamlit as st
import pickle
from fuzzywuzzy import process
from datetime import datetime
import re
import base64
from PIL import Image
from io import BytesIO

# -----------------------------
# Load Background Image
# -----------------------------
def load_image(image_path):
    image = Image.open(image_path)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# -----------------------------
# Load Models and Encoders
# -----------------------------
with open("gwl_prediction_model.pkl", "rb") as f:
    gwl_model = pickle.load(f)
with open("crop_recommendation_model.pkl", "rb") as f:
    crop_model = pickle.load(f)
with open("label_encoder_district.pkl", "rb") as f:
    district_encoder = pickle.load(f)
with open("label_encoder_month.pkl", "rb") as f:
    month_encoder = pickle.load(f)
with open("label_encoder_season.pkl", "rb") as f:
    season_encoder = pickle.load(f)
with open("label_encoder_crop.pkl", "rb") as f:
    crop_encoder = pickle.load(f)

# -----------------------------
# Intent Detection
# -----------------------------
def predict_intent(text):
    text = text.lower()
    if "groundwater" in text or "water level" in text:
        return "predict_gwl"
    elif "crop" in text or "recommend" in text or "grow" in text:
        return "recommend_crop"
    elif "hello" in text or "hi" in text or "namaste" in text:
        return "greeting"
    else:
        return "unknown"

# -----------------------------
# Season Mapping
# -----------------------------
def get_season(month):
    if not month:
        return None
    month = month.lower()
    if month in ['june', 'july', 'august', 'september']:
        return 'monsoon'
    elif month in ['october', 'november', 'december', 'january', 'february']:
        return 'winter'
    elif month in ['march', 'april', 'may']:
        return 'summer'
    return None

# -----------------------------
# Info Extraction with Safe Fuzzy Match
# -----------------------------
def extract_info(query):
    query = query.lower()
    
    district_match = process.extractOne(query, district_encoder.classes_)
    district = district_match[0] if district_match and district_match[1] >= 50 else None
    print(f"District Match: {district}")  # Debugging output

    month_match = process.extractOne(query, month_encoder.classes_)
    month = month_match[0] if month_match and month_match[1] >= 50 else None
    print(f"Month Match: {month}")  # Debugging output

    year_match = re.search(r"(20\d{2})", query)
    year = int(year_match.group()) if year_match else datetime.now().year
    print(f"Year Match: {year}")  # Debugging output

    season = get_season(month)

    if district:
        st.session_state.last_district = district
    if month:
        st.session_state.last_month = month
    if season:
        st.session_state.last_season = season
    if year:
        st.session_state.last_year = year

    return district, month, season, year

# -----------------------------
# Crop Suggestions based on GWL
# -----------------------------
def crop_suggestions(gwl):
    if gwl >= 70:
        return ["Paddy", "Sugarcane", "Banana"]
    elif gwl >= 40:
        return ["Maize", "Soybean", "Groundnut"]
    elif gwl >= 20:
        return ["Millet", "Cotton", "Chickpea"]
    else:
        return ["Mustard", "Horsegram", "Pearl Millet"]

# -----------------------------
# Irrigation Method Suggestion
# -----------------------------
def suggest_irrigation(gwl):
    if gwl >= 70:
        return "Flood irrigation or check dam-based irrigation is suitable."
    elif gwl >= 40:
        return "Sprinkler irrigation is recommended to optimize usage."
    elif gwl >= 20:
        return "Drip irrigation is highly recommended to conserve water."
    else:
        return "Use advanced drip irrigation with water harvesting techniques."

# -----------------------------
# Predict GWL
# -----------------------------
def predict_gwl(district, month, season, year):
    if district not in district_encoder.classes_:
        return f"⚠ Sorry, I couldn't find groundwater data for '{district.title()}'"
    features = [
        district_encoder.transform([district])[0],
        month_encoder.transform([month])[0],
        season_encoder.transform([season])[0],
        30.0, 60.0, 6.5, year
    ]
    return round(gwl_model.predict([features])[0], 2)

# -----------------------------
# Recommend Crops
# -----------------------------
def recommend_crop(district, month, season, year):
    if district not in district_encoder.classes_:
        return f"⚠ Sorry, I couldn't find crop data for '{district.title()}'"
    gwl = predict_gwl(district, month, season, year)
    if isinstance(gwl, str):
        return gwl
    crops = crop_suggestions(gwl)
    irrigation = suggest_irrigation(gwl)
    crop_list = "🌾 " + ", ".join(crops)
    return f"""✅ Based on groundwater level:
📍 Location: {district.title()}, Month: {month.title()}, Year: {year}
🌱 Suitable Crops: {crop_list}
💧 Groundwater Level: {gwl} meters
🚿 Suggested Irrigation: {irrigation}"""

# -----------------------------
# Background + Title
# -----------------------------
image_path = r"C:\Users\user\Desktop\colab\background.jpg.png"
bg_image = load_image(image_path)

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bg_image}");
        background-size: cover;
        background-attachment: fixed;
        color: black;
    }}
    .title {{
        background-color: rgba(255, 255, 255, 0.8);
        padding: 10px;
        border-radius: 12px;
        font-weight: bold;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="title">🚜 KrishiBot - Smart Farming Assistant</h1>', unsafe_allow_html=True)
st.subheader("🌱 Ask me about groundwater levels or the best crops for your season and area!")

# -----------------------------
# Session Memory
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_district" not in st.session_state:
    st.session_state.last_district = None
if "last_month" not in st.session_state:
    st.session_state.last_month = None
if "last_season" not in st.session_state:
    st.session_state.last_season = None
if "last_year" not in st.session_state:
    st.session_state.last_year = None

# -----------------------------
# Chat Input & Logic
# -----------------------------
user_input = st.chat_input("👩‍🌾 Ask your farming question here...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    intent = predict_intent(user_input)
    response = ""

    try:
        district, month, season, year = extract_info(user_input)
        district = district or st.session_state.last_district
        month = month or st.session_state.last_month
        season = season or st.session_state.last_season
        year = year or st.session_state.last_year

        if intent == "predict_gwl":
            if district and month and season:
                gwl = predict_gwl(district, month, season, year)
                if isinstance(gwl, str):
                    response = gwl
                else:
                    response = f"💧 In {district.title()}, during {month.title()} {year}, the estimated groundwater level is around {gwl} meters."
            else:
                response = "⚠ Please provide valid district and month to check groundwater level."

        elif intent == "recommend_crop":
            if district and month and season:
                response = recommend_crop(district, month, season, year)
            else:
                response = "⚠ Please provide valid district and month to suggest crops."

        elif intent == "greeting":
            response = "🙏 Namaste! I'm KrishiBot — your farming buddy. Ask me about crops or water levels."

        else:
            response = "🌽 Sorry, I didn’t catch that. Try asking like: 'What is groundwater level in Tirupati in June 2025?'"

    except Exception as e:
        response = f"⚠ Oops! Something went wrong: {e}"

    st.session_state.chat_history.append(("bot", response))

# -----------------------------
# Display Chat
# -----------------------------
for sender, msg in st.session_state.chat_history:
    if sender == "user":
        st.chat_message("user").markdown(f"👩‍🌾 {msg}")
    else:
        st.chat_message("assistant").markdown(msg)
