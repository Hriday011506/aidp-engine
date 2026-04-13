import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import calendar
import holidays
from sklearn.ensemble import RandomForestRegressor
from pymongo import MongoClient
import bcrypt
from serpapi import GoogleSearch

# ==============================
# 🔐 CONFIG
# ==============================
MONGO_URI = "mongodb+srv://hridaymahajan1979_db_user:hriday@aidp.jz72py2.mongodb.net/?retryWrites=true&w=majority"
SERPAPI_KEY = "bbc8aca8053bbe60b9c7017e236f71656667f6b4d2bbf3b2da695084ad8766b4"

# ==============================
# 🗄️ DB
# ==============================
client = MongoClient(MONGO_URI)
db = client["aidp_db"]
users_collection = db["users"]

# ==============================
# 🔐 SESSION
# ==============================
if "user" not in st.session_state:
    st.session_state.user = None
if "page" not in st.session_state:
    st.session_state.page = "welcome"

# ==============================
# 🎨 UI STYLE
# ==============================
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: white;
}
button {
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# 🔐 AUTH FUNCTIONS
# ==============================
def signup(email, password):
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    users_collection.insert_one({
        "email": email,
        "password": hashed
    })

def login(email, password):
    user = users_collection.find_one({"email": email})
    if user and bcrypt.checkpw(password.encode(), user["password"]):
        return user
    return None

# ==============================
# 💰 PRICE FIXED
# ==============================
def fetch_product_price(product_name):
    try:
        params = {
            "engine": "google_shopping",
            "q": product_name,
            "api_key": SERPAPI_KEY
        }
        results = GoogleSearch(params).get_dict()
        products = results.get("shopping_results", [])

        if products:
            price = products[0].get("price") or products[0].get("extracted_price")
            if price:
                return f"₹{price}"
    except:
        pass

    return "₹ Not Available"

# ==============================
# 🌦️ WEATHER
# ==============================
def get_weather(city):
    try:
        geo = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={city}").json()
        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]
        weather = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true").json()
        return weather["current_weather"]["temperature"]
    except:
        return 25

# ==============================
# 📅 HOLIDAYS
# ==============================
def get_holidays(year, month):
    india_holidays = holidays.India(years=year)
    total_days = calendar.monthrange(year, month)[1]
    return sum(
        1 for d in range(1, total_days + 1)
        if datetime.date(year, month, d).weekday() >= 5
        or datetime.date(year, month, d) in india_holidays
    )

# ==============================
# 🔥 VIRAL
# ==============================
def simulate_viral_score(product):
    np.random.seed(abs(hash(product)) % 100)
    return np.random.randint(30, 90)

# ==============================
# 🤖 MODEL
# ==============================
@st.cache_data
def load_data():
    df = pd.DataFrame({
        "holiday_count": np.random.randint(0, 10, 100),
        "avg_temp": np.random.randint(10, 40, 100),
        "viral_score": np.random.randint(0, 100, 100)
    })
    df["sales"] = 200 + df["holiday_count"]*50 + df["avg_temp"]*10 + df["viral_score"]*5
    return df

@st.cache_resource
def train_model(df):
    model = RandomForestRegressor()
    model.fit(df[["holiday_count","avg_temp","viral_score"]], df["sales"])
    return model

model = train_model(load_data())

# ==============================
# 🚀 WELCOME PAGE
# ==============================
if st.session_state.page == "welcome":

    st.markdown("<h1 style='text-align:center;'>🚀 Welcome to AIDP</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center;'>AI Demand Prediction Platform</h4>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        if st.button("Login", use_container_width=True):
            st.session_state.page = "login"
        if st.button("Signup", use_container_width=True):
            st.session_state.page = "signup"

# ==============================
# 🔐 LOGIN
# ==============================
elif st.session_state.page == "login":

    st.title("🔐 Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = login(email, password)
        if user:
            st.session_state.user = user
            st.session_state.page = "dashboard"
        else:
            st.error("Invalid credentials")

# ==============================
# 📝 SIGNUP
# ==============================
elif st.session_state.page == "signup":

    st.title("📝 Signup")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Create Account"):
        signup(email, password)
        st.success("Account created")
        st.session_state.page = "login"

# ==============================
# 📊 DASHBOARD
# ==============================
elif st.session_state.page == "dashboard" and st.session_state.user:

    st.title("📊 Dashboard")

    if st.button("Logout"):
        st.session_state.user = None
        st.session_state.page = "welcome"

    product = st.text_input("Product", "Wheat Flour")
    city = st.text_input("City", "Jaipur")

    year = 2025
    month = 5

    holiday = get_holidays(year, month)
    temp = get_weather(city)
    viral = simulate_viral_score(product)

    if st.button("Predict"):

        df = pd.DataFrame({
            "holiday_count":[holiday],
            "avg_temp":[temp],
            "viral_score":[viral]
        })

        pred = model.predict(df)[0]
        price = fetch_product_price(product)

        st.subheader("Results")

        col1, col2, col3 = st.columns(3)
        col1.metric("Sales", int(pred))
        col2.metric("Inventory", int(pred*1.1))
        col3.metric("Price", price)

        # ✅ Better readable charts
        chart_df = pd.DataFrame({
            "Type":["Sales","Inventory"],
            "Value":[pred, pred*1.1]
        })

        st.bar_chart(chart_df.set_index("Type"))

        trend = pd.DataFrame({
            "Month": list(range(1,13)),
            "Demand": np.linspace(200, pred, 12)
        })

        st.line_chart(trend.set_index("Month"))
