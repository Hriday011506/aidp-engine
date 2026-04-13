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
# 🗄️ DATABASE
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
st.set_page_config(page_title="AIDP Engine", layout="wide")

st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}

/* Glass cards */
.card {
    background: rgba(255, 255, 255, 0.05);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 20px;
}

/* KPI cards */
.kpi {
    background: linear-gradient(135deg, #1e293b, #020617);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 0 20px rgba(0,255,255,0.1);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #06b6d4, #3b82f6);
    color: white;
    border-radius: 10px;
    padding: 10px;
    border: none;
    font-weight: bold;
}

/* Titles */
h1, h2, h3 {
    color: #f1f5f9;
}

/* Input boxes */
input, .stSelectbox {
    background-color: #020617 !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)
# ==============================
# 🔐 AUTH FUNCTIONS
# ==============================
def signup(email, password):
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    users_collection.insert_one({"email": email, "password": hashed})

def login(email, password):
    user = users_collection.find_one({"email": email})
    if user and bcrypt.checkpw(password.encode(), user["password"]):
        return user
    return None

# ==============================
# 💰 PRICE FUNCTION 
# ==============================
def fetch_product_price(product_name):
    try:
        params = {
            "engine": "google_shopping",
            "q": product_name,
            "gl": "in",
            "hl": "en",
            "api_key": SERPAPI_KEY
        }

        results = GoogleSearch(params).get_dict()
        products = results.get("shopping_results", [])

        if products:
            p = products[0]

            price = p.get("price") or p.get("extracted_price")

            if price:
                price = str(price)

             
                if "₹" in price:
                    return price
                else:
                    return f"₹{price}"

    except:
        pass

    return "₹ Data unavailable"
# ==============================
# 🌦 WEATHER
# ==============================
def get_weather(city):
    try:
        geo = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={city}").json()
        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]

        weather = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        ).json()

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
# 🔥 VIRAL SCORE
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

    st.markdown("""
    <div style='text-align:center; padding:60px;'>
        <h1 style='font-size:60px;'>🚀 AIDP Engine</h1>
        <h3>AI-Powered Retail Demand Intelligence</h3>
        <p style='color:gray;'>Predict. Optimize. Grow.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        if st.button("🔐 Login", use_container_width=True):
            st.session_state.page = "login"

        if st.button("📝 Signup", use_container_width=True):
            st.session_state.page = "signup"

        st.markdown("</div>", unsafe_allow_html=True)

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
        st.success("Account created successfully")
        st.session_state.page = "login"

# ==============================
# 📊 DASHBOARD
# ==============================
elif st.session_state.page == "dashboard" and st.session_state.user:

    st.markdown("<h1>📊 AIDP Intelligence Dashboard</h1>", unsafe_allow_html=True)
    st.caption("AI-powered retail insights & forecasting")

    # Logout button
    col1, col2 = st.columns([8,1])
    with col2:
        if st.button("Logout"):
            st.session_state.user = None
            st.session_state.page = "welcome"

    st.divider()

    # ---------------- INPUT CARD ----------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📥 Business Inputs")

    col1, col2, col3 = st.columns(3)

    with col1:
        product = st.text_input("📦 Product", "Wheat Flour")

    with col2:
        city = st.text_input("📍 City", "Jaipur")

    with col3:
        month_name = st.selectbox("📅 Month", list(calendar.month_name)[1:])

    st.markdown("</div>", unsafe_allow_html=True)

    month = list(calendar.month_name).index(month_name)
    year = 2025

    # ---------------- DATA ----------------
    holiday = get_holidays(year, month)
    temp = get_weather(city)
    viral = simulate_viral_score(product)
    price = fetch_product_price(product)

    # ---------------- KPI CARDS ----------------
    st.subheader("📊 Market Intelligence")

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f"<div class='kpi'><h4>🌡 Temp</h4><h2>{temp:.1f}°C</h2></div>", unsafe_allow_html=True)

    with k2:
        st.markdown(f"<div class='kpi'><h4>📅 Holidays</h4><h2>{holiday}</h2></div>", unsafe_allow_html=True)

    with k3:
        st.markdown(f"<div class='kpi'><h4>🔥 Trend</h4><h2>{viral}</h2></div>", unsafe_allow_html=True)

    with k4:
        st.markdown(f"<div class='kpi'><h4>💰 Price</h4><h2>{price}</h2></div>", unsafe_allow_html=True)

    st.divider()

    # ---------------- PREDICTION ----------------
    if st.button("🚀 Predict Demand"):

        input_df = pd.DataFrame({
            "holiday_count":[holiday],
            "avg_temp":[temp],
            "viral_score":[viral]
        })

        pred = model.predict(input_df)[0]
        inventory = pred * 1.1

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.subheader("📈 AI Forecast Results")

        r1, r2 = st.columns(2)
        r1.metric("📦 Sales", int(pred))
        r2.metric("📊 Inventory", int(inventory))

        st.markdown("</div>", unsafe_allow_html=True)

        # Charts
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📊 Sales vs Inventory")

        chart_df = pd.DataFrame({
            "Type":["Sales","Inventory"],
            "Value":[pred, inventory]
        })

        st.bar_chart(chart_df.set_index("Type"))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📈 Demand Trend")

        trend = pd.DataFrame({
            "Month": list(range(1,13)),
            "Demand": np.linspace(200, pred, 12)
        })

        st.line_chart(trend.set_index("Month"))
        st.markdown("</div>", unsafe_allow_html=True)

        # Insights
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🧠 AI Insights")

        if viral > 70:
            st.success("🔥 High demand expected — trending product.")
        elif holiday > 6:
            st.info("📅 Demand boost due to holidays.")
        else:
            st.warning("⚖️ Stable demand expected.")

        st.markdown("</div>", unsafe_allow_html=True)

    # INPUTS
    st.subheader("📥 Inputs")

    col1, col2, col3 = st.columns(3)

    with col1:
        product = st.text_input("Product", "Wheat Flour")

    with col2:
        city = st.text_input("City", "Jaipur")

    with col3:
        month_name = st.selectbox("Month", list(calendar.month_name)[1:])

    month = list(calendar.month_name).index(month_name)
    year = 2025

    # FETCH DATA
    holiday = get_holidays(year, month)
    temp = get_weather(city)
    viral = simulate_viral_score(product)
    price = fetch_product_price(product)

    # KPI
    st.subheader("📊 Market Indicators")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🌡 Temperature", f"{temp:.1f} °C")
    k2.metric("📅 Holidays", holiday)
    k3.metric("🔥 Trend Score", viral)
    k4.metric("💰 Market Price", price)

    st.divider()

    # PREDICT
    if st.button("🚀 Predict Demand"):

        df = pd.DataFrame({
            "holiday_count":[holiday],
            "avg_temp":[temp],
            "viral_score":[viral]
        })

        pred = model.predict(df)[0]
        inventory = pred * 1.1

        st.subheader("📈 Results")

        r1, r2 = st.columns(2)
        r1.metric("📦 Sales", int(pred))
        r2.metric("📊 Inventory", int(inventory))

        # Charts
        st.subheader("📊 Sales vs Inventory")

        chart_df = pd.DataFrame({
            "Type":["Sales","Inventory"],
            "Value":[pred, inventory]
        })
        st.bar_chart(chart_df.set_index("Type"))

        st.subheader("📈 Demand Trend")

        trend = pd.DataFrame({
            "Month": list(range(1,13)),
            "Demand": np.linspace(200, pred, 12)
        })
        st.line_chart(trend.set_index("Month"))

        # Insights
        st.subheader("🧠 Insights")

        if viral > 70:
            st.success("🔥 High demand expected")
        elif holiday > 6:
            st.info("📅 Demand may increase due to holidays")
        else:
            st.warning("⚖️ Stable demand expected")
