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
# CONFIG
# ==============================
MONGO_URI = st.secrets.get("MONGO_URI", "")
SERPAPI_KEY = st.secrets.get("SERPAPI_KEY", "")

# ==============================
# DATABASE
# ==============================
@st.cache_resource
def get_database():
    if not MONGO_URI:
        return None
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)
        client.admin.command("ping")
        return client["aidp_db"]
    except Exception as exc:
        st.error(f"MongoDB connection failed: {exc}")
        return None

db = get_database()
users_collection = db["users"] if db is not None else None

# ==============================
# SESSION
# ==============================
if "user" not in st.session_state:
    st.session_state.user = None
if "page" not in st.session_state:
    st.session_state.page = "welcome"

# ==============================
# UI STYLE
# ==============================
st.set_page_config(page_title="AIDP Engine", layout="wide")

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: #e2e8f0;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #0f172a);
    border-right: 1px solid rgba(255,255,255,0.1);
}
.card {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 25px;
    transition: 0.3s;
}
.card:hover { transform: scale(1.02); }
.kpi {
    background: linear-gradient(135deg, #020617, #1e293b);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 0 25px rgba(0,255,255,0.15);
}
.stButton>button {
    background: linear-gradient(135deg, #06b6d4, #3b82f6);
    border: none;
    color: white;
    padding: 12px;
    border-radius: 12px;
    font-weight: bold;
}
input, textarea {
    background-color: #020617 !important;
    color: white !important;
}
h1 { font-size: 42px; }
h2 { font-size: 28px; }
</style>
""", unsafe_allow_html=True)

# ==============================
# AUTH FUNCTIONS
# ==============================
def signup(email, password, gst, turnover):
    if users_collection is None:
        return False, "Database is not configured. Add MONGO_URI in Streamlit Secrets."

    try:
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        user_doc = {
            "email": email,
            "password": hashed,
            "gst": gst,
            "turnover": turnover,
            "created_at": datetime.datetime.now()
        }
        users_collection.insert_one(user_doc)
        return True, "Account created successfully"
    except Exception as exc:
        return False, f"Signup failed: {exc}"


def login(email, password):
    if users_collection is None:
        return None
    try:
        user = users_collection.find_one({"email": email})
        if user and bcrypt.checkpw(password.encode(), user["password"]):
            return user
    except Exception:
        return None
    return None

# ==============================
# PRICE FUNCTION
# ==============================
def fetch_product_price(product_name):
    if not SERPAPI_KEY:
        return "₹ Data unavailable"

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
            price = products[0].get("price") or products[0].get("extracted_price")
            if price is not None:
                price = str(price)
                return price if "₹" in price else f"₹{price}"
    except Exception:
        pass

    return "₹ Data unavailable"

# ==============================
# WEATHER
# ==============================
def get_weather(city):
    try:
        response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=10
        )
        response.raise_for_status()
        geo = response.json()
        results = geo.get("results", [])
        if not results:
            return 25

        lat = results[0]["latitude"]
        lon = results[0]["longitude"]

        weather_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": "true"},
            timeout=10
        )
        weather_response.raise_for_status()
        return weather_response.json()["current_weather"]["temperature"]
    except Exception:
        return 25

# ==============================
# HOLIDAYS
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
# VIRAL SCORE
# ==============================
def simulate_viral_score(product):
    np.random.seed(abs(hash(product)) % 100)
    return np.random.randint(30, 90)

# ==============================
# MODEL
# ==============================
@st.cache_data
def load_data():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "holiday_count": rng.integers(0, 10, 100),
        "avg_temp": rng.integers(10, 40, 100),
        "viral_score": rng.integers(0, 100, 100)
    })
    df["sales"] = (
        200
        + df["holiday_count"] * 50
        + df["avg_temp"] * 10
        + df["viral_score"] * 5
    )
    return df


@st.cache_resource
def train_model(df):
    model = RandomForestRegressor(random_state=42)
    model.fit(
        df[["holiday_count", "avg_temp", "viral_score"]],
        df["sales"]
    )
    return model


model = train_model(load_data())

# ==============================
# WELCOME PAGE
# ==============================
if st.session_state.page == "welcome":
    st.markdown("""
    <div style='text-align:center; padding:80px;'>
        <h1 style='font-size:70px;'>🚀 AIDP Engine</h1>
        <h3>AI Demand Intelligence Platform</h3>
        <p style='color:gray;'>Predict demand. Optimize inventory. Maximize profit.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if st.button("🔐 Login", key="login_btn", use_container_width=True):
            st.session_state.page = "login"
        if st.button("📝 Signup", key="signup_btn", use_container_width=True):
            st.session_state.page = "signup"
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# LOGIN
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
            st.rerun()
        else:
            st.error("Invalid credentials or database unavailable.")

# ==============================
# SIGNUP
# ==============================
elif st.session_state.page == "signup":
    st.markdown("<h2>📝 Create Business Account</h2>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    email = st.text_input("📧 Business Email", key="signup_email")
    password = st.text_input("🔒 Password", type="password", key="signup_password")
    gst = st.text_input("🏢 GST Number", placeholder="Enter GSTIN", key="signup_gst")

    turnover = st.selectbox(
        "💰 Annual Turnover",
        ["1–5 Lakh", "5–10 Lakh", "10–15 Lakh", "15–50 Lakh", "50 Lakh+"],
        key="signup_turnover"
    )

    if st.button("🚀 Create Account", key="create_account_btn"):
        if not email or not password or not gst:
            st.error("Please fill all required fields")
        else:
            ok, message = signup(email, password, gst, turnover)
            if ok:
                st.success(f"✅ {message}")
                st.session_state.page = "login"
                st.rerun()
            else:
                st.error(message)

    st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# DASHBOARD
# ==============================
if st.session_state.page == "dashboard" and st.session_state.user:
    user = st.session_state.user

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🏢 Business Profile")
    col1, col2 = st.columns(2)
    col1.write(f"📧 Email: {user.get('email', 'N/A')}")
    col2.write(f"🏢 GST: {user.get('gst', 'N/A')}")
    col1.write(f"💰 Turnover: {user.get('turnover', 'N/A')}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<h1>📊 AI Intelligence Dashboard</h1>", unsafe_allow_html=True)

    product = st.session_state.get("product", "Wheat Flour")
    city = st.session_state.get("city", "Jaipur")
    month_name = st.session_state.get("month_name", "January")

    month = list(calendar.month_name).index(month_name)
    year = 2025

    holiday = get_holidays(year, month)
    temp = get_weather(city)
    viral = simulate_viral_score(product)
    price = fetch_product_price(product)

    st.subheader("📊 Market Intelligence")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"<div class='kpi'><h4>🌡 Temp</h4><h2>{temp:.1f}°C</h2></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi'><h4>📅 Holidays</h4><h2>{holiday}</h2></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi'><h4>🔥 Trend</h4><h2>{viral}</h2></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi'><h4>💰 Price</h4><h2>{price}</h2></div>", unsafe_allow_html=True)

    if st.button("🚀 Predict Demand", key="predict_btn_dashboard"):
        input_df = pd.DataFrame({
            "holiday_count": [holiday],
            "avg_temp": [temp],
            "viral_score": [viral]
        })

        pred = model.predict(input_df)[0]
        inventory = pred * 1.1

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📈 AI Forecast")
        c1, c2 = st.columns(2)
        c1.metric("📦 Sales", int(pred))
        c2.metric("📊 Inventory", int(inventory))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📊 Demand Analysis")
        chart_df = pd.DataFrame({
            "Type": ["Sales", "Inventory"],
            "Value": [pred, inventory]
        })
        st.bar_chart(chart_df.set_index("Type"))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🧠 AI Insights")
        if viral > 70:
            st.success("🔥 Trending product detected — high demand expected.")
        elif holiday > 6:
            st.info("📅 Seasonal demand increase expected.")
        else:
            st.warning("⚖️ Stable market demand.")
        st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([8, 1])
    with col2:
        if st.button("Logout"):
            st.session_state.user = None
            st.session_state.page = "welcome"
            st.rerun()
