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

st.set_page_config(page_title="AIDP Engine", page_icon="🚀", layout="wide", initial_sidebar_state="collapsed")

# ==============================
# CONFIG / SECRETS
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
if "product" not in st.session_state:
    st.session_state.product = "Wheat Flour"
if "city" not in st.session_state:
    st.session_state.city = "Jaipur"
if "month_name" not in st.session_state:
    st.session_state.month_name = datetime.datetime.now().strftime("%B")

# ==============================
# PROFESSIONAL UI
# ==============================
st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(59,130,246,.12), transparent 28%),
        radial-gradient(circle at 90% 5%, rgba(168,85,247,.12), transparent 25%),
        linear-gradient(135deg, #020617 0%, #0f172a 55%, #020617 100%);
    color: #e2e8f0;
}
.block-container { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1400px; }
.hero {
    padding: 32px 34px;
    border-radius: 28px;
    background: linear-gradient(135deg, rgba(15,23,42,.96), rgba(30,41,59,.82));
    border: 1px solid rgba(96,165,250,.2);
    box-shadow: 0 20px 50px rgba(0,0,0,.28);
    margin-bottom: 24px;
}
.hero h1 { margin: 0; font-size: 3.2rem; letter-spacing: -1px; }
.hero p { margin: 8px 0 0; color: #94a3b8; font-size: 1.05rem; }
.badge {
    display:inline-block; padding:6px 12px; border-radius:999px;
    background: rgba(34,197,94,.12); color:#86efac;
    border:1px solid rgba(34,197,94,.25); font-size:.78rem; font-weight:700;
    margin-bottom: 12px;
}
.card {
    background: rgba(15,23,42,.72);
    padding: 24px;
    border-radius: 22px;
    border: 1px solid rgba(148,163,184,.12);
    box-shadow: 0 14px 35px rgba(0,0,0,.22);
    margin-bottom: 18px;
}
.kpi {
    background: linear-gradient(135deg, rgba(15,23,42,.95), rgba(30,41,59,.88));
    padding: 22px;
    border-radius: 20px;
    border: 1px solid rgba(96,165,250,.14);
    text-align: center;
    min-height: 125px;
}
.kpi-label { color:#94a3b8; font-size:.78rem; text-transform:uppercase; letter-spacing:1px; }
.kpi-value { color:#f8fafc; font-size:1.8rem; font-weight:800; margin-top:7px; }
.section-title { font-size:1.65rem; font-weight:800; margin: 10px 0 14px; }
.tip {
    padding: 14px 16px; border-radius: 14px;
    background: rgba(59,130,246,.10); border:1px solid rgba(59,130,246,.18);
    color:#bfdbfe; margin-top: 10px;
}
.stButton > button {
    border: 0 !important; border-radius: 12px !important;
    background: linear-gradient(135deg, #06b6d4, #3b82f6) !important;
    color: white !important; font-weight: 800 !important;
    box-shadow: 0 8px 24px rgba(59,130,246,.2);
}
.stButton > button:hover { transform: translateY(-1px); }
[data-testid="stMetricValue"] { color:#f8fafc; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ==============================
# AUTH FUNCTIONS
# ==============================
def signup(email, password, gst, turnover):
    if users_collection is None:
        return False, "Database unavailable. Configure MONGO_URI in Streamlit Secrets."
    try:
        existing = users_collection.find_one({"email": email})
        if existing:
            return False, "An account with this email already exists."
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        users_collection.insert_one({
            "email": email,
            "password": hashed,
            "gst": gst,
            "turnover": turnover,
            "created_at": datetime.datetime.now()
        })
        return True, "Account created successfully."
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
# DATA SERVICES
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


def get_weather(city):
    try:
        response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1}, timeout=10
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            return 25.0
        lat, lon = results[0]["latitude"], results[0]["longitude"]
        weather_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": "true"}, timeout=10
        )
        weather_response.raise_for_status()
        return float(weather_response.json()["current_weather"]["temperature"])
    except Exception:
        return 25.0


def get_holidays(year, month):
    india_holidays = holidays.India(years=year)
    total_days = calendar.monthrange(year, month)[1]
    return sum(
        1 for d in range(1, total_days + 1)
        if datetime.date(year, month, d).weekday() >= 5
        or datetime.date(year, month, d) in india_holidays
    )


def simulate_viral_score(product):
    seed = sum(ord(c) for c in product)
    rng = np.random.default_rng(seed)
    return int(rng.integers(30, 90))

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
    df["sales"] = 200 + df["holiday_count"] * 50 + df["avg_temp"] * 10 + df["viral_score"] * 5
    return df


@st.cache_resource
def train_model(df):
    model = RandomForestRegressor(n_estimators=250, max_depth=12, random_state=42)
    model.fit(df[["holiday_count", "avg_temp", "viral_score"]], df["sales"])
    return model

model = train_model(load_data())

# ==============================
# SIDEBAR NAVIGATION
# ==============================
with st.sidebar:
    st.markdown("### 🚀 AIDP Engine")
    st.caption("AI Demand Intelligence Platform")
    st.divider()

    if st.session_state.user:
        st.write(f"**{st.session_state.user.get('email', 'User')}**")
        if st.button("📊 Dashboard", use_container_width=True):
            st.session_state.page = "dashboard"
        if st.button("⚙️ Settings", use_container_width=True):
            st.session_state.page = "settings"
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.user = None
            st.session_state.page = "welcome"
            st.rerun()

# ==============================
# WELCOME
# ==============================
if st.session_state.page == "welcome":
    st.markdown("""
    <div class='hero'>
        <div class='badge'>● SYSTEM ONLINE</div>
        <h1>🚀 AIDP Engine</h1>
        <p>AI Demand Intelligence Platform — Predict demand, optimize inventory, and support smarter pricing decisions.</p>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.5, 1])
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🧠 What AIDP Does")
        st.markdown("""
        **Demand Intelligence** — combines market signals for demand estimation.

        **Inventory Optimization** — recommends a safety-buffer inventory level.

        **Market Intelligence** — surfaces weather, holiday and shopping-price signals.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🔐 Business Access")
        if st.button("Login", use_container_width=True):
            st.session_state.page = "login"
            st.rerun()
        if st.button("Create Business Account", use_container_width=True):
            st.session_state.page = "signup"
            st.rerun()
        st.markdown("<div class='tip'>Secure access with hashed passwords and MongoDB-backed user accounts.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# LOGIN
# ==============================
elif st.session_state.page == "login":
    st.markdown("<div class='hero'><div class='badge'>SECURE ACCESS</div><h1>🔐 Welcome Back</h1><p>Sign in to access your business intelligence dashboard.</p></div>", unsafe_allow_html=True)
    with st.form("login_form"):
        email = st.text_input("Business Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", use_container_width=True)
        if submitted:
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
    st.markdown("<div class='hero'><div class='badge'>BUSINESS ONBOARDING</div><h1>📝 Create Account</h1><p>Set up your business profile to use AIDP intelligence features.</p></div>", unsafe_allow_html=True)
    with st.form("signup_form"):
        email = st.text_input("Business Email")
        password = st.text_input("Password", type="password")
        gst = st.text_input("GST Number", placeholder="Enter GSTIN")
        turnover = st.selectbox("Annual Turnover", ["1–5 Lakh", "5–10 Lakh", "10–15 Lakh", "15–50 Lakh", "50 Lakh+"])
        submitted = st.form_submit_button("Create Account", use_container_width=True)
        if submitted:
            if not email or not password or not gst:
                st.error("Please fill all required fields.")
            else:
                ok, message = signup(email, password, gst, turnover)
                if ok:
                    st.success(message)
                    st.session_state.page = "login"
                    st.rerun()
                else:
                    st.error(message)

# ==============================
# DASHBOARD
# ==============================
elif st.session_state.page == "dashboard" and st.session_state.user:
    user = st.session_state.user
    st.markdown("""
    <div class='hero'>
        <div class='badge'>● AI INTELLIGENCE DASHBOARD</div>
        <h1>📊 AIDP Control Center</h1>
        <p>Real-time business signals, demand forecasting and inventory recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    pcol1, pcol2, pcol3 = st.columns([2, 1, 1])
    with pcol1:
        product = st.text_input("Product", value=st.session_state.product)
    with pcol2:
        city = st.text_input("City", value=st.session_state.city)
    with pcol3:
        month_name = st.selectbox("Forecast Month", list(calendar.month_name)[1:], index=list(calendar.month_name)[1:].index(st.session_state.month_name))

    st.session_state.product = product
    st.session_state.city = city
    st.session_state.month_name = month_name

    month = list(calendar.month_name).index(month_name)
    year = datetime.datetime.now().year

    holiday = get_holidays(year, month)
    temp = get_weather(city)
    viral = simulate_viral_score(product)
    price = fetch_product_price(product)

    st.markdown("### 🌐 Market Intelligence")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"<div class='kpi'><div class='kpi-label'>Temperature</div><div class='kpi-value'>{temp:.1f}°C</div></div>", unsafe_allow_html=True)
    with k2:
        st.markdown(f"<div class='kpi'><div class='kpi-label'>Holiday Days</div><div class='kpi-value'>{holiday}</div></div>", unsafe_allow_html=True)
    with k3:
        st.markdown(f"<div class='kpi'><div class='kpi-label'>Trend Score</div><div class='kpi-value'>{viral}</div></div>", unsafe_allow_html=True)
    with k4:
        st.markdown(f"<div class='kpi'><div class='kpi-label'>Market Price</div><div class='kpi-value'>{price}</div></div>", unsafe_allow_html=True)

    st.markdown("### 🤖 Forecast Engine")
    if st.button("🚀 Generate AI Forecast", use_container_width=True):
        input_df = pd.DataFrame({
            "holiday_count": [holiday],
            "avg_temp": [temp],
            "viral_score": [viral]
        })
        pred = float(model.predict(input_df)[0])
        inventory = float(pred * 1.10)
        price_num = None
        try:
            price_num = float(''.join(c for c in price.replace(',', '') if c.isdigit() or c == '.'))
        except Exception:
            price_num = None
        optimized_price = price_num * (1 + min(max((pred - 300) / 3000, -0.10), 0.10)) if price_num else None

        st.session_state.prediction = {
            "pred": pred, "inventory": inventory, "optimized_price": optimized_price
        }

    result = st.session_state.get("prediction")
    if result:
        pred = result["pred"]
        inventory = result["inventory"]
        optimized_price = result["optimized_price"]

        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted Demand", f"{int(pred)} units")
        r2.metric("Recommended Inventory", f"{int(inventory)} units", delta="+10% safety buffer")
        r3.metric("Suggested Price", f"₹ {optimized_price:,.2f}" if optimized_price else "Unavailable")

        left, right = st.columns([1.35, 1])
        with left:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 📈 Demand vs Inventory")
            chart_df = pd.DataFrame({"Type": ["Forecast Demand", "Recommended Inventory"], "Units": [pred, inventory]})
            st.bar_chart(chart_df.set_index("Type"))
            st.markdown("</div>", unsafe_allow_html=True)
        with right:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 🧠 AI Insights")
            if viral >= 70:
                st.success("High trend signal detected. Consider increasing stock coverage.")
            elif holiday >= 8:
                st.info("Heavy holiday period detected. Seasonal demand uplift may occur.")
            elif holiday >= 5:
                st.info("Moderate seasonal activity detected. Monitor stock closely.")
            else:
                st.warning("Market conditions currently indicate relatively stable demand.")
            if temp >= 35:
                st.write("🌡️ High temperature may influence category-level demand patterns.")
            st.write("📦 Recommended inventory includes a 10% safety buffer.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🔍 Decision Summary")
        s1, s2, s3 = st.columns(3)
        s1.write(f"**Product:** {product}")
        s2.write(f"**City:** {city}")
        s3.write(f"**Forecast Period:** {month_name} {year}")
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# SETTINGS
# ==============================
elif st.session_state.page == "settings" and st.session_state.user:
    st.markdown("<div class='hero'><div class='badge'>ACCOUNT</div><h1>⚙️ Business Settings</h1><p>Review your registered business information.</p></div>", unsafe_allow_html=True)
    user = st.session_state.user
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write(f"**Business Email:** {user.get('email', 'N/A')}")
    st.write(f"**GST Number:** {user.get('gst', 'N/A')}")
    st.write(f"**Annual Turnover:** {user.get('turnover', 'N/A')}")
    st.markdown("</div>", unsafe_allow_html=True)
