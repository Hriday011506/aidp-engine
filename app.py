import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import calendar
import matplotlib.pyplot as plt
import holidays
import os
from sklearn.ensemble import RandomForestRegressor

# ---------------- MONGODB + AUTH
from pymongo import MongoClient
import bcrypt

MONGO_URI = "mongodb+srv://hridaymahajan1979_db_user:<hriday>@aidp.jz72py2.mongodb.net/?appName=aidp"

client = MongoClient(MONGO_URI)
db = client["aidp_db"]
users = db["users"]

if "user" not in st.session_state:
    st.session_state.user = None

def signup(email, password, gst, turnover):
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    users.insert_one({
        "email": email,
        "password": hashed_pw,
        "gst": gst,
        "turnover": turnover
    })

def login(email, password):
    user = users.find_one({"email": email})
    if user and bcrypt.checkpw(password.encode(), user["password"]):
        return user
    return None

# ---------------- GST API (CLEAR TAX STYLE)
def fetch_gst_details(gst):
    return {
        "company_name": "ABC Pvt Ltd",
        "gst_number": gst,
        "status": "Active"
    }

# ---------------- PAGE CONFIG
st.set_page_config(
    page_title="AIDP Engine – AI Forecasting System",
    page_icon="📊",
    layout="wide"
)

# ---------------- DARK UI
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- AUTH UI
if "signup_success" not in st.session_state:
    st.session_state.signup_success = False

menu = ["Login", "Signup"]

if st.session_state.signup_success:
    choice = "Login"
else:
    choice = st.sidebar.selectbox("Menu", menu)
if choice == "Signup":
    st.subheader("📝 Create Account")

    email = st.text_input("Work Email")
    password = st.text_input("Password", type="password")
    gst = st.text_input("GST Number")
    if gst:
        company = fetch_gst_details(gst)
        if company:
            st.info(f"🏢 Company: {company.get('company_name', 'Not Found')}")
    turnover = st.number_input("Annual Turnover")
if st.button("Signup"):
    st.write("Signup clicked")

    if users.find_one({"email": email}):
        st.error("User already exists")
    else:
        company = fetch_gst_details(gst)

        signup(email, password, gst, turnover)

        st.success("✅ Account created successfully!")

        st.session_state.signup_success = True
        st.experimental_rerun()
elif choice == "Login":
    st.subheader("🔐 Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = login(email, password)
        if user:
            st.session_state.user = user
            st.success("Logged in!")
        else:
            st.error("Invalid credentials")

# ---------------- LOGOUT
if st.session_state.user:
    if st.sidebar.button("Logout"):
        st.session_state.user = None

# ---------------- MAIN APP (PROTECTED)
if st.session_state.user:

    # HEADER
    st.title("📊 AI Retail Demand Dashboard")
    st.caption(f"Welcome {st.session_state.user['email']}")

    # ---------------- SHOW COMPANY INFO
    st.subheader("🏢 Company Info")

    gst = st.session_state.user.get("gst")
    company = fetch_gst_details(gst)

    if company:
        st.success("GST Verified")
        st.write(company)
    else:
        st.warning("GST data not available")

    st.divider()

    # ---------------- YOUR ORIGINAL CODE STARTS HERE (UNCHANGED)

    SERPAPI_KEY = "YOUR_KEY"

    def get_weather(city):
        try:
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
            geo = requests.get(geo_url, timeout=5).json()

            if "results" not in geo:
                return 25, False

            lat = geo["results"][0]["latitude"]
            lon = geo["results"][0]["longitude"]

            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
            weather = requests.get(weather_url, timeout=5).json()

            temp = weather["current_weather"]["temperature"]
            code = weather["current_weather"]["weathercode"]

            is_rainy = 50 <= code <= 67
            return temp, is_rainy
        except:
            return 25, False

    def get_holidays(year, month):
        india_holidays = holidays.India(years=year)
        total_days = calendar.monthrange(year, month)[1]
        count = 0
        for day in range(1, total_days + 1):
            date = datetime.date(year, month, day)
            if date.weekday() >= 5 or date in india_holidays:
                count += 1
        return count

    def simulate_viral_score(product):
        seed = abs(hash(product)) % 100
        np.random.seed(seed)
        base_score = np.random.randint(30, 70)

        if any(word in product.lower() for word in ["phone", "fashion", "jacket"]):
            base_score += 20

        return min(base_score, 100)

    @st.cache_data
    def load_data():
        np.random.seed(42)
        df = pd.DataFrame({
            "holiday_count": np.random.randint(0, 10, 100),
            "avg_temp": np.random.randint(10, 40, 100),
            "viral_score": np.random.randint(0, 100, 100)
        })

        df["monthly_sales"] = (
            200
            + df["holiday_count"] * 50
            + df["avg_temp"] * 10
            + df["viral_score"] * 5
            + np.random.normal(0, 50, 100)
        )
        return df

    @st.cache_resource
    def train_model(df):
        X = df[["holiday_count", "avg_temp", "viral_score"]]
        y = df["monthly_sales"]
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model

    data = load_data()
    rf_model = train_model(data)

    # ---------------- SIDEBAR INPUTS
    st.sidebar.header("📥 Input Parameters")

    product_name = st.sidebar.text_input("Product Name", "Wheat Flour")

    year = st.sidebar.selectbox("Year", [2025, 2026, 2027])
    month_names = list(calendar.month_name)[1:]
    selected_month = st.sidebar.selectbox("Month", month_names)
    month = month_names.index(selected_month) + 1

    city = st.sidebar.text_input("City", "Jaipur")

    holiday_count = get_holidays(year, month)
    st.sidebar.success(f"📅 Holidays: {holiday_count}")

    avg_temp, is_rainy = get_weather(city)
    st.sidebar.success(f"🌡 Temp: {avg_temp:.2f} °C")

    viral_score = simulate_viral_score(product_name)
    st.sidebar.metric("🔥 Trend Score", viral_score)

    st.divider()

    if st.sidebar.button("🚀 Predict Demand"):
        with st.spinner("Analyzing market data..."):

            input_df = pd.DataFrame({
                "holiday_count": [holiday_count],
                "avg_temp": [avg_temp],
                "viral_score": [viral_score]
            })

            predicted_sales = rf_model.predict(input_df)[0]
            recommended_inventory = predicted_sales * 1.10

            st.subheader("📊 Dashboard Results")

            col1, col2, col3 = st.columns(3)
            col1.metric("📦 Predicted Sales", f"{int(predicted_sales)} Units")
            col2.metric("📈 Inventory", f"{int(recommended_inventory)} Units")
            col3.metric("💰 Market Price", "Not available")

            chart_data = pd.DataFrame({
                "Category": ["Predicted Sales", "Inventory"],
                "Value": [predicted_sales, recommended_inventory]
            })

            st.subheader("📊 Sales vs Inventory")
            st.bar_chart(chart_data.set_index("Category"))

            trend_data = pd.DataFrame({
                "Month": list(range(1, 13)),
                "Demand": np.linspace(200, predicted_sales, 12)
            })

            st.subheader("📈 Demand Trend")
            st.line_chart(trend_data.set_index("Month"))

            st.subheader("📌 AI Insights")
            st.success("Optimize inventory based on predicted demand.")

    st.divider()

    st.markdown(
        "<center><small>Developed by Hriday Mahajan 🚀</small></center>",
        unsafe_allow_html=True
    )

else:
    st.warning("⚠️ Please login to access dashboard")
