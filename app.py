# debug-friendly app.py — paste over your current file
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

# ---------------- MONGODB + AUTH (defensive)
try:
    from pymongo import MongoClient
    import bcrypt
except Exception:
    # If packages are missing, show a helpful error below; the rest will still run with in-memory users_collection.
    pass

# Replace with st.secrets["MONGO_URI"] in deployment
MONGO_URI = "mongodb+srv://hridaymahajan1979_db_user:hriday@aidp.jz72py2.mongodb.net/?retryWrites=true&w=majority"

db_ok = False

_in_memory_users = []  # fallback for testing if Mongo not available

if MONGO_URI:
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=4000)
        client.server_info()

        db = client["aidp_db"]
        users_collection = db["users"] 
        users = users_collection# 

        db_ok = True

    except Exception as e:
        db_ok = False
        users_collection = None
else:
    db_ok = False

# We will use bcrypt if available; otherwise store plain text only for quick local test (not for production)
try:
    import bcrypt as _bcrypt
    have_bcrypt = True
except Exception:
    have_bcrypt = False

# session initialization
if "user" not in st.session_state:
    st.session_state.user = None
if "signup_success" not in st.session_state:
    st.session_state.signup_success = False
if "menu_choice" not in st.session_state:
    st.session_state["menu_choice"] = "Signup"  # default

# helper functions using DB if available, else in-memory
def _db_find_user(email):
    if db_ok and users_collection is not None:
        return users_collection.find_one({"email": email})
    else:
        for u in _in_memory_users:
            if u["email"] == email:
                return u
        return None

def _db_insert_user(doc):
    global _in_memory_users
    _in_memory_users.append(doc)

def signup(email, password, gst, turnover):
    import bcrypt

    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

   user_doc = {
    "email": email.lower().strip(),
    "password": hashed_pw,
    "gst": gst,
    "turnover": turnover
 }

if db_ok and users_collection is not None:
    users_collection.insert_one(user_doc)
else:
    _db_insert_user(user_doc)
    _db_insert_user(user_doc)

def login(email, password):
    user = _db_find_user(email.lower().strip())

    if not user:
        return None

    stored_pw = user.get("password")

    if have_bcrypt and isinstance(stored_pw, (bytes, bytearray)):
        try:
            if _bcrypt.checkpw(password.encode(), stored_pw):
                return user
            else:
                return None
        except:
            return None
    else:
        if stored_pw == password:
            return user
        else:
            return None
# ---------------- GST API (mocked for demo)
def fetch_gst_details(gst):
    # demo / mock response - replace with real API call when you have API access
    # If gst looks like 'valid', we return a realistic name, else still return mock
    if gst and gst.strip():
        return {
            "company_name": f"Demo Company for {gst}",
            "gst_number": gst,
            "status": "Active"
        }
    return None

# ---------------- PAGE CONFIG
st.set_page_config(
    page_title="AIDP Engine – AI Forecasting System (debug)",
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

# ---------------- AUTH UI (robust)
menu = ["Login", "Signup"]

# use the session key for the selectbox so we can set it programmatically
choice = st.sidebar.selectbox("Menu", menu, key="menu_choice")

# ---------- SIGNUP FLOW
if choice == "Signup":
    st.subheader("📝 Create Account")

    email = st.text_input("Work Email", key="signup_email")
    password = st.text_input("Password", type="password", key="signup_password")
    gst = st.text_input("GST Number", key="signup_gst")

    if gst:
        company = fetch_gst_details(gst)
        if company:
            st.info(f"🏢 Company: {company.get('company_name', 'Not Found')}")

    turnover = st.number_input("Annual Turnover", key="signup_turnover")

    if st.button("Signup"):
        try:
            email_clean = email.lower().strip()

            if _db_find_user(email_clean):
                st.error("User already exists")
            else:
                signup(email_clean, password, gst, turnover)

                st.success("✅ Account created successfully!")

                st.session_state.signup_success = True
                st.session_state["menu_choice"] = "Login"

                st.rerun()

        except Exception as e:
            st.error(f"Error: {e}")

# ---------- LOGIN FLOW
elif choice == "Login":
    st.subheader("🔐 Login")

    email_login = st.text_input("Email", key="login_email")
    password_login = st.text_input("Password", type="password", key="login_password")

    if st.button("Login"):
        st.write("Login clicked")
        user = login(email_login, password_login)
        if user:
            st.session_state.user = user
            # reset signup_success so the menu behaves normally next time
            st.session_state.signup_success = False
            # keep menu_choice on Login
            st.session_state["menu_choice"] = "Login"
            st.success("Logged in!")
        else:
            st.error("Invalid credentials")

# ---------------- LOGOUT
if st.session_state.user:
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.session_state["menu_choice"] = "Signup"
        st.success("Logged out")

# ---------------- PROTECTED DASHBOARD
if st.session_state.user:

    st.title("📊 AI Retail Demand Dashboard")
    st.caption(f"Welcome {st.session_state.user['email']}")

    st.subheader("🏢 Company Info")
    gst = st.session_state.user.get("gst")
    company = fetch_gst_details(gst)
    if company:
        st.success("GST Verified")
        st.write(company)
    else:
        st.warning("GST data not available")

    st.divider()

    # --- (original ML & UI code unchanged) ---
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
