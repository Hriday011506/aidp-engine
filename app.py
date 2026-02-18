import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import calendar
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="AIDP Engine – AI Forecasting System",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<h1 style='text-align: center; color: #1f4e79;'>AIDP Engine 📊</h1>
<h4 style='text-align: center; color: gray;'>
AI-Driven Sales Forecasting & Inventory Optimization System
</h4>
""", unsafe_allow_html=True)

st.divider()

# ---------------- WEATHER (Open-Meteo)
def get_weather(city):
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        geo = requests.get(geo_url, timeout=5).json()

        if "results" not in geo:
            return None, None

        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]

        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather = requests.get(weather_url, timeout=5).json()

        temp = weather["current_weather"]["temperature"]
        weather_code = weather["current_weather"]["weathercode"]

        return temp, weather_code
    except:
        return None, None


# ---------------- HOLIDAYS (Public + Weekends)
def get_holidays(year, month):
    try:
        url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/IN"
        response = requests.get(url, timeout=5)

        if response.status_code != 200:
            return 0

        public_holidays = response.json()
        holiday_dates = []

        for h in public_holidays:
            date_obj = datetime.datetime.strptime(h["date"], "%Y-%m-%d")
            if date_obj.month == month:
                holiday_dates.append(date_obj.date())

        total_days = calendar.monthrange(year, month)[1]
        weekend_count = 0

        for day in range(1, total_days + 1):
            date = datetime.date(year, month, day)
            if date.weekday() >= 5:
                weekend_count += 1

        return len(holiday_dates) + weekend_count

    except:
        return 0


# ---------------- DATA + MODEL
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
model = train_model(data)

# ---------------- INPUT SECTION
st.subheader("📥 Enter Business Parameters")

c1, c2, c3 = st.columns(3)

with c1:
    product_name = st.text_input("Enter Product Name", "Wheat Flour")

    year = st.selectbox("Select Year", [2025, 2026, 2027])

    month_names = list(calendar.month_name)[1:]
    selected_month_name = st.selectbox("Select Month", month_names)
    month = month_names.index(selected_month_name) + 1

    holiday_count = get_holidays(year, month)
    st.success(f"📅 Total Non-Working Days: {holiday_count}")

with c2:
    city = st.text_input("Enter City", "Jaipur")
    avg_temp, weather_code = get_weather(city)

    if avg_temp is not None:
        st.success(f"🌡 Temperature: {avg_temp:.2f} °C")
    else:
        avg_temp = 25
        weather_code = None
        st.warning("Using default 25°C")

with c3:
    viral_score = st.slider("Social Media Viral Score", 0, 100, 50)

st.divider()

# ---------------- FORECAST
if st.button("🚀 Generate Forecast"):

    input_df = pd.DataFrame({
        "holiday_count": [holiday_count],
        "avg_temp": [avg_temp],
        "viral_score": [viral_score]
    })

    predicted_sales = model.predict(input_df)[0]

    # ---------------- SMART PRODUCT + WEATHER LOGIC
    product = product_name.lower()
    reason = "Standard demand pattern."

    moisture_sensitive = ["flour", "wheat", "rice", "grains", "powder"]
    perishable = ["milk", "bread", "vegetable", "fruit"]
    winter_products = ["jacket", "coat", "heater"]
    summer_products = ["cooler", "fan", "ice cream"]

    is_rainy = weather_code is not None and 50 <= weather_code <= 67

    if any(word in product for word in moisture_sensitive):
        if is_rainy:
            predicted_sales *= 0.85
            reason = "Rainy weather increases moisture risk. Product prone to spoilage/ants. Maintain lower inventory."
        else:
            reason = "Moisture-sensitive product. Avoid excessive storage."

    elif any(word in product for word in perishable):
        predicted_sales *= 0.9
        reason = "Perishable item. Lower buffer stock recommended."

    elif any(word in product for word in winter_products):
        if avg_temp < 20:
            predicted_sales *= 1.2
            reason = "Cold weather detected. Higher seasonal demand expected."
        else:
            reason = "Winter product but temperature not low."

    elif any(word in product for word in summer_products):
        if avg_temp > 30:
            predicted_sales *= 1.2
            reason = "High temperature detected. Increased summer demand."
        else:
            reason = "Summer product but temperature moderate."

    recommended_inventory = predicted_sales * 1.10
    optimized_price = 500 + (predicted_sales * 0.02)

    st.subheader("📊 Forecast Results")

    m1, m2, m3 = st.columns(3)

    m1.metric("Predicted Sales", f"{int(predicted_sales)} Units")
    m2.metric("Recommended Inventory", f"{int(recommended_inventory)} Units")
    m3.metric("Optimized Price", f"₹{int(optimized_price)}")

    st.info(f"📌 Inventory Insight: {reason}")

    # ---------------- IMPROVED GRAPH
    fig, ax = plt.subplots()
    ax.bar(
        ["Predicted Sales", "Recommended Inventory"],
        [predicted_sales, recommended_inventory]
    )
    ax.set_ylabel("Units")
    ax.set_title("Sales vs Inventory Comparison")
    st.pyplot(fig)

st.divider()
st.markdown(
    "<center><small>Developed by Hriday Mahajan | Machine Learning Project | 2026</small></center>",
    unsafe_allow_html=True
)
