import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import calendar
import matplotlib.pyplot as plt
import holidays
import google.generativeai as genai
import os
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="AIDP Engine – AI Forecasting System",
    page_icon="📊",
    layout="wide"
)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

st.markdown("""
<h1 style='text-align: center; color: #1f4e79;'>AIDP Engine 📊</h1>
<h4 style='text-align: center; color: gray;'>
AI-Driven Sales Forecasting & Inventory Optimization System
</h4>
""", unsafe_allow_html=True)

st.divider()

# ---------------- WEATHER
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


# ---------------- HOLIDAYS (Offline Reliable)
def get_holidays(year, month):
    india_holidays = holidays.India(years=year)
    total_days = calendar.monthrange(year, month)[1]
    count = 0

    for day in range(1, total_days + 1):
        date = datetime.date(year, month, day)
        if date.weekday() >= 5 or date in india_holidays:
            count += 1

    return count


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
rf_model = train_model(data)


# ---------------- AI REASONING
def generate_inventory_reason(product, temp, rainy, holidays_count, viral):
    rain_status = "rainy" if rainy else "not rainy"

    prompt = f"""
    You are an inventory management expert.

    Product: {product}
    Temperature: {temp}°C
    Weather condition: {rain_status}
    Non-working days this month: {holidays_count}
    Social media trend score: {viral}

    Based on environmental risks, seasonal demand, and trend signals,
    explain clearly whether inventory should be increased or decreased.
    Give short professional reasoning.
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except:
        return "AI reasoning currently unavailable."


# ---------------- INPUT SECTION
st.subheader("📥 Enter Business Parameters")

c1, c2, c3 = st.columns(3)

with c1:
    product_name = st.text_input("Enter Product Name", "Wheat Flour")
    year = st.selectbox("Select Year", [2025, 2026, 2027])

    month_names = list(calendar.month_name)[1:]
    selected_month = st.selectbox("Select Month", month_names)
    month = month_names.index(selected_month) + 1

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

    predicted_sales = rf_model.predict(input_df)[0]

    is_rainy = weather_code is not None and 50 <= weather_code <= 67

    recommended_inventory = predicted_sales * 1.10
    optimized_price = 500 + (predicted_sales * 0.02)

    reason = generate_inventory_reason(
        product_name,
        avg_temp,
        is_rainy,
        holiday_count,
        viral_score
    )

    st.subheader("📊 Forecast Results")

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Sales", f"{int(predicted_sales)} Units")
    m2.metric("Recommended Inventory", f"{int(recommended_inventory)} Units")
    m3.metric("Optimized Price", f"₹{int(optimized_price)}")

    st.info(f"📌 Inventory Insight:\n\n{reason}")

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
