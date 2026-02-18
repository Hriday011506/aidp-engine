import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import calendar
import matplotlib.pyplot as plt
import holidays
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

# ---------------- WEATHER (SAFE)
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


# ---------------- HOLIDAYS (OFFLINE RELIABLE)
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
    avg_temp, is_rainy = get_weather(city)
    st.success(f"🌡 Temperature: {avg_temp:.2f} °C")

with c3:
    viral_score = st.slider("Social Media Viral Score", 0, 100, 50)

st.divider()

# ---------------- FORECAST BUTTON
if st.button("🚀 Generate Forecast"):

    input_df = pd.DataFrame({
        "holiday_count": [holiday_count],
        "avg_temp": [avg_temp],
        "viral_score": [viral_score]
    })

    predicted_sales = rf_model.predict(input_df)[0]

    # ---------------- SMART REASONING (DYNAMIC RULE BASED)
    product = product_name.lower()
    reason = "Standard demand pattern."

    if "flour" in product or "wheat" in product:
        if is_rainy:
            predicted_sales *= 0.85
            reason = "Rainy weather increases moisture risk. Wheat flour is prone to spoilage and ants. Maintain lower inventory."
        else:
            reason = "Dry product. Maintain moderate inventory levels."

    elif "milk" in product or "bread" in product:
        predicted_sales *= 0.9
        reason = "Perishable item. Lower inventory reduces spoilage risk."

    elif avg_temp > 32:
        predicted_sales *= 1.15
        reason = "High temperature detected. Seasonal demand likely higher."

    elif viral_score > 70:
        predicted_sales *= 1.2
        reason = "Strong social media trend suggests higher market demand."

    recommended_inventory = predicted_sales * 1.10
    optimized_price = 500 + (predicted_sales * 0.02)

    # ---------------- OUTPUT
    st.subheader("📊 Forecast Results")

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Sales", f"{int(predicted_sales)} Units")
    m2.metric("Recommended Inventory", f"{int(recommended_inventory)} Units")
    m3.metric("Optimized Price", f"₹{int(optimized_price)}")

    st.info(f"📌 Inventory Insight:\n\n{reason}")

    # ---------------- GRAPH
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
