import streamlit as st
import pandas as pd
import numpy as np
import requests
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

def get_weather(city):
    API_KEY = "a4e1c2d537eb57d0cab44b215e91bfae"
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        if response.status_code == 200:
            return data["main"]["temp"]
        else:
            return None
    except:
        return None

@st.cache_data
def load_data():
    np.random.seed(42)
    data = pd.DataFrame({
        "holiday_count": np.random.randint(0, 10, 100),
        "avg_temp": np.random.randint(10, 40, 100),
        "viral_score": np.random.randint(0, 100, 100)
    })
    data["monthly_sales"] = (
        200
        + data["holiday_count"] * 50
        + data["avg_temp"] * 10
        + data["viral_score"] * 5
        + np.random.normal(0, 50, 100)
    )
    return data

@st.cache_resource
def train_model(data):
    X = data[["holiday_count", "avg_temp", "viral_score"]]
    y = data["monthly_sales"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

data = load_data()
model = train_model(data)

st.subheader("📥 Enter Business Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    holiday_count = st.slider("Number of Holidays", 0, 15, 5)

with col2:
    city = st.text_input("Enter City for Live Weather", "Jaipur")
    avg_temp = get_weather(city)
    if avg_temp is not None:
        st.success(f"🌡 Live Temperature in {city}: {avg_temp:.2f} °C")
    else:
        st.warning("Could not fetch weather data. Using default 25°C.")
        avg_temp = 25

with col3:
    viral_score = st.slider("Social Media Viral Score", 0, 100, 50)

st.divider()

if st.button("🚀 Generate Forecast"):

    input_data = pd.DataFrame({
        "holiday_count": [holiday_count],
        "avg_temp": [avg_temp],
        "viral_score": [viral_score]
    })

    predicted_sales = model.predict(input_data)[0]
    recommended_inventory = predicted_sales * 1.10
    optimized_price = 500 + (predicted_sales * 0.02)

    st.subheader("📊 Forecast Results")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("📈 Predicted Monthly Sales", f"{int(predicted_sales)} Units")

    with c2:
        st.metric("📦 Recommended Inventory", f"{int(recommended_inventory)} Units")

    with c3:
        st.metric("💰 Optimized Price", f"₹{int(optimized_price)}")

    st.divider()

    chart_data = pd.DataFrame({
        "Category": ["Predicted Sales", "Recommended Inventory"],
        "Units": [predicted_sales, recommended_inventory]
    })

    st.bar_chart(chart_data.set_index("Category"))

st.divider()
st.markdown(
    "<center><small>Developed by Hriday Mahajan | Machine Learning Project | 2026</small></center>",
    unsafe_allow_html=True
)
