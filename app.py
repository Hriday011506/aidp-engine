import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# -----------------------------
# PAGE CONFIG (MUST BE FIRST)
# -----------------------------
st.set_page_config(
    page_title="AIDP Engine – AI Forecasting System",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# HEADER SECTION
# -----------------------------
st.markdown("""
<h1 style='text-align: center; color: #1f4e79;'>
AIDP Engine 📊
</h1>
<h4 style='text-align: center; text-align: center; color: gray;'>
AI-Driven Sales Forecasting & Inventory Optimization System
</h4>
""", unsafe_allow_html=True)

st.divider()

# -----------------------------
# LOAD / CREATE DATA
# -----------------------------
@st.cache_data
def load_data():
    # Creating synthetic dataset
    np.random.seed(42)
    data = pd.DataFrame({
        "holiday_count": np.random.randint(0, 10, 100),
        "avg_temp": np.random.randint(10, 40, 100),
        "viral_score": np.random.randint(0, 100, 100)
    })

    # Sales formula (simulated real-world relationship)
    data["monthly_sales"] = (
        200
        + data["holiday_count"] * 50
        + data["avg_temp"] * 10
        + data["viral_score"] * 5
        + np.random.normal(0, 50, 100)
    )

    return data


# -----------------------------
# TRAIN MODEL
# -----------------------------
@st.cache_resource
def train_model(data):
    X = data[["holiday_count", "avg_temp", "viral_score"]]
    y = data["monthly_sales"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model


data = load_data()
model = train_model(data)

# -----------------------------
# USER INPUT SECTION
# -----------------------------
st.subheader("📥 Enter Business Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    holiday_count = st.slider("Number of Holidays", 0, 15, 5)

with col2:
    avg_temp = st.slider("Average Temperature (°C)", 0, 50, 25)

with col3:
    viral_score = st.slider("Social Media Viral Score", 0, 100, 50)

st.divider()

# -----------------------------
# FORECAST BUTTON
# -----------------------------
if st.button("🚀 Generate Forecast"):

    input_data = pd.DataFrame({
        "holiday_count": [holiday_count],
        "avg_temp": [avg_temp],
        "viral_score": [viral_score]
    })

    predicted_sales = model.predict(input_data)[0]

    # Business Logic
    recommended_inventory = predicted_sales * 1.10   # 10% buffer
    optimized_price = 500 + (predicted_sales * 0.02)

    st.subheader("📊 Forecast Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📈 Predicted Monthly Sales", f"{int(predicted_sales)} Units")

    with col2:
        st.metric("📦 Recommended Inventory", f"{int(recommended_inventory)} Units")

    with col3:
        st.metric("💰 Optimized Price", f"₹{int(optimized_price)}")

    st.divider()

    # Bar Chart
    chart_data = pd.DataFrame({
        "Category": ["Predicted Sales", "Recommended Inventory"],
        "Units": [predicted_sales, recommended_inventory]
    })

    st.bar_chart(chart_data.set_index("Category"))

# -----------------------------
# FOOTER
# -----------------------------
st.divider()
st.markdown(
    "<center><small>Developed by Hriday Mahajan | Machine Learning Project | 2026</small></center>",
    unsafe_allow_html=True
)
