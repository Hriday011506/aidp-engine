import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

st.set_page_config(page_title="AIDP Engine", layout="wide")

st.title("🧠 AIDP Engine")
st.subheader("AI-Driven Inventory & Dynamic Pricing System")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/daily_data.csv")

df = load_data()

# -------------------------------
# Train Model Automatically
# -------------------------------
@st.cache_resource
def train_model(data):

    features = ["is_holiday", "avg_temp", "viral_score", "local_events"]
    target = "daily_sales"

    X = data[features]
    y = data[target]

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        random_state=42
    )

    model.fit(X, y)
    return model

model = train_model(df)

st.success("Model loaded successfully!")

# -------------------------------
# User Inputs
# -------------------------------
st.markdown("### 📊 Business Input Parameters")

col1, col2, col3, col4 = st.columns(4)

is_holiday = col1.selectbox("Is Holiday?", [0, 1])
avg_temp = col2.slider("Average Temperature (°C)", 15, 45, 30)
viral_score = col3.slider("Social Media Viral Score", 0, 100, 50)
local_events = col4.slider("Local Events Count", 0, 10, 2)

if st.button("🚀 Run Forecast"):

    input_data = np.array([[is_holiday, avg_temp, viral_score, local_events]])

    daily_prediction = model.predict(input_data)[0]
    monthly_forecast = daily_prediction * 30
    reorder_qty = monthly_forecast * 1.1

    base_price = 100
    elasticity = 0.01
    optimized_price = base_price * (1 + elasticity * (monthly_forecast / 10000))

    st.markdown("---")
    st.markdown("## 📈 Forecast Results")

    colA, colB, colC = st.columns(3)

    colA.metric("Predicted Daily Demand", f"{int(daily_prediction)} units")
    colB.metric("Predicted Monthly Demand", f"{int(monthly_forecast)} units")
    colC.metric("Recommended Reorder Quantity", f"{int(reorder_qty)} units")

    st.markdown("### 💰 Pricing Recommendation")
    st.metric("Suggested Optimized Price", f"₹ {round(optimized_price,2)}")

    fig, ax = plt.subplots()
    ax.bar(["Daily Forecast", "Monthly Forecast"],
           [daily_prediction, monthly_forecast])
    ax.set_ylabel("Units")
    st.pyplot(fig)

    st.success("AIDP Forecast Generated Successfully!")
