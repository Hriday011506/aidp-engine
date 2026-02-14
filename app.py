import streamlit as st
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt


# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="AIDP Engine",
    layout="wide"
)

st.title("🧠 AIDP Engine")
st.subheader("AI-Driven Inventory & Dynamic Pricing System")


# -------------------------------
# Load Trained Model
# -------------------------------
model_path = "models/demand_model.pkl"

if not os.path.exists(model_path):
    st.error("Model not found. Please run training first.")
    st.stop()

with open(model_path, "rb") as f:
    model = pickle.load(f)


# -------------------------------
# User Inputs
# -------------------------------
st.markdown("### 📊 Business Input Parameters")

col1, col2, col3, col4 = st.columns(4)

is_holiday = col1.selectbox("Is Holiday?", [0, 1])
avg_temp = col2.slider("Average Temperature (°C)", 15, 45, 30)
viral_score = col3.slider("Social Media Viral Score", 0, 100, 50)
local_events = col4.slider("Local Events Count", 0, 10, 2)


# -------------------------------
# Prediction Button
# -------------------------------
if st.button("🚀 Run Forecast"):

    # Prepare input
    input_data = np.array([[is_holiday, avg_temp, viral_score, local_events]])

    # Predict daily demand
    daily_prediction = model.predict(input_data)[0]

    # Monthly forecast (30 days)
    monthly_forecast = daily_prediction * 30

    # Inventory recommendation (10% safety buffer)
    reorder_qty = monthly_forecast * 1.1

    # Dynamic pricing logic
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

    # Visualization
    st.markdown("### 📊 Demand Visualization")

    fig, ax = plt.subplots()
    ax.bar(["Daily Forecast", "Monthly Forecast"],
           [daily_prediction, monthly_forecast])
    ax.set_ylabel("Units")
    st.pyplot(fig)

    st.success("AIDP Forecast Generated Successfully!")
