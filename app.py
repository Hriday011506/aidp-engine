import streamlit as st
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="AIDP Engine",
    layout="wide"
)

st.title("🧠 AIDP Engine")
st.subheader("AI-Driven Inventory & Dynamic Pricing System")

# ---------------------------------
# Load Monthly Dataset Safely
# ---------------------------------
@st.cache_data
def load_data():
    path = os.path.join("data", "processed", "monthly_data.csv")
    return pd.read_csv(path)

try:
    df = load_data()
except Exception as e:
    st.error("Dataset not found. Make sure monthly_data.csv is uploaded to data/processed/")
    st.stop()

# ---------------------------------
# Train Model (Runs Once)
# ---------------------------------
@st.cache_resource
def train_model(data):

    features = ["holiday_count", "avg_temp", "viral_score"]
    target = "monthly_sales"

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

st.success("Model trained successfully!")

# ---------------------------------
# User Inputs
# ---------------------------------
st.markdown("### 📊 Business Input Parameters")

col1, col2, col3 = st.columns(3)

holiday_count = col1.slider("Number of Holidays in Month", 0, 10, 2)
avg_temp = col2.slider("Average Temperature (°C)", 15, 45, 30)
viral_score = col3.slider("Social Media Viral Score", 0, 100, 50)

# ---------------------------------
# Forecast Button
# ---------------------------------
if st.button("🚀 Run Forecast"):

    input_data = np.array([[holiday_count, avg_temp, viral_score]])

    monthly_prediction = model.predict(input_data)[0]

    reorder_qty = monthly_prediction * 1.1

    base_price = 100
    elasticity = 0.01
    optimized_price = base_price * (1 + elasticity * (monthly_prediction / 100000))

    st.markdown("---")
    st.markdown("## 📈 Forecast Results")

    colA, colB = st.columns(2)

    colA.metric("Predicted Monthly Demand", f"{int(monthly_prediction)} units")
    colB.metric("Recommended Reorder Quantity", f"{int(reorder_qty)} units")

    st.markdown("### 💰 Pricing Recommendation")
    st.metric("Suggested Optimized Price", f"₹ {round(optimized_price,2)}")

    # Visualization
    fig, ax = plt.subplots()
    ax.bar(["Monthly Forecast"], [monthly_prediction])
    ax.set_ylabel("Units")
    st.pyplot(fig)

    st.success("AIDP Monthly Forecast Generated Successfully!")
