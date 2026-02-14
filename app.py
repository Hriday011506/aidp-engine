import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(page_title="AIDP Engine", layout="wide")

st.title("🧠 AIDP Engine")
st.subheader("AI-Driven Inventory & Dynamic Pricing System")

# ----------------------------------------------------
# DATA LOADING
# ----------------------------------------------------
DATA_FOLDER = os.path.join("data", "processed")
DATA_FILE = "monthly_data.csv"
DATA_PATH = os.path.join(DATA_FOLDER, DATA_FILE)

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

# Debug section (helps during deployment)
if not os.path.exists(DATA_FOLDER):
    st.error(f"Folder not found: {DATA_FOLDER}")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error(f"File not found: {DATA_PATH}")
    st.write("Files inside processed folder:")
    st.write(os.listdir(DATA_FOLDER))
    st.stop()

df = load_data(DATA_PATH)

st.success("Dataset loaded successfully!")

# ----------------------------------------------------
# MODEL TRAINING
# ----------------------------------------------------
@st.cache_resource
def train_model(data):
    required_columns = ["holiday_count", "avg_temp", "viral_score", "monthly_sales"]

    for col in required_columns:
        if col not in data.columns:
            st.error(f"Missing required column: {col}")
            st.stop()

    X = data[["holiday_count", "avg_temp", "viral_score"]]
    y = data["monthly_sales"]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(X, y)
    return model

model = train_model(df)

st.success("Model trained successfully!")

# ----------------------------------------------------
# USER INPUT SECTION
# ----------------------------------------------------
st.markdown("## 📊 Business Input Parameters")

col1, col2, col3 = st.columns(3)

holiday_count = col1.slider("Number of Holidays in Month", 0, 15, 2)
avg_temp = col2.slider("Average Temperature (°C)", 15, 45, 30)
viral_score = col3.slider("Social Media Viral Score", 0, 100, 50)

# ----------------------------------------------------
# FORECAST BUTTON
# ----------------------------------------------------
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
    st.metric("Suggested Optimized Price", f"₹ {round(optimized_price, 2)}")

    # Visualization
    fig, ax = plt.subplots()
    ax.bar(["Monthly Forecast"], [monthly_prediction])
    ax.set_ylabel("Units")
    ax.set_title("Predicted Monthly Demand")
    st.pyplot(fig)

    st.success("AIDP Forecast Generated Successfully!")
