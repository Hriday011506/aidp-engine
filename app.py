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

st.markdown("""
<style>
.big-title { font-size:40px; font-weight:800; }
.metric-box { padding:20px; border-radius:15px; background:#f0f2f6; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🧠 AIDP Engine</div>', unsafe_allow_html=True)
st.subheader("AI-Driven Inventory & Dynamic Pricing System")

# ----------------------------------------------------
# DATA LOADING WITH FALLBACK
# ----------------------------------------------------
DATA_PATH = os.path.join("data", "processed", "monthly_data.csv")

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        try:
            return pd.read_csv(DATA_PATH)
        except:
            st.warning("CSV corrupted. Using synthetic dataset instead.")
    else:
        st.warning("Dataset not found. Using synthetic dataset instead.")

    # Synthetic fallback dataset
    np.random.seed(42)
    months = 60
    df = pd.DataFrame({
        "holiday_count": np.random.randint(0, 5, months),
        "avg_temp": np.random.randint(20, 40, months),
        "viral_score": np.random.randint(10, 100, months),
    })

    df["monthly_sales"] = (
        50000 +
        df["holiday_count"] * 2000 +
        df["avg_temp"] * 300 +
        df["viral_score"] * 150 +
        np.random.randint(-5000, 5000, months)
    )

    return df

df = load_data()

st.success("Dataset Ready!")

# ----------------------------------------------------
# MODEL TRAINING
# ----------------------------------------------------
@st.cache_resource
def train_model(data):

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

st.success("Model Trained Successfully!")

# ----------------------------------------------------
# USER INPUTS
# ----------------------------------------------------
st.markdown("## 📊 Business Input Parameters")

col1, col2, col3 = st.columns(3)

holiday_count = col1.slider("Number of Holidays in Month", 0, 10, 2)
avg_temp = col2.slider("Average Temperature (°C)", 15, 45, 30)
viral_score = col3.slider("Social Media Viral Score", 0, 100, 50)

# ----------------------------------------------------
# FORECAST
# ----------------------------------------------------
if st.button("🚀 Generate Forecast"):

    input_data = np.array([[holiday_count, avg_temp, viral_score]])
    monthly_prediction = model.predict(input_data)[0]

    reorder_qty = monthly_prediction * 1.1
    optimized_price = 100 * (1 + 0.01 * (monthly_prediction / 100000))

    st.markdown("---")
    st.markdown("## 📈 Forecast Results")

    colA, colB, colC = st.columns(3)

    colA.metric("Predicted Monthly Demand", f"{int(monthly_prediction)} units")
    colB.metric("Recommended Inventory", f"{int(reorder_qty)} units")
    colC.metric("Optimized Price", f"₹ {round(optimized_price, 2)}")

    fig, ax = plt.subplots()
    ax.bar(["Monthly Forecast"], [monthly_prediction])
    ax.set_ylabel("Units")
    ax.set_title("Predicted Monthly Demand")
    st.pyplot(fig)

    st.success("AIDP Forecast Generated Successfully!")
