import streamlit as st
from src.predict import predict_demand
import matplotlib.pyplot as plt

st.set_page_config(page_title="AIDP Engine", layout="wide")

st.title("🧠 AIDP Engine")
st.subheader("AI-Driven Predictive Inventory & Dynamic Pricing")

st.markdown("""
This engine predicts monthly demand using:
- Weather signals
- Local event intensity
- Social media virality
""")

col1, col2, col3 = st.columns(3)

avg_temp = col1.slider("Average Temperature", 20, 45, 30)
local_events = col2.slider("Local Events", 0, 10, 3)
viral_score = col3.slider("Viral Score", 0, 100, 50)

if st.button("Run Forecast"):

    prediction = predict_demand(avg_temp, local_events, viral_score)

    reorder_qty = prediction * 1.1

    base_price = 100
    optimized_price = base_price * (1 + 0.01 * (prediction/1000))

    st.metric("Predicted Monthly Demand", round(prediction))
    st.metric("Recommended Inventory", round(reorder_qty))
    st.metric("Optimized Price", round(optimized_price,2))

    fig, ax = plt.subplots()
    ax.bar(["Predicted Demand"], [prediction])
    st.pyplot(fig)
