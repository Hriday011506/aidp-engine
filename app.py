import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import calendar
import matplotlib.pyplot as plt
import holidays
import os
from sklearn.ensemble import RandomForestRegressor

HF_API_KEY = os.getenv("HF_API_KEY", "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")
HF_MODEL_URL = "https://router.huggingface.co/hf-inference/models/google/flan-t5-large"

st.set_page_config(
    page_title="AIDP Engine – AI Forecasting System",
    page_icon="📊",
    layout="wide"
)


# ---------------- SERPAPI SETUP
SERPAPI_KEY = "bbc8aca8053bbe60b9c7017e236f71656667f6b4d2bbf3b2da695084ad8766b4"
# ---------------- HEADER
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


# ---------------- HOLIDAYS
def get_holidays(year, month):
    india_holidays = holidays.India(years=year)
    total_days = calendar.monthrange(year, month)[1]
    count = 0

    for day in range(1, total_days + 1):
        date = datetime.date(year, month, day)
        if date.weekday() >= 5 or date in india_holidays:
            count += 1

    return count


# ---------------- VIRAL SCORE (STABLE)
def simulate_viral_score(product):
    seed = abs(hash(product)) % 100
    np.random.seed(seed)
    base_score = np.random.randint(30, 70)

    trending_keywords = ["phone", "iphone", "fashion", "jacket", "sneaker"]
    if any(word in product.lower() for word in trending_keywords):
        base_score += 20

    return min(base_score, 100)


# ---------------- GOOGLE PRICE FETCH (SERPAPI)
def fetch_price(product):
    try:
        params = {
            "engine": "google_shopping",
            "q": product,
            "api_key": SERPAPI_KEY,
            "gl": "in",
            "hl": "en"
        }

        response = requests.get("https://serpapi.com/search", params=params, timeout=5)
        data = response.json()

        if "shopping_results" in data and len(data["shopping_results"]) > 0:
            return data["shopping_results"][0].get("price")
        else:
            return None
    except:
        return None
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


# ---------------- HUGGING FACE REASONING
def generate_reason(product, temp, rainy, holidays_count, viral):

    rain_text = "rainy" if rainy else "not rainy"

    prompt = f"""
    You are an inventory management expert.

    Product: {product}
    Temperature: {temp}°C
    Weather: {rain_text}
    Non-working days: {holidays_count}
    Trend score: {viral}

    Should inventory increase or decrease?
    Explain clearly in 4 lines.
    """

    if not HF_API_KEY:
        return "HF Error: Missing API key. Set the HF_API_KEY environment variable."

    try:
        response = requests.post(
            HF_MODEL_URL,
            headers={
                "Authorization": f"Bearer {HF_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "inputs": prompt,
                "parameters": {"max_new_tokens": 120, "temperature": 0.3},
                "options": {"wait_for_model": True}
            },
            timeout=30
        )

        if response.status_code != 200:
            return f"HF Error {response.status_code}: {response.text}"

        result = response.json()

        if isinstance(result, list) and result and "generated_text" in result[0]:
            return result[0]["generated_text"].strip()

        if isinstance(result, dict) and "generated_text" in result:
            return result["generated_text"].strip()

        if isinstance(result, dict) and "error" in result:
            return f"HF Error: {result['error']}"

        return f"Unexpected HF response format: {result}"

    except requests.RequestException as e:
        return f"HF Exception: {str(e)}"

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
    viral_score = simulate_viral_score(product_name)
    st.metric("📊 Social Media Trend Score", viral_score)

st.divider()

# ---------------- FORECAST
if st.button("🚀 Generate Forecast"):

    input_df = pd.DataFrame({
        "holiday_count": [holiday_count],
        "avg_temp": [avg_temp],
        "viral_score": [viral_score]
    })

    predicted_sales = rf_model.predict(input_df)[0]
    recommended_inventory = predicted_sales * 1.10

    # Fetch real price
    real_price = fetch_price(product_name)
    if real_price:
        optimized_price = real_price
    else:
        optimized_price = "Price not found"

    reason = generate_reason(
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
    m3.metric("Market Price", optimized_price)

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
