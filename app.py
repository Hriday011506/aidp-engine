# ==============================
# 📊 AIDP ENGINE – PROFESSIONAL VERSION
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import calendar
import holidays
import os
from sklearn.ensemble import RandomForestRegressor

# ==============================
# 🔐 CONFIGURATION (HARDCODED FOR DEMO)
# ==============================

MONGO_URI = "mongodb+srv://hridaymahajan1979_db_user:hriday@aidp.jz72py2.mongodb.net/?retryWrites=true&w=majority"

SERPAPI_KEY = "bbc8aca8053bbe60b9c7017e236f71656667f6b4d2bbf3b2da695084ad8766b4"

# ==============================
# 🗄️ DATABASE CONNECTION
# ==============================

try:
    from pymongo import MongoClient
    import bcrypt

    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=4000)
    client.server_info()

    db = client["aidp_db"]
    users_collection = db["users"]

    db_ok = True

except Exception as e:
    db_ok = False
    users_collection = None

# fallback for local testing
_in_memory_users = []

# ==============================
# 🔐 AUTH FUNCTIONS
# ==============================

def get_user(email):
    if db_ok:
        return users_collection.find_one({"email": email})
    return next((u for u in _in_memory_users if u["email"] == email), None)


def save_user(user):
    if db_ok:
        users_collection.insert_one(user)
    else:
        _in_memory_users.append(user)


def signup(email, password, gst, turnover, industry, location):
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

    user_doc = {
        "email": email.lower().strip(),
        "password": hashed_pw,
        "gst": gst,
        "turnover": turnover,
        "industry": industry,
        "location": location,
        "created_at": datetime.datetime.now()
    }

    save_user(user_doc)


def login(email, password):
    user = get_user(email.lower().strip())

    if not user:
        return None

    if bcrypt.checkpw(password.encode(), user["password"]):
        return user

    return None

# ==============================
# 🏢 GST MOCK API
# ==============================

def fetch_gst_details(gst):
    if gst:
        return {
            "company_name": f"{gst} Pvt Ltd",
            "gst_number": gst,
            "status": "Active"
        }
    return None

# ==============================
# 💰 PRICE FETCHING (SERP API)
# ==============================

from serpapi import GoogleSearch

def fetch_product_price(product_name):
    params = {
        "engine": "google_shopping",
        "q": product_name,
        "gl": "in",
        "hl": "en",
        "api_key": SERPAPI_KEY
    }

    try:
        results = GoogleSearch(params).get_dict()
        products = results.get("shopping_results", [])

        if products:
            product = products[0]
            title = product.get("title", product_name)
            price = product.get("price") or product.get("extracted_price")

            return title, price if price else "Not Available"

    except Exception:
        return product_name, "Error fetching"

    return product_name, "Not Available"

# ==============================
# 🌦️ WEATHER API
# ==============================

def get_weather(city):
    try:
        geo = requests.get(
            f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        ).json()

        if "results" not in geo:
            return 25, False

        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]

        weather = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        ).json()

        temp = weather["current_weather"]["temperature"]
        code = weather["current_weather"]["weathercode"]

        return temp, (50 <= code <= 67)

    except:
        return 25, False

# ==============================
# 📅 HOLIDAY COUNT
# ==============================

def get_holidays(year, month):
    india_holidays = holidays.India(years=year)
    total_days = calendar.monthrange(year, month)[1]

    return sum(
        1 for d in range(1, total_days + 1)
        if datetime.date(year, month, d).weekday() >= 5
        or datetime.date(year, month, d) in india_holidays
    )

# ==============================
# 🔥 VIRAL SCORE
# ==============================

def simulate_viral_score(product):
    seed = abs(hash(product)) % 100
    np.random.seed(seed)

    score = np.random.randint(30, 70)

    if any(word in product.lower() for word in ["phone", "fashion", "jacket"]):
        score += 20

    return min(score, 100)

# ==============================
# 📊 MODEL
# ==============================

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
