import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import calendar
import matplotlib.pyplot as plt
import holidays
from sklearn.ensemble import RandomForestRegressor
from pymongo import MongoClient
import bcrypt
from serpapi import GoogleSearch

# ---------------- MONGODB
MONGO_URI = "mongodb+srv://hridaymahajan1979_db_user:hriday@aidp.jz72py2.mongodb.net/?retryWrites=true&w=majority"

client = MongoClient(MONGO_URI)
db = client["aidp_db"]
users_collection = db["users"]

# ---------------- SESSION
if "user" not in st.session_state:
    st.session_state.user = None
if "market_price" not in st.session_state:
    st.session_state.market_price = "Not available"

# ---------------- AUTH FUNCTIONS
def signup(email, password, gst, turnover):
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

    user_doc = {
        "email": email.lower().strip(),
        "password": hashed_pw,
        "gst": gst,
        "turnover": turnover
    }

    users_collection.insert_one(user_doc)


def login(email, password):
    user = users_collection.find_one({"email": email.lower().strip()})

    if not user:
        return None

    if bcrypt.checkpw(password.encode(), user["password"]):
        return user
    return None


# ---------------- GST MOCK
def fetch_gst_details(gst):
    if gst:
        return {"company_name": f"Demo Company for {gst}"}
    return None


# ---------------- PRICE CONVERSION (ADDED)
def convert_to_inr(price_str):
    try:
        if "$" in price_str:
            price = float(price_str.replace("$", "").replace(",", ""))
            inr = price * 83
            return f"₹{round(inr, 2)}"
        return price_str
    except:
        return price_str


# ---------------- PRODUCT PRICE (UPDATED)
def fetch_product_price(product_name):
    params = {
        "engine": "google_shopping",
        "q": product_name,
        "location": "India",
        "api_key": "YOUR_SERPAPI_KEY"
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    try:
        products = results.get("shopping_results", [])
        for item in products:
            price = item.get("price")
            title = item.get("title", "")
            if price:
                price = convert_to_inr(price)   # ✅ FIX
                return title, price
    except:
        pass

    return None, "Not Available"


# ---------------- PAGE CONFIG
st.set_page_config(page_title="AIDP Engine", layout="wide")


# ---------------- PRODUCT INPUT (UPDATED)
product = st.text_input("Enter Product Name")

if product:
    title, price = fetch_product_price(product)

    st.session_state.market_price = price   # ✅ STORE

    if price != "Not Available":
        st.success(f"💰 Price: {price}")
        st.info(f"📦 Product: {title}")
    else:
        st.warning("Market price not available")


# ---------------- AUTH UI
menu = ["Login", "Signup"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Signup":
    st.subheader("Create Account")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    gst = st.text_input("GST Number")

    if gst:
        company = fetch_gst_details(gst)
        if company:
            st.info(f"Company: {company['company_name']}")

    turnover = st.number_input("Turnover")

    if st.button("Signup"):
        if users_collection.find_one({"email": email}):
            st.error("User already exists")
        else:
            signup(email, password, gst, turnover)
            st.success("Account created!")

elif choice == "Login":
    st.subheader("Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = login(email, password)
        if user:
            st.session_state.user = user
            st.success("Logged in!")
        else:
            st.error("Invalid credentials")


# ---------------- DASHBOARD
if st.session_state.user:

    st.title("📊 AI Retail Demand Dashboard")

    st.subheader("Dashboard Results")

    col1, col2, col3 = st.columns(3)

    predicted_sales = 1000
    inventory = 1100

    col1.metric("📦 Predicted Sales", f"{predicted_sales} Units")
    col2.metric("📈 Inventory", f"{inventory} Units")

    # ✅ FIXED MARKET PRICE DISPLAY
    col3.metric(
        "💰 Market Price",
        st.session_state.get("market_price", "Not available")
    )

else:
    st.warning("Please login to access dashboard")
