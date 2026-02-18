diff --git a/app.py b/app.py
index 0f469427a96648519ef823b4109626f2a9d19fb0..6ed32d3af040519a92820f36c3c4e40fd0a7ce74 100644
--- a/app.py
+++ b/app.py
@@ -1,36 +1,38 @@
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
-HF_API_KEY = "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"
+
+HF_API_KEY = os.getenv("HF_API_KEY", "")
+HF_MODEL_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
 
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
@@ -112,103 +114,103 @@ def load_data():
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
 
 
-# ---------------- GEMINI REASONING
+# ---------------- HUGGING FACE REASONING
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
 
+    if not HF_API_KEY:
+        return "HF Error: Missing API key. Set the HF_API_KEY environment variable."
+
     try:
         response = requests.post(
-            "https://router.huggingface.co/hf-inference/models/google/flan-t5-large",
+            HF_MODEL_URL,
             headers={
                 "Authorization": f"Bearer {HF_API_KEY}",
                 "Content-Type": "application/json"
             },
-            json={"inputs": prompt},
-            timeout=20
+            json={
+                "inputs": prompt,
+                "parameters": {"max_new_tokens": 120, "temperature": 0.3},
+                "options": {"wait_for_model": True}
+            },
+            timeout=30
         )
 
-        result = response.json()
-
-        if isinstance(result, list) and "generated_text" in result[0]:
-            return result[0]["generated_text"]
+        if response.status_code != 200:
+            return f"HF Error {response.status_code}: {response.text}"
 
-        if "error" in result:
-            return f"HF Error: {result['error']}"
-
-        return "Unexpected response format."
+        result = response.json()
 
-    except Exception as e:
-        return f"HF Exception: {str(e)}"
-        # SHOW RAW RESPONSE
-        st.write("HF Raw Response:", result)
+        if isinstance(result, list) and result and "generated_text" in result[0]:
+            return result[0]["generated_text"].strip()
 
-        if isinstance(result, list) and "generated_text" in result[0]:
-            return result[0]["generated_text"]
+        if isinstance(result, dict) and "generated_text" in result:
+            return result["generated_text"].strip()
 
-        if "error" in result:
+        if isinstance(result, dict) and "error" in result:
             return f"HF Error: {result['error']}"
 
-        return "Unexpected HF response format."
+        return f"Unexpected HF response format: {result}"
 
-    except Exception as e:
+    except requests.RequestException as e:
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
