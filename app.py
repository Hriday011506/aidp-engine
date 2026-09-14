import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import calendar
import holidays
from sklearn.ensemble import RandomForestRegressor
from google_search_results import GoogleSearch

st.set_page_config(page_title="AIDP Engine", page_icon="🚀", layout="wide", initial_sidebar_state="expanded")

SERPAPI_KEY = st.secrets.get("SERPAPI_KEY", "")

DEFAULTS = {
    "user": None,
    "page": "welcome",
    "product": "Wheat Flour",
    "city": "Jaipur",
    "month_name": datetime.datetime.now().strftime("%B"),
    "last_prediction": None,
    "users": []
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background: radial-gradient(circle at 8% 0%,rgba(37,99,235,.16),transparent 28%),radial-gradient(circle at 100% 8%,rgba(124,58,237,.14),transparent 26%),linear-gradient(135deg,#020617,#07111f 52%,#020617); color:#e2e8f0; }
.block-container { max-width: 1480px; padding-top: 1.4rem; padding-bottom: 4rem; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#020617,#0b1220); border-right:1px solid rgba(148,163,184,.12); }
[data-testid="stSidebar"] * { color:#dbeafe; }
.hero { padding:38px 40px; border-radius:32px; background:linear-gradient(135deg,rgba(15,23,42,.96),rgba(30,41,59,.78)); border:1px solid rgba(96,165,250,.16); box-shadow:0 28px 80px rgba(0,0,0,.3); margin-bottom:22px; }
.hero h1 { margin:0; font-size:3.35rem; line-height:1.04; letter-spacing:-2px; }
.hero p { color:#94a3b8; margin:11px 0 0; font-size:1.02rem; max-width:900px; }
.eyebrow { color:#67e8f9; text-transform:uppercase; font-size:.72rem; font-weight:800; letter-spacing:2.2px; margin-bottom:10px; }
.card { background:rgba(15,23,42,.7); border:1px solid rgba(148,163,184,.12); border-radius:24px; padding:24px; box-shadow:0 18px 52px rgba(0,0,0,.2); margin-bottom:18px; }
.kpi { background:linear-gradient(145deg,rgba(15,23,42,.98),rgba(30,41,59,.9)); border:1px solid rgba(96,165,250,.14); border-radius:20px; padding:21px; min-height:124px; box-shadow:inset 0 1px rgba(255,255,255,.03); }
.kpi small { color:#94a3b8; text-transform:uppercase; letter-spacing:1.2px; font-weight:700; }
.kpi strong { display:block; color:#f8fafc; font-size:1.9rem; margin-top:8px; }
.pill { display:inline-block; padding:7px 11px; border-radius:999px; background:rgba(34,197,94,.1); border:1px solid rgba(34,197,94,.2); color:#86efac; font-size:.72rem; font-weight:800; }
.stButton>button { border:0!important; border-radius:13px!important; min-height:46px!important; font-weight:800!important; background:linear-gradient(135deg,#06b6d4,#2563eb)!important; color:white!important; box-shadow:0 10px 28px rgba(37,99,235,.22); }
.stButton>button:hover { transform:translateY(-1px); }
[data-testid="stMetricValue"] { color:#f8fafc; }
[data-testid="stMetricLabel"] { color:#94a3b8; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ==============================
# LOCAL DEMO AUTH
# ==============================
def signup(email, password, gst, turnover):
    email = email.strip().lower()
    if any(u["email"] == email for u in st.session_state.users):
        return False, "An account with this email already exists."
    st.session_state.users.append({
        "email": email,
        "password": password,
        "gst": gst,
        "turnover": turnover,
        "created_at": datetime.datetime.now().isoformat()
    })
    return True, "Account created successfully."


def login(email, password):
    email = email.strip().lower()
    for user in st.session_state.users:
        if user["email"] == email and user["password"] == password:
            return user
    return None

# ==============================
# MARKET DATA SERVICES
# ==============================
def fetch_product_price(product_name):
    if not SERPAPI_KEY:
        return "₹ —"
    try:
        results = GoogleSearch({
            "engine": "google_shopping",
            "q": product_name,
            "gl": "in",
            "hl": "en",
            "api_key": SERPAPI_KEY
        }).get_dict()
        products = results.get("shopping_results", [])
        if products:
            price = products[0].get("price") or products[0].get("extracted_price")
            if price is not None:
                value = str(price)
                return value if "₹" in value else f"₹{value}"
    except Exception:
        pass
    return "₹ —"


def get_weather(city):
    try:
        response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1}, timeout=8
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            return 25.0
        lat, lon = results[0]["latitude"], results[0]["longitude"]
        weather = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": "true"}, timeout=8
        )
        weather.raise_for_status()
        return float(weather.json()["current_weather"]["temperature"])
    except Exception:
        return 25.0


def get_holidays(year, month):
    india_holidays = holidays.India(years=year)
    total_days = calendar.monthrange(year, month)[1]
    return sum(
        1 for d in range(1, total_days + 1)
        if datetime.date(year, month, d).weekday() >= 5
        or datetime.date(year, month, d) in india_holidays
    )


def simulate_viral_score(product):
    rng = np.random.default_rng(sum(ord(c) for c in product))
    return int(rng.integers(30, 90))

# ==============================
# MODEL
# ==============================
@st.cache_data
def load_data():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "holiday_count": rng.integers(0, 10, 100),
        "avg_temp": rng.integers(10, 40, 100),
        "viral_score": rng.integers(0, 100, 100)
    })
    df["sales"] = (
        200 + df["holiday_count"] * 50 + df["avg_temp"] * 10 + df["viral_score"] * 5
    )
    return df


@st.cache_resource
def train_model(df):
    model = RandomForestRegressor(n_estimators=250, max_depth=12, random_state=42)
    model.fit(df[["holiday_count", "avg_temp", "viral_score"]], df["sales"])
    return model

model = train_model(load_data())

# ==============================
# SIDEBAR
# ==============================
with st.sidebar:
    st.markdown("# 🚀 AIDP")
    st.caption("Demand Intelligence Platform")
    st.divider()
    if st.session_state.user:
        st.markdown(f"**{st.session_state.user.get('email','Business User')}**")
        if st.button("Overview", use_container_width=True):
            st.session_state.page = "dashboard"; st.rerun()
        if st.button("Forecast", use_container_width=True):
            st.session_state.page = "dashboard"; st.rerun()
        if st.button("Settings", use_container_width=True):
            st.session_state.page = "settings"; st.rerun()
        st.divider()
        if st.button("Sign out", use_container_width=True):
            st.session_state.user = None; st.session_state.page = "welcome"; st.rerun()
    else:
        st.markdown("### Built for modern retail")
        st.caption("Forecast demand. Protect inventory. Make faster decisions.")

# ==============================
# WELCOME
# ==============================
if st.session_state.page == "welcome":
    st.markdown("<div class='hero'><div class='eyebrow'>AI-POWERED RETAIL INTELLIGENCE</div><h1>Turn market signals into smarter decisions.</h1><p>AIDP brings demand forecasting, inventory planning, weather context, holiday patterns and market pricing into one business workspace.</p></div>", unsafe_allow_html=True)
    left, right = st.columns([1.35,.85], gap="large")
    with left:
        st.markdown("<div class='card'><span class='pill'>● SYSTEM READY</span><h2 style='margin-top:16px;'>One workspace. Smarter retail decisions.</h2><p style='color:#94a3b8;line-height:1.85;'>A customer-ready interface designed to move a business from reactive decisions to proactive demand planning.</p></div>", unsafe_allow_html=True)
        a,b,c = st.columns(3)
        a.metric("Demand", "AI Forecast")
        b.metric("Inventory", "+10% Buffer")
        c.metric("Signals", "Live Context")
    with right:
        st.markdown("<div class='card'><div class='eyebrow'>BUSINESS PORTAL</div><h2>Get started</h2><p style='color:#94a3b8;'>Create a workspace or sign in to access your intelligence console.</p></div>", unsafe_allow_html=True)
        if st.button("Sign in to AIDP", use_container_width=True): st.session_state.page="login"; st.rerun()
        if st.button("Create business account", use_container_width=True): st.session_state.page="signup"; st.rerun()

# ==============================
# LOGIN
# ==============================
elif st.session_state.page == "login":
    st.markdown("<div class='hero'><div class='eyebrow'>SECURE BUSINESS ACCESS</div><h1>Welcome back.</h1><p>Sign in to continue to your AIDP workspace.</p></div>", unsafe_allow_html=True)
    with st.form("login_form"):
        email = st.text_input("Business email")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", use_container_width=True):
            user = login(email, password)
            if user:
                st.session_state.user = user
                st.session_state.page = "dashboard"
                st.rerun()
            else:
                st.error("Invalid credentials.")

# ==============================
# SIGNUP
# ==============================
elif st.session_state.page == "signup":
    st.markdown("<div class='hero'><div class='eyebrow'>BUSINESS ONBOARDING</div><h1>Build your intelligence workspace.</h1><p>Create your business profile and start using AIDP.</p></div>", unsafe_allow_html=True)
    with st.form("signup_form"):
        email = st.text_input("Business email")
        password = st.text_input("Password", type="password")
        gst = st.text_input("GST Number")
        turnover = st.selectbox("Annual Turnover", ["1–5 Lakh", "5–10 Lakh", "10–15 Lakh", "15–50 Lakh", "50 Lakh+"])
        if st.form_submit_button("Create account", use_container_width=True):
            if not email or not password or not gst:
                st.error("Please fill all required fields.")
            else:
                ok, msg = signup(email, password, gst, turnover)
                if ok:
                    st.success(msg)
                    st.session_state.page = "login"
                    st.rerun()
                else:
                    st.error(msg)

# ==============================
# SETTINGS
# ==============================
elif st.session_state.page == "settings" and st.session_state.user:
    st.markdown("<div class='hero'><div class='eyebrow'>WORKSPACE</div><h1>Business settings</h1><p>Manage the business information associated with your workspace.</p></div>", unsafe_allow_html=True)
    user = st.session_state.user
    st.markdown("<div class='card'><div class='eyebrow'>ACCOUNT PROFILE</div><h2>Business details</h2></div>", unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    c1.metric("Account", "Active")
    c2.metric("Email", user.get("email", "N/A"))
    c3.metric("Turnover", user.get("turnover", "N/A"))
    st.info("AIDP demo authentication is active. SerpAPI is used only for market-price intelligence.")

# ==============================
# DASHBOARD
# ==============================
elif st.session_state.page == "dashboard" and st.session_state.user:
    user = st.session_state.user
    st.markdown("<div class='hero'><div class='eyebrow'>LIVE INTELLIGENCE CONSOLE</div><h1>Good decisions start with good signals.</h1><p>Configure your context, read the market pulse, generate a demand forecast and translate it into an inventory action.</p></div>", unsafe_allow_html=True)

    st.markdown("### Forecast setup")
    c1,c2,c3=st.columns([1.6,1,1])
    with c1: product=st.text_input("Product", value=st.session_state.product)
    with c2: city=st.text_input("City", value=st.session_state.city)
    with c3:
        months=list(calendar.month_name)[1:]
        month_name=st.selectbox("Forecast month", months, index=months.index(st.session_state.month_name))
    st.session_state.product,st.session_state.city,st.session_state.month_name=product,city,month_name

    month=months.index(month_name)+1
    year=datetime.datetime.now().year
    holiday=get_holidays(year,month)
    temp=get_weather(city)
    viral=simulate_viral_score(product)
    price=fetch_product_price(product)

    st.markdown("### Market pulse")
    k1,k2,k3,k4=st.columns(4)
    k1.markdown(f"<div class='kpi'><small>Temperature</small><strong>{temp:.1f}°C</strong></div>",unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi'><small>Holiday Days</small><strong>{holiday}</strong></div>",unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi'><small>Trend Score</small><strong>{viral}/100</strong></div>",unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi'><small>Market Price</small><strong>{price}</strong></div>",unsafe_allow_html=True)

    st.markdown("### Forecast engine")
    if st.button("Generate AI forecast", use_container_width=True):
        input_df=pd.DataFrame({"holiday_count":[holiday],"avg_temp":[temp],"viral_score":[viral]})
        pred=float(model.predict(input_df)[0])
        inventory=float(pred*1.10)
        st.session_state.last_prediction={"pred":pred,"inventory":inventory}

    result=st.session_state.last_prediction
    if result:
        pred=result["pred"]; inventory=result["inventory"]
        r1,r2,r3=st.columns([1.1,1.1,1])
        with r1:
            st.markdown("<div class='card'><div class='eyebrow'>AI FORECAST</div>",unsafe_allow_html=True)
            st.metric("Expected monthly demand", f"{int(pred):,}")
            st.caption("Model-estimated sales for the selected business context.")
            st.markdown("</div>",unsafe_allow_html=True)
        with r2:
            st.markdown("<div class='card'><div class='eyebrow'>INVENTORY ACTION</div>",unsafe_allow_html=True)
            st.metric("Recommended inventory", f"{int(inventory):,}")
            st.caption("Forecast plus a 10% planning buffer.")
            st.markdown("</div>",unsafe_allow_html=True)
        with r3:
            st.markdown("<div class='card'><div class='eyebrow'>MARKET SIGNAL</div>",unsafe_allow_html=True)
            st.metric("Trend score", f"{viral}/100")
            st.caption("Relative demand momentum indicator.")
            st.markdown("</div>",unsafe_allow_html=True)

        st.markdown("### Decision view")
        chart_df=pd.DataFrame({"Metric":["Demand","Recommended inventory"],"Units":[pred,inventory]}).set_index("Metric")
        st.bar_chart(chart_df, use_container_width=True)

        st.markdown("### AI recommendation")
        if viral >= 70:
            st.success("High demand momentum detected. Prioritize replenishment and monitor stock coverage closely.")
        elif holiday >= 8:
            st.info("Elevated holiday activity detected. Consider increasing availability before the forecast month.")
        else:
            st.warning("Demand conditions appear relatively stable. Maintain the recommended inventory buffer and monitor movement.")
    else:
        st.markdown("<div class='card'><div class='eyebrow'>READY</div><h2>Generate your first forecast</h2><p style='color:#94a3b8;'>Choose a product, location and month above, then generate the forecast to unlock the decision view.</p></div>",unsafe_allow_html=True)

    st.markdown("### Business context")
    b1,b2,b3=st.columns(3)
    b1.info("**Inventory:** Use the forecast as a planning baseline, then adjust for supplier lead time and current stock.")
    b2.info("**Pricing:** Market-price data is shown as a reference signal from shopping results.")
    b3.info("**External signals:** Weather and India holiday calendars are refreshed for the selected city/month.")
