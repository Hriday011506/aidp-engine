import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import calendar
import holidays
from sklearn.ensemble import RandomForestRegressor

try:
    from serpapi import GoogleSearch
except ImportError:
    GoogleSearch = None

st.set_page_config(page_title="OptiRetail AI", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

SERPAPI_KEY = st.secrets.get("SERPAPI_KEY", "")
MONGO_URI = st.secrets.get("MONGO_URI", "")

DEFAULTS = {
    "user": None,
    "page": "welcome",
    "product": "Wheat Flour",
    "city": "Jaipur",
    "month_name": datetime.datetime.now().strftime("%B"),
    "last_prediction": None,
    "current_price": None,
    "users": []
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background: radial-gradient(circle at 8% 0%,rgba(34,197,94,.10),transparent 28%),radial-gradient(circle at 100% 8%,rgba(16,185,129,.08),transparent 26%),linear-gradient(135deg,#f7fbf9,#ffffff 50%,#f3faf7); color:#0f172a; }
.block-container { max-width: 1480px; padding-top: 1.2rem; padding-bottom: 4rem; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#062b22,#071d18); border-right:1px solid rgba(255,255,255,.08); }
[data-testid="stSidebar"] * { color:#ecfdf5; }
.hero { padding:38px 42px; border-radius:30px; background:linear-gradient(135deg,#ffffff,#ecfdf5); border:1px solid #d1fae5; box-shadow:0 24px 70px rgba(15,118,110,.10); margin-bottom:22px; }
.hero h1 { margin:0; font-size:3.15rem; line-height:1.06; letter-spacing:-2px; color:#0f172a; }
.hero p { color:#64748b; margin:11px 0 0; font-size:1rem; max-width:920px; }
.eyebrow { color:#047857; text-transform:uppercase; font-size:.72rem; font-weight:800; letter-spacing:2px; margin-bottom:10px; }
.card { background:rgba(255,255,255,.92); border:1px solid #e2e8f0; border-radius:22px; padding:24px; box-shadow:0 14px 42px rgba(15,23,42,.06); margin-bottom:18px; }
.kpi { background:#ffffff; border:1px solid #e2e8f0; border-radius:18px; padding:20px; min-height:118px; box-shadow:0 10px 26px rgba(15,23,42,.05); }
.kpi small { color:#64748b; text-transform:uppercase; letter-spacing:1.1px; font-weight:800; font-size:.68rem; }
.kpi strong { display:block; color:#0f172a; font-size:1.8rem; margin-top:8px; }
.kpi span { color:#64748b; font-size:.78rem; }
.pill { display:inline-block; padding:7px 11px; border-radius:999px; background:#ecfdf5; border:1px solid #bbf7d0; color:#047857; font-size:.7rem; font-weight:800; }
.stButton>button { border:0!important; border-radius:12px!important; min-height:44px!important; font-weight:800!important; background:linear-gradient(135deg,#059669,#047857)!important; color:white!important; box-shadow:0 8px 22px rgba(5,150,105,.18); }
.stButton>button:hover { transform:translateY(-1px); }
[data-testid="stMetricValue"] { color:#0f172a; }
[data-testid="stMetricLabel"] { color:#64748b; }
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
    st.session_state.users.append({"email": email, "password": password, "gst": gst, "turnover": turnover, "created_at": datetime.datetime.now().isoformat()})
    return True, "Account created successfully."


def login(email, password):
    email = email.strip().lower()
    for user in st.session_state.users:
        if user["email"] == email and user["password"] == password:
            return user
    return None

# ==============================
# FAST MARKET PRICE LOOKUP
# ==============================
@st.cache_data(ttl=900, show_spinner=False)
def fetch_product_price(product_name):
    product_name = product_name.strip()
    if not product_name or not SERPAPI_KEY or GoogleSearch is None:
        return None
    try:
        results = GoogleSearch({
            "engine": "google_shopping_light",
            "q": f"{product_name} price",
            "gl": "in",
            "hl": "en",
            "num": 8,
            "api_key": SERPAPI_KEY
        }).get_dict()
        tokens = [t for t in product_name.lower().split() if len(t) > 2]
        candidates = []
        for item in results.get("shopping_results", []):
            raw = item.get("extracted_price", item.get("price"))
            try:
                if isinstance(raw, str): raw = raw.replace(",", "").replace("₹", "").strip()
                numeric = float(raw)
            except (TypeError, ValueError):
                continue
            if numeric <= 0: continue
            title = str(item.get("title", "")).lower(); source = str(item.get("source", "")).lower()
            hits = sum(token in title for token in tokens)
            score = hits * 3 + (10 if product_name.lower() in title else 0) + (1 if "india" in title or "india" in source else 0)
            candidates.append((score, numeric))
        if not candidates: return None
        candidates.sort(reverse=True)
        best_score = candidates[0][0]
        top_prices = [price for score, price in candidates if score == best_score]
        return float(np.median(top_prices[:5]))
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def get_weather(city):
    try:
        response = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": city, "count": 1}, timeout=5)
        response.raise_for_status(); results = response.json().get("results", [])
        if not results: return 25.0
        lat, lon = results[0]["latitude"], results[0]["longitude"]
        weather = requests.get("https://api.open-meteo.com/v1/forecast", params={"latitude":lat,"longitude":lon,"current_weather":"true"}, timeout=5)
        weather.raise_for_status(); return float(weather.json()["current_weather"]["temperature"])
    except Exception:
        return 25.0


@st.cache_data
def get_holidays(year, month):
    india_holidays = holidays.India(years=year); total_days = calendar.monthrange(year, month)[1]
    return sum(1 for d in range(1,total_days+1) if datetime.date(year,month,d).weekday() >= 5 or datetime.date(year,month,d) in india_holidays)


@st.cache_data
def trend_score(product):
    rng = np.random.default_rng(sum(ord(c) for c in product)); return int(rng.integers(30,90))

# ==============================
# MODEL
# ==============================
@st.cache_data
def load_data():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"holiday_count":rng.integers(0,10,100),"avg_temp":rng.integers(10,40,100),"viral_score":rng.integers(0,100,100)})
    df["sales"] = 200 + df["holiday_count"]*50 + df["avg_temp"]*10 + df["viral_score"]*5
    return df

@st.cache_resource
def train_model(df):
    model=RandomForestRegressor(n_estimators=200,max_depth=12,random_state=42,n_jobs=-1)
    model.fit(df[["holiday_count","avg_temp","viral_score"]],df["sales"]); return model

model=train_model(load_data())

# ==============================
# SIDEBAR
# ==============================
with st.sidebar:
    st.markdown("# 📊 OptiRetail AI")
    st.caption("Analyze · Predict · Price")
    st.divider()
    if st.session_state.user:
        st.markdown(f"**{st.session_state.user.get('email','Business User')}**")
        if st.button("Dashboard", use_container_width=True): st.session_state.page="dashboard"; st.rerun()
        if st.button("Product Analysis", use_container_width=True): st.session_state.page="dashboard"; st.rerun()
        if st.button("Demand Forecasting", use_container_width=True): st.session_state.page="dashboard"; st.rerun()
        if st.button("Dynamic Pricing", use_container_width=True): st.session_state.page="dashboard"; st.rerun()
        if st.button("Market Insights", use_container_width=True): st.session_state.page="dashboard"; st.rerun()
        if st.button("Settings", use_container_width=True): st.session_state.page="settings"; st.rerun()
        st.divider()
        if st.button("Sign out", use_container_width=True): st.session_state.user=None; st.session_state.page="welcome"; st.rerun()
    else:
        st.markdown("### Smarter retail decisions")
        st.caption("Forecast demand. Optimize inventory. Price with confidence.")

# ==============================
# WELCOME
# ==============================
if st.session_state.page == "welcome":
    st.markdown("<div class='hero'><div class='eyebrow'>OPTIRETAIL AI</div><h1>Turn market data into smarter decisions.</h1><p>Forecast demand, get market-price signals and translate them into practical inventory and pricing actions.</p></div>", unsafe_allow_html=True)
    a,b,c=st.columns(3)
    a.metric("Demand Forecasting","AI-powered")
    b.metric("Dynamic Pricing","Market-led")
    c.metric("Inventory","Action-ready")
    st.markdown("<div class='card'><div class='eyebrow'>GET STARTED</div><h2>One workspace for modern retail.</h2><p style='color:#64748b;'>Use OptiRetail AI to analyze a product, understand the demand context and generate a planning recommendation.</p></div>",unsafe_allow_html=True)
    x,y=st.columns(2)
    if x.button("Login →",use_container_width=True): st.session_state.page="login"; st.rerun()
    if y.button("Create account →",use_container_width=True): st.session_state.page="signup"; st.rerun()

# ==============================
# LOGIN
# ==============================
elif st.session_state.page == "login":
    st.markdown("<div class='hero'><div class='eyebrow'>WELCOME BACK</div><h1>Sign in to OptiRetail AI.</h1><p>Continue to your retail intelligence workspace.</p></div>",unsafe_allow_html=True)
    with st.form("login_form"):
        email=st.text_input("Email address"); password=st.text_input("Password",type="password")
        if st.form_submit_button("Sign in",use_container_width=True):
            user=login(email,password)
            if user: st.session_state.user=user; st.session_state.page="dashboard"; st.rerun()
            else: st.error("Invalid credentials.")

# ==============================
# SIGNUP
# ==============================
elif st.session_state.page == "signup":
    st.markdown("<div class='hero'><div class='eyebrow'>BUSINESS ONBOARDING</div><h1>Create your OptiRetail AI account.</h1><p>Start making data-driven retail decisions.</p></div>",unsafe_allow_html=True)
    with st.form("signup_form"):
        email=st.text_input("Email address"); password=st.text_input("Password",type="password"); gst=st.text_input("GST Number"); turnover=st.selectbox("Annual Turnover",["1–5 Lakh","5–10 Lakh","10–15 Lakh","15–50 Lakh","50 Lakh+"])
        if st.form_submit_button("Create account",use_container_width=True):
            if not email or not password or not gst: st.error("Please fill all required fields.")
            else:
                ok,msg=signup(email,password,gst,turnover)
                if ok: st.success(msg); st.session_state.page="login"; st.rerun()
                else: st.error(msg)

# ==============================
# SETTINGS
# ==============================
elif st.session_state.page == "settings" and st.session_state.user:
    st.markdown("<div class='hero'><div class='eyebrow'>WORKSPACE</div><h1>Business settings</h1><p>Manage the business information associated with your workspace.</p></div>",unsafe_allow_html=True)
    user=st.session_state.user; c1,c2,c3=st.columns(3); c1.metric("Account","Active"); c2.metric("Email",user.get("email","N/A")); c3.metric("Turnover",user.get("turnover","N/A"))
    st.info("OptiRetail AI uses SerpAPI for market-price intelligence.")

# ==============================
# DASHBOARD
# ==============================
elif st.session_state.page == "dashboard" and st.session_state.user:
    st.markdown("<div class='hero'><div class='eyebrow'>LIVE RETAIL INTELLIGENCE</div><h1>Make the next retail decision with confidence.</h1><p>Analyze a product, understand demand drivers, set a planning price and convert the forecast into a stock decision.</p></div>",unsafe_allow_html=True)
    c1,c2,c3=st.columns([1.6,1,1])
    with c1: product=st.text_input("Product",value=st.session_state.product,placeholder="e.g. Amul Taaza Milk 1L")
    with c2: city=st.text_input("City",value=st.session_state.city)
    with c3:
        months=list(calendar.month_name)[1:]; month_name=st.selectbox("Forecast month",months,index=months.index(st.session_state.month_name))
    st.session_state.product,st.session_state.city,st.session_state.month_name=product,city,month_name

    price_col, analyze_col = st.columns([1,3])
    with price_col:
        if st.button("Refresh market price",use_container_width=True):
            with st.spinner("Checking market prices…"):
                p=fetch_product_price(product)
            if p is None: st.warning("No reliable shopping price found. Use a specific product name.")
            else: st.session_state.current_price=p; st.success(f"Market reference: ₹{p:,.2f}")
            st.rerun()
    with analyze_col:
        if st.button("Generate AI Decision →",use_container_width=True):
            month=months.index(month_name)+1; year=datetime.datetime.now().year
            h=get_holidays(year,month); temp=get_weather(city); trend=trend_score(product)
            inp=pd.DataFrame({"holiday_count":[h],"avg_temp":[temp],"viral_score":[trend]})
            pred=float(model.predict(inp)[0]); stock=float(np.ceil(pred*1.10))
            market=st.session_state.get("current_price")
            suggested=float(market*(1+np.clip((pred-300)/3000,-0.08,0.08))) if isinstance(market,(int,float)) else None
            st.session_state.last_prediction={"pred":pred,"stock":stock,"suggested":suggested,"temp":temp,"holiday":h,"trend":trend,"market":market}

    current=st.session_state.get("current_price")
    current_text=f"₹{current:,.2f}" if isinstance(current,(int,float)) else "Refresh"
    prediction=st.session_state.get("last_prediction") or {}
    k1,k2,k3,k4=st.columns(4)
    k1.markdown(f"<div class='kpi'><small>Forecasted demand</small><strong>{int(prediction['pred']):,} units</strong><span>AI demand estimate</span></div>" if prediction else "<div class='kpi'><small>Forecasted demand</small><strong>—</strong><span>Generate a decision</span></div>",unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi'><small>Recommended stock</small><strong>{int(prediction['stock']):,} units</strong><span>Includes 10% planning buffer</span></div>" if prediction else "<div class='kpi'><small>Recommended stock</small><strong>—</strong><span>Inventory action</span></div>",unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi'><small>Market reference</small><strong>{current_text}</strong><span>SerpAPI shopping signal</span></div>",unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi'><small>Suggested price</small><strong>{('₹'+format(prediction['suggested'],',.2f')) if prediction and prediction.get('suggested') is not None else '—'}</strong><span>Demand-response guidance</span></div>",unsafe_allow_html=True)

    if prediction:
        r=prediction
        left,right=st.columns([1.5,1])
        with left:
            st.markdown("<div class='card'><div class='eyebrow'>DEMAND FORECAST</div><h2 style='margin:0 0 5px'>Demand outlook</h2><div style='color:#64748b;font-size:.85rem;'>Selected product + city + month</div></div>",unsafe_allow_html=True)
            chart=pd.DataFrame({"Metric":["Demand","Recommended Stock"],"Units":[r["pred"],r["stock"]]}).set_index("Metric")
            st.bar_chart(chart,use_container_width=True)
        with right:
            st.markdown("<div class='card'><div class='eyebrow'>PRICING GUIDANCE</div><h2 style='margin:0 0 8px'>Price optimization</h2><div style='color:#64748b;font-size:.85rem;'>Market reference + bounded demand response</div></div>",unsafe_allow_html=True)
            if r.get("suggested") is not None: st.metric("Suggested price",f"₹{r['suggested']:,.2f}")
            else: st.info("Refresh market price first to generate pricing guidance.")
            st.success("High demand momentum detected." if r["trend"]>=70 else "Demand conditions appear relatively stable.")
        st.markdown("### Market signals")
        s1,s2,s3=st.columns(3); s1.metric("Temperature",f"{r['temp']:.1f}°C"); s2.metric("Holiday / weekend days",r["holiday"]); s3.metric("Trend score",f"{r['trend']}/100")

    st.markdown("### Decision guidance")
    st.info("**Inventory:** use the recommended stock as a planning baseline and adjust for current stock, lead time and safety stock policy.")
    st.info("**Pricing:** the market reference is an external shopping signal; the suggested price is a bounded decision aid, not a guaranteed market price.")
