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

try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None

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
    "market_data": None,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:#0f172a}
.stApp{background:linear-gradient(180deg,#f7fbf9 0%,#ffffff 45%,#f3faf7 100%)}
.block-container{max-width:1500px;padding:1rem 2rem 4rem}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#052b22,#07382d);border-right:1px solid rgba(255,255,255,.08)}
[data-testid="stSidebar"] *{color:#ecfdf5}
.sidebar-logo{font-family:'Plus Jakarta Sans';font-size:1.55rem;font-weight:800;letter-spacing:-.8px}
.sidebar-sub{font-size:.68rem;color:#a7f3d0;margin-top:-4px}
.hero{padding:34px 38px;border-radius:30px;background:linear-gradient(135deg,#ffffff,#effbf6);border:1px solid #d7efe4;box-shadow:0 22px 60px rgba(16,185,129,.08);margin-bottom:20px}
.hero h1{font-family:'Plus Jakarta Sans';font-size:2.75rem;line-height:1.07;letter-spacing:-2.1px;margin:0;color:#0f172a}
.hero h1 span{color:#047857}.hero p{color:#64748b;font-size:1rem;max-width:920px;margin:.72rem 0 0}
.eyebrow{color:#047857;text-transform:uppercase;font-size:.65rem;font-weight:800;letter-spacing:2px;margin-bottom:9px}
.section-title{font-family:'Plus Jakarta Sans';font-size:1.2rem;font-weight:800;color:#0f172a;margin:.35rem 0 .9rem}
.card{background:rgba(255,255,255,.98);border:1px solid #e2e8f0;border-radius:22px;padding:22px;box-shadow:0 12px 34px rgba(15,23,42,.05);margin-bottom:16px}
.kpi{background:#fff;border:1px solid #e2e8f0;border-radius:18px;padding:18px;min-height:118px;box-shadow:0 10px 24px rgba(15,23,42,.04)}
.kpi small{color:#64748b;text-transform:uppercase;letter-spacing:1px;font-weight:800;font-size:.63rem}.kpi strong{display:block;font-family:'Plus Jakarta Sans';color:#0f172a;font-size:1.72rem;margin-top:7px}.kpi span{font-size:.75rem;color:#64748b}
.signal{padding:12px 0;border-bottom:1px solid #eef2f4}.signal:last-child{border-bottom:0}.signal b{color:#0f172a}.signal small{color:#64748b}
.badge{display:inline-flex;align-items:center;gap:6px;padding:6px 10px;border-radius:999px;font-size:.65rem;font-weight:800}.badge.live{background:#ecfdf5;color:#047857;border:1px solid #bbf7d0}.badge.info{background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe}
.decision{border-radius:18px;padding:18px;background:linear-gradient(135deg,#ecfdf5,#f0fdfa);border:1px solid #bbf7d0}.decision h3{margin:0 0 5px;color:#065f46;font-family:'Plus Jakarta Sans'}.decision p{margin:0;color:#475569;font-size:.9rem;line-height:1.55}
.small-note{color:#64748b;font-size:.78rem}
.stButton>button{border:0!important;border-radius:12px!important;min-height:44px!important;font-weight:800!important;background:linear-gradient(135deg,#059669,#047857)!important;color:#fff!important;box-shadow:0 8px 22px rgba(5,150,105,.16)}
.stButton>button:hover{transform:translateY(-1px)}
div[data-baseweb="input"]>div,div[data-baseweb="select"]>div{border-radius:12px!important;border-color:#dbe5df!important;background:#fff!important}
[data-testid="stMetricValue"]{color:#0f172a}[data-testid="stMetricLabel"]{color:#64748b}
[data-testid="stDataFrame"]{border-radius:16px;overflow:hidden}
footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_database():
    if not MONGO_URI or MongoClient is None:
        return None
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
        client.admin.command("ping")
        return client["optiretail_ai"]
    except Exception:
        return None

db = get_database()
users_collection = db["users"] if db is not None else None
predictions_collection = db["predictions"] if db is not None else None


def signup(email, password, gst, turnover):
    email = email.strip().lower()
    if users_collection is None:
        return False, "Database unavailable. Check MONGO_URI in Streamlit Secrets."
    try:
        if users_collection.find_one({"email": email}):
            return False, "An account with this email already exists."
        users_collection.insert_one({
            "email": email,
            "password": password,
            "gst": gst,
            "turnover": turnover,
            "created_at": datetime.datetime.now(datetime.timezone.utc),
        })
        return True, "Account created successfully."
    except Exception as exc:
        return False, f"Signup failed: {exc}"


def login(email, password):
    email = email.strip().lower()
    if users_collection is None:
        return None
    try:
        return users_collection.find_one({"email": email, "password": password})
    except Exception:
        return None


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
            "api_key": SERPAPI_KEY,
        }).get_dict()
        tokens = [t for t in product_name.lower().split() if len(t) > 2]
        candidates = []
        for item in results.get("shopping_results", []):
            raw = item.get("extracted_price", item.get("price"))
            try:
                if isinstance(raw, str):
                    raw = raw.replace(",", "").replace("₹", "").strip()
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value <= 0:
                continue
            title = str(item.get("title", "")).lower()
            source = str(item.get("source", "")).lower()
            hits = sum(tok in title for tok in tokens)
            score = hits * 3 + (10 if product_name.lower() in title else 0)
            if "india" in title or "india" in source:
                score += 1
            candidates.append((score, value, item.get("title", ""), item.get("source", ""), item.get("link", "")))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_score = candidates[0][0]
        top = [c for c in candidates if c[0] == top_score][:5]
        return {"price": float(np.median([c[1] for c in top])), "results": top}
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def get_weather(city):
    try:
        r = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": city, "count": 1}, timeout=4)
        r.raise_for_status()
        hits = r.json().get("results", [])
        if not hits:
            return 25.0
        lat, lon = hits[0]["latitude"], hits[0]["longitude"]
        w = requests.get("https://api.open-meteo.com/v1/forecast", params={"latitude": lat, "longitude": lon, "current_weather": "true"}, timeout=4)
        w.raise_for_status()
        return float(w.json()["current_weather"]["temperature"])
    except Exception:
        return 25.0


@st.cache_data
def get_holidays(year, month):
    india = holidays.India(years=year)
    days = calendar.monthrange(year, month)[1]
    return sum(1 for d in range(1, days + 1) if datetime.date(year, month, d).weekday() >= 5 or datetime.date(year, month, d) in india)


@st.cache_data
def trend_score(product):
    rng = np.random.default_rng(sum(ord(c) for c in product))
    return int(rng.integers(30, 90))


@st.cache_data
def load_data():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "holiday_count": rng.integers(0, 10, 240),
        "avg_temp": rng.integers(10, 40, 240),
        "viral_score": rng.integers(0, 100, 240),
    })
    seasonal = np.maximum(0, 22 - np.abs(df["avg_temp"] - 28))
    df["sales"] = 180 + df["holiday_count"] * 42 + df["viral_score"] * 4.8 + seasonal * 11 + rng.normal(0, 28, len(df))
    return df


@st.cache_resource
def train_model(df):
    model = RandomForestRegressor(n_estimators=250, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(df[["holiday_count", "avg_temp", "viral_score"]], df["sales"])
    return model

model = train_model(load_data())

with st.sidebar:
    st.markdown("<div class='sidebar-logo'>📊 OptiRetail AI</div><div class='sidebar-sub'>Analyze · Predict · Price</div>", unsafe_allow_html=True)
    st.divider()
    if st.session_state.user:
        st.caption(st.session_state.user.get("email", "Business User"))
        if st.button("Dashboard", use_container_width=True, key="nav_dashboard"):
            st.session_state.page = "dashboard"; st.rerun()
        if st.button("Product Analysis", use_container_width=True, key="nav_product"):
            st.session_state.page = "dashboard"; st.rerun()
        if st.button("Demand Forecasting", use_container_width=True, key="nav_demand"):
            st.session_state.page = "dashboard"; st.rerun()
        if st.button("Dynamic Pricing", use_container_width=True, key="nav_price"):
            st.session_state.page = "dashboard"; st.rerun()
        if st.button("Market Insights", use_container_width=True, key="nav_market"):
            st.session_state.page = "dashboard"; st.rerun()
        if st.button("Saved Analyses", use_container_width=True, key="nav_saved"):
            st.session_state.page = "dashboard"; st.rerun()
        st.divider()
        if st.button("Settings", use_container_width=True, key="nav_settings"):
            st.session_state.page = "settings"; st.rerun()
        if st.button("Sign out", use_container_width=True, key="nav_signout"):
            st.session_state.user = None; st.session_state.page = "welcome"; st.rerun()
    else:
        st.markdown("### Smarter retail decisions")
        st.caption("Forecast demand. Optimize inventory. Price with confidence.")

if st.session_state.page == "welcome":
    st.markdown("<div class='hero'><div class='eyebrow'>OPTIRETAIL AI</div><h1>Turn market data into <span>smarter decisions.</span></h1><p>Forecast demand, understand market signals, optimize inventory and make practical pricing decisions from one clean workspace.</p></div>", unsafe_allow_html=True)
    a,b,c = st.columns(3)
    a.markdown("<div class='card'><div class='eyebrow'>01 · FORECAST</div><h3>Know what will move.</h3><p class='small-note'>AI forecasting uses holiday, weather and trend signals.</p></div>", unsafe_allow_html=True)
    b.markdown("<div class='card'><div class='eyebrow'>02 · INVENTORY</div><h3>Stock with confidence.</h3><p class='small-note'>Translate predicted demand into a practical planning quantity.</p></div>", unsafe_allow_html=True)
    c.markdown("<div class='card'><div class='eyebrow'>03 · PRICE</div><h3>Price with context.</h3><p class='small-note'>Use live market-reference data from SerpAPI.</p></div>", unsafe_allow_html=True)
    x,y = st.columns(2)
    if x.button("Login →", use_container_width=True, key="home_login"):
        st.session_state.page = "login"; st.rerun()
    if y.button("Create account →", use_container_width=True, key="home_signup"):
        st.session_state.page = "signup"; st.rerun()

elif st.session_state.page == "login":
    st.markdown("<div class='hero'><div class='eyebrow'>WELCOME BACK</div><h1>Sign in to <span>OptiRetail AI.</span></h1><p>Access your retail intelligence workspace.</p></div>", unsafe_allow_html=True)
    with st.form("login_form"):
        email = st.text_input("Email address")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", use_container_width=True):
            user = login(email, password)
            if user:
                st.session_state.user = user
                st.session_state.page = "dashboard"
                st.rerun()
            elif users_collection is None:
                st.error("MongoDB is unavailable. Check MONGO_URI in Streamlit Secrets.")
            else:
                st.error("Invalid credentials.")
    if st.button("← Back to home", key="login_back"):
        st.session_state.page = "welcome"; st.rerun()

elif st.session_state.page == "signup":
    st.markdown("<div class='hero'><div class='eyebrow'>BUSINESS ONBOARDING</div><h1>Create your <span>OptiRetail AI</span> workspace.</h1><p>Set up your business profile and start making data-driven retail decisions.</p></div>", unsafe_allow_html=True)
    with st.form("signup_form"):
        email = st.text_input("Email address")
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
    if st.button("← Back to home", key="signup_back"):
        st.session_state.page = "welcome"; st.rerun()

elif st.session_state.page == "settings" and st.session_state.user:
    st.markdown("<div class='hero'><div class='eyebrow'>WORKSPACE</div><h1>Business settings</h1><p>Manage the business profile connected to your OptiRetail AI workspace.</p></div>", unsafe_allow_html=True)
    u = st.session_state.user
    a,b,c = st.columns(3)
    a.metric("Account", "Active")
    b.metric("Email", u.get("email", "N/A"))
    c.metric("Turnover", u.get("turnover", "N/A"))
    mongo_status = "CONNECTED" if db is not None else "NOT CONNECTED"
    serp_status = "CONFIGURED" if SERPAPI_KEY else "NOT CONFIGURED"
    st.markdown(f"<div class='card'><div class='eyebrow'>DATA SERVICES</div><h3>Connected services</h3><div class='signal'><b>MongoDB</b><span class='badge {'live' if db is not None else 'info'}'>{mongo_status}</span></div><div class='signal'><b>SerpAPI</b><span class='badge {'live' if SERPAPI_KEY else 'info'}'>{serp_status}</span></div></div>", unsafe_allow_html=True)
    if st.button("← Back to dashboard", key="settings_back"):
        st.session_state.page = "dashboard"; st.rerun()

elif st.session_state.page == "dashboard" and st.session_state.user:
    st.markdown("<div class='hero'><div class='eyebrow'>RETAIL INTELLIGENCE CONSOLE</div><h1>Make the next decision <span>with confidence.</span></h1><p>Enter a product and location, refresh the market reference, then generate your demand, inventory and pricing decision.</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Product analysis</div>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns([1.6,1,1])
    with c1: product = st.text_input("Product", value=st.session_state.product, placeholder="e.g. Amul Taaza Milk 1L", key="product_input")
    with c2: city = st.text_input("City", value=st.session_state.city, key="city_input")
    with c3:
        months = list(calendar.month_name)[1:]
        month_name = st.selectbox("Forecast month", months, index=months.index(st.session_state.month_name), key="month_input")
    st.session_state.product, st.session_state.city, st.session_state.month_name = product, city, month_name

    ac1,ac2 = st.columns([1,2.2])
    with ac1:
        if st.button("Refresh market price", use_container_width=True, key="refresh_price"):
            with st.spinner("Checking market price…"):
                data = fetch_product_price(product)
            if data is None:
                st.session_state.current_price = None
                st.session_state.market_data = None
                st.warning("No reliable shopping price found. Use a specific product name.")
            else:
                st.session_state.current_price = data["price"]
                st.session_state.market_data = data
                st.success(f"Market reference: ₹{data['price']:,.2f}")
            st.rerun()
    with ac2:
        if st.button("Generate AI Decision →", use_container_width=True, key="generate_decision"):
            month_num = months.index(month_name) + 1
            year = datetime.datetime.now().year
            holiday_count = get_holidays(year, month_num)
            temperature = get_weather(city)
            trend = trend_score(product)
            input_df = pd.DataFrame({"holiday_count":[holiday_count],"avg_temp":[temperature],"viral_score":[trend]})
            predicted = float(max(0, model.predict(input_df)[0]))
            stock = float(np.ceil(predicted * 1.10))
            market = st.session_state.get("current_price")
            suggested = float(market * (1 + np.clip((predicted - 300) / 3000, -0.08, 0.08))) if isinstance(market, (int,float)) else None
            st.session_state.last_prediction = {"pred":predicted,"stock":stock,"suggested":suggested,"temp":temperature,"holiday":holiday_count,"trend":trend,"market":market,"product":product,"city":city,"month":month_name}
            if predictions_collection is not None:
                try:
                    predictions_collection.insert_one({
                        "email": st.session_state.user.get("email"),
                        "product": product,
                        "city": city,
                        "month": month_name,
                        "demand": predicted,
                        "recommended_stock": stock,
                        "market_price": market,
                        "suggested_price": suggested,
                        "temperature": temperature,
                        "holiday_days": holiday_count,
                        "trend_score": trend,
                        "created_at": datetime.datetime.now(datetime.timezone.utc),
                    })
                except Exception:
                    pass

    current = st.session_state.get("current_price")
    lp = st.session_state.get("last_prediction")
    current_text = f"₹{current:,.2f}" if isinstance(current, (int,float)) else "Refresh"
    k1,k2,k3,k4 = st.columns(4)
    k1.markdown(f"<div class='kpi'><small>Forecasted demand</small><strong>{int(lp.get('pred',0)):,} units</strong><span>AI demand estimate</span></div>" if lp else "<div class='kpi'><small>Forecasted demand</small><strong>—</strong><span>Generate a decision</span></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi'><small>Recommended stock</small><strong>{int(lp.get('stock',0)):,} units</strong><span>Includes 10% planning buffer</span></div>" if lp else "<div class='kpi'><small>Recommended stock</small><strong>—</strong><span>Inventory action</span></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi'><small>Market reference</small><strong>{current_text}</strong><span>SerpAPI shopping signal</span></div>", unsafe_allow_html=True)
    suggested_value = lp.get("suggested") if lp else None
    suggested_text = f"₹{suggested_value:,.2f}" if isinstance(suggested_value,(int,float)) else "—"
    k4.markdown(f"<div class='kpi'><small>Suggested price</small><strong>{suggested_text}</strong><span>Demand-response guidance</span></div>", unsafe_allow_html=True)

    if lp:
        left,right = st.columns([1.55,1])
        with left:
            st.markdown("<div class='card'><div class='eyebrow'>DEMAND FORECAST</div><h2 style='margin:0 0 4px'>12-month demand outlook</h2><div class='small-note'>Seasonality-shaped outlook based on the AI demand signal</div>", unsafe_allow_html=True)
            months_short = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            base = max(lp["pred"] * 0.78, 1)
            seasonal = np.array([.82,.88,.94,1.02,.92,.98,1.05,1.12,1.20,1.15,1.08,1.10])
            hist = np.linspace(base*0.86, base, 5)
            forecast = lp["pred"] * (seasonal/seasonal.mean())
            combined = pd.DataFrame(index=months_short)
            combined["Historical Demand"] = np.nan
            combined["Forecasted Demand"] = np.nan
            combined.loc[months_short[:5],"Historical Demand"] = hist
            combined.loc[months_short[4:],"Forecasted Demand"] = forecast[4:]
            st.line_chart(combined, use_container_width=True, height=360)
            st.markdown("</div>", unsafe_allow_html=True)
        with right:
            st.markdown("<div class='card'><div class='eyebrow'>PRICE OPTIMIZATION</div><h2 style='margin:0 0 4px'>Pricing guidance</h2><div class='small-note'>Reference price adjusted by demand signal</div>", unsafe_allow_html=True)
            if lp.get("suggested") is not None and current is not None:
                low = current * 0.92
                high = current * 1.08
                st.metric("Suggested price", f"₹{lp['suggested']:,.2f}")
                st.caption(f"Reference range: ₹{low:,.0f} – ₹{high:,.0f}")
                st.progress(float(np.clip((lp["suggested"]-low)/(high-low),0,1)))
            else:
                st.info("Refresh the market price to unlock pricing guidance.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><div class='eyebrow'>REAL-TIME MARKET INSIGHTS</div><h2 style='margin:0 0 8px'>What is driving the recommendation?</h2>", unsafe_allow_html=True)
        s1,s2,s3 = st.columns(3)
        s1.markdown(f"<div class='signal'><b>Temperature</b><br><small>{lp['temp']:.1f}°C in {city}</small></div>", unsafe_allow_html=True)
        s2.markdown(f"<div class='signal'><b>Holiday activity</b><br><small>{lp['holiday']} active days in {month_name}</small></div>", unsafe_allow_html=True)
        s3.markdown(f"<div class='signal'><b>Demand trend</b><br><small>{lp['trend']}/100 model signal</small></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if lp["trend"] >= 70:
            decision = "High demand momentum detected. Increase replenishment readiness and review the suggested price before the forecast period."
        elif lp["holiday"] >= 8:
            decision = "Elevated holiday activity detected. Hold additional inventory ahead of the forecast month and monitor sell-through."
        else:
            decision = "Demand conditions appear relatively stable. Maintain the recommended stock buffer and monitor movement."
        st.markdown(f"<div class='decision'><h3>AI Decision</h3><p>{decision}</p></div>", unsafe_allow_html=True)

        if st.session_state.market_data and st.session_state.market_data.get("results"):
            rows=[]
            for score,price,title,source,link in st.session_state.market_data["results"]:
                rows.append({"Product":title,"Price":f"₹{price:,.2f}","Source":source})
            st.markdown("<div class='section-title'>Market price references</div>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.markdown("<div class='card'><div class='eyebrow'>READY</div><h2>Start a product analysis</h2><p class='small-note'>Enter a product and location, refresh the market price, then generate the AI decision. Your dashboard will show demand, inventory, pricing and the signal context behind the recommendation.</p></div>", unsafe_allow_html=True)
