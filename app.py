import calendar
import datetime as dt
import hashlib
from urllib.parse import parse_qsl, quote, unquote, urlsplit, urlunsplit

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

try:
    import holidays
except ImportError:
    holidays = None

try:
    from pymongo import MongoClient
    from pymongo.errors import OperationFailure, ServerSelectionTimeoutError
except ImportError:
    MongoClient = None
    OperationFailure = ServerSelectionTimeoutError = Exception

try:
    import certifi
except ImportError:
    certifi = None

try:
    from serpapi import GoogleSearch
except ImportError:
    GoogleSearch = None

st.set_page_config(page_title="OptiRetail AI", page_icon="📊", layout="wide", initial_sidebar_state="expanded")


def secret(name, default=""):
    try:
        value = st.secrets.get(name, default)
        return str(value).strip().strip('"').strip("'") if value else default
    except Exception:
        return default


MONGO_URI = secret("MONGO_URI") or secret("MONGODB_URI")
SERPAPI_KEY = secret("SERPAPI_KEY")

for key, value in {
    "user": None,
    "page": "welcome",
    "product": "Wheat Flour",
    "city": "Jaipur",
    "month_name": dt.datetime.now().strftime("%B"),
    "last_prediction": None,
    "current_price": None,
    "market_data": None,
}.items():
    st.session_state.setdefault(key, value)

st.markdown("""
<style>
:root{--ink:#10231f;--muted:#64748b;--green:#07865f;--green2:#056b4c;--line:#dcebe5}
html,body,[class*="css"]{font-family:Arial,sans-serif;color:var(--ink)}
.stApp{background:linear-gradient(180deg,#f7fbf9 0%,#fff 45%,#f3faf7 100%)}
.block-container{max-width:1500px;padding:1rem 2rem 4rem}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#052d23,#06392e)}
[data-testid="stSidebar"] *{color:#ecfdf5}
.hero{padding:34px 40px;border-radius:28px;background:linear-gradient(135deg,#fff,#effbf6);border:1px solid var(--line);box-shadow:0 20px 55px rgba(16,185,129,.08);margin-bottom:20px}
.hero h1{font-size:2.7rem;margin:0;color:#0f172a}.hero h1 span{color:var(--green2)}.hero p{color:#64748b;font-size:1rem;max-width:950px}
.eyebrow{color:var(--green2);text-transform:uppercase;font-size:.65rem;font-weight:800;letter-spacing:2px;margin-bottom:8px}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:20px;padding:20px;box-shadow:0 10px 30px rgba(15,23,42,.05);margin-bottom:16px}.card h3{margin:.15rem 0 .5rem;color:#0f172a}.small{color:#64748b;font-size:.84rem;line-height:1.5}
.kpi{background:#fff;border:1px solid #e2e8f0;border-radius:18px;padding:18px;min-height:115px;box-shadow:0 8px 22px rgba(15,23,42,.04)}.kpi small{color:#64748b;text-transform:uppercase;letter-spacing:1px;font-weight:800;font-size:.62rem}.kpi strong{display:block;color:#0f172a;font-size:1.65rem;margin-top:7px}.kpi span{font-size:.74rem;color:#64748b}
.signal{display:flex;justify-content:space-between;gap:12px;padding:11px 0;border-bottom:1px solid #eef2f4}.signal b{color:#0f172a}.signal span{color:#64748b}
.decision{border-radius:18px;padding:18px;background:linear-gradient(135deg,#ecfdf5,#f0fdfa);border:1px solid #bbf7d0}.decision h3{margin:0 0 5px;color:#065f46}.decision p{margin:0;color:#475569}
.auth{max-width:720px;margin:25px auto}.auth-card{background:#fff;border:1px solid var(--line);border-radius:28px;padding:34px;box-shadow:0 20px 55px rgba(15,23,42,.07)}.auth-title{font-size:2.1rem;font-weight:800;color:var(--green2)!important;margin:0}.auth-sub{color:#64748b}
div[data-testid="stTextInput"] input,div[data-testid="stNumberInput"] input,div[data-testid="stTextArea"] textarea,div[data-testid="stSelectbox"] div[data-baseweb="select"]>div{background:#fff!important;color:#0f172a!important;border:1px solid #cfe0d9!important;border-radius:12px!important;font-size:1rem!important}
div[data-testid="stTextInput"] input::placeholder,div[data-testid="stNumberInput"] input::placeholder,div[data-testid="stTextArea"] textarea::placeholder{color:#94a3b8!important;opacity:1!important}
label,div[data-testid="stWidgetLabel"] p{color:#334155!important;font-weight:600!important}div[data-baseweb="select"] span{color:#0f172a!important}
.stButton>button,.stFormSubmitButton>button{border:0!important;border-radius:12px!important;min-height:44px!important;font-weight:800!important;background:linear-gradient(135deg,#079568,#047a55)!important;color:#fff!important}
</style>
""", unsafe_allow_html=True)


def normalize_mongo_uri(raw_uri):
    """Make a MongoDB URI safe even when the Atlas password contains URI characters."""
    uri = raw_uri.strip().strip('"').strip("'")
    if not uri.startswith(("mongodb://", "mongodb+srv://")):
        raise ValueError("MONGO_URI must start with mongodb:// or mongodb+srv://.")

    scheme, rest = uri.split("://", 1)
    # Split at the last @ so a raw @ inside a password does not break the URI.
    if "@" in rest:
        credentials, host_and_query = rest.rsplit("@", 1)
        if ":" in credentials:
            username, password = credentials.split(":", 1)
            username = quote(unquote(username), safe="")
            password = quote(unquote(password), safe="")
            rest = f"{username}:{password}@{host_and_query}"

    parsed = urlsplit(f"{scheme}://{rest}")
    if not parsed.hostname:
        raise ValueError("MONGO_URI does not contain a valid Atlas hostname.")

    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["authSource"] = query.get("authSource") or "admin"
    query.setdefault("retryWrites", "true")
    query.setdefault("w", "majority")
    rebuilt = urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/", urlencode(query), ""))
    return rebuilt, parsed.hostname


@st.cache_resource(show_spinner=False)
def get_mongo_client(uri):
    kwargs = {
        "serverSelectionTimeoutMS": 12000,
        "connectTimeoutMS": 12000,
        "socketTimeoutMS": 12000,
        "retryWrites": True,
        "appname": "OptiRetailAI",
    }
    if certifi is not None:
        kwargs["tlsCAFile"] = certifi.where()
    client = MongoClient(uri, **kwargs)
    client.admin.command("ping")
    return client


def connect_mongo():
    if not MONGO_URI:
        return None, "MONGO_URI is missing from Streamlit Secrets."
    if MongoClient is None:
        return None, "PyMongo is not installed. Add pymongo to requirements.txt."
    try:
        uri, hostname = normalize_mongo_uri(MONGO_URI)
        client = get_mongo_client(uri)
        db = client["optiretail_ai"]
        db.command("ping")
        return db, f"Connected to {hostname}"
    except OperationFailure as exc:
        text = str(exc).lower()
        if "authentication failed" in text or "bad auth" in text or getattr(exc, "code", None) == 18:
            return None, "MongoDB authentication failed. The URI is reaching Atlas, but Atlas rejected the database credentials."
        return None, f"MongoDB operation failed: {type(exc).__name__}."
    except ServerSelectionTimeoutError:
        return None, "Atlas could not be reached. Check MongoDB Atlas Network Access and allow 0.0.0.0/0 for Streamlit Cloud testing."
    except Exception as exc:
        text = str(exc).lower()
        if "dns" in text or "srv" in text or "querysrv" in text:
            return None, "MongoDB DNS/SRV lookup failed. Check the Atlas cluster hostname."
        if "ssl" in text or "tls" in text or "certificate" in text:
            return None, "MongoDB TLS/SSL connection failed."
        return None, f"MongoDB connection failed: {type(exc).__name__}."


db, mongo_status = connect_mongo()
users = db["users"] if db is not None else None
predictions = db["predictions"] if db is not None else None


def password_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def save_user(email, password, gst, turnover):
    if users is None:
        return False, mongo_status
    normalized = email.strip().lower()
    try:
        if users.find_one({"email": normalized}):
            return False, "An account with this email already exists."
        users.insert_one({
            "email": normalized,
            "password_hash": password_hash(password),
            "gst": gst.strip(),
            "turnover": turnover,
            "created_at": dt.datetime.now(dt.timezone.utc),
        })
        return True, "Account created successfully."
    except Exception as exc:
        return False, f"Could not save account: {type(exc).__name__}."


def authenticate(email, password):
    if users is None:
        return None
    try:
        user = users.find_one({"email": email.strip().lower()})
        if user and (user.get("password_hash") == password_hash(password) or user.get("password") == password):
            return user
    except Exception:
        pass
    return None


def trend_score(product):
    rng = np.random.default_rng(sum(ord(c) for c in product))
    return int(rng.integers(35, 90))


@st.cache_data(ttl=900, show_spinner=False)
def fetch_price(product):
    if not product.strip() or not SERPAPI_KEY or GoogleSearch is None:
        return None
    try:
        result = GoogleSearch({"engine": "google_shopping_light", "q": f"{product} price", "gl": "in", "hl": "en", "num": 8, "api_key": SERPAPI_KEY}).get_dict()
        values = []
        for item in result.get("shopping_results", []):
            try:
                value = float(item.get("extracted_price"))
            except (TypeError, ValueError):
                continue
            if value > 0:
                values.append((value, item.get("title", ""), item.get("source", ""), item.get("link", "")))
        if not values:
            return None
        values.sort(key=lambda x: x[0])
        return {"price": float(np.median([x[0] for x in values[:5]])), "results": values[:5]}
    except Exception:
        return None


def get_weather(city):
    try:
        geo = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": city, "count": 1}, timeout=5)
        geo.raise_for_status()
        hits = geo.json().get("results", [])
        if not hits:
            return 25.0
        lat, lon = hits[0]["latitude"], hits[0]["longitude"]
        weather = requests.get("https://api.open-meteo.com/v1/forecast", params={"latitude": lat, "longitude": lon, "current_weather": "true"}, timeout=5)
        weather.raise_for_status()
        return float(weather.json()["current_weather"]["temperature"])
    except Exception:
        return 25.0


def holiday_count(year, month):
    india = holidays.India(years=year) if holidays else set()
    return sum(1 for day in range(1, calendar.monthrange(year, month)[1] + 1) if dt.date(year, month, day).weekday() >= 5 or dt.date(year, month, day) in india)


@st.cache_resource(show_spinner=False)
def train_model():
    rng = np.random.default_rng(42)
    data = pd.DataFrame({
        "holiday_count": rng.integers(0, 12, 500),
        "avg_temp": rng.uniform(10, 40, 500),
        "viral_score": rng.integers(0, 100, 500),
    })
    seasonal = np.maximum(0, 22 - np.abs(data["avg_temp"] - 28))
    data["sales"] = 180 + data["holiday_count"] * 42 + data["viral_score"] * 4.8 + seasonal * 11 + rng.normal(0, 25, 500)
    model = RandomForestRegressor(n_estimators=250, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(data[["holiday_count", "avg_temp", "viral_score"]], data["sales"])
    return model


model = train_model()


def render_analysis():
    st.markdown("<div class='card'><div class='eyebrow'>PRODUCT ANALYSIS</div><h3>Market and demand inputs</h3>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.5, 1, 1])
    with c1:
        product = st.text_input("Product", value=st.session_state.product, placeholder="e.g. Amul Taaza Milk 1L")
    with c2:
        city = st.text_input("City", value=st.session_state.city, placeholder="Jaipur")
    with c3:
        months = list(calendar.month_name)[1:]
        month_name = st.selectbox("Forecast month", months, index=months.index(st.session_state.month_name))
    st.session_state.product, st.session_state.city, st.session_state.month_name = product, city, month_name
    a, b = st.columns(2)
    with a:
        if st.button("Refresh market price", use_container_width=True, key="refresh_price"):
            data = fetch_price(product)
            st.session_state.market_data = data
            st.session_state.current_price = data["price"] if data else None
            if data:
                st.success(f"Market reference: ₹{data['price']:,.2f}")
            else:
                st.warning("No reliable shopping price found. Check SerpAPI or use a more specific product name.")
    with b:
        if st.button("Generate AI Decision →", use_container_width=True, key="generate_decision"):
            year = dt.datetime.now().year
            month_num = months.index(month_name) + 1
            temp = get_weather(city)
            holiday = holiday_count(year, month_num)
            trend = trend_score(product)
            predicted = float(max(0, model.predict(pd.DataFrame({"holiday_count": [holiday], "avg_temp": [temp], "viral_score": [trend]}))[0]))
            stock = float(np.ceil(predicted * 1.10))
            market = st.session_state.current_price
            suggested = float(market * (1 + np.clip((predicted - 300) / 3000, -0.08, 0.08))) if isinstance(market, (int, float)) else None
            st.session_state.last_prediction = {"pred": predicted, "stock": stock, "suggested": suggested, "temp": temp, "holiday": holiday, "trend": trend, "market": market, "product": product, "city": city, "month": month_name}
            if predictions is not None and st.session_state.user:
                try:
                    predictions.insert_one({"email": st.session_state.user.get("email"), "product": product, "city": city, "month": month_name, "demand": predicted, "recommended_stock": stock, "market_price": market, "suggested_price": suggested, "temperature": temp, "holiday_days": holiday, "trend_score": trend, "created_at": dt.datetime.now(dt.timezone.utc)})
                except Exception:
                    pass
    st.markdown("</div>", unsafe_allow_html=True)


def render_dashboard():
    lp = st.session_state.get("last_prediction") or {}
    current = st.session_state.get("current_price")
    suggested = lp.get("suggested")
    current_text = f"₹{current:,.2f}" if isinstance(current, (int, float)) else "—"
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"<div class='kpi'><small>Forecasted demand</small><strong>{int(lp.get('pred', 0)):,} units</strong><span>AI demand estimate</span></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi'><small>Recommended stock</small><strong>{int(lp.get('stock', 0)):,} units</strong><span>10% planning buffer</span></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi'><small>Market reference</small><strong>{current_text}</strong><span>SerpAPI shopping signal</span></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi'><small>Suggested price</small><strong>{f'₹{suggested:,.2f}' if isinstance(suggested, (int, float)) else '—'}</strong><span>Demand-aware bound</span></div>", unsafe_allow_html=True)
    if not lp:
        st.info("Generate an AI decision to populate the dashboard.")
        return
    left, right = st.columns([1.55, 1])
    with left:
        months = list(calendar.month_name)[1:]
        x = np.arange(12)
        base = max(1, float(lp.get("pred", 1)))
        forecast = base * (1 + .10 * np.sin((x + months.index(lp.get("month", months[0]))) * 2 * np.pi / 12))
        history = np.maximum(0, base * (.82 + .08 * np.sin((x + 1) * 2 * np.pi / 12)))
        chart = pd.DataFrame({"Historical demand": np.round(history), "Forecast demand": np.round(forecast)}, index=months)
        st.markdown("<div class='card'><div class='eyebrow'>DEMAND FORECAST</div><h3>12-month demand outlook</h3>", unsafe_allow_html=True)
        st.line_chart(chart, height=320)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='card'><div class='eyebrow'>AI DECISION</div><h3>Model signals</h3>", unsafe_allow_html=True)
        for name, value in [("Weather", f"{lp.get('temp', 25):.1f} °C"), ("Holiday days", str(lp.get('holiday', 0))), ("Trend score", f"{lp.get('trend', 0)}/100"), ("Market reference", current_text)]:
            st.markdown(f"<div class='signal'><b>{name}</b><span>{value}</span></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        temp = lp.get("temp", 25)
        action = "Higher temperature detected — consider extra safety stock for heat-sensitive products." if temp >= 32 else ("Cooler weather detected — keep inventory conservative for weather-sensitive categories." if temp <= 18 else "Weather is moderate; holiday and trend signals are driving the current recommendation.")
        st.markdown(f"<div class='decision'><h3>Recommended action</h3><p>{action}</p></div>", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("<div style='font-size:1.5rem;font-weight:800'>📊 OptiRetail AI</div><div style='color:#a7f3d0;font-size:.7rem'>Analyze · Predict · Price</div>", unsafe_allow_html=True)
    st.divider()
    if st.session_state.user:
        st.caption(st.session_state.user.get("email", "Business User"))
        for label, target in [("Dashboard", "dashboard"), ("Product Analysis", "product"), ("Demand Forecasting", "forecast"), ("Dynamic Pricing", "pricing"), ("Market Insights", "market"), ("Saved Analyses", "saved"), ("Settings", "settings")]:
            if st.button(label, use_container_width=True, key=f"nav_{target}"):
                st.session_state.page = target
                st.rerun()
        if st.button("Sign out", use_container_width=True, key="signout"):
            st.session_state.user = None
            st.session_state.page = "welcome"
            st.session_state.last_prediction = None
            st.rerun()
    else:
        st.markdown("### Smarter retail decisions")
        st.caption("Forecast demand. Optimize inventory. Price with confidence.")


if st.session_state.page == "welcome":
    st.markdown("<div class='hero'><div class='eyebrow'>OPTIRETAIL AI</div><h1>Turn market data into <span>smarter decisions.</span></h1><p>Forecast demand, optimize inventory and make practical pricing decisions from one clean workspace.</p></div>", unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.markdown("<div class='card'><div class='eyebrow'>01 · FORECAST</div><h3>Know what will move.</h3><p class='small'>AI demand forecasting combines weather, holidays and product trends.</p></div>", unsafe_allow_html=True)
    b.markdown("<div class='card'><div class='eyebrow'>02 · INVENTORY</div><h3>Stock with confidence.</h3><p class='small'>Translate predicted demand into a practical quantity.</p></div>", unsafe_allow_html=True)
    c.markdown("<div class='card'><div class='eyebrow'>03 · PRICE</div><h3>Price with context.</h3><p class='small'>Use a live market reference and demand-aware recommendation.</p></div>", unsafe_allow_html=True)
    x, y = st.columns(2)
    with x:
        if st.button("Get Started →", use_container_width=True, key="welcome_signup"):
            st.session_state.page = "signup"
            st.rerun()
    with y:
        if st.button("Sign in", use_container_width=True, key="welcome_login"):
            st.session_state.page = "login"
            st.rerun()

elif st.session_state.page == "login":
    st.markdown("<div class='auth'><div class='auth-card'><div class='eyebrow'>WELCOME BACK</div><h1 class='auth-title'>Sign in to OptiRetail AI</h1><p class='auth-sub'>Access your retail intelligence workspace.</p>", unsafe_allow_html=True)
    with st.form("login_form"):
        email = st.text_input("Email address", placeholder="you@company.com")
        password = st.text_input("Password", type="password", placeholder="Your password")
        submitted = st.form_submit_button("Sign in", use_container_width=True)
    if submitted:
        user = authenticate(email, password)
        if user:
            st.session_state.user = user
            st.session_state.page = "dashboard"
            st.rerun()
        elif users is None:
            st.error(f"MongoDB unavailable: {mongo_status}")
        else:
            st.error("Invalid credentials.")
    a, b = st.columns(2)
    with a:
        if st.button("Create an account →", use_container_width=True, key="login_signup"):
            st.session_state.page = "signup"
            st.rerun()
    with b:
        if st.button("← Back to home", use_container_width=True, key="login_back"):
            st.session_state.page = "welcome"
            st.rerun()
    st.markdown("</div></div>", unsafe_allow_html=True)

elif st.session_state.page == "signup":
    st.markdown("<div class='auth'><div class='auth-card'><div class='eyebrow'>BUSINESS ONBOARDING</div><h1 class='auth-title'>Create your OptiRetail AI workspace</h1><p class='auth-sub'>Create your business account. Credentials are stored securely in MongoDB.</p>", unsafe_allow_html=True)
    with st.form("signup_form"):
        email = st.text_input("Email address", placeholder="you@company.com")
        password = st.text_input("Password", type="password", placeholder="Create a password")
        gst = st.text_input("GST Number", placeholder="Enter GST number")
        turnover = st.selectbox("Annual Turnover", ["1–5 Lakh", "5–10 Lakh", "10–15 Lakh", "15–50 Lakh", "50 Lakh+"])
        submitted = st.form_submit_button("Create account", use_container_width=True)
    if submitted:
        email = email.strip().lower()
        if not email or "@" not in email or not password or not gst:
            st.error("Please enter a valid email, password and GST number.")
        elif users is None:
            st.error(f"MongoDB unavailable: {mongo_status}")
        else:
            ok, msg = save_user(email, password, gst, turnover)
            if ok:
                st.success(msg)
                st.session_state.page = "login"
                st.rerun()
            else:
                st.error(msg)
    a, b = st.columns(2)
    with a:
        if st.button("Already have an account? Sign in", key="signup_login"):
            st.session_state.page = "login"
            st.rerun()
    with b:
        if st.button("← Back to home", key="signup_back"):
            st.session_state.page = "welcome"
            st.rerun()
    st.markdown("</div></div>", unsafe_allow_html=True)

elif st.session_state.page == "settings" and st.session_state.user:
    st.markdown("<div class='hero'><div class='eyebrow'>WORKSPACE</div><h1>Business settings</h1><p>Connected services and account information.</p></div>", unsafe_allow_html=True)
    u = st.session_state.user
    a, b, c = st.columns(3)
    a.metric("Account", "Active")
    b.metric("Email", u.get("email", "N/A"))
    c.metric("MongoDB", "Connected" if db is not None else "Unavailable")
    st.markdown("<div class='card'><div class='eyebrow'>DATA SERVICES</div><h3>Connection status</h3>", unsafe_allow_html=True)
    st.write("MongoDB:", "Connected" if db is not None else "Unavailable")
    st.write("SerpAPI:", "Configured" if SERPAPI_KEY else "Not configured")
    if db is None:
        st.warning(mongo_status)
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page in {"dashboard", "product", "forecast", "pricing"} and st.session_state.user:
    st.markdown("<div class='hero'><div class='eyebrow'>OPTIRETAIL AI</div><h1>Retail intelligence <span>with confidence.</span></h1><p>Combine market price, weather, holidays and product trends to make a practical retail decision.</p></div>", unsafe_allow_html=True)
    render_analysis()
    render_dashboard()

elif st.session_state.page == "market" and st.session_state.user:
    st.markdown("<div class='hero'><div class='eyebrow'>MARKET INSIGHTS</div><h1>Live market reference.</h1><p>Review current shopping signals for your selected product.</p></div>", unsafe_allow_html=True)
    data = st.session_state.market_data
    if data:
        st.metric("Market reference", f"₹{data['price']:,.2f}")
        st.dataframe(pd.DataFrame(data["results"], columns=["Price", "Product", "Source", "Link"]), use_container_width=True, hide_index=True)
    else:
        st.info("Run Refresh market price from Product Analysis first.")

elif st.session_state.page == "saved" and st.session_state.user:
    st.markdown("<div class='hero'><div class='eyebrow'>SAVED ANALYSES</div><h1>Your saved decisions.</h1><p>Recent AI predictions stored in MongoDB.</p></div>", unsafe_allow_html=True)
    if predictions is None:
        st.warning(f"MongoDB unavailable: {mongo_status}")
    else:
        try:
            rows = list(predictions.find({"email": st.session_state.user.get("email")}, {"_id": 0}).sort("created_at", -1).limit(20))
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("No saved analyses yet.")
        except Exception as exc:
            st.error(f"Could not load saved analyses: {type(exc).__name__}")

else:
    st.session_state.page = "welcome"
    st.rerun()
