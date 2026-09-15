import calendar
import datetime as dt
import hashlib

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
    from serpapi import GoogleSearch
except ImportError:
    GoogleSearch = None

st.set_page_config(
    page_title="OptiRetail AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def secret(name, default=""):
    try:
        value = st.secrets.get(name, default)
        return str(value).strip().strip('"').strip("'") if value else default
    except Exception:
        return default


SERPAPI_KEY = secret("SERPAPI_KEY")

DEFAULTS = {
    "user": None,
    "local_users": {},
    "page": "welcome",
    "product": "Wheat Flour",
    "city": "Jaipur",
    "month_name": dt.datetime.now().strftime("%B"),
    "last_prediction": None,
    "current_price": None,
    "market_data": None,
    "saved_analyses": [],
    "forecast_table": None,
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)

st.markdown(
    """
<style>
:root{--ink:#10231f;--muted:#64748b;--green:#079568;--green2:#056b4c;--line:#dcebe5}
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
.nav-active button{background:linear-gradient(135deg,#0aa875,#087d59)!important}
</style>
""",
    unsafe_allow_html=True,
)


def password_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def save_local_user(email, password, gst, turnover):
    email = email.strip().lower()
    if email in st.session_state.local_users:
        return False, "An account with this email already exists in this session."
    st.session_state.local_users[email] = {
        "email": email,
        "password_hash": password_hash(password),
        "gst": gst.strip(),
        "turnover": turnover,
    }
    return True, "Account created successfully."


def authenticate_local(email, password):
    user = st.session_state.local_users.get(email.strip().lower())
    if user and user["password_hash"] == password_hash(password):
        return user
    return None


def trend_score(product):
    rng = np.random.default_rng(sum(ord(c) for c in product))
    return int(rng.integers(35, 90))


@st.cache_data(ttl=900, show_spinner=False)
def fetch_price(product):
    if not product.strip() or not SERPAPI_KEY or GoogleSearch is None:
        return None
    try:
        result = GoogleSearch({
            "engine": "google_shopping_light", "q": f"{product} price", "gl": "in", "hl": "en",
            "num": 8, "api_key": SERPAPI_KEY,
        }).get_dict()
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


@st.cache_data(ttl=3600, show_spinner=False)
def geocode_city(city):
    try:
        response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=8,
        )
        response.raise_for_status()
        hits = response.json().get("results", [])
        if hits:
            return float(hits[0]["latitude"]), float(hits[0]["longitude"]), hits[0].get("name", city)
    except Exception:
        pass
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def monthly_weather(city):
    """Return city-specific monthly average temperatures, rather than repeating current weather."""
    coords = geocode_city(city)
    if not coords:
        return None
    lat, lon, resolved_name = coords
    end_year = dt.datetime.now().year - 1
    start_year = end_year - 4
    try:
        response = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": f"{start_year}-01-01",
                "end_date": f"{end_year}-12-31",
                "daily": "temperature_2m_mean",
                "timezone": "auto",
            },
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        dates = payload.get("daily", {}).get("time", [])
        temps = payload.get("daily", {}).get("temperature_2m_mean", [])
        frame = pd.DataFrame({"date": pd.to_datetime(dates), "temperature": temps}).dropna()
        if not frame.empty:
            frame["month"] = frame["date"].dt.month
            monthly = frame.groupby("month")["temperature"].mean().to_dict()
            values = {calendar.month_name[m]: round(float(monthly.get(m, 25.0)), 1) for m in range(1, 13)}
            return {"city": resolved_name, "lat": lat, "lon": lon, "values": values, "source": "5-year historical monthly average"}
    except Exception:
        pass
    return {"city": resolved_name, "lat": lat, "lon": lon, "values": fallback_monthly_weather(lat), "source": "seasonal fallback"}


def fallback_monthly_weather(latitude):
    base = 25.0
    amplitude = min(12.0, 7.0 + abs(latitude) * 0.12)
    # Peak in the northern hemisphere around May/June; opposite for southern locations.
    phase = 5 if latitude >= 0 else 11
    values = {}
    for month in range(1, 13):
        seasonal = amplitude * np.cos((month - phase) * 2 * np.pi / 12)
        values[calendar.month_name[month]] = round(base + seasonal, 1)
    return values


def holiday_count(year, month):
    india = holidays.India(years=year) if holidays else set()
    return sum(
        1 for day in range(1, calendar.monthrange(year, month)[1] + 1)
        if dt.date(year, month, day).weekday() >= 5 or dt.date(year, month, day) in india
    )


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


def build_forecast(product, city):
    months = list(calendar.month_name)[1:]
    weather = monthly_weather(city)
    if weather:
        temps = weather["values"]
        resolved_city = weather["city"]
        weather_source = weather["source"]
    else:
        temps = fallback_monthly_weather(20.0)
        resolved_city = city
        weather_source = "seasonal fallback"
    trend = trend_score(product)
    year = dt.datetime.now().year
    rows = []
    for month_num, month_name in enumerate(months, start=1):
        temp = float(temps[month_name])
        holiday = holiday_count(year, month_num)
        features = pd.DataFrame({"holiday_count": [holiday], "avg_temp": [temp], "viral_score": [trend]})
        demand = float(max(0, model.predict(features)[0]))
        rows.append({"Month": month_name, "Temperature (°C)": temp, "Holiday days": holiday, "Trend score": trend, "Forecast demand": round(demand)})
    return pd.DataFrame(rows), resolved_city, weather_source


def generate_decision(product, city, month_name):
    table, resolved_city, weather_source = build_forecast(product, city)
    row = table.loc[table["Month"] == month_name].iloc[0]
    market = st.session_state.current_price
    predicted = float(row["Forecast demand"])
    suggested = float(market * (1 + np.clip((predicted - 300) / 3000, -0.08, 0.08))) if isinstance(market, (int, float)) else None
    prediction = {
        "pred": predicted,
        "stock": float(np.ceil(predicted * 1.10)),
        "suggested": suggested,
        "temp": float(row["Temperature (°C)"]),
        "holiday": int(row["Holiday days"]),
        "trend": int(row["Trend score"]),
        "market": market,
        "product": product,
        "city": resolved_city,
        "month": month_name,
        "weather_source": weather_source,
    }
    st.session_state.forecast_table = table
    st.session_state.last_prediction = prediction
    st.session_state.saved_analyses.insert(0, prediction.copy())
    return prediction


def render_header(eyebrow, title, subtitle):
    st.markdown(
        f"<div class='hero'><div class='eyebrow'>{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></div>",
        unsafe_allow_html=True,
    )


def render_analysis_inputs():
    st.markdown("<div class='card'><div class='eyebrow'>ANALYSIS INPUTS</div><h3>Choose product, location and forecast month</h3>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.5, 1, 1])
    months = list(calendar.month_name)[1:]
    with c1:
        product = st.text_input("Product", value=st.session_state.product, placeholder="e.g. Amul Taaza Milk 1L", key="analysis_product")
    with c2:
        city = st.text_input("City", value=st.session_state.city, placeholder="Jaipur", key="analysis_city")
    with c3:
        month_name = st.selectbox("Forecast month", months, index=months.index(st.session_state.month_name), key="analysis_month")
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
                st.warning("No market price found. Add SERPAPI_KEY or use a more specific product name.")
    with b:
        if st.button("Generate AI Decision →", use_container_width=True, key="generate_decision"):
            prediction = generate_decision(product, city, month_name)
            st.success(f"Forecast generated for {month_name} using city-specific monthly weather.")
            st.session_state.last_prediction = prediction
    st.markdown("</div>", unsafe_allow_html=True)


def render_dashboard():
    lp = st.session_state.get("last_prediction") or {}
    current = st.session_state.get("current_price")
    suggested = lp.get("suggested")
    current_text = f"₹{current:,.2f}" if isinstance(current, (int, float)) else "—"
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"<div class='kpi'><small>Forecasted demand</small><strong>{int(lp.get('pred', 0)):,} units</strong><span>{lp.get('month', 'Selected month')}</span></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi'><small>Recommended stock</small><strong>{int(lp.get('stock', 0)):,} units</strong><span>10% planning buffer</span></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi'><small>Market reference</small><strong>{current_text}</strong><span>Shopping market signal</span></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi'><small>Suggested price</small><strong>{f'₹{suggested:,.2f}' if isinstance(suggested, (int, float)) else '—'}</strong><span>Demand-aware recommendation</span></div>", unsafe_allow_html=True)
    table = st.session_state.get("forecast_table")
    if table is None:
        st.info("Go to Product Analysis and generate an AI decision to populate the dashboard.")
        return
    left, right = st.columns([1.55, 1])
    with left:
        st.markdown("<div class='card'><div class='eyebrow'>DEMAND FORECAST</div><h3>12-month demand outlook</h3><p class='small'>Forecast changes month-by-month using monthly weather, holidays and product trend signals.</p>", unsafe_allow_html=True)
        chart = table.set_index("Month")[["Forecast demand"]]
        st.line_chart(chart, height=340)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='card'><div class='eyebrow'>AI DECISION</div><h3>Selected month signals</h3>", unsafe_allow_html=True)
        for name, value in [
            ("Product", lp.get("product", "—")), ("City", lp.get("city", "—")),
            ("Weather", f"{lp.get('temp', 25):.1f} °C"), ("Holiday days", str(lp.get("holiday", 0))),
            ("Trend score", f"{lp.get('trend', 0)}/100"), ("Weather data", lp.get("weather_source", "—")),
        ]:
            st.markdown(f"<div class='signal'><b>{name}</b><span>{value}</span></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        temp = float(lp.get("temp", 25))
        if temp >= 32:
            action = "Higher temperature detected — consider extra safety stock for heat-sensitive products."
        elif temp <= 18:
            action = "Cooler weather detected — keep inventory conservative for weather-sensitive categories."
        else:
            action = "Moderate weather; holiday and product-trend signals are driving the recommendation."
        st.markdown(f"<div class='decision'><h3>Recommended action</h3><p>{action}</p></div>", unsafe_allow_html=True)


def render_dashboard_page():
    render_header("DASHBOARD", "Turn market data into <span>smarter decisions.</span>", "A clean overview of your latest demand, inventory and pricing recommendation.")
    if not st.session_state.last_prediction:
        st.markdown("<div class='card'><h3>Welcome to your retail intelligence workspace.</h3><p class='small'>Start with Product Analysis to select a product and city, then generate the 12-month forecast.</p></div>", unsafe_allow_html=True)
    render_dashboard()


def render_product_page():
    render_header("PRODUCT ANALYSIS", "Analyze a product with <span>real market context.</span>", "Select a product and location, refresh the market reference, and generate a complete AI decision.")
    render_analysis_inputs()
    lp = st.session_state.last_prediction
    if lp:
        st.markdown(f"<div class='card'><div class='eyebrow'>LATEST RESULT</div><h3>{lp['product']} · {lp['city']} · {lp['month']}</h3><p class='small'>Forecast demand: <b>{int(lp['pred']):,} units</b> · Weather: <b>{lp['temp']:.1f} °C</b> · Holiday days: <b>{lp['holiday']}</b> · Weather source: <b>{lp['weather_source']}</b></p></div>", unsafe_allow_html=True)


def render_forecast_page():
    render_header("DEMAND FORECASTING", "See demand change <span>across the year.</span>", "Each month is calculated with its own weather estimate instead of repeating the current temperature.")
    if st.button("Generate / refresh 12-month forecast →", use_container_width=False, key="forecast_refresh"):
        prediction = generate_decision(st.session_state.product, st.session_state.city, st.session_state.month_name)
        st.session_state.last_prediction = prediction
        st.rerun()
    table = st.session_state.forecast_table
    if table is None:
        st.info("No forecast yet. Click the button above or generate a decision from Product Analysis.")
        return
    weather = monthly_weather(st.session_state.city)
    source = weather["source"] if weather else "seasonal fallback"
    st.markdown(f"<div class='card'><div class='eyebrow'>WEATHER MODEL</div><h3>{st.session_state.city}</h3><p class='small'>Monthly temperatures come from {source.lower()}. Values are different by month and feed directly into the demand model.</p>", unsafe_allow_html=True)
    weather_chart = table.set_index("Month")[["Temperature (°C)"]]
    st.line_chart(weather_chart, height=280)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'><div class='eyebrow'>FORECAST TABLE</div><h3>Monthly demand and weather</h3>", unsafe_allow_html=True)
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_pricing_page():
    render_header("DYNAMIC PRICING", "Price with <span>demand-aware context.</span>", "Use the market reference and forecast demand to produce a practical suggested price.")
    lp = st.session_state.last_prediction
    if not lp:
        st.info("Generate an AI decision first from Product Analysis.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Market reference", f"₹{lp['market']:,.2f}" if isinstance(lp.get('market'), (int, float)) else "Not available")
    c2.metric("Suggested price", f"₹{lp['suggested']:,.2f}" if isinstance(lp.get('suggested'), (int, float)) else "Not available")
    c3.metric("Forecast demand", f"{int(lp['pred']):,} units")
    st.markdown("<div class='decision'><h3>Pricing recommendation</h3><p>The suggested price is bounded around the live market reference and adjusted according to forecast demand. Refresh the market price for a current reference.</p></div>", unsafe_allow_html=True)


def render_market_page():
    render_header("MARKET INSIGHTS", "Live market <span>reference.</span>", "Review shopping results for the selected product and use them as the pricing input.")
    product = st.session_state.product
    if st.button("Refresh live market data →", key="market_refresh"):
        data = fetch_price(product)
        st.session_state.market_data = data
        st.session_state.current_price = data["price"] if data else None
        st.rerun()
    data = st.session_state.market_data
    if data:
        st.metric("Market reference", f"₹{data['price']:,.2f}")
        st.dataframe(pd.DataFrame(data["results"], columns=["Price", "Product", "Source", "Link"]), use_container_width=True, hide_index=True)
    else:
        st.info("No market data yet. Add SERPAPI_KEY and click Refresh live market data.")


def render_saved_page():
    render_header("SAVED ANALYSES", "Your saved <span>decisions.</span>", "Review predictions generated during this Streamlit session.")
    rows = st.session_state.saved_analyses
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No saved analyses yet. Generate an AI decision first.")


def render_settings_page():
    render_header("WORKSPACE", "Business <span>settings.</span>", "Account information and current service configuration.")
    u = st.session_state.user
    a, b, c = st.columns(3)
    a.metric("Account", "Active")
    b.metric("Email", u.get("email", "N/A"))
    c.metric("Database", "Temporarily disabled")
    st.markdown("<div class='card'><div class='eyebrow'>DATA SERVICES</div><h3>Connection status</h3>", unsafe_allow_html=True)
    st.info("MongoDB is intentionally disabled for the presentation build. Accounts and saved analyses are held in the current Streamlit session.")
    st.write("SerpAPI:", "Configured" if SERPAPI_KEY else "Not configured")
    st.markdown("</div>", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("<div style='font-size:1.5rem;font-weight:800'>📊 OptiRetail AI</div><div style='color:#a7f3d0;font-size:.7rem'>Analyze · Predict · Price</div>", unsafe_allow_html=True)
    st.divider()
    if st.session_state.user:
        st.caption(st.session_state.user.get("email", "Business User"))
        nav_items = [
            ("Dashboard", "dashboard"), ("Product Analysis", "product"), ("Demand Forecasting", "forecast"),
            ("Dynamic Pricing", "pricing"), ("Market Insights", "market"), ("Saved Analyses", "saved"), ("Settings", "settings"),
        ]
        for label, target in nav_items:
            if st.button(label, use_container_width=True, key=f"nav_{target}"):
                st.session_state.page = target
                st.rerun()
        st.divider()
        if st.button("Sign out", use_container_width=True, key="signout"):
            st.session_state.user = None
            st.session_state.page = "welcome"
            st.session_state.last_prediction = None
            st.session_state.forecast_table = None
            st.rerun()
    else:
        st.markdown("### Smarter retail decisions")
        st.caption("Forecast demand. Optimize inventory. Price with confidence.")


if st.session_state.page == "welcome":
    render_header("OPTIRETAIL AI", "Turn market data into <span>smarter decisions.</span>", "Forecast demand, optimize inventory and make practical pricing decisions from one clean workspace.")
    a, b, c = st.columns(3)
    a.markdown("<div class='card'><div class='eyebrow'>01 · FORECAST</div><h3>Know what will move.</h3><p class='small'>AI demand forecasting combines monthly weather, holidays and product trends.</p></div>", unsafe_allow_html=True)
    b.markdown("<div class='card'><div class='eyebrow'>02 · INVENTORY</div><h3>Stock with confidence.</h3><p class='small'>Translate predicted demand into a practical quantity.</p></div>", unsafe_allow_html=True)
    c.markdown("<div class='card'><div class='eyebrow'>03 · PRICE</div><h3>Price with context.</h3><p class='small'>Use a market reference and demand-aware recommendation.</p></div>", unsafe_allow_html=True)
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
        user = authenticate_local(email, password)
        if user:
            st.session_state.user = user
            st.session_state.page = "dashboard"
            st.rerun()
        else:
            st.error("Invalid credentials. Create an account first in this session.")
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
    st.markdown("<div class='auth'><div class='auth-card'><div class='eyebrow'>BUSINESS ONBOARDING</div><h1 class='auth-title'>Create your OptiRetail AI workspace</h1><p class='auth-sub'>Create a demo account for the presentation. Database persistence is temporarily disabled.</p>", unsafe_allow_html=True)
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
        else:
            ok, msg = save_local_user(email, password, gst, turnover)
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

elif st.session_state.user:
    if st.session_state.page == "dashboard":
        render_dashboard_page()
    elif st.session_state.page == "product":
        render_product_page()
    elif st.session_state.page == "forecast":
        render_forecast_page()
    elif st.session_state.page == "pricing":
        render_pricing_page()
    elif st.session_state.page == "market":
        render_market_page()
    elif st.session_state.page == "saved":
        render_saved_page()
    elif st.session_state.page == "settings":
        render_settings_page()
    else:
        st.session_state.page = "dashboard"
        st.rerun()
else:
    st.session_state.page = "welcome"
    st.rerun()
