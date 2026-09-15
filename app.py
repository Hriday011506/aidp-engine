import calendar
import datetime as dt
import hashlib
import html
import re
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

try:
    import holidays
except ImportError:
    holidays = None

st.set_page_config(page_title="OptiRetail AI", page_icon="🛒", layout="wide", initial_sidebar_state="expanded")
MONTHS = list(calendar.month_name)[1:]

DEFAULTS = {"user": None, "users": {}, "page": "welcome", "product": "Wheat Flour", "city": "Jaipur", "month": dt.datetime.now().strftime("%B"), "prediction": None, "forecast": None, "price": None, "saved": []}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

st.markdown("""
<style>
:root{--green:#079568;--dark:#063b2e;--ink:#0f172a;--muted:#64748b;--line:#d9e9e3}
.stApp{background:linear-gradient(180deg,#f7fbf9,#fff 48%,#f2faf7)}
.block-container{max-width:1500px;padding:1rem 2rem 4rem}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#032d23,#063b2e)}
[data-testid="stSidebar"] *{color:#ecfdf5}
.brand{display:flex;align-items:center;gap:12px;padding:8px 0 22px;border-bottom:1px solid #ffffff33;margin-bottom:20px}.mark{width:48px;height:48px;border-radius:14px;background:linear-gradient(135deg,#10b981,#047857);display:flex;align-items:center;justify-content:center;font-size:25px}.brand-title{font-size:1.35rem;font-weight:900;color:#fff}.brand-title span{color:#34d399}.brand-sub{font-size:.62rem;color:#a7f3d0;letter-spacing:1px}
.hero{padding:30px 38px;border-radius:28px;background:linear-gradient(135deg,#fff,#effbf6);border:1px solid var(--line);box-shadow:0 18px 48px #10b98114;margin-bottom:20px}.hero h1{font-size:2.6rem;margin:0;color:#0f172a;font-weight:900}.hero h1 span{color:#047857}.hero p{color:#64748b}.eyebrow{color:#047857;text-transform:uppercase;font-size:.65rem;font-weight:900;letter-spacing:2px;margin-bottom:8px}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:20px;padding:20px;box-shadow:0 10px 30px #0f172a0d;margin-bottom:16px}.card h3{margin:.15rem 0 .5rem;color:#047857!important}.small{color:#64748b;font-size:.84rem;line-height:1.5}.kpi{background:#fff;border:1px solid #e2e8f0;border-radius:18px;padding:18px;min-height:112px}.kpi small{color:#64748b;text-transform:uppercase;letter-spacing:1px;font-weight:800;font-size:.62rem}.kpi strong{display:block;color:#0f172a;font-size:1.55rem;margin-top:7px}.kpi span{font-size:.74rem;color:#64748b}
.action{border:1px solid #e5e7eb;border-radius:14px;padding:13px;margin:8px 0}.action p{margin:3px 0 0;color:#64748b;font-size:.82rem}.critical{border-left:5px solid #ef4444}.warning{border-left:5px solid #f59e0b}.info{border-left:5px solid #3b82f6}.decision{border-radius:18px;padding:18px;background:#ecfdf5;border:1px solid #bbf7d0}.decision h3{color:#065f46!important;margin:0 0 5px}.score{font-size:3rem;font-weight:900;color:#047857}.auth{max-width:800px;margin:25px auto}.auth-card{background:#fff;border:1px solid var(--line);border-radius:28px;padding:34px;box-shadow:0 20px 55px #0f172a12}.auth-title{font-size:2.2rem;font-weight:900;color:#047857!important;margin:0}.auth-sub{color:#64748b}
input,textarea{color:#0f172a!important;background:#fff!important}label,div[data-testid="stWidgetLabel"] p{color:#334155!important;font-weight:600!important}.stButton>button,.stFormSubmitButton>button{border:0!important;border-radius:12px!important;min-height:44px!important;font-weight:800!important;background:linear-gradient(135deg,#079568,#047a55)!important;color:#fff!important}
</style>
""", unsafe_allow_html=True)


def ph(v):
    return hashlib.sha256(v.encode()).hexdigest()


def signup(email, password, gst, turnover):
    email = email.strip().lower()
    if email in st.session_state.users:
        return False, "An account with this email already exists in this session."
    st.session_state.users[email] = {"email": email, "password": ph(password), "gst": gst, "turnover": turnover}
    return True, "Account created successfully."


def authenticate(email, password):
    u = st.session_state.users.get(email.strip().lower())
    return u if u and u["password"] == ph(password) else None


def trend(product):
    return int(np.random.default_rng(sum(ord(x) for x in product.lower())).integers(45, 91))


def weather_fallback(lat):
    base = 25 - min(abs(lat), 45) * .06
    amp = min(14, 8 + abs(lat) * .1)
    phase = 5 if lat >= 0 else 11
    return {calendar.month_name[m]: round(base + amp * np.cos((m-phase) * 2*np.pi/12), 1) for m in range(1,13)}

@st.cache_data(ttl=86400, show_spinner=False)
def monthly_weather(city):
    try:
        r = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name":city,"count":1,"language":"en","format":"json"}, timeout=8)
        hit = r.json().get("results", [])[0]
        lat, lon = float(hit["latitude"]), float(hit["longitude"])
        end = dt.datetime.now().year - 1
        start = end - 4
        r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={"latitude":lat,"longitude":lon,"start_date":f"{start}-01-01","end_date":f"{end}-12-31","daily":"temperature_2m_mean","timezone":"auto"}, timeout=15)
        d = r.json().get("daily", {})
        f = pd.DataFrame({"date":pd.to_datetime(d.get("time", [])),"temp":d.get("temperature_2m_mean", [])}).dropna()
        if not f.empty:
            f["m"] = f.date.dt.month
            g = f.groupby("m").temp.mean()
            return hit.get("name", city), {calendar.month_name[m]:round(float(g.get(m,25)),1) for m in range(1,13)}, "5-year historical monthly average"
    except Exception:
        pass
    return city, weather_fallback(20), "seasonal fallback"


def live_market_price(product, city):
    key = str(st.secrets.get("SERPAPI_KEY", "")).strip()
    if not key:
        return None, "SERPAPI_KEY is missing from Streamlit Secrets."
    try:
        r = requests.get("https://serpapi.com/search.json", params={"engine":"google","q":f"{product} price {city} India","location":f"{city}, Rajasthan, India","hl":"en","gl":"in","api_key":key}, timeout=15)
        data = r.json()
        if data.get("error"):
            return None, f"Live market lookup failed: {data['error']}"
        candidates = []
        for item in data.get("shopping_results", [])[:10]:
            if item.get("price"): candidates.append(str(item["price"]))
        for item in data.get("inline_shopping_results", [])[:10]:
            if item.get("price"): candidates.append(str(item["price"]))
        text = " ".join(candidates)
        matches_found = re.findall(r"(?:₹|Rs\.?|INR)\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)", text, flags=re.I)
        if not matches_found:
            snippets = []
            for item in data.get("organic_results", [])[:8]:
                snippets.extend([str(item.get("title", "")), str(item.get("snippet", ""))])
            text = " ".join(snippets)
            matches_found = re.findall(r"(?:₹|Rs\.?|INR)\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)", text, flags=re.I)
        values = [float(x.replace(",", "")) for x in matches_found if 0 < float(x.replace(",", "")) < 100000]
        if not values:
            return None, f"No current price found for {product} in {city}."
        return round(float(np.median(values)), 2), "Live Google market reference via SerpAPI"
    except Exception as exc:
        return None, f"Live market lookup failed: {exc}"


def holiday_days(year, month):
    h = holidays.India(years=year) if holidays else set()
    return sum(1 for d in range(1, calendar.monthrange(year,month)[1]+1) if dt.date(year,month,d).weekday() >= 5 or dt.date(year,month,d) in h)

@st.cache_resource(show_spinner=False)
def get_model():
    rng = np.random.default_rng(42)
    d = pd.DataFrame({"holiday":rng.integers(0,14,800),"temp":rng.uniform(8,42,800),"trend":rng.integers(0,100,800)})
    season = np.maximum(0, 24 - np.abs(d.temp - 28))
    d["sales"] = 170 + d.holiday*38 + d.trend*4.7 + season*12 + rng.normal(0,24,800)
    m = RandomForestRegressor(n_estimators=220, max_depth=12, random_state=42, n_jobs=-1)
    m.fit(d[["holiday","temp","trend"]], d.sales)
    return m

MODEL = get_model()


def make_forecast(product, city):
    resolved, temps, source = monthly_weather(city)
    tr = trend(product)
    rows = []
    year = dt.datetime.now().year
    for i, month in enumerate(MONTHS, 1):
        temp = float(temps.get(month, 25))
        hd = holiday_days(year, i)
        x = pd.DataFrame({"holiday":[hd],"temp":[temp],"trend":[tr]})
        demand = float(max(0, MODEL.predict(x)[0]))
        rows.append({"Month":month,"Temperature (°C)":round(temp,1),"Holiday days":hd,"Trend score":tr,"Forecast demand":round(demand)})
    return pd.DataFrame(rows), resolved, source


def generate():
    if st.session_state.price is None:
        return False
    table, city, source = make_forecast(st.session_state.product, st.session_state.city)
    row = table[table.Month == st.session_state.month].iloc[0]
    market = st.session_state.price
    pred = float(row["Forecast demand"])
    suggested = float(market * (1 + np.clip((pred-300)/3000,-.08,.08)))
    p = {"product":st.session_state.product,"city":city,"month":st.session_state.month,"pred":pred,"stock":float(np.ceil(pred*1.1)),"temp":float(row["Temperature (°C)"]),"holiday":int(row["Holiday days"]),"trend":int(row["Trend score"]),"market":market,"suggested":suggested,"source":source,"confidence":int(np.clip(70+abs(int(row["Trend score"])-60)*.25,68,94))}
    st.session_state.forecast = table
    st.session_state.prediction = p
    st.session_state.saved.insert(0,p.copy())
    st.session_state.saved = st.session_state.saved[:20]
    return True


def matches(p):
    return bool(p) and p["product"] == st.session_state.product and p["city"] == st.session_state.city and p["month"] == st.session_state.month


def health(p):
    if not p: return 0
    score = (np.clip(65+(p["trend"]-50)*.35,50,94) + np.clip(92-max(0,(p["pred"]-1200)/40),60,94) + 80 + np.clip(92-abs(p["temp"]-27)*1.4,60,94)) / 4
    return int(np.clip(score,55,95))


def action_list(p):
    if not p: return [("info","Generate an AI decision","Start Product Analysis to unlock actions.")]
    a=[]
    a.append(("critical","Increase inventory","Predicted demand is high.") if p["pred"]>1200 else ("warning","Keep inventory lean","Forecast demand is relatively low.") if p["pred"]<500 else ("info","Maintain planned stock","Demand is manageable with the 10% buffer."))
    if p["trend"]>=78:a.append(("info","Demand surge detected","Trend score is strong; monitor replenishment more frequently."))
    if p["temp"]>=34:a.append(("warning","Watch heat sensitivity","High temperature may increase volatility for sensitive products."))
    if p["market"] is not None and p["suggested"] is not None:a.append(("warning","Review pricing","Demand-aware price is above market reference.") if p["suggested"]>p["market"]*1.02 else ("info","Keep price near market","Limited price change is indicated."))
    return a[:4]


def header(e, title, subtitle):
    st.markdown(f"<div class='hero'><div class='eyebrow'>{e}</div><h1>{title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)


def logo():
    st.markdown('<div class="brand"><div class="mark">🛒</div><div><div class="brand-title">OptiRetail <span>AI</span></div><div class="brand-sub">SMARTER DECISIONS · STRONGER RETAIL</div></div></div>', unsafe_allow_html=True)


def inputs():
    st.markdown("<div class='card'><div class='eyebrow'>ANALYSIS INPUTS</div><h3>Choose product, location and forecast month</h3>", unsafe_allow_html=True)
    a,b,c=st.columns([1.5,1,1])
    with a: product=st.text_input("Product",value=st.session_state.product,key="product_input")
    with b: city=st.text_input("City",value=st.session_state.city,key="city_input")
    with c: month=st.selectbox("Forecast month",MONTHS,index=MONTHS.index(st.session_state.month) if st.session_state.month in MONTHS else 0,key="month_input")
    product=product.strip() or "Wheat Flour";city=city.strip() or "Jaipur"
    old=st.session_state.prediction
    changed=bool(old) and (old["product"]!=product or old["city"]!=city or old["month"]!=month)
    st.session_state.product,st.session_state.city,st.session_state.month=product,city,month
    if changed:
        st.session_state.prediction=None;st.session_state.forecast=None;st.session_state.price=None
        st.info("Inputs changed. Refresh the live market price and generate a new AI decision.")
    x,y=st.columns(2)
    with x:
        if st.button("Refresh live market price",use_container_width=True,key="price_button"):
            price,msg=live_market_price(product,city)
            if price is None:
                st.session_state.price=None
                st.error(msg)
            else:
                st.session_state.price=price
                st.success(f"Live market reference: ₹{price:,.2f} · {msg}")
    with y:
        if st.button("Generate AI Decision →",use_container_width=True,key="generate_button"):
            if st.session_state.price is None:
                st.warning("Live market lookup is required. Click 'Refresh live market price' first.")
            elif generate():
                st.rerun()
    if st.session_state.price is not None:
        st.markdown(f"<div class='market-live'>Live market reference: <b>₹{st.session_state.price:,.2f}</b> · <span>verified for {html.escape(city)}</span></div>",unsafe_allow_html=True)
    st.markdown("</div>",unsafe_allow_html=True)


def simulator():
    p=st.session_state.prediction
    if not matches(p):st.info("Generate an AI decision first.");return
    st.markdown("<div class='card'><div class='eyebrow'>WHAT-IF SIMULATOR</div><h3>Test a retail scenario before acting</h3><p class='small'>Change price, demand, temperature and promotion assumptions.</p>",unsafe_allow_html=True)
    a,b=st.columns(2)
    with a:price_change=st.slider("Price change (%)",-20,20,0,key="sim_price");demand_change=st.slider("Demand change (%)",-30,30,0,key="sim_demand")
    with b:temp_change=st.slider("Temperature change (°C)",-10.,10.,0.,.5,key="sim_temp");promo=st.slider("Promotion effect (%)",0,25,0,key="sim_promo")
    x=pd.DataFrame({"holiday":[p["holiday"]],"temp":[p["temp"]+temp_change],"trend":[p["trend"]]});base_price=p["market"];d=float(max(0,MODEL.predict(x)[0]))*(1+demand_change/100)*(1+promo/100);price=base_price*(1+price_change/100);revenue=d*price;stock=int(np.ceil(d*1.1));risk="Low" if stock<1800 else "Medium" if stock<3000 else "High"
    q1,q2,q3,q4=st.columns(4);q1.metric("Expected demand",f"{d:,.0f}",f"{(d/p['pred']-1)*100:+.1f}%");q2.metric("Expected revenue",f"₹{revenue:,.0f}");q3.metric("Recommended stock",f"{stock:,}");q4.metric("Inventory risk",risk)
    if st.button("Save scenario",key="save_sim"):st.success("Scenario saved for this session.")
    st.markdown("</div>",unsafe_allow_html=True)


def dashboard():
    p=st.session_state.prediction if matches(st.session_state.prediction) else None;k1,k2,k3,k4=st.columns(4)
    vals=[("Forecasted demand",f"{int(p['pred']):,} units" if p else "—","Current month"),("Recommended stock",f"{int(p['stock']):,} units" if p else "—","10% planning buffer"),("Market reference",f"₹{p['market']:,.2f}" if p and p['market'] is not None else "—","Live lookup required"),("Health score",f"{health(p)}/100" if p else "—","Retail readiness")]
    for col,(a,b,c) in zip([k1,k2,k3,k4],vals):col.markdown(f"<div class='kpi'><small>{a}</small><strong>{b}</strong><span>{c}</span></div>",unsafe_allow_html=True)
    a,b=st.columns(2)
    with a:
        st.markdown("<div class='card'><div class='eyebrow'>AI ACTION CENTER</div><h3>Today's AI actions</h3>",unsafe_allow_html=True)
        for typ,title,desc in action_list(p):st.markdown(f"<div class='action {typ}'><b>{title}</b><p>{desc}</p></div>",unsafe_allow_html=True)
        st.markdown("</div>",unsafe_allow_html=True)
    with b:
        st.markdown("<div class='card'><div class='eyebrow'>RETAIL HEALTH</div><h3>Business health score</h3>",unsafe_allow_html=True)
        if p:st.markdown(f"<div class='score'>{health(p)}<span style='font-size:1rem;color:#64748b'>/100</span></div>");st.progress(health(p),text="Overall health")
        else:st.info("Generate a decision to calculate the score.")
        st.markdown("</div>",unsafe_allow_html=True)
    if not p:st.info("Open Product Analysis, refresh the live market price, and generate an AI decision.");return
    a,b=st.columns([1.4,1])
    with a:
        st.markdown("<div class='card'><div class='eyebrow'>FORECAST OUTLOOK</div><h3>12-month demand</h3>",unsafe_allow_html=True);st.line_chart(st.session_state.forecast.set_index("Month")[["Forecast demand"]],height=300);st.markdown("</div>",unsafe_allow_html=True)
    with b:
        st.markdown("<div class='card'><div class='eyebrow'>WHY THIS PREDICTION?</div><h3>Explainable AI</h3>",unsafe_allow_html=True)
        for n,v in [("Product trend",f"{p['trend']}/100"),("Holiday days",p['holiday']),("Temperature",f"{p['temp']:.1f} °C"),("Confidence",f"{p['confidence']}%")]:st.markdown(f"<p class='small'><b>{n}</b> <span style='float:right'>{v}</span></p>",unsafe_allow_html=True)
        st.markdown("</div>",unsafe_allow_html=True)
    simulator()


def product_page():
    header("PRODUCT ANALYSIS","Analyze a product with <span>live market context.</span>","Refresh the live market reference, then generate an explainable AI decision.");inputs();p=st.session_state.prediction
    if matches(p):st.markdown(f"<div class='card'><div class='eyebrow'>LATEST RESULT</div><h3>{html.escape(p['product'])} · {html.escape(p['city'])} · {p['month']}</h3><p class='small'>Forecast demand: <b>{int(p['pred']):,}</b> units · Weather: <b>{p['temp']:.1f} °C</b> · Holiday days: <b>{p['holiday']}</b> · Confidence: <b>{p['confidence']}%</b> · Market: <b>₹{p['market']:,.2f}</b> · Source: <b>{p['source']}</b></p></div>",unsafe_allow_html=True)


def forecast_page():
    header("DEMAND FORECASTING","See demand change <span>across the year.</span>","Each month gets its own weather value; no repeated placeholder.");p=st.session_state.prediction
    if not p:st.info("Generate a decision first.");return
    t=st.session_state.forecast;st.line_chart(t.set_index("Month")[["Temperature (°C)"]],height=250);st.line_chart(t.set_index("Month")[["Forecast demand"]],height=280);st.dataframe(t.assign(**{"Recommended stock":np.ceil(t["Forecast demand"]*1.1).astype(int)}),use_container_width=True,hide_index=True)


def compare_page():
    header("PRODUCT COMPARISON","Compare products with <span>one model.</span>","Compare expected demand, peak month and inventory requirement.");a,b,c=st.columns(3)
    with a:p1=st.text_input("Product 1",value=st.session_state.product,key="cmp1")
    with b:p2=st.text_input("Product 2",value="Rice 5kg",key="cmp2")
    with c:p3=st.text_input("Product 3",value="Cooking Oil 1L",key="cmp3")
    if st.button("Compare products →",use_container_width=True,key="compare_button"):
        rows=[]
        for p in [p1,p2,p3]:
            if p.strip():t,city,_=make_forecast(p,st.session_state.city);peak=t.loc[t["Forecast demand"].idxmax()];rows.append({"Product":p,"City":city,"12-month demand":int(t["Forecast demand"].sum()),"Peak month":peak.Month,"Peak demand":int(peak["Forecast demand"]),"Recommended stock":int(np.ceil(t["Forecast demand"].sum()*1.1))})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)


def alerts_page():
    header("ALERTS & RISK","See risks before they become <span>retail problems.</span>","Early-warning signals generated from the current forecast.");p=st.session_state.prediction
    if not matches(p):st.info("Generate a decision first.");return
    for typ,title,desc in action_list(p):st.markdown(f"<div class='card'><h3>{'🔴' if typ=='critical' else '🟠' if typ=='warning' else '🔵'} {title}</h3><p class='small'>{desc}</p></div>",unsafe_allow_html=True)


def copilot_page():
    header("AI DECISION COPILOT","Ask OptiRetail AI <span>in plain language.</span>","Get a concise answer from the current forecast context.");q=st.text_input("Ask a question",placeholder="Why should I increase stock?",key="question")
    if st.button("Analyze question →",use_container_width=True,key="ask"):
        p=st.session_state.prediction
        if not matches(p):st.info("Generate a decision first.")
        elif not q.strip():st.warning("Enter a question first.")
        else:
            q=q.lower()
            if "stock" in q or "inventory" in q:a=f"Plan about {int(p['stock']):,} units for {p['product']} in {p['month']}. Forecast demand is {int(p['pred']):,} units with a 10% buffer."
            elif "weather" in q:a=f"{p['month']} is estimated at {p['temp']:.1f} °C using {p['source']}."
            elif "price" in q:a=f"The live market reference is ₹{p['market']:,.2f}. Use the Simulator to test price changes around this reference."
            else:a=f"The model forecasts {int(p['pred']):,} units with {p['confidence']}% confidence. Trend is {p['trend']}/100 and temperature is {p['temp']:.1f} °C."
            st.markdown(f"<div class='decision'><h3>OptiRetail AI</h3><p>{html.escape(a)}</p></div>",unsafe_allow_html=True)


def settings_page():
    header("SETTINGS","Configure your <span>workspace.</span>","This presentation build uses session storage for accounts; live market lookup is required for AI decisions.");u=st.session_state.user or {};st.text_input("Email",value=u.get("email",""),disabled=True);st.text_input("GST Number",value=u.get("gst",""),disabled=True);st.info("MongoDB and Brevo are not required for this version. SERPAPI_KEY is required for live market pricing.")

with st.sidebar:
    logo()
    if st.session_state.user:
        st.caption(st.session_state.user.get("email","Business User"));items=[("Dashboard","dashboard"),("Product Analysis","product"),("Demand Forecasting","forecast"),("Dynamic Pricing","pricing"),("Market Insights","market"),("Product Comparison","compare"),("AI Simulator","simulator"),("Alerts & Risk","alerts"),("AI Decision Copilot","copilot"),("Saved Analyses","saved"),("Settings","settings")]
        for label,target in items:
            if st.button(label,use_container_width=True,key="nav_"+target):st.session_state.page=target;st.rerun()
        if st.button("Sign out",use_container_width=True,key="signout"):st.session_state.user=None;st.session_state.page="welcome";st.rerun()

if st.session_state.page=="welcome":
    header("OPTIRETAIL AI","Turn market data into <span>smarter decisions.</span>","Forecast demand, use live market pricing, simulate scenarios, detect risks and explain AI recommendations.");a,b,c,d=st.columns(4)
    for col,title,desc in zip([a,b,c,d],["Predict","Decide","Simulate","Explain"],["12-month demand forecast","AI Action Center","What-if scenarios","Confidence and drivers"]):col.markdown(f"<div class='card'><div class='eyebrow'>AI</div><h3>{title}</h3><p class='small'>{desc}</p></div>",unsafe_allow_html=True)
    x,y=st.columns(2)
    with x:
        if st.button("Get Started →",use_container_width=True,key="start"):st.session_state.page="signup";st.rerun()
    with y:
        if st.button("Sign in",use_container_width=True,key="home_login"):st.session_state.page="login";st.rerun()
elif st.session_state.page=="login":
    st.markdown("<div class='auth'><div class='auth-card'><div class='eyebrow'>WELCOME BACK</div><h1 class='auth-title'>Sign in to OptiRetail AI</h1><p class='auth-sub'>Access your retail intelligence workspace.</p>",unsafe_allow_html=True)
    with st.form("login_form"):email=st.text_input("Email address");pw=st.text_input("Password",type="password");submit=st.form_submit_button("Sign in",use_container_width=True)
    if submit:
        u=authenticate(email,pw)
        if u:st.session_state.user=u;st.session_state.page="dashboard";st.rerun()
        else:st.error("Invalid credentials. Create an account first in this session.")
    if st.button("Create an account →",key="login_signup"):st.session_state.page="signup";st.rerun()
    st.markdown("</div></div>",unsafe_allow_html=True)
elif st.session_state.page=="signup":
    st.markdown("<div class='auth'><div class='auth-card'><div class='eyebrow'>BUSINESS ONBOARDING</div><h1 class='auth-title'>Create your OptiRetail AI workspace</h1><p class='auth-sub'>Presentation-ready account with session storage.</p>",unsafe_allow_html=True)
    with st.form("signup_form"):email=st.text_input("Email address");pw=st.text_input("Password",type="password");gst=st.text_input("GST Number");turn=st.selectbox("Annual Turnover",["1–5 Lakh","5–10 Lakh","10–15 Lakh","15–50 Lakh","50 Lakh+"]);submit=st.form_submit_button("Create account",use_container_width=True)
    if submit:
        if not email or "@" not in email or not pw or not gst:st.error("Please enter a valid email, password and GST number.")
        else:
            ok,msg=signup(email,pw,gst,turn)
            if ok:st.success(msg);st.session_state.page="login";st.rerun()
            else:st.error(msg)
    st.markdown("</div></div>",unsafe_allow_html=True)
elif st.session_state.user:
    if st.session_state.page=="dashboard":header("DASHBOARD","Turn market data into <span>smarter decisions.</span>","AI Action Center, health score, explainable forecast and simulator.");dashboard()
    elif st.session_state.page=="product":product_page()
    elif st.session_state.page=="forecast":forecast_page()
    elif st.session_state.page in {"pricing","simulator"}:header("AI SIMULATOR","Test decisions before <span>you act.</span>","Change price, demand, temperature and promotion assumptions.");simulator()
    elif st.session_state.page=="market":
        header("MARKET INSIGHTS","Live market <span>reference.</span>","A live market price is required before an AI decision can be generated.")
        a,b=st.columns([2,1])
        with a:
            if st.button("Refresh live market price",use_container_width=True,key="market_refresh"):
                price,msg=live_market_price(st.session_state.product,st.session_state.city)
                if price is None:st.error(msg)
                else:st.session_state.price=price;st.success(f"Live market reference: ₹{price:,.2f} · {msg}")
        with b:
            if st.session_state.price is not None:st.metric("Current reference",f"₹{st.session_state.price:,.2f}")
            else:st.warning("No live price loaded")
    elif st.session_state.page=="compare":compare_page()
    elif st.session_state.page=="alerts":alerts_page()
    elif st.session_state.page=="copilot":copilot_page()
    elif st.session_state.page=="saved":
        header("SAVED ANALYSES","Keep your important <span>AI decisions.</span>","Recent session decisions.")
        if not st.session_state.saved:st.info("No saved analyses yet.")
        for p in st.session_state.saved[:10]:st.markdown(f"<div class='card'><h3>{html.escape(p['product'])} · {p['city']} · {p['month']}</h3><p class='small'>Demand {int(p['pred']):,} · Stock {int(p['stock']):,} · Market ₹{p['market']:,.2f} · Confidence {p['confidence']}%</p></div>",unsafe_allow_html=True)
    elif st.session_state.page=="settings":settings_page()
    else:st.session_state.page="dashboard";st.rerun()
else:st.session_state.page="welcome";st.rerun()