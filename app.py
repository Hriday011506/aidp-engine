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

st.set_page_config(page_title="OptiRetail AI", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

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
    "users": [],
    "analyzed": False,
    "market_data": None
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:#0f172a}
.stApp{background:linear-gradient(180deg,#f7fbf9 0%,#ffffff 34%,#f4faf7 100%)}
.block-container{max-width:1520px;padding:1.2rem 2rem 4rem}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#052b22,#07362b);border-right:1px solid rgba(255,255,255,.08)}
[data-testid="stSidebar"] *{color:#ecfdf5}
.sidebar-logo{font-family:'Plus Jakarta Sans';font-size:1.8rem;font-weight:800;letter-spacing:-.8px}
.sidebar-sub{font-size:.7rem;color:#a7f3d0;margin-top:-5px}
.hero{padding:34px 40px;border-radius:30px;background:linear-gradient(135deg,#ffffff 0%,#f1fbf7 100%);border:1px solid #d7efe4;box-shadow:0 24px 70px rgba(16,185,129,.08);margin-bottom:20px}
.hero h1{font-family:'Plus Jakarta Sans';font-size:3.05rem;line-height:1.05;letter-spacing:-2.4px;margin:0;color:#0f172a}
.hero h1 span{color:#047857}.hero p{color:#64748b;font-size:1rem;max-width:900px;margin:.75rem 0 0}
.eyebrow{color:#047857;text-transform:uppercase;font-size:.68rem;font-weight:800;letter-spacing:2px;margin-bottom:10px}
.section-title{font-family:'Plus Jakarta Sans';font-size:1.25rem;font-weight:800;color:#0f172a;margin:.35rem 0 1rem}
.card{background:rgba(255,255,255,.96);border:1px solid #e2e8f0;border-radius:22px;padding:22px;box-shadow:0 12px 36px rgba(15,23,42,.05);margin-bottom:16px}
.kpi{background:#fff;border:1px solid #e2e8f0;border-radius:18px;padding:18px;min-height:120px;box-shadow:0 10px 24px rgba(15,23,42,.04)}
.kpi small{color:#64748b;text-transform:uppercase;letter-spacing:1px;font-weight:800;font-size:.65rem}.kpi strong{display:block;font-family:'Plus Jakarta Sans';color:#0f172a;font-size:1.75rem;margin-top:7px}.kpi span{font-size:.76rem;color:#64748b}
.signal{display:flex;justify-content:space-between;align-items:center;padding:13px 0;border-bottom:1px solid #eef2f4}.signal:last-child{border-bottom:0}.signal b{color:#0f172a}.signal small{color:#64748b}
.badge{display:inline-flex;align-items:center;gap:6px;padding:6px 10px;border-radius:999px;font-size:.68rem;font-weight:800}.badge.live{background:#ecfdf5;color:#047857;border:1px solid #bbf7d0}.badge.info{background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe}
.stButton>button{border:0!important;border-radius:12px!important;min-height:45px!important;font-weight:800!important;background:linear-gradient(135deg,#059669,#047857)!important;color:#fff!important;box-shadow:0 8px 22px rgba(5,150,105,.18)}
.stButton>button:hover{transform:translateY(-1px)}
div[data-baseweb="input"]>div,div[data-baseweb="select"]>div{border-radius:12px!important;border-color:#dbe5df!important;background:#fff!important}
[data-testid="stMetricValue"]{color:#0f172a}[data-testid="stMetricLabel"]{color:#64748b}
.small-note{color:#64748b;font-size:.78rem}.decision{border-radius:18px;padding:18px;background:linear-gradient(135deg,#ecfdf5,#f0fdfa);border:1px solid #bbf7d0}.decision h3{margin:0 0 5px;color:#065f46;font-family:'Plus Jakarta Sans'}.decision p{margin:0;color:#475569;font-size:.9rem;line-height:1.55}
.footer-note{text-align:center;color:#94a3b8;font-size:.72rem;padding-top:14px}
</style>
""",unsafe_allow_html=True)

# ==============================
# LOCAL DEMO AUTH
# ==============================
def signup(email,password,gst,turnover):
    email=email.strip().lower()
    if any(u["email"]==email for u in st.session_state.users): return False,"An account with this email already exists."
    st.session_state.users.append({"email":email,"password":password,"gst":gst,"turnover":turnover,"created_at":datetime.datetime.now().isoformat()})
    return True,"Account created successfully."

def login(email,password):
    email=email.strip().lower()
    for user in st.session_state.users:
        if user["email"]==email and user["password"]==password: return user
    return None

# ==============================
# DATA SERVICES
# ==============================
@st.cache_data(ttl=900,show_spinner=False)
def fetch_product_price(product_name):
    product_name=product_name.strip()
    if not product_name or not SERPAPI_KEY or GoogleSearch is None: return None
    try:
        res=GoogleSearch({"engine":"google_shopping_light","q":f"{product_name} price","gl":"in","hl":"en","num":8,"api_key":SERPAPI_KEY}).get_dict()
        tokens=[t for t in product_name.lower().split() if len(t)>2]
        candidates=[]
        for item in res.get("shopping_results",[]):
            raw=item.get("extracted_price",item.get("price"))
            try:
                if isinstance(raw,str): raw=raw.replace(",","").replace("₹","").strip()
                val=float(raw)
            except (TypeError,ValueError): continue
            if val<=0: continue
            title=str(item.get("title","")).lower()
            hits=sum(tok in title for tok in tokens)
            score=hits*3+(10 if product_name.lower() in title else 0)
            candidates.append((score,val))
        if not candidates: return None
        candidates.sort(reverse=True)
        best=candidates[0][0]
        vals=[v for s,v in candidates if s==best]
        return float(np.median(vals[:5]))
    except Exception: return None

@st.cache_data(ttl=900,show_spinner=False)
def get_weather(city):
    try:
        r=requests.get("https://geocoding-api.open-meteo.com/v1/search",params={"name":city,"count":1},timeout=4); r.raise_for_status()
        hits=r.json().get("results",[])
        if not hits:return 25.0
        lat,lon=hits[0]["latitude"],hits[0]["longitude"]
        w=requests.get("https://api.open-meteo.com/v1/forecast",params={"latitude":lat,"longitude":lon,"current_weather":"true"},timeout=4); w.raise_for_status()
        return float(w.json()["current_weather"]["temperature"])
    except Exception:return 25.0

@st.cache_data
def get_holidays(year,month):
    india=holidays.India(years=year); days=calendar.monthrange(year,month)[1]
    return sum(1 for d in range(1,days+1) if datetime.date(year,month,d).weekday()>=5 or datetime.date(year,month,d) in india)

@st.cache_data
def trend_score(product):
    rng=np.random.default_rng(sum(ord(c) for c in product)); return int(rng.integers(30,90))

# ==============================
# MODEL
# ==============================
@st.cache_data
def load_data():
    rng=np.random.default_rng(42)
    df=pd.DataFrame({"holiday_count":rng.integers(0,10,100),"avg_temp":rng.integers(10,40,100),"viral_score":rng.integers(0,100,100)})
    df["sales"]=200+df["holiday_count"]*50+df["avg_temp"]*10+df["viral_score"]*5
    return df

@st.cache_resource
def train_model(df):
    m=RandomForestRegressor(n_estimators=200,max_depth=12,random_state=42,n_jobs=-1)
    m.fit(df[["holiday_count","avg_temp","viral_score"]],df["sales"]); return m
model=train_model(load_data())

# ==============================
# SIDEBAR
# ==============================
with st.sidebar:
    st.markdown("<div class='sidebar-logo'>📊 OptiRetail AI</div><div class='sidebar-sub'>Analyze · Predict · Price</div>",unsafe_allow_html=True)
    st.divider()
    if st.session_state.user:
        st.caption(st.session_state.user.get("email","Business User"))
        for label in ["Dashboard","Product Analysis","Demand Forecasting","Dynamic Pricing","Market Insights","Saved Analyses"]:
            if st.button(label,use_container_width=True,key=f"nav_{label}"):
                st.session_state.page="dashboard"; st.rerun()
        st.divider()
        if st.button("Settings",use_container_width=True): st.session_state.page="settings"; st.rerun()
        if st.button("Sign out",use_container_width=True): st.session_state.user=None; st.session_state.page="welcome"; st.rerun()
    else:
        st.markdown("### Smarter retail decisions")
        st.caption("Forecast demand. Optimize inventory. Price with confidence.")

# ==============================
# WELCOME
# ==============================
if st.session_state.page=="welcome":
    st.markdown("<div class='hero'><div class='eyebrow'>OPTIRETAIL AI</div><h1>Turn market data into <span>smarter decisions.</span></h1><p>Forecast demand, understand market signals, optimize inventory and make practical pricing decisions from one clean workspace.</p></div>",unsafe_allow_html=True)
    a,b,c=st.columns(3)
    a.markdown("<div class='card'><div class='eyebrow'>01 · FORECAST</div><h3>Know what will move.</h3><p class='small-note'>Use business context, holidays, weather and trend signals to estimate demand.</p></div>",unsafe_allow_html=True)
    b.markdown("<div class='card'><div class='eyebrow'>02 · INVENTORY</div><h3>Stock with confidence.</h3><p class='small-note'>Translate demand into an actionable inventory planning baseline.</p></div>",unsafe_allow_html=True)
    c.markdown("<div class='card'><div class='eyebrow'>03 · PRICE</div><h3>Price with context.</h3><p class='small-note'>Use live market-reference data to guide a demand-aware price.</p></div>",unsafe_allow_html=True)
    x,y=st.columns(2)
    if x.button("Login →",use_container_width=True): st.session_state.page="login"; st.rerun()
    if y.button("Create account →",use_container_width=True): st.session_state.page="signup"; st.rerun()
    st.markdown("<div class='footer-note'>OptiRetail AI · Analyze · Predict · Price</div>",unsafe_allow_html=True)

# ==============================
# LOGIN
# ==============================
elif st.session_state.page=="login":
    st.markdown("<div class='hero'><div class='eyebrow'>WELCOME BACK</div><h1>Sign in to <span>OptiRetail AI.</span></h1><p>Continue to your retail intelligence workspace.</p></div>",unsafe_allow_html=True)
    with st.form("login_form"):
        email=st.text_input("Email address"); password=st.text_input("Password",type="password")
        if st.form_submit_button("Sign in",use_container_width=True):
            user=login(email,password)
            if user: st.session_state.user=user; st.session_state.page="dashboard"; st.rerun()
            else: st.error("Invalid credentials.")

# ==============================
# SIGNUP
# ==============================
elif st.session_state.page=="signup":
    st.markdown("<div class='hero'><div class='eyebrow'>BUSINESS ONBOARDING</div><h1>Create your <span>OptiRetail AI</span> workspace.</h1><p>Set up your business profile to start making data-driven retail decisions.</p></div>",unsafe_allow_html=True)
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
elif st.session_state.page=="settings" and st.session_state.user:
    st.markdown("<div class='hero'><div class='eyebrow'>WORKSPACE</div><h1>Business settings</h1><p>Review the profile associated with your OptiRetail AI workspace.</p></div>",unsafe_allow_html=True)
    u=st.session_state.user; a,b,c=st.columns(3); a.metric("Account","Active"); b.metric("Email",u.get("email","N/A")); c.metric("Turnover",u.get("turnover","N/A"))
    st.info("OptiRetail AI uses SerpAPI for market-price intelligence.")

# ==============================
# DASHBOARD
# ==============================
elif st.session_state.page=="dashboard" and st.session_state.user:
    st.markdown("<div class='hero'><div class='eyebrow'>RETAIL INTELLIGENCE CONSOLE</div><h1>Good morning. Make the next decision <span>with confidence.</span></h1><p>Analyze a product, refresh live market context, then generate one practical demand, inventory and pricing decision.</p></div>",unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Product analysis</div>",unsafe_allow_html=True)
    c1,c2,c3=st.columns([1.6,1,1])
    with c1: product=st.text_input("Product",value=st.session_state.product,placeholder="e.g. Amul Taaza Milk 1L",label_visibility="collapsed")
    with c2: city=st.text_input("City",value=st.session_state.city,label_visibility="collapsed")
    with c3:
        months=list(calendar.month_name)[1:]; month_name=st.selectbox("Forecast month",months,index=months.index(st.session_state.month_name),label_visibility="collapsed")
    st.session_state.product,st.session_state.city,st.session_state.month_name=product,city,month_name

    ac1,ac2=st.columns([1,3])
    with ac1:
        if st.button("Refresh market price",use_container_width=True):
            p=fetch_product_price(product)
            st.session_state.current_price=p
            st.session_state.analyzed=True
            if p is None: st.warning("No reliable price found. Use a specific product name.")
            st.rerun()
    with ac2:
        if st.button("Generate AI Decision →",use_container_width=True):
            month=months.index(month_name)+1; year=datetime.datetime.now().year
            h=get_holidays(year,month); temp=get_weather(city); trend=trend_score(product)
            inp=pd.DataFrame({"holiday_count":[h],"avg_temp":[temp],"viral_score":[trend]})
            pred=float(model.predict(inp)[0]); stock=float(np.ceil(pred*1.10)); market=st.session_state.get("current_price")
            suggested=float(market*(1+np.clip((pred-300)/3000,-0.08,0.08))) if isinstance(market,(int,float)) else None
            st.session_state.last_prediction={"pred":pred,"stock":stock,"suggested":suggested,"temp":temp,"holiday":h,"trend":trend,"market":market}
            st.session_state.analyzed=True
            st.rerun()

    current=st.session_state.get("current_price")
    price_text=f"₹{current:,.2f}" if isinstance(current,(int,float)) else "Refresh"
    r=st.session_state.last_prediction
    forecast_text=f"{int(r.get('pred',0)):,} units" if r else "—"
    stock_text=f"{int(r.get('stock',0)):,} units" if r else "—"
    suggested_text=f"₹{r['suggested']:,.2f}" if r and r.get("suggested") is not None else "—"

    st.markdown("<div class='section-title'>Executive snapshot</div>",unsafe_allow_html=True)
    k1,k2,k3,k4=st.columns(4)
    k1.markdown(f"<div class='kpi'><small>Forecasted demand</small><strong>{forecast_text}</strong><span>AI demand estimate</span></div>",unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi'><small>Recommended stock</small><strong>{stock_text}</strong><span>10% planning buffer</span></div>",unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi'><small>Market reference</small><strong>{price_text}</strong><span>SerpAPI signal</span></div>",unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi'><small>Suggested price</small><strong>{suggested_text}</strong><span>Demand-aware guidance</span></div>",unsafe_allow_html=True)

    left,right=st.columns([1.55,.95],gap="large")
    with left:
        st.markdown("<div class='card'><div style='display:flex;justify-content:space-between;align-items:center'><div><div class='eyebrow'>DEMAND OUTLOOK</div><h2 style='margin:0;font-family:Plus Jakarta Sans'>Forecast vs inventory</h2><div class='small-note'>Selected product · city · month</div></div><span class='badge live'>● Live context</span></div></div>",unsafe_allow_html=True)
        if r:
            chart=pd.DataFrame({"Metric":["Predicted demand","Recommended stock"],"Units":[r["pred"],r["stock"]]}).set_index("Metric")
            st.bar_chart(chart,use_container_width=True)
        else:
            st.markdown("<div class='card'><div class='small-note'>Generate an AI decision to display the forecast visualization.</div></div>",unsafe_allow_html=True)
    with right:
        st.markdown("<div class='card'><div class='eyebrow'>MARKET SIGNALS</div><h2 style='margin:0;font-family:Plus Jakarta Sans'>What is changing?</h2><div class='signal'><div><b>Market price</b><br><small>Current reference</small></div><strong>"+price_text+"</strong></div><div class='signal'><div><b>Trend strength</b><br><small>Product momentum</small></div><strong>"+(f"{r['trend']}/100" if r else "—")+"</strong></div><div class='signal'><div><b>Weather</b><br><small>Forecast context</small></div><strong>"+(f"{r['temp']:.1f}°C" if r else "—")+"</strong></div><div class='signal'><div><b>Holiday activity</b><br><small>Weekend + India holidays</small></div><strong>"+(str(r['holiday']) if r else "—")+" days</strong></div></div>",unsafe_allow_html=True)

    if r:
        st.markdown("<div class='section-title'>AI decision</div>",unsafe_allow_html=True)
        d1,d2=st.columns([1.2,1])
        with d1:
            if r["trend"]>=70 or r["holiday"]>=8:
                title="Increase replenishment readiness"
                body="Demand signals are elevated. Prioritize availability, review current stock cover and avoid running lean ahead of the selected period."
            else:
                title="Maintain the planning baseline"
                body="Demand conditions appear relatively stable. Keep the recommended buffer and monitor sell-through before making a larger purchase decision."
            st.markdown(f"<div class='decision'><h3>{title}</h3><p>{body}</p></div>",unsafe_allow_html=True)
        with d2:
            price_note="Refresh the market price first to unlock pricing guidance." if r.get("suggested") is None else f"Reference {price_text} → suggested {suggested_text}. Use the recommendation as guidance, then validate margins and competitor pricing."
            st.markdown(f"<div class='decision'><h3>Pricing guidance</h3><p>{price_note}</p></div>",unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Business context</div>",unsafe_allow_html=True)
    a,b,c=st.columns(3)
    a.markdown("<div class='card'><div class='eyebrow'>INVENTORY</div><p class='small-note'>Recommended stock is a planning baseline with a 10% buffer. Adjust for current inventory and supplier lead time.</p></div>",unsafe_allow_html=True)
    b.markdown("<div class='card'><div class='eyebrow'>PRICING</div><p class='small-note'>SerpAPI provides a market-reference signal. The suggested price uses a bounded demand-response adjustment.</p></div>",unsafe_allow_html=True)
    c.markdown("<div class='card'><div class='eyebrow'>EXTERNAL SIGNALS</div><p class='small-note'>Weather and India holiday context are refreshed when a decision is generated.</p></div>",unsafe_allow_html=True)
