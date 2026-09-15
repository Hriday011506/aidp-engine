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
:root { --green:#047857; --green2:#10b981; --ink:#0f172a; --muted:#64748b; --line:#e2e8f0; --soft:#f8fafc; }
html, body, [class*="css"] { font-family:'Plus Jakarta Sans',sans-serif; }
.stApp { background:linear-gradient(180deg,#f7fbfa 0%,#f8fafc 42%,#eef7f4 100%); color:var(--ink); }
.block-container { max-width:1500px; padding:1.2rem 2rem 3rem; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#06281f,#07382b); border-right:1px solid rgba(255,255,255,.08); }
[data-testid="stSidebar"] * { color:#ecfdf5!important; }
[data-testid="stSidebar"] .stButton>button { background:transparent!important; box-shadow:none!important; border:1px solid transparent!important; text-align:left!important; }
.topbar { display:flex; align-items:center; justify-content:space-between; gap:18px; margin-bottom:18px; }
.brand { display:flex; align-items:center; gap:12px; }
.logo { width:46px; height:46px; border-radius:14px; background:linear-gradient(135deg,#10b981,#047857); color:white; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:20px; box-shadow:0 10px 24px rgba(4,120,87,.2); }
.brand-title { font-size:1.55rem; font-weight:800; line-height:1; }
.brand-sub { color:#64748b; font-size:.72rem; margin-top:5px; }
.hero { padding:34px 36px; border-radius:28px; background:linear-gradient(135deg,#ffffff 0%,#ecfdf5 100%); border:1px solid #dcefe8; box-shadow:0 18px 45px rgba(15,23,42,.06); margin-bottom:20px; }
.hero h1 { margin:0; font-size:3rem; line-height:1.05; letter-spacing:-1.7px; }
.hero p { margin:10px 0 0; color:#64748b; max-width:920px; font-size:1rem; }
.green-text { color:#047857; }
.eyebrow { color:#059669; text-transform:uppercase; letter-spacing:2px; font-weight:800; font-size:.68rem; margin-bottom:9px; }
.section-title { font-size:1.35rem; font-weight:800; margin:.7rem 0 .9rem; }
.card { background:#fff; border:1px solid var(--line); border-radius:22px; padding:22px; box-shadow:0 12px 34px rgba(15,23,42,.05); margin-bottom:16px; }
.kpi { background:#fff; border:1px solid var(--line); border-radius:20px; padding:20px; min-height:120px; box-shadow:0 10px 28px rgba(15,23,42,.04); }
.kpi small { color:#64748b; text-transform:uppercase; letter-spacing:1px; font-size:.64rem; font-weight:800; }
.kpi strong { display:block; font-size:1.65rem; margin-top:8px; color:#0f172a; }
.kpi span { display:block; color:#059669; font-size:.72rem; margin-top:4px; font-weight:700; }
.feature { background:linear-gradient(135deg,#06281f,#047857); color:#ecfdf5; border-radius:24px; padding:24px; box-shadow:0 18px 38px rgba(4,120,87,.18); }
.feature h3 { margin:0 0 7px; font-size:1.15rem; }
.feature p { color:#bbf7d0; margin:0; line-height:1.7; font-size:.88rem; }
.pill { display:inline-block; padding:7px 11px; border-radius:999px; background:#ecfdf5; border:1px solid #bbf7d0; color:#047857; font-size:.7rem; font-weight:800; }
.stButton>button { border:0!important; border-radius:12px!important; min-height:44px!important; font-weight:800!important; background:linear-gradient(135deg,#10b981,#047857)!important; color:white!important; box-shadow:0 10px 25px rgba(4,120,87,.16); }
.stButton>button:hover { transform:translateY(-1px); }
.stTextInput input,.stSelectbox div[data-baseweb="select"]>div { border-radius:12px!important; border-color:#d7e3df!important; }
[data-testid="stMetricValue"] { color:#0f172a; }
[data-testid="stMetricLabel"] { color:#64748b; }
.small-note { color:#64748b; font-size:.76rem; }
hr { border-color:#e2e8f0!important; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ==============================
# LOCAL SESSION AUTH
# ==============================
def signup(email, password, gst, turnover):
    email = email.strip().lower()
    if any(u["email"] == email for u in st.session_state.users):
        return False, "An account with this email already exists."
    st.session_state.users.append({"email":email,"password":password,"gst":gst,"turnover":turnover,"created_at":datetime.datetime.now().isoformat()})
    return True, "Account created successfully."

def login(email, password):
    email = email.strip().lower()
    for user in st.session_state.users:
        if user["email"] == email and user["password"] == password:
            return user
    return None

# ==============================
# SERVICES
# ==============================
@st.cache_data(ttl=900, show_spinner=False)
def fetch_product_price(product_name):
    product_name = product_name.strip()
    if not product_name or not SERPAPI_KEY or GoogleSearch is None:
        return None
    try:
        results = GoogleSearch({"engine":"google_shopping_light","q":f"{product_name} price","gl":"in","hl":"en","num":8,"api_key":SERPAPI_KEY}).get_dict()
        tokens=[t for t in product_name.lower().split() if len(t)>2]
        candidates=[]
        for item in results.get("shopping_results",[]):
            raw=item.get("extracted_price",item.get("price"))
            try:
                if isinstance(raw,str): raw=raw.replace(",","").replace("₹","").strip()
                numeric=float(raw)
            except (TypeError,ValueError):
                continue
            if numeric<=0: continue
            title=str(item.get("title","")).lower()
            hits=sum(token in title for token in tokens)
            score=hits*3+(10 if product_name.lower() in title else 0)
            candidates.append((score,numeric))
        if not candidates: return None
        candidates.sort(reverse=True)
        best=candidates[0][0]
        prices=[p for s,p in candidates if s==best]
        return float(np.median(prices[:5]))
    except Exception:
        return None

@st.cache_data(ttl=900, show_spinner=False)
def get_weather(city):
    try:
        r=requests.get("https://geocoding-api.open-meteo.com/v1/search",params={"name":city,"count":1},timeout=4)
        r.raise_for_status(); rs=r.json().get("results",[])
        if not rs: return 25.0
        lat,lon=rs[0]["latitude"],rs[0]["longitude"]
        w=requests.get("https://api.open-meteo.com/v1/forecast",params={"latitude":lat,"longitude":lon,"current_weather":"true"},timeout=4)
        w.raise_for_status(); return float(w.json()["current_weather"]["temperature"])
    except Exception: return 25.0

@st.cache_data
def get_holidays(year, month):
    india_holidays=holidays.India(years=year)
    days=calendar.monthrange(year,month)[1]
    return sum(1 for d in range(1,days+1) if datetime.date(year,month,d).weekday()>=5 or datetime.date(year,month,d) in india_holidays)

@st.cache_data
def trend_score(product):
    rng=np.random.default_rng(sum(ord(c) for c in product))
    return int(rng.integers(30,90))

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
    model=RandomForestRegressor(n_estimators=160,max_depth=12,random_state=42,n_jobs=-1)
    model.fit(df[["holiday_count","avg_temp","viral_score"]],df["sales"])
    return model
model=train_model(load_data())

# ==============================
# NAVIGATION
# ==============================
with st.sidebar:
    st.markdown("## 📊 OptiRetail AI")
    st.caption("Analyze · Predict · Price")
    st.divider()
    if st.session_state.user:
        st.caption(st.session_state.user.get("email","Business User"))
        if st.button("⌂  Dashboard",use_container_width=True): st.session_state.page="dashboard"; st.rerun()
        if st.button("⌕  Product Analysis",use_container_width=True): st.session_state.page="dashboard"; st.rerun()
        if st.button("▥  Demand Forecasting",use_container_width=True): st.session_state.page="dashboard"; st.rerun()
        if st.button("⌁  Dynamic Pricing",use_container_width=True): st.session_state.page="dashboard"; st.rerun()
        if st.button("◎  Market Insights",use_container_width=True): st.session_state.page="dashboard"; st.rerun()
        if st.button("☆  Saved Analyses",use_container_width=True): st.session_state.page="dashboard"; st.rerun()
        st.divider()
        if st.button("◯  Account",use_container_width=True): st.session_state.page="settings"; st.rerun()
        if st.button("⚙  Settings",use_container_width=True): st.session_state.page="settings"; st.rerun()
        if st.button("Sign out",use_container_width=True): st.session_state.user=None; st.session_state.page="welcome"; st.rerun()

# ==============================
# WELCOME
# ==============================
if st.session_state.page=="welcome":
    st.markdown("""
    <div class='hero'>
      <div class='eyebrow'>AI-POWERED RETAIL INTELLIGENCE</div>
      <h1>Turn market data into <span class='green-text'>smarter decisions.</span></h1>
      <p>OptiRetail AI combines demand forecasting, inventory planning, live market pricing and external signals in one clear workspace.</p>
    </div>
    """,unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1:
        st.markdown("<div class='card'><span class='pill'>DEMAND</span><h3>Forecast with context</h3><p class='small-note'>Use holidays, weather and demand signals to estimate upcoming requirements.</p></div>",unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><span class='pill'>PRICE</span><h3>Understand the market</h3><p class='small-note'>Use current shopping-market data as a reference before pricing decisions.</p></div>",unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='feature'><h3>Ready for smarter retail?</h3><p>Create an account and open your intelligence workspace.</p></div>",unsafe_allow_html=True)
    b1,b2=st.columns([1,1])
    with b1:
        if st.button("Get Started →",use_container_width=True): st.session_state.page="signup"; st.rerun()
    with b2:
        if st.button("Login",use_container_width=True): st.session_state.page="login"; st.rerun()

# ==============================
# LOGIN
# ==============================
elif st.session_state.page=="login":
    st.markdown("<div class='hero'><div class='eyebrow'>ACCOUNT ACCESS</div><h1>Welcome <span class='green-text'>back.</span></h1><p>Sign in to continue to your OptiRetail AI workspace.</p></div>",unsafe_allow_html=True)
    with st.form("login_form"):
        email=st.text_input("Business email")
        password=st.text_input("Password",type="password")
        if st.form_submit_button("Login",use_container_width=True):
            user=login(email,password)
            if user:
                st.session_state.user=user; st.session_state.page="dashboard"; st.rerun()
            else: st.error("Invalid credentials. Please create an account first.")

# ==============================
# SIGNUP
# ==============================
elif st.session_state.page=="signup":
    st.markdown("<div class='hero'><div class='eyebrow'>BUSINESS ONBOARDING</div><h1>Create your <span class='green-text'>OptiRetail AI</span> workspace.</h1><p>Set up your profile and start making data-backed retail decisions.</p></div>",unsafe_allow_html=True)
    with st.form("signup_form"):
        email=st.text_input("Business email")
        password=st.text_input("Password",type="password")
        gst=st.text_input("GST Number")
        turnover=st.selectbox("Annual Turnover",["1–5 Lakh","5–10 Lakh","10–15 Lakh","15–50 Lakh","50 Lakh+"])
        if st.form_submit_button("Create Account",use_container_width=True):
            if not email or not password or not gst: st.error("Please fill all required fields.")
            else:
                ok,msg=signup(email,password,gst,turnover)
                if ok: st.success(msg); st.session_state.page="login"; st.rerun()
                else: st.error(msg)

# ==============================
# SETTINGS
# ==============================
elif st.session_state.page=="settings" and st.session_state.user:
    st.markdown("<div class='hero'><div class='eyebrow'>WORKSPACE</div><h1>Business <span class='green-text'>settings.</span></h1><p>Manage the business information associated with your OptiRetail AI workspace.</p></div>",unsafe_allow_html=True)
    user=st.session_state.user
    c1,c2,c3=st.columns(3)
    c1.metric("Account","Active")
    c2.metric("Email",user.get("email","N/A"))
    c3.metric("Turnover",user.get("turnover","N/A"))
    st.info("OptiRetail AI is using SerpAPI for live market-price intelligence.")

# ==============================
# DASHBOARD
# ==============================
elif st.session_state.page=="dashboard" and st.session_state.user:
    st.markdown("<div class='topbar'><div><div class='brand'><div class='logo'>↗</div><div><div class='brand-title'>OptiRetail AI</div><div class='brand-sub'>AI Demand Intelligence · Analyze · Predict · Price</div></div></div></div><div class='small-note'>Data is the compass for better business decisions.</div></div>",unsafe_allow_html=True)
    st.markdown("<div class='hero'><div class='eyebrow'>PRODUCT INTELLIGENCE</div><h1>Make your next <span class='green-text'>retail decision</span> with confidence.</h1><p>Enter a product, select the market and forecast period, then generate one clear view of demand, stock and pricing guidance.</p></div>",unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Analyse a product</div>",unsafe_allow_html=True)
    c1,c2,c3=st.columns([1.6,1,1])
    with c1: product=st.text_input("Product",value=st.session_state.product,placeholder="e.g. iPhone 15, Nike Air Force 1, Wheat Flour")
    with c2: city=st.text_input("Market / City",value=st.session_state.city)
    with c3:
        months=list(calendar.month_name)[1:]
        month_name=st.selectbox("Forecast month",months,index=months.index(st.session_state.month_name))
    st.session_state.product,st.session_state.city,st.session_state.month_name=product,city,month_name

    analyze,forecast=st.columns([1,2])
    with analyze:
        if st.button("Refresh market price",use_container_width=True):
            with st.spinner("Fetching market price..."):
                p=fetch_product_price(product)
            if p is None: st.warning("No reliable shopping price found. Use a specific product name.")
            else: st.session_state.current_price=p; st.success(f"Market reference: ₹{p:,.2f}")
            st.rerun()
    with forecast:
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
    k1,k2,k3,k4=st.columns(4)
    k1.markdown(f"<div class='kpi'><small>Forecasted demand</small><strong>{int(st.session_state.last_prediction['pred']):,} units</strong><span>AI demand estimate</span></div>" if st.session_state.last_prediction else "<div class='kpi'><small>Forecasted demand</small><strong>—</strong><span>Generate a decision</span></div>",unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi'><small>Recommended stock</small><strong>{int(st.session_state.last_prediction['stock']):,} units</strong><span>Includes 10% planning buffer</span></div>" if st.session_state.last_prediction else "<div class='kpi'><small>Recommended stock</small><strong>—</strong><span>Inventory action</span></div>",unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi'><small>Market reference</small><strong>{current_text}</strong><span>SerpAPI shopping signal</span></div>",unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi'><small>Suggested price</small><strong>{('₹'+format(st.session_state.last_prediction['suggested'],',.2f')) if st.session_state.last_prediction and st.session_state.last_prediction.get('suggested') is not None else '—'}</strong><span>Demand-response guidance</span></div>",unsafe_allow_html=True)

    if st.session_state.last_prediction:
        r=st.session_state.last_prediction
        left,right=st.columns([1.5,1])
        with left:
            st.markdown("<div class='card'><div class='eyebrow'>DEMAND FORECAST</div><h2 style='margin:0 0 5px'>Demand outlook</h2><div class='small-note'>Selected context: product + city + forecast month</div></div>",unsafe_allow_html=True)
            chart=pd.DataFrame({"Metric":["Demand","Recommended Stock"],"Units":[r["pred"],r["stock"]]}).set_index("Metric")
            st.bar_chart(chart,use_container_width=True)
        with right:
            st.markdown("<div class='card'><div class='eyebrow'>PRICING GUIDANCE</div><h2 style='margin:0 0 8px'>Price optimization</h2><div class='small-note'>Reference price + bounded demand response</div></div>",unsafe_allow_html=True)
            if r.get("suggested") is not None:
                st.metric("Suggested selling price",f"₹{r['suggested']:,.2f}")
                st.success("Pricing guidance is available using the current market reference.")
            else:
                st.info("Refresh the market price first to calculate pricing guidance.")

        s1,s2,s3=st.columns(3)
        s1.info(f"**Weather:** {r['temp']:.1f}°C")
        s2.info(f"**Holiday / weekend days:** {r['holiday']}")
        s3.info(f"**Trend signal:** {r['trend']}/100")

        st.markdown("### AI recommendation")
        if r["trend"]>=70:
            st.success("High demand momentum: keep replenishment readiness high and review the suggested price before the forecast period.")
        elif r["holiday"]>=8:
            st.info("Elevated calendar activity: consider carrying additional safety stock.")
        else:
            st.warning("Stable conditions: use the recommended stock as a planning baseline and monitor sell-through.")
    else:
        st.markdown("<div class='feature'><h3>Your decision workspace is ready.</h3><p>Refresh the market price for your product, then generate the AI decision to see demand, stock and pricing guidance together.</p></div>",unsafe_allow_html=True)
        st.markdown("### What OptiRetail AI brings together")
        a,b,c=st.columns(3)
        a.info("**Predict** · Demand forecast based on business signals")
        b.info("**Protect** · Recommended stock with planning buffer")
        c.info("**Price** · Market reference + bounded pricing guidance")
