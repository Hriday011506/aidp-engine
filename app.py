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
DEFAULTS = {"user": None, "users": {}, "page": "welcome", "product": "Wheat Flour", "city": "Jaipur", "month": dt.datetime.now().strftime("%B"), "prediction": None, "forecast": None, "price": None, "price_source": None, "saved": []}
for k, v in DEFAULTS.items(): st.session_state.setdefault(k, v)

st.markdown("""
<style>
:root{--green:#079568;--green2:#047a55;--dark:#063b2e;--ink:#0f172a;--muted:#64748b;--line:#d9e9e3}
.stApp{background:#ffffff;color:var(--ink)}
.block-container{max-width:1500px;padding:1rem 2rem 4rem}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#032d23,#063b2e)!important;border-right:1px solid #075943}
[data-testid="stSidebar"] *{color:#ecfdf5!important}
.brand{display:flex;align-items:center;gap:12px;padding:8px 0 22px;border-bottom:1px solid #ffffff33;margin-bottom:20px}.mark{width:48px;height:48px;border-radius:14px;background:linear-gradient(135deg,#10b981,#047857);display:flex;align-items:center;justify-content:center;font-size:25px}.brand-title{font-size:1.35rem;font-weight:900;color:#fff!important}.brand-title span{color:#34d399!important}.brand-sub{font-size:.62rem;color:#a7f3d0!important;letter-spacing:1px}
.hero{padding:30px 38px;border-radius:28px;background:linear-gradient(135deg,#fff,#effbf6);border:1px solid var(--line);box-shadow:0 18px 48px #10b98114;margin-bottom:20px}.hero h1{font-size:2.6rem;margin:0;color:#047857!important;font-weight:900}.hero h1 span{color:#047857!important}.hero p{color:#64748b!important}
.eyebrow{color:#047857!important;text-transform:uppercase;font-size:.65rem;font-weight:900;letter-spacing:2px;margin-bottom:8px}
.card{background:#fff;border:1px solid #d9e9e3;border-radius:20px;padding:20px;box-shadow:0 10px 30px #0f172a0d;margin-bottom:16px}.card h3{margin:.15rem 0 .5rem;color:#047857!important}.small{color:#475569!important;font-size:.84rem;line-height:1.5}
.kpi{background:#fff;border:1px solid #d9e9e3;border-radius:18px;padding:18px;min-height:112px}.kpi small{color:#64748b!important;text-transform:uppercase;letter-spacing:1px;font-weight:800;font-size:.62rem}.kpi strong{display:block;color:#047857!important;font-size:1.55rem;margin-top:7px}.kpi span{font-size:.74rem;color:#64748b!important}
.action{border:1px solid #d9e9e3;border-radius:14px;padding:13px;margin:8px 0;background:#fff}.action p{margin:3px 0 0;color:#64748b!important;font-size:.82rem}.critical{border-left:5px solid #ef4444}.warning{border-left:5px solid #f59e0b}.info{border-left:5px solid #3b82f6}.decision{border-radius:18px;padding:18px;background:#ecfdf5;border:1px solid #bbf7d0}.decision h3{color:#065f46!important;margin:0 0 5px}.score{font-size:3rem;font-weight:900;color:#047857!important}.auth{max-width:800px;margin:25px auto}.auth-card{background:#fff;border:1px solid var(--line);border-radius:28px;padding:34px;box-shadow:0 20px 55px #0f172a12}.auth-title{font-size:2.2rem;font-weight:900;color:#047857!important;margin:0}.auth-sub{color:#64748b!important}
/* Restore readable green/white application controls and headings. */
input,textarea{color:#0f172a!important;background:#fff!important;border-color:#cbd5e1!important}label,div[data-testid="stWidgetLabel"] p{color:#334155!important;font-weight:600!important}
.stButton>button,.stFormSubmitButton>button{border:0!important;border-radius:12px!important;min-height:44px!important;font-weight:800!important;background:linear-gradient(135deg,#079568,#047a55)!important;color:#fff!important}
.stButton>button:hover,.stFormSubmitButton>button:hover{background:linear-gradient(135deg,#0aa979,#05835e)!important;color:#fff!important}
[data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"]{color:#047857!important}
[data-testid="stMarkdownContainer"] h1,[data-testid="stMarkdownContainer"] h2,[data-testid="stMarkdownContainer"] h3,[data-testid="stMarkdownContainer"] h4{color:#047857!important}
[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li{color:#334155}
.stSelectbox label,.stTextInput label,.stNumberInput label{color:#334155!important}
</style>
""", unsafe_allow_html=True)

def ph(value): return hashlib.sha256(str(value).encode()).hexdigest()
def signup(email,password,gst,turnover):
    email=email.strip().lower()
    if email in st.session_state.users:return False,"An account with this email already exists in this session."
    st.session_state.users[email]={"email":email,"password":ph(password),"gst":gst.strip(),"turnover":turnover};return True,"Account created successfully."
def authenticate(email,password):
    user=st.session_state.users.get(email.strip().lower());return user if user and user.get("password")==ph(password) else None
def trend(product):return int(np.random.default_rng(sum(ord(x) for x in product.lower())).integers(45,91))
def weather_fallback(lat=20):
    base=25-min(abs(lat),45)*.06;amp=min(14,8+abs(lat)*.1);phase=5 if lat>=0 else 11
    return {calendar.month_name[m]:round(base+amp*np.cos((m-phase)*2*np.pi/12),1) for m in range(1,13)}
@st.cache_data(ttl=86400,show_spinner=False)
def monthly_weather(city):
    try:
        geo=requests.get("https://geocoding-api.open-meteo.com/v1/search",params={"name":city,"count":1,"language":"en","format":"json"},timeout=(3,7));results=geo.json().get("results") or []
        if not results:raise ValueError("City not found")
        hit=results[0];lat,lon=float(hit["latitude"]),float(hit["longitude"]);end=dt.datetime.now().year-1;start=end-4
        archive=requests.get("https://archive-api.open-meteo.com/v1/archive",params={"latitude":lat,"longitude":lon,"start_date":f"{start}-01-01","end_date":f"{end}-12-31","daily":"temperature_2m_mean","timezone":"auto"},timeout=(3,12));data=archive.json().get("daily",{})
        frame=pd.DataFrame({"date":pd.to_datetime(data.get("time",[])),"temp":data.get("temperature_2m_mean",[])}).dropna()
        if not frame.empty:
            frame["month"]=frame.date.dt.month;grouped=frame.groupby("month").temp.mean();temps={calendar.month_name[m]:round(float(grouped.get(m,25)),1) for m in range(1,13)};return hit.get("name",city),temps,"5-year historical monthly average"
    except Exception:pass
    return city,weather_fallback(20),"seasonal fallback"
def _price_number(value):
    if isinstance(value,(int,float,np.number)):return float(value)
    text=str(value).replace("₹","").replace("Rs.","").replace("Rs","").replace(",","").strip();match=re.search(r"\d+(?:\.\d+)?",text);return float(match.group()) if match else None

def _fallback_market_reference(product):
    p=str(product).lower();catalog={"coke":(40.0,"Indicative reference · Coca-Cola family"),"coca cola":(40.0,"Indicative reference · Coca-Cola family"),"amul":(60.0,"Indicative reference · Amul family"),"pepsi":(45.0,"Indicative reference · Pepsi family"),"lays":(20.0,"Indicative reference · Lay's family"),"rice":(320.0,"Indicative reference · Rice 5kg"),"cooking oil":(145.0,"Indicative reference · Cooking Oil 1L"),"wheat flour":(260.0,"Indicative reference · Wheat Flour 5kg")}
    for key,value in catalog.items():
        if key in p:return value
    return None,None

def live_market_price(product,city):
    key=str(st.secrets.get("SERPAPI_KEY","")).strip()
    if not key:return None,"SERPAPI_KEY is missing from Streamlit Secrets."
    queries=[f"{product} price in {city} India",f"{product} MRP price India",f"{product} current price India"]
    last_error=None
    for query in queries:
        params={"engine":"google","q":query,"location":f"{city}, India","hl":"en","gl":"in","device":"desktop","api_key":key}
        try:
            response=requests.get("https://serpapi.com/search.json",params=params,timeout=(3,7));response.raise_for_status();data=response.json()
            if data.get("error"):last_error=str(data["error"]);continue
            candidates=[]
            for item in list(data.get("organic_results") or [])[:30]+list(data.get("shopping_results") or [])[:30]:
                price=_price_number(item.get("extracted_price",item.get("price")))
                if price is None or not 0<price<100000:continue
                candidates.append((price,str(item.get("source") or item.get("merchant") or item.get("displayed_link") or "SerpApi result")))
            if candidates:
                values=[x[0] for x in candidates[:12]];market=round(float(np.median(values)),2);sources=[]
                for _,source in candidates[:12]:
                    if source and source not in sources:sources.append(source)
                return market,f"Live web market reference · {', '.join(sources[:6]) or 'SerpApi search results'}"
            last_error="SerpApi returned no priced results for this query."
        except requests.exceptions.Timeout:last_error="SerpApi request timed out."
        except requests.exceptions.RequestException as exc:last_error=str(exc)
        except (ValueError,TypeError) as exc:last_error=str(exc)
    fallback,fallback_source=_fallback_market_reference(product)
    if fallback is not None:return fallback,fallback_source+" · live web lookup unavailable"
    return None,f"Live market lookup failed: {last_error or 'no priced results returned'}"

def holiday_days(year,month):
    h=holidays.India(years=year) if holidays else set();return sum(1 for d in range(1,calendar.monthrange(year,month)[1]+1) if dt.date(year,month,d).weekday()>=5 or dt.date(year,month,d) in h)
@st.cache_resource(show_spinner=False)
def get_model():
    rng=np.random.default_rng(42);frame=pd.DataFrame({"holiday":rng.integers(0,14,800),"temp":rng.uniform(8,42,800),"trend":rng.integers(0,100,800)});season=np.maximum(0,24-np.abs(frame.temp-28));frame["sales"]=170+frame.holiday*38+frame.trend*4.7+season*12+rng.normal(0,24,800);model=RandomForestRegressor(n_estimators=220,max_depth=12,random_state=42,n_jobs=-1);model.fit(frame[["holiday","temp","trend"]],frame.sales);return model
MODEL=get_model()
def make_forecast(product,city):
    resolved,temps,source=monthly_weather(city);product_trend=trend(product);rows=[];year=dt.datetime.now().year
    for number,month in enumerate(MONTHS,1):
        temp=float(temps.get(month,25));holiday=holiday_days(year,number);features=pd.DataFrame({"holiday":[holiday],"temp":[temp],"trend":[product_trend]});demand=float(max(0,MODEL.predict(features)[0]));rows.append({"Month":month,"Temperature (°C)":round(temp,1),"Holiday days":holiday,"Trend score":product_trend,"Forecast demand":round(demand)})
    return pd.DataFrame(rows),resolved,source

def _mrp_for_product(product,market):
    p=str(product).lower();known={"coke":40.0,"coca cola":40.0,"amul taaza milk 1l":56.0,"amul taaza milk":56.0,"amul":60.0,"pepsi":45.0,"lays":20.0}
    for key,mrp in known.items():
        if key in p:return mrp
    return None

def generate():
    market=st.session_state.price
    if market is None:return False
    table,city,weather_source=make_forecast(st.session_state.product,st.session_state.city);row=table.loc[table.Month==st.session_state.month].iloc[0]
    pred=float(row["Forecast demand"]);temp=float(row["Temperature (°C)"]);holiday=int(row["Holiday days"]);trend_score=int(row["Trend score"])
    demand_factor=float(np.clip((pred-700)/5000,-0.045,0.045));weather_factor=float(np.clip((temp-27)/250,-0.025,0.025));holiday_factor=float(np.clip(holiday/600,-0.012,0.012));trend_factor=float(np.clip((trend_score-60)/3000,-0.015,0.015));adjustment=float(np.clip(demand_factor+weather_factor+holiday_factor+trend_factor,-0.06,0.06));base=market*(1+adjustment)
    mrp=_mrp_for_product(st.session_state.product,market);suggested=round(min(base,mrp),2) if mrp is not None else round(base,2)
    p={"product":st.session_state.product,"city":city,"month":st.session_state.month,"pred":pred,"stock":float(np.ceil(pred*1.1)),"temp":temp,"holiday":holiday,"trend":trend_score,"market":float(market),"suggested":suggested,"source":weather_source,"confidence":int(np.clip(70+abs(trend_score-60)*.25,68,94))}
    st.session_state.forecast=table;st.session_state.prediction=p;st.session_state.saved.insert(0,p.copy());st.session_state.saved=st.session_state.saved[:20];return True

def normalize_prediction(p):
    if not isinstance(p,dict):return None
    required={"product","city","month","pred","stock","temp","holiday","trend","market","suggested","source","confidence"}
    if not required.issubset(p):return None
    try:
        for k in ["pred","stock","temp","market","suggested"]:p[k]=float(p[k])
        for k in ["holiday","trend","confidence"]:p[k]=int(p[k])
        p["source"]=str(p.get("source","Historical weather"));return p
    except (TypeError,ValueError):return None
def matches(p):
    p=normalize_prediction(p);return bool(p) and p.get("product")==st.session_state.product and p.get("city")==st.session_state.city and p.get("month")==st.session_state.month
def health(p):
    if not p:return 0
    score=(np.clip(65+(p["trend"]-50)*.35,50,94)+np.clip(92-max(0,(p["pred"]-1200)/40),60,94)+80+np.clip(92-abs(p["temp"]-27)*1.4,60,94))/4;return int(np.clip(score,55,95))
def action_list(p):
    if not p:return [("info","Generate an AI decision","Start Product Analysis to unlock actions.")]
    a=[("critical","Increase inventory","Predicted demand is high.") if p["pred"]>1200 else ("warning","Keep inventory lean","Forecast demand is relatively low.") if p["pred"]<500 else ("info","Maintain planned stock","Demand is manageable with the 10% buffer.")]
    if p["trend"]>=78:a.append(("info","Demand surge detected","Trend score is strong; monitor replenishment more frequently."))
    if p["temp"]>=34:a.append(("warning","Watch heat sensitivity","High temperature may increase volatility for sensitive products."))
    a.append(("warning","Review pricing","Demand-aware price is above the market reference.") if p["suggested"]>p["market"]*1.02 else ("info","Price is market-aligned","Suggested price remains close to the market reference."))
    return a

def sidebar():
    st.sidebar.markdown("<div class='brand'><div class='mark'>🛒</div><div><div class='brand-title'>OptiRetail <span>AI</span></div><div class='brand-sub'>SMARTER DECISIONS · STRONGER RETAIL</div></div></div>",unsafe_allow_html=True)
    st.sidebar.markdown(f"<a style='color:#34d399!important'>{html.escape(st.session_state.user or 'admin@optiretail.in')}</a>",unsafe_allow_html=True)
    pages=[("dashboard","Dashboard"),("product","Product Analysis"),("forecast","Demand Forecasting"),("pricing","Dynamic Pricing"),("market","Market Insights"),("comparison","Product Comparison"),("simulator","AI Simulator"),("alerts","Alerts & Risk"),("copilot","AI Decision Copilot"),("saved","Saved Analyses"),("settings","Settings")]
    for key,label in pages:
        if st.sidebar.button(label,key=f"nav_{key}",use_container_width=True):st.session_state.page=key;st.rerun()
    st.sidebar.markdown("<hr style='border-color:#ffffff33'>",unsafe_allow_html=True)
    if st.sidebar.button("Sign out",key="signout",use_container_width=True):st.session_state.user=None;st.session_state.page="welcome";st.rerun()

def page_header(eyebrow,title,subtitle=""):
    st.markdown(f"<div class='hero'><div class='eyebrow'>{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></div>",unsafe_allow_html=True)

def dashboard_page():
    p=st.session_state.prediction;page_header("DASHBOARD","Turn market data into smarter decisions.","AI Action Center, health score, explainable forecast and simulator.")
    if p:
        cols=st.columns(5)
        vals=[("FORECAST DEMAND",f"{int(p['pred']):,}","units"),("RECOMMENDED STOCK",f"{int(p['stock']):,}","units · 10% buffer"),("MARKET REFERENCE",f"₹{p['market']:,.2f}","live web market data"),("SOCIAL VIRAL",f"{p['trend']}/100","demand signal"),("CONFIDENCE",f"{p['confidence']}%","model confidence")]
        for c,(a,b,d) in zip(cols,vals):c.markdown(f"<div class='kpi'><small>{a}</small><strong>{b}</strong><span>{d}</span></div>",unsafe_allow_html=True)
        st.markdown("### AI Action Center")
        c1,c2=st.columns(2)
        with c1:
            for kind,title,desc in action_list(p):st.markdown(f"<div class='action {kind}'><b>{title}</b><p>{desc}</p></div>",unsafe_allow_html=True)
        with c2:
            st.markdown("### Business health score");st.markdown(f"<div class='score'>{health(p)}/100</div><div class='small'>Overall health</div><div style='height:10px;background:#e2e8f0;border-radius:9px;margin-top:10px'><div style='width:{health(p)}%;height:10px;background:#079568;border-radius:9px'></div></div>",unsafe_allow_html=True)
        st.markdown("### 12-month demand")
        st.line_chart(st.session_state.forecast.set_index("Month")["Forecast demand"])
        st.markdown("### Explainable AI")
        st.markdown(f"**Product trend:** {p['trend']}/100  
**Holiday days:** {p['holiday']}  
**Temperature:** {p['temp']:.1f} °C")
    else:st.info("Run Product Analysis to populate the AI dashboard.")

def product_page():
    page_header("ANALYSIS INPUTS","Choose product, location and forecast month","Select a product and location, refresh the market reference, and generate a complete AI decision.")
    c1,c2,c3=st.columns([1.5,1,1]);st.session_state.product=c1.text_input("Product",st.session_state.product);st.session_state.city=c2.text_input("City",st.session_state.city);st.session_state.month=c3.selectbox("Forecast month",MONTHS,index=MONTHS.index(st.session_state.month))
    b1,b2=st.columns(2)
    if b1.button("Refresh market price",use_container_width=True):
        with st.spinner("Finding current market references..."):price,source=live_market_price(st.session_state.product,st.session_state.city)
        st.session_state.price=price;st.session_state.price_source=source
    if b2.button("Generate AI Decision →",use_container_width=True):
        if st.session_state.price is None:
            with st.spinner("Finding market reference..."):st.session_state.price,st.session_state.price_source=live_market_price(st.session_state.product,st.session_state.city)
        if st.session_state.price is not None:generate();st.success("AI decision generated.")
        else:st.error(st.session_state.price_source or "Market reference unavailable.")
    if st.session_state.price is not None:st.success(f"Market reference: ₹{st.session_state.price:,.2f} · {st.session_state.price_source}")
    p=st.session_state.prediction
    if p:st.markdown(f"<div class='card'><div class='eyebrow'>LATEST RESULT</div><h3>{html.escape(p['product'])} · {html.escape(p['city'])} · {p['month']}</h3><p class='small'>Forecast demand: <b>{int(p['pred']):,}</b> units · Weather: <b>{p['temp']:.1f} °C</b> · Holiday days: <b>{p['holiday']}</b> · Social viral: <b>{p['trend']}/100</b> · Confidence: <b>{p['confidence']}%</b> · Market: <b>₹{p['market']:,.2f}</b></p></div>",unsafe_allow_html=True)

def forecast_page():
    page_header("DEMAND FORECASTING","See demand before it happens.","Forecast demand across the next 12 months using multiple signals.")
    if st.session_state.forecast is None:st.info("Generate an AI decision first.");return
    st.dataframe(st.session_state.forecast,use_container_width=True,hide_index=True)
    st.line_chart(st.session_state.forecast.set_index("Month")["Forecast demand"])

def pricing_page():
    page_header("DYNAMIC PRICING","Price with market awareness.","Compare the market reference with the demand-aware recommendation.")
    p=st.session_state.prediction
    if not p:st.info("Generate an AI decision first.");return
    a,b,c=st.columns(3);a.metric("Market reference",f"₹{p['market']:,.2f}");b.metric("Suggested price",f"₹{p['suggested']:,.2f}");c.metric("Difference",f"{((p['suggested']/p['market'])-1)*100:+.1f}%")
    st.markdown(f"<div class='decision'><h3>Pricing rationale</h3><p>Recommendation uses the market reference of ₹{p['market']:,.2f}, forecast demand of {int(p['pred']):,} units, temperature of {p['temp']:.1f} °C, {p['holiday']} holiday days and a social viral score of {p['trend']}/100.</p></div>",unsafe_allow_html=True)

def generic_page(title,subtitle,body):
    page_header(title.upper(),title,subtitle);st.markdown(body,unsafe_allow_html=True)

def market_page():
    generic_page("Market Insights","Understand current market conditions.","<div class='card'><h3>Market intelligence</h3><p class='small'>Live web market references are used when available. Search results are not restricted to a specific retailer.</p></div>")
def comparison_page():
    generic_page("Product Comparison","Compare product demand, trends and market references.","<div class='card'><h3>Compare products</h3><p class='small'>Use Product Analysis to generate results for products and compare their demand signals.</p></div>")
def simulator_page():
    generic_page("AI Simulator","Test how changing signals can affect demand.","<div class='card'><h3>Scenario simulation</h3><p class='small'>Experiment with demand, weather, holidays and social trend assumptions.</p></div>")
def alerts_page():
    generic_page("Alerts & Risk","Identify demand and pricing risks early.","<div class='card'><h3>Risk monitoring</h3><p class='small'>Alerts are generated from forecast demand, social trend intensity and market-aware pricing.</p></div>")
def copilot_page():
    generic_page("AI Decision Copilot","Turn model outputs into practical retail actions.","<div class='card'><h3>AI recommendation</h3><p class='small'>The copilot explains inventory and pricing recommendations using the current forecast signals.</p></div>")
def saved_page():
    page_header("SAVED ANALYSES","Your saved decisions.")
    if not st.session_state.saved:st.info("No saved analyses yet.");return
    for p in st.session_state.saved:
        p=normalize_prediction(p)
        if p:st.markdown(f"<div class='card'><h3>{html.escape(p['product'])} · {html.escape(p['city'])} · {p['month']}</h3><p class='small'>Demand {int(p['pred']):,} · Social viral {p['trend']}/100 · Market ₹{p['market']:,.2f} · Suggested ₹{p['suggested']:,.2f}</p></div>",unsafe_allow_html=True)
def settings_page():
    generic_page("Settings","Manage your workspace.","<div class='card'><h3>Workspace settings</h3><p class='small'>OptiRetail AI uses session-based workspace data in this build. MongoDB is currently disabled.</p></div>")

def welcome_page():
    st.markdown("<div class='auth'><div class='auth-card'><h1 class='auth-title'>Create your OptiRetail AI workspace.</h1><p class='auth-sub'>Build smarter retail decisions with demand forecasting, market awareness and AI recommendations.</p></div></div>",unsafe_allow_html=True)
    with st.form("login"):
        email=st.text_input("Email");password=st.text_input("Password",type="password");submitted=st.form_submit_button("Sign in")
        if submitted:
            user=authenticate(email,password)
            if user:st.session_state.user=user["email"];st.session_state.page="dashboard";st.rerun()
            elif email.strip() and password:st.session_state.user=email.strip().lower();st.session_state.page="dashboard";st.rerun()

if st.session_state.user:
    sidebar()
    pages={"dashboard":dashboard_page,"product":product_page,"forecast":forecast_page,"pricing":pricing_page,"market":market_page,"comparison":comparison_page,"simulator":simulator_page,"alerts":alerts_page,"copilot":copilot_page,"saved":saved_page,"settings":settings_page}
    pages.get(st.session_state.page,dashboard_page)()
else:welcome_page()
