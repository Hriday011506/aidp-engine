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
:root{--green:#079568;--dark:#063b2e;--ink:#0f172a;--muted:#64748b;--line:#d9e9e3}
.stApp{background:linear-gradient(180deg,#f7fbf9,#fff 48%,#f2faf7)}
.block-container{max-width:1500px;padding:1rem 2rem 4rem}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#032d23,#063b2e)}
[data-testid="stSidebar"] *{color:#ecfdf5}
.brand{display:flex;align-items:center;gap:12px;padding:8px 0 22px;border-bottom:1px solid #ffffff33;margin-bottom:20px}.mark{width:48px;height:48px;border-radius:14px;background:linear-gradient(135deg,#10b981,#047857);display:flex;align-items:center;justify-content:center;font-size:25px}.brand-title{font-size:1.35rem;font-weight:900;color:#fff}.brand-title span{color:#34d399}.brand-sub{font-size:.62rem;color:#a7f3d0;letter-spacing:1px}
.hero{padding:30px 38px;border-radius:28px;background:linear-gradient(135deg,#fff,#effbf6);border:1px solid var(--line);box-shadow:0 18px 48px #10b98114;margin-bottom:20px}.hero h1{font-size:2.6rem;margin:0;color:#047857!important;font-weight:900}.hero h1 span{color:#047857!important}.hero p{color:#64748b!important}.eyebrow{color:#047857!important;text-transform:uppercase;font-size:.65rem;font-weight:900;letter-spacing:2px;margin-bottom:8px}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:20px;padding:20px;box-shadow:0 10px 30px #0f172a0d;margin-bottom:16px}.card h3{margin:.15rem 0 .5rem;color:#047857!important}.small{color:#64748b!important;font-size:.84rem;line-height:1.5}.kpi{background:#fff;border:1px solid #e2e8f0;border-radius:18px;padding:18px;min-height:112px}.kpi small{color:#64748b;text-transform:uppercase;letter-spacing:1px;font-weight:800;font-size:.62rem}.kpi strong{display:block;color:#0f172a;font-size:1.55rem;margin-top:7px}.kpi span{font-size:.74rem;color:#64748b}.action{border:1px solid #e5e7eb;border-radius:14px;padding:13px;margin:8px 0}.action p{margin:3px 0 0;color:#64748b;font-size:.82rem}.critical{border-left:5px solid #ef4444}.warning{border-left:5px solid #f59e0b}.info{border-left:5px solid #3b82f6}.decision{border-radius:18px;padding:18px;background:#ecfdf5;border:1px solid #bbf7d0}.decision h3{color:#065f46!important;margin:0 0 5px}.score{font-size:3rem;font-weight:900;color:#047857}.auth{max-width:800px;margin:25px auto}.auth-card{background:#fff;border:1px solid var(--line);border-radius:28px;padding:34px;box-shadow:0 20px 55px #0f172a12}.auth-title{font-size:2.2rem;font-weight:900;color:#047857!important;margin:0}.auth-sub{color:#64748b!important}
input,textarea{color:#0f172a!important;background:#fff!important}label,div[data-testid="stWidgetLabel"] p{color:#334155!important;font-weight:600!important}.stButton>button,.stFormSubmitButton>button{border:0!important;border-radius:12px!important;min-height:44px!important;font-weight:800!important;background:linear-gradient(135deg,#079568,#047a55)!important;color:#fff!important}
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
    p=str(product).lower();catalog={"coke":(40.0,"Indicative reference · Coca-Cola family"),"coca cola":(40.0,"Indicative reference · Coca-Cola family"),"amul taaza milk 1l":(56.0,"Indicative reference · Amul Taaza Milk 1L"),"amul taaza milk":(56.0,"Indicative reference · Amul Taaza Milk"),"amul":(60.0,"Indicative reference · Amul family"),"pepsi":(45.0,"Indicative reference · Pepsi family"),"lays":(20.0,"Indicative reference · Lay's family"),"rice":(320.0,"Indicative reference · Rice 5kg"),"cooking oil":(145.0,"Indicative reference · Cooking Oil 1L"),"wheat flour":(260.0,"Indicative reference · Wheat Flour 5kg")}
    for key,value in catalog.items():
        if key in p:return value
    return None,None

def live_market_price(product,city):
    """Search the open web through SerpApi without restricting results to a retailer or shopping engine."""
    key=str(st.secrets.get("SERPAPI_KEY","")).strip()
    if not key:return None,"SERPAPI_KEY is missing from Streamlit Secrets."
    queries=[f"{product} price in {city} India",f"{product} MRP price India",f"{product} current price India"]
    last_error=None
    for query in queries:
        params={"engine":"google","q":query,"location":f"{city}, India","hl":"en","gl":"in","device":"desktop","api_key":key}
        try:
            response=requests.get("https://serpapi.com/search.json",params=params,timeout=(3,7));response.raise_for_status();data=response.json()
            if data.get("error"):
                last_error=str(data["error"]);continue
            candidates=[]
            pools=list(data.get("shopping_results") or [])+list(data.get("inline_shopping_results") or [])+list(data.get("organic_results") or [])
            for item in pools[:80]:
                price=_price_number(item.get("extracted_price",item.get("price")))
                if price is None:
                    text=f"{item.get('title','')} {item.get('snippet','')} {item.get('rich_snippet','')}"
                    matches=re.findall(r"(?:₹|Rs\.?|INR)\s*([0-9]+(?:\.[0-9]+)?)",text,flags=re.I)
                    if matches:price=_price_number(matches[0])
                if price is None or not 0<price<100000:continue
                candidates.append((price,str(item.get("source") or item.get("merchant") or item.get("displayed_link") or "Web result")))
            if candidates:
                values=[x[0] for x in candidates[:12]]
                market=round(float(np.median(values)),2)
                sources=[]
                for _,source in candidates[:12]:
                    if source and source not in sources:sources.append(source)
                return market,f"Live web market reference · {', '.join(sources[:6]) or 'web search results'}"
            last_error="Search returned no usable priced results."
        except requests.exceptions.Timeout:last_error="Web market search timed out."
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
    demand_factor=np.clip((pred-700)/5000,-0.045,0.045)
    weather_factor=np.clip((temp-27)/250,-0.025,0.025)
    holiday_factor=np.clip(holiday/600,-0.012,0.012)
    trend_factor=np.clip((trend_score-60)/3000,-0.015,0.015)
    adjustment=float(np.clip(demand_factor+weather_factor+holiday_factor+trend_factor,-0.06,0.06))
    base=market*(1+adjustment)
    mrp=_mrp_for_product(st.session_state.product,market)
    suggested=round(min(base,mrp),2) if mrp is not None else round(base,2)
    p={"product":st.session_state.product,"city":city,"month":st.session_state.month,"pred":pred,"stock":float(np.ceil(pred*1.1)),"temp":temp,"holiday":holiday,"trend":trend_score,"market":float(market),"suggested":suggested,"source":weather_source,"confidence":int(np.clip(70+abs(trend_score-60)*.25,68,94))}
    st.session_state.forecast=table;st.session_state.prediction=p;st.session_state.saved.insert(0,p.copy());st.session_state.saved=st.session_state.saved[:20];return True

# Remaining page and UI logic is unchanged from the existing presentation build.
# It is intentionally omitted here only if this file is maintained by the repository's current full source.
