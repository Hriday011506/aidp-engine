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
    # Never reuse the previous product's price. A failed lookup must not leak a stale value
    # from another product or city into the current analysis.
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
    p=str(product).lower()
    known={"coke":40.0,"coca cola":40.0,"amul taaza milk 1l":56.0,"amul taaza milk":56.0,"amul":60.0,"pepsi":45.0,"lays":20.0}
    for key,mrp in known.items():
        if key in p:return mrp
    return None

def generate():
    market=st.session_state.price
    if market is None:return False
    table,city,weather_source=make_forecast(st.session_state.product,st.session_state.city);row=table.loc[table.Month==st.session_state.month].iloc[0];pred=float(row["Forecast demand"])
    base=market*(1+np.clip((pred-700)/7000,-0.06,0.06))
    mrp=_mrp_for_product(st.session_state.product,market)
    if mrp is not None:
        suggested=round(min(base,mrp),2)
    else:
        suggested=round(base,2)
    p={"product":st.session_state.product,"city":city,"month":st.session_state.month,"pred":pred,"stock":float(np.ceil(pred*1.1)),"temp":float(row["Temperature (°C)"]),"holiday":int(row["Holiday days"]),"trend":int(row["Trend score"]),"market":float(market),"suggested":suggested,"source":weather_source,"confidence":int(np.clip(70+abs(int(row["Trend score"])-60)*.25,68,94))};st.session_state.forecast=table;st.session_state.prediction=p;st.session_state.saved.insert(0,p.copy());st.session_state.saved=st.session_state.saved[:20];return True

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
    a.append(("warning","Review pricing","Demand-aware price is above the market reference.") if p["suggested"]>p["market"]*1.02 else ("info","Keep price near market","Limited price change is indicated."));return a[:4]
def header(eyebrow,title,subtitle):st.markdown(f"<div class='hero'><div class='eyebrow'>{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></div>",unsafe_allow_html=True)
def logo():st.markdown('<div class="brand"><div class="mark">🛒</div><div><div class="brand-title">OptiRetail <span>AI</span></div><div class="brand-sub">SMARTER DECISIONS · STRONGER RETAIL</div></div></div>',unsafe_allow_html=True)
def inputs():
    st.markdown("<div class='card'><div class='eyebrow'>ANALYSIS INPUTS</div><h3>Choose product, location and forecast month</h3>",unsafe_allow_html=True);a,b,c=st.columns([1.5,1,1])
    with a:product=st.text_input("Product",value=st.session_state.product,key="product_input")
    with b:city=st.text_input("City",value=st.session_state.city,key="city_input")
    with c:month=st.selectbox("Forecast month",MONTHS,index=MONTHS.index(st.session_state.month) if st.session_state.month in MONTHS else 0,key="month_input")
    product=product.strip() or "Wheat Flour";city=city.strip() or "Jaipur";old=st.session_state.prediction;changed=bool(old) and (old.get("product")!=product or old.get("city")!=city or old.get("month")!=month);st.session_state.product,st.session_state.city,st.session_state.month=product,city,month
    if changed:st.session_state.prediction=None;st.session_state.forecast=None;st.session_state.price=None;st.session_state.price_source=None
    x,y=st.columns(2)
    with x:
        if st.button("Refresh live market price",use_container_width=True,key="price_button"):
            price,message=live_market_price(product,city)
            if price is None:st.session_state.price=None;st.session_state.price_source=None;st.error(message)
            else:st.session_state.price=price;st.session_state.price_source=message;st.success(f"Market reference: ₹{price:,.2f} · {message}")
    with y:
        if st.button("Generate AI Decision →",use_container_width=True,key="generate_button"):
            if st.session_state.price is None:st.error("Refresh the market price first. A market reference is required for an AI decision.")
            elif generate():st.success("AI decision generated successfully.")
    if st.session_state.price is not None:st.markdown(f"<div style='background:#dcfce7;border-radius:12px;padding:14px 18px;color:#047857;font-weight:800'>Market reference: ₹{st.session_state.price:,.2f} · {html.escape(str(st.session_state.price_source or 'Live web market data'))}</div>",unsafe_allow_html=True)
    st.markdown("</div>",unsafe_allow_html=True)
def simulator():
    p=normalize_prediction(st.session_state.prediction)
    if not p:st.info("Generate a decision first to use the simulator.");return
    header("AI SIMULATOR","Test decisions before <span>you act.</span>","Change demand, price and promotion assumptions without changing the saved forecast.");a,b,c=st.columns(3)
    with a:price_factor=st.slider("Price change",-20,20,0,1,format="%d%%")
    with b:demand_factor=st.slider("Demand change",-30,30,0,1,format="%d%%")
    with c:promotion=st.slider("Promotion intensity",0,30,0,1,format="%d%%")
    simulated=max(0,p["pred"]*(1+demand_factor/100)*(1+promotion/250)*(1-price_factor/400));stock=int(np.ceil(simulated*1.1));x,y=st.columns(2)
    with x:st.markdown(f"<div class='kpi'><small>SIMULATED DEMAND</small><strong>{int(simulated):,} units</strong><span>Base forecast {int(p['pred']):,}</span></div>",unsafe_allow_html=True)
    with y:st.markdown(f"<div class='kpi'><small>RECOMMENDED STOCK</small><strong>{stock:,} units</strong><span>Includes 10% buffer</span></div>",unsafe_allow_html=True)
def dashboard():
    p=normalize_prediction(st.session_state.prediction);cols=st.columns(4);values=[("FORECAST DEMAND",f"{int(p['pred']):,} units" if p else "—","Current month"),("RECOMMENDED STOCK",f"{int(p['stock']):,} units" if p else "—","10% planning buffer"),("MARKET REFERENCE",f"₹{p['market']:,.2f}" if p else "—","Live web market data"),("CONFIDENCE",f"{p['confidence']}%" if p else "—","Model confidence")]
    for col,(label,value,note) in zip(cols,values):col.markdown(f"<div class='kpi'><small>{label}</small><strong>{value}</strong><span>{note}</span></div>",unsafe_allow_html=True)
    a,b=st.columns([1.4,1])
    with a:
        st.markdown("<div class='card'><div class='eyebrow'>AI ACTION CENTER</div><h3>Today's AI actions</h3>",unsafe_allow_html=True)
        for typ,title,desc in action_list(p):st.markdown(f"<div class='action {typ}'><b>{title}</b><p>{desc}</p></div>",unsafe_allow_html=True)
        st.markdown("</div>",unsafe_allow_html=True)
    with b:
        st.markdown("<div class='card'><div class='eyebrow'>RETAIL HEALTH</div><h3>Business health score</h3>",unsafe_allow_html=True)
        if p:score=health(p);st.markdown(f"<div class='score'>{score}<span style='font-size:1rem;color:#64748b'>/100</span></div>",unsafe_allow_html=True);st.progress(score,text="Overall health")
        else:st.info("Generate a decision to calculate the score.")
        st.markdown("</div>",unsafe_allow_html=True)
    if not p or st.session_state.forecast is None:st.info("Open Product Analysis and generate an AI decision.");return
    a,b=st.columns([1.4,1])
    with a:st.markdown("<div class='card'><div class='eyebrow'>FORECAST OUTLOOK</div><h3>12-month demand</h3>",unsafe_allow_html=True);st.line_chart(st.session_state.forecast.set_index("Month")[["Forecast demand"]],height=300);st.markdown("</div>",unsafe_allow_html=True)
    with b:
        st.markdown("<div class='card'><div class='eyebrow'>WHY THIS PREDICTION?</div><h3>Explainable AI</h3>",unsafe_allow_html=True)
        for name,value in [("Product trend",f"{p['trend']}/100"),("Holiday days",p["holiday"]),("Temperature",f"{p['temp']:.1f} °C"),("Confidence",f"{p['confidence']}%")]:st.markdown(f"<p class='small'><b>{name}</b> <span style='float:right'>{value}</span></p>",unsafe_allow_html=True)
        st.markdown("</div>",unsafe_allow_html=True)
    simulator()
def product_page():
    header("PRODUCT ANALYSIS","Analyze a product with <span>live market context.</span>","Select a product and location, refresh the live market reference, then generate an explainable AI decision.");inputs();p=normalize_prediction(st.session_state.prediction)
    if matches(p):st.markdown(f"<div class='card'><div class='eyebrow'>LATEST RESULT</div><h3>{html.escape(str(p.get('product','')))} · {html.escape(str(p.get('city','')))} · {html.escape(str(p.get('month','')))}</h3><p class='small'>Forecast demand: <b>{int(p.get('pred',0)):,}</b> units · Weather: <b>{p.get('temp',0):.1f} °C</b> · Holiday days: <b>{p.get('holiday',0)}</b> · Confidence: <b>{p.get('confidence',0)}%</b> · Market: <b>₹{p.get('market',0):,.2f}</b> · Weather source: <b>{html.escape(str(p.get('source','')))}</b></p></div>",unsafe_allow_html=True)
def forecast_page():
    header("DEMAND FORECASTING","See demand change <span>across the year.</span>","Each month gets its own weather value; no repeated placeholder.");p=normalize_prediction(st.session_state.prediction)
    if not p or st.session_state.forecast is None:st.info("Generate a decision first.");return
    table=st.session_state.forecast;st.line_chart(table.set_index("Month")[["Temperature (°C)"]],height=250);st.line_chart(table.set_index("Month")[["Forecast demand"]],height=280);st.dataframe(table.assign(**{"Recommended stock":np.ceil(table["Forecast demand"]*1.1).astype(int)}),use_container_width=True,hide_index=True)
def pricing_page():
    header("DYNAMIC PRICING","Price with <span>market awareness.</span>","Compare the market reference with the demand-aware recommendation.");p=normalize_prediction(st.session_state.prediction)
    if not p:st.info("Generate a decision first.");return
    a,b,c=st.columns(3);a.metric("Market reference",f"₹{p['market']:,.2f}");b.metric("Suggested price",f"₹{p['suggested']:,.2f}");c.metric("Difference",f"{((p['suggested']/p['market'])-1)*100:+.1f}%");st.markdown(f"<div class='decision'><h3>Pricing rationale</h3><p>Recommendation uses the market reference of ₹{p['market']:,.2f} and the current demand forecast of {int(p['pred']):,} units.</p></div>",unsafe_allow_html=True)
def compare_page():
    header("PRODUCT COMPARISON","Compare products with <span>one model.</span>","Compare expected demand, peak month and inventory requirement.");a,b,c=st.columns(3)
    with a:p1=st.text_input("Product 1",value=st.session_state.product,key="cmp1")
    with b:p2=st.text_input("Product 2",value="Rice 5kg",key="cmp2")
    with c:p3=st.text_input("Product 3",value="Cooking Oil 1L",key="cmp3")
    if st.button("Compare products →",use_container_width=True,key="compare_button"):
        rows=[]
        for product in [p1,p2,p3]:
            if product.strip():table,resolved_city,_=make_forecast(product,st.session_state.city);peak=table.loc[table["Forecast demand"].idxmax()];rows.append({"Product":product,"City":resolved_city,"12-month demand":int(table["Forecast demand"].sum()),"Peak month":peak.Month,"Peak demand":int(peak["Forecast demand"]),"Recommended stock":int(np.ceil(table["Forecast demand"].sum()*1.1))})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
def alerts_page():
    header("ALERTS & RISK","See risks before they become <span>retail problems.</span>","Early-warning signals generated from the current forecast.");p=normalize_prediction(st.session_state.prediction)
    if not p:st.info("Generate a decision first.");return
    for typ,title,desc in action_list(p):st.markdown(f"<div class='card'><h3>{'🔴' if typ=='critical' else '🟠' if typ=='warning' else '🔵'} {title}</h3><p class='small'>{desc}</p></div>",unsafe_allow_html=True)
def copilot_page():
    header("AI DECISION COPILOT","Ask OptiRetail AI <span>in plain language.</span>","Get a concise answer from the current forecast context.");question=st.text_input("Ask a question",placeholder="Why should I increase stock?",key="question")
    if st.button("Analyze question →",use_container_width=True,key="ask"):
        p=normalize_prediction(st.session_state.prediction)
        if not p:st.info("Generate a decision first.")
        elif not question.strip():st.warning("Enter a question first.")
        else:
            query=question.lower()
            if "stock" in query or "inventory" in query:answer=f"Plan about {int(p['stock']):,} units for {p['product']} in {p['month']}. Forecast demand is {int(p['pred']):,} units with a 10% buffer."
            elif "weather" in query:answer=f"{p['month']} is estimated at {p['temp']:.1f} °C using {p['source']}."
            elif "price" in query:answer=f"The market reference is ₹{p['market']:,.2f}; the demand-aware recommendation is ₹{p['suggested']:,.2f}."
            else:answer=f"The model forecasts {int(p['pred']):,} units with {p['confidence']}% confidence. Trend is {p['trend']}/100 and temperature is {p['temp']:.1f} °C."
            st.markdown(f"<div class='decision'><h3>OptiRetail AI</h3><p>{html.escape(answer)}</p></div>",unsafe_allow_html=True)
def settings_page():
    header("SETTINGS","Configure your <span>workspace.</span>","Presentation build settings and account information.");user=st.session_state.user or {};st.text_input("Email",value=user.get("email",""),disabled=True);st.text_input("GST Number",value=user.get("gst",""),disabled=True);st.text_input("Annual Turnover",value=user.get("turnover",""),disabled=True);st.info("MongoDB and Brevo are disabled in this presentation build. Account data is kept in the Streamlit session.")
with st.sidebar:
    logo()
    if st.session_state.user:
        st.caption(st.session_state.user.get("email","Business User"));items=[("Dashboard","dashboard"),("Product Analysis","product"),("Demand Forecasting","forecast"),("Dynamic Pricing","pricing"),("Market Insights","market"),("Product Comparison","compare"),("AI Simulator","simulator"),("Alerts & Risk","alerts"),("AI Decision Copilot","copilot"),("Saved Analyses","saved"),("Settings","settings")]
        for label,target in items:
            if st.button(label,use_container_width=True,key="nav_"+target):st.session_state.page=target;st.rerun()
        if st.button("Sign out",use_container_width=True,key="signout"):st.session_state.user=None;st.session_state.page="welcome";st.rerun()
if st.session_state.page=="welcome":
    header("OPTIRETAIL AI","Turn market data into <span>smarter decisions.</span>","Forecast demand, use live web market intelligence, simulate scenarios, detect risks and explain AI recommendations.");a,b,c,d=st.columns(4)
    for col,title,desc in zip([a,b,c,d],["Predict","Decide","Simulate","Explain"],["12-month demand forecast","AI Action Center","What-if scenarios","Confidence and drivers"]):col.markdown(f"<div class='card'><div class='eyebrow'>AI</div><h3>{title}</h3><p class='small'>{desc}</p></div>",unsafe_allow_html=True)
    x,y=st.columns(2)
    with x:
        if st.button("Get Started →",use_container_width=True,key="start"):st.session_state.page="signup";st.rerun()
    with y:
        if st.button("Sign in",use_container_width=True,key="home_login"):st.session_state.page="login";st.rerun()
elif st.session_state.page=="login":
    st.markdown("<div class='auth'><div class='auth-card'><div class='eyebrow'>WELCOME BACK</div><h1 class='auth-title'>Sign in to OptiRetail AI</h1><p class='auth-sub'>Access your retail intelligence workspace.</p>",unsafe_allow_html=True)
    with st.form("login_form"):email=st.text_input("Email address");password=st.text_input("Password",type="password");submit=st.form_submit_button("Sign in",use_container_width=True)
    if submit:
        user=authenticate(email,password)
        if user:st.session_state.user=user;st.session_state.page="dashboard";st.rerun()
        else:st.error("Invalid credentials. Create an account first in this session.")
    if st.button("Create an account →",key="login_signup"):st.session_state.page="signup";st.rerun()
    st.markdown("</div></div>",unsafe_allow_html=True)
elif st.session_state.page=="signup":
    st.markdown("<div class='auth'><div class='auth-card'><div class='eyebrow'>BUSINESS ONBOARDING</div><h1 class='auth-title'>Create your OptiRetail AI workspace</h1><p class='auth-sub'>Presentation-ready account with session storage.</p>",unsafe_allow_html=True)
    with st.form("signup_form"):email=st.text_input("Email address");password=st.text_input("Password",type="password");gst=st.text_input("GST Number");turnover=st.selectbox("Annual Turnover",["1–5 Lakh","5–10 Lakh","10–15 Lakh","15–50 Lakh","50 Lakh+"]);submit=st.form_submit_button("Create account",use_container_width=True)
    if submit:
        if not email or "@" not in email or not password or not gst:st.error("Please enter a valid email, password and GST number.")
        else:
            ok,message=signup(email,password,gst,turnover)
            if ok:st.success(message);st.session_state.page="login";st.rerun()
            else:st.error(message)
    st.markdown("</div></div>",unsafe_allow_html=True)
elif st.session_state.user:
    if st.session_state.page=="dashboard":header("DASHBOARD","Turn market data into <span>smarter decisions.</span>","AI Action Center, health score, explainable forecast and simulator.");dashboard()
    elif st.session_state.page=="product":product_page()
    elif st.session_state.page=="forecast":forecast_page()
    elif st.session_state.page=="pricing":pricing_page()
    elif st.session_state.page=="simulator":simulator()
    elif st.session_state.page=="market":
        header("MARKET INSIGHTS","Live web <span>market intelligence.</span>","Search the open web for a current market reference without restricting the lookup to a specific retailer.");price,message=live_market_price(st.session_state.product,st.session_state.city)
        if price is None:st.error(message)
        else:st.session_state.price=price;st.session_state.price_source=message;st.metric("Current market reference",f"₹{price:,.2f}");st.caption(message)
    elif st.session_state.page=="compare":compare_page()
    elif st.session_state.page=="alerts":alerts_page()
    elif st.session_state.page=="copilot":copilot_page()
    elif st.session_state.page=="saved":
        header("SAVED ANALYSES","Keep your important <span>AI decisions.</span>","Recent session decisions.")
        if not st.session_state.saved:st.info("No saved analyses yet.")
        for saved in st.session_state.saved[:10]:
            p=normalize_prediction(saved)
            if p:st.markdown(f"<div class='card'><h3>{html.escape(str(p['product']))} · {html.escape(str(p['city']))} · {p['month']}</h3><p class='small'>Demand {int(p['pred']):,} · Stock {int(p['stock']):,} · Market ₹{p['market']:,.2f} · Confidence {p['confidence']}%</p></div>",unsafe_allow_html=True)
    elif st.session_state.page=="settings":settings_page()
    else:st.session_state.page="dashboard";st.rerun()
else:st.session_state.page="welcome";st.rerun()
