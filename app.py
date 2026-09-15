import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import calendar
import holidays
import secrets
import time
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
BREVO_API_KEY = st.secrets.get("BREVO_API_KEY", "")
BREVO_SENDER_EMAIL = st.secrets.get("BREVO_SENDER_EMAIL", "")
BREVO_SENDER_NAME = st.secrets.get("BREVO_SENDER_NAME", "OptiRetail AI")

DEFAULTS = {
    "user": None,
    "page": "welcome",
    "product": "Wheat Flour",
    "city": "Jaipur",
    "month_name": datetime.datetime.now().strftime("%B"),
    "last_prediction": None,
    "current_price": None,
    "market_data": None,
    "otp_email": None,
    "otp_value": None,
    "otp_created_at": None,
    "pending_signup": None,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');
:root{--ink:#10231f;--muted:#6b7b76;--green:#07865f;--green2:#056b4c;--line:#dcebe5}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--ink)}
.stApp{background:linear-gradient(180deg,#f7fbf9 0%,#ffffff 44%,#f3faf7 100%)}
.block-container{max-width:1500px;padding:1rem 2rem 4rem}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#052d23,#06392e);border-right:1px solid rgba(255,255,255,.08)}
[data-testid="stSidebar"] *{color:#ecfdf5}
.sidebar-logo{font-family:'Plus Jakarta Sans';font-size:1.55rem;font-weight:800;letter-spacing:-.8px}.sidebar-sub{font-size:.68rem;color:#a7f3d0;margin-top:-4px}
.hero{padding:36px 40px;border-radius:30px;background:linear-gradient(135deg,#ffffff,#effbf6);border:1px solid var(--line);box-shadow:0 22px 60px rgba(16,185,129,.08);margin-bottom:22px}
.hero h1{font-family:'Plus Jakarta Sans';font-size:2.75rem;line-height:1.07;letter-spacing:-2.1px;margin:0;color:#0f172a}.hero h1 span{color:var(--green2)}.hero p{color:#64748b;font-size:1rem;max-width:950px;margin:.7rem 0 0}
.eyebrow{color:var(--green2);text-transform:uppercase;font-size:.65rem;font-weight:800;letter-spacing:2px;margin-bottom:9px}.section-title{font-family:'Plus Jakarta Sans';font-size:1.2rem;font-weight:800;color:#0f172a;margin:.35rem 0 .9rem}
.card{background:rgba(255,255,255,.98);border:1px solid #e2e8f0;border-radius:22px;padding:22px;box-shadow:0 12px 34px rgba(15,23,42,.05);margin-bottom:16px}.card h3{margin:.15rem 0 .45rem;font-family:'Plus Jakarta Sans';font-size:1.05rem}
.kpi{background:#fff;border:1px solid #e2e8f0;border-radius:18px;padding:18px;min-height:118px;box-shadow:0 10px 24px rgba(15,23,42,.04)}.kpi small{color:#64748b;text-transform:uppercase;letter-spacing:1px;font-weight:800;font-size:.62rem}.kpi strong{display:block;font-family:'Plus Jakarta Sans';color:#0f172a;font-size:1.72rem;margin-top:7px}.kpi span{font-size:.75rem;color:#64748b}
.signal{padding:12px 0;border-bottom:1px solid #eef2f4}.signal:last-child{border-bottom:0}.signal b{color:#0f172a}.signal span{color:#64748b}
.badge{display:inline-flex;align-items:center;gap:6px;padding:6px 10px;border-radius:999px;font-size:.65rem;font-weight:800}.badge.live{background:#ecfdf5;color:#047857;border:1px solid #bbf7d0}.badge.warn{background:#fff7ed;color:#c2410c;border:1px solid #fed7aa}.badge.info{background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe}
.decision{border-radius:18px;padding:18px;background:linear-gradient(135deg,#ecfdf5,#f0fdfa);border:1px solid #bbf7d0}.decision h3{margin:0 0 5px;color:#065f46;font-family:'Plus Jakarta Sans'}.decision p{margin:0;color:#475569;font-size:.9rem;line-height:1.55}.small-note{color:#64748b;font-size:.78rem}
.auth-wrap{max-width:720px;margin:28px auto}.auth-card{background:rgba(255,255,255,.99);border:1px solid var(--line);border-radius:28px;padding:36px;box-shadow:0 20px 55px rgba(15,23,42,.06)}.auth-title{font-family:'Plus Jakarta Sans';font-size:2rem;font-weight:800;letter-spacing:-1px;color:#0f172a;margin:0}.auth-sub{color:#64748b;margin:.4rem 0 1.4rem}
.stButton>button,.stFormSubmitButton>button{border:0!important;border-radius:12px!important;min-height:44px!important;font-weight:800!important;background:linear-gradient(135deg,#079568,#047a55)!important;color:#fff!important;box-shadow:0 8px 22px rgba(5,150,105,.16)}.stButton>button:hover,.stFormSubmitButton>button:hover{transform:translateY(-1px)}
div[data-testid="stTextInput"] input,div[data-testid="stNumberInput"] input,div[data-testid="stSelectbox"] div[data-baseweb="select"]>div,div[data-testid="stTextArea"] textarea{background:#fff!important;color:#0f172a!important;border:1px solid #cfe0d9!important;border-radius:12px!important;caret-color:#047857!important;box-shadow:none!important;font-size:1rem!important}
div[data-testid="stTextInput"] input:focus,div[data-testid="stNumberInput"] input:focus,div[data-testid="stTextArea"] textarea:focus{border:2px solid #56b99a!important;box-shadow:0 0 0 2px rgba(5,150,105,.08)!important}
div[data-testid="stTextInput"] input::placeholder,div[data-testid="stNumberInput"] input::placeholder,div[data-testid="stTextArea"] textarea::placeholder{color:#94a3b8!important;opacity:1!important}label,div[data-testid="stWidgetLabel"] p{color:#334155!important;font-weight:600!important}div[data-baseweb="select"] span{color:#0f172a!important}
[data-testid="stMetricValue"]{color:#0f172a}[data-testid="stMetricLabel"]{color:#64748b}[data-testid="stDataFrame"]{border-radius:16px;overflow:hidden}footer{visibility:hidden}
</style>
""",unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_database():
    if not MONGO_URI or MongoClient is None: return None,"MONGO_URI missing or PyMongo unavailable"
    try:
        client=MongoClient(MONGO_URI,serverSelectionTimeoutMS=8000,connectTimeoutMS=8000,socketTimeoutMS=8000)
        client.admin.command("ping")
        return client["optiretail_ai"],"Connected"
    except Exception as exc:
        return None,f"MongoDB connection failed: {type(exc).__name__}: {exc}"

db,mongo_status=get_database(); users_collection=db["users"] if db is not None else None; predictions_collection=db["predictions"] if db is not None else None

def generate_otp(): return f"{secrets.randbelow(1_000_000):06d}"
def otp_is_valid(created_at,ttl_seconds=300): return bool(created_at) and (time.time()-created_at)<=ttl_seconds

def send_otp_email(to_email,otp):
    if not BREVO_API_KEY or not BREVO_SENDER_EMAIL: return False,"Brevo email settings are missing in Streamlit Secrets."
    try:
        r=requests.post("https://api.brevo.com/v3/smtp/email",headers={"accept":"application/json","api-key":BREVO_API_KEY,"content-type":"application/json"},json={"sender":{"name":BREVO_SENDER_NAME,"email":BREVO_SENDER_EMAIL},"to":[{"email":to_email}],"subject":"Your OptiRetail AI verification code","htmlContent":f"<div style='font-family:Arial;padding:24px'><h2>Verify your OptiRetail AI account</h2><p>Your verification code is:</p><div style='font-size:32px;font-weight:700;letter-spacing:8px;color:#047857'>{otp}</div><p>This code expires in 5 minutes.</p></div>"},timeout=10)
        if 200<=r.status_code<300: return True,"OTP sent to your email."
        return False,f"Email service error ({r.status_code}). Verify your Brevo sender settings."
    except Exception as exc: return False,f"Could not send OTP: {exc}"

def save_user(email,password,gst,turnover):
    if users_collection is None: return False,"MongoDB is unavailable. Check MONGO_URI and Atlas Network Access."
    try:
        email=email.strip().lower()
        if users_collection.find_one({"email":email}): return False,"An account with this email already exists."
        users_collection.insert_one({"email":email,"password":password,"gst":gst.strip(),"turnover":turnover,"created_at":datetime.datetime.now(datetime.timezone.utc)})
        return True,"Account created successfully."
    except Exception as exc: return False,f"Could not save account: {exc}"

def login(email,password):
    if users_collection is None:return None
    try:return users_collection.find_one({"email":email.strip().lower(),"password":password})
    except Exception:return None

@st.cache_data(ttl=900,show_spinner=False)
def fetch_product_price(product_name):
    if not product_name.strip() or not SERPAPI_KEY or GoogleSearch is None:return None
    try:
        result=GoogleSearch({"engine":"google_shopping_light","q":f"{product_name} price","gl":"in","hl":"en","num":8,"api_key":SERPAPI_KEY}).get_dict();tokens=[t for t in product_name.lower().split() if len(t)>2];candidates=[]
        for item in result.get("shopping_results",[]):
            try:value=float(str(item.get("extracted_price",item.get("price"))).replace(",","").replace("₹","").strip())
            except (TypeError,ValueError):continue
            if value<=0:continue
            title=str(item.get("title","" )).lower();source=str(item.get("source","")).lower();score=sum(tok in title for tok in tokens)*3+(10 if product_name.lower() in title else 0)+(1 if "india" in title or "india" in source else 0);candidates.append((score,value,item.get("title",""),item.get("source",""),item.get("link","")))
        if not candidates:return None
        candidates.sort(reverse=True);best=candidates[0][0];top=[x for x in candidates if x[0]==best][:5];return {"price":float(np.median([x[1] for x in top])),"results":top}
    except Exception:return None

@st.cache_data(ttl=900,show_spinner=False)
def get_weather(city):
    try:
        r=requests.get("https://geocoding-api.open-meteo.com/v1/search",params={"name":city,"count":1},timeout=4);r.raise_for_status();hits=r.json().get("results",[])
        if not hits:return 25.0
        lat,lon=hits[0]["latitude"],hits[0]["longitude"];w=requests.get("https://api.open-meteo.com/v1/forecast",params={"latitude":lat,"longitude":lon,"current_weather":"true"},timeout=4);w.raise_for_status();return float(w.json()["current_weather"]["temperature"])
    except Exception:return 25.0

@st.cache_data
def get_holidays(year,month):
    india=holidays.India(years=year);days=calendar.monthrange(year,month)[1];return sum(1 for d in range(1,days+1) if datetime.date(year,month,d).weekday()>=5 or datetime.date(year,month,d) in india)

@st.cache_data
def trend_score(product):
    rng=np.random.default_rng(sum(ord(c) for c in product));return int(rng.integers(30,90))

@st.cache_data
def load_data():
    rng=np.random.default_rng(42);df=pd.DataFrame({"holiday_count":rng.integers(0,10,240),"avg_temp":rng.integers(10,40,240),"viral_score":rng.integers(0,100,240)});seasonal=np.maximum(0,22-np.abs(df["avg_temp"]-28));df["sales"]=180+df["holiday_count"]*42+df["viral_score"]*4.8+seasonal*11+rng.normal(0,28,240);return df

@st.cache_resource
def train_model(df):
    m=RandomForestRegressor(n_estimators=250,max_depth=12,min_samples_leaf=2,random_state=42,n_jobs=-1);m.fit(df[["holiday_count","avg_temp","viral_score"]],df["sales"]);return m
model=train_model(load_data())

with st.sidebar:
    st.markdown("<div class='sidebar-logo'>📊 OptiRetail AI</div><div class='sidebar-sub'>Analyze · Predict · Price</div>",unsafe_allow_html=True);st.divider()
    if st.session_state.user:
        st.caption(st.session_state.user.get("email","Business User"));nav=[("Dashboard","dashboard"),("Product Analysis","product"),("Demand Forecasting","forecast"),("Dynamic Pricing","pricing"),("Market Insights","market"),("Saved Analyses","saved")]
        for label,target in nav:
            if st.button(label,use_container_width=True,key=f"nav_{target}"):st.session_state.page=target;st.rerun()
        st.divider()
        if st.button("Settings",use_container_width=True,key="nav_settings"):st.session_state.page="settings";st.rerun()
        if st.button("Sign out",use_container_width=True,key="nav_signout"):st.session_state.user=None;st.session_state.page="welcome";st.rerun()
    else:st.markdown("### Smarter retail decisions");st.caption("Forecast demand. Optimize inventory. Price with confidence.")

if st.session_state.page=="welcome":
    st.markdown("<div class='hero'><div class='eyebrow'>OPTIRETAIL AI</div><h1>Turn market data into <span>smarter decisions.</span></h1><p>Forecast demand, understand market signals, optimize inventory and make practical pricing decisions from one clean workspace.</p></div>",unsafe_allow_html=True);a,b,c=st.columns(3)
    a.markdown("<div class='card'><div class='eyebrow'>01 · FORECAST</div><h3>Know what will move.</h3><p class='small-note'>AI demand forecasting uses holiday, weather and trend signals.</p></div>",unsafe_allow_html=True);b.markdown("<div class='card'><div class='eyebrow'>02 · INVENTORY</div><h3>Stock with confidence.</h3><p class='small-note'>Translate predicted demand into a practical planning quantity.</p></div>",unsafe_allow_html=True);c.markdown("<div class='card'><div class='eyebrow'>03 · PRICE</div><h3>Price with context.</h3><p class='small-note'>Use a live market reference from SerpAPI and bounded demand response.</p></div>",unsafe_allow_html=True);x,y=st.columns(2)
    if x.button("Login →",use_container_width=True,key="home_login"):st.session_state.page="login";st.rerun()
    if y.button("Create account →",use_container_width=True,key="home_signup"):st.session_state.page="signup";st.rerun()

elif st.session_state.page=="login":
    st.markdown("<div class='auth-wrap'><div class='auth-card'><div class='eyebrow'>WELCOME BACK</div><h2 class='auth-title'>Sign in to OptiRetail AI</h2><p class='auth-sub'>Access your retail intelligence workspace.</p>",unsafe_allow_html=True)
    with st.form("login_form"):
        email=st.text_input("Email address",placeholder="you@company.com");password=st.text_input("Password",type="password",placeholder="Enter your password")
        if st.form_submit_button("Sign in",use_container_width=True):
            user=login(email,password)
            if user:st.session_state.user=user;st.session_state.page="dashboard";st.rerun()
            elif users_collection is None:st.error(mongo_status)
            else:st.error("Invalid credentials. Please check your email and password.")
    if st.button("Create an account →",key="login_signup"):st.session_state.page="signup";st.rerun()
    if st.button("← Back to home",key="login_back"):st.session_state.page="welcome";st.rerun()
    st.markdown("</div></div>",unsafe_allow_html=True)

elif st.session_state.page=="signup":
    st.markdown("<div class='auth-wrap'><div class='auth-card'><div class='eyebrow'>BUSINESS ONBOARDING</div><h2 class='auth-title'>Create your OptiRetail AI workspace</h2><p class='auth-sub'>Verify your email before your account is saved.</p>",unsafe_allow_html=True)
    if not st.session_state.otp_value:
        with st.form("signup_form"):
            email=st.text_input("Email address",placeholder="you@company.com");password=st.text_input("Password",type="password",placeholder="Create a password");gst=st.text_input("GST Number",placeholder="Enter GST number");turnover=st.selectbox("Annual Turnover",["1–5 Lakh","5–10 Lakh","10–15 Lakh","15–50 Lakh","50 Lakh+"])
            if st.form_submit_button("Send verification OTP",use_container_width=True):
                email=email.strip().lower()
                if not email or not password or not gst:st.error("Please fill in Email, Password and GST Number.")
                elif users_collection is None:st.error(mongo_status)
                else:
                    try:existing=users_collection.find_one({"email":email})
                    except Exception as exc:existing=None;st.error(f"MongoDB error: {exc}")
                    if existing:st.error("An account with this email already exists.")
                    else:
                        otp=generate_otp();ok,msg=send_otp_email(email,otp)
                        if ok:st.session_state.otp_email=email;st.session_state.otp_value=otp;st.session_state.otp_created_at=time.time();st.session_state.pending_signup={"email":email,"password":password,"gst":gst,"turnover":turnover};st.rerun()
                        else:st.error(msg)
    else:
        st.success(f"Verification code sent to {st.session_state.otp_email}")
        with st.form("otp_form"):
            otp_input=st.text_input("6-digit OTP",max_chars=6,placeholder="000000");verify=st.form_submit_button("Verify email & create account",use_container_width=True)
        if verify:
            created=st.session_state.otp_created_at
            if not created or not otp_is_valid(created):st.error("OTP expired. Please request a new one.")
            elif otp_input.strip()!=st.session_state.otp_value:st.error("Incorrect OTP. Please try again.")
            else:
                p=st.session_state.pending_signup;ok,msg=save_user(**p)
                if ok:st.session_state.otp_value=None;st.session_state.otp_email=None;st.session_state.otp_created_at=None;st.session_state.pending_signup=None;st.session_state.page="login";st.success(msg);st.rerun()
                else:st.error(msg)
        if st.button("Resend OTP",key="resend_otp"):
            p=st.session_state.pending_signup;new_otp=generate_otp();ok,msg=send_otp_email(p["email"],new_otp)
            if ok:st.session_state.otp_value=new_otp;st.session_state.otp_created_at=time.time();st.success("New OTP sent.");st.rerun()
            else:st.error(msg)
    if st.button("Already have an account? Sign in →",key="signup_login"):st.session_state.otp_value=None;st.session_state.pending_signup=None;st.session_state.page="login";st.rerun()
    if st.button("← Back to home",key="signup_back"):st.session_state.otp_value=None;st.session_state.pending_signup=None;st.session_state.page="welcome";st.rerun()
    st.markdown("</div></div>",unsafe_allow_html=True)

elif st.session_state.page=="settings" and st.session_state.user:
    st.markdown("<div class='hero'><div class='eyebrow'>WORKSPACE</div><h1>Business settings</h1><p>Manage your OptiRetail AI workspace and connected services.</p></div>",unsafe_allow_html=True);u=st.session_state.user;a,b,c=st.columns(3);a.metric("Account","Active");b.metric("Email",u.get("email","N/A"));c.metric("Turnover",u.get("turnover","N/A"));mongo_class="live" if db is not None else "warn";mongo_badge="CONNECTED" if db is not None else "UNAVAILABLE";serp_class="live" if SERPAPI_KEY else "warn";serp_badge="CONFIGURED" if SERPAPI_KEY else "MISSING";brevo_class="live" if BREVO_API_KEY and BREVO_SENDER_EMAIL else "warn";brevo_badge="CONFIGURED" if BREVO_API_KEY and BREVO_SENDER_EMAIL else "MISSING";st.markdown(f"<div class='card'><div class='eyebrow'>DATA SERVICES</div><h3>Connected services</h3><div class='signal'><b>MongoDB</b><span class='badge {mongo_class}'>{mongo_badge}</span></div><div class='signal'><b>SerpAPI</b><span class='badge {serp_class}'>{serp_badge}</span></div><div class='signal'><b>Brevo Email OTP</b><span class='badge {brevo_class}'>{brevo_badge}</span></div><div class='small-note' style='margin-top:10px'>{mongo_status}</div></div>",unsafe_allow_html=True)

# Shared analysis must be defined before the page branch that calls it.
def render_analysis():
    st.markdown("<div class='section-title'>Product analysis</div>",unsafe_allow_html=True);c1,c2,c3=st.columns([1.6,1,1])
    with c1:product=st.text_input("Product",value=st.session_state.product,placeholder="e.g. Amul Taaza Milk 1L",label_visibility="collapsed")
    with c2:city=st.text_input("City",value=st.session_state.city,placeholder="Jaipur",label_visibility="collapsed")
    with c3:
        months=list(calendar.month_name)[1:];month_name=st.selectbox("Forecast month",months,index=months.index(st.session_state.month_name),label_visibility="collapsed")
    st.session_state.product,st.session_state.city,st.session_state.month_name=product,city,month_name;a,b=st.columns([1,1.2])
    with a:
        if st.button("Refresh market price",use_container_width=True,key="refresh_price"):
            with st.spinner("Checking market price…"):data=fetch_product_price(product)
            if data is None:st.session_state.current_price=None;st.session_state.market_data=None;st.warning("No reliable shopping price found. Try a specific product name.")
            else:st.session_state.current_price=data["price"];st.session_state.market_data=data;st.success(f"Market reference: ₹{data['price']:,.2f}")
            st.rerun()
    with b:
        if st.button("Generate AI Decision →",use_container_width=True,key="generate_decision"):
            month_num=months.index(month_name)+1;year=datetime.datetime.now().year;holiday_count=get_holidays(year,month_num);temperature=get_weather(city);trend=trend_score(product);input_df=pd.DataFrame({"holiday_count":[holiday_count],"avg_temp":[temperature],"viral_score":[trend]});predicted=float(max(0,model.predict(input_df)[0]));stock=float(np.ceil(predicted*1.10));market=st.session_state.get("current_price");suggested=float(market*(1+np.clip((predicted-300)/3000,-0.08,0.08))) if isinstance(market,(int,float)) else None;st.session_state.last_prediction={"pred":predicted,"stock":stock,"suggested":suggested,"temp":temperature,"holiday":holiday_count,"trend":trend,"market":market,"product":product,"city":city,"month":month_name}
            if predictions_collection is not None:
                try:predictions_collection.insert_one({"email":st.session_state.user.get("email"),"product":product,"city":city,"month":month_name,"demand":predicted,"recommended_stock":stock,"market_price":market,"suggested_price":suggested,"temperature":temperature,"holiday_days":holiday_count,"trend_score":trend,"created_at":datetime.datetime.now(datetime.timezone.utc)})
                except Exception:pass
    return product,city,month_name

elif st.session_state.page in {"dashboard","product","forecast","pricing","market","saved"} and st.session_state.user:
    titles={"dashboard":"Good morning. Make the next decision","product":"Analyze a product","forecast":"Demand forecasting","pricing":"Dynamic pricing","market":"Market insights","saved":"Saved analyses"};subtitles={"dashboard":"Enter a product and location, refresh the market reference, then generate a demand, inventory and pricing decision.","product":"Combine market price, weather, holidays and trend signals for a practical product decision.","forecast":"Forecast demand using the current weather, holiday and trend inputs.","pricing":"Use the market reference and predicted demand to calculate a bounded suggested selling price.","market":"Review live price reference together with environmental and demand signals.","saved":"Review the most recent predictions stored in MongoDB."};st.markdown(f"<div class='hero'><div class='eyebrow'>OPTIRETAIL AI · {titles[st.session_state.page].upper()}</div><h1>{titles[st.session_state.page]} <span>with confidence.</span></h1><p>{subtitles[st.session_state.page]}</p></div>",unsafe_allow_html=True)
    if st.session_state.page=="saved":
        if predictions_collection is None:st.warning("MongoDB is unavailable, so saved analyses cannot be loaded.")
        else:
            try:
                rows=list(predictions_collection.find({"email":st.session_state.user.get("email")}).sort("created_at",-1).limit(20))
                if rows:
                    display=pd.DataFrame([{"Date":r.get("created_at"),"Product":r.get("product"),"City":r.get("city"),"Demand":round(r.get("demand",0),1),"Stock":round(r.get("recommended_stock",0),1),"Market Price":r.get("market_price"),"Suggested Price":r.get("suggested_price")} for r in rows]);st.dataframe(display,use_container_width=True,hide_index=True)
                else:st.info("No saved analyses yet. Generate a decision from the dashboard.")
            except Exception as exc:st.error(f"Could not load saved analyses: {exc}")
    else:
        product,city,month_name=render_analysis();lp=st.session_state.get("last_prediction") or {};current=st.session_state.get("current_price");current_text=f"₹{current:,.2f}" if isinstance(current,(int,float)) else "—";suggested_value=lp.get("suggested");k1,k2,k3,k4=st.columns(4)
        k1.markdown(f"<div class='kpi'><small>Forecasted demand</small><strong>{int(lp.get('pred',0)):,} units</strong><span>{'AI demand estimate' if lp else 'Generate a decision'}</span></div>",unsafe_allow_html=True);k2.markdown(f"<div class='kpi'><small>Recommended stock</small><strong>{int(lp.get('stock',0)):,} units</strong><span>10% planning buffer</span></div>",unsafe_allow_html=True);k3.markdown(f"<div class='kpi'><small>Market reference</small><strong>{current_text}</strong><span>SerpAPI shopping signal</span></div>",unsafe_allow_html=True);k4.markdown(f"<div class='kpi'><small>Suggested price</small><strong>{f'₹{suggested_value:,.2f}' if isinstance(suggested_value,(int,float)) else '—'}</strong><span>Demand-aware bound</span></div>",unsafe_allow_html=True)
        if lp:
            c1,c2=st.columns([1.5,1])
            with c1:
                st.markdown("<div class='card'><div class='eyebrow'>DEMAND FORECAST</div><h3>12-month demand outlook</h3>",unsafe_allow_html=True);months=list(calendar.month_name)[1:];x=np.arange(12);base=max(1,lp.get('pred',1));seasonal=1+0.10*np.sin((x+months.index(lp['month']))*2*np.pi/12);forecast=base*seasonal;hist=np.maximum(0,base*(0.82+0.08*np.sin((x+1)*2*np.pi/12)));chart=pd.DataFrame({"Month":months,"Historical demand":np.round(hist),"Forecast demand":np.round(forecast)});st.line_chart(chart.set_index("Month"),height=320);st.markdown("</div>",unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='card'><div class='eyebrow'>AI DECISION</div><h3>Model signals</h3><div class='signal'><b>Weather</b><span>{lp.get('temp',0):.1f} °C</span></div><div class='signal'><b>Holiday days</b><span>{lp.get('holiday',0)}</span></div><div class='signal'><b>Trend score</b><span>{lp.get('trend',0)}/100</span></div><div class='signal'><b>Market reference</b><span>{current_text}</span></div></div>",unsafe_allow_html=True);temp=lp.get('temp',25);action="Demand signal elevated by warm weather — keep additional safety stock for heat-sensitive categories." if temp>=32 else ("Cooler weather detected — demand may shift toward seasonal categories; keep inventory conservative." if temp<=18 else "Weather is within a moderate range; holiday and trend signals drive the current decision.");st.markdown(f"<div class='decision'><h3>Recommended action</h3><p>{action}</p></div>",unsafe_allow_html=True)
        else:st.markdown("<div class='card'><div class='eyebrow'>READY</div><h3>Generate your first AI decision</h3><p class='small-note'>Refresh the market reference, then click Generate AI Decision to populate demand, inventory, pricing and the forecast chart.</p></div>",unsafe_allow_html=True)
