import calendar
import datetime as dt
import hashlib
import secrets
import time
from urllib.parse import urlsplit

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
except ImportError:
    MongoClient = None
try:
    from serpapi import GoogleSearch
except ImportError:
    GoogleSearch = None

st.set_page_config(page_title="OptiRetail AI", page_icon="📊", layout="wide", initial_sidebar_state="expanded")


def secret_text(name):
    try:
        value = st.secrets.get(name, "")
        return str(value).strip().strip('"').strip("'") if value else ""
    except Exception:
        return ""


SERPAPI_KEY = secret_text("SERPAPI_KEY")
MONGO_URI = secret_text("MONGO_URI") or secret_text("MONGODB_URI")
BREVO_API_KEY = secret_text("BREVO_API_KEY")
BREVO_SENDER_EMAIL = secret_text("BREVO_SENDER_EMAIL")
BREVO_SENDER_NAME = secret_text("BREVO_SENDER_NAME") or "OptiRetail AI"

for key, value in {
    "user": None,
    "page": "welcome",
    "product": "Wheat Flour",
    "city": "Jaipur",
    "month_name": dt.datetime.now().strftime("%B"),
    "last_prediction": None,
    "current_price": None,
    "market_data": None,
    "otp_email": None,
    "otp_value": None,
    "otp_created_at": None,
    "pending_signup": None,
}.items():
    st.session_state.setdefault(key, value)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');
:root{--ink:#10231f;--muted:#64748b;--green:#07865f;--green2:#056b4c;--line:#dcebe5}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--ink)}
.stApp{background:linear-gradient(180deg,#f7fbf9 0%,#fff 45%,#f3faf7 100%)}
.block-container{max-width:1500px;padding:1rem 2rem 4rem}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#052d23,#06392e);border-right:1px solid rgba(255,255,255,.08)}
[data-testid="stSidebar"] *{color:#ecfdf5}
.sidebar-logo{font-family:'Plus Jakarta Sans';font-size:1.5rem;font-weight:800}.sidebar-sub{font-size:.68rem;color:#a7f3d0}
.hero{padding:34px 40px;border-radius:28px;background:linear-gradient(135deg,#fff,#effbf6);border:1px solid var(--line);box-shadow:0 20px 55px rgba(16,185,129,.08);margin-bottom:20px}
.hero h1{font-family:'Plus Jakarta Sans';font-size:2.7rem;line-height:1.08;letter-spacing:-2px;margin:0;color:#0f172a}.hero h1 span{color:var(--green2)}.hero p{color:#64748b;font-size:1rem;max-width:950px;margin:.7rem 0 0}
.eyebrow{color:var(--green2);text-transform:uppercase;font-size:.65rem;font-weight:800;letter-spacing:2px;margin-bottom:8px}.card{background:#fff;border:1px solid #e2e8f0;border-radius:20px;padding:20px;box-shadow:0 10px 30px rgba(15,23,42,.05);margin-bottom:16px}.card h3{font-family:'Plus Jakarta Sans';margin:.15rem 0 .5rem;color:#0f172a}.small{color:#64748b;font-size:.84rem;line-height:1.5}.kpi{background:#fff;border:1px solid #e2e8f0;border-radius:18px;padding:18px;min-height:115px;box-shadow:0 8px 22px rgba(15,23,42,.04)}.kpi small{color:#64748b;text-transform:uppercase;letter-spacing:1px;font-weight:800;font-size:.62rem}.kpi strong{display:block;font-family:'Plus Jakarta Sans';color:#0f172a;font-size:1.65rem;margin-top:7px}.kpi span{font-size:.74rem;color:#64748b}.signal{display:flex;justify-content:space-between;gap:12px;padding:11px 0;border-bottom:1px solid #eef2f4}.signal:last-child{border-bottom:0}.signal b{color:#0f172a}.signal span{color:#64748b;text-align:right}.decision{border-radius:18px;padding:18px;background:linear-gradient(135deg,#ecfdf5,#f0fdfa);border:1px solid #bbf7d0}.decision h3{margin:0 0 5px;color:#065f46}.decision p{margin:0;color:#475569;line-height:1.5}.auth{max-width:720px;margin:25px auto}.auth-card{background:#fff;border:1px solid var(--line);border-radius:28px;padding:34px;box-shadow:0 20px 55px rgba(15,23,42,.07)}.auth-title{font-family:'Plus Jakarta Sans';font-size:2.1rem;font-weight:800;color:var(--green2);margin:0}.auth-sub{color:#64748b;margin:.4rem 0 1.4rem}
div[data-testid="stTextInput"] input,div[data-testid="stNumberInput"] input,div[data-testid="stTextArea"] textarea,div[data-testid="stSelectbox"] div[data-baseweb="select"]>div{background:#fff!important;color:#0f172a!important;border:1px solid #cfe0d9!important;border-radius:12px!important;box-shadow:none!important;font-size:1rem!important}
div[data-testid="stTextInput"] input::placeholder,div[data-testid="stNumberInput"] input::placeholder,div[data-testid="stTextArea"] textarea::placeholder{color:#94a3b8!important;opacity:1!important}label,div[data-testid="stWidgetLabel"] p{color:#334155!important;font-weight:600!important}div[data-baseweb="select"] span{color:#0f172a!important}
.stButton>button,.stFormSubmitButton>button{border:0!important;border-radius:12px!important;min-height:44px!important;font-weight:800!important;background:linear-gradient(135deg,#079568,#047a55)!important;color:#fff!important;box-shadow:0 7px 18px rgba(5,150,105,.15)}
</style>
""",unsafe_allow_html=True)


def get_database():
    if not MONGO_URI:
        return None,"MONGO_URI is missing from Streamlit Secrets."
    if MongoClient is None:
        return None,"PyMongo is not installed."
    if not (MONGO_URI.startswith("mongodb://") or MONGO_URI.startswith("mongodb+srv://")):
        return None,"MONGO_URI must start with mongodb:// or mongodb+srv://."
    if "<password>" in MONGO_URI or "<db_password>" in MONGO_URI:
        return None,"MONGO_URI still contains a password placeholder."
    try:
        parsed=urlsplit(MONGO_URI)
        if not parsed.hostname:
            return None,"MongoDB URI has no hostname."
        uri=MONGO_URI
        if "authSource=" not in uri:
            uri += ("&" if "?" in uri else "?") + "authSource=admin"
        client=MongoClient(uri,serverSelectionTimeoutMS=10000,connectTimeoutMS=10000,socketTimeoutMS=10000,retryWrites=True,appname="OptiRetailAI")
        client.admin.command("ping")
        database=client["optiretail_ai"]
        database.command("ping")
        return database,f"Connected to {parsed.hostname}"
    except Exception as exc:
        text=str(exc).lower();name=type(exc).__name__.lower()
        if "authentication failed" in text or "bad auth" in text or "authenticationfailure" in name:
            return None,"MongoDB authentication failed for the Atlas database user. Check the Atlas username/password and authSource."
        if "serverselectiontimeouterror" in name:
            return None,"Atlas could not be reached. Check MongoDB Atlas Network Access/IP allowlist."
        if "dns" in text or "querysrv" in name:
            return None,"MongoDB DNS/SRV lookup failed. Check the Atlas cluster hostname in MONGO_URI."
        return None,f"MongoDB connection failed: {type(exc).__name__}. Check the URI and Atlas configuration."


db,mongo_status=get_database()
users=db["users"] if db is not None else None
predictions=db["predictions"] if db is not None else None


def password_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def generate_otp():
    return f"{secrets.randbelow(1_000_000):06d}"


def send_otp(email,otp):
    if not BREVO_API_KEY or not BREVO_SENDER_EMAIL:
        return False,"Brevo API key or verified sender email is missing in Streamlit Secrets."
    try:
        response=requests.post("https://api.brevo.com/v3/smtp/email",headers={"accept":"application/json","api-key":BREVO_API_KEY,"content-type":"application/json"},json={"sender":{"name":BREVO_SENDER_NAME,"email":BREVO_SENDER_EMAIL},"to":[{"email":email}],"subject":"Your OptiRetail AI verification code","htmlContent":f"<div style='font-family:Arial;padding:24px'><h2>Verify your OptiRetail AI account</h2><p>Your verification code is:</p><div style='font-size:32px;font-weight:700;letter-spacing:8px;color:#047857'>{otp}</div><p>This code expires in 5 minutes.</p></div>"},timeout=10)
        if 200<=response.status_code<300:return True,"OTP sent to your email."
        return False,f"Brevo error {response.status_code}: {response.text[:200]}"
    except Exception as exc:return False,f"Could not send OTP: {exc}"


def save_user(email,password,gst,turnover):
    if users is None:return False,f"MongoDB is unavailable. {mongo_status}"
    normalized=email.strip().lower()
    try:
        if users.find_one({"email":normalized}):return False,"An account with this email already exists."
        users.insert_one({"email":normalized,"password_hash":password_hash(password),"gst":gst.strip(),"turnover":turnover,"created_at":dt.datetime.now(dt.timezone.utc)})
        return True,"Account created successfully."
    except Exception as exc:return False,f"Could not save account: {type(exc).__name__}: {exc}"


def authenticate(email,password):
    if users is None:return None
    try:
        user=users.find_one({"email":email.strip().lower()})
        if not user:return None
        if user.get("password_hash")==password_hash(password) or user.get("password")==password:return user
        return None
    except Exception:return None


@st.cache_data(ttl=900,show_spinner=False)
def fetch_price(product):
    if not product.strip() or not SERPAPI_KEY or GoogleSearch is None:return None
    try:
        result=GoogleSearch({"engine":"google_shopping_light","q":f"{product} price","gl":"in","hl":"en","num":8,"api_key":SERPAPI_KEY}).get_dict();values=[]
        for item in result.get("shopping_results",[]):
            try:value=float(item.get("extracted_price"))
            except (TypeError,ValueError):continue
            if value>0:values.append((value,item.get("title",""),item.get("source",""),item.get("link","")))
        if not values:return None
        values.sort(key=lambda x:x[0]);return {"price":float(np.median([x[0] for x in values[:5]])),"results":values[:5]}
    except Exception:return None


@st.cache_data(ttl=900,show_spinner=False)
def get_weather(city):
    try:
        geo=requests.get("https://geocoding-api.open-meteo.com/v1/search",params={"name":city,"count":1},timeout=5);geo.raise_for_status();hits=geo.json().get("results",[])
        if not hits:return 25.0
        lat,lon=hits[0]["latitude"],hits[0]["longitude"];weather=requests.get("https://api.open-meteo.com/v1/forecast",params={"latitude":lat,"longitude":lon,"current_weather":"true"},timeout=5);weather.raise_for_status();return float(weather.json()["current_weather"]["temperature"])
    except Exception:return 25.0


@st.cache_data
def holiday_count(year,month):
    total=0;india=holidays.India(years=year) if holidays else set()
    for day in range(1,calendar.monthrange(year,month)[1]+1):
        date=dt.date(year,month,day)
        if date.weekday()>=5 or date in india:total+=1
    return total


def trend_score(product):
    rng=np.random.default_rng(sum(ord(c) for c in product));return int(rng.integers(35,90))


@st.cache_resource(show_spinner=False)
def train_model():
    rng=np.random.default_rng(42);data=pd.DataFrame({"holiday_count":rng.integers(0,12,500),"avg_temp":rng.uniform(10,40,500),"viral_score":rng.integers(0,100,500)})
    seasonal=np.maximum(0,22-np.abs(data["avg_temp"]-28));data["sales"]=180+data["holiday_count"]*42+data["viral_score"]*4.8+seasonal*11+rng.normal(0,25,500)
    model=RandomForestRegressor(n_estimators=250,max_depth=12,min_samples_leaf=2,random_state=42,n_jobs=-1);model.fit(data[["holiday_count","avg_temp","viral_score"]],data["sales"]);return model

model=train_model()


def clear_otp():
    for key in ("otp_email","otp_value","otp_created_at","pending_signup"):st.session_state[key]=None


def render_analysis():
    st.markdown("<div class='section-title'>Product analysis</div>",unsafe_allow_html=True)
    c1,c2,c3=st.columns([1.5,1,1])
    with c1:product=st.text_input("Product",value=st.session_state.product,placeholder="e.g. Amul Taaza Milk 1L")
    with c2:city=st.text_input("City",value=st.session_state.city,placeholder="Jaipur")
    with c3:
        months=list(calendar.month_name)[1:];month_name=st.selectbox("Forecast month",months,index=months.index(st.session_state.month_name))
    st.session_state.product,st.session_state.city,st.session_state.month_name=product,city,month_name
    a,b=st.columns(2)
    with a:
        if st.button("Refresh market price",use_container_width=True,key="refresh_price"):
            with st.spinner("Checking market price..."):data=fetch_price(product)
            st.session_state.market_data=data;st.session_state.current_price=data["price"] if data else None
            if data:st.success(f"Market reference: ₹{data['price']:,.2f}")
            else:st.warning("No reliable shopping price found. Check SerpAPI or use a specific product name.")
    with b:
        if st.button("Generate AI Decision →",use_container_width=True,key="generate_decision"):
            month_num=months.index(month_name)+1;year=dt.datetime.now().year;temp=get_weather(city);holiday=holiday_count(year,month_num);trend=trend_score(product)
            inputs=pd.DataFrame({"holiday_count":[holiday],"avg_temp":[temp],"viral_score":[trend]});predicted=float(max(0,model.predict(inputs)[0]));stock=float(np.ceil(predicted*1.10));market=st.session_state.current_price
            suggested=float(market*(1+np.clip((predicted-300)/3000,-0.08,0.08))) if isinstance(market,(int,float)) else None
            st.session_state.last_prediction={"pred":predicted,"stock":stock,"suggested":suggested,"temp":temp,"holiday":holiday,"trend":trend,"market":market,"product":product,"city":city,"month":month_name}
            if predictions is not None and st.session_state.user:
                try:predictions.insert_one({"email":st.session_state.user.get("email"),"product":product,"city":city,"month":month_name,"demand":predicted,"recommended_stock":stock,"market_price":market,"suggested_price":suggested,"temperature":temp,"holiday_days":holiday,"trend_score":trend,"created_at":dt.datetime.now(dt.timezone.utc)})
                except Exception as exc:st.warning(f"Decision generated, but MongoDB could not save it: {exc}")


def render_dashboard():
    lp=st.session_state.get("last_prediction") or {};current=st.session_state.get("current_price");current_text=f"₹{current:,.2f}" if isinstance(current,(int,float)) else "—";suggested=lp.get("suggested")
    k1,k2,k3,k4=st.columns(4)
    k1.markdown(f"<div class='kpi'><small>Forecasted demand</small><strong>{int(lp.get('pred',0)):,} units</strong><span>AI demand estimate</span></div>",unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi'><small>Recommended stock</small><strong>{int(lp.get('stock',0)):,} units</strong><span>10% planning buffer</span></div>",unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi'><small>Market reference</small><strong>{current_text}</strong><span>SerpAPI shopping signal</span></div>",unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi'><small>Suggested price</small><strong>{f'₹{suggested:,.2f}' if isinstance(suggested,(int,float)) else '—'}</strong><span>Demand-aware bound</span></div>",unsafe_allow_html=True)
    if not lp:
        st.markdown("<div class='card'><div class='eyebrow'>READY</div><h3>Generate your first AI decision</h3><p class='small'>Choose a product and city above, refresh the market reference, then generate a decision.</p></div>",unsafe_allow_html=True);return
    left,right=st.columns([1.55,1])
    with left:
        months=list(calendar.month_name)[1:];x=np.arange(12);base=max(1,lp.get("pred",1));forecast=base*(1+.10*np.sin((x+months.index(lp["month"]))*2*np.pi/12));history=np.maximum(0,base*(.82+.08*np.sin((x+1)*2*np.pi/12)));chart=pd.DataFrame({"Historical demand":np.round(history),"Forecast demand":np.round(forecast)},index=months)
        st.markdown("<div class='card'><div class='eyebrow'>DEMAND FORECAST</div><h3>12-month demand outlook</h3>",unsafe_allow_html=True);st.line_chart(chart,height=320);st.markdown("</div>",unsafe_allow_html=True)
    with right:
        st.markdown("<div class='card'><div class='eyebrow'>AI DECISION</div><h3>Model signals</h3>",unsafe_allow_html=True)
        for name,value in [("Weather",f"{lp.get('temp',25):.1f} °C"),("Holiday days",str(lp.get('holiday',0))),("Trend score",f"{lp.get('trend',0)}/100"),("Market reference",current_text)]:st.markdown(f"<div class='signal'><b>{name}</b><span>{value}</span></div>",unsafe_allow_html=True)
        st.markdown("</div>",unsafe_allow_html=True)
        temp=lp.get("temp",25);action="Higher temperature detected — consider extra safety stock for heat-sensitive products." if temp>=32 else ("Cooler weather detected — keep inventory conservative for weather-sensitive categories." if temp<=18 else "Weather is moderate; holiday and trend signals are driving the current recommendation.")
        st.markdown(f"<div class='decision'><h3>Recommended action</h3><p>{action}</p></div>",unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<div class='sidebar-logo'>📊 OptiRetail AI</div><div class='sidebar-sub'>Analyze · Predict · Price</div>",unsafe_allow_html=True);st.divider()
    if st.session_state.user:
        st.caption(st.session_state.user.get("email","Business User"))
        for label,target in [("Dashboard","dashboard"),("Product Analysis","product"),("Demand Forecasting","forecast"),("Dynamic Pricing","pricing"),("Market Insights","market"),("Saved Analyses","saved"),("Settings","settings")]:
            if st.button(label,use_container_width=True,key=f"nav_{target}"):st.session_state.page=target;st.rerun()
        st.divider()
        if st.button("Sign out",use_container_width=True,key="signout"):st.session_state.user=None;st.session_state.page="welcome";st.session_state.last_prediction=None;st.rerun()
    else:
        st.markdown("### Smarter retail decisions");st.caption("Forecast demand. Optimize inventory. Price with confidence.")

if st.session_state.page=="welcome":
    st.markdown("<div class='hero'><div class='eyebrow'>OPTIRETAIL AI</div><h1>Turn market data into <span>smarter decisions.</span></h1><p>Forecast demand, understand market signals, optimize inventory and make practical pricing decisions from one clean workspace.</p></div>",unsafe_allow_html=True)
    a,b,c=st.columns(3)
    a.markdown("<div class='card'><div class='eyebrow'>01 · FORECAST</div><h3>Know what will move.</h3><p class='small'>AI demand forecasting combines weather, holidays and product trend signals.</p></div>",unsafe_allow_html=True)
    b.markdown("<div class='card'><div class='eyebrow'>02 · INVENTORY</div><h3>Stock with confidence.</h3><p class='small'>Turn predicted demand into a practical quantity with a planning buffer.</p></div>",unsafe_allow_html=True)
    c.markdown("<div class='card'><div class='eyebrow'>03 · PRICE</div><h3>Price with context.</h3><p class='small'>Use a live market reference and demand-aware pricing recommendation.</p></div>",unsafe_allow_html=True)
    x,y=st.columns(2)
    with x:
        if st.button("Get Started →",use_container_width=True,key="welcome_signup"):st.session_state.page="signup";st.rerun()
    with y:
        if st.button("Sign in",use_container_width=True,key="welcome_login"):st.session_state.page="login";st.rerun()

elif st.session_state.page=="login":
    st.markdown("<div class='auth'><div class='auth-card'><div class='eyebrow'>WELCOME BACK</div><h1 class='auth-title'>Sign in to OptiRetail AI.</h1><p class='auth-sub'>Access your retail intelligence workspace.</p>",unsafe_allow_html=True)
    with st.form("login_form"):
        email=st.text_input("Email address",placeholder="you@company.com");password=st.text_input("Password",type="password",placeholder="Your password");submitted=st.form_submit_button("Sign in",use_container_width=True)
    if submitted:
        user=authenticate(email,password)
        if user:st.session_state.user=user;st.session_state.page="dashboard";st.rerun()
        elif users is None:st.error(f"MongoDB is unavailable. {mongo_status}")
        else:st.error("Invalid credentials.")
    a,b=st.columns(2)
    with a:
        if st.button("Create an account →",use_container_width=True,key="login_signup"):st.session_state.page="signup";st.rerun()
    with b:
        if st.button("← Back to home",use_container_width=True,key="login_back"):st.session_state.page="welcome";st.rerun()
    st.markdown("</div></div>",unsafe_allow_html=True)

elif st.session_state.page=="signup":
    st.markdown("<div class='auth'><div class='auth-card'><div class='eyebrow'>BUSINESS ONBOARDING</div><h1 class='auth-title'>Create your OptiRetail AI workspace.</h1><p class='auth-sub'>Verify your email before the account is saved to MongoDB.</p>",unsafe_allow_html=True)
    if not st.session_state.otp_value:
        with st.form("signup_form"):
            email=st.text_input("Email address",placeholder="you@company.com");password=st.text_input("Password",type="password",placeholder="Create a password");gst=st.text_input("GST Number",placeholder="Enter GST number");turnover=st.selectbox("Annual Turnover",["1–5 Lakh","5–10 Lakh","10–15 Lakh","15–50 Lakh","50 Lakh+"]);send=st.form_submit_button("Send verification OTP",use_container_width=True)
        if send:
            email=email.strip().lower()
            if not email or "@" not in email or not password or not gst:st.error("Please enter a valid email, password and GST number.")
            elif users is None:st.error(f"MongoDB is unavailable. {mongo_status}")
            else:
                try:exists=users.find_one({"email":email})
                except Exception as exc:exists=None;st.error(f"MongoDB error: {type(exc).__name__}: {exc}")
                if not exists:
                    otp=generate_otp();ok,msg=send_otp(email,otp)
                    if ok:st.session_state.otp_email=email;st.session_state.otp_value=otp;st.session_state.otp_created_at=time.time();st.session_state.pending_signup={"email":email,"password":password,"gst":gst,"turnover":turnover};st.rerun()
                    else:st.error(msg)
                else:st.error("An account with this email already exists.")
    else:
        st.success(f"Verification code sent to {st.session_state.otp_email}")
        with st.form("otp_form"):
            code=st.text_input("6-digit OTP",max_chars=6,placeholder="000000");verify=st.form_submit_button("Verify email & create account",use_container_width=True)
        if verify:
            if not st.session_state.otp_created_at or time.time()-st.session_state.otp_created_at>300:st.error("OTP expired. Please resend a new code.")
            elif code.strip()!=st.session_state.otp_value:st.error("Incorrect OTP.")
            else:
                ok,msg=save_user(**st.session_state.pending_signup)
                if ok:clear_otp();st.session_state.page="login";st.success(msg);st.rerun()
                else:st.error(msg)
        if st.button("Resend OTP",key="resend_otp"):
            pending=st.session_state.pending_signup;new_code=generate_otp();ok,msg=send_otp(pending["email"],new_code)
            if ok:st.session_state.otp_value=new_code;st.session_state.otp_created_at=time.time();st.success("New OTP sent.");st.rerun()
            else:st.error(msg)
    a,b=st.columns(2)
    with a:
        if st.button("Already have an account? Sign in",key="signup_login"):clear_otp();st.session_state.page="login";st.rerun()
    with b:
        if st.button("← Back to home",key="signup_back"):clear_otp();st.session_state.page="welcome";st.rerun()
    st.markdown("</div></div>",unsafe_allow_html=True)

elif st.session_state.page=="settings" and st.session_state.user:
    st.markdown("<div class='hero'><div class='eyebrow'>WORKSPACE</div><h1>Business settings</h1><p>Connected services and account information.</p></div>",unsafe_allow_html=True)
    u=st.session_state.user;a,b,c=st.columns(3);a.metric("Account","Active");b.metric("Email",u.get("email","N/A"));c.metric("Turnover",u.get("turnover","N/A"))
    st.markdown("<div class='card'><div class='eyebrow'>DATA SERVICES</div><h3>Connection status</h3>",unsafe_allow_html=True);st.write("MongoDB:","Connected" if db is not None else "Unavailable");st.write("SerpAPI:","Configured" if SERPAPI_KEY else "Not configured");st.write("Brevo email:","Configured" if BREVO_API_KEY and BREVO_SENDER_EMAIL else "Not configured");
    if db is None:st.warning(mongo_status)
    st.markdown("</div>",unsafe_allow_html=True)

elif st.session_state.page in {"dashboard","product","forecast","pricing"} and st.session_state.user:
    titles={"dashboard":"Smarter retail decisions","product":"Analyze a product","forecast":"Demand forecasting","pricing":"Dynamic pricing"}
    st.markdown(f"<div class='hero'><div class='eyebrow'>OPTIRETAIL AI</div><h1>{titles[st.session_state.page]} <span>with confidence.</span></h1><p>Combine market price, weather, holidays and product trends to make a practical retail decision.</p></div>",unsafe_allow_html=True)
    render_analysis();render_dashboard()

elif st.session_state.page=="market" and st.session_state.user:
    st.markdown("<div class='hero'><div class='eyebrow'>MARKET INSIGHTS</div><h1>Live market reference.</h1><p>Review the latest SerpAPI shopping signals for the selected product.</p></div>",unsafe_allow_html=True)
    data=st.session_state.market_data
    if data:
        st.metric("Market reference",f"₹{data['price']:,.2f}");st.dataframe(pd.DataFrame(data["results"],columns=["Price","Product","Source","Link"]),use_container_width=True,hide_index=True)
    else:st.info("Run Refresh market price from Product Analysis first.")

elif st.session_state.page=="saved" and st.session_state.user:
    st.markdown("<div class='hero'><div class='eyebrow'>SAVED ANALYSES</div><h1>Your saved decisions.</h1><p>Recent AI predictions stored in MongoDB.</p></div>",unsafe_allow_html=True)
    if predictions is None:st.warning(f"MongoDB is unavailable. {mongo_status}")
    else:
        try:
            rows=list(predictions.find({"email":st.session_state.user.get("email")},{"_id":0}).sort("created_at",-1).limit(20))
            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True) if rows else st.info("No saved analyses yet.")
        except Exception as exc:st.error(f"Could not load saved analyses: {type(exc).__name__}: {exc}")
else:
    if st.session_state.page!="welcome":st.session_state.page="welcome";st.rerun()
