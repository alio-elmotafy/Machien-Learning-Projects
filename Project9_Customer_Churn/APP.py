import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #05080f;
    color: #c9d8f0;
}
.main { background: #05080f; }
section.main > div { padding-top: 1.2rem; }

/* animated dot-grid background */
.main::before {
    content: "";
    position: fixed; inset: 0;
    background-image: radial-gradient(rgba(0,210,255,0.07) 1px, transparent 1px);
    background-size: 32px 32px;
    pointer-events: none; z-index: 0;
}

/* ── hero ── */
.hero { margin-bottom: 1.6rem; }
.hero-title {
    font-weight: 800; font-size: clamp(1.9rem, 4vw, 3.2rem);
    letter-spacing: -0.03em; line-height: 1.05;
    background: linear-gradient(130deg, #00d2ff 0%, #a855f7 55%, #f472b6 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin: 0;
}
.hero-sub {
    font-family: 'Space Mono', monospace; font-size: 0.72rem;
    color: #3d5a80; letter-spacing: 0.22em; text-transform: uppercase; margin-top: 0.4rem;
}

/* ── section headers ── */
.sec-label {
    font-family: 'Space Mono', monospace; font-size: 0.68rem;
    letter-spacing: 0.22em; text-transform: uppercase;
    color: #00d2ff; margin: 0 0 1rem 0;
    display: flex; align-items: center; gap: 8px;
}
.sec-label::after {
    content: ""; flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(0,210,255,0.3), transparent);
}

/* ── cards ── */
.card {
    background: linear-gradient(150deg, #0b1526 0%, #09111f 100%);
    border: 1px solid rgba(0,210,255,0.10);
    border-radius: 18px; padding: 1.5rem 1.6rem; margin-bottom: 1rem;
    position: relative; overflow: hidden;
}
.card::before {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #00d2ff, #a855f7, #f472b6);
}

/* ── input labels ── */
label[data-testid="stWidgetLabel"] p,
div[data-testid="stSlider"] label p,
div[data-testid="stNumberInput"] label p {
    color: #6b9ab8 !important; font-size: 0.8rem !important;
    font-weight: 600 !important; letter-spacing: 0.05em !important;
}

/* ── inputs ── */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stNumberInput"] input {
    background: #0b1526 !important;
    border: 1px solid rgba(0,210,255,0.15) !important;
    color: #c9d8f0 !important; border-radius: 8px !important;
}

/* ── predict button ── */
div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #00d2ff 0%, #a855f7 60%, #f472b6 100%);
    color: #fff; font-family: 'Syne', sans-serif;
    font-weight: 800; font-size: 1rem; letter-spacing: 0.1em;
    border: none; border-radius: 14px; padding: 1rem 2rem;
    cursor: pointer; transition: all .3s ease;
    box-shadow: 0 0 28px rgba(0,210,255,0.2);
    margin-top: 0.5rem;
}
div.stButton > button:hover {
    box-shadow: 0 0 48px rgba(168,85,247,0.45);
    transform: translateY(-3px) scale(1.01);
}

/* ── result containers ── */
.res-high {
    background: linear-gradient(145deg, rgba(239,68,68,0.12), rgba(185,28,28,0.06));
    border: 1px solid rgba(239,68,68,0.45); border-radius: 18px;
    padding: 2.2rem 1.8rem; text-align: center;
    animation: glowRed 2.5s ease-in-out infinite;
}
.res-low {
    background: linear-gradient(145deg, rgba(16,185,129,0.1), rgba(5,150,105,0.05));
    border: 1px solid rgba(16,185,129,0.4); border-radius: 18px;
    padding: 2.2rem 1.8rem; text-align: center;
    animation: glowGreen 2.5s ease-in-out infinite;
}
.res-med {
    background: linear-gradient(145deg, rgba(245,158,11,0.12), rgba(180,83,9,0.06));
    border: 1px solid rgba(245,158,11,0.4); border-radius: 18px;
    padding: 2.2rem 1.8rem; text-align: center;
    animation: glowAmber 2.5s ease-in-out infinite;
}
@keyframes glowRed   { 0%,100%{box-shadow:0 0 18px rgba(239,68,68,.15)}  50%{box-shadow:0 0 40px rgba(239,68,68,.35)} }
@keyframes glowGreen { 0%,100%{box-shadow:0 0 18px rgba(16,185,129,.1)}  50%{box-shadow:0 0 40px rgba(16,185,129,.28)} }
@keyframes glowAmber { 0%,100%{box-shadow:0 0 18px rgba(245,158,11,.12)} 50%{box-shadow:0 0 40px rgba(245,158,11,.3)} }

.res-pct { font-family:'Syne',sans-serif; font-weight:800; font-size:clamp(3rem,7vw,5rem); line-height:1; }
.res-pct-high  { color:#ef4444; }
.res-pct-low   { color:#10b981; }
.res-pct-med   { color:#f59e0b; }
.res-lbl { font-family:'Space Mono',monospace; font-size:0.7rem; letter-spacing:.2em; text-transform:uppercase; opacity:.55; margin-bottom:.5rem; }
.res-desc { font-size:0.88rem; opacity:.65; margin-top:.7rem; }

/* ── gauge ── */
.gauge-wrap { margin: 1.4rem auto 0; max-width:360px; }
.gauge-track { height:10px; background:rgba(255,255,255,.07); border-radius:99px; overflow:hidden; }
.gauge-fill  { height:100%; border-radius:99px; transition:width 1.2s cubic-bezier(.4,0,.2,1); }

/* ── metric chips ── */
.chips { display:flex; gap:.8rem; margin-top:1.3rem; flex-wrap:wrap; justify-content:center; }
.chip  {
    flex:1; min-width:100px; max-width:130px;
    background:rgba(0,210,255,0.06); border:1px solid rgba(0,210,255,0.14);
    border-radius:12px; padding:.7rem .9rem; text-align:center;
}
.chip-val { font-family:'Space Mono',monospace; font-size:1.1rem; font-weight:700; color:#00d2ff; }
.chip-lbl { font-size:0.63rem; opacity:.45; letter-spacing:.1em; text-transform:uppercase; margin-top:3px; }

/* ── feature-importance mini bar ── */
.fi-bar-wrap { margin-top:.5rem; }
.fi-row { display:flex; align-items:center; gap:.7rem; margin-bottom:.45rem; }
.fi-name { font-family:'Space Mono',monospace; font-size:.65rem; color:#6b9ab8; width:160px; text-align:right; flex-shrink:0; }
.fi-track { flex:1; height:6px; background:rgba(255,255,255,.06); border-radius:99px; overflow:hidden; }
.fi-fill  { height:100%; border-radius:99px; background:linear-gradient(90deg,#00d2ff,#a855f7); }

/* ── divider & footer ── */
hr { border:none; border-top:1px solid rgba(0,210,255,0.08); margin:1.8rem 0; }
.footer {
    font-family:'Space Mono',monospace; font-size:.62rem;
    opacity:.22; text-align:center; letter-spacing:.14em; padding:.8rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── ASSET LOADING ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Loading model assets…")
def load_assets():
    candidates = [".", os.path.dirname(os.path.abspath(__file__))]
    def find(fn):
        for d in candidates:
            p = os.path.join(d, fn)
            if os.path.exists(p):
                return p
        return None

    mp = find("lgbm_model.pkl")
    sp = find("scaler.pkl")
    cp = find("model_columns.pkl")

    missing = [n for n, p in [("lgbm_model.pkl", mp), ("scaler.pkl", sp), ("model_columns.pkl", cp)] if p is None]
    if missing:
        return None, None, None, missing

    return joblib.load(mp), joblib.load(sp), joblib.load(cp), []


model, scaler, model_columns, missing_files = load_assets()


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <p class="hero-title">📡 Churn Intelligence</p>
  <p class="hero-sub">Customer Risk Prediction · LightGBM · Telco Dataset</p>
</div>
""", unsafe_allow_html=True)

if missing_files:
    st.error(f"⚠️ Missing model files: **{', '.join(missing_files)}** — place them in the repo root and redeploy.")
    st.stop()


# ── LAYOUT: 3 columns ─────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([1.05, 1.05, 1.1], gap="large")

# ═══════════════════════════════════════
# COLUMN 1 — Financial & Usage
# ═══════════════════════════════════════
with c1:
    # ── Top Features (from importance chart) ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="sec-label">💳 Financial & Usage</p>', unsafe_allow_html=True)

    cltv              = st.number_input("CLTV",                          min_value=0,   value=4500, step=100)
    monthly_charge    = st.number_input("Monthly Charge ($)",            min_value=0.0, value=70.0, step=1.0,  format="%.2f")
    total_charges     = st.number_input("Total Charges ($)",             min_value=0.0, value=2000.0,step=50.0,format="%.2f")
    total_refunds     = st.number_input("Total Refunds ($)",             min_value=0.0, value=0.0,  step=1.0,  format="%.2f")
    total_extra_data  = st.number_input("Total Extra Data Charges ($)",  min_value=0.0, value=0.0,  step=1.0,  format="%.2f")
    avg_long_dist     = st.number_input("Avg Monthly Long Distance ($)", min_value=0.0, value=25.0, step=1.0,  format="%.2f")
    total_long_dist   = st.number_input("Total Long Distance Charges ($)",min_value=0.0,value=300.0,step=10.0, format="%.2f")
    avg_gb_download   = st.number_input("Avg Monthly GB Download",       min_value=0.0, value=15.0, step=1.0,  format="%.1f")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Location ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="sec-label">📍 Location</p>', unsafe_allow_html=True)
    city        = st.text_input("City", value="Los Angeles")
    zip_code    = st.number_input("Zip Code",    min_value=0, value=90001, step=1)
    latitude    = st.number_input("Latitude",    min_value=-90.0,  max_value=90.0,  value=34.05, format="%.4f")
    longitude   = st.number_input("Longitude",   min_value=-180.0, max_value=180.0, value=-118.24, format="%.4f")
    population  = st.number_input("Area Population", min_value=0, value=25000, step=500)
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════
# COLUMN 2 — Profile & Services
# ═══════════════════════════════════════
with c2:
    # ── Customer Profile ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="sec-label">👤 Customer Profile</p>', unsafe_allow_html=True)

    age              = st.slider("Age", 18, 100, 35)
    churn_score      = st.slider("Churn Score (0–100)", 0, 100, 50)
    satisfaction     = st.slider("Satisfaction Score (1–5)", 1, 5, 3)
    tenure_months    = st.number_input("Tenure in Months", min_value=0, value=24, step=1)
    num_referrals    = st.number_input("Number of Referrals", min_value=0, value=0, step=1)
    num_dependents   = st.number_input("Number of Dependents", min_value=0, value=0, step=1)

    p1, p2 = st.columns(2)
    with p1:
        gender         = st.selectbox("Gender",          ["Male", "Female"])
        married        = st.selectbox("Married",         ["Yes", "No"])
        dependents     = st.selectbox("Dependents",      ["Yes", "No"])
        senior         = st.selectbox("Senior Citizen",  ["Yes", "No"])
    with p2:
        referred       = st.selectbox("Referred Friend", ["Yes", "No"])
        paperless      = st.selectbox("Paperless Billing",["Yes", "No"])
        contract       = st.selectbox("Contract",        ["Month-to-Month", "One Year", "Two Year"])
        payment_method = st.selectbox("Payment Method",  ["Bank Withdrawal", "Credit Card", "Mailed Check"])

    st.markdown('</div>', unsafe_allow_html=True)

    # ── CLTV / Score Categories ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="sec-label">🏷️ Score Categories</p>', unsafe_allow_html=True)

    cltv_cat = st.selectbox("CLTV Category", [
        "2000-2500","2501-3000","3001-3500","3501-4000","4001-4500",
        "4501-5000","5001-5500","5501-6000","6001-6500","6501-7000"
    ], index=4)
    churn_score_cat = st.selectbox("Churn Score Category", [
        "0-10","11-20","21-30","31-40","41-50",
        "51-60","61-70","71-80","81-90","91-100"
    ], index=4)
    sat_label = st.selectbox("Satisfaction Score Label", [
        "Very Unsatisfied","Unsatisfied","Neutral","Satisfied","Very Satisfied"
    ], index=2)

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════
# COLUMN 3 — Services + Result
# ═══════════════════════════════════════
with c3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="sec-label">🌐 Services</p>', unsafe_allow_html=True)

    phone_service   = st.selectbox("Phone Service",          ["Yes", "No"])
    multiple_lines  = st.selectbox("Multiple Lines",         ["Yes", "No"])
    internet_service= st.selectbox("Internet Service",       ["Fiber Optic", "Cable", "DSL", "No"])
    s1, s2 = st.columns(2)
    with s1:
        online_security  = st.selectbox("Online Security",       ["Yes", "No"])
        online_backup    = st.selectbox("Online Backup",          ["Yes", "No"])
        device_protect   = st.selectbox("Device Protection",      ["Yes", "No"])
        premium_tech     = st.selectbox("Premium Tech Support",   ["Yes", "No"])
    with s2:
        streaming_tv     = st.selectbox("Streaming TV",           ["Yes", "No"])
        streaming_movies = st.selectbox("Streaming Movies",       ["Yes", "No"])
        streaming_music  = st.selectbox("Streaming Music",        ["Yes", "No"])
        unlimited_data   = st.selectbox("Unlimited Data",         ["Yes", "No"])

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Predict Button ──
    predict = st.button("⚡  PREDICT CHURN RISK", use_container_width=True)
    result_box = st.empty()

    # ── Risk guide ──
    st.markdown("""
    <div class="card" style="margin-top:.8rem">
      <p class="sec-label">📊 Risk Legend</p>
      <div style="display:flex;gap:.5rem;flex-wrap:wrap;font-size:.78rem">
        <span style="background:rgba(16,185,129,.12);border:1px solid rgba(16,185,129,.35);
              border-radius:8px;padding:4px 12px;color:#10b981">0–30 % · Low</span>
        <span style="background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.35);
              border-radius:8px;padding:4px 12px;color:#f59e0b">30–60 % · Medium</span>
        <span style="background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.35);
              border-radius:8px;padding:4px 12px;color:#ef4444">60–100 % · High</span>
      </div>
      <div class="fi-bar-wrap" style="margin-top:1rem">
        <p style="font-family:'Space Mono',monospace;font-size:.62rem;color:#3d5a80;
                  letter-spacing:.15em;text-transform:uppercase;margin-bottom:.7rem">
          Top Churn Drivers (LGBM)
        </p>
        <div class="fi-row"><span class="fi-name">CLTV</span><div class="fi-track"><div class="fi-fill" style="width:100%"></div></div></div>
        <div class="fi-row"><span class="fi-name">Churn Score</span><div class="fi-track"><div class="fi-fill" style="width:86%"></div></div></div>
        <div class="fi-row"><span class="fi-name">Age</span><div class="fi-track"><div class="fi-fill" style="width:76%"></div></div></div>
        <div class="fi-row"><span class="fi-name">Population</span><div class="fi-track"><div class="fi-fill" style="width:72%"></div></div></div>
        <div class="fi-row"><span class="fi-name">Avg LDist Charges</span><div class="fi-track"><div class="fi-fill" style="width:71%"></div></div></div>
        <div class="fi-row"><span class="fi-name">Monthly Charge</span><div class="fi-track"><div class="fi-fill" style="width:64%"></div></div></div>
        <div class="fi-row"><span class="fi-name">Total Charges</span><div class="fi-track"><div class="fi-fill" style="width:61%"></div></div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── PREDICTION ────────────────────────────────────────────────────────────────
if predict:
    raw = {
        # ── Numeric features (exact training names) ──
        "Age":                                age,
        "Number of Dependents":               num_dependents,
        "Zip Code":                           zip_code,
        "Latitude":                           latitude,
        "Longitude":                          longitude,
        "Population":                         population,
        "Number of Referrals":                num_referrals,
        "Tenure in Months":                   tenure_months,
        "Avg Monthly Long Distance Charges":  avg_long_dist,
        "Avg Monthly GB Download":            avg_gb_download,
        "Monthly Charge":                     monthly_charge,
        "Total Charges":                      total_charges,
        "Total Refunds":                      total_refunds,
        "Total Extra Data Charges":           total_extra_data,
        "Total Long Distance Charges":        total_long_dist,
        "Satisfaction Score":                 satisfaction,
        "Churn Score":                        churn_score,
        "CLTV":                               cltv,
        # ── Categorical features ──
        "City":                               city,
        "Gender":                             gender,
        "Senior Citizen":                     senior,
        "Married":                            married,
        "Dependents":                         dependents,
        "Referred a Friend":                  referred,
        "Phone Service":                      phone_service,
        "Multiple Lines":                     multiple_lines,
        "Internet Service":                   internet_service,
        "Online Security":                    online_security,
        "Online Backup":                      online_backup,
        "Device Protection Plan":             device_protect,
        "Premium Tech Support":               premium_tech,
        "Streaming TV":                       streaming_tv,
        "Streaming Movies":                   streaming_movies,
        "Streaming Music":                    streaming_music,
        "Unlimited Data":                     unlimited_data,
        "Contract":                           contract,
        "Paperless Billing":                  paperless,
        "Payment Method":                     payment_method,
        "Churn Score Category":               churn_score_cat,
        "CLTV Category":                      cltv_cat,
        "Satisfaction Score Label":           sat_label,
    }

    try:
        df_in  = pd.DataFrame([raw])
        df_enc = pd.get_dummies(df_in)                                    # one-hot encode
        df_fin = df_enc.reindex(columns=model_columns, fill_value=0)      # align to training cols
        df_fin = df_fin.astype(float)

        scaled     = scaler.transform(df_fin)
        prediction = model.predict(scaled)[0]
        proba      = model.predict_proba(scaled)[0]
        prob_churn = float(proba[1])
        prob_stay  = float(proba[0])
        pct        = prob_churn * 100

        # risk band
        if pct < 30:
            band, cls, gauge_css = "LOW", "res-low", "linear-gradient(90deg,#10b981,#34d399)"
            pct_cls, icon = "res-pct-low",  "✅"
            action = "Customer is likely to stay. No immediate action required."
        elif pct < 60:
            band, cls, gauge_css = "MEDIUM", "res-med", "linear-gradient(90deg,#f59e0b,#fbbf24)"
            pct_cls, icon = "res-pct-med", "⚠️"
            action = "Moderate risk. Consider a proactive check-in or targeted offer."
        else:
            band, cls, gauge_css = "HIGH", "res-high", "linear-gradient(90deg,#ef4444,#f87171)"
            pct_cls, icon = "res-pct-high", "🚨"
            action = "High churn risk! Escalate to retention team immediately."

        result_box.markdown(f"""
        <div class="{cls}">
          <div class="res-lbl">Churn Probability</div>
          <div class="res-pct {pct_cls}">{pct:.1f}%</div>
          <div class="res-desc">{icon} <strong>{band} RISK</strong> — {action}</div>
          <div class="gauge-wrap">
            <div class="gauge-track">
              <div class="gauge-fill" style="width:{pct:.1f}%;background:{gauge_css}"></div>
            </div>
          </div>
          <div class="chips">
            <div class="chip">
              <div class="chip-val">{pct:.1f}%</div>
              <div class="chip-lbl">Churn Risk</div>
            </div>
            <div class="chip">
              <div class="chip-val">{prob_stay*100:.1f}%</div>
              <div class="chip-lbl">Stay Prob</div>
            </div>
            <div class="chip">
              <div class="chip-val">{band}</div>
              <div class="chip-lbl">Risk Band</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        result_box.error(f"Prediction failed: {e}")
        with st.expander("🔍 Debug — model columns vs input columns"):
            df_in  = pd.DataFrame([raw])
            df_enc = pd.get_dummies(df_in)
            st.write("**Input columns after encoding:**", sorted(df_enc.columns.tolist()))
            st.write("**Model expects:**", sorted(model_columns))
            diff_a = set(model_columns) - set(df_enc.columns)
            diff_b = set(df_enc.columns) - set(model_columns)
            if diff_a: st.warning(f"Missing from input (filled with 0): {diff_a}")
            if diff_b: st.info(f"Extra in input (ignored): {diff_b}")


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""<hr>
<p class="footer">CHURN INTELLIGENCE DASHBOARD · LIGHTGBM · TELCO DATASET · BUILT BY ALI OSAMA</p>
""", unsafe_allow_html=True)
