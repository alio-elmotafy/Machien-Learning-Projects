import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be the very first st call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence · Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

/* ── reset & base ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #060b14;
    color: #c9d8f0;
}
.main { background: #060b14; }
section.main > div { padding-top: 1.5rem; }

/* ── animated grid background ── */
.main::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,210,255,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,210,255,0.04) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    z-index: 0;
}

/* ── header ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: clamp(1.8rem, 4vw, 3rem);
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #00d2ff 0%, #7b2ff7 60%, #ff6eaa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
}
.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #4a6fa5;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

/* ── cards ── */
.card {
    background: linear-gradient(145deg, #0c1624 0%, #0a1220 100%);
    border: 1px solid rgba(0,210,255,0.12);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00d2ff, #7b2ff7);
    border-radius: 16px 16px 0 0;
}
.card-title {
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #00d2ff;
    margin-bottom: 1.1rem;
}

/* ── input label overrides ── */
label[data-testid="stWidgetLabel"] p,
div[data-testid="stSlider"] label p {
    color: #7ea8c9 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
}

/* ── button ── */
div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #00d2ff, #7b2ff7);
    color: #ffffff;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.08em;
    border: none;
    border-radius: 12px;
    padding: 0.85rem 2rem;
    cursor: pointer;
    transition: all 0.25s ease;
    box-shadow: 0 0 20px rgba(0,210,255,0.25);
    margin-top: 1rem;
}
div.stButton > button:hover {
    box-shadow: 0 0 35px rgba(123,47,247,0.5);
    transform: translateY(-2px);
}

/* ── result boxes ── */
.result-high {
    background: linear-gradient(145deg, rgba(255,75,75,0.12), rgba(200,40,40,0.06));
    border: 1px solid rgba(255,75,75,0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    animation: pulse-red 2s infinite;
}
.result-low {
    background: linear-gradient(145deg, rgba(0,235,147,0.1), rgba(0,180,100,0.05));
    border: 1px solid rgba(0,235,147,0.35);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    animation: pulse-green 2s infinite;
}
@keyframes pulse-red  { 0%,100%{box-shadow:0 0 20px rgba(255,75,75,0.15)} 50%{box-shadow:0 0 40px rgba(255,75,75,0.35)} }
@keyframes pulse-green{ 0%,100%{box-shadow:0 0 20px rgba(0,235,147,0.1)}  50%{box-shadow:0 0 40px rgba(0,235,147,0.3)} }

.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    opacity: 0.7;
}
.result-pct {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: clamp(2.5rem, 6vw, 4rem);
    line-height: 1;
}
.result-pct-high { color: #ff4b4b; }
.result-pct-low  { color: #00eb93; }
.result-desc {
    font-size: 0.9rem;
    opacity: 0.65;
    margin-top: 0.6rem;
}

/* ── metric row ── */
.metric-row { display: flex; gap: 1rem; margin-top: 1.2rem; flex-wrap: wrap; }
.metric-chip {
    flex: 1;
    min-width: 110px;
    background: rgba(0,210,255,0.07);
    border: 1px solid rgba(0,210,255,0.15);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    text-align: center;
}
.metric-chip-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.15rem;
    font-weight: 700;
    color: #00d2ff;
}
.metric-chip-lbl { font-size: 0.68rem; opacity: 0.5; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 2px; }

/* ── gauge ── */
.gauge-wrap { margin: 1.2rem auto 0; max-width: 340px; }
.gauge-track {
    height: 10px;
    background: rgba(255,255,255,0.08);
    border-radius: 99px;
    overflow: hidden;
}
.gauge-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 1s cubic-bezier(.4,0,.2,1);
}

/* ── divider ── */
hr { border: none; border-top: 1px solid rgba(0,210,255,0.1); margin: 1.5rem 0; }

/* ── selectbox / number input tweaks ── */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stNumberInput"] input {
    background-color: #0c1624 !important;
    border-color: rgba(0,210,255,0.18) !important;
    color: #c9d8f0 !important;
    border-radius: 8px !important;
}

/* ── error / warning banners ── */
.warn-box {
    background: rgba(255,200,0,0.08);
    border: 1px solid rgba(255,200,0,0.3);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.85rem;
    color: #ffd966;
    font-family: 'Space Mono', monospace;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ASSET LOADING  (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model assets…")
def load_assets():
    search_dirs = [
        ".",
        "Project9_Customer_Churn",
        os.path.dirname(__file__),
        os.path.join(os.path.dirname(__file__), "Project9_Customer_Churn"),
    ]

    def find(filename):
        for d in search_dirs:
            p = os.path.join(d, filename)
            if os.path.exists(p):
                return p
        return None

    model_path  = find("lgbm_model.pkl")
    scaler_path = find("scaler.pkl")
    cols_path   = find("model_columns.pkl")

    missing = [n for n, p in [("lgbm_model.pkl", model_path),
                               ("scaler.pkl",     scaler_path),
                               ("model_columns.pkl", cols_path)] if p is None]
    if missing:
        return None, None, None, missing

    model         = joblib.load(model_path)
    scaler        = joblib.load(scaler_path)
    model_columns = joblib.load(cols_path)
    return model, scaler, model_columns, []


model, scaler, model_columns, missing_files = load_assets()


# ─────────────────────────────────────────────
# HELPER — build & encode input row
# ─────────────────────────────────────────────
def build_input(raw: dict, model_columns: list) -> np.ndarray:
    """
    Converts raw UI values → one-hot encoded → reindexed to model_columns → scaled.
    Returns the scaled numpy array ready for model.predict / predict_proba.
    """
    df = pd.DataFrame([raw])

    # Identify categorical columns (object dtype) and apply get_dummies
    df_encoded = pd.get_dummies(df)

    # Ensure every column the model expects is present; fill unknowns with 0
    df_final = df_encoded.reindex(columns=model_columns, fill_value=0)

    # Cast to float to be safe
    df_final = df_final.astype(float)

    scaled = scaler.transform(df_final)
    return scaled


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:1.8rem">
  <p class="hero-title">📡 Churn Intelligence</p>
  <p class="hero-sub">Customer Risk Prediction · LightGBM Model</p>
</div>
""", unsafe_allow_html=True)

if missing_files:
    st.markdown(f"""
    <div class="warn-box">
    ⚠️  Model files not found: <b>{', '.join(missing_files)}</b><br>
    Place them alongside <code>app.py</code> (or in a <code>Project9_Customer_Churn/</code> subfolder)
    and restart the app.  Predictions are disabled until all files are loaded.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    # ── Financial card ──
    st.markdown('<div class="card"><p class="card-title">💳 Financial Metrics</p>', unsafe_allow_html=True)
    cltv            = st.number_input("Customer Lifetime Value (CLTV)", min_value=0, value=4000, step=100)
    monthly_charge  = st.number_input("Monthly Charge ($)", min_value=0.0, value=70.0, step=1.0, format="%.2f")
    total_charges   = st.number_input("Total Charges ($)", min_value=0.0, value=2000.0, step=50.0, format="%.2f")
    avg_long_dist   = st.number_input("Avg Monthly Long-Distance Charges ($)", min_value=0.0, value=25.0, step=1.0, format="%.2f")
    avg_gb_download = st.number_input("Avg Monthly GB Download", min_value=0.0, value=15.0, step=1.0, format="%.1f")
    tenure_months   = st.number_input("Tenure (months)", min_value=0, value=24, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Service card ──
    st.markdown('<div class="card"><p class="card-title">🌐 Service Details</p>', unsafe_allow_html=True)
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        internet_type   = st.selectbox("Internet Type",   ["Fiber Optic", "Cable", "DSL", "None"])
        contract        = st.selectbox("Contract",        ["Month-to-Month", "One Year", "Two Year"])
        payment_method  = st.selectbox("Payment Method",  ["Bank Transfer", "Credit Card", "Mailed Check", "Electronic Check"])
    with s_col2:
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        tech_support    = st.selectbox("Tech Support",    ["Yes", "No", "No internet service"])
        streaming_tv    = st.selectbox("Streaming TV",    ["Yes", "No", "No internet service"])
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    # ── Profile card ──
    st.markdown('<div class="card"><p class="card-title">👤 Customer Profile</p>', unsafe_allow_html=True)
    age            = st.slider("Age", 18, 100, 35)
    churn_score    = st.slider("Churn Score (0–100)", 0, 100, 50)
    number_calls   = st.slider("Number of Customer Service Calls", 0, 20, 2)
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        gender         = st.selectbox("Gender",            ["Male", "Female"])
        married        = st.selectbox("Married",           ["Yes", "No"])
        dependents     = st.selectbox("Dependents",        ["Yes", "No"])
    with p_col2:
        senior         = st.selectbox("Senior Citizen",    ["Yes", "No"])
        partner        = st.selectbox("Partner",           ["Yes", "No"])
        paperless      = st.selectbox("Paperless Billing", ["Yes", "No"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Predict button + result live here ──
    predict_clicked = st.button("⚡  PREDICT CHURN RISK", use_container_width=True)

    result_placeholder = st.empty()

    # ── Risk Guide ──
    st.markdown("""
    <div class="card" style="margin-top:0.5rem">
      <p class="card-title">📊 Risk Guide</p>
      <div style="display:flex;gap:0.6rem;flex-wrap:wrap;font-size:0.8rem">
        <span style="background:rgba(0,235,147,0.15);border:1px solid rgba(0,235,147,0.35);border-radius:6px;padding:4px 10px;color:#00eb93">0–30 % · Low</span>
        <span style="background:rgba(255,200,0,0.1);border:1px solid rgba(255,200,0,0.3);border-radius:6px;padding:4px 10px;color:#ffd966">30–60 % · Medium</span>
        <span style="background:rgba(255,75,75,0.12);border:1px solid rgba(255,75,75,0.35);border-radius:6px;padding:4px 10px;color:#ff4b4b">60–100 % · High</span>
      </div>
      <p style="font-size:0.76rem;opacity:0.45;margin-top:0.9rem;font-family:'Space Mono',monospace;line-height:1.6">
        Model: LightGBM · Features scaled with StandardScaler
      </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────────
if predict_clicked:
    if missing_files:
        result_placeholder.error("Cannot predict — model files are missing. See warning above.")
    else:
        # ── Assemble raw dict matching original training feature names ──
        raw = {
            # numeric
            "CLTV":                              cltv,
            "Churn Score":                       churn_score,
            "Age":                               age,
            "Avg Monthly Long Distance Charges": avg_long_dist,
            "Monthly Charge":                    monthly_charge,
            "Total Charges":                     total_charges,
            "Avg Monthly GB Download":           avg_gb_download,
            "Tenure in Months":                  tenure_months,
            "Number of Customer Service Calls":  number_calls,
            # categorical  (get_dummies will one-hot these)
            "Gender":           gender,
            "Married":          married,
            "Dependents":       dependents,
            "Senior Citizen":   senior,
            "Partner":          partner,
            "Paperless Billing":    paperless,
            "Internet Type":        internet_type,
            "Contract":             contract,
            "Payment Method":       payment_method,
            "Online Security":      online_security,
            "Tech Support":         tech_support,
            "Streaming TV":         streaming_tv,
        }

        try:
            scaled_input = build_input(raw, model_columns)
            prediction   = model.predict(scaled_input)[0]
            proba        = model.predict_proba(scaled_input)[0]
            prob_churn   = float(proba[1])          # probability of class=1 (churn)
            prob_stay    = float(proba[0])

            # colour-code gauge
            if prob_churn < 0.30:
                gauge_color = "linear-gradient(90deg,#00eb93,#00d2a0)"
            elif prob_churn < 0.60:
                gauge_color = "linear-gradient(90deg,#ffd966,#ff9a3c)"
            else:
                gauge_color = "linear-gradient(90deg,#ff6060,#ff2222)"

            pct = prob_churn * 100

            if prediction == 1:
                result_placeholder.markdown(f"""
                <div class="result-high">
                  <div class="result-label">Churn Probability</div>
                  <div class="result-pct result-pct-high">{pct:.1f}%</div>
                  <div class="result-desc">🚨 This customer is <strong>likely to churn</strong>. Consider a retention offer.</div>
                  <div class="gauge-wrap">
                    <div class="gauge-track">
                      <div class="gauge-fill" style="width:{pct:.1f}%;background:{gauge_color}"></div>
                    </div>
                  </div>
                  <div class="metric-row">
                    <div class="metric-chip">
                      <div class="metric-chip-val">{pct:.1f}%</div>
                      <div class="metric-chip-lbl">Churn Risk</div>
                    </div>
                    <div class="metric-chip">
                      <div class="metric-chip-val">{prob_stay*100:.1f}%</div>
                      <div class="metric-chip-lbl">Retention</div>
                    </div>
                    <div class="metric-chip">
                      <div class="metric-chip-val">{'High' if pct>=60 else 'Med'}</div>
                      <div class="metric-chip-lbl">Risk Band</div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                result_placeholder.markdown(f"""
                <div class="result-low">
                  <div class="result-label">Churn Probability</div>
                  <div class="result-pct result-pct-low">{pct:.1f}%</div>
                  <div class="result-desc">✅ This customer is <strong>likely to stay</strong>. Keep up the good work!</div>
                  <div class="gauge-wrap">
                    <div class="gauge-track">
                      <div class="gauge-fill" style="width:{pct:.1f}%;background:{gauge_color}"></div>
                    </div>
                  </div>
                  <div class="metric-row">
                    <div class="metric-chip">
                      <div class="metric-chip-val">{pct:.1f}%</div>
                      <div class="metric-chip-lbl">Churn Risk</div>
                    </div>
                    <div class="metric-chip">
                      <div class="metric-chip-val">{prob_stay*100:.1f}%</div>
                      <div class="metric-chip-lbl">Retention</div>
                    </div>
                    <div class="metric-chip">
                      <div class="metric-chip-val">Low</div>
                      <div class="metric-chip-lbl">Risk Band</div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            result_placeholder.error(f"Prediction error: {e}")
            st.exception(e)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<hr>
<p style="font-family:'Space Mono',monospace;font-size:0.68rem;opacity:0.3;text-align:center;letter-spacing:0.12em">
CHURN INTELLIGENCE DASHBOARD · LGBM · STANDARDSCALER · FOR INTERNAL USE ONLY
</p>
""", unsafe_allow_html=True)
