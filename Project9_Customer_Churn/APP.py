import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. LOAD ASSETS ---
@st.cache_resource
def load_ml_assets():
    model = joblib.load('lgbm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, scaler, model_columns

model, scaler, model_columns = load_ml_assets()

# --- 2. PRETTY UI ---
st.set_page_config(page_title="Churn Predictor", layout="wide")

# Custom CSS for "Beauty"
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stNumberInput label, .stSlider label { color: #00d2ff !important; font-weight: bold; }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 25px;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Customer Churn Intelligence Dashboard")
st.markdown("---")

# --- 3. INPUT FORM ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Financial Metrics")
    cltv = st.number_input("CLTV", value=4000)
    monthly_charge = st.number_input("Monthly Charge", value=70.0)
    total_charges = st.number_input("Total Charges", value=2000.0)
    avg_long_dist = st.number_input("Avg Long Distance Charges", value=25.0)

with col2:
    st.subheader("Customer Profile")
    age = st.slider("Age", 18, 100, 35)
    churn_score = st.slider("Churn Score", 0, 100, 50)
    # Categorical example based on your dataset
    married = st.selectbox("Married", ["Yes", "No"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

# --- 4. PREDICTION LOGIC ---
if st.button("PREDICT CHURN RISK"):
    # Create a dataframe with the raw inputs
    input_df = pd.DataFrame({
        'CLTV': [cltv],
        'Churn Score': [churn_score],
        'Age': [age],
        'Avg Monthly Long Distance Charges': [avg_long_dist],
        'Monthly Charge': [monthly_charge],
        'Total Charges': [total_charges],
        'Married': [married],
        'Paperless Billing': [paperless]
        # ADD ALL OTHER COLUMNS FROM YOUR CSV HERE
    })

    # Transform categories to match 'pd.get_dummies' format
    input_df_encoded = pd.get_dummies(input_df)
    
    # Reindex to match the EXACT columns the model saw during training
    # This fills missing columns with 0
    input_df_final = input_df_encoded.reindex(columns=model_columns, fill_value=0)

    # Apply Scaling
    scaled_input = scaler.transform(input_df_final)

    # Predict
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    # --- 5. RESULTS ---
    if prediction == 1:
        st.markdown(f"""<div class='prediction-box' style='background-color: rgba(255, 75, 75, 0.2); border: 2px solid #ff4b4b;'>
                    🚨 HIGH RISK: {probability*100:.1f}% Churn Probability</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class='prediction-box' style='background-color: rgba(0, 235, 147, 0.2); border: 2px solid #00eb93;'>
                    ✅ LOW RISK: {probability*100:.1f}% Churn Probability</div>""", unsafe_allow_html=True)