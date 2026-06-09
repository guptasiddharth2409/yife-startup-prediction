"""
app/streamlit_app.py

YIFE Startup Success Predictor - Interactive Streamlit Dashboard
Predicts startup success probability using the XGBoost YIFE model.

Run: streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="YIFE Startup Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Styling ----
st.markdown("""
<style>
.metric-box { background:#f0f2f6; border-radius:8px; padding:16px; text-align:center; }
.success-badge { background:#d4edda; color:#155724; border-radius:4px; padding:4px 10px; font-weight:600; }
.fail-badge    { background:#f8d7da; color:#721c24; border-radius:4px; padding:4px 10px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

MODEL_DIR = Path(__file__).resolve().parent.parent / 'models'


@st.cache_resource
def load_artifacts():
    model   = joblib.load(MODEL_DIR / 'xgboost.pkl')
    scaler  = joblib.load(MODEL_DIR / 'scaler.pkl')
    imputer = joblib.load(MODEL_DIR / 'imputer.pkl')
    return model, scaler, imputer


def predict(inputs: dict, model, imputer, scaler):
    feature_order = [
        'total_funding_usd', 'num_funding_rounds', 'seed_round_size',
        'team_size', 'faang_experience', 'elite_edu',
        'github_repo_count', 'github_commit_freq',
        'batch_year_encoded', 'batch_size',
        'industry_category', 'ai_flag', 'geo_cluster', 'tier1_vc_investor'
    ]
    # Encode categoricals
    industry_map = {'B2B':0,'AI':1,'FinTech':2,'Consumer':3,'Healthcare':4,'DevTools':5,'Other':6}
    geo_map      = {'SF Bay':0,'NY':1,'International':2,'Other':3}
    inputs['industry_category'] = industry_map.get(inputs['industry_category'], 6)
    inputs['geo_cluster']       = geo_map.get(inputs['geo_cluster'], 3)

    X = np.array([[inputs[f] for f in feature_order]])
    X = imputer.transform(X)
    prob = model.predict_proba(X)[0][1]
    pred = int(prob >= 0.5)
    return pred, prob


# ---- UI ----
st.title("🚀 YIFE: Early-Stage Startup Success Predictor")
st.markdown(
    "Enter your startup's metrics below to predict the probability of "
    "reaching Series A or a successful exit, powered by the "
    "**YC-Inspired Feature Engineering (YIFE)** XGBoost model "
    "(F1=0.85, AUROC=0.91 on held-out YC test set)."
)
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("💰 Funding")
    total_funding   = st.number_input("Total Funding Raised (USD)", 0.0, 5e9, 1_500_000.0, step=100_000.0, format="%.0f")
    seed_size       = st.number_input("Seed Round Size (USD)", 0.0, 1e9,   500_000.0,   step=50_000.0,  format="%.0f")
    funding_rounds  = st.slider("Number of Funding Rounds", 0, 10, 2)
    tier1_vc        = st.selectbox("Tier-1 VC Backed?", [0, 1], format_func=lambda x: "Yes" if x else "No")

with col2:
    st.subheader("👥 Team & Tech")
    team_size        = st.slider("Co-founder Count", 1, 6, 2)
    faang_exp        = st.selectbox("Prior FAANG Experience?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    elite_edu        = st.selectbox("Elite University (Top-20)?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    github_repos     = st.number_input("Total Public GitHub Repos", 0, 500, 25)
    github_commits   = st.number_input("Avg Weekly Commits (pre-batch)", 0.0, 100.0, 12.5)

with col3:
    st.subheader("🌐 Context")
    batch_year       = st.slider("YC Batch Year", 2005, 2025, 2023)
    batch_size       = st.number_input("Batch Size (# companies)", 10, 500, 250)
    industry         = st.selectbox("Industry Category", ['B2B','AI','FinTech','Consumer','Healthcare','DevTools','Other'])
    geo              = st.selectbox("Location", ['SF Bay','NY','International','Other'])
    ai_flag          = st.selectbox("AI-Core Product?", [0, 1], format_func=lambda x: "Yes" if x else "No")

st.divider()

if st.button("🔮 Predict Startup Success", type="primary", use_container_width=True):
    try:
        model, scaler, imputer = load_artifacts()
        inputs = {
            'total_funding_usd':  total_funding,
            'num_funding_rounds': funding_rounds,
            'seed_round_size':    seed_size,
            'team_size':          team_size,
            'faang_experience':   faang_exp,
            'elite_edu':          elite_edu,
            'github_repo_count':  github_repos,
            'github_commit_freq': github_commits,
            'batch_year_encoded': batch_year,
            'batch_size':         batch_size,
            'industry_category':  industry,
            'ai_flag':            ai_flag,
            'geo_cluster':        geo,
            'tier1_vc_investor':  tier1_vc,
        }
        pred, prob = predict(inputs, model, imputer, scaler)
        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Success Probability", f"{prob*100:.1f}%")
        with r2:
            if pred == 1:
                st.success("🎉 **Predicted: Successful Trajectory**")
            else:
                st.error("⚠️ **Predicted: High Risk of Stagnation**")
        with r3:
            st.metric("Model Confidence", "High" if abs(prob - 0.5) > 0.2 else "Moderate")
        st.divider()
        st.subheader("📈 Key Signals from SHAP Analysis")
        st.markdown("""
        Based on the paper's SHAP analysis (Table 6), the strongest predictors are:
        - 🥇 **Funding rounds** (SHAP=0.187): More rounds = higher success signal
        - 📅 **Batch year** (SHAP=0.163): 2020+ AI-era batches show higher success rates
        - 👥 **Team size** (SHAP=0.141): Optimal range is **2–4 co-founders**
        - 💰 **Total funding** (SHAP=0.128): Positive but with diminishing returns
        - 🤖 **AI flag** (SHAP=0.112): AI-core products strongly positive post-2020
        - ❌ **FAANG experience** (SHAP=0.031): Surprisingly weak — challenges common VC heuristic
        """)
    except FileNotFoundError:
        st.warning(
            "Model artifacts not found. Please run:\n"
            "```\npython scripts/02_train_model.py\n```"
            "\nfirst to train and save the models."
        )

# ---- Sidebar ----
with st.sidebar:
    st.header("📊 About YIFE")
    st.markdown("""
    **Paper:** Predicting Early-Stage Startup Success Using ML: A YC-Inspired Feature Engineering Approach  
    **Author:** Siddharth Gupta, GNIOT  
    **Dataset:** 4,323 YC companies (2005–2024)  
    **Champion Model:** XGBoost  
    **F1-Score:** 0.85  
    **AUROC:** 0.91  
    **Test Set:** W21–S24 batches (n=863)
    """)
    st.divider()
    st.subheader("Model Performance")
    results = {
        "Model": ["Logistic Reg", "Random Forest", "XGBoost ★", "SVM", "MLP"],
        "F1":    [0.71, 0.80, 0.85, 0.76, 0.79],
        "AUROC": [0.79, 0.88, 0.91, 0.83, 0.86],
    }
    st.dataframe(pd.DataFrame(results), hide_index=True)
