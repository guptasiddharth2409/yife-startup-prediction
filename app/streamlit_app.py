"""YIFE Startup Outcome Demo - Interactive Streamlit Dashboard.

This interface is a demonstration/educational wrapper around a trained XGBoost
YIFE pipeline. It is not investment advice and is not a strict ex-ante forecasting
tool because some study predictors can contain post-YC information.

Run: streamlit run app/streamlit_app.py
"""

import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="YIFE Startup Outcome Demo",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_DIR / "xgboost.pkl")


def predict(inputs: dict, model):
    # The current training pipeline owns imputation, scaling, and one-hot encoding.
    # Pass raw feature values so inference uses exactly the same preprocessing path.
    X = pd.DataFrame([inputs])
    prob = float(model.predict_proba(X)[0, 1])
    pred = int(prob >= 0.5)
    return pred, prob


st.title("🚀 YIFE: Startup Outcome Classification Demo")
st.markdown(
    "Explore the published YIFE feature framework with the trained XGBoost model. "
    "The published model achieved **F1=0.85** and **AUROC=0.91** on the held-out "
    "W21–S24 YC cohort (n=863)."
)
st.info(
    "Research-use note: the published study is a retrospective, domain-contextualized "
    "classification approach. Some funding and investor-network variables may contain "
    "post-incubation information, so this demo should not be treated as investment advice "
    "or a strict application-time forecasting system."
)
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("💰 Funding")
    total_funding = st.number_input("Total Funding Raised (USD)", 0.0, 5e9, 1_500_000.0, step=100_000.0, format="%.0f")
    seed_size = st.number_input("Seed Round Size (USD)", 0.0, 1e9, 500_000.0, step=50_000.0, format="%.0f")
    funding_rounds = st.slider("Number of Funding Rounds", 0, 10, 2)
    tier1_vc = st.selectbox("Tier-1 VC Backed?", [0, 1], format_func=lambda x: "Yes" if x else "No")

with col2:
    st.subheader("👥 Team & Tech")
    team_size = st.slider("Co-founder Count", 1, 6, 2)
    faang_exp = st.selectbox("Prior FAANG Experience?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    elite_edu = st.selectbox("Elite University (Top-20)?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    github_repos = st.number_input("Total Public GitHub Repos", 0, 500, 25)
    github_commits = st.number_input("Avg Weekly Commits (pre-YC)", 0.0, 100.0, 12.5)

with col3:
    st.subheader("🌐 Context")
    batch_year = st.slider("YC Batch Year", 2005, 2024, 2023)
    batch_size = st.number_input("Batch Size (# companies)", 10, 500, 250)
    industry = st.selectbox("Industry Category", ["B2B", "AI", "FinTech", "Consumer", "Healthcare", "DevTools", "Other"])
    geo = st.selectbox("Location", ["SF Bay", "NY", "International", "Other"])
    ai_flag = st.selectbox("AI-Core Product?", [0, 1], format_func=lambda x: "Yes" if x else "No")

st.divider()

if st.button("🔮 Evaluate Startup Outcome", type="primary", use_container_width=True):
    try:
        model = load_model()
        inputs = {
            "total_funding_usd": total_funding,
            "num_funding_rounds": funding_rounds,
            "seed_round_size": seed_size,
            "team_size": team_size,
            "faang_experience": faang_exp,
            "elite_edu": elite_edu,
            "github_repo_count": github_repos,
            "github_commit_freq": github_commits,
            "batch_year_encoded": batch_year,
            "batch_size": batch_size,
            "industry_category": industry,
            "ai_flag": ai_flag,
            "geo_cluster": geo,
            "tier1_vc_investor": tier1_vc,
        }
        pred, prob = predict(inputs, model)
        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Predicted Success Probability", f"{prob * 100:.1f}%")
        with r2:
            if pred == 1:
                st.success("**Predicted: Successful Outcome Class**")
            else:
                st.error("**Predicted: Unsuccessful Outcome Class**")
        with r3:
            st.metric("Decision Threshold", "0.50")

        st.divider()
        st.subheader("📈 Global SHAP Signals")
        st.caption("Mean absolute SHAP values show relative attribution magnitude; they do not establish causal direction or monotonic effects.")
        shap_df = pd.DataFrame({
            "Feature": [
                "num_funding_rounds", "batch_year_encoded", "team_size",
                "total_funding_usd", "ai_flag", "industry_category (B2B)",
                "github_commit_freq", "geo_cluster (San Francisco Bay)",
                "elite_edu", "faang_experience",
            ],
            "Mean |SHAP|": [0.187, 0.163, 0.141, 0.128, 0.112, 0.094, 0.087, 0.071, 0.052, 0.031],
        })
        st.dataframe(shap_df, hide_index=True, use_container_width=True)
    except FileNotFoundError:
        st.warning(
            "The trained XGBoost artifact was not found. Run the training pipeline first:\n\n"
            "```bash\npython src/train/trainer.py\n```"
        )

with st.sidebar:
    st.header("📊 About YIFE")
    st.markdown(
        "**Paper:** Predicting Startup Outcomes Using Explainable Machine Learning and Y Combinator-Inspired Feature Engineering  \n"
        "**Authors:** Siddharth Gupta, Pratham Namdev, Shubham Nagar, Sunny Singh, Anjali Deshwal  \n"
        "**Dataset:** 4,323 YC companies (2005–2024)  \n"
        "**Champion:** XGBoost + YIFE  \n"
        "**Held-out cohort:** W21–S24 (n=863)  \n"
        "**F1:** 0.85  \n"
        "**AUROC:** 0.91  \n"
        "**DOI:** 10.7759/s44389-026-00254-0"
    )
    st.divider()
    st.subheader("Published Model Performance")
    results = {
        "Model": ["Logistic Reg B1", "Logistic Reg YIFE", "RF B2", "RF YIFE", "XGBoost YIFE ★", "SVM YIFE", "MLP YIFE"],
        "F1": [0.66, 0.71, 0.77, 0.80, 0.85, 0.76, 0.79],
        "AUROC": [0.74, 0.79, 0.85, 0.88, 0.91, 0.83, 0.86],
    }
    st.dataframe(pd.DataFrame(results), hide_index=True, use_container_width=True)
