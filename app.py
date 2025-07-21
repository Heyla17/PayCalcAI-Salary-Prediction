import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Page Config
# ------------------------------------------------------------------
st.set_page_config(page_title="PayCalAI", page_icon="💼", layout="centered")
st.title("💼 PayCalAI – Startup Salary Prediction")
st.write("Predict salaries for new tech hires based on **Job Title, Education, and Skills**. Compare Linear Regression and Gradient Boosting models.")

# ------------------------------------------------------------------
# Load models & scaler
# ------------------------------------------------------------------
@st.cache_resource
def load_linear_model():
    return joblib.load("model_lr.pkl")

@st.cache_resource
def load_gb_model():
    return joblib.load("model_gb.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

model_lr = load_linear_model()
model_gb = load_gb_model()
scaler = load_scaler()

# ------------------------------------------------------------------
# Encoding maps (must match what you used during training!)
# ------------------------------------------------------------------
job_title_map = {
    "AI Research Scientist": 0,
    "AI Software Engineer": 1,
    "AI Specialist": 2,
    "NLP Engineer": 3,
    "AI Consultant": 4
}

education_map = {
    "Bachelor": 0,
    "Master": 1,
    "PhD": 2,
    "Associate": 3
}

all_skills = [
    "Python", "SQL", "AWS", "Java", "Linux", "Docker",
    "TensorFlow", "PyTorch", "Tableau", "Hadoop", "Scala",
    "Kubernetes", "NLP", "MLOps", "Git"
]

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def preprocess_inputs(job_title, education, skill_list):
    job_title_encoded = job_title_map[job_title]
    education_encoded = education_map[education]
    skill_count = len(skill_list)

    X = np.array([[job_title_encoded, education_encoded, skill_count]])
    X_scaled = scaler.transform(X)
    return X_scaled, {
        "job_title_encoded": job_title_encoded,
        "education_encoded": education_encoded,
        "skill_count": skill_count
    }

def predict_linear(X_scaled):
    return float(model_lr.predict(X_scaled)[0])

def predict_gb(X_scaled):
    return float(model_gb.predict(X_scaled)[0])

# ------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------
st.subheader("🔮 Single Salary Prediction")

col1, col2 = st.columns(2)
with col1:
    job_title = st.selectbox("Job Title", list(job_title_map.keys()))
    education = st.selectbox("Education Level", list(education_map.keys()))
with col2:
    chosen_skills = st.multiselect("Select Skills", all_skills, default=["Python", "SQL"])
    extra_skill_text = st.text_input("Add custom skills (comma separated)", "")
    if extra_skill_text.strip():
        extra = [s.strip() for s in extra_skill_text.split(",") if s.strip()]
        chosen_skills = list(dict.fromkeys(chosen_skills + extra))

model_choice = st.radio("Model to Use", ["Linear Regression", "Gradient Boosting"], horizontal=True)

if st.button("Predict Salary 💰"):
    X_scaled, feats = preprocess_inputs(job_title, education, chosen_skills)
    if model_choice == "Linear Regression":
        salary = predict_linear(X_scaled)
        used_model = "Linear Regression"
    else:
        salary = predict_gb(X_scaled)
        used_model = "Gradient Boosting"

    st.success(f"**Predicted Salary (USD): ${salary:,.2f}**")
    with st.expander("Feature Encoding Details"):
        st.json(feats)
    st.caption(f"Model used: {used_model}")
