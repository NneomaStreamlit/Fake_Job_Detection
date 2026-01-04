import os
import pickle
import streamlit as st


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# -----------------------------
# Load models safely
# -----------------------------
with open(os.path.join(MODELS_DIR, "tfidf.pkl"), "rb") as f:
    tfidf = pickle.load(f)

with open(os.path.join(MODELS_DIR, "logreg_model.pkl"), "rb") as f:
    log_reg = pickle.load(f)

with open(os.path.join(MODELS_DIR, "rf_model.pkl"), "rb") as f:
    rf_model = pickle.load(f)

with open(os.path.join(MODELS_DIR, "xgb_model.pkl"), "rb") as f:
    xgb_model = pickle.load(f)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Fake Job Detector", layout="wide")

st.title("🕵️ Fake Job Posting Detector Public App")
st.write(
    "This public demo uses **Machine Learning models** "
    "(Logistic Regression, Random Forest, and XGBoost) "
    "to assess whether a job post may be fraudulent."
)

job_text = st.text_area("Paste Job Description", height=250)

if st.button("Analyze Job Post"):
    if not job_text.strip():
        st.warning("Please enter a job description.")
        st.stop()

    X = tfidf.transform([job_text])

    lr_p = logreg.predict_proba(X)[0][1]
    rf_p = rf.predict_proba(X)[0][1]
    xgb_p = xgb.predict_proba(X)[0][1]

    st.subheader("🔎 Model Predictions")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Logistic Regression", "Fake" if lr_p >= 0.5 else "Real", f"P(fake)={lr_p:.3f}")

    with col2:
        st.metric("Random Forest", "Fake" if rf_p >= 0.5 else "Real", f"P(fake)={rf_p:.3f}")

    with col3:
        st.metric("XGBoost", "Fake" if xgb_p >= 0.5 else "Real", f"P(fake)={xgb_p:.3f}")

    st.success("Analysis complete.")
