import os
import pickle
import streamlit as st

# ======================================================
#                 PATHS (Cloud + Local Safe)
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))     # .../public_app
MODELS_DIR = os.path.join(BASE_DIR, "models")             # .../public_app/models

# ======================================================
#                 LOAD MODELS (CACHED)
# ======================================================
@st.cache_resource
def load_models():
    with open(os.path.join(MODELS_DIR, "tfidf.pkl"), "rb") as f:
        tfidf = pickle.load(f)

    with open(os.path.join(MODELS_DIR, "logreg_model.pkl"), "rb") as f:
        log_reg = pickle.load(f)

    with open(os.path.join(MODELS_DIR, "rf_model.pkl"), "rb") as f:
        rf_model = pickle.load(f)

    with open(os.path.join(MODELS_DIR, "xgb_model.pkl"), "rb") as f:
        xgb_model = pickle.load(f)

    return tfidf, log_reg, rf_model, xgb_model


tfidf, log_reg, rf_model, xgb_model = load_models()

# ======================================================
#                 STREAMLIT UI
# ======================================================
st.set_page_config(page_title="Fake Job Detector (Public ML App)", layout="wide")

st.title("🕵️ Fake Job Posting Detector (Public ML App)")
st.write(
    "This public demo uses **Machine Learning models** "
    "(Logistic Regression, Random Forest, and XGBoost) "
    "to assess whether a job post may be fraudulent."
)

# Optional: quick debug panel (helps when deploying)
with st.expander("🔧 Debug (paths)", expanded=False):
    st.write("BASE_DIR:", BASE_DIR)
    st.write("MODELS_DIR:", MODELS_DIR)
    st.write("Files found in models/:", os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else "❌ models/ not found")

job_text = st.text_area("Paste Job Description", height=250)

# ======================================================
#                 PREDICTION
# ======================================================
def label_from_prob(p: float) -> str:
    return "Fake" if p >= 0.5 else "Real"


if st.button("Analyze Job Post"):
    if not job_text.strip():
        st.warning("Please enter a job description.")
        st.stop()

    X = tfidf.transform([job_text])

    lr_p = float(log_reg.predict_proba(X)[0][1])
    rf_p = float(rf_model.predict_proba(X)[0][1])
    xgb_p = float(xgb_model.predict_proba(X)[0][1])

    st.subheader("🔎 Model Predictions")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Logistic Regression",
            label_from_prob(lr_p),
            f"P(fake) = {lr_p:.3f}",
        )

    with col2:
        st.metric(
            "Random Forest",
            label_from_prob(rf_p),
            f"P(fake) = {rf_p:.3f}",
        )

    with col3:
        st.metric(
            "XGBoost",
            label_from_prob(xgb_p),
            f"P(fake) = {xgb_p:.3f}",
        )

    st.success("Analysis complete ✅")

