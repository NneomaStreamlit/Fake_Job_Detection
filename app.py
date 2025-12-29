import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


# ======================================================
#               LOAD MODELS & PREPROCESSORS
# ======================================================

# ---- ML Models ----
log_reg = pickle.load(open("logreg_model.pkl", "rb"))     # fixed filename
rf_model = pickle.load(open("rf_model.pkl", "rb"))        # fixed filename
xgb_model = pickle.load(open("xgb_model.pkl", "rb"))      # fixed filename

# ---- TF-IDF Vectorizer ----
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# ---- Deep Learning Models ----
lstm_model = load_model("lstm_model.h5")
bilstm_model = load_model("bilstm_model.h5")
gru_model = load_model("gru_model.h5")

# ---- Tokenizer + Padding Length ----
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
MAX_LEN = pickle.load(open("max_len.pkl", "rb"))


# ======================================================
#              TEXT PREPROCESSING FUNCTION
# ======================================================

def preprocess_for_dl(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    return padded


# ======================================================
#              PREDICTION FUNCTIONS
# ======================================================

def predict_ml(text):
    vec = tfidf.transform([text])

    lr_p = log_reg.predict_proba(vec)[0][1]
    rf_p = rf_model.predict_proba(vec)[0][1]
    xgb_p = xgb_model.predict_proba(vec)[0][1]

    return lr_p, rf_p, xgb_p


def predict_dl(text):
    padded = preprocess_for_dl(text)

    lstm_p = lstm_model.predict(padded, verbose=0)[0][0]
    bilstm_p = bilstm_model.predict(padded, verbose=0)[0][0]
    gru_p = gru_model.predict(padded, verbose=0)[0][0]

    return lstm_p, bilstm_p, gru_p


# ======================================================
#                    STREAMLIT UI
# ======================================================

st.set_page_config(page_title="Fake Job Detector", layout="wide")
st.title("🕵️‍♂️ Fake Job Posting Detector")
st.write("Enter a job description below to check if it is **real or potentially fraudulent**.")

text_input = st.text_area("Job Description", height=250)

if st.button("Analyze Job Post"):
    if len(text_input.strip()) == 0:
        st.warning("Please enter a job description.")
        st.stop()

    # ----------------------------------------
    # ML Predictions
    # ----------------------------------------
    lr_p, rf_p, xgb_p = predict_ml(text_input)

    lr_label = "Fake" if lr_p >= 0.5 else "Real"
    rf_label = "Fake" if rf_p >= 0.5 else "Real"
    xgb_label = "Fake" if xgb_p >= 0.5 else "Real"

    st.subheader("🔎 Machine Learning Models")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("### Logistic Regression")
        st.write(f"**Prediction:** {lr_label}")
        st.write(f"**P(fake):** {lr_p:.3f}")

    with col2:
        st.write("### Random Forest")
        st.write(f"**Prediction:** {rf_label}")
        st.write(f"**P(fake):** {rf_p:.3f}")

    with col3:
        st.write("### XGBoost")
        st.write(f"**Prediction:** {xgb_label}")
        st.write(f"**P(fake):** {xgb_p:.3f}")

    # ----------------------------------------
    # Deep Learning Predictions
    # ----------------------------------------
    lstm_p, bilstm_p, gru_p = predict_dl(text_input)

    lstm_label = "Fake" if lstm_p >= 0.5 else "Real"
    bilstm_label = "Fake" if bilstm_p >= 0.5 else "Real"
    gru_label = "Fake" if gru_p >= 0.5 else "Real"

    st.subheader("🤖 Deep Learning Models")

    col4, col5, col6 = st.columns(3)

    with col4:
        st.write("### LSTM")
        st.write(f"**Prediction:** {lstm_label}")
        st.write(f"**P(fake):** {lstm_p:.3f}")

    with col5:
        st.write("### BiLSTM")
        st.write(f"**Prediction:** {bilstm_label}")
        st.write(f"**P(fake):** {bilstm_p:.3f}")

    with col6:
        st.write("### GRU")
        st.write(f"**Prediction:** {gru_label}")
        st.write(f"**P(fake):** {gru_p:.3f}")

    st.success("Analysis complete!")
