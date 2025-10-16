# ============================================================
# 🚀 Spam URL Detection - Streamlit Web App (RF + SVM)
# ============================================================

import streamlit as st
import pandas as pd
import joblib
import re
from datetime import datetime
from scipy.sparse import hstack

# ============================================================
# Load Models and Vectorizers
# ============================================================
try:
    rf_model = joblib.load("spam_url_model.pkl")
    rf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

    svm_model = joblib.load("svm_spam_model.pkl")
    svm_vectorizer = joblib.load("svm_vectorizer.pkl")
except FileNotFoundError:
    st.error("One or more model files not found. Ensure spam_url_model.pkl, tfidf_vectorizer.pkl, svm_spam_model.pkl, svm_vectorizer.pkl are in the directory.")
    st.stop()

# ============================================================
# Feature Preprocessing
# ============================================================
def clean_url(url):
    url = str(url).lower()
    url = re.sub(r'https?://', '', url)
    url = re.sub(r'www\.?', '', url)
    url = url.strip().strip('/')
    return url

def extract_features(url):
    return {
        "url_length": len(url),
        "count_digits": sum(c.isdigit() for c in url),
        "count_dots": url.count('.'),
        "count_hyphens": url.count('-'),
        "count_at": url.count('@'),
        "count_question": url.count('?'),
        "count_equals": url.count('='),
        "has_ip": 1 if re.search(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', url) else 0,
        "has_suspicious_word": 1 if any(w in url for w in 
            ['login','verify','update','free','click','secure',
            'account','bank','signin','confirm','password']) else 0
    }

# ============================================================
# Prediction Function
# ============================================================
def predict_url(url, model_choice="Both"):
    clean = clean_url(url)
    features_dict = extract_features(clean)
    numeric_features = pd.DataFrame([features_dict]).values

    results = {}

    # Random Forest uses combined TF-IDF + numeric features
    if model_choice in ["RF", "Both"]:
        tfidf_rf = rf_vectorizer.transform([clean])
        combined_rf = hstack([tfidf_rf, numeric_features])
        pred_rf = rf_model.predict(combined_rf)[0]
        results["RF"] = "🚨 SPAM" if pred_rf == 1 else "✅ SAFE"

    # SVM uses only its TF-IDF features
    if model_choice in ["SVM", "Both"]:
        tfidf_svm = svm_vectorizer.transform([clean])
        pred_svm = svm_model.predict(tfidf_svm)[0]
        results["SVM"] = "🚨 SPAM" if pred_svm == 1 else "✅ SAFE"

    return results

# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="Spam URL Detector", page_icon="🔒", layout="centered")
st.markdown("<h1 style='text-align:center; color:#2C3E50;'>🔍 Spam URL Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:16px;'>Check if a URL is safe or potentially malicious using RF and SVM models.</p>", unsafe_allow_html=True)

url_input = st.text_input("Enter a URL to analyze:")

model_choice = st.selectbox("Choose Model:", ["Both", "RF", "SVM"])

if st.button("🔍 Analyze"):
    if not url_input.strip():
        st.warning("Please enter a valid URL.")
    else:
        with st.spinner("Analyzing... Please wait..."):
            results = predict_url(url_input, model_choice)

        # Display results
        for model, result in results.items():
            if "SPAM" in result:
                st.error(f"{model} Prediction: {result}")
            else:
                st.success(f"{model} Prediction: {result}")

        # Logging
        log_entry = pd.DataFrame({
            "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "URL": [url_input],
            "Model": [", ".join(results.keys())],
            "Prediction": [", ".join(results.values())]
        })
        try:
            logs_df = pd.read_csv("logs.csv")
            log_entry.to_csv("logs.csv", mode='a', header=False, index=False)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            log_entry.to_csv("logs.csv", index=False)

# Display Logs
if st.checkbox("📜 Show Prediction History"):
    try:
        logs = pd.read_csv("logs.csv")
        st.dataframe(logs, use_container_width=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        st.info("No logs yet. Start predicting!")

st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>Spam URL Detection using RF & SVM Models</p>", unsafe_allow_html=True)
