import streamlit as st
st.cache_resource.clear()   # <--- ADD THIS LINE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib



st.set_page_config(page_title="AAPL Forecast Dashboard", layout="wide")

st.title("📈 Apple Stock Forecast Dashboard")

# -------------------------------
# Load Models
# -------------------------------
@st.cache_resource
def load_all_models():
    # Correct filenames exactly as they appear in your repository
    rf = joblib.load("rf.joblib")  
    xgb = joblib.load("xg.joblib")  
    scaler = joblib.load("scaler.joblib")
    lstm = None  # Since you removed LSTM
    results = pd.read_csv("model_results.csv")

    return rf, xgb, scaler, lstm, results


rf, best_xgb, scaler_xgb, lstm_model, results_df = load_all_models()



# -------------------------------
# Upload CSV
# -------------------------------
st.sidebar.header("📂 Upload File")
uploaded = st.sidebar.file_uploader("Upload AAPL.csv", type=['csv'])

if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv("AAPL.csv")

df['Date'] = pd.to_datetime(df['Date'])

st.subheader("📌 Close Price Trend")
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(df['Date'], df['Close'])
ax.set_title("AAPL Closing Price")
st.pyplot(fig)


# -------------------------------
# Model Accuracy Table
# -------------------------------
st.subheader("📊 Model Accuracy Table")
st.dataframe(results_df)


# -------------------------------
# Choose Model
# -------------------------------
model_choice = st.selectbox(
    "Select Model for 30-Day Forecast",
    ["Random Forest", "XGBoost", ]
)

X = df.drop(['Date', 'Close'], axis=1, errors='ignore').iloc[-1:].copy()

if model_choice == "Random Forest":
    pred = rf.predict(X)[0]
    st.success(f"📈 RF Prediction: {pred}")

elif model_choice == "XGBoost":
    scaled = scaler_xgb.transform(X)
    pred = best_xgb.predict(scaled)[0]
    st.success(f"🚀 XGBoost Prediction: {pred}")



# -------------------------------
# Technical Indicators
# -------------------------------
st.subheader("📉 Technical Indicators")

df['MA7'] = df['Close'].rolling(7).mean()
df['MA21'] = df['Close'].rolling(21).mean()
df['Volatility'] = df['Close'].rolling(7).std()

# Moving Averages
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(df['Date'], df['Close'], label="Close")
ax.plot(df['Date'], df['MA7'], label="MA7")
ax.plot(df['Date'], df['MA21'], label="MA21")
ax.legend()
st.pyplot(fig)

# Volatility
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(df['Date'], df['Volatility'])
ax.set_title("Volatility (7-Day STD)")
st.pyplot(fig)
