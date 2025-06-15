import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("model.keras")
scaler = joblib.load("scaler.pkl")

st.title("🌱 Agri-Food CO₂ Emissions Predictor")
st.write("Upload a CSV file with input features to predict emissions.")

# File uploader
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    st.write("### Preview of uploaded data:")
    st.dataframe(input_df.head())

    # Preprocess
    input_scaled = scaler.transform(input_df)

    # Predict
    predictions = model.predict(input_scaled)

    # Display predictions
    st.write("### Predicted CO₂ Emissions:")
    input_df["Predicted Emissions"] = predictions
    st.dataframe(input_df)

    # Optional: download
    csv = input_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download predictions as CSV", csv, "predictions.csv", "text/csv")
