import streamlit as st
import keras
import joblib
import numpy as np

def load_model_and_scalers():
    model = keras.models.load_model('agrifood_co2_emissions_model_v2.keras')
    feature_scaler = joblib.load('feature_scaler.pkl')
    target_scaler = joblib.load('target_scaler.pkl')
    return model, feature_scaler, target_scaler

model, feature_scaler, target_scaler = load_model_and_scalers()

# Example prediction function
def make_prediction(input_data):
    # Scale the input features
    scaled_features = feature_scaler.transform(input_data)
    
    # Make prediction
    scaled_prediction = model.predict(scaled_features)
    
    # Inverse transform to get actual values
    actual_prediction = target_scaler.inverse_transform(scaled_prediction)
    
    return actual_prediction

# Streamlit app
st.title("AgriFood CO2 Emissions Prediction")

# Add your input widgets here
#For example:
input_value = st.number_input("Enter feature value:")
if st.button("Predict"):
    prediction = make_prediction([[input_value]])
    st.write(f"Predicted CO2 Emissions: {prediction[0][0]}")
