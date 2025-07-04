import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load the model and scalers
model = load_model('emission_model.keras')
scaler_data = joblib.load('scalers.pkl')
feature_scaler = scaler_data['feature_scaler']
target_scaler = scaler_data['target_scaler']
ohe = scaler_data['one_hot_encoder']
features = scaler_data['feature_names']
targets = scaler_data['target_names']

# Function to make predictions
def recursive_forecast(model, feature_scaler, target_scaler, ohe, df, area, start_year, predict_until_year, time_steps, feature_names, target_names, n_output_steps=3):
    # Prepare initial input sequence: last 'time_steps' years for the area
    df_area = df[df['Area'] == area].sort_values('Year')

    df_area['Year'] = pd.to_numeric(df_area['Year'], errors='coerce')
    seq_df = df_area[df_area['Year'] <= start_year].tail(time_steps)
    if len(seq_df) < time_steps:
        return None, "Not enough historical data for recursive forecasting."
    
    # Build initial input features for the sequence
    X_seq = []
    for _, row in seq_df.iterrows():
        row_feats = []
        for f in feature_names:
            if f == 'area':
                continue
            row_feats.append(row[f] if f in row else 0)
        area_ohe = ohe.transform([[row['Area']]])
        row_feats.extend(area_ohe[0])
        X_seq.append(row_feats)

    X_seq = np.array(X_seq)
    print("Shape of X_seq before scaling:", X_seq.shape)  # Debugging line
    X_seq_scaled = feature_scaler.transform(X_seq)

    current_seq = X_seq_scaled.copy()
    predictions = []
    current_year = start_year
    
    while current_year < predict_until_year:
        input_seq = np.expand_dims(current_seq, axis=0)  # shape (1, time_steps, n_features)
        y_pred_scaled = model.predict(input_seq)  # shape (1, n_output_steps, n_targets)
        y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, len(target_names))).reshape(-1, len(target_names))
        
        for pred in y_pred:
            current_year += 1
            predictions.append((current_year, pred))
            if current_year >= predict_until_year:
                break
        
        # Prepare next input sequence by sliding window forward by n_output_steps
        last_known_features = current_seq[-1].copy()
        new_rows = []
        for _ in range(n_output_steps):
            new_row = last_known_features.copy()
            new_rows.append(new_row)
        new_rows = np.array(new_rows)
        current_seq = np.vstack([current_seq[n_output_steps:], new_rows])
    
    return predictions, None

# Load your dataset
df = pd.read_csv('data/Agrofood_co2_emission.csv')

# Streamlit app layout
st.title("Global Agrifood CO2 Emissions Forecasting")
st.write("Select a country and a future year to forecast CO2 emissions.")

# User input for country selection
unique_areas = df['Area'].unique()
selected_area = st.selectbox("Select Country/Area", unique_areas)

# User input for year selection
current_year = df['Year'].max()
future_year = st.number_input("Select Future Year", min_value=current_year + 1, max_value=current_year + 100, value=current_year + 1)

# Button to trigger prediction
if st.button("Forecast"):
    predictions, error = recursive_forecast(model, feature_scaler, target_scaler, ohe, df, selected_area, current_year, future_year, 5, features, targets)
    
    if error:
        st.error(error)
    else:
        st.success("Forecasting completed!")
        
        # Convert predictions into a DataFrame
        df_results = pd.DataFrame(predictions, columns=['Year', 'Predicted Values'])

        # Convert predicted values to numpy array for inspection
        predicted_values_array = np.array(df_results['Predicted Values'].tolist())
        print("Shape of predicted values array:", predicted_values_array.shape)
        
        try:
            # If model outputs 3 values but we only want Total Emission and Temperature
            if predicted_values_array.shape[1] == 3:
                df_results[['Total Emission', 'Average Temperature (°C)']] = predicted_values_array[:, [0, 2]]  # Select first and third columns
            # Else handle other cases as needed
            else:
                st.error(f"Unexpected prediction shape: {predicted_values_array.shape}")
                st.stop()
        except Exception as e:
            st.error(f"Error processing predictions: {str(e)}")
            st.stop()
            
        # Drop the original 'Predicted Values' column
        df_results.drop(columns='Predicted Values', inplace=True)
        # ===== END OF NEW CODE =====

        # Round the values for better display
        df_results = df_results.round(2)

        # Display as a table
        st.subheader("📋 Forecast Results")
        st.dataframe(df_results, use_container_width=True)
