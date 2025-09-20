import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/capacity_model.pkl")

st.title("EU Power Plant Capacity Predictor")

energy = st.selectbox("Energy Source", ["Natural gas", "Hard coal", "Oil", "Renewable"])
tech = st.text_input("Technology (e.g., Steam turbine, Unknown)")
year = st.number_input("Commissioning Year", min_value=1900, max_value=2025, value=2000)
country = st.text_input("Country (e.g., NL, DE, FR)")

if st.button("Predict Capacity"):
    # Encode inputs (for demo, using simple mapping)
    df = pd.DataFrame([[energy, tech, year, country]], columns=['energy_source','technology','commissioned','country'])
    # For real use, ensure encoding matches training
    pred = model.predict(df)
    st.success(f"Predicted Capacity: {pred[0]:.2f} MW")
