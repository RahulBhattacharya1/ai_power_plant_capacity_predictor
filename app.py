import os, joblib
import streamlit as st
import pandas as pd

st.set_page_config(page_title="EU Power Plant Capacity Predictor", page_icon="⚡")

MODEL_PATH = "models/capacity_pipeline.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)  # this is a scikit-learn Pipeline

pipe = load_model()

st.title("EU Power Plant Capacity Predictor")

col1, col2 = st.columns(2)
with col1:
    energy_source = st.text_input("Energy source", "Natural gas")
    technology    = st.text_input("Technology", "Combined cycle")
with col2:
    commissioned  = st.number_input("Commissioning year", min_value=1900, max_value=2030, value=2000, step=1)
    country       = st.text_input("Country", "DE")

if st.button("Predict capacity (MW)"):
    X_new = pd.DataFrame([{
        "energy_source": energy_source,
        "technology": technology,
        "commissioned": int(commissioned),
        "country": country
    }])
    pred = float(pipe.predict(X_new)[0])
    st.success(f"Predicted capacity: {pred:.2f} MW")
