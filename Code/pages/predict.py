import streamlit as st
from utils import load_data

st.markdown("""
<style>
    .stMetric { text-align: center; }
</style>
""", unsafe_allow_html=True)

df = load_data()

st.title("Rent Predictor")

size = st.slider("Living Space (sqm)", 20, 200, 70)
rooms = st.slider("Rooms", 1, 6, 2)
amenities = st.slider("Amenities Score", 0, 6, 3)

# Simple estimate
base_price = df["price_per_sqm"].mean()

pred_price = base_price * size * (1 + amenities * 0.05)

st.metric("Estimated Rent", f"€{int(pred_price)}")

st.info("Replace with ML model for final version.")
