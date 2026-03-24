import streamlit as st
import pandas as pd
from utils import load_data

st.markdown("""
<style>
    .stMetric { text-align: center; }
</style>
""", unsafe_allow_html=True)

df = load_data()

st.title("Explore Listings")

# Sidebar filters
city = st.sidebar.selectbox("City", sorted(df["city"].unique()))

budget = st.sidebar.radio(
    "Budget",
    ["All", "Budget", "Mid", "Premium"]
)

filtered = df[df["city"] == city]

if budget == "Budget":
    filtered = filtered[filtered["price_per_sqm"] < 10]
elif budget == "Mid":
    filtered = filtered[(filtered["price_per_sqm"] >= 10) & (filtered["price_per_sqm"] < 20)]
elif budget == "Premium":
    filtered = filtered[filtered["price_per_sqm"] >= 20]

# KPIs
c1, c2, c3, c4 = st.columns(4)

c1.metric("Avg €/sqm", round(filtered["price_per_sqm"].mean(), 2))
c2.metric("Median Rent", int(filtered["baseRent"].median()))
c3.metric("Avg Size", int(filtered["livingSpace"].mean()))
c4.metric("Listings", len(filtered))

st.divider()

# Listings cards
for _, row in filtered.head(15).iterrows():
    col1, col2 = st.columns([1, 3])

    col1.image("https://via.placeholder.com/150", use_container_width=True)

    col2.markdown(f"""
    ### €{int(row['baseRent'])}
    {row['livingSpace']} sqm · {row['noRooms']} rooms  
    **€{row['price_per_sqm']}/sqm**

    Amenities: {row['amenity_score']}
    """)

    st.divider()
