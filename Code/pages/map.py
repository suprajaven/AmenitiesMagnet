import streamlit as st
import plotly.express as px
from utils import load_data

st.markdown("""
<style>
    .stMetric { text-align: center; }
</style>
""", unsafe_allow_html=True)

df = load_data()

st.title("Map View")

if "lat" in df.columns:
    fig = px.scatter_mapbox(
        df.sample(2000),
        lat="lat",
        lon="lon",
        color="price_per_sqm",
        zoom=5,
        height=600
    )
    fig.update_layout(mapbox_style="open-street-map")

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No location data available — add lat/lon to enable map view.")
