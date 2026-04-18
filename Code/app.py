import streamlit as st


st.set_page_config(page_title="Amenities Magnet", layout="wide")

st.title("Amenities Magnet")
st.caption("Rental price intelligence for Germany")

metric_cols = st.columns(3)
metric_cols[0].metric("Listings", "268,632")
metric_cols[1].metric("Cities", "419")
metric_cols[2].metric("Markets", "16 states")

nav1, nav2, nav3 = st.columns(3)
nav1.info("Map View")
nav2.info("Explore Listings")
nav3.info("Rental Price Estimator")

if hasattr(st, "page_link"):
    link1, link2, link3, link4 = st.columns(4)
    link1.page_link("pages/map.py", label="Open Map")
    link2.page_link("pages/explore.py", label="Explore Data")
    link3.page_link("pages/predict.py", label="Open Predictor")
    link4.page_link("pages/insights.py", label="Model Insights")

with st.expander("About"):
    st.write(
        "Explore rental patterns, estimate apartment prices, and compare similar listings across Germany."
    )
