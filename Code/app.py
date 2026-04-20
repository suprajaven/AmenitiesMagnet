import streamlit as st


st.set_page_config(page_title="Amenities Magnet", layout="wide")

st.title("Amenities Magnet")
st.caption("Rental price intelligence for Germany")

st.markdown(
    """
    **About**

    Amenities Magnet predicts residential rental prices across German cities and explains
    why a property costs what it does. Built on ImmoScout24 listings data, it combines
    market exploration, explainable machine learning, and apartment matching in one dashboard.
    """
)

about_metrics = st.columns(4)
about_metrics[0].metric("Listings", "268,632")
about_metrics[1].metric("Cities", "419")
about_metrics[2].metric("States", "16")
about_metrics[3].metric("User groups", "4")
st.caption(
    "Designed for tenants, investors, developers, and policymakers who want a clearer view of German rental markets."
)

if hasattr(st, "page_link"):
    top_row = st.columns(2)
    bottom_row = st.columns(2)
    top_row[0].page_link("pages/map.py", label="Map View")
    top_row[1].page_link("pages/explore.py", label="Explore Listings")
    bottom_row[0].page_link("pages/predict.py", label="Price Estimator")
    bottom_row[1].page_link("pages/insights.py", label="Model Insights")
