import streamlit as st

st.set_page_config(
    page_title="Amenities Magnet",
    layout="wide",
    page_icon=None
)

st.markdown("""
<style>
    .stMetric { text-align: center; }
</style>
""", unsafe_allow_html=True)

st.title("Amenities Magnet")
st.markdown("""
### Discover rental prices like a pro

- Explore listings  
- Visualize locations  
- Understand price drivers  
- Predict rent  
""")

st.info("Use the sidebar to navigate between pages.")
