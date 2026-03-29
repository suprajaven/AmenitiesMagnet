import streamlit as st

from utils import MODEL_CATEGORICAL_FEATURES, MODEL_NUMERIC_FEATURES


st.set_page_config(page_title="Amenities Magnet", layout="wide")

st.title("Amenities Magnet")
st.caption("Rental price intelligence for Germany's housing markets")

hero_left, hero_right = st.columns([1.4, 1])

with hero_left:
    st.subheader("Germany rental market overview")
    st.markdown(
        """
        Compare regional rental patterns, inspect the strongest market drivers,
        and benchmark property profiles against similar listings.
        """
    )

with hero_right:
    st.metric("Listings", "268,632")
    st.metric("Modeling rows", "268,568")
    st.metric("Markets", "419 cities")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Numeric features")
    st.selectbox("Feature", MODEL_NUMERIC_FEATURES, key="numeric_feature_home")

with col2:
    st.subheader("Categorical features")
    st.selectbox("Feature ", MODEL_CATEGORICAL_FEATURES, key="categorical_feature_home")
