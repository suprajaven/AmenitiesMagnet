import streamlit as st

from utils import comparable_listing_estimate, load_data, load_model_data


st.title("Predictor")
st.caption("Property benchmark")

raw_df = load_data()
model_df = load_model_data()

left, right = st.columns([1.2, 1])

with left:
    city = st.selectbox("City", sorted(raw_df["city"].dropna().unique()))
    size = st.slider("Living space (sqm)", 20, 200, 70)
    rooms = st.slider("Rooms", 1, 8, 2)
    amenities = st.slider("Amenity score", 0, 6, 3)
    has_kitchen = st.toggle("Has kitchen", value=True)
    lift = st.toggle("Lift", value=False)

with right:
    estimate, comparable_count = comparable_listing_estimate(
        raw_df,
        city=city,
        size=size,
        rooms=rooms,
        amenity_score=amenities,
        has_kitchen=int(has_kitchen),
        lift=int(lift),
    )

    city_baseline = raw_df.loc[raw_df["city"] == city, "baseRent"].median()
    city_price_per_sqm = raw_df.loc[raw_df["city"] == city, "price_per_sqm"].median()

    if estimate is None:
        st.warning("Not enough comparable listings were found for that exact profile.")
    else:
        st.metric("Comparable-listing estimate", f"EUR {estimate:,.0f}")
        st.metric("Comparable listings used", f"{comparable_count:,}")

    st.metric("City median rent", f"EUR {city_baseline:,.0f}")
    st.metric("City median EUR/sqm", f"{city_price_per_sqm:.2f}")

st.divider()

col1, col2, col3 = st.columns(3)
col1.metric("Modeling rows", f"{len(model_df):,}")
col2.metric("Feature columns", f"{model_df.shape[1] - 1}")
col3.metric("Target", "log_price_per_sqm")
