import plotly.express as px
import streamlit as st

from utils import load_data


st.title("Explore Listings")
st.caption("Raw market view built from the original cleaned dataset")

df = load_data()

state_options = ["All"] + sorted(df["state"].dropna().unique().tolist())
selected_state = st.sidebar.selectbox("State", state_options)

city_df = df if selected_state == "All" else df[df["state"] == selected_state]
city_options = ["All"] + sorted(city_df["city"].dropna().unique().tolist())
selected_city = st.sidebar.selectbox("City", city_options)

filtered = city_df.copy()
if selected_city != "All":
    filtered = filtered[filtered["city"] == selected_city]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Listings", f"{len(filtered):,}")
c2.metric("Median rent", f"EUR {filtered['baseRent'].median():,.0f}")
c3.metric("Median EUR/sqm", f"{filtered['price_per_sqm'].median():.2f}")
c4.metric("Avg amenity score", f"{filtered['amenity_score'].mean():.2f}")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    scatter = px.scatter(
        filtered.sample(min(3000, len(filtered)), random_state=42),
        x="livingSpace",
        y="baseRent",
        color="price_per_sqm",
        hover_data=["city", "state", "amenity_score", "condition_score", "interior_score"],
        title="Rent vs living space",
        color_continuous_scale="YlOrRd",
    )
    scatter.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(scatter, use_container_width=True)

with chart_col2:
    city_summary = (
        filtered.groupby("city", as_index=False)
        .agg(median_price_per_sqm=("price_per_sqm", "median"), listings=("city", "size"))
        .sort_values("listings", ascending=False)
        .head(15)
    )
    city_bar = px.bar(
        city_summary.sort_values("median_price_per_sqm"),
        x="median_price_per_sqm",
        y="city",
        color="listings",
        orientation="h",
        title="Top cities in current slice",
        color_continuous_scale="Blues",
    )
    city_bar.update_layout(margin=dict(l=20, r=20, t=50, b=20), yaxis_title="")
    st.plotly_chart(city_bar, use_container_width=True)

st.subheader("Sample listings")
display_cols = [
    "city",
    "state",
    "baseRent",
    "livingSpace",
    "noRooms",
    "price_per_sqm",
    "amenity_score",
    "condition_score",
    "interior_score",
]
st.dataframe(
    filtered[display_cols].sort_values("price_per_sqm", ascending=False).head(25),
    use_container_width=True,
    hide_index=True,
)
