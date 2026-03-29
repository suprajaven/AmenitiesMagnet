import plotly.express as px
import streamlit as st

from utils import load_data, prepare_state_market_view


st.title("Map View-tobeupd")
st.caption("State-level market map for Germany using aggregated listing metrics")

df = load_data()
state_view = prepare_state_market_view(df)

metric_choice = st.selectbox(
    "Map color",
    ["avg_price_per_sqm", "median_price_per_sqm", "avg_amenity_score"],
    format_func=lambda value: {
        "avg_price_per_sqm": "Average EUR/sqm",
        "median_price_per_sqm": "Median EUR/sqm",
        "avg_amenity_score": "Average amenity score",
    }[value],
)

geo = px.scatter_geo(
    state_view,
    lat="lat",
    lon="lon",
    size="listings",
    color=metric_choice,
    hover_name="state",
    hover_data={
        "listings": ":,",
        "avg_price_per_sqm": ":.2f",
        "median_price_per_sqm": ":.2f",
        "avg_amenity_score": ":.2f",
        "lat": False,
        "lon": False,
    },
    projection="mercator",
    scope="europe",
    color_continuous_scale="YlOrRd",
    title="Regional rental intensity across Germany",
)
geo.update_geos(
    center={"lat": 51.0, "lon": 10.2},
    lataxis_range=[47.0, 55.5],
    lonaxis_range=[5.0, 16.5],
    showcountries=True,
    countrycolor="LightGray",
    showsubunits=True,
    subunitcolor="White",
)
geo.update_layout(height=650, margin=dict(l=20, r=20, t=60, b=20))
st.plotly_chart(geo, use_container_width=True)

table = state_view.sort_values("avg_price_per_sqm", ascending=False).rename(
    columns={
        "avg_price_per_sqm": "Avg EUR/sqm",
        "median_price_per_sqm": "Median EUR/sqm",
        "avg_amenity_score": "Avg amenity score",
        "listings": "Listings",
    }
)

st.subheader("State summary")
st.dataframe(
    table[["state", "Listings", "Avg EUR/sqm", "Median EUR/sqm", "Avg amenity score"]],
    use_container_width=True,
    hide_index=True,
)
