import streamlit as st
import plotly.express as px
from CSE6242_Project.Code.utils import load_data

st.markdown("""
<style>
    .stMetric { text-align: center; }
</style>
""", unsafe_allow_html=True)

df = load_data()

st.title("Price Insights")

city = st.selectbox("Select City", sorted(df["city"].unique()))
filtered = df[df["city"] == city]

# Distribution
fig1 = px.histogram(filtered, x="price_per_sqm", nbins=40)
st.plotly_chart(fig1, width=True)

# Amenity impact
impact = filtered.groupby("amenity_score")["price_per_sqm"].mean().reset_index()

fig2 = px.bar(impact, x="amenity_score", y="price_per_sqm")
st.plotly_chart(fig2, width=True)

# Smart insight
high = filtered[filtered["amenity_score"] >= 4]["price_per_sqm"].mean()
low = filtered[filtered["amenity_score"] <= 2]["price_per_sqm"].mean()

st.success(f"""
Apartments with high amenities cost **{round(high,2)} €/sqm**  
vs low amenities at **{round(low,2)} €/sqm**
""")
