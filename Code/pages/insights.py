import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import MODEL_NUMERIC_FEATURES, load_model_data, make_quantile_bins


st.title("Modeling Insights")
st.caption("Transformed modelling dataset and feature-level explanation patterns")

model_df = load_model_data()

top_cities = (
    model_df["city"].value_counts().head(12).index.tolist()
)

selected_city = st.selectbox("City focus", ["All cities"] + top_cities)
selected_feature = st.selectbox("Numeric feature heatmap", MODEL_NUMERIC_FEATURES)

filtered = model_df if selected_city == "All cities" else model_df[model_df["city"] == selected_city]

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Rows in current slice", f"{len(filtered):,}")
metric_col2.metric("Avg log EUR/sqm", f"{filtered['log_price_per_sqm'].mean():.3f}")
metric_col3.metric("Median amenity score", f"{filtered['amenity_score'].median():.0f}")

st.caption(
    "These charts show how the model features behave in the transformed training data."
)

feature_band = make_quantile_bins(
    filtered[selected_feature],
    ["Low", "Lower-mid", "Upper-mid", "High"],
)

heatmap_frame = (
    filtered.assign(feature_band=feature_band)
    .groupby(["feature_band", "interior_score"], as_index=False)["log_price_per_sqm"]
    .mean()
)

heatmap = px.density_heatmap(
    heatmap_frame,
    x="interior_score",
    y="feature_band",
    z="log_price_per_sqm",
    histfunc="avg",
    color_continuous_scale="YlOrRd",
    title=f"Average log price per sqm by {selected_feature} band and interior score",
)
heatmap.update_layout(margin=dict(l=20, r=20, t=60, b=20))
st.caption(
    "Darker cells indicate higher average log price per sqm for that combination."
)
st.plotly_chart(heatmap, use_container_width=True)

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.caption(
        "City ranking by average log price per sqm."
    )
    city_rank = (
        model_df[model_df["city"].isin(top_cities)]
        .groupby("city", as_index=False)
        .agg(avg_log_price=("log_price_per_sqm", "mean"))
        .sort_values("avg_log_price", ascending=True)
    )
    rank_chart = px.bar(
        city_rank,
        x="avg_log_price",
        y="city",
        orientation="h",
        title="Average log price per sqm across top cities",
        color="avg_log_price",
        color_continuous_scale="Tealgrn",
    )
    rank_chart.update_layout(margin=dict(l=20, r=20, t=50, b=20), yaxis_title="")
    st.plotly_chart(rank_chart, use_container_width=True)

with chart_col2:
    st.caption(
        "Correlation view of the retained numeric features."
    )
    st.info("Legend: red = positive correlation, blue = negative correlation, pale colors = weak correlation.")
    corr = filtered[MODEL_NUMERIC_FEATURES + ["log_price_per_sqm"]].corr(numeric_only=True)
    corr_chart = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmid=0,
            text=corr.round(2).values,
            texttemplate="%{text}",
        )
    )
    corr_chart.update_layout(
        title="Correlation view of retained numeric features",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(corr_chart, use_container_width=True)
