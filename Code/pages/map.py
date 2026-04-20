import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import load_data, prepare_state_market_view


def _build_pillar_map(frame, metric):
    chart_frame = frame.copy()
    metric_min = chart_frame[metric].min()
    metric_range = chart_frame[metric].max() - metric_min
    if metric_range == 0:
        metric_range = 1

    chart_frame["pillar_height"] = 0.12 + 0.48 * (
        (chart_frame[metric] - metric_min) / metric_range
    )
    chart_frame["pillar_top_lat"] = chart_frame["lat"] + chart_frame["pillar_height"]
    chart_frame["top_size"] = 6 + 10 * (
        chart_frame["listings"] / chart_frame["listings"].max()
    ) ** 0.55
    chart_frame["base_size"] = chart_frame["top_size"] + 2

    pillar_lons = []
    pillar_lats = []
    for row in chart_frame.itertuples():
        pillar_lons.extend([row.lon, row.lon, None])
        pillar_lats.extend([row.lat, row.pillar_top_lat, None])

    fig = go.Figure()
    fig.add_trace(
        go.Scattergeo(
            lon=pillar_lons,
            lat=pillar_lats,
            mode="lines",
            line=dict(color="rgba(20, 53, 87, 0.5)", width=2),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scattergeo(
            lon=chart_frame["lon"],
            lat=chart_frame["lat"],
            mode="markers",
            marker=dict(
                size=chart_frame["base_size"],
                color="rgba(20, 53, 87, 0.16)",
                line=dict(color="rgba(20, 53, 87, 0.10)", width=0.5),
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scattergeo(
            lon=chart_frame["lon"],
            lat=chart_frame["pillar_top_lat"],
            mode="markers",
            marker=dict(
                size=chart_frame["top_size"],
                color=chart_frame[metric],
                colorscale=[
                    [0.0, "#ffd166"],
                    [0.45, "#f4a261"],
                    [1.0, "#d1495b"],
                ],
                cmin=chart_frame[metric].min(),
                cmax=chart_frame[metric].max(),
                line=dict(color="white", width=0.8),
                colorbar=dict(title="Selected metric", x=0.98),
                opacity=0.95,
            ),
            customdata=chart_frame[
                [
                    "state",
                    "listings",
                    "avg_price_per_sqm",
                    "median_price_per_sqm",
                    "avg_amenity_score",
                ]
            ],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Listings: %{customdata[1]:,.0f}<br>"
                "Avg EUR/sqm: %{customdata[2]:.2f}<br>"
                "Median EUR/sqm: %{customdata[3]:.2f}<br>"
                "Avg amenity score: %{customdata[4]:.2f}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scattergeo(
            lon=chart_frame["lon"],
            lat=chart_frame["lat"],
            mode="markers",
            marker=dict(size=3, color="#143557", line=dict(color="white", width=0.6)),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_geos(
        scope="europe",
        projection_type="mercator",
        center={"lat": 51.0, "lon": 10.2},
        lataxis_range=[47.0, 55.8],
        lonaxis_range=[4.8, 16.8],
        showcountries=True,
        countrycolor="rgba(20, 53, 87, 0.18)",
        showsubunits=True,
        subunitcolor="rgba(255, 255, 255, 0.95)",
        showland=True,
        landcolor="#f8f1e7",
        showocean=True,
        oceancolor="#e3edf6",
        showlakes=True,
        lakecolor="#e3edf6",
        coastlinecolor="rgba(20, 53, 87, 0.12)",
        bgcolor="white",
    )
    fig.update_layout(
        height=620,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig, chart_frame


st.title("Map View")
st.caption("State market map")

st.info(
    "Deeper red means a higher value for the selected metric. Slightly taller spikes show stronger state-level intensity."
)

df = load_data()
state_view = prepare_state_market_view(df).dropna(subset=["lat", "lon"]).copy()

metric_choice = st.selectbox(
    "Map metric",
    ["avg_price_per_sqm", "median_price_per_sqm", "avg_amenity_score"],
    format_func=lambda value: {
        "avg_price_per_sqm": "Average EUR/sqm",
        "median_price_per_sqm": "Median EUR/sqm",
        "avg_amenity_score": "Average amenity score",
    }[value],
)

map_col, rank_col = st.columns([1.9, 1])
fig, chart_frame = _build_pillar_map(state_view, metric_choice)

with map_col:
    st.caption("Legend: light yellow = lower values, orange = mid-range values, deep red = higher values.")
    st.plotly_chart(fig, use_container_width=True)

with rank_col:
    st.caption("Ranking panel for the selected metric.")
    top_states = chart_frame.sort_values(metric_choice, ascending=False).copy()
    ranking = px.bar(
        top_states.head(8).sort_values(metric_choice),
        x=metric_choice,
        y="state",
        orientation="h",
        color=metric_choice,
        color_continuous_scale=[
            [0.0, "#ffd166"],
            [0.45, "#f4a261"],
            [1.0, "#d1495b"],
        ],
        labels={metric_choice: "Selected metric", "state": ""},
        title="Top states",
    )
    ranking.update_layout(
        height=700,
        margin=dict(l=10, r=10, t=55, b=10),
        coloraxis_showscale=False,
    )
    st.plotly_chart(ranking, use_container_width=True)

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Highest value", f"{chart_frame[metric_choice].max():.2f}")
metric_col2.metric("Median value", f"{chart_frame[metric_choice].median():.2f}")
metric_col3.metric("States shown", f"{len(chart_frame):,}")

table = chart_frame.sort_values("avg_price_per_sqm", ascending=False).rename(
    columns={
        "avg_price_per_sqm": "Avg EUR/sqm",
        "median_price_per_sqm": "Median EUR/sqm",
        "avg_amenity_score": "Avg amenity score",
        "listings": "Listings",
    }
)

st.subheader("State summary")
st.caption("Table view: exact state-level values behind the map for quick comparison.")
st.dataframe(
    table[["state", "Listings", "Avg EUR/sqm", "Median EUR/sqm", "Avg amenity score"]],
    use_container_width=True,
    hide_index=True,
)
