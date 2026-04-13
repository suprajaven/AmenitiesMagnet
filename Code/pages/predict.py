"""
Code/pages/predict.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


_MODEL_DIR = Path(__file__).resolve().parents[2] / "Analysis" / "Modeling"
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

from model_utils import (
    artifacts_ready,
    find_similar,
    get_label_options,
    get_shap_ranking,
    get_state_model_info,
    predict_price,
)
from utils import comparable_listing_estimate, load_data, load_model_data

# ─────────────────────────────────────────────────────────────
st.title("Predictor")
st.caption("Property benchmark · XGBoost · State insight · Similar listings")

raw_df   = load_data()
model_df = load_model_data()

# ── Check artifacts ───────────────────────────────────────────
if not artifacts_ready():
    st.error(
        "Model artifacts not found."
    )
    st.stop()

label_opts = get_label_options()
state_info = get_state_model_info()

# ── Left / Right layout  ───────
left, right = st.columns([1.2, 1])

with left:
    city = st.selectbox("City", sorted(raw_df["city"].dropna().unique()))

    # Derive state from selected city
    city_state_map = raw_df.dropna(subset=["city","state"]).drop_duplicates("city").set_index("city")["state"].to_dict()
    state = city_state_map.get(city, None)

    size    = st.slider("Living space (sqm)", 20, 200, 70)
    rooms   = st.slider("Rooms", 1, 8, 2)
    amenities = st.slider("Amenity score", 0, 6, 3)
    has_kitchen = st.toggle("Has kitchen", value=True)
    lift        = st.toggle("Lift", value=False)

    # Advanced inputs in expander 
    with st.expander("More details (optional — improves XGBoost accuracy)"):
        building_age    = st.slider("Building age (years)", 0, 150, 40)
        service_charge  = st.number_input("Service charge (€/month)", 0, 1000, 120, step=10)
        condition_score = st.slider("Condition score (0–9)", 0, 9, 5)
        interior_score  = st.slider("Interior score (1–4)", 1, 4, 2)
        condition     = st.selectbox("Condition",        label_opts.get("condition",    ["unknown"]))
        interior_qual = st.selectbox("Interior quality", label_opts.get("interiorQual", ["normal"]))
        type_of_flat  = st.selectbox("Type of flat",     label_opts.get("typeOfFlat",   ["apartment"]))
        heating_type  = st.selectbox("Heating type",     label_opts.get("heatingType",  ["central_heating"]))
        floor         = st.selectbox("Floor",            label_opts.get("floor",        ["2"]))

# Build input dict for XGBoost
input_dict = {
    "serviceCharge":   service_charge  if "service_charge"  in dir() else 120,
    "noRooms":         rooms,
    "building_age":    building_age    if "building_age"    in dir() else 40,
    "amenity_score":   amenities,
    "condition_score": condition_score if "condition_score" in dir() else 5,
    "interior_score":  interior_score  if "interior_score"  in dir() else 2,
    "hasKitchen":      int(has_kitchen),
    "lift":            int(lift),
    "condition":       condition       if "condition"       in dir() else "unknown",
    "interiorQual":    interior_qual   if "interior_qual"   in dir() else "normal",
    "typeOfFlat":      type_of_flat    if "type_of_flat"    in dir() else "apartment",
    "heatingType":     heating_type    if "heating_type"    in dir() else "central_heating",
    "floor":           str(floor)      if "floor"           in dir() else "2",
    "city":            city,
    "state":           state or "unknown",
}

# ── Predictions ───────────────────────────────────────────────
xgb_result = predict_price(input_dict, state=state)

# Original comparable-listing estimate (kept for reference)
estimate, comparable_count = comparable_listing_estimate(
    raw_df,
    city=city,
    size=size,
    rooms=rooms,
    amenity_score=amenities,
    has_kitchen=int(has_kitchen),
    lift=int(lift),
)

city_baseline      = raw_df.loc[raw_df["city"] == city, "baseRent"].median()
city_price_per_sqm = raw_df.loc[raw_df["city"] == city, "price_per_sqm"].median()

price_global = xgb_result["price_per_sqm_global"]
price_state  = xgb_result["price_per_sqm_state"]
state_r2     = xgb_result["state_r2"]

with right:
    # ── XGBoost prediction (primary) ──────────────────────────
    if price_global:
        st.metric(
            "XGBoost prediction (€/sqm)",
            f"€ {price_global:.2f}",
            help="Global XGBoost model trained on all 268,568 listings.",
        )
        if price_state:
            delta = price_state - price_global
            st.metric(
                f"State model — {state}",
                f"€ {price_state:.2f}",
                delta=f"{delta:+.2f} vs global  |  R²={state_r2:.3f}",
                help=f"XGBoost trained on {state} listings only.",
            )

    st.divider()

    # ── Original comparable-listing estimate (kept) ────────────
    if estimate is None:
        st.warning("Not enough comparable listings found for that profile.")
    else:
        st.metric("Comparable-listing estimate", f"EUR {estimate:,.0f}")
        st.metric("Comparable listings used", f"{comparable_count:,}")

    st.metric("City median rent",    f"EUR {city_baseline:,.0f}")
    st.metric("City median EUR/sqm", f"{city_price_per_sqm:.2f}")

st.divider()

# ── SHAP + Distribution in two columns ────────────────────────
shap_col, dist_col = st.columns(2)

with shap_col:
    st.subheader("Feature importance")
    st.caption(f"What drives price/m² in {city}?")

    shap_df = get_shap_ranking(state=state, city=city)

    if not shap_df.empty:
        fig = px.bar(
            shap_df.sort_values("mean_abs_shap"),
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            color="mean_abs_shap",
            color_continuous_scale="Blues",
            labels={"mean_abs_shap": "Mean |SHAP|", "feature": ""},
        )
        fig.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10),
            height=260,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("SHAP data not available for this location.")

with dist_col:
    st.subheader("Price distribution")
    st.caption(f"All listings in {city}")

    city_data = raw_df.loc[
        raw_df["city"] == city,
        "price_per_sqm",
    ].dropna()
    city_data = city_data[city_data.between(1, 50)]

    if len(city_data) > 10:
        fig2 = px.histogram(
            city_data,
            nbins=35,
            color_discrete_sequence=["#003057"],
            labels={"value": "€/m²"},
        )
        if price_global:
            fig2.add_vline(
                x=price_global,
                line_dash="dash",
                line_color="#EAB900",
                annotation_text=f"XGBoost €{price_global:.1f}",
                annotation_position="top right",
            )
        fig2.add_vline(
            x=city_price_per_sqm,
            line_dash="dot",
            line_color="#377117",
            annotation_text=f"Median €{city_price_per_sqm:.1f}",
            annotation_position="top left",
        )
        fig2.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=260,
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Not enough listings for distribution chart.")

st.divider()

# ── Similar listings ──────────────────────────────────────────
st.subheader("Similar listings")

tab_text, tab_struct = st.tabs(["🔍 Search by description", "📊 Search by features"])

with tab_text:
    query_text = st.text_area(
        "Describe what you are looking for",
        placeholder="e.g. renovated flat near city centre with balcony and modern kitchen",
        height=70,
    )
    same_city = st.checkbox("Limit to same city", value=True)
    if st.button("Find similar", key="txt"):
        if query_text.strip():
            with st.spinner("Searching …"):
                sim = find_similar(query_text, city=city if same_city else None, k=8)
            if not sim.empty:
                cols = [c for c in ["city","state","price_per_sqm","noRooms","amenity_score","similarity_score"] if c in sim.columns]
                st.dataframe(sim[cols], use_container_width=True, hide_index=True)
            else:
                st.info("No similar listings found. Try a broader description.")
        else:
            st.warning("Enter a description first.")

with tab_struct:
    st.caption("Uses your current inputs to find structurally similar listings.")
    if st.button("Find similar", key="struct"):
        with st.spinner("Searching …"):
            sim2 = find_similar(input_dict, city=city, k=8)
        if not sim2.empty:
            cols = [c for c in ["city","state","price_per_sqm","noRooms","amenity_score","similarity_score"] if c in sim2.columns]
            st.dataframe(sim2[cols], use_container_width=True, hide_index=True)
        else:
            st.info("No similar listings found.")

st.divider()

# ── Bottom metadata row — same as original ────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Modeling rows",    f"{len(model_df):,}")
col2.metric("Feature columns",  f"{model_df.shape[1] - 1}")
col3.metric("Target", "log_price_per_sqm → exp() → €/m²")

# ── State model details (collapsed) ───────────────────────────
with st.expander("State model details"):
    if state_info:
        info_df = (
            pd.DataFrame(state_info).T
            .reset_index()
            .rename(columns={"index": "State", "r2": "R²", "rmse": "RMSE", "n_train": "N train"})
            .sort_values("R²", ascending=False)
        )
        st.dataframe(info_df[["State","R²","RMSE","N train"]],
                     use_container_width=True, hide_index=True)
    else:
        st.info("No state models loaded.")
