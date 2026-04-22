"""
Code/pages/predict.py
"""

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

_MODEL_DIR = Path(__file__).resolve().parents[2] / "Analysis" / "Modeling"
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

from model_utils import (
    artifacts_ready,
    find_similar_by_description,
    find_similar_by_features,
    get_label_options,
    get_shap_ranking,
    predict_price,
)
from utils import apply_branding, comparable_listing_estimate, load_data


FEATURE_LABELS = {
    "noRooms": "Number of rooms",
    "building_age": "Building age",
    "hasKitchen": "Has kitchen",
    "amenity_score": "Amenities",
    "interior_score": "Interior quality",
    "condition_score": "Property condition",
    "serviceCharge": "Service charge",
    "lift": "Has lift",
}


apply_branding()


st.title("Rental Price Estimator")
st.caption("Estimate rent and compare similar apartments")

raw_df = load_data()

if not artifacts_ready():
    st.error(
        "Price estimation is unavailable in this environment because the model dependency `xgboost` is not installed."
    )
    st.caption("Install project requirements to enable the estimator page.")
    st.stop()

label_opts = get_label_options()
city_state_map = (
    raw_df.dropna(subset=["city", "state"])
    .drop_duplicates("city")
    .set_index("city")["state"]
    .to_dict()
)

left, right = st.columns([1, 1.2])

with left:
    st.subheader("Apartment details")
    city = st.selectbox("City", sorted(raw_df["city"].dropna().unique()))
    state = city_state_map.get(city)

    size = st.slider("Size (sqm)", 20, 200, 70)
    rooms = st.slider("Number of rooms", 1, 8, 2)

    st.markdown("**Amenities**")
    amenity_col1, amenity_col2, amenity_col3 = st.columns(3)
    with amenity_col1:
        has_balcony = st.checkbox("Balcony", value=True)
        has_cellar = st.checkbox("Cellar", value=True)
    with amenity_col2:
        has_garden = st.checkbox("Garden", value=False)
        has_kitchen = st.checkbox("Kitchen", value=True)
    with amenity_col3:
        is_new_build = st.checkbox("New build", value=False)
        has_lift = st.checkbox("Lift", value=False)

    amenities = int(
        has_balcony
        + has_cellar
        + has_garden
        + has_kitchen
        + has_lift
        + is_new_build
    )
    st.caption(f"Amenities selected: {amenities}/6")

    with st.expander("Open advanced property details"):
        building_age = st.slider("Building age (years)", 0, 150, 40)
        service_charge = st.number_input(
            "Monthly service charge (EUR)", 0, 1000, 120, step=10
        )
        condition_score = st.slider(
            "Property condition (0 = poor, 9 = brand new)", 0, 9, 5
        )
        interior_score = st.slider("Interior quality (1 = basic, 4 = luxury)", 1, 4, 2)
        condition = st.selectbox(
            "Condition", label_opts.get("condition", ["unknown"])
        )
        interior_qual = st.selectbox(
            "Interior quality", label_opts.get("interiorQual", ["normal"])
        )
        type_of_flat = st.selectbox(
            "Apartment type", label_opts.get("typeOfFlat", ["apartment"])
        )
        heating_type = st.selectbox(
            "Heating type", label_opts.get("heatingType", ["central_heating"])
        )
        floor_sel = st.selectbox("Floor", label_opts.get("floor", ["2"]))

_ba = building_age if "building_age" in dir() else 40
_sc = service_charge if "service_charge" in dir() else 120
_cs = condition_score if "condition_score" in dir() else 5
_is = interior_score if "interior_score" in dir() else 2
_cnd = condition if "condition" in dir() else "unknown"
_iq = interior_qual if "interior_qual" in dir() else "normal"
_tof = type_of_flat if "type_of_flat" in dir() else "apartment"
_ht = heating_type if "heating_type" in dir() else "central_heating"
_fl = str(floor_sel) if "floor_sel" in dir() else "2"

input_dict = {
    "serviceCharge": _sc,
    "noRooms": rooms,
    "building_age": _ba,
    "amenity_score": amenities,
    "condition_score": _cs,
    "interior_score": _is,
    "hasKitchen": int(has_kitchen),
    "lift": int(has_lift),
    "condition": _cnd,
    "interiorQual": _iq,
    "typeOfFlat": _tof,
    "heatingType": _ht,
    "floor": _fl,
    "city": city,
    "state": state or "unknown",
}

result = predict_price(input_dict, state=state)
estimate, comparable_count = comparable_listing_estimate(
    raw_df,
    city=city,
    size=size,
    rooms=rooms,
    amenity_score=amenities,
    has_kitchen=int(has_kitchen),
    lift=int(has_lift),
)
city_mean_rent = raw_df.loc[raw_df["city"] == city, "baseRent"].mean()
city_mean_ppsm = raw_df.loc[raw_df["city"] == city, "price_per_sqm"].mean()
price_global = result["price_per_sqm_global"]
price_state = result["price_per_sqm_state"]
est_monthly = round(price_global * size, 0) if price_global else None

with right:
    st.subheader("Estimate")

    if est_monthly:
        st.metric(
            "Estimated rent",
            f"EUR {est_monthly:,.0f} / month",
            help=f"Based on EUR {price_global:.2f}/sqm x {size} sqm",
        )
    st.metric("Price per sqm", f"EUR {price_global:.2f} / sqm")
    st.metric("City average rent", f"EUR {city_mean_rent:,.0f} / month")
    st.metric("City avg EUR/sqm", f"EUR {city_mean_ppsm:.2f}")

    if price_global and city_mean_ppsm:
        diff_pct = ((price_global - city_mean_ppsm) / city_mean_ppsm) * 100
        badge = (
            "Below city mean"
            if diff_pct < -5
            else "Around city mean"
            if diff_pct <= 5
            else "Above city mean"
        )
        st.success(f"{badge} for {city}")

    if estimate:
        st.caption(f"Comparable evidence: {comparable_count:,} similar listings in {city}")

    if price_state and state:
        diff = price_state - price_global
        direction = (
            "slightly higher"
            if diff > 0.05
            else "slightly lower"
            if diff < -0.05
            else "similar"
        )
        st.info(
            f"{state} signal: local data suggests prices here are {direction} than the national estimate "
            f"(EUR {price_state:.2f}/sqm vs EUR {price_global:.2f}/sqm)."
        )

st.divider()

tab_estimate, tab_explain, tab_match = st.tabs(
    ["Price Estimate", "What Drives the Price", "Similar Apartments"]
)

with tab_estimate:
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Selected city", city)
    metric_col2.metric("Apartment profile", f"{size} sqm | {rooms} rooms")
    metric_col3.metric("Amenities selected", f"{amenities}/6")
    st.caption(
        "Compares the predicted price per sqm against the city average."
    )

    if est_monthly and city_mean_ppsm:
        benchmark_frame = {
            "Label": ["Your estimate", f"{city} mean"],
            "EUR/sqm": [price_global, city_mean_ppsm],
        }
        benchmark_fig = px.bar(
            benchmark_frame,
            x="Label",
            y="EUR/sqm",
            color="Label",
            color_discrete_map={
                "Your estimate": "#d88a13",
                f"{city} mean": "#0c2d48",
            },
            title="Predicted price per sqm versus city mean",
        )
        benchmark_fig.update_layout(
            margin=dict(l=10, r=10, t=50, b=10),
            showlegend=False,
            yaxis_title="EUR per sqm",
        )
        st.plotly_chart(benchmark_fig, use_container_width=True)
    else:
        st.info("Benchmark chart is unavailable for the current selection.")

with tab_explain:
    explain_left, explain_right = st.columns(2)

    with explain_left:
        st.subheader(f"What matters most in {city}")
        st.caption(
            "Longer bars indicate features with greater influence on the prediction in this market."
        )

        shap_df = get_shap_ranking(state=state, city=city)
        if not shap_df.empty:
            shap_df = shap_df.copy()
            shap_df["label"] = shap_df["feature"].map(
                lambda feature: FEATURE_LABELS.get(
                    feature, feature.replace("_", " ").replace("state ", "").title()
                )
            )
            shap_df = shap_df.drop_duplicates("label").sort_values("mean_abs_shap")
            fig = px.bar(
                shap_df,
                x="mean_abs_shap",
                y="label",
                orientation="h",
                color="mean_abs_shap",
                color_continuous_scale="Blues",
                labels={"mean_abs_shap": "Relative influence", "label": ""},
            )
            fig.update_layout(
                showlegend=False,
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=10, b=10),
                height=320,
                xaxis_title="Relative influence on prediction",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Area insights not available yet.")

    with explain_right:
        st.subheader(f"Rent spread in {city}")
        st.caption(
            "Shows where the predicted EUR/sqm sits within the full city distribution."
        )
        city_data = raw_df.loc[raw_df["city"] == city, "price_per_sqm"].dropna()
        city_data = city_data[city_data.between(1, 50)]
        if len(city_data) > 10:
            fig2 = px.histogram(
                city_data,
                nbins=35,
                color_discrete_sequence=["#0c2d48"],
                labels={"value": "EUR/sqm"},
            )
            if price_global:
                fig2.add_vline(
                    x=price_global,
                    line_dash="dash",
                    line_color="#d88a13",
                    annotation_text="Your estimate",
                    annotation_position="top right",
                )
            fig2.add_vline(
                x=city_mean_ppsm,
                line_dash="dot",
                line_color="#377117",
                annotation_text="City mean",
                annotation_position="top left",
            )
            fig2.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=320,
                showlegend=False,
                xaxis_title="EUR per sqm",
                yaxis_title="Number of listings",
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Not enough listings for a distribution chart.")

with tab_match:
    st.caption(
        "Search by description or use the current apartment settings to find similar listings."
    )
    match_tab1, match_tab2 = st.tabs(
        ["Describe What You Want", "Match My Apartment Profile"]
    )

    with match_tab1:
        left_query, right_query = st.columns([3, 1])
        with left_query:
            query = st.text_area(
                "What kind of apartment are you looking for?",
                placeholder="e.g. affordable studio near city centre with lift in Munich",
                height=110,
            )
        with right_query:
            same_city_only = st.checkbox("Same city only", value=True)
            k_results = st.selectbox("Show", [5, 8, 10], index=0, key="k_desc")

        if st.button("Find matches from description", use_container_width=True, key="btn_desc"):
            if query.strip():
                with st.spinner("Searching through 268,000 listings..."):
                    results = find_similar_by_description(
                        raw_df,
                        query=query,
                        city=city if same_city_only else None,
                        k=k_results,
                    )
                if not results.empty:
                    st.success(f"Found {len(results)} matching apartments")
                    st.dataframe(results, use_container_width=True, hide_index=True)
                else:
                    st.warning(
                        "No close matches found. Try a broader description or uncheck 'Same city only'."
                    )
            else:
                st.info("Type a description and then run the search.")

    with match_tab2:
        k_feat = st.selectbox("Number of results", [5, 8, 10], index=0, key="k_feat")

        if st.button(
            "Find matches from apartment profile",
            use_container_width=True,
            key="btn_feat",
        ):
            with st.spinner("Searching for similar apartments..."):
                results_feat = find_similar_by_features(
                    raw_df,
                    city=city,
                    rooms=rooms,
                    amenity_score=amenities,
                    has_kitchen=int(has_kitchen),
                    lift=int(has_lift),
                    size=size,
                    k=k_feat,
                )
            if not results_feat.empty:
                st.success(f"Found {len(results_feat)} similar apartments in {city}")
                st.dataframe(results_feat, use_container_width=True, hide_index=True)
            else:
                st.info("No close matches found in this city. Try a different city.")
