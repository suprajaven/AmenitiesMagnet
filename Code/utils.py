from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
ORIGINAL_DATA_PATH = DATA_DIR / "immo_data_clean.csv"
TRANSFORMED_DATA_PATH = DATA_DIR / "immo_data_clean_transformed.csv"

MODEL_NUMERIC_FEATURES = [
    "serviceCharge",
    "noRooms",
    "building_age",
    "amenity_score",
    "condition_score",
    "interior_score",
]

MODEL_CATEGORICAL_FEATURES = [
    "hasKitchen",
    "lift",
    "condition",
    "interiorQual",
    "typeOfFlat",
    "heatingType",
    "floor",
    "city",
    "state",
]

STATE_CENTROIDS = {
    "Baden Württemberg": {"lat": 48.6616, "lon": 9.3501},
    "Bayern": {"lat": 48.7904, "lon": 11.4979},
    "Berlin": {"lat": 52.5200, "lon": 13.4050},
    "Brandenburg": {"lat": 52.4125, "lon": 12.5316},
    "Bremen": {"lat": 53.0793, "lon": 8.8017},
    "Hamburg": {"lat": 53.5511, "lon": 9.9937},
    "Hessen": {"lat": 50.6521, "lon": 9.1624},
    "Mecklenburg Vorpommern": {"lat": 53.6127, "lon": 12.4296},
    "Niedersachsen": {"lat": 52.6367, "lon": 9.8451},
    "Nordrhein Westfalen": {"lat": 51.4332, "lon": 7.6616},
    "Rheinland Pfalz": {"lat": 49.9929, "lon": 7.8467},
    "Saarland": {"lat": 49.3964, "lon": 7.0230},
    "Sachsen": {"lat": 51.1045, "lon": 13.2017},
    "Sachsen Anhalt": {"lat": 51.9503, "lon": 11.6923},
    "Schleswig Holstein": {"lat": 54.2194, "lon": 9.6961},
    "Thüringen": {"lat": 50.9848, "lon": 11.0299},
}


@st.cache_data
def load_data():
    return pd.read_csv(ORIGINAL_DATA_PATH)


@st.cache_data
def load_model_data():
    return pd.read_csv(TRANSFORMED_DATA_PATH)


def make_quantile_bins(series: pd.Series, labels: list[str]) -> pd.Series:
    ranked = series.rank(method="first")
    return pd.qcut(ranked, q=len(labels), labels=labels)


def prepare_state_market_view(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("state", as_index=False)
        .agg(
            listings=("price_per_sqm", "size"),
            avg_price_per_sqm=("price_per_sqm", "mean"),
            median_price_per_sqm=("price_per_sqm", "median"),
            avg_amenity_score=("amenity_score", "mean"),
        )
    )
    coords = pd.DataFrame.from_dict(STATE_CENTROIDS, orient="index").reset_index()
    coords.columns = ["state", "lat", "lon"]
    return summary.merge(coords, on="state", how="left")


def comparable_listing_estimate(
    df: pd.DataFrame,
    city: str,
    size: float,
    rooms: float,
    amenity_score: float,
    has_kitchen: int,
    lift: int,
) -> tuple[float | None, int]:
    subset = df[df["city"] == city].copy()
    subset = subset[subset["livingSpace"].between(size - 15, size + 15)]
    subset = subset[subset["noRooms"].between(rooms - 1, rooms + 1)]
    subset = subset[subset["amenity_score"].between(amenity_score - 1, amenity_score + 1)]
    subset = subset[subset["hasKitchen"] == has_kitchen]
    subset = subset[subset["lift"] == lift]

    if len(subset) < 15:
        subset = df[df["city"] == city].copy()
        subset = subset[subset["livingSpace"].between(size - 25, size + 25)]
        subset = subset[subset["noRooms"].between(rooms - 1, rooms + 1)]
        subset = subset[subset["amenity_score"].between(amenity_score - 2, amenity_score + 2)]

    if subset.empty:
        return None, 0

    return float(subset["baseRent"].median()), int(len(subset))
