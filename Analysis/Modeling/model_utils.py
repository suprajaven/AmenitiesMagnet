"""
TF-IDF similarity search uses scikit-learn.
"""

from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

_HERE        = Path(__file__).resolve().parent
ARTIFACT_DIR = _HERE / "artifacts"

NUM_FEATS  = ["serviceCharge","noRooms","building_age",
              "amenity_score","condition_score","interior_score"]
BOOL_FEATS = ["hasKitchen","lift"]
CAT_FEATS  = ["condition","interiorQual","typeOfFlat",
              "heatingType","floor","city","state"]
ALL_FEATS  = NUM_FEATS + BOOL_FEATS + CAT_FEATS


# ── Listing description builder ───────────────────────────────
def _make_listing_description(row) -> str:
   
    parts = []

    # Location
    city  = str(row.get("city",  "")).replace("_"," ").strip()
    state = str(row.get("state", "")).replace("_"," ").strip()
    if city:  parts.append(f"city {city}")
    if state: parts.append(f"state {state}")

    # Apartment type
    tof = str(row.get("typeOfFlat","")).replace("_"," ")
    if tof and tof != "nan":
        parts.append(f"{tof} apartment")

    # Rooms and size
    rooms = row.get("noRooms", None)
    if pd.notna(rooms):
        r = int(rooms)
        label = {1:"studio",2:"two room",3:"three room",
                 4:"four room",5:"five room"}.get(r, f"{r} room")
        parts.append(f"{label} flat")

    size = row.get("livingSpace", None)
    if pd.notna(size):
        parts.append(f"{int(size)} sqm")

    # Condition
    cond = str(row.get("condition","")).replace("_"," ")
    if cond and cond not in ("nan","unknown"):
        parts.append(cond)

    # Interior quality
    iq = str(row.get("interiorQual","")).replace("_"," ")
    if iq and iq not in ("nan","unknown"):
        parts.append(f"{iq} interior")

    # Heating
    ht = str(row.get("heatingType","")).replace("_"," ")
    if ht and ht not in ("nan","unknown"):
        parts.append(ht)

    # Amenities as words
    amenity_words = []
    if row.get("hasKitchen", False): amenity_words.append("kitchen")
    if row.get("lift",        False): amenity_words.append("lift elevator")
    if row.get("balcony",     False): amenity_words.append("balcony")
    if row.get("garden",      False): amenity_words.append("garden")
    if row.get("cellar",      False): amenity_words.append("cellar storage")
    if row.get("newlyConst",  False): amenity_words.append("new construction newly built")
    if amenity_words:
        parts.extend(amenity_words)

    # Building age qualitative label
    age = row.get("building_age", None)
    if pd.notna(age):
        if   age <= 5:   parts.append("new building modern")
        elif age <= 20:  parts.append("recent construction")
        elif age <= 50:  parts.append("established building")
        elif age <= 80:  parts.append("older building")
        else:            parts.append("historic old building")

    # Floor
    floor = row.get("floor", None)
    if pd.notna(floor):
        f = int(float(str(floor))) if str(floor).replace(".","").isdigit() else None
        if f is not None:
            if   f == 0: parts.append("ground floor")
            elif f == 1: parts.append("first floor")
            elif f == 2: parts.append("second floor")
            elif f >= 5: parts.append("high floor top floor")
            else:        parts.append(f"floor {f}")

    # Price tier
    ppsm = row.get("price_per_sqm", None)
    if pd.notna(ppsm):
        if   ppsm < 6:   parts.append("affordable budget low cost")
        elif ppsm < 10:  parts.append("mid range average price")
        elif ppsm < 15:  parts.append("above average premium")
        else:            parts.append("luxury expensive high end")

    return " ".join(parts).lower()


# ── TF-IDF index ────────────────────
@st.cache_resource(show_spinner=False)
def _build_tfidf_index(raw_df: pd.DataFrame):
    """
    Build TF-IDF matrix from synthetic listing descriptions.
    Cached — runs once per session.
    """
    desc = raw_df.apply(_make_listing_description, axis=1).tolist()
    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(desc)
    return vectorizer, matrix


# ── Model loaders ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Preparing price model …")
def _load_preprocessor():
    p = ARTIFACT_DIR / "preprocessor.pkl"
    return joblib.load(p) if p.exists() else None

@st.cache_resource(show_spinner=False)
def _load_global_model():
    p = ARTIFACT_DIR / "xgb_global.json"
    if not p.exists():
        return None
    m = xgb.XGBRegressor()
    m.load_model(str(p))
    return m

@st.cache_resource(show_spinner=False)
def _load_state_models():
    rp = ARTIFACT_DIR / "state_model_registry.json"
    if not rp.exists():
        return {}
    with open(rp) as f:
        registry = json.load(f)
    models = {}
    for state, meta in registry.items():
        mp = ARTIFACT_DIR / meta["file"]
        if mp.exists():
            m = xgb.XGBRegressor()
            m.load_model(str(mp))
            models[state] = {"model": m, **{k: v for k, v in meta.items() if k != "model"}}
    return models

@st.cache_data(show_spinner=False)
def _load_shap_rankings():
    sp = ARTIFACT_DIR / "state_shap_rankings.csv"
    cp = ARTIFACT_DIR / "city_shap_rankings.csv"
    return (
        pd.read_csv(sp) if sp.exists() else pd.DataFrame(),
        pd.read_csv(cp) if cp.exists() else pd.DataFrame(),
    )

@st.cache_resource(show_spinner=False)
def _load_label_encoders():
    p = ARTIFACT_DIR / "label_encoders.pkl"
    return joblib.load(p) if p.exists() else {}


# ── Public API ────────────────────────────────────────────────

def artifacts_ready() -> bool:
    required = ["xgb_global.json", "preprocessor.pkl",
                "state_shap_rankings.csv", "city_shap_rankings.csv"]
    return all((ARTIFACT_DIR / f).exists() for f in required)


def predict_price(input_dict: dict, state: str | None = None) -> dict:
    preprocessor = _load_preprocessor()
    global_model = _load_global_model()
    state_models = _load_state_models()

    if preprocessor is None or global_model is None:
        return {"price_per_sqm_global": None, "price_per_sqm_state": None,
                "log_pred_global": None, "model_used": "unavailable", "state_r2": None}

    row = pd.DataFrame([{f: input_dict.get(f, 0) for f in ALL_FEATS}])
    for c in BOOL_FEATS:
        row[c] = row[c].astype(int)
    for c in CAT_FEATS:
        row[c] = row[c].astype(str)

    X            = preprocessor.transform(row)
    log_global   = float(global_model.predict(X)[0])
    price_global = float(np.exp(log_global))

    price_state = None
    state_r2    = None
    model_used  = "global"

    if state and state in state_models:
        log_state   = float(state_models[state]["model"].predict(X)[0])
        price_state = float(np.exp(log_state))
        state_r2    = state_models[state].get("r2")
        model_used  = "state+global"

    return {
        "price_per_sqm_global": round(price_global, 2),
        "price_per_sqm_state":  round(price_state, 2) if price_state else None,
        "log_pred_global":      round(log_global, 4),
        "model_used":           model_used,
        "state_r2":             state_r2,
    }


def find_similar_by_description(
    raw_df:      pd.DataFrame,
    query:       str,
    city:        str  = None,
    k:           int  = 6,
) -> pd.DataFrame:
    """
    TF-IDF similarity search using scikit-learn only.
    Converts user's free-text query into a vector and finds
    the most similar listings by cosine similarity.
    Optionally filters to the same city first.
    """
    if not query.strip():
        return pd.DataFrame()

    vectorizer, matrix = _build_tfidf_index(raw_df)

    # Optionally narrow to city
    if city:
        city_mask = raw_df["city"] == city
        if city_mask.sum() >= 20:
            sub_df     = raw_df[city_mask].reset_index(drop=True)
            sub_matrix = matrix[city_mask.values]
        else:
            sub_df     = raw_df.reset_index(drop=True)
            sub_matrix = matrix
    else:
        sub_df     = raw_df.reset_index(drop=True)
        sub_matrix = matrix

    # Vectorise query and compute similarity
    q_vec  = vectorizer.transform([query.lower()])
    sims   = cosine_similarity(q_vec, sub_matrix).flatten()
    top_idx = sims.argsort()[::-1][:k]

    result = sub_df.iloc[top_idx].copy()
    result["Match score"] = (sims[top_idx] * 100).round(1)

    # Build display table with friendly column names
    keep = {}
    col_map = {
        "city":          "City",
        "state":         "State",
        "baseRent":      "Rent (€/mo)",
        "price_per_sqm": "€/sqm",
        "livingSpace":   "Size (sqm)",
        "noRooms":       "Rooms",
        "condition":     "Condition",
        "interiorQual":  "Interior",
        "heatingType":   "Heating",
        "amenity_score": "Amenities",
        "hasKitchen":    "Kitchen",
        "lift":          "Lift",
        "building_age":  "Building age",
        "Match score":   "Match score",
    }
    available = {k: v for k, v in col_map.items() if k in result.columns}
    display = result[list(available.keys())].copy()
    display.columns = list(available.values())

    # Clean up boolean columns
    for col in ["Kitchen", "Lift"]:
        if col in display.columns:
            display[col] = display[col].map({True: "✓", False: "✗", 1: "✓", 0: "✗"})

    # Clean up text columns
    for col in ["Condition", "Interior", "Heating"]:
        if col in display.columns:
            display[col] = display[col].str.replace("_", " ").str.title()

    display = display.round({"Rent (€/mo)": 0, "€/sqm": 2,
                              "Size (sqm)": 0, "Building age": 0}).reset_index(drop=True)
    # Only show results with non-zero match
    display = display[display["Match score"] > 0]
    return display


def find_similar_by_features(
    raw_df:       pd.DataFrame,
    city:         str,
    rooms:        float,
    amenity_score: float,
    has_kitchen:  int,
    lift:         int,
    size:         float,
    k:            int = 6,
) -> pd.DataFrame:
    """Pure-pandas fallback: find structurally similar listings."""
    subset = raw_df[raw_df["city"] == city].copy()
    if len(subset) < 5:
        subset = raw_df.copy()

    subset = subset.dropna(subset=["noRooms","amenity_score","livingSpace","price_per_sqm"])
    if subset.empty:
        return pd.DataFrame()

    subset["_score"] = (
        (subset["noRooms"]       - rooms).abs() * 2.0 +
        (subset["amenity_score"] - amenity_score).abs() * 1.5 +
        (subset["livingSpace"]   - size).abs() / 20.0 +
        (subset["hasKitchen"].astype(int) - has_kitchen).abs() * 1.0 +
        (subset["lift"].astype(int)       - lift).abs() * 0.5
    )

    top = subset.sort_values("_score").head(k).reset_index(drop=True)

    col_map = {
        "city": "City", "state": "State",
        "baseRent": "Rent (€/mo)", "price_per_sqm": "€/sqm",
        "livingSpace": "Size (sqm)", "noRooms": "Rooms",
        "condition": "Condition", "interiorQual": "Interior",
        "amenity_score": "Amenities",
        "hasKitchen": "Kitchen", "lift": "Lift",
        "building_age": "Building age",
    }
    available = {k: v for k, v in col_map.items() if k in top.columns}
    display = top[list(available.keys())].copy()
    display.columns = list(available.values())

    for col in ["Kitchen","Lift"]:
        if col in display.columns:
            display[col] = display[col].map({True:"✓",False:"✗",1:"✓",0:"✗"})
    for col in ["Condition","Interior"]:
        if col in display.columns:
            display[col] = display[col].str.replace("_"," ").str.title()

    return display.round({"Rent (€/mo)":0,"€/sqm":2,"Size (sqm)":0}).reset_index(drop=True)


def get_shap_ranking(state: str = None, city: str = None) -> pd.DataFrame:
    state_df, city_df = _load_shap_rankings()

    if city and not city_df.empty:
        sub = city_df[city_df["city"] == city].drop_duplicates("feature")
        if not sub.empty:
            return sub.sort_values("rank")[["rank","feature","mean_abs_shap"]].reset_index(drop=True)

    if state and not state_df.empty:
        sub = state_df[state_df["state"] == state].drop_duplicates("feature")
        if not sub.empty:
            return sub.sort_values("rank")[["rank","feature","mean_abs_shap"]].reset_index(drop=True)

    return pd.DataFrame()


def get_state_model_info() -> dict:
    return {s: {k: v for k, v in m.items() if k != "model"}
            for s, m in _load_state_models().items()}


def get_label_options() -> dict:
    return _load_label_encoders()
