"""
============================================================
AMENITIES MAGNET — Analysis/Modeling/model_utils.py
============================================================

============================================================
"""

from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import faiss
import streamlit as st

warnings.filterwarnings("ignore")

# model_utils.py lives at:  Analysis/Modeling/model_utils.py
# artifacts/ lives at:      Analysis/Modeling/artifacts/
_HERE        = Path(__file__).resolve().parent
ARTIFACT_DIR = _HERE / "artifacts"

# ── Feature schema — must match train_models.py exactly ───────
NUM_FEATS  = ["serviceCharge","noRooms","building_age",
              "amenity_score","condition_score","interior_score"]
BOOL_FEATS = ["hasKitchen","lift"]
CAT_FEATS  = ["condition","interiorQual","typeOfFlat",
              "heatingType","floor","city","state"]
ALL_FEATS  = NUM_FEATS + BOOL_FEATS + CAT_FEATS


@st.cache_resource(show_spinner="Loading prediction model …")
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

@st.cache_resource(show_spinner=False)
def _load_faiss():
    ip = ARTIFACT_DIR / "faiss_index.bin"
    mp = ARTIFACT_DIR / "listing_index.parquet"
    if not ip.exists() or not mp.exists():
        return None, None
    return faiss.read_index(str(ip)), pd.read_parquet(str(mp))

@st.cache_resource(show_spinner=False)
def _load_tfidf():
    p = ARTIFACT_DIR / "tfidf_vectorizer.pkl"
    return joblib.load(p) if p.exists() else None

@st.cache_resource(show_spinner=False)
def _load_faiss_scaler():
    p = ARTIFACT_DIR / "faiss_scaler.pkl"
    return joblib.load(p) if p.exists() else None

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


def find_similar(query, city: str = None, k: int = 8) -> pd.DataFrame:
    index, meta = _load_faiss()
    if index is None:
        return pd.DataFrame()

    vec = None
    if isinstance(query, str) and query.strip():
        tfidf = _load_tfidf()
        if tfidf is not None:
            vec = tfidf.transform([query.lower()]).toarray().astype("float32")

    if vec is None:
        scaler = _load_faiss_scaler()
        if scaler is None:
            return pd.DataFrame()
        vals = [float(query.get(f, 0)) if isinstance(query, dict) else 0.0
                for f in NUM_FEATS + BOOL_FEATS]
        vec = scaler.transform([vals]).astype("float32")

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm

    scores, indices = index.search(vec, k * 3)
    valid  = indices[0] >= 0
    result = meta.iloc[indices[0][valid]].copy()
    result["similarity_score"] = (scores[0][valid] * 100).round(1)

    if city and "city" in result.columns:
        cr = result[result["city"] == city]
        result = cr if len(cr) >= 3 else result

    return result.head(k).reset_index(drop=True)


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
