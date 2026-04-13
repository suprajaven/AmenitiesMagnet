"""
============================================================
AMENITIES MAGNET — train_models.py
============================================================

Artifacts 
  xgb_global.json           — global XGBoost model
  xgb_state_<name>.json     — one XGBoost per state (16 total)
  preprocessor.pkl          — fitted sklearn ColumnTransformer
  tfidf_vectorizer.pkl      — fitted TF-IDF on description text
  faiss_index.bin           — FAISS index for similarity search
  listing_index.pkl         — mapping: FAISS row → listing metadata
  state_shap_rankings.csv   — top-5 SHAP per state (dashboard lookup)
  city_shap_rankings.csv    — top-5 SHAP per city  (dashboard lookup)
  label_encoders.pkl        — unique values for each categorical
  feature_names.pkl         — encoded feature name list

============================================================
"""

import os, warnings, pickle, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler, OneHotEncoder
from sklearn.compose         import ColumnTransformer
from sklearn.metrics         import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import shap
import faiss
import joblib

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[2]  
DATA_DIR     = ROOT / "Data"
ARTIFACT_DIR = ROOT / "Analysis" / "Modeling" / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TRANSFORMED_DATA = DATA_DIR / "immo_data_clean_transformed.csv"
ORIGINAL_DATA    = DATA_DIR / "immo_data_clean.csv"   

RANDOM_SEED  = 7406
MIN_STATE_N  = 500      # min training rows to build a state model
MIN_CITY_N   = 150      # min rows for city SHAP lookup
TOP_CITIES   = 30

# ──  XGBoost params 
XGB_PARAMS = {
    "max_depth":        6,
    "learning_rate":    0.08,
    "colsample_bytree": 0.9,
    "subsample":        0.8,
    "min_child_weight": 5,
    "gamma":            0.1,
    "reg_lambda":       2,
    "reg_alpha":        0,
    "n_estimators":     1000,
    "early_stopping_rounds": 30,
    "eval_metric":      "rmse",
    "random_state":     RANDOM_SEED,
    "n_jobs":           -1,
    "verbosity":        0,
}

print("=" * 60)
print("  AMENITIES MAGNET — Model Training Pipeline")
print("=" * 60)

# ── 1. Load data ─────────────────────────────────────────────
print("\n[1] Loading transformed data …")
df = pd.read_csv(TRANSFORMED_DATA, low_memory=False)
df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")],
        inplace=True, errors="ignore")
for c in ["hasKitchen", "lift"]:
    if c in df.columns:
        df[c] = df[c].astype(int)
if "floor" in df.columns:
    df["floor"] = df["floor"].astype(str)
print(f"    Shape: {df.shape}")

# ── 2. Feature definition ─────────────────────────────────────
NUM_FEATS  = ["serviceCharge","noRooms","building_age",
              "amenity_score","condition_score","interior_score"]
BOOL_FEATS = ["hasKitchen","lift"]
CAT_FEATS  = ["condition","interiorQual","typeOfFlat",
              "heatingType","floor","city","state"]

NUM_FEATS  = [c for c in NUM_FEATS  if c in df.columns]
BOOL_FEATS = [c for c in BOOL_FEATS if c in df.columns]
CAT_FEATS  = [c for c in CAT_FEATS  if c in df.columns]
ALL_FEATS  = NUM_FEATS + BOOL_FEATS + CAT_FEATS
TARGET     = "log_price_per_sqm"

model_df = df[ALL_FEATS + [TARGET]].dropna()
print(f"    Modelling rows: {len(model_df):,}")

# Save label encoders (unique values per categorical)
label_encoders = {c: sorted(model_df[c].astype(str).unique().tolist())
                  for c in CAT_FEATS}
joblib.dump(label_encoders, ARTIFACT_DIR / "label_encoders.pkl")

# ── 3. Train/test split ───────────────────────────────────────
X = model_df[ALL_FEATS]
y = model_df[TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED)
print(f"\n[3] Split → train: {len(X_tr):,}  test: {len(X_te):,}")

# ── 4. Preprocessor ───────────────────────────────────────────
print("\n[4] Fitting preprocessor …")
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUM_FEATS + BOOL_FEATS),
    ("cat", OneHotEncoder(handle_unknown="ignore",
                          sparse_output=False, drop="first"), CAT_FEATS),
])
X_tr_p = preprocessor.fit_transform(X_tr)
X_te_p  = preprocessor.transform(X_te)
joblib.dump(preprocessor, ARTIFACT_DIR / "preprocessor.pkl")

# Feature names
ohe_names  = (preprocessor.named_transformers_["cat"]
               .get_feature_names_out(CAT_FEATS).tolist())
feat_names_raw = NUM_FEATS + BOOL_FEATS + ohe_names
seen = {}; feat_names = []
for f in feat_names_raw:
    key = f
    while key in seen:
        key = f + f"_dup{seen.get(f,0)}"
    seen[key] = True; feat_names.append(key)
joblib.dump(feat_names, ARTIFACT_DIR / "feature_names.pkl")
print(f"    Encoded dims: {X_tr_p.shape[1]}")

# ── 5. Global XGBoost ─────────────────────────────────────────
print("\n[5] Training global XGBoost …")
t0 = time.time()
X_tr2, X_val, y_tr2, y_val = train_test_split(
    X_tr_p, y_tr, test_size=0.10, random_state=RANDOM_SEED)

global_model = xgb.XGBRegressor(**XGB_PARAMS)
global_model.fit(X_tr_p, y_tr,
                 eval_set=[(X_val, y_val)],
                 verbose=False)
global_model.save_model(str(ARTIFACT_DIR / "xgb_global.json"))

pred_te = global_model.predict(X_te_p)
rmse = np.sqrt(mean_squared_error(y_te, pred_te))
r2   = r2_score(y_te, pred_te)
print(f"    Global XGBoost — RMSE={rmse:.4f}  R²={r2:.4f}  "
      f"iter={global_model.best_iteration}  [{time.time()-t0:.1f}s]")

# ── 6. SHAP lookups (state + city) ───────────────────────────
print("\n[6] Computing SHAP rankings …")
BG_N = min(500, len(X_tr_p))
bg_i = np.random.default_rng(RANDOM_SEED).choice(len(X_tr_p), BG_N, replace=False)
explainer = shap.TreeExplainer(global_model,
                                data=X_tr_p[bg_i],
                                feature_names=feat_names)

# Core non-location features for SHAP ranking
CORE = [f for f in feat_names
        if not f.startswith("state_") and not f.startswith("city_")]

def compute_shap_ranking(mask, label, n_sample=600):
    idx = np.where(mask)[0]
    if len(idx) < 30:
        return None
    cap  = min(n_sample, len(idx))
    pick = np.random.default_rng(RANDOM_SEED).choice(len(idx), cap, replace=False)
    sv   = np.abs(explainer(X_te_p[idx[pick]]).values)
    mean_sv = pd.Series(sv.mean(axis=0), index=feat_names)
    mean_sv = mean_sv.groupby(level=0).first()
    core_sv = mean_sv[[f for f in CORE if f in mean_sv.index]]
    top5    = core_sv.sort_values(ascending=False).head(5)
    return [{"rank": i+1, "feature": f, "mean_abs_shap": round(v, 6)}
            for i, (f, v) in enumerate(top5.items())]

state_rows, city_rows = [], []

for state in X_te["state"].value_counts().index:
    mask = X_te["state"].values == state
    ranking = compute_shap_ranking(mask, state)
    if ranking:
        for r in ranking:
            state_rows.append({"state": state, **r})
        print(f"    ✓ state: {state}")

for city in X_te["city"].value_counts().head(TOP_CITIES).index:
    mask = X_te["city"].values == city
    ranking = compute_shap_ranking(mask, city, n_sample=400)
    if ranking:
        for r in ranking:
            city_rows.append({"city": city, **r})

pd.DataFrame(state_rows).to_csv(ARTIFACT_DIR / "state_shap_rankings.csv", index=False)
pd.DataFrame(city_rows).to_csv(ARTIFACT_DIR / "city_shap_rankings.csv", index=False)
print(f"    Saved state_shap_rankings.csv ({len(state_rows)} rows)")
print(f"    Saved city_shap_rankings.csv  ({len(city_rows)} rows)")

# ── 7. State-level XGBoost models ────────────────────────────
print("\n[7] Training state-level XGBoost models …")
state_model_registry = {}   

for state in model_df["state"].unique():
    state_df   = model_df[model_df["state"] == state]
    n          = len(state_df)
    if n < MIN_STATE_N:
        print(f"    ✗ {state} — only {n} rows (< {MIN_STATE_N}), skipping")
        continue

    Xs = state_df[ALL_FEATS]
    ys = state_df[TARGET]
    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(
        Xs, ys, test_size=0.20, random_state=RANDOM_SEED)

    Xs_tr_p = preprocessor.transform(Xs_tr)
    Xs_te_p = preprocessor.transform(Xs_te)
    Xs_tr2, Xs_val, ys_tr2, ys_val = train_test_split(
        Xs_tr_p, ys_tr, test_size=0.10, random_state=RANDOM_SEED)

    state_xgb = xgb.XGBRegressor(**XGB_PARAMS)
    state_xgb.fit(Xs_tr_p, ys_tr,
                  eval_set=[(Xs_val, ys_val)],
                  verbose=False)

    pred_s = state_xgb.predict(Xs_te_p)
    rmse_s = np.sqrt(mean_squared_error(ys_te, pred_s))
    r2_s   = r2_score(ys_te, pred_s)

    safe   = state.replace(" ", "_").replace("/", "_")
    fname  = f"xgb_state_{safe}.json"
    state_xgb.save_model(str(ARTIFACT_DIR / fname))

    state_model_registry[state] = {
        "file":      fname,
        "rmse":      round(rmse_s, 4),
        "r2":        round(r2_s, 4),
        "n_train":   len(Xs_tr),
        "best_iter": state_xgb.best_iteration,
    }
    print(f"    ✓ {state:<30} n={n:>6,}  RMSE={rmse_s:.4f}  R²={r2_s:.4f}")

with open(ARTIFACT_DIR / "state_model_registry.json", "w") as f:
    json.dump(state_model_registry, f, indent=2)
print(f"    Saved state_model_registry.json "
      f"({len(state_model_registry)} state models)")

# ── 8. TF-IDF + FAISS similarity index ──────────────────────
print("\n[8] Building TF-IDF + FAISS similarity index …")


try:
    orig_df = pd.read_csv(ORIGINAL_DATA, low_memory=False)
    has_text = ("description" in orig_df.columns or
                "facilities" in orig_df.columns)
except Exception:
    has_text = False

if has_text:
    # Merge available text columns
    text_cols = [c for c in ["description","facilities"] if c in orig_df.columns]
    orig_df["_text"] = orig_df[text_cols].fillna("").agg(" ".join, axis=1)

    # Keep only rows present in transformed data (same index alignment)
    merged = orig_df[["_text","city","state","baseRent","price_per_sqm",
                       "livingSpace","noRooms","amenity_score"]].copy()
    merged = merged.dropna(subset=["_text"])
    merged["_text"] = merged["_text"].str.lower().str.strip()
    merged = merged[merged["_text"].str.len() > 10].reset_index(drop=True)

    # TF-IDF (German text, trim to 5000 features for FAISS efficiency)
    print(f"    Fitting TF-IDF on {len(merged):,} listings …")
    tfidf = TfidfVectorizer(max_features=5000, min_df=5,
                            strip_accents="unicode", sublinear_tf=True)
    tfidf_matrix = tfidf.fit_transform(merged["_text"]).toarray().astype("float32")
    joblib.dump(tfidf, ARTIFACT_DIR / "tfidf_vectorizer.pkl")

    # Normalise for cosine similarity via inner product
    norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    tfidf_norm = tfidf_matrix / norms

    # FAISS flat index (exact cosine via normalised inner product)
    dim   = tfidf_norm.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(tfidf_norm)
    faiss.write_index(index, str(ARTIFACT_DIR / "faiss_index.bin"))

    # Listing metadata for display
    listing_meta = merged[["city","state","baseRent","price_per_sqm",
                            "livingSpace","noRooms","amenity_score"]].copy()
    listing_meta.to_parquet(ARTIFACT_DIR / "listing_index.parquet", index=False)

    print(f"    FAISS index: {index.ntotal:,} vectors  dim={dim}")
    print("    Saved faiss_index.bin + listing_index.parquet + tfidf_vectorizer.pkl")

else:
    # No text data available — build FAISS on structured features instead
    # This is the fallback: find similar listings by numeric feature similarity
    print("    No description text found — building structured FAISS fallback …")
    struct_data  = model_df[NUM_FEATS + BOOL_FEATS].fillna(0).values.astype("float32")
    scaler_faiss = StandardScaler().fit(struct_data)
    struct_norm  = scaler_faiss.transform(struct_data).astype("float32")

    norms = np.linalg.norm(struct_norm, axis=1, keepdims=True)
    norms[norms == 0] = 1
    struct_norm /= norms

    dim   = struct_norm.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(struct_norm)
    faiss.write_index(index, str(ARTIFACT_DIR / "faiss_index.bin"))

    listing_meta = model_df[["city","state",TARGET,
                              "noRooms","amenity_score","building_age"]].copy()
    listing_meta.to_parquet(ARTIFACT_DIR / "listing_index.parquet", index=False)

    joblib.dump(scaler_faiss, ARTIFACT_DIR / "faiss_scaler.pkl")
    joblib.dump(False,        ARTIFACT_DIR / "faiss_is_tfidf.pkl")

    print(f"    FAISS index: {index.ntotal:,} structured vectors  dim={dim}")
    print("    Saved faiss_index.bin + listing_index.parquet + faiss_scaler.pkl")

# ── 9. Summary ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TRAINING COMPLETE")
print("=" * 60)
print(f"\n  Artifacts saved to: {ARTIFACT_DIR}")
print(f"""
  ├── xgb_global.json              (global XGBoost, R²={r2:.4f})
  ├── xgb_state_<name>.json        ({len(state_model_registry)} state models)
  ├── state_model_registry.json    (metadata for all state models)
  ├── preprocessor.pkl             (fitted ColumnTransformer)
  ├── feature_names.pkl            (encoded feature name list)
  ├── label_encoders.pkl           (unique values per categorical)
  ├── state_shap_rankings.csv      (top-5 SHAP features per state)
  ├── city_shap_rankings.csv       (top-5 SHAP features per city)
  ├── faiss_index.bin              (similarity search index)
  ├── listing_index.parquet        (listing metadata for display)
  └── tfidf_vectorizer.pkl / faiss_scaler.pkl
""")
