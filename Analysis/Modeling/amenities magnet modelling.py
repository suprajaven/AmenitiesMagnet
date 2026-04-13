"""
============================================================
AMENITIES MAGNET 
============================================================

"""

import os, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection  import (train_test_split, cross_val_score,
                                      GridSearchCV, RandomizedSearchCV)
from sklearn.preprocessing    import StandardScaler, OneHotEncoder
from sklearn.compose          import ColumnTransformer
from sklearn.pipeline         import Pipeline
from sklearn.linear_model     import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition    import PCA
from sklearn.svm              import LinearSVR
from sklearn.ensemble         import (RandomForestRegressor,
                                      GradientBoostingRegressor)
from sklearn.metrics          import (mean_squared_error, r2_score,
                                      mean_absolute_error,
                                      mean_absolute_percentage_error,
                                      classification_report, confusion_matrix)
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")

# ── 0  SETTINGS ──────────────────────────────────────────────
DATA_PATH   = "immo_data_clean_transformed.csv"  
OUTPUT_DIR  = "model_outputs_v2"
RANDOM_SEED = 7406
TARGET      = "log_price_per_sqm"
TRAIN_SIZE  = 0.80
CV_LINEAR   = 3     
CV_TREE     = 5     
SAMPLE_N    = 100_000
N_JOBS      = -1

NAVY   = "#003057"
GOLD   = "#EAB900"
GREEN  = "#377117"

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})

print("  AMENITIES MAGNET  MODELLING")

# ── 1  LOAD DATA ─────────────────────────────────────────────
print("\n[1] Loading data …")
df = pd.read_csv(DATA_PATH, low_memory=False)

# Drop index column if present
drop_cols = [c for c in df.columns if c.startswith("Unnamed")]
if drop_cols:
    df.drop(columns=drop_cols, inplace=True)

print(f"    Shape: {df.shape}")
print(f"    Columns: {df.columns.tolist()}")
print(f"    Missing values: {df.isnull().sum().sum()}")

# ── 2  FEATURE DEFINITION ─────────────────────────────────────

print("\n[2] Feature sets (as per Phase 1 decisions) …")

NUM_FEATS  = ['serviceCharge', 'noRooms', 'building_age',
              'amenity_score', 'condition_score', 'interior_score']
BOOL_FEATS = ['hasKitchen', 'lift']        
CAT_FEATS  = ['condition', 'interiorQual', 'typeOfFlat',
              'heatingType', 'floor', 'city', 'state']

# Verify columns exist
NUM_FEATS  = [c for c in NUM_FEATS  if c in df.columns]
BOOL_FEATS = [c for c in BOOL_FEATS if c in df.columns]
CAT_FEATS  = [c for c in CAT_FEATS  if c in df.columns]
ALL_FEATS  = NUM_FEATS + BOOL_FEATS + CAT_FEATS

print(f"    Numeric  ({len(NUM_FEATS)}): {NUM_FEATS}")
print(f"    Boolean  ({len(BOOL_FEATS)}): {BOOL_FEATS}")
print(f"    Categorical ({len(CAT_FEATS)}): {CAT_FEATS}")

# Convert booleans to int
for c in BOOL_FEATS:
    df[c] = df[c].astype(int)


if 'floor' in df.columns:
    df['floor'] = df['floor'].astype(str)

model_df = df[ALL_FEATS + [TARGET]].dropna()
print(f"\n    Modelling rows: {len(model_df):,}")

# ── 3  TRAIN / TEST SPLIT ─────────────────────────────────────
print("\n[3] Train/test split 80/20 …")
X = model_df[ALL_FEATS]
y = model_df[TARGET]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=1 - TRAIN_SIZE, random_state=RANDOM_SEED)
print(f"    Train: {len(X_tr):,}  |  Test: {len(X_te):,}")

# ── 4  PREPROCESSING ─────────────────────────────────────────
print("\n[4] Fitting preprocessor …")


preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUM_FEATS + BOOL_FEATS),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                          drop="first"), CAT_FEATS),
])

X_tr_p = preprocessor.fit_transform(X_tr)
X_te_p = preprocessor.transform(X_te)

# Feature names after encoding
ohe_names = (preprocessor.named_transformers_["cat"]
             .get_feature_names_out(CAT_FEATS).tolist())
feat_names = NUM_FEATS + BOOL_FEATS + ohe_names
print(f"    Encoded dimensions: {X_tr_p.shape[1]}")
print(f"    (6 numeric + 2 boolean + {len(ohe_names)} OHE = {X_tr_p.shape[1]} total)")

# Sub-sample for linear model CV
s_idx  = np.random.choice(len(X_tr_p), min(SAMPLE_N, len(X_tr_p)), replace=False)
X_tr_s = X_tr_p[s_idx]
y_tr_s = y_tr.iloc[s_idx]
print(f"    Linear CV sample: {len(X_tr_s):,} rows / {CV_LINEAR} folds")

# ── 5  HELPERS ───────
results = []   

def metrics(y_true, y_pred, name=""):
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    r2    = r2_score(y_true, y_pred)
    mape  = mean_absolute_percentage_error(y_true, y_pred) * 100
    # Back-transform RMSE to price_per_sqm scale for interpretability
    rmse_bt = np.sqrt(mean_squared_error(np.expm1(y_true),
                                          np.expm1(y_pred)))
    print(f"    {name:22s}  RMSE={rmse:.4f}  MAE={mae:.4f}  "
          f"R²={r2:.4f}  MAPE={mape:.2f}%  RMSE_€/sqm={rmse_bt:.4f}")
    return dict(Model=name,
                RMSE=round(rmse,4), MAE=round(mae,4),
                R2=round(r2,4), MAPE_pct=round(mape,2),
                RMSE_ppsm=round(rmse_bt,4))

def fi_bar(features, importances, title, fname, color=NAVY, n=20):
    """Save a horizontal bar chart of top-n feature importances."""
    fi = (pd.DataFrame({"Feature": features, "Importance": importances})
            .sort_values("Importance", ascending=False).head(n))
    fig, ax = plt.subplots(figsize=(10, 7))
    fi.sort_values("Importance").plot(
        kind="barh", x="Feature", y="Importance",
        ax=ax, color=color, alpha=0.82, legend=False)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{fname}")
    plt.close()
    print(f"    Saved {fname}")
    return fi

# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  REGRESSION — TARGET: log_price_per_sqm")
print(f"  (Back-transform: np.exp(pred) → price in €/sqm)")
print("=" * 65)

# ── 6  NULL MODEL ─────────────────────────────────────────────
print("\n[6] Null model (mean) …")
null_pred = np.full(len(y_te), y_tr.mean())
results.append(metrics(y_te, null_pred, "Null(mean)"))

# ── 7  OLS  ───────────────────────────────────────────────────

print("\n[7] OLS (Phase 1 benchmark replication) …")
t0 = time.time()
ols = LinearRegression().fit(X_tr_p, y_tr)
ols_pred = ols.predict(X_te_p)
results.append(metrics(y_te, ols_pred, "OLS"))
cv_ols = -cross_val_score(LinearRegression(), X_tr_s, y_tr_s,
    cv=CV_LINEAR, scoring="neg_root_mean_squared_error", n_jobs=N_JOBS)
print(f"    CV-RMSE (sample, {CV_LINEAR}-fold): {cv_ols.mean():.4f} ± {cv_ols.std():.4f}  "
      f"[{time.time()-t0:.1f}s]")

# Top coefficients
coef_df = (pd.DataFrame({"Feature": feat_names, "Coef": ols.coef_})
            .reindex(pd.Series(ols.coef_).abs().sort_values(ascending=False).index))
print("\n    Top 15 OLS coefficients (standardised numeric, OHE categorical):")
print(coef_df.head(15).to_string(index=False))

# Save OLS coefficient plot
fig, ax = plt.subplots(figsize=(10, 7))
coef_top = coef_df.head(20)
colors = [GREEN if c >= 0 else "#C00000" for c in coef_top["Coef"]]
ax.barh(coef_top["Feature"], coef_top["Coef"], color=colors, alpha=0.80)
ax.axvline(0, color="black", lw=0.8)
ax.set_title("OLS — Top 20 Coefficients (standardised)", fontsize=12, fontweight="bold")
ax.set_xlabel("Coefficient value")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/ols_coefficients.png")
plt.close()
print("    Saved ols_coefficients.png")

# ── 8  RIDGE ─────────────────────────────────────────────────
print("\n[8] Ridge …")
t0 = time.time()
ridge_cv = GridSearchCV(Ridge(),
    {"alpha": [0.001, 0.01, 0.1, 1, 10, 50, 100]},
    cv=CV_LINEAR, scoring="neg_root_mean_squared_error", n_jobs=N_JOBS)
ridge_cv.fit(X_tr_s, y_tr_s)
ridge = Ridge(alpha=ridge_cv.best_params_["alpha"]).fit(X_tr_p, y_tr)
ridge_pred = ridge.predict(X_te_p)
results.append(metrics(y_te, ridge_pred, "Ridge"))
print(f"    Best alpha={ridge_cv.best_params_['alpha']}  [{time.time()-t0:.1f}s]")

# ── 9  LASSO ─────────────────────────────────────────────────
print("\n[9] LASSO …")
t0 = time.time()
lasso_cv = GridSearchCV(
    Lasso(max_iter=10_000, warm_start=True),
    {"alpha": [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]},
    cv=CV_LINEAR, scoring="neg_root_mean_squared_error", n_jobs=N_JOBS)
lasso_cv.fit(X_tr_s, y_tr_s)
best_lasso_alpha = lasso_cv.best_params_["alpha"]
lasso = Lasso(alpha=best_lasso_alpha, max_iter=10_000).fit(X_tr_p, y_tr)
lasso_pred = lasso.predict(X_te_p)
results.append(metrics(y_te, lasso_pred, "LASSO"))
n_zero = (lasso.coef_ == 0).sum()
print(f"    Best alpha={best_lasso_alpha}  Features zeroed: {n_zero}/{len(lasso.coef_)}  "
      f"[{time.time()-t0:.1f}s]")

lasso_kept = (pd.DataFrame({"Feature": feat_names, "Coef": lasso.coef_})
               .query("Coef != 0")
               .assign(AbsCoef=lambda d: d["Coef"].abs())
               .sort_values("AbsCoef", ascending=False))
print(f"    Top 15 retained LASSO features:")
print(lasso_kept[["Feature","Coef"]].head(15).to_string(index=False))

# ── 10  ELASTIC NET ───────────────────────────────────────────
print("\n[10] Elastic Net …")
t0 = time.time()
en_cv = GridSearchCV(
    ElasticNet(max_iter=10_000, warm_start=True),
    {"alpha": [0.001, 0.01, 0.05, 0.1, 0.5],
     "l1_ratio": [0.2, 0.5, 0.7, 0.9]},
    cv=CV_LINEAR, scoring="neg_root_mean_squared_error", n_jobs=N_JOBS)
en_cv.fit(X_tr_s, y_tr_s)
en = ElasticNet(**en_cv.best_params_, max_iter=10_000).fit(X_tr_p, y_tr)
en_pred = en.predict(X_te_p)
results.append(metrics(y_te, en_pred, "ElasticNet"))
print(f"    Best params={en_cv.best_params_}  [{time.time()-t0:.1f}s]")

# ── 11  PCR ───────────────────────────────────────────────────
print("\n[11] PCR (PCA + OLS) …")
t0 = time.time()
n_feat = X_tr_p.shape[1]
pcr_cv = GridSearchCV(
    Pipeline([("pca", PCA()), ("ols", LinearRegression())]),
    {"pca__n_components": list(range(10, min(n_feat, 80), 10))},
    cv=CV_LINEAR, scoring="neg_root_mean_squared_error", n_jobs=N_JOBS)
pcr_cv.fit(X_tr_s, y_tr_s)
best_nc = pcr_cv.best_params_["pca__n_components"]
pcr = Pipeline([("pca", PCA(n_components=best_nc)),
                ("ols", LinearRegression())]).fit(X_tr_p, y_tr)
pcr_pred = pcr.predict(X_te_p)
results.append(metrics(y_te, pcr_pred, "PCR"))
print(f"    Best n_components={best_nc}  [{time.time()-t0:.1f}s]")

# PCR explained variance plot
ev = pcr.named_steps["pca"].explained_variance_ratio_.cumsum()
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(ev)+1), ev, "o-", color=NAVY, lw=2, markersize=4)
ax.axhline(0.90, color="red", linestyle="--", lw=1.2, label="90% threshold")
ax.axvline(best_nc, color=GOLD, linestyle="--", lw=1.5, label=f"Best n={best_nc}")
ax.set_xlabel("Number of Components"); ax.set_ylabel("Cumulative Explained Variance")
ax.set_title("PCR — Explained Variance by Components", fontweight="bold"); ax.legend()
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/pcr_explained_variance.png"); plt.close()

# ── 12  PLS ───────────────────────────────────────────────────
print("\n[12] PLS Regression …")
t0 = time.time()
pls_cv = GridSearchCV(PLSRegression(),
    {"n_components": [3, 5, 8, 10, 12, 15, 20]},
    cv=CV_LINEAR, scoring="neg_root_mean_squared_error", n_jobs=N_JOBS)
pls_cv.fit(X_tr_s, y_tr_s)
best_pls = pls_cv.best_params_["n_components"]
pls = PLSRegression(n_components=best_pls).fit(X_tr_p, y_tr)
pls_pred = pls.predict(X_te_p).flatten()
results.append(metrics(y_te, pls_pred, "PLS"))
print(f"    Best n_components={best_pls}  [{time.time()-t0:.1f}s]")

# ── 13  SVR-LINEAR ────────────────────────────────────────────
print("\n[13] SVR-Linear (30 k sample) …")
t0 = time.time()
svr_idx = np.random.choice(len(X_tr_p), min(30_000, len(X_tr_p)), replace=False)
svr_cv  = GridSearchCV(LinearSVR(max_iter=3_000),
    {"C": [0.1, 0.5, 1.0, 5.0, 10.0]},
    cv=3, scoring="neg_root_mean_squared_error", n_jobs=N_JOBS)
svr_cv.fit(X_tr_p[svr_idx], y_tr.iloc[svr_idx])
svr = LinearSVR(C=svr_cv.best_params_["C"],
                max_iter=3_000).fit(X_tr_p[svr_idx], y_tr.iloc[svr_idx])
svr_pred = svr.predict(X_te_p)
results.append(metrics(y_te, svr_pred, "SVR-Linear"))
print(f"    Best C={svr_cv.best_params_['C']}  (30 k sample)  [{time.time()-t0:.1f}s]")

# ── 14  RANDOM FOREST ─────────────────────────────────────────
print("\n[14] Random Forest (RandomisedSearch, 12 iters) …")
t0 = time.time()
rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=N_JOBS),
    {"n_estimators": [200, 300, 500],
     "max_features": ["sqrt", 0.33, 0.4],
     "min_samples_leaf": [3, 5, 10],
     "max_depth": [None, 20, 30]},
    n_iter=12, cv=CV_TREE, scoring="neg_root_mean_squared_error",
    random_state=RANDOM_SEED, n_jobs=1, refit=True, verbose=0)
rf_search.fit(X_tr_p, y_tr)
rf      = rf_search.best_estimator_
rf_pred = rf.predict(X_te_p)
results.append(metrics(y_te, rf_pred, "RandomForest"))
cv_rf = -cross_val_score(rf, X_tr_p, y_tr, cv=CV_TREE,
    scoring="neg_root_mean_squared_error", n_jobs=N_JOBS)
print(f"    Best params: {rf_search.best_params_}")
print(f"    CV-RMSE: {cv_rf.mean():.4f} ± {cv_rf.std():.4f}  [{time.time()-t0:.1f}s]")
fi_bar(feat_names, rf.feature_importances_,
    "Random Forest — Top 20 Feature Importances",
    "rf_feature_importance.png", NAVY)

# ── 15  GBM ───────────────────────────────────────────────────
print("\n[15] GBM (RandomisedSearch, 10 iters) …")
t0 = time.time()
gbm_search = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=RANDOM_SEED),
    {"n_estimators": [200, 300],
     "learning_rate": [0.05, 0.08, 0.1],
     "max_depth": [4, 5, 6],
     "subsample": [0.7, 0.8],
     "min_samples_leaf": [5, 10]},
    n_iter=10, cv=CV_TREE, scoring="neg_root_mean_squared_error",
    random_state=RANDOM_SEED, n_jobs=N_JOBS, refit=True, verbose=0)
gbm_search.fit(X_tr_p, y_tr)
gbm      = gbm_search.best_estimator_
gbm_pred = gbm.predict(X_te_p)
results.append(metrics(y_te, gbm_pred, "GBM"))
cv_gbm = -cross_val_score(gbm, X_tr_p, y_tr, cv=CV_TREE,
    scoring="neg_root_mean_squared_error", n_jobs=N_JOBS)
print(f"    Best params: {gbm_search.best_params_}")
print(f"    CV-RMSE: {cv_gbm.mean():.4f} ± {cv_gbm.std():.4f}  [{time.time()-t0:.1f}s]")
fi_bar(feat_names, gbm.feature_importances_,
    "GBM — Top 20 Feature Importances", "gbm_feature_importance.png", GREEN)

# ── 16  XGBOOST (EARLY STOPPING) ─────────────────────────────
print("\n[16] XGBoost (RandomisedSearch + early stopping) …")
t0 = time.time()
X_tr2, X_val, y_tr2, y_val = train_test_split(
    X_tr_p, y_tr, test_size=0.10, random_state=RANDOM_SEED)

xgb_search = RandomizedSearchCV(
    xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=30,
                     eval_metric="rmse", random_state=RANDOM_SEED,
                     n_jobs=N_JOBS, verbosity=0),
    {"max_depth":        [4, 5, 6, 7],
     "learning_rate":    [0.03, 0.05, 0.08],
     "colsample_bytree": [0.7, 0.8, 0.9],
     "subsample":        [0.7, 0.8],
     "min_child_weight": [3, 5, 10],
     "gamma":            [0, 0.1],
     "reg_alpha":        [0, 0.1, 0.5],
     "reg_lambda":       [1, 2]},
    n_iter=20, cv=3, scoring="neg_root_mean_squared_error",
    random_state=RANDOM_SEED, n_jobs=1, refit=False, verbose=0)
xgb_search.fit(X_tr2, y_tr2, eval_set=[(X_val, y_val)], verbose=False)

bp = xgb_search.best_params_
print(f"    Best XGB params: {bp}")

# Refit on full training data with early stopping
xgb_model = xgb.XGBRegressor(**bp, n_estimators=1000,
    early_stopping_rounds=30, eval_metric="rmse",
    random_state=RANDOM_SEED, n_jobs=N_JOBS, verbosity=0)
xgb_model.fit(X_tr_p, y_tr, eval_set=[(X_te_p, y_te)], verbose=False)
xgb_pred = xgb_model.predict(X_te_p)
results.append(metrics(y_te, xgb_pred, "XGBoost"))
cv_xgb = -cross_val_score(
    xgb.XGBRegressor(**bp, n_estimators=xgb_model.best_iteration+1,
                      eval_metric="rmse", random_state=RANDOM_SEED,
                      n_jobs=N_JOBS, verbosity=0),
    X_tr_p, y_tr, cv=CV_TREE,
    scoring="neg_root_mean_squared_error", n_jobs=1)
print(f"    Best iteration={xgb_model.best_iteration}  "
      f"CV-RMSE: {cv_xgb.mean():.4f} ± {cv_xgb.std():.4f}  [{time.time()-t0:.1f}s]")
fi_bar(feat_names, xgb_model.feature_importances_,
    "XGBoost — Top 20 Feature Importances", "xgb_feature_importance.png", GOLD)

# ── 17  COMPARISON TABLE ──────────────────────────────────────
print("\n" + "=" * 65)
print("  MODEL COMPARISON — log_price_per_sqm")
print("=" * 65)
res_df = pd.DataFrame(results).sort_values("RMSE")
print(res_df.to_string(index=False))
res_df.to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=False)

# CV summary
cv_summary = pd.DataFrame([
    {"Model": "OLS",         "CV_RMSE": round(cv_ols.mean(),4),  "CV_STD": round(cv_ols.std(),4)},
    {"Model": "RandomForest","CV_RMSE": round(cv_rf.mean(),4),   "CV_STD": round(cv_rf.std(),4)},
    {"Model": "GBM",         "CV_RMSE": round(cv_gbm.mean(),4),  "CV_STD": round(cv_gbm.std(),4)},
    {"Model": "XGBoost",     "CV_RMSE": round(cv_xgb.mean(),4),  "CV_STD": round(cv_xgb.std(),4)},
]).sort_values("CV_RMSE")
cv_summary.to_csv(f"{OUTPUT_DIR}/cv_summary.csv", index=False)

# Comparison plots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
clrs = [GOLD if m == res_df.iloc[0]["Model"] else
        ("#cccccc" if m == "Null(mean)" else NAVY)
        for m in res_df["Model"]]

res_df.plot(kind="barh", x="Model", y="RMSE", ax=axes[0],
            color=clrs, legend=False)
axes[0].set_title("Test RMSE (log scale) — lower is better", fontweight="bold")
axes[0].invert_yaxis(); axes[0].set_xlabel("RMSE (log €/sqm)")

res_df.plot(kind="barh", x="Model", y="R2", ax=axes[1],
            color=clrs, legend=False)
axes[1].set_title("Test R² — higher is better", fontweight="bold")
axes[1].invert_yaxis(); axes[1].set_xlabel("R²")

fig.suptitle("Model Comparison — log_price_per_sqm", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/model_comparison_chart.png"); plt.close()
print("    Saved model_comparison_chart.png")

# Null reference bar
fig, ax = plt.subplots(figsize=(10, 6))
bar_c = [GOLD if m == res_df.iloc[0]["Model"] else
         ("#cccccc" if m == "Null(mean)" else NAVY)
         for m in res_df["Model"]]
ax.barh(res_df["Model"], res_df["RMSE"], color=bar_c, alpha=0.85)
null_val = res_df.loc[res_df["Model"]=="Null(mean)", "RMSE"].values[0]
ax.axvline(null_val, color="red", linestyle="--", lw=1.5, label="Null RMSE")
ax.invert_yaxis(); ax.set_xlabel("RMSE (log €/sqm)")
ax.set_title("All Models vs Null Baseline", fontsize=12, fontweight="bold")
ax.legend(); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/model_comparison_vs_null.png"); plt.close()

# RMSE on back-transformed scale
fig, ax = plt.subplots(figsize=(10, 6))
res_df_bt = res_df[res_df["Model"] != "Null(mean)"].copy()
ax.barh(res_df_bt["Model"], res_df_bt["RMSE_ppsm"], color=NAVY, alpha=0.80)
ax.invert_yaxis(); ax.set_xlabel("RMSE (€/sqm — back-transformed)")
ax.set_title("RMSE on Price/sqm Scale (back-transformed)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/model_comparison_backtransformed.png"); plt.close()
print("    Saved model_comparison_backtransformed.png")

# ── 18  ACTUAL VS PREDICTED — BEST MODEL ─────────────────────
best_name = res_df.iloc[0]["Model"]
pred_map  = {
    "Null(mean)": null_pred, "OLS": ols_pred,
    "Ridge": ridge_pred, "LASSO": lasso_pred,
    "ElasticNet": en_pred, "PCR": pcr_pred,
    "PLS": pls_pred, "SVR-Linear": svr_pred,
    "RandomForest": rf_pred, "GBM": gbm_pred, "XGBoost": xgb_pred,
}
best_pred = pred_map[best_name]
si = np.random.choice(len(y_te), min(6000, len(y_te)), replace=False)

print(f"\n[18] Actual vs Predicted — Best model: {best_name}")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# Log scale
axes[0].scatter(y_te.iloc[si], best_pred[si], alpha=0.25, s=6, color=NAVY)
lims = [y_te.quantile(0.01), y_te.quantile(0.99)]
axes[0].plot(lims, lims, "--", color=GOLD, lw=2, label="Perfect prediction")
axes[0].set_xlabel("Actual log_price_per_sqm"); axes[0].set_ylabel("Predicted")
axes[0].set_title(f"Actual vs Predicted (log scale) — {best_name}"); axes[0].legend()
# Back-transformed
y_te_bt   = np.exp(y_te.values)
best_bt   = np.exp(best_pred)
axes[1].scatter(y_te_bt[si], best_bt[si], alpha=0.25, s=6, color=GREEN)
lims_bt = [np.exp(y_te.quantile(0.01)), min(np.exp(y_te.quantile(0.98)), 80)]
axes[1].plot(lims_bt, lims_bt, "--", color="red", lw=2, label="Perfect prediction")
axes[1].set_xlabel("Actual price/sqm (€)"); axes[1].set_ylabel("Predicted price/sqm (€)")
axes[1].set_title(f"Actual vs Predicted (€/sqm back-transformed) — {best_name}")
axes[1].legend()
fig.suptitle(f"Best Model: {best_name}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/actual_vs_predicted_best.png"); plt.close()
print("    Saved actual_vs_predicted_best.png")

# Residuals
resid = y_te.values - best_pred
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].scatter(best_pred[si], resid[si], alpha=0.2, s=5, color=NAVY)
axes[0].axhline(0, color="red", linestyle="--", lw=1.5)
axes[0].set_xlabel("Fitted"); axes[0].set_ylabel("Residual")
axes[0].set_title(f"Residuals vs Fitted — {best_name}")
axes[1].hist(resid, bins=80, color=NAVY, alpha=0.75, edgecolor="white", density=True)
from scipy import stats
xr = np.linspace(resid.min(), resid.max(), 300)
axes[1].plot(xr, stats.norm.pdf(xr, resid.mean(), resid.std()),
             color="red", linestyle="--", lw=2, label="Normal PDF")
axes[1].set_xlabel("Residual"); axes[1].set_title("Residual Distribution")
axes[1].legend()
fig.suptitle(f"Residual Diagnostics — {best_name}", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/residual_diagnostics.png"); plt.close()
print("    Saved residual_diagnostics.png")

# All-model residual KDE
fig, ax = plt.subplots(figsize=(12, 6))
for nm, pred in pred_map.items():
    if nm == "Null(mean)": continue
    sns.kdeplot(y_te.values - pred, ax=ax, label=nm, fill=False, linewidth=1.2)
ax.axvline(0, color="black", lw=1.5, linestyle="--")
ax.set_xlim(-1.5, 1.5); ax.set_xlabel("Residual (log scale)")
ax.set_title("Residual Distributions — All Models", fontsize=12, fontweight="bold")
ax.legend(fontsize=8); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/residual_distributions_all.png"); plt.close()
print("    Saved residual_distributions_all.png")

# ── 19  SHAP — XGBoost ────────────────────────────────────────
print("\n[19] SHAP interpretation (XGBoost) …")
BG_N = min(500,  len(X_tr_p))
SH_N = min(2000, len(X_te_p))
bg_i = np.random.choice(len(X_tr_p), BG_N, replace=False)
sh_i = np.random.choice(len(X_te_p), SH_N, replace=False)

explainer   = shap.TreeExplainer(xgb_model, data=X_tr_p[bg_i],
                                  feature_names=feat_names)
shap_values = explainer(X_te_p[sh_i])

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_te_p[sh_i],
    feature_names=feat_names, show=False, max_display=20)
plt.title("SHAP Summary — XGBoost (log_price_per_sqm)",
          fontsize=13, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/shap_summary.png"); plt.close()
print("    Saved shap_summary.png")

plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_te_p[sh_i],
    feature_names=feat_names, plot_type="bar", show=False, max_display=20)
plt.title("SHAP Mean |Value| — XGBoost (log_price_per_sqm)",
          fontsize=12, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/shap_bar.png"); plt.close()
print("    Saved shap_bar.png")

mean_shap = pd.DataFrame({
    "Feature":     feat_names,
    "MeanAbsSHAP": np.abs(shap_values.values).mean(axis=0)
}).sort_values("MeanAbsSHAP", ascending=False)
mean_shap.to_csv(f"{OUTPUT_DIR}/shap_feature_importance.csv", index=False)
print("\n    Top 15 features by mean |SHAP|:")
print(mean_shap.head(15).to_string(index=False))

# Dependence plots — top 2 features
for top_feat in mean_shap.head(2)["Feature"].tolist():
    try:
        fi = feat_names.index(top_feat)
        plt.figure(figsize=(9, 5))
        shap.dependence_plot(fi, shap_values.values, X_te_p[sh_i],
                             feature_names=feat_names, show=False)
        safe = top_feat.replace("/","_").replace(" ","_")[:40]
        plt.title(f"SHAP Dependence — {top_feat}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/shap_dependence_{safe}.png"); plt.close()
        print(f"    Saved shap_dependence_{safe}.png")
    except Exception as e:
        print(f"    (Dependence skipped for {top_feat}: {e})")

# ── 20  OPTIONAL: PRICE TIER CLASSIFICATION ───────────────────
print("\n" + "=" * 65)
print("  OPTIONAL: Price Tier Classification (on log_price_per_sqm)")
print("=" * 65)
# Use log-scale tertiles so tiers are meaningful
q33 = y.quantile(0.33)
q67 = y.quantile(0.67)

def price_tier(x):
    if x <= q33: return 0    # Low
    if x <= q67: return 1    # Mid
    return 2                  # High

y_tier = y.map(price_tier)
print(f"    Log-scale thresholds: Low≤{q33:.3f}  Mid≤{q67:.3f}  High>{q67:.3f}")
print(f"    Back-transformed: Low≤€{np.exp(q33):.2f}/sqm  "
      f"Mid≤€{np.exp(q67):.2f}/sqm  High>€{np.exp(q67):.2f}/sqm")
print(y_tier.map({0:"Low",1:"Mid",2:"High"}).value_counts().to_string())

Xt_tr, Xt_te, yt_tr, yt_te = train_test_split(
    X, y_tier, test_size=1-TRAIN_SIZE,
    random_state=RANDOM_SEED, stratify=y_tier)
Xt_tr_p = preprocessor.transform(Xt_tr)
Xt_te_p = preprocessor.transform(Xt_te)

clf = xgb.XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_SEED, n_jobs=N_JOBS,
    eval_metric="mlogloss", verbosity=0)
clf.fit(Xt_tr_p, yt_tr)
clf_pred = clf.predict(Xt_te_p)
print("\n    Classification Report:")
print(classification_report(yt_te, clf_pred,
      target_names=["Low","Mid","High"]))

cm_arr = confusion_matrix(yt_te, clf_pred, labels=[0,1,2])
cm_df  = pd.DataFrame(cm_arr, index=["Low","Mid","High"],
                      columns=["Low","Mid","High"])
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax, linewidths=0.5)
ax.set_title("Confusion Matrix — Price Tier (XGBoost)")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_tier.png"); plt.close()
print("    Saved confusion_matrix_tier.png")

# ── 21  FINAL SUMMARY ─────────────────────────────────────────
print(f"\n  Target: log_price_per_sqm  (Phase 1 OLS benchmark R²=0.752)")
print(f"  Best model: {res_df.iloc[0]['Model']}")
print(f"  Best RMSE (log): {res_df.iloc[0]['RMSE']}")
print(f"  Best R²:         {res_df.iloc[0]['R2']}")
print(f"  Best RMSE (€/sqm back-transformed): {res_df.iloc[0]['RMSE_ppsm']}")
print(f"\n  Full results:")
print(res_df.to_string(index=False))
print(f"\n  All outputs saved to: ./{OUTPUT_DIR}/")
