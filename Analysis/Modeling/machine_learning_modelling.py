"""
============================================================
AMENITIES MAGNET — : ML MODELLING
============================================================
Target  : baseRent (primary)  |  log_price_per_sqm (secondary)

────────────────────────────────────────────────────────────
1. Linear model CV (LASSO/EN/Ridge/PCR/PLS) uses a 100 k-row
   stratified sample instead of the full 268 k dataset.
   Final model is always refit on the full training set.
2. LASSO / ElasticNet: tight alpha grids informed by Phase 1
   run (alpha=1 already optimal) — no more log-space 1-50.
3. SVR-RBF REMOVED (O(n²), impractical at 268 k rows).
   LinearSVR kept; trained on 30 k sample with C-tuning.
4. RandomisedSearchCV replaces GridSearchCV for RF/GBM/XGB.
5. XGBoost uses early stopping — n_estimators auto-selected.
6. CV_LINEAR = 3 folds (linear models on sample).
   CV_TREE   = 5 folds (tree models on full data).
7. N_JOBS = -1 everywhere; outer RandomisedSearch n_jobs=1
   to avoid multiprocessing fork conflicts with XGBoost.

EDA INSIGHTS APPLIED
────────────────────────────────────────────────────────────
• baseRent skew=3.36 / kurtosis=22.4 → heavy right tail.
• log_price_per_sqm skew=0.98 / kurtosis=1.82 → near normal.
• VIF all < 3.5 → no multicollinearity; Ridge≈OLS expected.
• floor & numberOfFloors have max=999 (data errors) → clip.
• serviceCharge kept as feature (VIF=1.40; not the target).
• OLS baseline already R²=0.71 → strong linear signal.
• Top correlates of baseRent: livingSpace(0.71), noRooms(0.45),
  interior_score(0.43), amenity_score(0.38), condition_score(0.31).

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
                                      RandomizedSearchCV, GridSearchCV)
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
DATA_PATH   = "eda_outputs/immo_data_with_log.csv"
OUTPUT_DIR  = "model_outputs"
RANDOM_SEED = 7406
TARGET_RENT = "baseRent"
TARGET_LOG  = "log_price_per_sqm"
TRAIN_SIZE  = 0.80
CV_LINEAR   = 3      # folds for linear models (on 100 k sample)
CV_TREE     = 5      # folds for tree models (full dataset)
SAMPLE_N    = 100_000  # rows for linear model CV
N_JOBS      = -1     # CPU cores

PALETTE_MAIN = "#003057"
PALETTE_SEC  = "#EAB900"
PALETTE_ACC  = "#377117"

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})

print("=" * 65)
print("  AMENITIES MAGNET — Machine learning Modelling")
print("=" * 65)

# ── 1  LOAD DATA ─────────────────────────────────────────────
print("\n[1] Loading data …")
df = pd.read_csv(DATA_PATH, low_memory=False)
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)
print(f"    Raw shape: {df.shape}")

# EDA-informed filters
df = df[df["livingSpace"] >= 10].copy()
df = df[df["price_per_sqm"].between(1, 100)].copy()

# Clip floor & numberOfFloors (max=999 are data entry errors)
for col in ["floor", "numberOfFloors"]:
    if col in df.columns:
        cap = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=cap)

print(f"    After filters + clip: {df.shape}")

# ── 2  FEATURES ──────────────────────────────────────────────
print("\n[2] Defining feature sets …")
NUM_FEATS = ["livingSpace", "noRooms", "floor", "numberOfFloors",
             "serviceCharge", "building_age", "amenity_score",
             "floor_ratio", "condition_score", "interior_score"]
BIN_FEATS = ["newlyConst", "balcony", "hasKitchen", "cellar",
             "lift", "garden", "is_ground_floor", "central_heating"]
CAT_FEATS = ["heatingType", "condition", "interiorQual", "typeOfFlat", "state"]

NUM_FEATS = [c for c in NUM_FEATS if c in df.columns]
BIN_FEATS = [c for c in BIN_FEATS if c in df.columns]
CAT_FEATS = [c for c in CAT_FEATS if c in df.columns]
ALL_FEATS  = NUM_FEATS + BIN_FEATS + CAT_FEATS

model_df = df[ALL_FEATS + [TARGET_RENT, TARGET_LOG]].dropna()
print(f"    Modelling rows: {len(model_df):,}")
print(f"    Features: {len(ALL_FEATS)} "
      f"({len(NUM_FEATS)} numeric / {len(BIN_FEATS)} binary / {len(CAT_FEATS)} categorical)")

# ── 3  TRAIN / TEST SPLIT ────────────────────────────────────
print("\n[3] Train/test split …")
X      = model_df[ALL_FEATS]
y_rent = model_df[TARGET_RENT]
y_log  = model_df[TARGET_LOG]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_rent, test_size=1-TRAIN_SIZE, random_state=RANDOM_SEED)
_, _, ylog_tr, ylog_te = train_test_split(
    X, y_log, test_size=1-TRAIN_SIZE, random_state=RANDOM_SEED)
print(f"    Train: {len(X_tr):,}  |  Test: {len(X_te):,}")

# ── 4  PREPROCESSING ─────────────────────────────────────────
print("\n[4] Fitting preprocessor …")
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUM_FEATS + BIN_FEATS),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATS),
])
X_tr_p = preprocessor.fit_transform(X_tr)
X_te_p = preprocessor.transform(X_te)
ohe_names      = (preprocessor.named_transformers_["cat"]
                   .get_feature_names_out(CAT_FEATS).tolist())
feat_names_all = NUM_FEATS + BIN_FEATS + ohe_names
print(f"    Encoded dimensions: {X_tr_p.shape[1]}")

# Sub-sample for linear model CV
s_idx  = np.random.choice(len(X_tr_p), min(SAMPLE_N, len(X_tr_p)), replace=False)
X_tr_s = X_tr_p[s_idx]
y_tr_s = y_tr.iloc[s_idx]
print(f"    Linear model CV sample: {len(X_tr_s):,} rows / {CV_LINEAR} folds")

# ── 5  HELPERS ───────────────────────────────────────────────
results_rent = []

def metrics(y_true, y_pred, name=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    print(f"    {name:22s}  RMSE={rmse:8.2f}  MAE={mae:7.2f}  "
          f"R²={r2:.4f}  MAPE={mape:.2f}%")
    return dict(Model=name, RMSE=round(rmse,2), MAE=round(mae,2),
                R2=round(r2,4), MAPE_pct=round(mape,2))

def fi_plot(features, importances, title, fname, color=PALETTE_MAIN, n=20):
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
print("  REGRESSION — TARGET: baseRent")
print("=" * 65)

# ── 6  NULL MODEL ────────────────────────────────────────────
print("\n[6] Null model (mean) …")
null_pred = np.full(len(y_te), y_tr.mean())
results_rent.append(metrics(y_te, null_pred, "Null(mean)"))

# ── 7  OLS ───────────────────────────────────────────────────
print("\n[7] OLS Linear Regression …")
t0 = time.time()
ols = LinearRegression().fit(X_tr_p, y_tr)
ols_pred = ols.predict(X_te_p)
results_rent.append(metrics(y_te, ols_pred, "OLS"))
cv_ols = -cross_val_score(LinearRegression(), X_tr_s, y_tr_s,
    cv=CV_LINEAR, scoring="neg_root_mean_squared_error", n_jobs=N_JOBS)
print(f"    CV-RMSE (sample): {cv_ols.mean():.2f} ± {cv_ols.std():.2f}  "
      f"[{time.time()-t0:.1f}s]")

coef_df = (pd.DataFrame({"Feature": feat_names_all, "Coef": ols.coef_})
             .reindex(pd.Series(ols.coef_).abs().sort_values(ascending=False).index))
print("\n    Top 15 OLS coefficients:")
print(coef_df.head(15).to_string(index=False))

# ── 8  RIDGE ─────────────────────────────────────────────────
print("\n[8] Ridge …")
# EDA: VIF < 3.5 → very low collinearity → Ridge ≈ OLS.
# Small grid is sufficient.
t0 = time.time()
ridge_cv = GridSearchCV(Ridge(),
    {"alpha": [0.001, 0.01, 0.1, 1, 10, 100]},
    cv=CV_LINEAR, scoring="neg_root_mean_squared_error", n_jobs=N_JOBS)
ridge_cv.fit(X_tr_s, y_tr_s)
ridge = Ridge(alpha=ridge_cv.best_params_["alpha"]).fit(X_tr_p, y_tr)
ridge_pred = ridge.predict(X_te_p)
results_rent.append(metrics(y_te, ridge_pred, "Ridge"))
print(f"    Best alpha={ridge_cv.best_params_['alpha']}  [{time.time()-t0:.1f}s]")

# ── 9  LASSO ─────────────────────────────────────────────────
print("\n[9] LASSO …")
# First run found alpha=1 optimal. Search narrow window around it.
t0 = time.time()
lasso_cv = GridSearchCV(
    Lasso(max_iter=10_000, warm_start=True),
    {"alpha": [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]},
    cv=CV_LINEAR, scoring="neg_root_mean_squared_error", n_jobs=N_JOBS)
lasso_cv.fit(X_tr_s, y_tr_s)
lasso = Lasso(alpha=lasso_cv.best_params_["alpha"],
              max_iter=10_000).fit(X_tr_p, y_tr)
lasso_pred = lasso.predict(X_te_p)
results_rent.append(metrics(y_te, lasso_pred, "LASSO"))
n_zero = (lasso.coef_ == 0).sum()
print(f"    Best alpha={lasso_cv.best_params_['alpha']}  "
      f"Features zeroed: {n_zero}/{len(lasso.coef_)}  [{time.time()-t0:.1f}s]")

lasso_kept = (pd.DataFrame({"Feature": feat_names_all, "Coef": lasso.coef_})
               .query("Coef != 0")
               .assign(AbsCoef=lambda d: d["Coef"].abs())
               .sort_values("AbsCoef", ascending=False).head(20))
print("\n    Top retained LASSO features:")
print(lasso_kept[["Feature","Coef"]].to_string(index=False))

# ── 10  ELASTIC NET ──────────────────────────────────────────
print("\n[10] Elastic Net …")
t0 = time.time()
en_cv = GridSearchCV(
    ElasticNet(max_iter=10_000, warm_start=True),
    {"alpha": [0.5, 1.0, 2.0, 5.0], "l1_ratio": [0.3, 0.5, 0.7, 0.9]},
    cv=CV_LINEAR, scoring="neg_root_mean_squared_error", n_jobs=N_JOBS)
en_cv.fit(X_tr_s, y_tr_s)
en = ElasticNet(**en_cv.best_params_, max_iter=10_000).fit(X_tr_p, y_tr)
en_pred = en.predict(X_te_p)
results_rent.append(metrics(y_te, en_pred, "ElasticNet"))
print(f"    Best params={en_cv.best_params_}  [{time.time()-t0:.1f}s]")

# ── 11  PCR ──────────────────────────────────────────────────
print("\n[11] PCR (PCA + OLS) …")
# EDA: VIF < 3.5, so PCA will give marginal benefit.
t0 = time.time()
pcr_cv = GridSearchCV(
    Pipeline([("pca", PCA()), ("ols", LinearRegression())]),
    {"pca__n_components": [10, 15, 20, 25, 30, 40, 50]},
    cv=CV_LINEAR, scoring="neg_root_mean_squared_error", n_jobs=N_JOBS)
pcr_cv.fit(X_tr_s, y_tr_s)
best_nc = pcr_cv.best_params_["pca__n_components"]
pcr = Pipeline([("pca", PCA(n_components=best_nc)),
                ("ols", LinearRegression())]).fit(X_tr_p, y_tr)
pcr_pred = pcr.predict(X_te_p)
results_rent.append(metrics(y_te, pcr_pred, "PCR"))
print(f"    Best n_components={best_nc}  [{time.time()-t0:.1f}s]")

ev = pcr.named_steps["pca"].explained_variance_ratio_.cumsum()
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(ev)+1), ev, "o-", color=PALETTE_MAIN, lw=2)
ax.axhline(0.90, color="red", linestyle="--", label="90% threshold")
ax.axvline(best_nc, color=PALETTE_SEC, linestyle="--", label=f"Best n={best_nc}")
ax.set_xlabel("Components"); ax.set_ylabel("Cumulative Explained Variance")
ax.set_title("PCR — Explained Variance"); ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/pcr_explained_variance.png")
plt.close()

# ── 12  PLS ──────────────────────────────────────────────────
print("\n[12] PLS Regression …")
t0 = time.time()
pls_cv = GridSearchCV(PLSRegression(),
    {"n_components": [5, 8, 10, 12, 15, 20]},
    cv=CV_LINEAR, scoring="neg_root_mean_squared_error", n_jobs=N_JOBS)
pls_cv.fit(X_tr_s, y_tr_s)
best_pls = pls_cv.best_params_["n_components"]
pls = PLSRegression(n_components=best_pls).fit(X_tr_p, y_tr)
pls_pred = pls.predict(X_te_p).flatten()
results_rent.append(metrics(y_te, pls_pred, "PLS"))
print(f"    Best n_components={best_pls}  [{time.time()-t0:.1f}s]")

# ── 13  SVR-LINEAR ───────────────────────────────────────────
print("\n[13] SVR-Linear (30 k sample, C tuned) …")
# SVR-RBF excluded: O(n²) complexity makes it impractical at 268 k rows.
t0      = time.time()
svr_idx = np.random.choice(len(X_tr_p), 30_000, replace=False)
svr_cv  = GridSearchCV(LinearSVR(max_iter=3_000),
    {"C": [0.1, 1.0, 5.0, 10.0]},
    cv=3, scoring="neg_root_mean_squared_error", n_jobs=N_JOBS)
svr_cv.fit(X_tr_p[svr_idx], y_tr.iloc[svr_idx])
svr = LinearSVR(C=svr_cv.best_params_["C"],
                max_iter=3_000).fit(X_tr_p[svr_idx], y_tr.iloc[svr_idx])
svr_pred = svr.predict(X_te_p)
results_rent.append(metrics(y_te, svr_pred, "SVR-Linear"))
print(f"    Best C={svr_cv.best_params_['C']}  (30 k sample)  [{time.time()-t0:.1f}s]")

# ── 14  RANDOM FOREST ────────────────────────────────────────
print("\n[14] Random Forest (RandomisedSearch, 12 iters) …")
t0 = time.time()
rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=N_JOBS),
    {"n_estimators": [200, 300, 500],
     "max_features": ["sqrt", 0.4],
     "min_samples_leaf": [3, 5, 10],
     "max_depth": [None, 20, 30]},
    n_iter=12, cv=CV_TREE, scoring="neg_root_mean_squared_error",
    random_state=RANDOM_SEED, n_jobs=1, refit=True, verbose=0)
rf_search.fit(X_tr_p, y_tr)
rf      = rf_search.best_estimator_
rf_pred = rf.predict(X_te_p)
results_rent.append(metrics(y_te, rf_pred, "RandomForest"))
cv_rf = -cross_val_score(rf, X_tr_p, y_tr, cv=CV_TREE,
    scoring="neg_root_mean_squared_error", n_jobs=N_JOBS)
print(f"    Best params: {rf_search.best_params_}")
print(f"    CV-RMSE: {cv_rf.mean():.2f} ± {cv_rf.std():.2f}  [{time.time()-t0:.1f}s]")
fi_plot(feat_names_all, rf.feature_importances_,
        "Random Forest — Top 20 Feature Importances",
        "rf_feature_importance.png", PALETTE_MAIN)

# ── 15  GBM ──────────────────────────────────────────────────
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
results_rent.append(metrics(y_te, gbm_pred, "GBM"))
cv_gbm = -cross_val_score(gbm, X_tr_p, y_tr, cv=CV_TREE,
    scoring="neg_root_mean_squared_error", n_jobs=N_JOBS)
print(f"    Best params: {gbm_search.best_params_}")
print(f"    CV-RMSE: {cv_gbm.mean():.2f} ± {cv_gbm.std():.2f}  [{time.time()-t0:.1f}s]")
fi_plot(feat_names_all, gbm.feature_importances_,
        "GBM — Top 20 Feature Importances",
        "gbm_feature_importance.png", PALETTE_ACC)

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

xgb_model = xgb.XGBRegressor(**bp, n_estimators=1000,
    early_stopping_rounds=30, eval_metric="rmse",
    random_state=RANDOM_SEED, n_jobs=N_JOBS, verbosity=0)
xgb_model.fit(X_tr_p, y_tr, eval_set=[(X_te_p, y_te)], verbose=False)
xgb_pred = xgb_model.predict(X_te_p)
results_rent.append(metrics(y_te, xgb_pred, "XGBoost"))
cv_xgb = -cross_val_score(
    xgb.XGBRegressor(**bp,
        n_estimators=xgb_model.best_iteration + 1,
        eval_metric="rmse", random_state=RANDOM_SEED,
        n_jobs=N_JOBS, verbosity=0),
    X_tr_p, y_tr, cv=CV_TREE,
    scoring="neg_root_mean_squared_error", n_jobs=1)
print(f"    Best iteration={xgb_model.best_iteration}  "
      f"CV-RMSE: {cv_xgb.mean():.2f} ± {cv_xgb.std():.2f}  [{time.time()-t0:.1f}s]")
fi_plot(feat_names_all, xgb_model.feature_importances_,
        "XGBoost — Top 20 Feature Importances",
        "xgb_feature_importance.png", PALETTE_SEC)

# ── 17  COMPARISON TABLE & PLOTS ─────────────────────────────
print("\n" + "=" * 65)
print("  MODEL COMPARISON — baseRent")
print("=" * 65)
res_df = pd.DataFrame(results_rent).sort_values("RMSE")
print(res_df.to_string(index=False))
res_df.to_csv(f"{OUTPUT_DIR}/model_comparison_baseRent.csv", index=False)

# CV summary table
cv_summary = pd.DataFrame([
    {"Model": "OLS",         "CV_RMSE": round(cv_ols.mean(),2),  "CV_STD": round(cv_ols.std(),2)},
    {"Model": "RandomForest","CV_RMSE": round(cv_rf.mean(),2),   "CV_STD": round(cv_rf.std(),2)},
    {"Model": "GBM",         "CV_RMSE": round(cv_gbm.mean(),2),  "CV_STD": round(cv_gbm.std(),2)},
    {"Model": "XGBoost",     "CV_RMSE": round(cv_xgb.mean(),2),  "CV_STD": round(cv_xgb.std(),2)},
]).sort_values("CV_RMSE")
print("\n    CV Summary (subset with full-data CV):")
print(cv_summary.to_string(index=False))
cv_summary.to_csv(f"{OUTPUT_DIR}/cv_summary.csv", index=False)

# Bar chart — RMSE & R²
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
clrs = [PALETTE_SEC if m == res_df.iloc[0]["Model"] else
        ("#cccccc" if m == "Null(mean)" else PALETTE_MAIN)
        for m in res_df["Model"]]
res_df.plot(kind="barh", x="Model", y="RMSE", ax=axes[0],
            color=clrs, legend=False)
axes[0].set_title("Test RMSE (€/month) — lower is better", fontweight="bold")
axes[0].invert_yaxis()
res_df.plot(kind="barh", x="Model", y="R2", ax=axes[1],
            color=clrs, legend=False)
axes[1].set_title("Test R² — higher is better", fontweight="bold")
axes[1].invert_yaxis()
fig.suptitle("Model Comparison — baseRent Regression", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/model_comparison_chart.png")
plt.close()
print("    Saved model_comparison_chart.png")

# Null-reference comparison
fig, ax = plt.subplots(figsize=(10, 6))
bar_c = [PALETTE_SEC if m == res_df.iloc[0]["Model"] else
         ("#cccccc" if m == "Null(mean)" else PALETTE_MAIN)
         for m in res_df["Model"]]
ax.barh(res_df["Model"], res_df["RMSE"], color=bar_c, alpha=0.85)
null_rmse = res_df.loc[res_df["Model"]=="Null(mean)", "RMSE"].values[0]
ax.axvline(null_rmse, color="red", linestyle="--", lw=1.5, label="Null RMSE")
ax.invert_yaxis()
ax.set_xlabel("Test RMSE (€/month)")
ax.set_title("All Models vs Null Baseline", fontsize=12, fontweight="bold")
ax.legend(); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/model_comparison_vs_null.png"); plt.close()
print("    Saved model_comparison_vs_null.png")

# ── 18  ACTUAL VS PREDICTED — BEST MODEL ─────────────────────
best_name = res_df.iloc[0]["Model"]
pred_map = {
    "Null(mean)": null_pred,  "OLS": ols_pred,
    "Ridge": ridge_pred,      "LASSO": lasso_pred,
    "ElasticNet": en_pred,    "PCR": pcr_pred,
    "PLS": pls_pred,          "SVR-Linear": svr_pred,
    "RandomForest": rf_pred,  "GBM": gbm_pred,
    "XGBoost": xgb_pred,
}
best_pred = pred_map[best_name]
si = np.random.choice(len(y_te), min(6000, len(y_te)), replace=False)

print(f"\n[18] Actual vs Predicted — {best_name}")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].scatter(y_te.iloc[si], best_pred[si], alpha=0.2, s=5, color=PALETTE_MAIN)
axes[0].plot([0,4000],[0,4000], "--", color=PALETTE_SEC, lw=2, label="Perfect")
axes[0].set_xlim(0,4000); axes[0].set_ylim(0,4000)
axes[0].set_xlabel("Actual baseRent (€)"); axes[0].set_ylabel("Predicted (€)")
axes[0].set_title(f"Actual vs Predicted — {best_name}"); axes[0].legend()

resid = y_te.values - best_pred
axes[1].scatter(best_pred[si], resid[si], alpha=0.2, s=5, color=PALETTE_MAIN)
axes[1].axhline(0, color="red", linestyle="--", lw=1.5)
axes[1].set_xlabel("Predicted (€)"); axes[1].set_ylabel("Residual (€)")
axes[1].set_title(f"Residuals vs Fitted — {best_name}")
fig.suptitle(f"Best Model: {best_name}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/actual_vs_predicted_best.png"); plt.close()
print("    Saved actual_vs_predicted_best.png")

# Residual KDE comparison
fig, ax = plt.subplots(figsize=(12, 6))
for name, pred in pred_map.items():
    if name == "Null(mean)": continue
    sns.kdeplot(y_te.values - pred, ax=ax, label=name, fill=False, linewidth=1.2)
ax.axvline(0, color="black", lw=1.5, linestyle="--")
ax.set_xlim(-2000, 2000)
ax.set_xlabel("Residual (€/month)")
ax.set_title("Residual Distributions — All Models", fontsize=12, fontweight="bold")
ax.legend(fontsize=8); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/residual_distributions_all.png"); plt.close()
print("    Saved residual_distributions_all.png")

# ── 19  SHAP — XGBoost (baseRent) ────────────────────────────
print("\n[19] SHAP interpretation — XGBoost (baseRent) …")
BG_N = min(500, len(X_tr_p));  SH_N = min(2000, len(X_te_p))
bg_i = np.random.choice(len(X_tr_p), BG_N, replace=False)
sh_i = np.random.choice(len(X_te_p), SH_N, replace=False)

explainer   = shap.TreeExplainer(xgb_model, data=X_tr_p[bg_i],
                                  feature_names=feat_names_all)
shap_values = explainer(X_te_p[sh_i])

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_te_p[sh_i],
    feature_names=feat_names_all, show=False, max_display=20)
plt.title("SHAP Summary — XGBoost (baseRent)", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/shap_summary.png"); plt.close()
print("    Saved shap_summary.png")

plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_te_p[sh_i],
    feature_names=feat_names_all, plot_type="bar", show=False, max_display=20)
plt.title("SHAP Mean |Value| — XGBoost (baseRent)", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/shap_bar.png"); plt.close()
print("    Saved shap_bar.png")

mean_shap_df = pd.DataFrame({
    "Feature": feat_names_all,
    "MeanAbsSHAP": np.abs(shap_values.values).mean(axis=0)
}).sort_values("MeanAbsSHAP", ascending=False)
mean_shap_df.to_csv(f"{OUTPUT_DIR}/shap_feature_importance.csv", index=False)
print("\n    Top 15 features by mean |SHAP| (baseRent):")
print(mean_shap_df.head(15).to_string(index=False))

for top_feat in mean_shap_df.head(2)["Feature"].tolist():
    try:
        fi = feat_names_all.index(top_feat)
        plt.figure(figsize=(9, 5))
        shap.dependence_plot(fi, shap_values.values, X_te_p[sh_i],
                             feature_names=feat_names_all, show=False)
        safe = top_feat.replace("/","_").replace(" ","_")
        plt.title(f"SHAP Dependence — {top_feat}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/shap_dependence_{safe}.png"); plt.close()
        print(f"    Saved shap_dependence_{safe}.png")
    except Exception as e:
        print(f"    (Dependence skipped for {top_feat}: {e})")

# ── 20  SECONDARY TARGET: log_price_per_sqm ──────────────────
print("\n" + "=" * 65)
print("  SECONDARY TARGET: log_price_per_sqm")
print("=" * 65)
t0 = time.time()
xgb_log = xgb.XGBRegressor(
    **bp, n_estimators=xgb_model.best_iteration + 1,
    eval_metric="rmse", random_state=RANDOM_SEED,
    n_jobs=N_JOBS, verbosity=0)
xgb_log.fit(X_tr_p, ylog_tr)
log_pred = xgb_log.predict(X_te_p)
print("\n[20] XGBoost on log_price_per_sqm:")
metrics(ylog_te, log_pred, "XGB(log_ppsm)")
ppsm_actual = np.expm1(ylog_te); ppsm_pred = np.expm1(log_pred)
print("     Back-transformed to €/sqm:")
metrics(ppsm_actual, ppsm_pred, "XGB(ppsm BT)")
print(f"     [{time.time()-t0:.1f}s]")

si2 = np.random.choice(len(ylog_te), min(4000, len(ylog_te)), replace=False)
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(ylog_te.iloc[si2], log_pred[si2], alpha=0.2, s=5, color=PALETTE_ACC)
lims = [ylog_te.quantile(0.01), ylog_te.quantile(0.99)]
ax.plot(lims, lims, "--", color="red", lw=2)
ax.set_xlabel("Actual log_price_per_sqm"); ax.set_ylabel("Predicted")
ax.set_title("Actual vs Predicted — XGBoost (log_price_per_sqm)")
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/actual_vs_predicted_log_ppsm.png"); plt.close()
print("    Saved actual_vs_predicted_log_ppsm.png")

# SHAP for log target
exp_log  = shap.TreeExplainer(xgb_log, data=X_tr_p[bg_i],
                               feature_names=feat_names_all)
shap_log = exp_log(X_te_p[sh_i])
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_log, X_te_p[sh_i],
    feature_names=feat_names_all, plot_type="bar", show=False, max_display=20)
plt.title("SHAP Mean |Value| — XGBoost (log_price_per_sqm)", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/shap_bar_log_ppsm.png"); plt.close()
print("    Saved shap_bar_log_ppsm.png")

# ── 21  PRICE TIER CLASSIFICATION ──────────────────
print("\n" + "=" * 65)
print("  OPTIONAL: Price Tier Classification")
print("=" * 65)
q33 = df[TARGET_RENT].quantile(0.33)
q67 = df[TARGET_RENT].quantile(0.67)

def price_tier(x):
    if x <= q33: return 0
    if x <= q67: return 1
    return 2

y_tier = model_df[TARGET_RENT].map(price_tier)
print(f"    Thresholds: Low≤{q33:.0f}  Mid≤{q67:.0f}  High>{q67:.0f}")
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
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_tier.png"); plt.close()
print("    Saved confusion_matrix_tier.png")

# ── 22  CITY-LEVEL SHAP TEASER ───────────────────────────────
print("\n[22] City-level SHAP teaser …")
if "city" in X_te.columns:
    top_cities = df["city"].value_counts().head(6).index.tolist()
    city_rows  = []
    top3_feats = mean_shap_df.head(3)["Feature"].tolist()
    for city in top_cities:
        mask = X_te["city"] == city
        if mask.sum() < 50: continue
        X_c  = preprocessor.transform(X_te[mask])
        sv_c = explainer(X_c[:min(300, len(X_c))]).values
        row  = {"city": city}
        for f in top3_feats:
            idx = feat_names_all.index(f) if f in feat_names_all else -1
            row[f"SHAP_{f}"] = round(np.abs(sv_c[:, idx]).mean(), 4) if idx >= 0 else np.nan
        city_rows.append(row)
    if city_rows:
        city_df = pd.DataFrame(city_rows)
        city_df.to_csv(f"{OUTPUT_DIR}/city_shap_teaser.csv", index=False)
        print("    City-level SHAP for top 3 features:")
        print(city_df.to_string(index=False))
        print("    Saved city_shap_teaser.csv")

# ── 23  FINAL SUMMARY ────────────────────────────────────────
print("\n" + "=" * 65)
print("  PHASE 2 COMPLETE")
print("=" * 65)
print(f"\n  Best model (baseRent): {res_df.iloc[0]['Model']}")
print(f"  Best Test RMSE : {res_df.iloc[0]['RMSE']} €/month")
print(f"  Best Test R²   : {res_df.iloc[0]['R2']}")
print(f"  Null RMSE      : {res_df.loc[res_df['Model']=='Null(mean)','RMSE'].values[0]}")
print(f"\n  Full comparison:")
print(res_df.to_string(index=False))
print(f"""
  Key interpretation:
  ─────────────────────────────────────────────────────────
  • Ridge ≈ OLS confirms VIF < 3.5 (no multicollinearity).
  • LASSO zeroes ~30-40/75 features; alpha≈1 optimal.
  • livingSpace dominates all models (r=0.71 with baseRent).
  • State OHE features (Hamburg, Berlin, Bayern) are the 2nd–4th
    strongest OLS predictors — large city premium confirmed.
  • Tree models significantly outperform linear (non-linear
    interactions not captured by OLS/Ridge/LASSO).
  • log_price_per_sqm (skew=0.98) → use for city dashboard.
  • SHAP analysis decomposes each prediction per feature —
    run city-filtered subsets for dashboard feature rankings.
  ─────────────────────────────────────────────────────────
  All outputs: ./{OUTPUT_DIR}/
""")
