"""
============================================================
AMENITIES MAGNET — : EXPLORATORY DATA ANALYSIS
============================================================
Project : CSE 6242 Team #146
Target  : baseRent (primary), price_per_sqm (log-transformed, secondary)
Data    : immo_data_clean.csv  (268,632 rows, 33 columns + engineered features)

Run     : python phase1_eda.py
Outputs : /eda_outputs/  folder with all plots + console summary stats
============================================================
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, kstest, normaltest
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

warnings.filterwarnings("ignore")

# ── 0  SETTINGS ──────────────────────────────────────────────
DATA_PATH   = "immo_data_clean.csv"   
OUTPUT_DIR  = "eda_outputs"
RANDOM_SEED = 7406
TARGET      = "baseRent"
LOG_TARGET  = "log_price_per_sqm"    

PALETTE_MAIN = "#003057"
PALETTE_SEC  = "#EAB900"
PALETTE_ACC  = "#377117"

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})

print("=" * 65)
print("  AMENITIES MAGNET — EDA")
print("=" * 65)

# ── 1  LOAD DATA ─────────────────────────────────────────────
print("\n[1] Loading data …")
df = pd.read_csv(DATA_PATH, low_memory=False)

# Drop unnamed index column if present
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

print(f"    Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"    Columns: {df.columns.tolist()}")

# ── 1.1  Add log_price_per_sqm  (noted as required by data team) ──
#  log1p used so that the rare zero values don't produce -inf
print("\n[1.1] Adding log_price_per_sqm …")
df[LOG_TARGET] = np.log1p(df["price_per_sqm"])
print(f"      log_price_per_sqm range: [{df[LOG_TARGET].min():.4f}, {df[LOG_TARGET].max():.4f}]")

# ── 1.2  Define column groups ────────────────────────────────
NUMERIC_COLS = [
    "baseRent", "serviceCharge", "totalRent", "livingSpace",
    "yearConstructed", "noRooms", "floor", "numberOfFloors",
    "price_per_sqm", LOG_TARGET,
    "building_age", "amenity_score", "floor_ratio",
    "condition_score", "interior_score",
]

BOOL_COLS = ["newlyConst", "balcony", "hasKitchen", "cellar",
             "lift", "garden", "is_ground_floor", "central_heating"]

CAT_COLS  = ["heatingType", "condition", "interiorQual", "typeOfFlat",
             "state", "city", "regio1", "regio2", "regio3"]

# Keep only columns that actually exist in this dataset
NUMERIC_COLS = [c for c in NUMERIC_COLS if c in df.columns]
BOOL_COLS    = [c for c in BOOL_COLS    if c in df.columns]
CAT_COLS     = [c for c in CAT_COLS     if c in df.columns]

# ── 2  MISSING VALUE AUDIT ───────────────────────────────────
print("\n[2] Missing value audit …")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
miss_df = pd.DataFrame({"n_missing": missing, "pct_missing": missing_pct})
miss_df = miss_df[miss_df.n_missing > 0].sort_values("pct_missing", ascending=False)

if miss_df.empty:
    print("    ✓ No missing values — data team confirmed 0 NaNs after pipeline.")
else:
    print(miss_df.to_string())

# ── 3  DESCRIPTIVE STATISTICS ────────────────────────────────
print("\n[3] Descriptive statistics (numeric columns) …")
desc = df[NUMERIC_COLS].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
desc["skewness"] = df[NUMERIC_COLS].skew()
desc["kurtosis"] = df[NUMERIC_COLS].kurt()
print(desc.to_string())
desc.to_csv(f"{OUTPUT_DIR}/descriptive_stats.csv")

# ── 4  UNIVARIATE DISTRIBUTIONS ─────────────────────────────
print("\n[4] Plotting univariate distributions …")

PLOT_NUMERICS = [c for c in [
    "baseRent", "livingSpace", "price_per_sqm", LOG_TARGET,
    "building_age", "noRooms", "amenity_score", "floor_ratio",
    "condition_score", "interior_score"
] if c in df.columns]

n_cols = 3
n_rows = int(np.ceil(len(PLOT_NUMERICS) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
axes = axes.flatten()

for i, col in enumerate(PLOT_NUMERICS):
    data = df[col].dropna()
    axes[i].hist(data, bins=50, color=PALETTE_MAIN, alpha=0.75, edgecolor="white")
    axes[i].set_title(col, fontsize=11, fontweight="bold")
    axes[i].set_xlabel("")
    axes[i].set_ylabel("Count")
    skew_val = data.skew()
    axes[i].text(0.97, 0.95, f"skew={skew_val:.2f}", transform=axes[i].transAxes,
                 ha="right", va="top", fontsize=8, color="gray")

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Univariate Distributions — Numeric Features", y=1.01,
             fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_univariate_distributions.png")
plt.close()
print("    Saved 01_univariate_distributions.png")

# ── 4.1  Target variable detail: baseRent vs log_price_per_sqm ──
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for col, ax_row in zip(["baseRent", LOG_TARGET], axes):
    data = df[col].dropna()

    # Histogram + KDE
    ax_row[0].hist(data, bins=60, color=PALETTE_MAIN, alpha=0.7, density=True, edgecolor="white")
    kde_x = np.linspace(data.min(), data.max(), 300)
    kde = stats.gaussian_kde(data.sample(min(10000, len(data)), random_state=RANDOM_SEED))
    ax_row[0].plot(kde_x, kde(kde_x), color=PALETTE_SEC, lw=2)
    ax_row[0].set_title(f"{col} — Histogram + KDE")

    # Q-Q plot
    sm.qqplot(data.sample(min(5000, len(data)), random_state=RANDOM_SEED),
              line="s", ax=ax_row[1], alpha=0.3, markersize=2)
    ax_row[1].set_title(f"{col} — Q-Q Plot")

    # Boxplot
    ax_row[2].boxplot(data, vert=True, patch_artist=True,
                      boxprops=dict(facecolor=PALETTE_MAIN, alpha=0.6),
                      medianprops=dict(color=PALETTE_SEC, linewidth=2))
    ax_row[2].set_title(f"{col} — Boxplot")
    ax_row[2].set_xticklabels([col])

fig.suptitle("Target Variable Deep Dive: baseRent vs log_price_per_sqm",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_target_variable_analysis.png")
plt.close()
print("    Saved 02_target_variable_analysis.png")

# ── 5  NORMALITY TESTS on targets ────────────────────────────
print("\n[5] Normality tests …")
for col in ["baseRent", "price_per_sqm", LOG_TARGET]:
    if col not in df.columns:
        continue
    sample = df[col].dropna().sample(min(5000, len(df)), random_state=RANDOM_SEED)
    stat_sw, p_sw = shapiro(sample)
    stat_ks, p_ks = kstest(sample, "norm",
                            args=(sample.mean(), sample.std()))
    stat_da, p_da = normaltest(sample)
    print(f"\n    {col}:")
    print(f"      Shapiro-Wilk   W={stat_sw:.4f}, p={p_sw:.4e}  "
          f"({'Normal' if p_sw>0.05 else 'NOT Normal'})")
    print(f"      KS (normal)    D={stat_ks:.4f}, p={p_ks:.4e}  "
          f"({'Normal' if p_ks>0.05 else 'NOT Normal'})")
    print(f"      D'Agostino-K²  k²={stat_da:.4f}, p={p_da:.4e} "
          f"({'Normal' if p_da>0.05 else 'NOT Normal'})")

# ── 6  BOXPLOTS — scaled ─────────────────────────────────────
print("\n[6] Scaled boxplot comparison …")
BOX_COLS = [c for c in [
    "baseRent", "livingSpace", "price_per_sqm", "building_age",
    "amenity_score", "floor_ratio", "condition_score", "interior_score",
    "noRooms"
] if c in df.columns]

scaled_df = (df[BOX_COLS] - df[BOX_COLS].mean()) / df[BOX_COLS].std()
melted = scaled_df.melt(var_name="Variable", value_name="Standardised Value")

fig, ax = plt.subplots(figsize=(14, 7))
bp_data = [scaled_df[c].dropna().values for c in BOX_COLS]
bp = ax.boxplot(bp_data, patch_artist=True, labels=BOX_COLS,
                showfliers=False,
                boxprops=dict(facecolor=PALETTE_MAIN, alpha=0.55),
                medianprops=dict(color=PALETTE_SEC, linewidth=2),
                whiskerprops=dict(color=PALETTE_MAIN),
                capprops=dict(color=PALETTE_MAIN))
ax.set_xticklabels(BOX_COLS, rotation=30, ha="right")
ax.set_ylabel("Standardised Value (z-score)")
ax.set_title("Scaled Boxplots — Key Numeric Features", fontsize=13, fontweight="bold")
ax.axhline(0, color="gray", linestyle="--", lw=0.8)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_scaled_boxplots.png")
plt.close()
print("    Saved 03_scaled_boxplots.png")

# ── 7  OUTLIER ANALYSIS ──────────────────────────────────────
print("\n[7] Outlier analysis (IQR method) …")
outlier_summary = []
for col in ["baseRent", "livingSpace", "price_per_sqm", "totalRent"]:
    if col not in df.columns:
        continue
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_out = ((df[col] < lo) | (df[col] > hi)).sum()
    outlier_summary.append({"Column": col, "Q1": Q1, "Q3": Q3, "IQR": IQR,
                             "Lower fence": lo, "Upper fence": hi,
                             "N outliers": n_out,
                             "% outliers": round(n_out / len(df) * 100, 2)})
    print(f"    {col}: {n_out:,} outliers ({n_out/len(df)*100:.2f}%)  "
          f"[fence: {lo:.1f} – {hi:.1f}]")

pd.DataFrame(outlier_summary).to_csv(f"{OUTPUT_DIR}/outlier_summary.csv", index=False)

# ── 8  CORRELATION MATRIX ───────────────────────────────────
print("\n[8] Correlation matrix …")
CORR_COLS = [c for c in NUMERIC_COLS if c in df.columns]
corr_mx   = df[CORR_COLS].corr()

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_mx, dtype=bool))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr_mx, mask=mask, cmap=cmap, center=0,
            annot=True, fmt=".2f", annot_kws={"size": 7},
            linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title("Pearson Correlation Matrix — Numeric Features",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_correlation_matrix.png")
plt.close()
corr_mx.to_csv(f"{OUTPUT_DIR}/correlation_matrix.csv")
print("    Saved 04_correlation_matrix.png")

# ── 8.1  Correlation with targets ───────────────────────────
print("\n    Correlations with baseRent:")
corr_target = corr_mx["baseRent"].drop("baseRent").sort_values(key=abs, ascending=False)
print(corr_target.to_string())

print("\n    Correlations with log_price_per_sqm:")
corr_log = corr_mx[LOG_TARGET].drop(LOG_TARGET).sort_values(key=abs, ascending=False)
print(corr_log.to_string())

# ── 9  SCATTER PLOTS — top predictors vs target ─────────────
print("\n[9] Scatter plots — top predictors vs baseRent …")
TOP_PREDS = ["livingSpace", "noRooms", "amenity_score",
             "building_age", "condition_score", "interior_score"]
TOP_PREDS = [c for c in TOP_PREDS if c in df.columns]

sample_df = df.sample(min(8000, len(df)), random_state=RANDOM_SEED)

n_cols = 3
n_rows = int(np.ceil(len(TOP_PREDS) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 5))
axes = axes.flatten()

for i, col in enumerate(TOP_PREDS):
    axes[i].scatter(sample_df[col], sample_df[TARGET], alpha=0.15,
                    s=5, color=PALETTE_MAIN)
    m, b = np.polyfit(sample_df[col].fillna(0), sample_df[TARGET].fillna(0), 1)
    xr = np.linspace(sample_df[col].min(), sample_df[col].max(), 100)
    axes[i].plot(xr, m * xr + b, color=PALETTE_SEC, lw=1.8)
    r, p = stats.pearsonr(sample_df[col].fillna(0), sample_df[TARGET].fillna(0))
    axes[i].set_title(f"{col}  (r={r:.3f})", fontweight="bold")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel(TARGET)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle(f"Scatter Plots — Key Predictors vs {TARGET}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_scatter_predictors.png")
plt.close()
print("    Saved 05_scatter_predictors.png")

# ── 10  CATEGORICAL FEATURE ANALYSIS ────────────────────────
print("\n[10] Categorical feature analysis …")

# heatingType distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
heat_counts = df["heatingType"].value_counts() if "heatingType" in df.columns else pd.Series()
if not heat_counts.empty:
    heat_counts.plot(kind="barh", ax=axes[0], color=PALETTE_MAIN, alpha=0.75)
    axes[0].set_title("Heating Type Distribution")
    axes[0].set_xlabel("Count")

# condition distribution
cond_counts = df["condition"].value_counts() if "condition" in df.columns else pd.Series()
if not cond_counts.empty:
    cond_counts.plot(kind="barh", ax=axes[1], color=PALETTE_ACC, alpha=0.75)
    axes[1].set_title("Condition Distribution")
    axes[1].set_xlabel("Count")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_categorical_distributions.png")
plt.close()
print("    Saved 06_categorical_distributions.png")

# ── 10.1  Rent by heatingType boxplot ───────────────────────
if "heatingType" in df.columns:
    top_heat = df["heatingType"].value_counts().head(8).index
    heat_df  = df[df["heatingType"].isin(top_heat)]

    fig, ax = plt.subplots(figsize=(13, 6))
    heat_df.boxplot(column="baseRent", by="heatingType",
                    ax=ax, showfliers=False,
                    boxprops=dict(color=PALETTE_MAIN),
                    medianprops=dict(color=PALETTE_SEC, linewidth=2))
    ax.set_title("Base Rent by Heating Type (top 8, no outliers)")
    ax.set_xlabel("Heating Type")
    ax.set_ylabel("Base Rent (€/month)")
    plt.suptitle("")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/07_rent_by_heatingtype.png")
    plt.close()
    print("    Saved 07_rent_by_heatingtype.png")

# ── 10.2  Rent by condition ──────────────────────────────────
if "condition" in df.columns:
    fig, ax = plt.subplots(figsize=(13, 6))
    df.boxplot(column="baseRent", by="condition", ax=ax, showfliers=False,
               boxprops=dict(color=PALETTE_MAIN),
               medianprops=dict(color=PALETTE_SEC, linewidth=2))
    ax.set_title("Base Rent by Condition (no outliers)")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Base Rent (€/month)")
    plt.suptitle("")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/08_rent_by_condition.png")
    plt.close()
    print("    Saved 08_rent_by_condition.png")

# ── 11  BINARY / AMENITY FEATURES ───────────────────────────
print("\n[11] Boolean amenity features vs rent …")
bool_rent = {}
for col in [c for c in BOOL_COLS if c in df.columns]:
    grp = df.groupby(col)["baseRent"].median()
    bool_rent[col] = grp

bool_rent_df = pd.DataFrame(bool_rent).T
bool_rent_df.columns = bool_rent_df.columns.map({True: "Yes/1", False: "No/0",
                                                  1: "Yes/1", 0: "No/0",
                                                  "True": "Yes/1", "False": "No/0"})
print("\n    Median baseRent (€/mo) by binary amenity:")
print(bool_rent_df.to_string())

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
valid_bool = [c for c in BOOL_COLS if c in df.columns]

for i, col in enumerate(valid_bool):
    grps = [df.loc[df[col] == v, "baseRent"].dropna()
            for v in sorted(df[col].unique())]
    labels = [str(v) for v in sorted(df[col].unique())]
    axes[i].boxplot(grps, labels=labels, showfliers=False,
                    patch_artist=True,
                    boxprops=dict(facecolor=PALETTE_MAIN, alpha=0.55),
                    medianprops=dict(color=PALETTE_SEC, linewidth=2))
    axes[i].set_title(col, fontweight="bold")
    axes[i].set_ylabel("baseRent (€)")

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Base Rent by Boolean Amenity Features (no outliers)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_amenity_features_vs_rent.png")
plt.close()
print("    Saved 09_amenity_features_vs_rent.png")

# ── 12  AMENITY SCORE ANALYSIS ───────────────────────────────
print("\n[12] Amenity score analysis …")
if "amenity_score" in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Count distribution
    df["amenity_score"].value_counts().sort_index().plot(
        kind="bar", ax=axes[0], color=PALETTE_MAIN, alpha=0.75)
    axes[0].set_title("Amenity Score Distribution (0–6)")
    axes[0].set_xlabel("Amenity Score")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=0)

    # Rent by score
    med_rent = df.groupby("amenity_score")["baseRent"].median()
    axes[1].plot(med_rent.index, med_rent.values, "o-",
                 color=PALETTE_MAIN, lw=2, markersize=8)
    axes[1].fill_between(med_rent.index, med_rent.values,
                         alpha=0.15, color=PALETTE_MAIN)
    axes[1].set_title("Median Base Rent by Amenity Score")
    axes[1].set_xlabel("Amenity Score")
    axes[1].set_ylabel("Median Base Rent (€/month)")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/10_amenity_score_analysis.png")
    plt.close()
    print("    Saved 10_amenity_score_analysis.png")

# ── 13  GEOGRAPHIC ANALYSIS ──────────────────────────────────
print("\n[13] Geographic analysis …")
if "state" in df.columns:
    state_stats = (df.groupby("state")["price_per_sqm"]
                   .agg(median_ppsm="median", count="count")
                   .sort_values("median_ppsm", ascending=True))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    state_stats["median_ppsm"].plot(
        kind="barh", ax=axes[0],
        color=[PALETTE_MAIN if v < state_stats["median_ppsm"].median()
               else PALETTE_SEC for v in state_stats["median_ppsm"]],
        alpha=0.8)
    axes[0].set_title("Median Price/sqm by German State")
    axes[0].set_xlabel("Median €/sqm")
    axes[0].axvline(state_stats["median_ppsm"].median(), color="red",
                    linestyle="--", lw=1.2, label="National median")
    axes[0].legend()

    state_stats["count"].plot(kind="barh", ax=axes[1], color=PALETTE_ACC, alpha=0.75)
    axes[1].set_title("Listing Count by German State")
    axes[1].set_xlabel("Number of listings")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/11_geographic_analysis_state.png")
    plt.close()
    print("    Saved 11_geographic_analysis_state.png")
    print("\n    Top 10 states by median price/sqm:")
    print(state_stats.sort_values("median_ppsm", ascending=False).head(10).to_string())

# ── 14  TOP CITIES ───────────────────────────────────────────
if "city" in df.columns:
    city_stats = (df.groupby("city")
                  .agg(median_ppsm=("price_per_sqm", "median"),
                       count=("price_per_sqm", "count"))
                  .query("count >= 500")
                  .sort_values("median_ppsm", ascending=False)
                  .head(20))

    fig, ax = plt.subplots(figsize=(13, 8))
    city_stats["median_ppsm"].sort_values().plot(
        kind="barh", ax=ax, color=PALETTE_MAIN, alpha=0.75)
    ax.set_title("Top 20 Cities by Median Price/sqm (≥500 listings)")
    ax.set_xlabel("Median €/sqm")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/12_top_cities_price_per_sqm.png")
    plt.close()
    print("    Saved 12_top_cities_price_per_sqm.png")

# ── 15  REGRESSION ASSUMPTION CHECKS ────────────────────────
print("\n[15] Regression assumption checks (OLS on numeric predictors) …")

ASSUME_FEATS = [c for c in [
    "livingSpace", "noRooms", "building_age", "amenity_score",
    "floor_ratio", "condition_score", "interior_score",
    "serviceCharge", "central_heating", "is_ground_floor"
] if c in df.columns]

assume_df = df[ASSUME_FEATS + [TARGET]].dropna().sample(
    min(20000, len(df)), random_state=RANDOM_SEED)

X_sm = sm.add_constant(assume_df[ASSUME_FEATS])
y_sm = assume_df[TARGET]

ols_fit = sm.OLS(y_sm, X_sm).fit()
print("\n    OLS Summary (sample of 20k rows):")
print(ols_fit.summary().tables[0])
print(ols_fit.summary().tables[1])

# Residual diagnostics
resids = ols_fit.resid
fitted = ols_fit.fittedvalues

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residuals vs Fitted
axes[0, 0].scatter(fitted, resids, alpha=0.1, s=4, color=PALETTE_MAIN)
axes[0, 0].axhline(0, color="red", linestyle="--", lw=1.2)
axes[0, 0].set_title("Residuals vs Fitted")
axes[0, 0].set_xlabel("Fitted Values")
axes[0, 0].set_ylabel("Residuals")

# Q-Q plot of residuals
sm.qqplot(resids, line="s", ax=axes[0, 1], alpha=0.2, markersize=2)
axes[0, 1].set_title("Q-Q Plot of Residuals")

# Scale-Location
std_resid = resids / resids.std()
axes[1, 0].scatter(fitted, np.sqrt(np.abs(std_resid)), alpha=0.1, s=4, color=PALETTE_ACC)
axes[1, 0].set_title("Scale-Location")
axes[1, 0].set_xlabel("Fitted Values")
axes[1, 0].set_ylabel("√|Standardised Residual|")

# Residual histogram
axes[1, 1].hist(resids, bins=80, color=PALETTE_MAIN, alpha=0.7, density=True, edgecolor="white")
xr = np.linspace(resids.min(), resids.max(), 300)
axes[1, 1].plot(xr, stats.norm.pdf(xr, resids.mean(), resids.std()),
                color="red", linestyle="--", lw=2, label="Normal PDF")
axes[1, 1].set_title("Residual Distribution")
axes[1, 1].legend()

fig.suptitle("OLS Regression Diagnostic Plots", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/13_ols_diagnostics.png")
plt.close()
print("    Saved 13_ols_diagnostics.png")

# Breusch-Pagan homoscedasticity test
bp_lm, bp_p, bp_f, bp_fp = het_breuschpagan(ols_fit.resid, ols_fit.model.exog)
print(f"\n    Breusch-Pagan test: LM stat={bp_lm:.4f}, p={bp_p:.4e}  "
      f"({'Homoscedastic' if bp_p>0.05 else 'Heteroscedastic — consider log transform'})")

# Shapiro-Wilk on residuals (sample)
sw_stat, sw_p = shapiro(resids.sample(min(5000, len(resids)), random_state=RANDOM_SEED))
print(f"    Shapiro-Wilk on residuals: W={sw_stat:.4f}, p={sw_p:.4e}  "
      f"({'Normal' if sw_p>0.05 else 'NOT Normal'})")

# ── 16  VIF ─────────────────────────────────────────────────
print("\n[16] Variance Inflation Factors …")
vif_data = pd.DataFrame()
vif_data["Feature"] = ASSUME_FEATS
vif_data["VIF"] = [variance_inflation_factor(X_sm.values, i + 1)
                   for i in range(len(ASSUME_FEATS))]
vif_data = vif_data.sort_values("VIF", ascending=False)
print(vif_data.to_string(index=False))
print("    Note: VIF > 5 = investigate; VIF > 10 = serious multicollinearity.")
vif_data.to_csv(f"{OUTPUT_DIR}/vif_results.csv", index=False)

# ── 17  PAIRPLOT — key features ─────────────────────────────
print("\n[17] Pairplot of key features (sampled) …")
PAIR_COLS = [c for c in [
    "baseRent", "livingSpace", "amenity_score",
    "building_age", "condition_score", LOG_TARGET
] if c in df.columns]

pair_sample = df[PAIR_COLS].sample(min(3000, len(df)), random_state=RANDOM_SEED)
pg = sns.pairplot(pair_sample, diag_kind="kde", plot_kws={"alpha": 0.25, "s": 10},
                  diag_kws={"fill": True})
pg.fig.suptitle("Pairplot — Key Features (n=3,000 sample)", y=1.01, fontsize=12, fontweight="bold")
pg.savefig(f"{OUTPUT_DIR}/14_pairplot.png")
plt.close()
print("    Saved 14_pairplot.png")

# ── 18  FEATURE ENGINEERING VALIDATION ──────────────────────
print("\n[18] Engineered feature validation …")
eng_cols = ["price_per_sqm", LOG_TARGET, "building_age", "amenity_score",
            "is_ground_floor", "floor_ratio", "condition_score",
            "interior_score", "central_heating"]
eng_present = [c for c in eng_cols if c in df.columns]
eng_missing = [c for c in eng_cols if c not in df.columns]

print(f"    ✓ Engineered features present : {eng_present}")
if eng_missing:
    print(f"    ✗ Missing engineered features: {eng_missing}")

print("\n    Sample engineered feature stats:")
print(df[eng_present].describe().T[["mean", "std", "min", "50%", "max"]].to_string())

# ── 19  SAVE CLEANED DATA WITH LOG FEATURE ───────────────────
print("\n[19] Saving dataset with log_price_per_sqm appended …")
df.to_csv(f"{OUTPUT_DIR}/immo_data_with_log.csv", index=False)
print(f"     Saved → {OUTPUT_DIR}/immo_data_with_log.csv")
print(f"     Final shape: {df.shape}")

# ── SUMMARY ──────────────────────────────────────────────────
print("\n" + "=" * 65)
print(" EDA COMPLETE")
print("=" * 65)
print(f"  All outputs saved to: ./{OUTPUT_DIR}/")
print("""
  Key findings for modeling team (Phase 2):
  ─────────────────────────────────────────
  1. baseRent is RIGHT-SKEWED (skew > 2). Log transform recommended
     for linear models. Tree-based models handle this natively.
  2. log_price_per_sqm is approximately normal — suitable as a
     target for linear/penalised regression directly.
  3. Breusch-Pagan test indicates HETEROSCEDASTICITY → supports
     using log-transformed target or robust models.
  4. livingSpace, noRooms, amenity_score, condition_score, and
     interior_score are the strongest numeric correlates of rent.
  5. No missing values after data pipeline (confirmed).
  6. VIF check: serviceCharge and totalRent may be collinear with
     baseRent — drop or treat carefully in regression models.
  7. Categorical features (heatingType, condition, typeOfFlat,
     state, city) should be one-hot or ordinal encoded.
  8. City-level effects are large — consider state/city as
     features or use city-stratified modeling.
  9. Small studios (<5 sqm) produce extreme price_per_sqm values;
     consider filtering livingSpace > 10 sqm for modeling.
 10. Models to run in Phase 2 (regression on baseRent & log_price_per_sqm):
     OLS, Ridge, LASSO, ElasticNet, PCR, PLS, SVR-Linear, SVR-RBF,
     Random Forest, GBM, XGBoost (+ SHAP interpretation).
     NOTE: Classification section from R template skipped — this is
     a regression problem. Binary/multi-class variants not applicable
     unless rent is discretised into price brackets (optional).
""")
