"""
H2 - Insurance Claims Cost/Risk Predictive Model
Builds features from CMS DE-SynPUF Sample 1 (2008 Beneficiary Summary + 2008-2010 Inpatient Claims),
trains an XGBoost classifier to flag likely top-decile-cost claimants, explains it with SHAP,
and saves the artifacts (model.pkl, shap_summary.png, scored dataset) used by the notebook,
the Streamlit app, and the README.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

ROOT = Path(__file__).parent
RAW = ROOT / "Data" / "raw"
PROCESSED = ROOT / "Data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Step 1: Load
# ---------------------------------------------------------------------------
beneficiary = pd.read_csv(RAW / "DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv")
claims = pd.read_csv(RAW / "DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv")

# ---------------------------------------------------------------------------
# Step 2: Feature engineering
# ---------------------------------------------------------------------------
# Age as of the start of the study window (2008-01-01), derived from BENE_BIRTH_DT (YYYYMMDD int)
birth_year = (beneficiary["BENE_BIRTH_DT"] // 10000).astype(int)
beneficiary["BENE_AGE"] = 2008 - birth_year

# Prospective framing: features come from PRIOR utilization (2008), the label comes from
# FUTURE cost (2009-2010). This avoids the circularity of using this-year's claim count to
# predict this-year's spend (trivially: 0 claims -> 0 cost) and mirrors how payers actually
# score members for care-management outreach -- using known history to flag future risk.
claims = claims.dropna(subset=["CLM_FROM_DT"])
claims["year"] = (claims["CLM_FROM_DT"] // 10000).astype(int)

prior = claims[claims["year"] <= 2008]
future = claims[claims["year"] >= 2009]

prior_agg = prior.groupby("DESYNPUF_ID").agg(
    prior_cost=("CLM_PMT_AMT", "sum"),
    num_claims=("CLM_ID", "count"),
).reset_index()

future_agg = future.groupby("DESYNPUF_ID").agg(
    future_cost=("CLM_PMT_AMT", "sum"),
).reset_index()

df = beneficiary.merge(prior_agg, on="DESYNPUF_ID", how="left")
df = df.merge(future_agg, on="DESYNPUF_ID", how="left")
df["num_claims"] = df["num_claims"].fillna(0)
df["prior_cost"] = df["prior_cost"].fillna(0)
df["future_cost"] = df["future_cost"].fillna(0)

# label: top 10% of 2009-2010 cost = high-cost claimant (this is what we're predicting)
threshold = df["future_cost"].quantile(0.90)
df["high_cost"] = (df["future_cost"] >= threshold).astype(int)

# chronic condition count (DE-SynPUF flags are 1 = yes, 2 = no)
chronic_cols = [c for c in df.columns if c.startswith("SP_")]
df["num_chronic"] = df[chronic_cols].apply(lambda x: (x == 1).sum(), axis=1)

# a couple of additional low-cost, clinically meaningful features
df["is_male"] = (df["BENE_SEX_IDENT_CD"] == 1).astype(int)
df["has_esrd"] = (df["BENE_ESRD_IND"] == "Y").astype(int)

print(f"Beneficiaries: {len(df):,}")
print(f"High-cost threshold (90th pct of 2009-2010 inpatient cost): ${threshold:,.0f}")
print(f"High-cost claimants (2009-2010): {df['high_cost'].sum():,} ({df['high_cost'].mean():.1%})")

# ---------------------------------------------------------------------------
# Step 3: Train/test split and model
# ---------------------------------------------------------------------------
features = ["BENE_AGE", "num_chronic", "num_claims", "prior_cost", "is_male", "has_esrd"]
X = df[features]
y = df["high_cost"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.1, eval_metric="logloss"
)
model.fit(X_train, y_train)

preds = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, preds)
report = classification_report(y_test, (preds > 0.5).astype(int))
print("\nAUC:", round(auc, 4))
print(report)

# The label is imbalanced (~10.6% positive), so a 0.5 probability cutoff is the wrong lens --
# in practice a care-management program enrolls a fixed top slice of members by score, not
# everyone above 0.5. Report precision/recall at the top-decile operating point instead.
test_eval = pd.DataFrame({"y_true": y_test.values, "score": preds}).sort_values(
    "score", ascending=False
)
top_decile_n = max(1, int(len(test_eval) * 0.10))
top_decile = test_eval.head(top_decile_n)
capture_rate = top_decile["y_true"].sum() / y_test.sum()
precision_at_decile = top_decile["y_true"].mean()
print(f"Top-decile operating point (flag top {top_decile_n} of {len(test_eval)} members by score):")
print(f"  Precision@decile: {precision_at_decile:.1%} (vs. {y_test.mean():.1%} base rate)")
print(f"  Capture rate: {capture_rate:.1%} of all actual high-cost members caught in that top 10%")

# ---------------------------------------------------------------------------
# Step 4: SHAP explainability
# ---------------------------------------------------------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig(ROOT / "shap_summary.png", bbox_inches="tight", dpi=150)
plt.close()

mean_abs_shap = pd.Series(
    np.abs(shap_values.values).mean(axis=0), index=features
).sort_values(ascending=False)
print("\nTop SHAP drivers (mean |SHAP value|):")
print(mean_abs_shap)

# ---------------------------------------------------------------------------
# Step 5: Save model + scored dataset for the README / one-pager / dashboard
# ---------------------------------------------------------------------------
pickle.dump(model, open(ROOT / "model.pkl", "wb"))

df["predicted_risk"] = model.predict_proba(X)[:, 1]
df[
    ["DESYNPUF_ID", "BENE_AGE", "num_chronic", "num_claims", "prior_cost", "is_male", "has_esrd",
     "future_cost", "high_cost", "predicted_risk"]
].to_csv(PROCESSED / "claims_scored.csv", index=False)

with open(ROOT / "model_metrics.txt", "w") as f:
    f.write(f"Beneficiaries: {len(df):,}\n")
    f.write(f"High-cost threshold (90th pct of 2009-2010 inpatient cost): ${threshold:,.0f}\n")
    f.write(f"High-cost claimants (2009-2010): {df['high_cost'].sum():,} ({df['high_cost'].mean():.1%})\n")
    f.write(f"AUC: {auc:.4f}\n\n")
    f.write(report)
    f.write(f"\nTop-decile operating point (flag top {top_decile_n} of {len(test_eval)} members by score):\n")
    f.write(f"  Precision@decile: {precision_at_decile:.1%} (vs. {y_test.mean():.1%} base rate)\n")
    f.write(f"  Capture rate: {capture_rate:.1%} of all actual high-cost members caught in that top 10%\n")
    f.write("\nTop SHAP drivers (mean |SHAP value|):\n")
    f.write(mean_abs_shap.to_string())

print("\nSaved model.pkl, shap_summary.png, model_metrics.txt, Data/processed/claims_scored.csv")
