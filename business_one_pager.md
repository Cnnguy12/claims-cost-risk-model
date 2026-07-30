# High-Cost Claimant Risk Tool — Business One-Pager

## Problem
Rising claims costs are concentrated in a small share of members. Payers and health systems need to identify likely high-cost claimants *before* the cost is incurred, so care-management resources (case managers, outreach programs) can be targeted proactively rather than reactively.

## Approach
Trained an XGBoost classifier on CMS DE-SynPUF synthetic Medicare claims (116,352 beneficiaries). Features are built strictly from **prior-year (2008)** information — age, chronic condition count, prior inpatient claim count and cost, sex, ESRD status — to predict whether a beneficiary lands in the **top 10% of inpatient spend over the following two years (2009-2010)**. This prospective framing mirrors how a real risk-scoring program would be deployed: score members today on what's already known, flag the highest-risk slice, act before the cost happens.

SHAP was used to make the model's reasoning interpretable at both the population and individual-member level (see `shap_summary.png` and the Streamlit tool).

## Result
- **Test AUC: 0.73** — meaningful separation given the label is inherently noisy (predicting 2 years out from limited features) and rare (10.6% base rate).
- **Operating point that matters for the business:** flagging the top 10% of members by predicted risk score captures a **25.7% actual high-cost rate vs. a 10.6% population base rate — a 2.4x lift** — and catches **24.3% of all future high-cost members** within that top decile.
- **Top predictor: chronic condition count**, followed by ESRD status and age — consistent with clinical expectations and directly actionable (multi-morbid members are the clearest targets for outreach).

## Business Impact (illustrative — assumption stated explicitly)
If the top-decile-flagged members were enrolled in a care-management program that reduced their inpatient spend by even 10%, and the average flagged member's 2-year inpatient cost is roughly $20,000-$25,000 (based on this sample's high-cost threshold and distribution), that implies **roughly $200,000-$250,000 in avoided cost per 1,000 flagged members enrolled** over the two-year window. This is a directional estimate for portfolio purposes, not a validated ROI — a real deployment would need a randomized or matched-control pilot to confirm the causal savings.

## Caveats
- Data is fully synthetic (CMS DE-SynPUF) — no real patient information, and synthetic data can understate real-world signal.
- AUC and precision/recall are reported at the top-decile operating point because the label is imbalanced (~10.6% positive); a 0.5 probability cutoff is not the right way to read this model.
- This is a portfolio project, not a validated actuarial or clinical model.
