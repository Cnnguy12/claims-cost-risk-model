# Insurance Claims Cost/Risk Predictive Model

XGBoost + SHAP model that flags Medicare beneficiaries likely to become top-decile inpatient cost claimants, packaged behind a Streamlit front-end for a non-technical business user. Payers need to identify high-cost members *before* the cost is incurred so care-management resources can be targeted proactively — this scores members on prior-year utilization and demographics to predict who becomes a top-10% inpatient spender over the following two years.

## Key results

| # | Step | Key result |
|---|------|------------|
| 1 | Feature engineering (`pipeline.py`, [`notebooks/claims_risk_model.ipynb`](notebooks/claims_risk_model.ipynb)) | Prospective split: 2008 utilization/demographics → predicts top-decile inpatient spend in **2009-2010** |
| 2 | Model (XGBoost, 200 trees, depth 4) | Test **AUC 0.73** |
| 3 | Business-relevant evaluation | Flagging the highest-risk 10% of members captures a **25.7%** actual high-cost rate vs. a **10.6%** population base rate — **2.4x lift**, catching 24.3% of all future high-cost members |
| 4 | SHAP explainability | Top driver is **chronic condition count**, then ESRD status and age (see `shap_summary.png`) |
| 5 | Front-end (`app.py`) | Streamlit app — enter a member profile, get a risk score plus per-member SHAP drivers |

## Notes on results

The original framing of this label was circular: using a beneficiary's claim count to predict their own spend in the same year is close to tautological (zero claims trivially means zero cost). The features here are built strictly from **2008** data and the label is **2009-2010** spend, which mirrors how a real prospective risk model is actually deployed and avoids that circularity — at the cost of a lower, more honest AUC (0.73 rather than the ~0.96 the circular version produced). The label is also imbalanced (10.6% positive), so a 0.5 probability cutoff is the wrong lens; the top-decile operating point above reflects how this would actually be used — rank members by score and enroll the highest-risk slice, not threshold at 50%.

## Data

[CMS DE-SynPUF Sample 1](https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files/cms-2008-2010-data-entrepreneurs-synthetic-public-use-file-de-synpuf/de10-sample-1) — 2008 Beneficiary Summary File + 2008-2010 Inpatient Claims. Fully synthetic Medicare data (no PHI), 116,352 beneficiaries.

Raw source files aren't tracked in this repo (see `.gitignore`) — download both from the link above into `Data/raw/`. The scored population output *is* tracked at `Data/processed/claims_scored.csv`.

## Structure

```
pipeline.py               end-to-end script: feature engineering -> model -> SHAP -> saved artifacts
notebooks/                same pipeline as an executed, annotated notebook
app.py                    Streamlit front-end
model.pkl                 trained XGBoost model
shap_summary.png          SHAP summary plot
model_metrics.txt         full evaluation output
business_one_pager.md     problem / approach / result / impact writeup
Data/raw/                 CMS DE-SynPUF source files (not tracked, see Data section)
Data/processed/           scored population output
```

## Setup

```bash
pip install -r requirements.txt
python pipeline.py          # rebuilds model.pkl, shap_summary.png, scored dataset
streamlit run app.py        # launches the risk tool at localhost:8501
```

To deploy: push this repo to GitHub (done), then go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub → New app → select this repo → `app.py` as the entry point.

## License

MIT — see [LICENSE](LICENSE).
