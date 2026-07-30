import streamlit as st
import shap
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "model.pkl"

st.set_page_config(page_title="High-Cost Claimant Risk Tool", page_icon="🏥")

@st.cache_resource
def load_model():
    return pickle.load(open(MODEL_PATH, "rb"))

model = load_model()
explainer = shap.TreeExplainer(model)

st.title("High-Cost Claimant Risk Tool")
st.write(
    "Estimate a Medicare beneficiary's likelihood of becoming a top-decile inpatient "
    "cost claimant over the next two years, based on prior-year utilization and demographics."
)

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 100, 72)
    num_chronic = st.slider("Number of chronic conditions (of 11 tracked)", 0, 11, 2)
    num_claims = st.number_input("Inpatient claims in prior year", 0, 50, 0)
with col2:
    prior_cost = st.number_input("Total inpatient cost in prior year ($)", 0, 200000, 0, step=500)
    sex = st.radio("Sex", ["Female", "Male"])
    has_esrd = st.checkbox("End-stage renal disease (ESRD)")

if st.button("Predict", type="primary"):
    input_df = pd.DataFrame(
        [[age, num_chronic, num_claims, prior_cost, 1 if sex == "Male" else 0, int(has_esrd)]],
        columns=["BENE_AGE", "num_chronic", "num_claims", "prior_cost", "is_male", "has_esrd"],
    )
    score = model.predict_proba(input_df)[0][1]
    st.metric("High-Cost Risk (next 2 years)", f"{score:.0%}")

    if score >= 0.30:
        st.warning("Above the model's top-decile risk band on the training population — consider proactive care management outreach.")
    else:
        st.info("Below the top-decile risk band.")

    st.write("**Top risk drivers for this member:**")
    shap_vals = explainer(input_df)
    fig, ax = plt.subplots()
    shap.plots.bar(shap_vals[0], show=False)
    st.pyplot(fig)

st.caption(
    "Model: XGBoost classifier trained on CMS DE-SynPUF Sample 1 (synthetic Medicare claims, no PHI). "
    "Test AUC 0.73; flagging the top 10% of members by score captures ~2.4x the base high-cost rate. "
    "Portfolio project — not a validated clinical or actuarial tool."
)
