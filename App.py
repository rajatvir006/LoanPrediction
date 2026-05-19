"""
Loan Approval Prediction — app.py
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib
import pathlib
import subprocess
import sys

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")

# ── Load models ───────────────────────────────────────────────────────────────
# Dataset bounds (from the real Kaggle CSV) — used to cap UI sliders
# so users can't enter values the model has never seen during training.
DATA_MAX_APP_INCOME  = 81000   # max ApplicantIncome in train.csv
DATA_MAX_COAPP_INCOME = 41667  # max CoapplicantIncome in train.csv

BASE = pathlib.Path(__file__).parent   # always resolves relative to this file

@st.cache_resource
def load():
    # Use sys.executable — always the correct Python binary (python3 on Linux)
    # Using pathlib.Path so paths always resolve correctly regardless of
    # which directory the app is launched from.
    needed = [BASE / "preprocessor.pkl", BASE / "lr_model.pkl", BASE / "knn_model.pkl"]
    if any(not p.exists() for p in needed):
        subprocess.run([sys.executable, str(BASE / "Train.py")], check=True)
    return (joblib.load(BASE / "preprocessor.pkl"),
            joblib.load(BASE / "lr_model.pkl"),
            joblib.load(BASE / "knn_model.pkl"))

try:
    preprocessor, lr, knn = load()
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.info("This usually means Train.py failed. Check that train.csv is present.")
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏦 Loan Approval Predictor")
st.write("Fill in the applicant details and click **Predict** to see the result.")
st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    gender        = st.selectbox("Gender",          ["Male","Female"])
    married       = st.selectbox("Married",         ["Yes","No"])
    dependents    = st.selectbox("Dependents",      ["0","1","2","3+"])
    education     = st.selectbox("Education",       ["Graduate","Not Graduate"])
    self_employed = st.selectbox("Self Employed",   ["No","Yes"])
    property_area = st.selectbox("Property Area",   ["Urban","Semiurban","Rural"])

with c2:
    app_income   = st.number_input("Applicant Income (per month)",
                                   min_value=0, max_value=DATA_MAX_APP_INCOME,
                                   value=5000, step=500,
                                   help=f"Max in training data: {DATA_MAX_APP_INCOME:,}")
    coapp_income = st.number_input("Co-applicant Income (per month)",
                                   min_value=0, max_value=DATA_MAX_COAPP_INCOME,
                                   value=0, step=500,
                                   help=f"Max in training data: {DATA_MAX_COAPP_INCOME:,}")
    loan_amount  = st.number_input("Loan Amount (in thousands)",
                                   min_value=9, max_value=700,
                                   value=120, step=5,
                                   help="e.g. enter 120 for a loan of 120,000")
    loan_term   = st.selectbox("Loan Term (months)", [120,180,240,300,360,480], index=4)
    credit_hist = st.radio("Credit History",
                           [1.0, 0.0],
                           format_func=lambda x: "✅ Good" if x==1.0 else "❌ Bad / None")

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("🔍 Predict", use_container_width=True, type="primary"):

    row = pd.DataFrame([{
        "Gender":gender, "Married":married, "Dependents":dependents,
        "Education":education, "Self_Employed":self_employed,
        "ApplicantIncome":float(app_income), "CoapplicantIncome":float(coapp_income),
        "LoanAmount":float(loan_amount), "Loan_Amount_Term":float(loan_term),
        "Credit_History":float(credit_hist), "Property_Area":property_area,
    }])

    X = preprocessor.transform(row)

    lr_pred  = lr.predict(X)[0]
    lr_prob  = lr.predict_proba(X)[0][1]
    knn_pred = knn.predict(X)[0]
    knn_prob = knn.predict_proba(X)[0][1]

    st.subheader("Results")

    for name, pred, prob in [("Logistic Regression", lr_pred, lr_prob),
                              ("KNN",                 knn_pred, knn_prob)]:
        label = "✅ APPROVED" if pred == 1 else "❌ REJECTED"
        (st.success if pred==1 else st.error)(f"**{name}** → {label}  ({prob*100:.1f}% approval probability)")
        st.progress(float(prob))

    if lr_pred == knn_pred:
        st.info("Both models agree ✔")
    else:
        st.warning("Models disagree — consider the one with higher probability.")


# ── About ──────────────────────────────────────────────────────────────────────
with st.expander("ℹ️ How it works"):
    st.markdown("""
**Pipeline:** Raw input → `preprocessor.pkl` (Imputer + Scaler + OneHotEncoder) → Model prediction

**Models trained with:**
- `SimpleImputer` to fill missing values
- `StandardScaler` to normalise numbers
- `OneHotEncoder` to convert categories to numbers
- `SMOTE` to balance the training data
- `LogisticRegression` + `KNN` for classification
- `joblib` to save and reload everything
""")