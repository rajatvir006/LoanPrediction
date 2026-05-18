"""
Loan Approval Prediction — app.py
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib, os

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load():
    needed = ["preprocessor.pkl", "lr_model.pkl", "knn_model.pkl"]
    # If models are missing, or if they crash due to version mismatch, retrain them on the fly!
    if any(not os.path.exists(f) for f in needed):
        os.system("python Train.py")
    try:
        return (joblib.load("preprocessor.pkl"),
                joblib.load("lr_model.pkl"),
                joblib.load("knn_model.pkl"))
    except Exception:
        os.system("python Train.py")
        return (joblib.load("preprocessor.pkl"),
                joblib.load("lr_model.pkl"),
                joblib.load("knn_model.pkl"))

preprocessor, lr, knn = load()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏦 Loan Approval Predictor")
st.write("Fill in the applicant details and click **Predict** to see the result.")
st.divider()

if preprocessor is None:
    st.error("Models not found. Please run  `python train.py`  first.")
    st.stop()

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
    app_income  = st.number_input("Applicant Income (₹/month)",   0, 500000, 5000, 500)
    coapp_income= st.number_input("Co-applicant Income (₹/month)",0, 200000, 0,    500)
    loan_amount = st.number_input("Loan Amount (₹ thousands)",    9, 700,    120,  5)
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