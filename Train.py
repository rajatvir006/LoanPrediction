"""
Loan Approval Prediction — train.py
Classical ML Mini Project | Python + scikit-learn

Flow:
  Data → Split → Preprocess → SMOTE → Train (LR + KNN) → Evaluate → Save
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection    import train_test_split
from sklearn.impute             import SimpleImputer
from sklearn.preprocessing      import StandardScaler, OneHotEncoder
from sklearn.compose            import ColumnTransformer
from sklearn.pipeline           import Pipeline
from sklearn.linear_model       import LogisticRegression
from sklearn.neighbors          import KNeighborsClassifier
from sklearn.decomposition      import PCA
from sklearn.metrics            import (confusion_matrix, ConfusionMatrixDisplay,
                                        accuracy_score, precision_score,
                                        recall_score, f1_score,
                                        classification_report)
from imblearn.over_sampling     import SMOTE

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
# Real Kaggle Loan Prediction dataset (Analytics Vidhya).
# 614 rows, 12 features + 1 target (Loan_Status: Y/N → 1/0).
df = pd.read_csv("train.csv")
df = df.drop(columns=["Loan_ID"])                           # ID column — not a feature
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0}) # Map text labels to binary
print(f"Dataset: {df.shape}  |  Approved: {df.Loan_Status.sum()}  Rejected: {(df.Loan_Status==0).sum()}")



# ─────────────────────────────────────────────
# 2. TRAIN-TEST SPLIT  (always first — prevents leakage)
# ─────────────────────────────────────────────
X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

# ─────────────────────────────────────────────
# 3. PREPROCESSING PIPELINE
# ─────────────────────────────────────────────
num_cols = ["ApplicantIncome","CoapplicantIncome","LoanAmount",
            "Loan_Amount_Term","Credit_History"]
cat_cols = ["Gender","Married","Dependents","Education",
            "Self_Employed","Property_Area"]

# Numerical: fill missing with median, then scale to mean=0 std=1
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

# Categorical: fill missing with most common value, then one-hot encode
# drop='first' removes one dummy column per feature to avoid multicollinearity
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
])

# ColumnTransformer applies the right pipeline to the right columns
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols),
])

# Fit ONLY on training data, then transform both sets
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)      # no fitting here!

print(f"After preprocessing — Train shape: {X_train_proc.shape}")

# ─────────────────────────────────────────────
# 4. SMOTE  (only on training data!)
# ─────────────────────────────────────────────
# SMOTE creates synthetic minority-class samples so both classes are equal.
# Never apply it to the test set — that would corrupt our evaluation.

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train)
print(f"After SMOTE — Train shape: {X_train_bal.shape}  "
      f"|  Class counts: {dict(pd.Series(y_train_bal).value_counts())}")

# ─────────────────────────────────────────────
# 5. TRAIN MODELS
# ─────────────────────────────────────────────
lr  = LogisticRegression(C=1.0, max_iter=300, random_state=42)
knn = KNeighborsClassifier(n_neighbors=7, weights="distance")

lr.fit(X_train_bal, y_train_bal)
knn.fit(X_train_bal, y_train_bal)

lr_pred  = lr.predict(X_test_proc)
knn_pred = knn.predict(X_test_proc)

# ─────────────────────────────────────────────
# 6. EVALUATE
# ─────────────────────────────────────────────
def show_metrics(name, y_true, y_pred):
    print(f"\n{'-'*40}")
    print(f"  {name}")
    print(f"{'-'*40}")
    print(f"  Accuracy : {accuracy_score(y_true, y_pred):.3f}")
    print(f"  Precision: {precision_score(y_true, y_pred):.3f}")
    print(f"  Recall   : {recall_score(y_true, y_pred):.3f}")
    print(f"  F1-Score : {f1_score(y_true, y_pred):.3f}")
    print(classification_report(y_true, y_pred,
          target_names=["Rejected","Approved"], zero_division=0))

show_metrics("Logistic Regression", y_test, lr_pred)
show_metrics("KNN", y_test, knn_pred)

# ─────────────────────────────────────────────
# 7. PLOTS  (confusion matrices + PCA)
# ─────────────────────────────────────────────

# — Confusion Matrices —
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Confusion Matrices", fontweight="bold")
for ax, (name, preds) in zip(axes, [("Logistic Regression", lr_pred),
                                     ("KNN",                 knn_pred)]):
    ConfusionMatrixDisplay(confusion_matrix(y_test, preds),
                           display_labels=["Rejected","Approved"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name)
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=120)
plt.close()

# — PCA Scatter (2D view of training data after SMOTE) —
pca   = PCA(n_components=2, random_state=42)
X_2d  = pca.fit_transform(X_train_bal)
ev    = pca.explained_variance_ratio_ * 100

fig, ax = plt.subplots(figsize=(7, 5))
for label, color, name in [(0,"#EF5350","Rejected"), (1,"#42A5F5","Approved")]:
    m = y_train_bal == label
    ax.scatter(X_2d[m,0], X_2d[m,1], c=color, label=name, alpha=0.4, s=15)
ax.set_xlabel(f"PC1 ({ev[0]:.1f}% variance)")
ax.set_ylabel(f"PC2 ({ev[1]:.1f}% variance)")
ax.set_title("PCA — Training Data (after SMOTE)", fontweight="bold")
ax.legend(); ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("pca_plot.png", dpi=120)
plt.close()

print("\nPlots saved: confusion_matrices.png  |  pca_plot.png")

# ─────────────────────────────────────────────
# 8. SAVE MODELS
# ─────────────────────────────────────────────
joblib.dump(preprocessor, "preprocessor.pkl")
joblib.dump(lr,  "lr_model.pkl")
joblib.dump(knn, "knn_model.pkl")
print("Models saved: preprocessor.pkl  |  lr_model.pkl  |  knn_model.pkl")
print("\nDone! Run:  streamlit run app.py")
