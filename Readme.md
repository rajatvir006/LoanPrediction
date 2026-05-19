# 🏦 Loan Approval Prediction System
### Classical ML Mini Project — scikit-learn + Streamlit

---

## 📁 Project Files

```
loan_approval_project/
├── Train.py            ← Run this first  (ML pipeline)
├── App.py              ← Run this second (Streamlit app)
├── requirements.txt    ← Dependencies
├── README.md           ← This file
│
│   (created after running Train.py)
├── preprocessor.pkl
├── lr_model.pkl
├── knn_model.pkl
├── confusion_matrices.png
└── pca_plot.png
```

---

## ▶️ How to Run (Step-by-Step)

```bash
# Step 1 — install packages
pip install -r requirements.txt

# Step 2 — train and save models
python Train.py

# Step 3 — launch the web app
streamlit run App.py
# Opens at http://localhost:8501
```

---

## 🔄 Pipeline Flow (Easy to Explain in Viva)

```
Raw Data (with missing values + imbalance)
        │
        ▼
   Train-Test Split  ← ALWAYS first to prevent leakage
        │
   ┌────┴────────────────────────────┐
   │                                 │
   Numerical cols                Categorical cols
   SimpleImputer(median)         SimpleImputer(most_frequent)
   StandardScaler                OneHotEncoder(drop='first')
   │                                 │
   └────────── ColumnTransformer ────┘
                    │
                  SMOTE  ← only on training data
                    │
          ┌─────────┴──────────┐
          │                    │
   Logistic Regression        KNN
          │                    │
          └──── Evaluate ──────┘
                    │
             Save with joblib
                    │
             Streamlit App
```

---

## 🧠 ML Concepts — Simple Viva Explanations

### Train-Test Split
Split data 80/20 *before* any preprocessing. The test set simulates real unseen data. Fitting the scaler on the full dataset would leak test statistics into training — giving falsely high accuracy.

### SimpleImputer
Fills missing values automatically.
- `strategy='median'` for numbers → robust to outliers in income columns
- `strategy='most_frequent'` for text → fills with the most common category

### OneHotEncoder
Converts text categories to 0/1 columns so the model can process them.
`drop='first'` removes one dummy per feature to avoid the **dummy variable trap** (perfect multicollinearity).

### StandardScaler
Transforms numbers to mean=0, std=1.
Critical for **KNN** — without it, ApplicantIncome (range 1k–80k) would completely overpower Credit_History (0 or 1) when computing distances.

### ColumnTransformer
Applies different pipelines to different columns in one step. Keeps preprocessing clean and leak-free.

### Pipeline
Chains steps (impute → scale) so `.fit_transform()` runs them in order. Prevents accidentally applying a step in the wrong order.

### SMOTE
Our data has ~85% Approved, ~15% Rejected. A model that always predicts "Approved" would get 85% accuracy without learning anything.
SMOTE creates *synthetic* minority-class points by interpolating between real ones — balancing the training set. Applied **only on training data**.

### Logistic Regression
Predicts probability using the sigmoid function: `P = 1 / (1 + e^-z)`.
If P > 0.5 → Approved. Linear, fast, interpretable. Great baseline.

### KNN
Classifies a new point by majority vote among its *k* nearest neighbours.
Non-parametric (no formula assumed). Sensitive to scale → StandardScaler is essential.

### Confusion Matrix
Shows TP, FP, TN, FN. In loan approval, a False Positive (approving a bad loan) costs money, so we also watch Precision, not just Accuracy.

### Accuracy / Precision / Recall / F1
- **Accuracy** = correct / total
- **Precision** = of all predicted Approved, how many actually were?
- **Recall** = of all actual Approved, how many did we catch?
- **F1** = harmonic mean of Precision & Recall (best single metric for imbalanced data)

### PCA
Reduces many features to 2 dimensions for visualisation. Shows whether Approved and Rejected clusters are separable in the feature space.

### joblib
Saves the trained preprocessor and models to `.pkl` files. The Streamlit app loads them to predict on new input — no retraining needed.

---

## ❓ Quick Viva Q&A

**Q: Why split before preprocessing?**
To prevent data leakage. If you fit the scaler on all data, test-set statistics influence training — inflating accuracy artificially.

**Q: Why median for numerical imputation?**
Income is right-skewed. Mean is pulled up by outliers. Median is the middle value and stays stable.

**Q: Why drop='first' in OneHotEncoder?**
With k categories you only need k-1 columns. The k-th is implied when all others are 0. Keeping all k creates perfect multicollinearity.

**Q: Why SMOTE only on training data?**
The test set must reflect real-world class distribution. Oversampling it would contaminate evaluation.

**Q: Why use StandardScaler before KNN?**
KNN uses Euclidean distance. Without scaling, high-range features dominate low-range ones, making KNN ignore important features.

**Q: What is your best metric and why?**
F1-Score — it balances Precision and Recall and is reliable on imbalanced datasets where Accuracy can be misleading.

**Q: What is the dummy variable trap?**
With k categories and k dummy columns, the last column is always predictable from the others → perfect multicollinearity → unstable model. Drop one column to fix it.

---

## ⚠️ Common Mistakes

| Mistake | Fix |
|---|---|
| Fitting scaler on full data | Split first, fit on train only |
| Applying SMOTE before split | Apply SMOTE after split, train only |
| Not scaling before KNN | Always scale numerical features |
| Using accuracy on imbalanced data | Use F1-Score |
| `fit_transform` on test data | Use `.transform()` only on test |

---

## 📦 Requirements

```
scikit-learn, imbalanced-learn, pandas, numpy, matplotlib, streamlit, joblib
```