# Loan Approval Prediction using Machine Learning

A binary classification model that predicts whether a loan application should be **approved or rejected** from the applicant's profile, built with a Random Forest classifier in Python.

Built during my 1-month internship at **JNNC Technologies Pvt. Ltd.** (15 March - 15 April 2026).

## Problem

Manual loan approval is slow, subjective and hard to scale. This project automates the eligibility decision with machine learning and finds which applicant features drive it.

## Dataset

The dataset is **synthetic**: 1,000 simulated applications generated in `loan_approval.py` (seed 42), so no real customer data is used. The approval label follows a built-in rule: applicants with good credit history and combined income above 5,000 are approved with high probability, and others with low probability.

| Feature | Type |
|---|---|
| Gender, Married, Dependents, Education, Self_Employed, Property_Area | Categorical |
| ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term | Numerical |
| Credit_History | Binary (1.0 good, 0.0 poor) |
| **Loan_Status** | Target (Y = approved, N = rejected) |

Class balance: 771 approved, 229 rejected. The generated data is saved in `data/loan_data.csv`.

## Method

1. Generate the synthetic dataset
2. Exploratory data analysis (class distribution, correlation heatmap)
3. Preprocessing: drop `Loan_ID`, label-encode categorical columns
4. 80/20 train-test split (`random_state=42`)
5. Train `RandomForestClassifier(n_estimators=100)`
6. Evaluate with accuracy, confusion matrix, precision/recall and feature importance

## Results (200 test applications)

| Metric | Value |
|---|---|
| Accuracy | **90.5%** |
| Precision / Recall (Approved) | 0.93 / 0.96 |
| Precision / Recall (Rejected) | 0.82 / 0.72 |

Confusion matrix (rows = actual, columns = predicted): 31 rejected correctly, 12 rejected applicants wrongly approved, 7 approved applicants wrongly rejected, 150 approved correctly.

![Confusion matrix](images/confusion_matrix.png)
![Feature importance](images/feature_importance.png)

**Top features:** Credit_History (0.23), ApplicantIncome (0.20), CoapplicantIncome (0.17), LoanAmount (0.17).

## Limitations

- The data is simulated and its labels come from a simple rule, so the 90.5% shows the model can recover that pattern. It is **not** a measure of performance on real lending data.
- Features with no role in the label rule (Gender, Married, Education, etc.) still show small importance values of about 0.02. That is noise, not signal.
- The 12 wrongly approved applicants are the costly error for a lender; threshold tuning would be needed to reduce them.

## How to run

```bash
git clone https://github.com/<your-username>/loan-approval-prediction.git
cd loan-approval-prediction
pip install -r requirements.txt
python loan_approval.py
```

This regenerates `data/loan_data.csv` and the plots in `images/`.

## Future work

- Compare against Logistic Regression and Decision Tree
- Tune hyper-parameters with `GridSearchCV`
- Test on a real public loan dataset
- Deploy as a Streamlit app for instant predictions

## Tech stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn
