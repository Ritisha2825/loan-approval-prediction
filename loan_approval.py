"""
Loan Approval Prediction using Machine Learning (Random Forest)

Built during the JNNC Technologies internship (15 Mar - 15 Apr 2026).
Run:  python loan_approval.py
Outputs: data/loan_data.csv and plots in images/
"""
import os

import matplotlib
matplotlib.use("Agg")  # save plots to files; no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

os.makedirs("data", exist_ok=True)
os.makedirs("images", exist_ok=True)

# ==========================================
# STEP 1: Create Synthetic Dataset
# ==========================================
# A simulated dataset of loan applications (no real customer data).
np.random.seed(42)
data_size = 1000

data = {
    "Loan_ID": [f"LP{i}" for i in range(1000, 1000 + data_size)],
    "Gender": np.random.choice(["Male", "Female"], data_size),
    "Married": np.random.choice(["Yes", "No"], data_size),
    "Dependents": np.random.choice(["0", "1", "2", "3+"], data_size),
    "Education": np.random.choice(["Graduate", "Not Graduate"], data_size),
    "Self_Employed": np.random.choice(["Yes", "No"], data_size),
    "ApplicantIncome": np.random.randint(2000, 20000, data_size),
    "CoapplicantIncome": np.random.randint(0, 10000, data_size),
    "LoanAmount": np.random.randint(50, 500, data_size),
    "Loan_Amount_Term": np.random.choice([120, 240, 360], data_size),
    "Credit_History": np.random.choice([1.0, 0.0], data_size, p=[0.8, 0.2]),
    "Property_Area": np.random.choice(["Rural", "Semi-urban", "Urban"], data_size),
}


def determine_status(row):
    """Good credit history + higher combined income -> higher chance of approval."""
    if row["Credit_History"] == 1.0 and (row["ApplicantIncome"] + row["CoapplicantIncome"]) > 5000:
        return np.random.choice(["Y", "N"], p=[0.9, 0.1])
    return np.random.choice(["Y", "N"], p=[0.3, 0.7])


df = pd.DataFrame(data)
df["Loan_Status"] = df.apply(determine_status, axis=1)
df.to_csv("data/loan_data.csv", index=False)

print("Dataset created successfully!")
print(df.head())
print("\nLoan_Status counts:\n", df["Loan_Status"].value_counts())

# ==========================================
# STEP 2: Exploratory Data Analysis (EDA)
# ==========================================
plt.figure(figsize=(6, 4))
sns.countplot(x="Loan_Status", hue="Loan_Status", data=df, palette="viridis", legend=False)
plt.title("Distribution of Loan Approval Status")
plt.tight_layout()
plt.savefig("images/loan_status_distribution.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("images/correlation_heatmap.png", dpi=150)
plt.close()

# ==========================================
# STEP 3: Data Preprocessing
# ==========================================
df = df.drop("Loan_ID", axis=1)  # identifier, not a predictive feature

le = LabelEncoder()
categorical_cols = ["Gender", "Married", "Dependents", "Education",
                    "Self_Employed", "Property_Area", "Loan_Status"]
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]  # encoded: N = 0, Y = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# STEP 4: Model Building & Training
# ==========================================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ==========================================
# STEP 5: Evaluation
# ==========================================
print("\n--- Model Evaluation ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["N (Rejected)", "Y (Approved)"]))

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Rejected", "Approved"], yticklabels=["Rejected", "Approved"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("images/confusion_matrix.png", dpi=150)
plt.close()

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
print("\nFeature importances:\n", feat_importances.sort_values(ascending=False).round(3))

plt.figure(figsize=(8, 5))
feat_importances.nlargest(10).sort_values().plot(kind="barh")
plt.title("Top Features influencing Loan Approval")
plt.tight_layout()
plt.savefig("images/feature_importance.png", dpi=150)
plt.close()
