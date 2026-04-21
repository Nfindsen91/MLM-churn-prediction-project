import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import joblib

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUTS_PATH = PROJECT_ROOT / "outputs"
OUTPUTS_PATH.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

# Clean TotalCharges column
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# Split features and target
y = df["Churn"].map({"Yes": 1, "No": 0})
X = df.drop(columns=["Churn", "customerID"])

# Identify columns
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "string"]).columns

# Preprocessing
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Models
models = {
    "Logistic Regression": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]
    ),
    "Decision Tree": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(
                random_state=42,
                max_depth=5
            ))
        ]
    )
}

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train + Evaluate
results = []

for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    y_pred = mdl.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1
    })

    print(f"\n{name} Performance:")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")

# Print comparison summary
results_df = pd.DataFrame(results).sort_values(by="F1", ascending=False)
print("\nModel Comparison (sorted by F1):")
print(results_df.to_string(index=False))

# Pick the best model by F1
best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]

# Visualizations

# 1) Churn distribution
plt.figure()
df["Churn"].value_counts().plot(kind="bar")
plt.title("Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUTPUTS_PATH / "churn_distribution.png")
plt.close()

# 2) Monthly Charges by Churn
plt.figure()
df.boxplot(column="MonthlyCharges", by="Churn")
plt.title("Monthly Charges by Churn")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.savefig(OUTPUTS_PATH / "monthly_charges_by_churn.png")
plt.close()

# 3) Confusion matrix for best model
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()
plt.savefig(OUTPUTS_PATH / "confusion_matrix_best_model.png")
plt.close()

# Save best model
joblib.dump(best_model, OUTPUTS_PATH / "churn_model_best.joblib")

print(f"\nBest model saved: {best_model_name}")
print("Artifacts saved to the outputs/ directory.")


# User Application

def prompt_int(prompt: str, min_val: int, max_val: int) -> int:
    while True:
        try:
            val = int(input(prompt).strip())
            if min_val <= val <= max_val:
                return val
            print(f"Please enter a value between {min_val} and {max_val}.")
        except ValueError:
            print("Please enter a whole number.")


def prompt_float(prompt: str, min_val: float = 0.0) -> float:
    while True:
        try:
            val = float(input(prompt).strip())
            if val >= min_val:
                return val
            print(f"Please enter a value >= {min_val}.")
        except ValueError:
            print("Please enter a number.")


def prompt_choice(prompt: str, options: list[str]) -> str:
    options_lower = [o.lower() for o in options]
    while True:
        val = input(prompt).strip()
        if val.lower() in options_lower:
            # return canonical option with matching case
            return options[options_lower.index(val.lower())]
        print(f"Please enter one of: {', '.join(options)}")


def build_user_input_row(template_X: pd.DataFrame) -> pd.DataFrame:
    row = template_X.iloc[[0]].copy()

    print("\n-- Churn Risk Estimator --")
    print("Enter a few customer details to estimate churn risk.\n")

    tenure = prompt_int("Tenure in months (0-72): ", 0, 72)
    monthly = prompt_float("Monthly Charges (ex., 29.85): ", 0.0)
    contract = prompt_choice(
        "Contract type (Month-to-month / One year / Two year): ",
        ["Month-to-month", "One year", "Two year"]
    )

    # Overwrite fields that exist in the dataset
    if "tenure" in row.columns:
        row.loc[row.index[0], "tenure"] = tenure
    if "MonthlyCharges" in row.columns:
        row.loc[row.index[0], "MonthlyCharges"] = monthly
    if "Contract" in row.columns:
        row.loc[row.index[0], "Contract"] = contract

    # set a simple estimate
    if "TotalCharges" in row.columns:
        row.loc[row.index[0], "TotalCharges"] = tenure * monthly

    return row


answer = input("\nWould you like to score a customer now? (y/n): ").strip().lower()
if answer == "y":
    # Create a one row input with a few user provided values
    user_row = build_user_input_row(X)

    # Predict churn probability if available
    if hasattr(best_model, "predict_proba"):
        churn_prob = best_model.predict_proba(user_row)[0][1]
    else:
        # Fallback
        churn_prob = float(best_model.predict(user_row)[0])

    predicted_class = "Churn" if churn_prob >= 0.5 else "No Churn"

    print("\nChurn Risk Result:")
    print(f"Model used: {best_model_name}")
    print(f"Predicted churn probability: {churn_prob:.3f}")
    print(f"Predicted outcome: {predicted_class}")

    if churn_prob >= 0.5:
        print("Recommendation: Flag this customer for retention outreach.")
    else:
        print("Recommendation: No immediate retention action needed.")
else:
    print("\nUser scoring skipped.")
