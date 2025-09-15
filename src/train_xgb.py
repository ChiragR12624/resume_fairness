# ===================================================
# src/train_xgb.py
# Train XGBoost on cleaned resumes dataset
# Fully functional version
# ===================================================

import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt

def main():
    # ---------- Load Data ----------
    data = pd.read_csv("data/cleaned_resumes.csv")
    print("Dataset shape:", data.shape)
    print("Columns:", data.columns.tolist())

    # ---------- Target & Features ----------
    target_col = "hire_label"
    if target_col not in data.columns:
        if "label" in data.columns:
            data.rename(columns={"label": target_col}, inplace=True)
        else:
            raise ValueError(f"Target column '{target_col}' not found! Available columns: {data.columns.tolist()}")

    X = data.drop(columns=[target_col, "candidate_id"])
    y = data[target_col]

    # ---------- One-hot encode categorical variables ----------
    X = pd.get_dummies(X, drop_first=True)

    # ---------- Sanitize column names for XGBoost ----------
    X.columns = [c.replace("[", "_")
                  .replace("]", "_")
                  .replace("<", "_")
                  .replace(">", "_")
                  .replace(",", "_")
                  .replace(";", "_")  # also replace semicolons
                 for c in X.columns]

    print("Feature matrix shape after encoding:", X.shape)

    # ---------- Train-Test Split ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------- Train XGBoost ----------
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=1,
        use_label_encoder=False
    )

    xgb.fit(X_train, y_train)

    # ---------- Predict & Evaluate ----------
    y_pred = xgb.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\n=== XGBoost Model Performance ===")
    print(f"Accuracy: {acc:.4f}")
    print(report)

    # ---------- Save Model ----------
    Path("models").mkdir(exist_ok=True)
    joblib.dump(xgb, "models/xgb_baseline.pkl")
    print("✅ Model saved at models/xgb_baseline.pkl")

    # ---------- Save Metrics Report ----------
    Path("reports").mkdir(exist_ok=True)
    report_path = Path("reports/metrics_xgb.txt")
    with open(report_path, "w") as f:
        f.write("=== XGBoost Model Performance ===\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)
    print(f"📄 Metrics report saved at {report_path}")

    # ---------- Save Feature Importance Plot ----------
    plt.figure(figsize=(10,6))
    plot_importance(xgb, max_num_features=10, importance_type="weight")
    plt.title("Top 10 Feature Importances (XGBoost)")
    plt.tight_layout()
    fig_path = Path("reports/feature_importance_xgb.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"📊 Feature importance plot saved at {fig_path}")


if __name__ == "__main__":
    main()
