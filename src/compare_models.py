# ===================================================
# src/compare_models.py
# Train & compare Logistic Regression and XGBoost
# Fully functional version
# ===================================================

import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
import numpy as np

def sanitize_columns(df):
    df.columns = [c.replace("[", "_")
                  .replace("]", "_")
                  .replace("<", "_")
                  .replace(">", "_")
                  .replace(",", "_")
                  .replace(";", "_")
                  for c in df.columns]
    return df

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
            raise ValueError(f"Target column '{target_col}' not found! Columns: {data.columns.tolist()}")

    X = data.drop(columns=[target_col, "candidate_id"])
    y = data[target_col]

    # ---------- One-hot encode categorical variables ----------
    X = pd.get_dummies(X, drop_first=True)
    X = sanitize_columns(X)
    print("Feature matrix shape after encoding:", X.shape)

    # ---------- Train-Test Split ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------- Standardize features for Logistic Regression ----------
    scaler = StandardScaler()
    X_train_lr = scaler.fit_transform(X_train)
    X_test_lr = scaler.transform(X_test)

    # ---------- Train Logistic Regression ----------
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_train_lr, y_train)

    # ---------- Predict & Evaluate Logistic Regression ----------
    y_pred_lr = logreg.predict(X_test_lr)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    report_lr = classification_report(y_test, y_pred_lr)

    print("\n=== Logistic Regression Performance ===")
    print(f"Accuracy: {acc_lr:.4f}")
    print(report_lr)

    # Save Logistic Regression model
    Path("models").mkdir(exist_ok=True)
    joblib.dump(logreg, "models/logreg.pkl")
    joblib.dump(scaler, "models/logreg_scaler.pkl")
    print("✅ Logistic Regression model saved at models/logreg.pkl")

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

    # ---------- Predict & Evaluate XGBoost ----------
    y_pred_xgb = xgb.predict(X_test)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    report_xgb = classification_report(y_test, y_pred_xgb)

    print("\n=== XGBoost Performance ===")
    print(f"Accuracy: {acc_xgb:.4f}")
    print(report_xgb)

    # Save XGBoost model
    joblib.dump(xgb, "models/xgb.pkl")
    print("✅ XGBoost model saved at models/xgb.pkl")

    # ---------- Save Metrics Reports ----------
    Path("reports").mkdir(exist_ok=True)
    with open("reports/metrics_logreg.txt", "w") as f:
        f.write("=== Logistic Regression Performance ===\n")
        f.write(f"Accuracy: {acc_lr:.4f}\n\n")
        f.write(report_lr)

    with open("reports/metrics_xgb.txt", "w") as f:
        f.write("=== XGBoost Performance ===\n")
        f.write(f"Accuracy: {acc_xgb:.4f}\n\n")
        f.write(report_xgb)

    print("📄 Metrics reports saved at reports/")

    # ---------- Save Feature Importance Plot for XGBoost ----------
    plt.figure(figsize=(10,6))
    plot_importance(xgb, max_num_features=10, importance_type="weight")
    plt.title("Top 10 Feature Importances (XGBoost)")
    plt.tight_layout()
    plt.savefig("reports/feature_importance_xgb.png")
    plt.close()
    print("📊 Feature importance plot saved at reports/feature_importance_xgb.png")


if __name__ == "__main__":
    main()
