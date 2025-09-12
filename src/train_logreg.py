# ===================================================
# src/train_logreg.py
# Project: Bias & Fairness in Resume Screening
# Day 4: Baseline ML Model (Logistic Regression)
# ===================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10,6)

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
csv_path = '/Users/chiragrgowda/Documents/resume-fairness/data/cleaned_resumes.csv'
df = pd.read_csv(csv_path)

# Use consistent target column
if 'label' in df.columns:
    df.rename(columns={'label':'hire_label'}, inplace=True)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# -----------------------------
# Step 2: Select Features & Target
# -----------------------------
y = df['hire_label']

# Features: numeric + categorical
features = ['years_experience', 'num_skills', 'education', 'gender', 'ethnicity']
X = df[features]

# One-hot encode categorical features
X = pd.get_dummies(X, columns=['education', 'gender', 'ethnicity'], drop_first=True)
print("Feature shape after encoding:", X.shape)

# -----------------------------
# Step 3: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# -----------------------------
# Step 4: Train Logistic Regression
# -----------------------------
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)
print("Training complete!")

# -----------------------------
# Step 5: Model Evaluation
# -----------------------------
y_pred = logreg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n=== Baseline Logistic Regression Performance ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# -----------------------------
# Step 6: Feature Importance
# -----------------------------
feature_importance = pd.Series(logreg.coef_[0], index=X.columns).sort_values(ascending=False)
print("\nTop features influencing hire decisions:")
print(feature_importance.head(10))

# -----------------------------
# Step 7: Save the Trained Model
# -----------------------------
model_folder = '/Users/chiragrgowda/Documents/resume-fairness/models'
os.makedirs(model_folder, exist_ok=True)

model_path = os.path.join(model_folder, 'logistic_regression_baseline.pkl')
joblib.dump(logreg, model_path)
print(f"\nModel saved at {model_path}")
