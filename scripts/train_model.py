# scripts/train_model.py
import os
import pickle
import pandas as pd
import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler   # 🔄 swapped StandardScaler → RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ========== Paths ==========
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_IN = os.path.join(ROOT, "data", "cleaned_resumes.csv")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ========== Load ==========
df = pd.read_csv(DATA_IN)
print("Loaded dataset:", df.shape)

# ========== Preprocessing: group rare categories ==========
def group_rare(df, col, min_count=20):
    counts = df[col].value_counts()
    rare = counts[counts < min_count].index
    df[col] = df[col].replace({v: "other" for v in rare})
    return df

for col in ["gender", "ethnicity", "education"]:
    if col in df.columns:
        df = group_rare(df, col)

# ========== Train/Test Split ==========
X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ========== Feature Groups ==========
num_cols = ["years_experience", "num_skills"]
cat_cols = ["gender", "ethnicity", "education"]
text_col = "skills_text" if "skills_text" in df.columns else None

# ========== Pipelines ==========
num_pipe = Pipeline([("scaler", RobustScaler())])   # 🔄 swapped scaler
cat_pipe = Pipeline([("ohe", OneHotEncoder(drop="first", handle_unknown="ignore"))])

transformers = [
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
]

if text_col:
    tfidf_pipe = Pipeline([("tfidf", TfidfVectorizer(min_df=5, ngram_range=(1, 2)))])
    transformers.append(("skills", tfidf_pipe, text_col))

preprocessor = ColumnTransformer(transformers)

# Final pipeline
pipeline = Pipeline([
    ("pre", preprocessor),
    ("clf", LogisticRegression(max_iter=1000))
])

# ========== Train ==========
pipeline.fit(X_train, y_train)

# Save pipeline
with open(os.path.join(MODELS_DIR, "model_pipeline.pkl"), "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model trained and saved at models/model_pipeline.pkl")

# ========== Evaluate ==========
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("\nAccuracy:", acc)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ========== Fairness Checks ==========
def group_accuracy(X, y_true, y_pred, group_col):
    df_eval = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        group_col: X[group_col].values
    })
    results = {}
    for group in df_eval[group_col].unique():
        mask = df_eval[group_col] == group
        acc = accuracy_score(df_eval.loc[mask, "y_true"], df_eval.loc[mask, "y_pred"])
        results[group] = acc
    return results

print("\n📊 Group Accuracy by Gender:")
print(group_accuracy(X_test, y_test, y_pred, "gender"))

print("\n📊 Group Accuracy by Ethnicity:")
print(group_accuracy(X_test, y_test, y_pred, "ethnicity"))
