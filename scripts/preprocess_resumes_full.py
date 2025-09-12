# ===============================
# Day 2: Resume Dataset Preprocessing Pipeline
# ===============================

import pandas as pd
import re
import json
from collections import Counter
import os

# -------------------------------
# 0️⃣ Ensure folders exist
# -------------------------------
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# -------------------------------
# 1️⃣ Load dataset
# -------------------------------
input_path = "data/synthetic_resumes.csv"  # adjust if needed
if not os.path.exists(input_path):
    raise FileNotFoundError(f"{input_path} not found! Run your synthetic generator first.")

df = pd.read_csv(input_path, low_memory=False)
print("✅ Dataset loaded. Shape:", df.shape)
print(df.head(), "\n")

# -------------------------------
# 2️⃣ Handle missing values
# -------------------------------
df['gender'] = df['gender'].fillna('unknown')
df['ethnicity'] = df['ethnicity'].fillna('unknown')
df['education'] = df['education'].fillna('unknown')
df['years_experience'] = df['years_experience'].fillna(0)
df['skills'] = df['skills'].fillna('')

# -------------------------------
# 3️⃣ Normalize skills column
# -------------------------------
df['skills'] = df['skills'].astype(str)
df['skills'] = df['skills'].str.replace(r'[,\|/]+', ';', regex=True)  # normalize separators
df['skills'] = df['skills'].str.replace(r';{2,}', ';', regex=True)   # remove duplicates
df['skills'] = df['skills'].str.lower().str.strip()

def clean_skill_token(tok):
    tok = tok.strip()
    tok = re.sub(r'[^\w\s\+\#\.\-]', '', tok)  # keep +, #, ., -
    tok = re.sub(r'\s+', ' ', tok)
    return tok

df['skills_list'] = df['skills'].apply(
    lambda s: [clean_skill_token(t) for t in s.split(';') if t.strip()]
)

# Optional: map common synonyms
skill_map = {
    'py': 'python',
    'python3': 'python',
    'ml': 'machine learning',
    'nlp': 'natural language processing',
    'js': 'javascript'
}
df['skills_list'] = df['skills_list'].apply(
    lambda lst: [skill_map.get(t, t) for t in lst]
)

# -------------------------------
# 4️⃣ Generate numeric/text features
# -------------------------------
df['num_skills'] = df['skills_list'].apply(len)
df['skills_text'] = df['skills_list'].apply(lambda lst: " ".join(lst))

# -------------------------------
# 5️⃣ Save cleaned dataset
# -------------------------------
output_path = "data/cleaned_resumes.csv"
df.to_csv(output_path, index=False)
print(f"✅ Cleaned dataset saved to {output_path}")

# -------------------------------
# 6️⃣ Save top-K skill vocabulary
# -------------------------------
all_skills = [skill for lst in df['skills_list'] for skill in lst]
top_k = [s for s, _ in Counter(all_skills).most_common(100)]  # top 100

with open("models/skills_top_k.json", "w", encoding="utf-8") as f:
    json.dump(top_k, f, indent=2, ensure_ascii=False)

print("✅ Top-K skill vocabulary saved to models/skills_top_k.json")

# -------------------------------
# 7️⃣ Quick QA / check
# -------------------------------
print("\n📊 Sample processed skills:")
print(df[['skills_list', 'num_skills', 'skills_text']].head(), "\n")

print("📊 Top 20 skills frequency:")
print(Counter(all_skills).most_common(20))


# -------------------------------
# 8️⃣ Save small sample for GitHub
# -------------------------------
sample_path = "data/sample_cleaned_resumes.csv"
df.sample(n=min(20, len(df)), random_state=42).to_csv(sample_path, index=False)
print(f"✅ Sample dataset saved to {sample_path} (safe to commit to GitHub)")



# -------------------------------
# 9️⃣ Quick Exploratory Bias Checks
# -------------------------------
print("\n📊 Label distribution by gender:")
print(df.groupby('gender')['label'].mean())

print("\n📊 Label distribution by ethnicity:")
print(df.groupby('ethnicity')['label'].mean())

print("\nCounts per group (gender):")
print(df['gender'].value_counts())

print("\nCounts per group (ethnicity):")
print(df['ethnicity'].value_counts())
