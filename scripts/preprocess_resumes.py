import pandas as pd
import re
import json
from collections import Counter
import os


print("Current working directory:", os.getcwd())
print("Files in data/:", os.listdir("data"))



# 1) Load dataset
df = pd.read_csv("../data/synthetic_resumes.csv", low_memory=False)

# 2) Handle missing values
df['gender'] = df['gender'].fillna('unknown')
df['ethnicity'] = df['ethnicity'].fillna('unknown')
df['education'] = df['education'].fillna('unknown')
df['years_experience'] = df['years_experience'].fillna(0)
df['skills'] = df['skills'].fillna('')

# 3) Normalize skills text
# Replace separators with semicolon
df['skills'] = df['skills'].str.replace(r'[,\|/]+', ';', regex=True)
df['skills'] = df['skills'].str.replace(r';{2,}', ';', regex=True)
df['skills'] = df['skills'].str.lower().str.strip()

# Split into list and clean tokens
def clean_skill_token(tok):
    tok = tok.strip()
    tok = re.sub(r'[^\w\s\+\#\.\-]', '', tok)
    tok = re.sub(r'\s+', ' ', tok)
    return tok

df['skills_list'] = df['skills'].apply(lambda s: [clean_skill_token(t) for t in s.split(';') if t.strip()])

# Optional: map synonyms
skill_map = {
    'py': 'python',
    'python3': 'python',
    'ml': 'machine learning',
    'nlp': 'natural language processing',
    'js': 'javascript'
}
df['skills_list'] = df['skills_list'].apply(lambda lst: [skill_map.get(t, t) for t in lst])

# 4) Numeric/text features
df['num_skills'] = df['skills_list'].apply(len)
df['skills_text'] = df['skills_list'].apply(lambda lst: " ".join(lst))

# 5) Save cleaned dataset
df.to_csv("../data/cleaned_resumes.csv", index=False)
print("✅ File exists?", os.path.exists("data/cleaned_resumes.csv"))
print("✅ Cleaned dataset saved to data/cleaned_resumes.csv")

# 6) Save top-K skill vocabulary (optional)
all_skills = [skill for lst in df['skills_list'] for skill in lst]
top_k = [s for s,_ in Counter(all_skills).most_common(100)]
with open('models/skills_top_k.json', 'w') as f:
    json.dump(top_k, f, indent=2)
print("✅ Top-K skill vocabulary saved to models/skills_top_k.json")
