"""
Day 6 – Transformer Model for Resume–Job Matching
-------------------------------------------------
This script fine-tunes DistilBERT (or BERT), extracts embeddings for resumes & jobs,
computes similarity, and finds top matches.

Deliverable: Fully functional training + embedding pipeline.
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW  # Use PyTorch's AdamW
from transformers import AutoTokenizer, AutoModel, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------
# 1. Config
# ------------------------------
MODEL_NAME = "distilbert-base-uncased"
DATA_PATH = "data/resumes_jobs.csv"  # CSV must have 'resume_text', 'job_text', 'label'
BATCH_SIZE = 8
EPOCHS = 2
LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# 2. Load Data
# ------------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
required_cols = {"resume_text", "job_text", "label"}
if not required_cols.issubset(set(df.columns)):
    raise ValueError(f"❌ CSV must have {required_cols}")

print(f"✅ Loaded dataset: {df.shape}")
print(df.head())

# ------------------------------
# 3. Dataset Class
# ------------------------------
class ResumeJobDataset(Dataset):
    def __init__(self, resumes, jobs, labels, tokenizer, max_len=128):
        self.resumes = resumes
        self.jobs = jobs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.resumes)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.resumes[idx],
            self.jobs[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ------------------------------
# 4. Load Tokenizer & Model
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModel.from_pretrained(MODEL_NAME)

class MatchingModel(torch.nn.Module):
    def __init__(self, base_model, hidden_size=768, num_labels=2):
        super(MatchingModel, self).__init__()
        self.bert = base_model
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return logits, loss

model = MatchingModel(base_model).to(DEVICE)

# ------------------------------
# 5. Train/Test Split
# ------------------------------
train_texts, test_texts, train_jobs, test_jobs, y_train, y_test = train_test_split(
    df["resume_text"].tolist(),
    df["job_text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
)

train_dataset = ResumeJobDataset(train_texts, train_jobs, y_train, tokenizer)
test_dataset = ResumeJobDataset(test_texts, test_jobs, y_test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ------------------------------
# 6. Optimizer & Scheduler
# ------------------------------
optimizer = AdamW(model.parameters(), lr=LR)
num_training_steps = EPOCHS * len(train_loader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# ------------------------------
# 7. Training Loop
# ------------------------------
print("\n🚀 Training started...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        logits, loss = model(input_ids, attention_mask, labels)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

# ------------------------------
# 8. Evaluation
# ------------------------------
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        logits, _ = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\n📊 Evaluation Results:")
print("Accuracy:", accuracy_score(all_labels, all_preds))
print(classification_report(all_labels, all_preds))

# ------------------------------
# 9. Embedding Extraction
# ------------------------------
def get_embeddings(texts, batch_size=8):
    model.eval()
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        with torch.no_grad():
            tokens = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
            tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
            outputs = base_model(**tokens)
            embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)

resume_embeddings = get_embeddings(df["resume_text"].tolist())
job_embeddings = get_embeddings(df["job_text"].tolist())

np.save("resume_embeddings.npy", resume_embeddings.numpy())
np.save("job_embeddings.npy", job_embeddings.numpy())
print("\n✅ Saved embeddings: resume_embeddings.npy, job_embeddings.npy")

# ------------------------------
# 10. Matching Function
# ------------------------------
similarity_scores = torch.matmul(resume_embeddings, job_embeddings.T) / (
    resume_embeddings.norm(dim=1)[:, None] * job_embeddings.norm(dim=1)[None, :]
)

def find_best_matches(resume_idx, top_n=3):
    scores = similarity_scores[resume_idx].numpy()
    top_indices = scores.argsort()[-top_n:][::-1]
    return [(df["job_text"].iloc[j], round(scores[j], 4)) for j in top_indices]

print("\n🎯 Example Matches:")
for i in range(min(3, len(df))):
    print(f"\nResume {i+1}: {df['resume_text'].iloc[i][:80]}...")
    matches = find_best_matches(i, top_n=3)
    for job, score in matches:
        print(f"   → {job[:80]}... (Score: {score})")
