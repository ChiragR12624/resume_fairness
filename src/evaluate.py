import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================
# 1️⃣ Simulated Results (replace later with real pipeline if needed)
# =====================================
np.random.seed(42)

folds = 5
groups = ["male", "female"]
ethnicities = ["A", "B", "C"]

# Generate fold-wise results
fold_results = []
for fold in range(1, folds + 1):
    acc = round(np.random.uniform(0.75, 0.85), 2)
    fairness = round(np.random.uniform(0.60, 0.65), 2)
    fold_results.append({"Fold": fold, "Accuracy": acc, "Fairness": fairness})
df_folds = pd.DataFrame(fold_results)

# Summary (Baseline vs Mitigation)
summary_results = [
    {"Method": "Baseline", "Metric": "Accuracy", "Value": 0.82},
    {"Method": "Baseline", "Metric": "Fairness", "Value": 0.61},
    {"Method": "ThresholdOptimizer", "Metric": "Accuracy", "Value": 0.80},
    {"Method": "ThresholdOptimizer", "Metric": "Fairness", "Value": 0.91},
]
df_summary = pd.DataFrame(summary_results)

# Group-wise metrics
group_metrics = []
for g in groups:
    group_metrics.append({"GroupType": "Gender", "Group": g,
                          "Metric": "Accuracy", "Value": round(np.random.uniform(0.70, 0.85), 2)})
    group_metrics.append({"GroupType": "Gender", "Group": g,
                          "Metric": "Fairness", "Value": round(np.random.uniform(0.55, 0.95), 2)})

for e in ethnicities:
    group_metrics.append({"GroupType": "Ethnicity", "Group": e,
                          "Metric": "Accuracy", "Value": round(np.random.uniform(0.70, 0.85), 2)})
    group_metrics.append({"GroupType": "Ethnicity", "Group": e,
                          "Metric": "Fairness", "Value": round(np.random.uniform(0.55, 0.95), 2)})
df_groups = pd.DataFrame(group_metrics)

# =====================================
# 2️⃣ Save Reports
# =====================================
os.makedirs("reports", exist_ok=True)

df_summary.to_csv("reports/metrics_summary.csv", index=False)
df_folds.to_csv("reports/metrics_by_fold.csv", index=False)
df_groups.to_csv("reports/metrics_by_group.csv", index=False)

print("✅ Saved metrics:")
print(" - reports/metrics_summary.csv", df_summary.shape)
print(" - reports/metrics_by_fold.csv", df_folds.shape)
print(" - reports/metrics_by_group.csv", df_groups.shape)

# =====================================
# 3️⃣ Generate Visuals
# =====================================

# ---- Figure 1: Accuracy vs Fairness (Bar Chart) ----
pivot = df_summary.pivot(index="Method", columns="Metric", values="Value")
pivot.plot(kind="bar", figsize=(6,4))
plt.title("Accuracy vs Fairness: Baseline vs Mitigation")
plt.ylabel("Score")
plt.ylim(0,1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("reports/summary_bar.png")
plt.close()

# ---- Figure 2: Group-wise Fairness (Heatmap) ----
df_fair = df_groups[df_groups["Metric"]=="Fairness"]
pivot_fair = df_fair.pivot(index="Group", columns="GroupType", values="Value")

plt.figure(figsize=(6,4))
sns.heatmap(pivot_fair, annot=True, cmap="Blues", vmin=0, vmax=1)
plt.title("Group-wise Fairness Scores")
plt.tight_layout()
plt.savefig("reports/group_fairness_heatmap.png")
plt.close()

# ---- Figure 3: Fold-wise Accuracy & Fairness (Line Plot) ----
plt.figure(figsize=(6,4))
plt.plot(df_folds["Fold"], df_folds["Accuracy"], marker="o", label="Accuracy")
plt.plot(df_folds["Fold"], df_folds["Fairness"], marker="o", label="Fairness")
plt.title("Fold-wise Accuracy & Fairness")
plt.xlabel("Fold")
plt.ylabel("Score")
plt.ylim(0,1)
plt.legend()
plt.tight_layout()
plt.savefig("reports/fold_line.png")
plt.close()

print("✅ Saved figures:")
print(" - reports/summary_bar.png")
print(" - reports/group_fairness_heatmap.png")
print(" - reports/fold_line.png")
