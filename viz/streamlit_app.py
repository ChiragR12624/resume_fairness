import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import os

st.set_page_config(page_title="Fairness Dashboard", layout="wide")

st.title("📊 Fairness Dashboard — Baseline vs Mitigation")

# ==============================
# 1️⃣ Load Data
# ==============================
def load_csv(path, fallback_df):
    if os.path.exists(path):
        return pd.read_csv(path)
    return fallback_df

df_summary = load_csv("reports/metrics_summary.csv",
                      pd.DataFrame({"Method": [], "Metric": [], "Value": []}))
df_folds = load_csv("reports/metrics_by_fold.csv",
                    pd.DataFrame({"Fold": [], "Accuracy": [], "Fairness": []}))
df_groups = load_csv("reports/metrics_by_group.csv",
                     pd.DataFrame({"GroupType": [], "Group": [], "Metric": [], "Value": []}))

# ==============================
# 2️⃣ Tabs
# ==============================
tab1, tab2, tab3 = st.tabs(["📊 Summary", "🔍 By Fold", "⚖️ Group Comparisons"])

# ----- Tab 1: Summary -----
with tab1:
    st.subheader("Model Summary Metrics")
    st.dataframe(df_summary)

    if not df_summary.empty:
        st.bar_chart(df_summary.pivot(index="Metric", columns="Method", values="Value"))

        # Download CSV
        csv = df_summary.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Summary CSV", csv, "metrics_summary.csv")

# ----- Tab 2: Fold-wise -----
with tab2:
    st.subheader("Fold-wise Metrics")
    st.dataframe(df_folds)

    if not df_folds.empty:
        st.line_chart(df_folds.set_index("Fold"))

        csv = df_folds.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Fold CSV", csv, "metrics_by_fold.csv")

# ----- Tab 3: Group Comparisons -----
with tab3:
    st.subheader("Group-wise Metrics")

    if not df_groups.empty:
        group_type = st.selectbox("Select Group Type", df_groups["GroupType"].unique())
        df_filtered = df_groups[df_groups["GroupType"] == group_type]

        # Show table
        st.dataframe(df_filtered)

        # Plot
        pivot = df_filtered.pivot(index="Group", columns="Metric", values="Value")
        st.bar_chart(pivot)

        # Download CSV
        csv = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(f"📥 Download {group_type} CSV", csv, f"metrics_by_{group_type.lower()}.csv")

# ==============================
# 3️⃣ Downloadable Plot Example
# ==============================
if not df_summary.empty:
    fig, ax = plt.subplots()
    df_summary.pivot(index="Metric", columns="Method", values="Value").plot(kind="bar", ax=ax)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.download_button("📸 Download Plot", buf.getvalue(), "fairness_plot.png", "image/png")
