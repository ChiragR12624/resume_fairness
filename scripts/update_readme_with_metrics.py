#!/usr/bin/env python3
"""
Reads reports/metrics_summary.csv (Method,Metric,Value)
and injects a small metrics block into README.template.md producing README.md.
"""
import pandas as pd
from pathlib import Path
import sys

CSV = Path("reports/metrics_summary.csv")
TEMPLATE = Path("README.template.md")
OUT = Path("README.md")

if not CSV.exists():
    print(f"ERROR: {CSV} not found. Place your metrics file at {CSV}.")
    sys.exit(1)

df = pd.read_csv(CSV)
required = {"Method", "Metric", "Value"}
if not required.issubset(set(df.columns)):
    print(f"ERROR: CSV must contain columns: {required}. Found: {list(df.columns)}")
    sys.exit(1)

# Pivot so we have metrics x methods
pivot = df.pivot_table(index="Metric", columns="Method", values="Value", aggfunc="first")

# pick baseline / mitigated method names
methods = list(pivot.columns)
baseline = "Baseline" if "Baseline" in methods else methods[0] if methods else None
mitigated = "ThresholdOptimizer" if "ThresholdOptimizer" in methods else (methods[1] if len(methods) > 1 else None)

def fmt(x):
    # format numeric values to 3 decimals, else show as-is
    try:
        if pd.isna(x):
            return "N/A"
        return f"{float(x):.3f}"
    except Exception:
        return str(x)

lines = []
lines.append("### Key metrics (Baseline → Mitigated)")
lines.append("")
if baseline is None:
    lines.append("_No method columns found in CSV._")
else:
    for metric in pivot.index:
        before = pivot.loc[metric, baseline] if baseline in pivot.columns else None
        after = pivot.loc[metric, mitigated] if (mitigated and mitigated in pivot.columns) else None
        lines.append(f"- **{metric}**: {fmt(before)} → {fmt(after)}")

metrics_md = "\n".join(lines) + "\n"

# Read template
if TEMPLATE.exists():
    tpl = TEMPLATE.read_text()
else:
    print("WARNING: README.template.md not found — trying to read existing README.md as template.")
    tpl = OUT.read_text() if OUT.exists() else ""
    if tpl == "":
        print("ERROR: No template to update. Create README.template.md first.")
        sys.exit(1)

if "<!-- METRICS-START -->" in tpl:
    # replace the whole block between start and end markers
    before, rest = tpl.split("<!-- METRICS-START -->", 1)
    _, after = rest.split("<!-- METRICS-END -->", 1) if "<!-- METRICS-END -->" in rest else ("", "")
    new_readme = before + "<!-- METRICS-START -->\n" + metrics_md + "<!-- METRICS-END -->" + after
else:
    # fallback: insert after the "## Quick results" heading (first occurrence)
    if "## Quick results" in tpl:
        new_readme = tpl.replace("## Quick results", "## Quick results\n\n" + metrics_md, 1)
    else:
        # append metrics at the top if nothing else
        new_readme = metrics_md + "\n" + tpl

OUT.write_text(new_readme)
print("README.md generated/updated successfully.")