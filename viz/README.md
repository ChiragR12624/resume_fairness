# viz/

Streamlit dashboard for Day 13 (Fairness Dashboard)

## Run locally
1. From repo root:
   python -m venv .venv
   source .venv/bin/activate    # Windows: .venv\Scripts\activate
2. pip install -r viz/requirements.txt
3. streamlit run viz/streamlit_app.py

## Input data
Place Day 12 outputs in:
- reports/metrics_by_fold.csv   (preferred)
- reports/metrics_summary.csv   (fallback)
