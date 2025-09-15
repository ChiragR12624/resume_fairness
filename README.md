ResumeFairness — Bias detection & mitigation in resume screening

An explainable resume-screening pipeline that detects gender bias and applies mitigation (post-processing) to improve fairness while preserving predictive performance.

“This project investigates bias in machine learning models for resume screening, evaluates fairness using standard metrics, and applies algorithmic debiasing techniques to improve equity in automated hiring decisions.” 

Quick results
- Dataset: `data/sample_cleaned_resume.csv`
- Main model: `<model architecture / name>`  
- Key metrics (before → after):  
  - Overall AUC: `<AUC_before>` → `<AUC_after>`  
  - Demographic Parity Difference (gender): `<dp_before>` → `<dp_after>`  
  - Equalized Odds gap: `<eo_before>` → `<eo_after>`

Repo layout
- `src/` — training, evaluation, fairness scripts  
- `notebooks/` — EDA, training experiments, fairness analysis  
- `viz/` — Streamlit dashboard (`viz/streamlit_app.py`)  
- `reports/` — final report, metrics CSV, screenshots

Reproducibility
1. Create a venv and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
2. Run a smoke test (train a tiny model):
   `python src/quick_train.py --data data/sample_cleaned_resume.csv --out               models/demo_model.pt`
3. Run the dashboard locally:
   `cd viz
   streamlit run streamlit_app.py`

How the fairness pipeline works
-  Preprocessing: text cleaning → embedding via <method>
-  Classifier: <model> trained with seed 42
-  Fairness correction: post-processing via ThresholdOptimizer (equalized odds) or     reweighing (link to notebook). See `notebooks/fairness_mitigation.ipynb`.

Files of interest
-  `reports/final_report.pdf` — downloadable full write-up
-  `reports/metrics_summary.csv` — fold & group metrics used in the paper
-  `viz/streamlit_app.py` — interactive dashboard to compare before/after

Key metrics (Baseline → Mitigated)

-  Accuracy: 0.820 → 0.800
-  Fairness: 0.610 → 0.910


License
-  MIT License

Contact
-  Chirag Gowda — `chirag12624@gmail.com`

 

