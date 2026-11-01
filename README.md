# Fairness-Aware Machine Learning Framework for Bias Mitigation in AI-Driven Recruitment Systems

<p align="center">
  <b>Investigating Algorithmic Bias and Promoting Responsible AI in Automated Recruitment</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange" />
  <img src="https://img.shields.io/badge/Fairlearn-Fairness%20Evaluation-green" />
  <img src="https://img.shields.io/badge/IBM-AIF360-purple" />
  <img src="https://img.shields.io/badge/Research-Published-success" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

## 1. Research Overview

Artificial Intelligence is increasingly transforming recruitment processes through automated resume screening, candidate ranking, and predictive hiring systems. Despite their efficiency, these systems introduce significant ethical challenges when machine learning models learn discriminatory patterns embedded in historical recruitment data.

This research investigates algorithmic bias in automated recruitment systems and explores fairness-aware machine learning approaches to identify, quantify, and mitigate demographic disparities in candidate evaluation.

The project implements an experimental framework that integrates conventional machine learning classification with statistical fairness assessment and bias mitigation techniques.

The central research objective is to examine the relationship between predictive performance and algorithmic fairness, investigating whether fairness-aware interventions can improve equitable treatment across demographic groups while maintaining acceptable classification performance.

**Research Focus:** Responsible AI | Algorithmic Fairness | Bias Mitigation | Machine Learning | Automated Recruitment

## 2. Research Problem

Conventional machine learning models are generally optimized for predictive performance. However, high predictive accuracy does not necessarily imply equitable outcomes.

In automated recruitment, biased training data and learned associations can produce systematically different predictions across demographic groups.

This research addresses the following questions:

- How can demographic bias be identified in machine learning-based resume screening?
- Which statistical fairness metrics can effectively quantify disparities?
- How do fairness-aware mitigation techniques influence predictive performance?
- What trade-offs emerge between model accuracy and demographic fairness?

## 3. Research Objectives

1. Investigate potential demographic bias in automated resume-screening systems.
2. Develop baseline machine learning classification models.
3. Evaluate model performance and demographic fairness using established statistical measures.
4. Implement bias mitigation techniques using fairness-aware machine learning libraries.
5. Compare baseline and mitigated models through quantitative evaluation.
6. Examine the relationship between predictive performance and algorithmic fairness.

## 4. Methodology

The framework follows a structured experimental pipeline.

### 4.1 Dataset Preparation

A synthetic recruitment dataset is used to simulate candidate information and recruitment-related classification outcomes.

The dataset undergoes preprocessing and feature preparation before being used for model development and fairness evaluation.

The use of synthetic data enables controlled experimentation without exposing actual candidate information.

### 4.2 Baseline Model Development

Supervised machine learning techniques are employed to establish baseline predictive performance.

The experimental framework explores classification approaches including:

- Logistic Regression
- XGBoost

### 4.3 Fairness Assessment

The framework incorporates statistical fairness evaluation to investigate disparities across demographic groups.

Key evaluation dimensions include:

- **Disparate Impact:** Examines differences in favorable prediction rates between demographic groups.
- **Equal Opportunity:** Evaluates differences in true positive rates across protected groups.
- **Predictive Performance:** Measures classification performance using appropriate evaluation metrics.

### 4.4 Bias Mitigation

Fairness-aware interventions are explored using established open-source responsible AI frameworks.

- **Fairlearn:** Used to investigate fairness assessment and mitigation through techniques such as ThresholdOptimizer.
- **IBM AI Fairness 360:** Provides additional capabilities for evaluating and mitigating algorithmic bias, including reweighing approaches.

### 4.5 Comparative Evaluation

Baseline and mitigated models are evaluated to understand:

- Changes in classification performance.
- Improvements in demographic fairness.
- Effectiveness of mitigation interventions.
- Trade-offs between predictive performance and fairness.

## 5. Experimental Results

The experimental evaluation demonstrates the importance of incorporating fairness considerations into conventional machine learning pipelines.

### Baseline vs. Mitigated Model

| Evaluation Metric | Baseline | Mitigated |
| ----------------- | -------: | --------: |
| Accuracy          |    0.820 |     0.800 |
| Fairness Score\*  |    0.610 |     0.910 |

*The fairness score must be interpreted according to the specific metric and calculation implemented in the experimental pipeline.*

### Key Observations

- Classification accuracy changed from 82% to 80% following mitigation.
- The reported fairness score increased from 0.61 to 0.91.
- The experimental results motivate evaluating fairness alongside predictive performance rather than relying exclusively on accuracy.

These results are specific to the experimental dataset and evaluation configuration and should not be interpreted as evidence of universal fairness or real-world hiring effectiveness.

## 6. Technology Stack

| Component             | Technologies                   |
| --------------------- | ------------------------------ |
| Programming           | Python                         |
| Machine Learning      | Scikit-learn, XGBoost          |
| Fairness Assessment   | Fairlearn, IBM AI Fairness 360 |
| Data Processing       | Pandas, NumPy                  |
| Visualization         | Matplotlib, Seaborn            |
| Interactive Dashboard | Streamlit                      |
| Experimentation       | Jupyter Notebook               |

## 7. Repository Structure

```text
resume_fairness/
│
├── data/                   # Recruitment dataset
│
├── models/                 # Model artifacts
│
├── notebooks/              # Research experiments and analysis
│
├── reports/                # Experimental reports and metrics
│
├── scripts/                # Execution and utility scripts
│
├── src/                    # Core ML and fairness implementation
│
├── viz/                    # Interactive Streamlit dashboard
│
├── requirements.txt        # Project dependencies
│
├── .gitignore
│
└── README.md
```

## 8. Reproducibility

The implementation is publicly available to support transparency and further experimentation.

### Clone the Repository

```bash
git clone https://github.com/ChiragR12624/resume_fairness.git

cd resume_fairness
```

### Set Up the Environment

```bash
python -m venv venv
```

Activate the environment:

**macOS / Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch the Interactive Dashboard

```bash
streamlit run viz/streamlit_app.py
```

Refer to the notebooks and source scripts for the complete experimental workflow and implementation details.

## 9. Research Contributions

This research contributes to the exploration of responsible machine learning in high-impact decision-making environments.

Its contributions include:

- A structured experimental framework for investigating algorithmic bias in automated recruitment.
- Integration of machine learning classification with statistical fairness evaluation.
- Exploration of fairness-aware mitigation techniques using established open-source frameworks.
- Comparative analysis of predictive performance and fairness before and after mitigation.
- An openly accessible implementation supporting reproducibility and further research.

## 10. Ethical Considerations and Limitations

Algorithmic fairness is a multidimensional concept, and optimizing one fairness metric does not necessarily guarantee equitable outcomes across all demographic groups.

The experimental framework uses synthetic recruitment data, which may not fully capture the complexity of real-world recruitment processes.

Furthermore, fairness interventions can introduce trade-offs between different statistical fairness criteria and predictive performance.

The framework is intended for research and educational purposes. It is not designed to make autonomous employment decisions or replace human oversight in recruitment.

## 11. Future Research Directions

Potential extensions include:

- Investigating intersectional fairness across multiple protected attributes.
- Exploring explainable AI techniques for understanding recruitment model predictions.
- Evaluating additional pre-processing, in-processing and post-processing mitigation strategies.
- Investigating fairness under distribution shifts.
- Extending the framework to appropriately sourced real-world datasets.
- Developing comprehensive fairness-monitoring and visualization capabilities.

## 12. Research Publication

**Fairness-Aware Machine Learning Framework for Bias Mitigation in AI-Driven Recruitment Systems**

This project is associated with a peer-reviewed research publication investigating algorithmic bias and fairness-aware mitigation in automated recruitment systems.

The implementation is open-sourced to encourage transparency, reproducibility and further research into responsible AI.

## 13. Author

**Chirag R. Gowda**

Research Interests: Machine Learning, Responsible AI, Algorithmic Fairness, Explainable AI and Deep Learning.

GitHub: [https://github.com/ChiragR12624](https://github.com/ChiragR12624)

---

If you find this research useful, consider starring the repository and contributing to further exploration of fairness-aware machine learning.
