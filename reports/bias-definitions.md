# Bias Definitions for Resume Screening Project

## 1. Gender
- **Categories**: Male, Female, Non-binary/Other, Unknown
- **Privileged group**: Male
- **Unprivileged groups**: Female, Non-binary/Other
- **Justification**: In many hiring datasets, males historically have higher selection rates. Using male as privileged aligns with fairness evaluation conventions.

---

## 2. Ethnicity
- **Categories (example for India)**: SC/ST, OBC, General, Other
- **Privileged group**: General
- **Unprivileged groups**: SC/ST, OBC
- **Justification**: General category applicants typically have higher representation and fewer systemic barriers, so they are treated as the privileged baseline.

*(If working with US context, categories could be: White, Black, Hispanic, Asian, Other; privileged = White)*

---

## Notes
- If self-reported attributes are available, always prefer them.  
- If attributes are inferred (e.g., using name-based ethnicity inference), clearly mark them as **inferred** and be cautious in analysis.  
- These definitions will guide our fairness metrics such as disparate impact, demographic parity, and equal opportunity.


# Fairness Metrics for Resume Screening

This section defines the fairness metrics we will use to evaluate our ML model. Each metric is described in **plain English** and **mathematical formula**.

---

## 1. Disparate Impact (DI)
**Formula:**
DI = P(ŷ = 1 | A = unprivileged) / P(ŷ = 1 | A = privileged)


**Interpretation:**  
- DI ≈ 1 is ideal (selection rates are equal).  
- The “80% rule”: if DI < 0.8, the unprivileged group may be experiencing adverse impact.

---

## 2. Demographic Parity (Statistical Parity)
**Requirement:**  
P(ŷ = 1 | A = a) is equal ∀ groups a



**Interpretation:**  
- All groups should have the same probability of being selected, regardless of protected attributes.  
- Can also report **difference**:  
Δ = P(ŷ=1 | unprivileged) - P(ŷ=1 | privileged

- Δ ≈ 0 is ideal.

---

## 3. Equal Opportunity
**Focus:** True Positive Rate (TPR)  
**Formula:**  
EOD = TPR_unpriv - TPR_priv
= P(ŷ = 1 | Y = 1, A = unprivileged) - P(ŷ = 1 | Y = 1, A = privileged)


**Interpretation:**  
- Measures whether qualified candidates in unprivileged groups are selected at the same rate as the privileged group.  
- Goal: EOD ≈ 0.

---

## 4. Equalized Odds
**Requirement:**  
- Both TPR and FPR should be similar across groups.  

**Interpretation:**  
- Ensures that the model is fair for both positive and negative outcomes.  

---

## 5. Other Useful Statistics
- **Group counts** — number of individuals in each group  
- **Selection rates** — fraction of candidates selected  
- **TPR / FPR per group** — true positive / false positive rates  
- **Precision per group** — fraction of selected candidates who are actually qualified

