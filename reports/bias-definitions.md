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
