# Synthetic Resume Dataset Plan

This document describes the structure of the CSV dataset we will use for the Bias & Fairness in Resume Screening project.

---

## Columns

| Column Name        | Description |
|-------------------|-------------|
| candidate_id       | Unique identifier for each candidate |
| gender             | Protected attribute: male/female/non-binary/unknown |
| ethnicity          | Protected attribute: context-specific categories (e.g., White/Black/Asian/Hispanic/Other or SC/ST/OBC/General) |
| education_level    | Highest degree: bachelors, masters, phd |
| years_experience   | Numeric: total years of work experience |
| skills             | Semi-colon separated list of skills (e.g., Python;SQL;ML) |
| job_title          | The role candidate is applying for |
| resume_text        | Optional: raw resume text, stripped of PII |
| label              | Target variable: 1 = suitable/hire, 0 = not suitable |
| notes              | Metadata, e.g., whether data is synthetic or inferred |

---

## Data Quality Rules

1. Missing protected attributes → mark as `unknown` (do not drop rows unless documented).  
2. Do not store raw PII (names, emails, phone numbers) in the repository.  
   - If you use them for local experiments, add `data/.gitignore` and keep those CSVs outside version control.  
3. Ensure categories are consistent:
   - `gender`: male, female, non-binary/other, unknown  
   - `ethnicity`: define per context (e.g., India or US)  
4. Skills column is semi-colon separated; no duplicates within a candidate.  
5. Labels should be binary (0 or 1) for ML classification.

---

## Example Row (CSV)

candidate_id,gender,ethnicity,education_level,years_experience,skills,job_title,resume_text,label,notes
1,female,SC/ST,bachelors,3,Python;SQL;Data Analysis,Data Analyst,"",1,synthetic
2,male,General,masters,5,Java;AWS;DevOps,Software Engineer,"",0,synthetic