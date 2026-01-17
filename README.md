# Credit Risk Prediction 💳

This project focuses on building a machine learning model to predict the probability that somebody will experience financial distress in the next two years. It utilizes historical credit data to perform deep exploratory data analysis (EDA), data cleaning, and predictive modeling.

## 🚀 Overview
The primary goal is to identify high-risk borrowers using features like monthly utilization, age, debt ratio, and past-due history. The project handles significant data challenges including missing values, extreme outliers, and class imbalance.

## 📊 Dataset Information
The project uses the "Give Me Some Credit" dataset. Key features include:
- **Target Variable:** `SeriousDlqin2yrs` (Person experienced 90 days past due delinquency or worse).
- **Predictors:** - `monthly_utilization`: Total balance on credit cards and personal lines of credit.
  - `age`: Age of borrower in years.
  - `DebtRatio`: Monthly debt payments, alimony, living costs divided by monthly gross income.
  - `MonthlyIncome`: Monthly income.
  - `NumberOfOpenCreditLinesAndLoans`: Number of open loans (installment and real estate).
  - `NumberOfTime30-59DaysPastDueNotWorse`: Number of times borrower has been 30-59 days past due.

## 🛠️ Data Preprocessing & Cleaning
The notebook implements a robust cleaning pipeline:
* **Missing Value Imputation:** Handled missing `MonthlyIncome` and `NumberOfDependents` using median-based imputation.
* **Outlier Treatment:** Addressed extreme values in `DebtRatio` and `MonthlyIncome` by capping them at the 95th/97.5th percentiles.
* **Feature Engineering:** Cleaned "error" values in delinquency counts (e.g., replacing values > 13 with representative high-risk buckets).
* **Data Saving:** Processed data is exported to `../data/processed/credit_model_df.csv` for modeling.

## 🤖 Modeling
The project employs a **Logistic Regression** model specifically tuned for imbalanced data:
- **Algorithm:** Logistic Regression.
- **Configuration:** `class_weight='balanced'`, `max_iter=1000`.
- **Serialization:** The trained model and feature list are saved using `joblib` as `credit_risk_logreg.pkl`.

## ⚙️ Requirements
To run this notebook, you will need:
- Python 3.x
- pandas
- numpy
- scikit-learn
- joblib

```bash
pip install pandas numpy scikit-learn joblib
