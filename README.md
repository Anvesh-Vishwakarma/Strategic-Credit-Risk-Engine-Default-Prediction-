# Credit Risk Prediction 💳
**Web App Link** - https://strategic-credit-risk-engine-default.onrender.com

This project focuses on building a machine learning model to predict the probability that somebody will experience financial distress in the next two years. It utilizes historical credit data to perform deep exploratory data analysis (EDA), data cleaning, and predictive modeling.

## Overview
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
  - 
## 📌 Features
* Real-time Risk Prediction: Instant classification of borrowers based on their financial history.
* Explainable AI (XAI): Provides Odds Ratios for each feature, allowing users to understand which factors (e.g., Debt Ratio, Late Payments) most influence the risk score.
* Robust Data Cleaning: Automated pipeline to handle missing values, cap outliers at the 99th percentile, and bucket high-frequency categories for better model stability.
* Imbalance Handling: Utilizes "Balanced" class weights and SMOTE (Synthetic Minority Over-sampling Technique) to ensure the model accurately identifies rare high-risk events.
* Performance Visualization: Built-in ROC curve demonstration to track model discriminative power.

## 🧠 Problem Statement
Financial institutions lose billions annually due to credit defaults. Traditional manual underwriting is:

* Slow: Hindering the customer experience.
* Subjective: Leading to inconsistent risk thresholds.
* Inefficient: Difficult to scale across thousands of applications.

Identifying "Serious Delinquency" (defaulting within 2 years) is critical for maintaining bank liquidity and protecting investor interests.

## 💡 Solution Overview
This project implements a Logistic Regression pipeline to automate credit scoring. The solution uses a data-driven approach: it analyzes 150,000 historical records to learn patterns between borrower behavior and default risk. It handles real-world data issues like missing income records and extreme outliers before feeding them into a model optimized for imbalanced datasets.

## 🛠️ Data Preprocessing & Cleaning
The notebook implements a robust cleaning pipeline:
* **Missing Value Imputation:** Handled missing `MonthlyIncome` and `NumberOfDependents` using median-based imputation.
* **Outlier Treatment:** Addressed extreme values in `DebtRatio` and `MonthlyIncome` by capping them at the 95th/97.5th percentiles.
* **Feature Engineering:** Cleaned "error" values in delinquency counts (e.g., replacing values > 13 with representative high-risk buckets).
* **Data Saving:** Processed data is exported to `../data/processed/credit_model_df.csv` for modeling.

## 🤖 Model Explainability
Each feature coefficient represents its impact on default risk:

* **Positive coefficient** → Increases default probability
* **Negative coefficient** → Decreases default probability

Odds ratios were used to explain how a unit increase in a feature changes default risk.

## 🎯 Business Decision Threshold
Instead of using a default 0.5 threshold, a 0.4 probability cutoff was selected to:

* Improve recall of high-risk customers
* Balance false positives and false negatives
* Reflect real-world credit approval trade-offs

## 🖥️ Live Demo (Streamlit)
The model is deployed as an interactive web application where users can:

* Input customer details
* View default probability
* Receive a credit risk decision (Low / High risk)

## 🚀 How to run 

### 1️. Clone the Repository
```
https://github.com/Anvesh-Vishwakarma/Strategic-Credit-Risk-Engine-Default-Prediction-.git
```
### 2. Install Dependencies
```
pip install -r requirements.txt
```

### 3. Run the streamlit app

```
streamlit run app.py
```












