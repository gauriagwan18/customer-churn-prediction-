# Customer Churn Prediction

An end-to-end machine learning pipeline that predicts telecom customer churn, deployed as an interactive web app.

## Problem

Customer churn directly impacts recurring revenue. This project identifies customers at high risk of leaving so a business can act early — offering retention incentives before it's too late.

## Dataset

- **Source:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- 7,043 customers, 21 features (demographics, account info, services subscribed)
- Target: `Churn` (Yes/No) — 26.5% churn rate (imbalanced)

## Approach

1. **Data Cleaning** — handled missing values in `TotalCharges`, removed non-predictive ID column
2. **EDA** — analyzed churn patterns by contract type and tenure
3. **Modeling** — trained and compared three classifiers:
   - Logistic Regression
   - Random Forest
   - XGBoost
4. **Class Imbalance Handling** — applied `class_weight='balanced'` to improve detection of the minority (churn) class
5. **Deployment** — built a Streamlit web app for real-time churn prediction

## Key Findings

- **Month-to-month contracts** churn far more than one/two-year contracts
- **Low-tenure customers** (0–10 months) are the highest churn risk
- Top churn drivers (by model coefficient): **Paperless Billing, Internet Service type, Senior Citizen status**

## Results

| Model | Accuracy | F1-Score |
|---|---|---|
| Logistic Regression (baseline) | 81.6% | 0.62 |
| Random Forest | 79.6% | 0.55 |
| XGBoost | 79.2% | 0.57 |
| **Logistic Regression (balanced)** | **75.7%** | **0.645** |

The balanced model was selected as final, since it improves recall on actual churners from 58% to 84% — catching far more at-risk customers, which matters more than raw accuracy for a retention use case.

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Streamlit

## Project Structure

```
├── churn_prediction_analysis.ipynb   # Full analysis: EDA, modeling, evaluation
├── churn_model.pkl                   # Saved trained model
├── app.py                            # Streamlit web app
├── requirements.txt                  # Dependencies
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Live Demo

[Add Streamlit Cloud link here once deployed]

## Author

Gauri Agwan — [LinkedIn](https://linkedin.com/in/gauriagwan)
