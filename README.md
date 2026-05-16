# 📡 Telco Customer Churn Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A professional, end-to-end machine learning system that predicts whether a telecom customer will churn. Built to production-engineering standards with a modular codebase, reproducible pipeline, and an interactive Streamlit web application.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Project Structure](#project-structure)
4. [ML Pipeline Explanation](#ml-pipeline-explanation)
5. [Model Results](#model-results)
6. [Key Insights from EDA](#key-insights-from-eda)
7. [Installation](#installation)
8. [How to Run](#how-to-run)
9. [Launch the Streamlit App](#launch-the-streamlit-app)
10. [Feature Importance](#feature-importance)

---

## Project Overview

Customer churn is one of the most expensive problems in the telecoms industry — acquiring a new customer costs 5–25× more than retaining an existing one. This project builds a **binary classification** system to identify at-risk customers before they leave, enabling proactive retention campaigns.

**Problem type:** Binary Classification  
**Target variable:** `Churn` (Yes → 1, No → 0)  
**Best model:** Random Forest (ROC-AUC = 0.8438)

---

## Dataset Description

| Attribute        | Detail                                              |
|------------------|-----------------------------------------------------|
| **Source**       | [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Rows**         | 7,043 customers                                     |
| **Columns**      | 21 features                                         |
| **Target**       | `Churn` — whether the customer left in the last month |
| **Class balance**| 73.5% Stay / 26.5% Churn                            |

### Feature Groups

| Group          | Features                                                                         |
|----------------|----------------------------------------------------------------------------------|
| Demographics   | `gender`, `SeniorCitizen`, `Partner`, `Dependents`                               |
| Account info   | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`                        |
| Billing        | `MonthlyCharges`, `TotalCharges`                                                 |
| Phone services | `PhoneService`, `MultipleLines`                                                  |
| Internet svcs  | `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |

---

## Project Structure

```
customer-churn-project/
│
├── data/
│   └── telco_churn.csv          # Raw dataset
│
├── notebooks/
│   ├── EDA.ipynb                # Exploratory Data Analysis notebook
│   └── figures/                 # Auto-generated EDA plots
│       ├── churn_distribution.png
│       ├── numeric_distributions.png
│       ├── churn_by_contract.png
│       ├── churn_by_tenure.png
│       ├── churn_by_charges.png
│       ├── churn_by_payment.png
│       ├── correlation_heatmap.png
│       ├── feature_importance.png
│       ├── roc_curves.png
│       └── cm_*.png             # Per-model confusion matrices
│
├── src/
│   ├── preprocess.py            # Full preprocessing pipeline
│   ├── train_model.py           # Model training, evaluation & selection
│   └── predict.py               # Prediction script / inference API
│
├── models/
│   ├── churn_model.pkl          # Saved best model (Random Forest)
│   ├── scaler.pkl               # Fitted StandardScaler
│   └── feature_names.pkl        # Ordered training feature list
│
├── app/
│   └── streamlit_app.py         # Interactive prediction web app
│
├── requirements.txt
└── README.md
```

---

## ML Pipeline Explanation

```
Raw CSV
  │
  ▼
preprocess.py
  ├─ Load & validate data
  ├─ Fix TotalCharges (object → float, impute 11 NaN with median)
  ├─ Encode target: Yes→1 / No→0
  ├─ One-hot encode 15 categorical features (drop_first=True)
  ├─ StandardScaler on numeric columns (tenure, MonthlyCharges, TotalCharges)
  └─ Stratified 80/20 train-test split
       │
       ▼
train_model.py
  ├─ Logistic Regression   (class_weight=balanced)
  ├─ Random Forest         (200 trees, max_depth=8, balanced)
  ├─ Gradient Boosting     (200 estimators, lr=0.05)
  │
  ├─ Evaluate each: Accuracy, Precision, Recall, F1, ROC-AUC
  ├─ Plot: ROC curves, Confusion matrices, Feature importances
  └─ Save best model → models/churn_model.pkl
       │
       ▼
predict.py / streamlit_app.py
  ├─ Load model + scaler + feature_names
  ├─ Preprocess raw customer input (same pipeline)
  └─ Return: prediction (0/1), probability, human-readable message
```

---

## Model Results

All models evaluated on the held-out **20% test set** (1,409 customers).

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.7388   | 0.5052    | 0.7834 | 0.6143   | 0.8417  |
| **Random Forest**   | **0.7630** | **0.5366** | **0.7834** | **0.6370** | **0.8438** |
| Gradient Boosting   | 0.7999   | 0.6533    | 0.5241 | 0.5816   | 0.8431  |

> **Winner: Random Forest** — selected on ROC-AUC (0.8438).  
> It achieves the best balance of Recall (catching churners) and F1, critical  
> in a business context where missed churners are costly.

### Why Recall matters here

A false negative (predicting "Stay" when the customer actually churns) costs the business a lost customer. We therefore favour models with **high Recall** over pure Accuracy, accepting some false positives (sending a retention offer to a customer who would have stayed anyway — low cost).

---

## Key Insights from EDA

| Finding | Business Implication |
|---------|---------------------|
| Month-to-month customers churn at **43%** vs 3% for 2-year contracts | Offer discounts to upgrade to annual plans |
| **New customers (0–12 months)** churn at ~50% | Implement 90-day onboarding loyalty programmes |
| **Electronic check** payers churn at ~45% | Promote auto-pay enrolment with a small discount |
| **Fiber optic** users churn more than DSL | Audit service quality; introduce SLA guarantees |
| Customers paying **$86+/month** have 35%+ churn | Bundle add-ons to increase perceived value |
| `tenure` is the single strongest predictor (−0.35 corr.) | Long-term retention compounds — invest early |

---

## Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/customer-churn-project.git
cd customer-churn-project

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Step 1 — Preprocess the data

```bash
python src/preprocess.py
```

Validates the dataset, fixes data quality issues, and saves the fitted scaler.

### Step 2 — Train models

```bash
python src/train_model.py
```

Trains Logistic Regression, Random Forest, and Gradient Boosting classifiers, prints the comparison table, saves the best model to `models/churn_model.pkl`, and writes all evaluation plots to `notebooks/figures/`.

### Step 3 — Run a prediction

```bash
python src/predict.py
```

Runs an example customer through the saved model and prints the churn prediction and probability.

You can also import `predict_churn` directly:

```python
from src.predict import predict_churn

result = predict_churn({
    'SeniorCitizen'   : 0,
    'tenure'          : 5,
    'MonthlyCharges'  : 80.0,
    'TotalCharges'    : 400.0,
    'gender'          : 'Female',
    'Partner'         : 'No',
    'Dependents'      : 'No',
    'PhoneService'    : 'Yes',
    'MultipleLines'   : 'No',
    'InternetService' : 'Fiber optic',
    'OnlineSecurity'  : 'No',
    'OnlineBackup'    : 'No',
    'DeviceProtection': 'No',
    'TechSupport'     : 'No',
    'StreamingTV'     : 'Yes',
    'StreamingMovies' : 'Yes',
    'Contract'        : 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod'   : 'Electronic check',
})

print(result['message'])           # ⚠️  Customer likely to CHURN
print(result['probability'])       # 0.54
```

### Step 4 — Open the EDA notebook

```bash
jupyter notebook notebooks/EDA.ipynb
```

---

## Launch the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The app opens at **http://localhost:8501**.

**Usage:**
1. Fill in the customer's account details using the sliders and dropdowns.
2. Click **🔍 Predict Churn**.
3. The app displays the prediction, churn probability, and risk level.

> **Note:** The trained model (`models/churn_model.pkl`) must exist before launching the app. Run `python src/train_model.py` first if needed.

---

## Feature Importance

Top features identified by the Random Forest model:

| Rank | Feature                    | Importance |
|------|----------------------------|------------|
| 1    | `tenure`                   | Highest    |
| 2    | `TotalCharges`             | High       |
| 3    | `Contract_Two year`        | High       |
| 4    | `MonthlyCharges`           | High       |
| 5    | `InternetService_Fiber optic` | Medium  |

See `notebooks/figures/feature_importance.png` for the full chart.

---

## License

This project is licensed under the MIT License. Free to use for academic, research, and personal projects.

---

*Built with ❤️ using Python · scikit-learn · pandas · Streamlit*
