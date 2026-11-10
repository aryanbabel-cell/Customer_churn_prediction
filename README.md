# Customer Churn Prediction

A Machine Learning project that predicts whether a telecom customer is likely to leave the service using Logistic Regression.

## Project Overview

Customer churn is a major challenge for telecom companies. This project uses customer information and service usage patterns to identify customers who are likely to discontinue their subscriptions.

The project includes a machine learning model and an interactive Streamlit dashboard for exploring churn data and predicting customer churn.

## Features

- Customer churn prediction using Logistic Regression
- Exploratory Data Analysis (EDA)
- Customer churn and retention statistics
- Churn analysis by contract type
- Interactive customer input form
- Churn probability and risk classification
- Personalized customer retention suggestions

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib (if used in analysis)

## Machine Learning Workflow

1. Data collection
2. Data cleaning and preprocessing
3. Exploratory Data Analysis
4. Feature encoding and scaling
5. Model training
6. Model evaluation
7. Streamlit application development

## Model Performance

**Final Model: Logistic Regression**

| Metric | Score |
|---|---:|
| Accuracy | 80.55% |
| Precision | 65.72% |
| Recall | 55.88% |
| F1 Score | 60.40% |

## Project Structure

```text
Customer_churn_prediction/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   └── churn_model.pkl
├── app.py
└── train_model.py
```

## How to Run the Project

1. Clone this repository:

```bash
git clone https://github.com/aryanbabel-cell/Customer_churn_prediction.git
```

2. Navigate to the project folder:

```bash
cd Customer_churn_prediction
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit application:

```bash
streamlit run app.py
```

## Dataset

IBM Telco Customer Churn dataset, containing customer demographics, account information, services and churn status.

## Future Improvements

- Improve recall using model tuning and class imbalance techniques.
- Compare additional machine learning algorithms.
- Deploy the application online.
- Add more advanced customer retention insights.

## Author

Aryan Babel
