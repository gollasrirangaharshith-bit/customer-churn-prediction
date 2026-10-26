# Customer Churn Prediction

Predicts whether a telecom customer is likely to leave, using machine learning, with a Streamlit web app.

## Dataset
Telco Customer Churn (7,043 customers, 21 columns). After cleaning: 7,032 rows. About 73% stayed and 27% left.

## What I did
1. Cleaned the data and explored churn patterns (contract type, tenure)
2. Built a preprocessing pipeline (one-hot encoding and scaling)
3. Compared four models: Logistic Regression, Decision Tree, Random Forest, XGBoost
4. Saved the best model and built a Streamlit app

## Model comparison

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 80.38 | 64.85 | 57.22 | 60.80 |
| Decision Tree | 79.03 | 61.65 | 55.88 | 58.63 |
| XGBoost | 78.89 | 62.38 | 51.87 | 56.64 |
| Random Forest | 78.39 | 62.32 | 47.33 | 53.80 |

Logistic Regression scored highest on all four metrics, so it is used in the app.

## Run it locally
pip install -r requirements.txt
streamlit run app.py

## Files
- app.py: Streamlit web app
- train_model.py: trains and saves the model
- churn_model.joblib: the trained model
