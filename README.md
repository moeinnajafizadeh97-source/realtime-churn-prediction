# Customer Churn Prediction with Continual Learning and Explainability

**[Try the live dashboard](https://realtime-churn-prediction.streamlit.app/)**

Most churn prediction models are trained once and left alone, even though customer behaviour keeps changing. This project asks whether a model that keeps learning holds up better when it does.

Three models were compared on the IBM Telco Customer Churn dataset (7,043 customers): Logistic Regression, LightGBM with Focal Loss, and an Adaptive Random Forest from the River library that updates after every customer. Halfway through a simulated stream of customer data, the pattern of who churns was deliberately changed while the overall churn rate stayed the same.

## Findings

- Before the change, all three models performed the same (F2 between 0.742 and 0.750).
- After it, the two static models lost around 0.12 of F2 and never recovered.
- The continual learning model lost about half as much, recovered to 0.750 by the final batch, and beat both static models in every changed batch.
- Logistic Regression matched the tuned LightGBM throughout, which suggests that for this problem, whether a model keeps learning matters more than how sophisticated it is.

Predictions are explained per customer using SHAP. Contract type dominates every other feature.

## Dashboard

The dashboard simulates new customer data arriving one month at a time. The model predicts each month before seeing the outcomes, then learns from them before the next month arrives, which is how it would work in practice.

## Repository

| Path | Contents |
|---|---|
| `app.py` | Streamlit dashboard |
| `notebook/churn_pipeline.ipynb` | Exploratory data analysis, preprocessing, drift simulation, model training, evaluation, statistical tests and SHAP |
| `files/` | Data and trained model used by the dashboard |

## Running it yourself

Download or clone this repository, then from inside the folder run:
```
pip install -r requirements.txt
streamlit run app.py
```
## Built with

Python, River, LightGBM, scikit-learn, SHAP, Streamlit

## Data

IBM Telco Customer Churn, available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). The data describes a fictional telecoms company and contains no real customer information.

---

Moein Najafizadeh, MSc Data Science, Manchester Metropolitan University, 2026
