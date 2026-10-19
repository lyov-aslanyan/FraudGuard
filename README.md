# FraudGuard — Credit Card Fraud Detection API

FraudGuard is a machine learning project that detects potentially fraudulent credit card transactions using a Random Forest classifier. It exposes predictions through a REST API built with FastAPI and can be run in Docker.

## Project Highlights

- Trained and evaluated Logistic Regression and Random Forest classifiers.
- Addressed severe class imbalance in the credit card fraud dataset.
- Evaluated models using Average Precision, ROC-AUC, Precision, Recall, and F1-score.
- Selected a classification threshold of `0.37`.
- Built a REST API with FastAPI.
- Added automated API tests with pytest.
- Containerized the application using Docker and Docker Compose.

## Final Model Performance

The following results are from the held-out test set.

| Metric | Result |
|---|---:|
| Average Precision (AP) | 0.8444 |
| ROC-AUC | 0.9604 |
| Precision | 93.83% |
| Recall | 80.00% |
| F1-score | 86.36% |
| Classification threshold | 0.37 |

### Confusion Matrix

| | Predicted legitimate | Predicted fraud |
|---|---:|---:|
| Actual legitimate | 56,646 | 5 |
| Actual fraud | 19 | 76 |

At the selected threshold, the model identified 76 of 95 fraudulent transactions, with 5 false positives among 56,651 legitimate transactions.

These results apply to the held-out test set and may differ on real-world transactions.

## Tech Stack

- **Language:** Python 3.12
- **Machine Learning:** scikit-learn
- **Data Processing:** pandas, NumPy
- **API:** FastAPI, Uvicorn
- **Testing:** pytest
- **Model Serialization:** joblib
- **Deployment:** Docker, Docker Compose

## Project Structure

```text
FraudGuard/
├── app.py
├── check_api.py
├── inspect_data.py
├── requirements.txt
├── Dockerfile
├── compose.yaml
├── .dockerignore
├── .gitignore
├── models/
│   ├── fraud_model.joblib
│   ├── threshold.joblib
│   └── metrics.json
├── reports/
│   ├── feature_importance.csv
│   └── model_comparison.csv
├── tests/
│   └── test_api.py
└── data/
    └── raw/
        └── creditcard.csv
```

The dataset and virtual environment are excluded from version control.

## Dataset

This project uses the Credit Card Fraud Detection dataset from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

nano README.mdThe dataset contains anonymized transaction features and a binary target indicating whether a transaction is fraudulent.

Fraudulent transactions represent a very small proportion of the original dataset, making this an imbalanced classification problem.

The dataset is not included in this repository.

## Run Locally

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd FraudGuard
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the API

```bash
uvicorn app:app --reload
```

The API will be available at:

- API root: http://127.0.0.1:8000/
- Interactive API documentation: http://127.0.0.1:8000/docs

## Run with Docker

Build and start the service:

```bash
sudo docker compose up --build -d
```

Check its status:

```bash
sudo docker compose ps
```

The service is exposed on port `8001` on the host.

- API root: http://127.0.0.1:8001/
- Interactive API documentation: http://127.0.0.1:8001/docs

Stop the service:

```bash
sudo docker compose down
```

## API Usage

The prediction endpoint is:

`POST /predict`

It accepts a JSON object containing exactly 30 numeric features in the same order used during model training.

Example request format:

```json
{
  "features": [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
  ]
}
```

The zeros illustrate the required format only; use actual transaction feature values for meaningful predictions.

The response includes:
- `fraud_probability`: predicted fraud probability
- `threshold`: the selected classification threshold
- `prediction`: `1` for fraud or `0` for legitimate
- `label`: human-readable prediction

Explore and test the endpoint at `/docs`.


POST /predict
```

It accepts a JSON object containing the 30 numeric transaction features under the `features` key. Supply the feature names and values expected by the trained model.

Example request structure:

```json
{
  "features": {
    "Time": 0.0,
    "V1": 0.1,
    "V2": -0.2
  }
}
```

**Important:** This is only a partial example. The actual request must include all 30 expected features. Open `/docs` to inspect and test the endpoint.

The response includes the fraud probability, selected threshold, prediction, and human-readable label.

## Run Tests

```bash
pytest -v
```

The API test suite currently passes all four tests, covering the root endpoint, valid predictions, incorrect feature count, and missing features.

## Future Improvements

- Add input validation and clearer error messages.
- Compare additional models and tune the decision threshold using validation data.
- Monitor model performance and data drift.
- Add CI with GitHub Actions.
- Deploy the API to a cloud platform.
- Improve API logging and operational monitoring.

## Disclaimer

FraudGuard is an educational machine learning project, not a production-ready financial fraud prevention system. Its predictions should not be used as the sole basis for real financial decisions.
