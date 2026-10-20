from pathlib import Path
import math
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field ,field_validator


# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "fraud_model.joblib"
THRESHOLD_PATH = MODELS_DIR / "threshold.joblib"

# Check that the trained artifacts exist
if not MODEL_PATH.exists() or not THRESHOLD_PATH.exists():
    raise FileNotFoundError(
        "Model files not found. Run inspect_data.py first."
    )

# Load the trained model and selected threshold
model = joblib.load(MODEL_PATH)
threshold = float(joblib.load(THRESHOLD_PATH))

# Recover the exact feature names used during training
feature_names = list(model.feature_names_in_)

app = FastAPI(
    title="FraudGuard API",
    description="Credit card fraud detection API",
    version="1.0.0",
)


class Transaction(BaseModel):
    features: list[float] = Field(
        ...,
        description=(
            "30 transaction features in the same order "
            "as the training dataset."
        ),
    )

    @field_validator("features")
    @classmethod
    def features_must_be_finite(
        cls, values: list[float]
    ) -> list[float]:
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                "All feature values must be finite numbers."
            )

        return values

@app.get("/")
def home():
    return {
        "message": "FraudGuard API is running",
        "model": "Random Forest",
    }


@app.post("/predict")
def predict(transaction: Transaction):
    if len(transaction.features) != len(feature_names):
        raise HTTPException(
            status_code=422,
            detail=f"Expected {len(feature_names)} features, "
                   f"received {len(transaction.features)}.",
        )

    # Preserve feature names and their training order
    input_data = pd.DataFrame(
        [transaction.features],
        columns=feature_names,
    )

    # Estimate fraud probability
    probability = float(
        model.predict_proba(input_data)[0, 1]
    )

    # Convert probability into a class using the saved threshold
    prediction = int(probability >= threshold)

    return {
        "fraud_probability": round(probability, 6),
        "threshold": threshold,
        "prediction": prediction,
        "label": "fraud" if prediction == 1 else "legitimate",
    }