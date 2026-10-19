from pathlib import Path

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"

df = pd.read_csv(DATA_PATH)

# Take one transaction without its target label.
transaction = df.drop(columns=["Class"]).iloc[0].tolist()

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"features": transaction},
    timeout=10,
)

print("Status code:", response.status_code)
print("Response:", response.json())

response.raise_for_status()