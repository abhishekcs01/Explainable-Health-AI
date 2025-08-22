
from __future__ import annotations
import pandas as pd

RENAMES = {"gluc": "glucose", "cardio": "heart_disease", "alco": "alcohol"}

REQUIRED_BASE = [
    "age","height","weight","systolic","diastolic","cholesterol","glucose",
    "gender","smoke","alcohol","active","heart_disease"
]

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Standardize column names used downstream
    df = df.rename(columns=RENAMES)
    missing = [c for c in REQUIRED_BASE if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df
