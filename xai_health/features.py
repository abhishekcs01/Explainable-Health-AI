
from __future__ import annotations
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from .config import APPLY_WEIGHTING

FEATURES = [
    "age", "systolic", "diastolic", "cholesterol", "glucose", "gender",
    "smoke", "alcohol", "active", "bmi", "pulse_pressure", "bmi_age_interaction"
]
TARGET = "heart_disease"

def engineer(df: pd.DataFrame, apply_weighting: bool = APPLY_WEIGHTING) -> pd.DataFrame:
    df = df.copy()
    # Derived features
    df["bmi"] = df["weight"] / (df["height"] / 100.0) ** 2
    df["pulse_pressure"] = df["systolic"] - df["diastolic"]
    df["bmi_age_interaction"] = df["bmi"] * df["age"]

    # Optional manual multipliers (disabled by default)
    if apply_weighting:
        df.loc[:, "age"] *= 1.1
        df.loc[:, "systolic"] *= 1.2
        df.loc[:, "diastolic"] *= 1.1
        df.loc[:, "cholesterol"] *= 1.5
        df.loc[:, "glucose"] *= 1.4
        df.loc[:, "smoke"] *= 1.3
        df.loc[:, "alcohol"] *= 1.2
        df.loc[:, "active"] *= -0.8
        df.loc[:, "bmi"] *= 1.4
        df.loc[:, "pulse_pressure"] *= 1.2
        df.loc[:, "bmi_age_interaction"] *= 1.3

    return df

def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    return X, y

def build_scaler() -> MinMaxScaler:
    return MinMaxScaler()
