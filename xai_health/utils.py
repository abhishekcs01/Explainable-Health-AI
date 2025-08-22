
from __future__ import annotations
import os
import pandas as pd
from datetime import datetime

LOG_PATH = "prediction_log.csv"

def log_prediction(row: dict):
    df = pd.DataFrame([row])
    if os.path.exists(LOG_PATH):
        df.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_PATH, index=False)

def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
