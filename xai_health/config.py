
import os

# Paths
DATA_PATH = os.environ.get("XAI_DATA_PATH", "data/heartdisease_data.csv")

# Reproducibility
RANDOM_STATE = int(os.environ.get("XAI_RANDOM_STATE", 42))
TEST_SIZE = float(os.environ.get("XAI_TEST_SIZE", 0.2))

# Class imbalance
USE_SMOTE = os.environ.get("XAI_USE_SMOTE", "0") == "1"

# Engineering tweaks (kept OFF by default)
# The original notebook multiplied certain features to "manually adjust correlations".
# That can distort distributions; it's kept here but OFF unless you flip this flag.
APPLY_WEIGHTING = os.environ.get("XAI_APPLY_WEIGHTING", "0") == "1"

# XGBoost defaults (good starting point; tune as needed)
MODEL_PARAMS = {
    "n_estimators": int(os.environ.get("XAI_N_ESTIMATORS", 600)),
    "max_depth": int(os.environ.get("XAI_MAX_DEPTH", 7)),
    "learning_rate": float(os.environ.get("XAI_LEARNING_RATE", 0.025)),
    "min_child_weight": int(os.environ.get("XAI_MIN_CHILD_WEIGHT", 4)),
    "subsample": float(os.environ.get("XAI_SUBSAMPLE", 0.85)),
    "colsample_bytree": float(os.environ.get("XAI_COLSAMPLE_BYTREE", 0.85)),
    "eval_metric": os.environ.get("XAI_EVAL_METRIC", "logloss"),
}
