
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb

from .config import RANDOM_STATE, TEST_SIZE, MODEL_PARAMS, USE_SMOTE
from .features import FEATURES

def _scale_pos_weight(y: np.ndarray) -> float:
    # ratio = (neg / pos)
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    return float(neg) / float(max(pos, 1))

def make_splits(X, y):
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

def maybe_smote(X_train, y_train):
    if USE_SMOTE:
        sm = SMOTE(random_state=RANDOM_STATE)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        return X_res, y_res
    return X_train, y_train

def train_model(X_train, y_train) -> xgb.XGBClassifier:
    params = dict(MODEL_PARAMS)
    params["scale_pos_weight"] = _scale_pos_weight(y_train)
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test) -> Dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, target_names=["No Disease", "Disease"])

    cm = confusion_matrix(y_test, y_pred)
    fig = plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    return {"accuracy": acc, "auc": auc, "report": report, "confusion_fig": fig}
