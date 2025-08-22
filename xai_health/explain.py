
from __future__ import annotations
from typing import Tuple, List
import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import shap
import lime.lime_tabular as llt

def build_explainers(model, X_train, feature_names: List[str]):
    shap_explainer = shap.TreeExplainer(model)
    lime_explainer = llt.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=["No Disease", "Disease"],
        discretize_continuous=True
    )
    return shap_explainer, lime_explainer

def explain_instance(shap_explainer, lime_explainer, model, scaler, feature_names, raw_instance: np.ndarray):
    # raw_instance: shape (n_features,) in raw units; scale before model
    data_scaled = scaler.transform(raw_instance.reshape(1, -1))

    # SHAP (global-ish bar for this instance using abs mean)
    shap_vals = shap_explainer.shap_values(data_scaled)
    if isinstance(shap_vals, list):  # older SHAP returns list per class
        shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
    shap_vals = np.asarray(shap_vals)
    shap_mean = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(shap_mean)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=shap_mean[order], y=np.array(feature_names)[order])
    plt.title("SHAP Feature Importance")
    buf_shap = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf_shap, format="png", dpi=150)
    plt.close()
    buf_shap.seek(0)
    shap_img = Image.open(buf_shap)

    # LIME local explanation
    lime_exp = lime_explainer.explain_instance(data_scaled[0], model.predict_proba, num_features=min(8, len(feature_names)))
    lime_fig = lime_exp.as_pyplot_figure()
    buf_lime = io.BytesIO()
    lime_fig.savefig(buf_lime, format="png", dpi=150, bbox_inches="tight")
    plt.close(lime_fig)
    buf_lime.seek(0)
    lime_img = Image.open(buf_lime)

    # Pred prob for display
    prob = float(model.predict_proba(data_scaled)[0, 1]) * 100.0
    prob = max(0.0, min(100.0, prob))
    return shap_img, lime_img, prob
