
from __future__ import annotations
import numpy as np
from datetime import datetime
import gradio as gr
from PIL import Image
from .features import FEATURES
from .recommendations import text_recommendations, dynamic_flags

def _risk_bucket(p: float) -> str:
    return "Low" if p < 30 else ("Moderate" if p < 70 else "High")

def build_interface(model, scaler, shap_explainer, lime_explainer):
    with gr.Blocks() as demo:
        gr.Markdown("# 🩺 Healthcare Dashboard")

        gr.Markdown("### 👤 Enter Your Health Info")

        age = gr.Number(value=30, label="Age (years)")
        height = gr.Number(value=170, label="Height (cm)")
        weight = gr.Number(value=70, label="Weight (kg)")
        systolic = gr.Number(value=120, label="Systolic BP (mmHg)")
        diastolic = gr.Number(value=80, label="Diastolic BP (mmHg)")
        gender = gr.Radio(choices=["Male", "Female"], value="Male", label="Gender")
        cholesterol = gr.Number(value=190, label="Cholesterol")
        glucose = gr.Number(value=95, label="Glucose")
        smoking = gr.Radio(choices=["Yes", "No"], value="No", label="Do you smoke?")
        alcohol = gr.Radio(choices=["Yes", "No"], value="No", label="Consume alcohol?")
        activity = gr.Radio(choices=["Yes", "No"], value="Yes", label="Physically active?")

        with gr.Tab("🧠 Immediate Feedback"):
            feedback_output = gr.Textbox(label="Live Health Assessment", lines=6, interactive=False)
            for comp in [age, height, weight, systolic, cholesterol, glucose]:
                comp.change(
                    fn=dynamic_flags,
                    inputs=[age, height, weight, systolic, cholesterol, glucose],
                    outputs=feedback_output
                )

        with gr.Tab("❤️ Heart Disease Prediction"):
            output_bmi = gr.Textbox(label="BMI")
            output_prob = gr.Textbox(label="Heart Disease Probability")
            output_shap_image = gr.Image(label="SHAP Feature Importance (Static)")
            output_lime_image = gr.Image(label="LIME Explanation")
            output_dynamic_recs = gr.Textbox(label="Dynamic Health Recommendations")
            output_personal_recs = gr.Textbox(label="Personalized Health Recommendations")
            output_confidence = gr.Textbox(label="Confidence Interval (±5% heuristic)")

            def _predict(age, gender, height, weight, systolic, diastolic, cholesterol, glucose, smoking, alcohol, activity):
                # Binary encodings
                g = 1 if gender == "Male" else 0
                s = 1 if smoking == "Yes" else 0
                a = 1 if alcohol == "Yes" else 0
                act = 1 if activity == "Yes" else 0

                bmi = weight / (height / 100.0) ** 2
                pulse_pressure = systolic - diastolic
                bmi_age_interaction = bmi * age

                raw = np.array([
                    age, systolic, diastolic, cholesterol, glucose, g,
                    s, a, act, bmi, pulse_pressure, bmi_age_interaction
                ], dtype=float)

                # Build SHAP/LIME + probability
                from .explain import explain_instance
                shap_img, lime_img, prob = explain_instance(
                    shap_explainer, lime_explainer, model, scaler, FEATURES, raw
                )

                ci_low = max(0.0, prob - 5.0)
                ci_high = min(100.0, prob + 5.0)
                risk_level = _risk_bucket(prob)

                dyn = []
                if systolic > 130 or diastolic > 85: dyn.append("⚠️ Blood pressure is elevated.")
                if cholesterol > 240: dyn.append("⚠️ Cholesterol is high.")
                if glucose > 125: dyn.append("⚠️ Glucose suggests possible diabetes.")
                if bmi > 30: dyn.append("⚠️ High BMI may increase cardiovascular risk.")
                if s: dyn.append("⚠️ Smoking detected — consider quitting.")
                if a: dyn.append("⚠️ Alcohol consumption could affect heart.")
                if not act: dyn.append("⚠️ Lack of activity — increase physical movement.")
                dynamic_output = "\n".join(dyn) if dyn else "✅ All dynamic metrics within healthy range."

                personal_recs = text_recommendations(raw.tolist())

                return (
                    f"{bmi:.2f}",
                    f"{prob:.2f}% ({risk_level} Risk)",
                    shap_img,
                    lime_img,
                    dynamic_output,
                    personal_recs,
                    f"{ci_low:.2f}% - {ci_high:.2f}%"
                )

            predict_btn = gr.Button("🔍 Predict Heart Disease Risk")
            predict_btn.click(
                fn=_predict,
                inputs=[age, gender, height, weight, systolic, diastolic, cholesterol, glucose, smoking, alcohol, activity],
                outputs=[
                    output_bmi, output_prob, output_shap_image, output_lime_image,
                    output_dynamic_recs, output_personal_recs, output_confidence
                ]
            )

    return demo
