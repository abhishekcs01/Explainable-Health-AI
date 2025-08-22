
from __future__ import annotations
from typing import List

def text_recommendations(values: list) -> str:
    # values are in RAW units aligned to feature order in ui.py
    advice: List[str] = []
    age, systolic, diastolic, cholesterol, glucose, gender, smoking, alcohol, active, bmi, pulse_pressure, bmi_age_interaction = values

    if systolic > 140:
        advice.append("High systolic BP detected. Reduce sodium and consider regular aerobic exercise.")
    if cholesterol > 200:
        advice.append("High cholesterol. Prefer fiber-rich diet and limit saturated fats.")
    if glucose > 126:
        advice.append("Elevated glucose. Reduce simple sugars and consult a clinician.")
    if bmi > 30:
        advice.append("BMI suggests overweight/obesity. Increase activity and adopt a balanced diet.")
    if smoking == 1:
        advice.append("Smoking increases risk substantially. Seek cessation support.")
    if alcohol == 1:
        advice.append("Frequent alcohol intake may affect heart health. Consider reducing intake.")
    if active == 0:
        advice.append("Low physical activity. Aim for 150+ minutes/week of moderate activity.")

    return "\n".join(advice) if advice else "Your metrics look good — keep up a heart-healthy lifestyle!"

def dynamic_flags(age, height, weight, systolic, cholesterol, glucose):
    bmi = weight / (height / 100.0) ** 2
    msgs = []
    if systolic > 130: msgs.append("🔴 Elevated systolic BP.")
    if cholesterol > 200: msgs.append("🟠 High cholesterol.")
    if glucose > 125: msgs.append("🟠 Elevated glucose.")
    if bmi > 30: msgs.append("🟡 High BMI.")
    return "\n".join(msgs) if msgs else "✅ All metrics within healthy range."
