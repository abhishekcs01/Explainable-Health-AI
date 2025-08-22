# 🩺 Explainable AI for Heart Disease Risk (Gradio + SHAP + LIME)

An end‑to‑end, **explainable** ML app for estimating heart disease risk with **XGBoost**, featuring **SHAP** and **LIME** explanations and a clean **Gradio** dashboard — split into a proper, production‑style Python package layout.

> ⚠️ **Medical disclaimer:** This tool is for **education and demonstration only**. It is **not** a medical device and must not be used for diagnosis or treatment decisions.

---

## ✨ Highlights

- Modular Python package (`xai_health/`) with clear separation of **data**, **features**, **modeling**, **explainability**, **recommendations**, and **UI**.
- **Gradio** app with number inputs (no sliders) and **live feedback** while you type.
- **SHAP** global bar chart + **LIME** local explanation per prediction.
- Prediction **logging** to `prediction_log.csv` with timestamp and inputs.
- Easy **CLI**: `train`, `evaluate`, `serve`.

---

## 📁 Repo Structure

```
xai-healthcare/
├─ app.py                         # CLI entrypoint: train/evaluate/serve
├─ requirements.txt
├─ README.md
├─ .gitignore
├─ data/
│  └─ heartdisease_data.csv       # (Put your dataset here)
├─ xai_health/
│  ├─ __init__.py
│  ├─ config.py                   # Paths, hyperparams, flags
│  ├─ data.py                     # Load & validate data
│  ├─ features.py                 # Feature engineering + scaling
│  ├─ model.py                    # Train & evaluate XGBoost
│  ├─ explain.py                  # SHAP/LIME builders and plots
│  ├─ recommendations.py          # Health guidance helpers
│  ├─ ui.py                       # Gradio Blocks interface
│  └─ utils.py                    # Logging and helpers
└─ tests/
   └─ test_smoke.py               # Basic import & run smoke tests
```

---

## 🧰 Setup

```bash
# 1) Clone your repo, then create a virtual env
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Add data
# Put your CSV at: data/heartdisease_data.csv
```

### 📄 Expected CSV Columns

Your dataset should include (names are case‑sensitive):

- `age` (years)
- `height` (cm)
- `weight` (kg)
- `systolic` (mmHg)
- `diastolic` (mmHg)
- `cholesterol` (mg/dL or categorical score)
- `gluc` **or** `glucose` (mg/dL or categorical score)
- `gender` (0/1 or string ('Male', 'Female'))
- `smoke` (0/1)
- `alco` **or** `alcohol` (0/1)
- `active` (0/1)
- `cardio` **or** `heart_disease` (target: 0/1)

The loader will **auto‑rename** `gluc→glucose`, `alco→alcohol`, `cardio→heart_disease` if present.

---

## 🚀 Usage

### 1) Train
```bash
python app.py train
```

### 2) Evaluate (prints metrics, saves confusion matrix)
```bash
python app.py evaluate
```

### 3) Serve the Gradio app
```bash
python app.py serve --host 0.0.0.0 --port 7860
```
Then open the printed URL.

---

## 🧠 Features & Explainability

- Engineered features: **BMI**, **pulse_pressure**, **bmi_age_interaction**.
- **MinMaxScaler** normalization before model training.
- **Class imbalance** handled using `scale_pos_weight` (neg/pos ratio). Optionally enable **SMOTE** in `config.py`.

> The UI returns: BMI, predicted risk % with a coarse **risk level**, **SHAP** bar chart, **LIME** explanation, dynamic flags (e.g., high BP), personalized tips, and a simple ±5% pseudo‑interval for quick intuition — not a statistical CI.

---

## ⚙️ Configuration

Edit `xai_health/config.py`:

- Data & seed: `DATA_PATH`, `RANDOM_STATE`, `TEST_SIZE`
- Modeling: `MODEL_PARAMS` (XGBoost hyperparams)
- Imbalance: `USE_SMOTE` (default False)
- Engineering: `APPLY_WEIGHTING` (off by default; the original notebook used manual multipliers — kept here behind a flag)

---

## 🧪 Tests

Basic smoke tests ensure modules import and minimal flows run.

```bash
pytest -q
```

---

## 📜 License

MIT — see `LICENSE` if you add one.

---

## 🙌 Credits

Built with ❤️ using **Gradio**, **SHAP**, **LIME**, **XGBoost**, and **scikit‑learn**.