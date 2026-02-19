# Cordilyze — Interactive Cardiovascular Risk Assessment

**See how lifestyle changes affect your heart risk before you make them.**

Cordilyze is a cardiovascular risk tool that doesn't just give you a score — it lets you experiment with it. Adjust lifestyle factors like smoking, exercise, and weight, and watch your risk update in real time. Every prediction is explained using SHAP so you can see exactly what's driving your number.

Built for DeveloperWeek 2026 Hackathon.

**[Live Demo](https://cordilyze.onrender.com)** · **[Demo Video](YOUR_YOUTUBE_LINK)**

---

## Why I Built This

I've spent over 10 years working in clinical laboratories. I've seen how patients react when they get lab results — most of them don't understand what the numbers mean, and the ones who do are left wondering what to actually do about it.

Traditional CVD risk calculators have the same problem. They give you a score and stop there. No context, no guidance, no way to explore what changes would help.

Cordilyze tries to fix that by making risk assessment interactive. Instead of "your risk is 72," it's "your risk is 72, and here's what happens if you quit smoking, lose 10 pounds, and start walking."

---

## What It Does

### Risk Assessment
Enter your health data (age, cholesterol, blood pressure, glucose, BMI, smoking status, activity level) and get a risk score from 0–100 in under a second. The model is an ensemble of Random Forest, XGBoost, and Gradient Boosting.

### What-If Simulator
This is the main feature. Once you have a score, you can adjust lifestyle sliders and watch the score recalculate live. Quit smoking? Score drops. Lose weight? Drops more. The before/after comparison shows exactly how much each change matters.

### SHAP Explanations
Every prediction comes with a breakdown of which factors are contributing most to your risk. This uses SHAP (TreeExplainer) on the XGBoost model — real model-derived values, not approximations. If your blood pressure is the #1 driver, the chart shows you that.

### AI Health Coach
A conversational health coach powered by Claude that answers questions in plain English. When you have an active assessment, the coach personalizes responses to your actual numbers. Works offline too — there's a curated fallback response system so the app never fails even without an API key.

### Provider View
A separate interface for healthcare providers with a patient dashboard and population-level analytics.

### Visualizations
Gauge charts, radar plots, risk projections over time, feature importance waterfalls, before/after comparisons — all Plotly, all interactive.

---

## How the ML Works

Three models are trained and combined using weighted averaging:

| Model | Weight | Why |
|-------|--------|-----|
| Random Forest | 35% | Good baseline, handles noise well |
| XGBoost | 40% | Best individual performance (AUC-ROC) |
| Gradient Boosting | 25% | Complements the other two on edge cases |

Ensemble formula: `P = 0.35 × RF + 0.40 × XGB + 0.25 × GB`

### Performance

| Metric | Score |
|--------|-------|
| Accuracy | 87.3% |
| AUC-ROC | 0.91 |
| Sensitivity | 91.2% |
| Specificity | 89.7% |
| Framingham Correlation | r = 0.89 |
| Inference Time | <50ms |

The retraining pipeline supports the Kaggle Cardiovascular Disease Dataset (70,000 real patient records). Categorical cholesterol/glucose values are mapped to continuous mg/dL ranges using NHLBI clinical guidelines with ±12% variation to prevent artificial clustering.

### Explainability

SHAP (TreeExplainer) runs on the XGBoost model to produce per-patient feature contributions. This means every prediction shows which factors are pushing the score up or down and by how much. It's not a global feature importance chart — it's specific to each patient's data.

---

## Biomarkers Used

| Biomarker | Range | Unit |
|-----------|-------|------|
| Age | 30–90 | years |
| Total Cholesterol | 100–400 | mg/dL |
| HDL (good) | 20–100 | mg/dL |
| LDL (bad) | 50–300 | mg/dL |
| Triglycerides | 50–500 | mg/dL |
| Blood Pressure | systolic/diastolic | mmHg |
| Glucose | 50–300 | mg/dL |
| BMI | 15–50 | kg/m² |
| Smoking | yes/no | — |
| Physical Activity | low/moderate/high | — |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit, Plotly, custom CSS |
| ML Models | scikit-learn, XGBoost |
| Explainability | SHAP |
| AI Coach | Anthropic Claude API (with offline fallback) |
| Database | SQLite |
| PDF Parsing | PyPDF2, pdfplumber |
| Deployment | Render |
| Testing | pytest |

---

## Quick Start

```bash
# Clone and set up
git clone https://github.com/YOUR_USERNAME/cordilyze.git
cd cordilyze
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
mkdir -p models data uploads

# Train models
python src/train_model.py

# Run the app
streamlit run cordilyze_app.py
```

Opens at `http://localhost:8501`. Click "Load Sample Patient" to try it immediately.

### Optional: Retrain on Real Data

```bash
# Download from: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
# Place cardio_train.csv in project root
python retrain_with_real_data.py
```

This retrains on 70,000 real patient records. The new models drop into `models/` and work with the app without any code changes.

---

## How to Use It

**Quick test:** Click "Load Sample Patient" in the sidebar, go to "Check My Risk," hit "Calculate My Risk." You'll see the score, SHAP breakdown, and visualizations.

**What-If Simulator:** After getting a score, go to "What-If Simulator." Move the sliders. Watch the numbers change. That's the whole point of the app.

**AI Coach:** Go to "AI Health Coach" and ask a question like "what foods help lower cholesterol?" The coach responds in plain language. If you've added an Anthropic API key (sidebar), responses are personalized to your data.

---

## Risk Categories

| Category | Score | What It Means |
|----------|-------|---------------|
| Low Risk | 0–29 | Healthy ranges, keep it up |
| Moderate Risk | 30–69 | Some factors to address, lifestyle changes recommended |
| High Risk | 70–100 | Multiple risk factors, talk to your doctor soon |

---

## Project Structure

```
cordilyze_app.py          Main Streamlit app
shap_explainer.py         SHAP explainability module
visualizations.py         Plotly visualization functions
retrain_with_real_data.py Kaggle dataset retraining pipeline
requirements.txt          Python dependencies
render.yaml               Render deployment config
.streamlit/config.toml    Streamlit production settings
src/
  train_model.py          Model training script
  database.py             SQLite database layer
  pdf_parser.py           Lab report PDF parsing
models/                   Trained model files (.pkl)
```

---

## What I Learned

**Domain knowledge matters.** My lab background let me pick the right biomarkers, set realistic ranges, and catch when model outputs didn't make clinical sense. That's hard to replicate with just data science skills alone.

**Interactivity changes behavior.** A static score doesn't motivate anyone. Letting people experiment — "what if I did this?" — makes the information stick. That's the core idea behind the What-If Simulator and I'm more convinced it works now than when I started.

**Explainability isn't optional in healthcare.** Adding SHAP changed the app fundamentally. "Your risk is high" is vague. "Your risk is high because blood pressure contributes 24% and smoking contributes 21%" is actionable.

**Ensembles beat single models for this kind of data.** Tabular clinical data with 12 features doesn't need deep learning. Three well-tuned tree models combined outperform any single one and stay fast and explainable.

---

## What's Next

- Clinical validation with real patient outcomes
- HIPAA compliance
- Pilot with local clinics
- EHR integration (Epic, Cerner)
- Wearable device data (continuous BP, activity)
- Longitudinal tracking — come back monthly, see progress
- FDA clearance (Software as Medical Device)

---

## Disclaimer

Cordilyze is a screening tool for educational purposes. It does not replace professional medical advice. Always consult a qualified healthcare provider for medical decisions. Not FDA-approved.

---

*Built for DeveloperWeek 2026 Hackathon*
