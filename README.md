# BiasLens 
### Universal AI Bias Detection & Mitigation Framework

> Upload any dataset → automatically detect bias across demographic groups → get specific mitigation recommendations with before/after comparison.

---

## What is BiasLens?

BiasLens is a Streamlit web application that audits datasets and AI models for demographic bias. It works on **any CSV dataset** — no machine learning knowledge required. You upload your data, the tool finds the bias, explains it in plain English, and tells you exactly how to fix it.

**Author:** Ananyaa P (2127220501014)
**Guide:** Dr. N Rajganesh

---

## The Problem It Solves

AI systems used in hiring, loan approval, healthcare, and criminal justice have been shown to discriminate against certain demographic groups:

- **Amazon's hiring tool (2018)** — downranked CVs containing the word "women's"
- **COMPAS recidivism algorithm** — flagged Black defendants as high-risk at twice the rate of white defendants
- **Facial recognition tools** — error rates up to 34% for dark-skinned women vs under 1% for light-skinned men

BiasLens detects these biases automatically and quantifies them using 6 internationally recognised fairness metrics.

---

## Features

- **Upload any CSV** — hiring, loans, healthcare, criminal justice data
- **Auto-detects sensitive columns** — identifies gender, race, age, and outcome columns automatically
- **No model training needed** — audits tabular data directly
- **6 fairness metrics** — Bias Gap, Demographic Parity, Equalized Odds, Disparate Impact, Predictive Parity, Chi-square Imbalance
- **Plain-English verdicts** — explains what each failing metric means
- **Actionable mitigation** — reweighing, resampling, threshold tuning with before/after comparison
- **Image mode** — optional ResNet-18 training + Grad-CAM explainability for image datasets

---

## Demo Datasets

These real-world datasets are built into the app — no download needed:

| Dataset | Domain | Known Bias |
|---|---|---|
| COMPAS Recidivism | Criminal justice | Race bias in risk scores |
| Adult Income (UCI) | Census / income | Gender & race bias |
| German Credit | Loan approval | Age & gender bias |

---

## Fairness Metrics

| Metric | What it measures | Ideal value |
|---|---|---|
| Bias Gap | Max − min outcome rate across groups | < 0.05 |
| Demographic Parity Diff | Gap in positive decision rates | 0 |
| Equalized Odds Diff | TPR + FPR gap across groups | 0 |
| Disparate Impact Ratio | Min rate / max rate (legal standard) | ≥ 0.8 |
| Predictive Parity Gap | Precision gap across groups | 0 |
| Chi-square Imbalance | Dataset representation test | p > 0.05 |

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Ananyaa-Perumalsamy/BiasLens.git
cd BiasLens

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## Project Structure

```
BiasLens/
│
├── app.py                  # Main entry point, navigation
│
├── core/
│   ├── dataset.py          # Universal CSV loader, column auto-detection
│   ├── bias_metrics.py     # All 6 fairness metric calculations
│   ├── model.py            # BiasAwareCNN (ResNet-18 + embedding head)
│   ├── trainer.py          # Training loop with early stopping
│   └── xai.py              # Grad-CAM and SmoothGradCAM
│
├── pages/
│   ├── home.py             # Landing page
│   ├── upload.py           # Dataset upload and column configuration
│   ├── bias.py             # Bias report page
│   ├── dashboard.py        # Visual analytics dashboard
│   ├── mitigation.py       # Mitigation strategies with before/after
│   ├── train.py            # Image model training (optional)
│   └── xai.py              # Grad-CAM visualisation page
│
└── requirements.txt
```

