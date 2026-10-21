# Tabular ML Workbench: Credit Risk Modeling & Experiments

A modular sandbox and benchmarking workspace for exploring tabular data classification, feature engineering, and gradient boosting models (LightGBM, XGBoost) with reproducible cross-validation pipelines. Designed for seamless execution both on local workstations and Google Colab.

---

## 📌 Project Overview

This repository contains end-to-end experiments focusing on tabular risk assessment:
- **Task**: Binary classification (credit default risk prediction).
- **Core Models**: LightGBM, XGBoost.
- **Evaluation Metric**: Area Under the ROC Curve (ROC-AUC) via Stratified 5-Fold Cross Validation.
- **Goal**: Hands-on practice with systematic feature ablation, ratio-based financial engineering, and out-of-fold (OOF) ensembling.

---

## 🗂️ Directory Structure

```text
.
├── Competition/
│   ├── home_credit_improved.py         # Main modular training & evaluation pipeline
│   ├── train_xgboost.py                # Standalone XGBoost training script
│   ├── ablation_features.py            # Feature importance & ablation experiment runner
│   ├── feature_ablation_v3.py          # Extended feature study (ratios & external sources)
│   ├── home_credit_improved_colab.ipynb# Interactive notebook optimized for Google Colab
│   └── input/                          # Dataset directory (train.csv, test.csv, sample_submission.csv)
├── .gitignore                          # Standard git ignore rules (venv, checkpoints, system files)
└── README.md                           # Project documentation
```

---

## 🚀 Quick Start

### 1. Local Environment Setup

Clone the repository and install required packages:

```bash
git clone https://github.com/nguyendangcole/colab-world.git
cd colab-world

# Optional: Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install core dependencies
pip install lightgbm xgboost scikit-learn pandas numpy
```

### 2. Training Pipelines

Execute the full training pipeline with Stratified 5-Fold CV:

```bash
# Run LightGBM pipeline
python Competition/home_credit_improved.py \
  --input-dir Competition/input \
  --output-dir Competition/output_lightgbm_final \
  --models lightgbm \
  --folds 5 \
  --seed 42

# Run XGBoost baseline
python Competition/train_xgboost.py
```

### 3. Google Colab Workflow

To run experiments in Google Colab with GPU/TPU or cloud CPU:

```python
# 1. Clone repository into Colab workspace
!git clone https://github.com/nguyendangcole/colab-world.git
%cd colab-world

# 2. Install dependencies
!pip install -q lightgbm xgboost

# 3. Open and run:
# Competition/home_credit_improved_colab.ipynb
```

---

## 🔬 Experimentation Highlights

- **Ratio-Based Engineering**: Generating domain-specific indicators (e.g., credit-to-annuity, income-to-credit, payment-to-income ratios).
- **External Score Interactions**: Exploring non-linear interactions across normalized external credit indicators.
- **Leak-Free Cross Validation**: Stratified K-Fold setup ensuring categorical encoding, imputation, and metric calculations do not leak test/validation sets.
- **Automated Validation**: Integrated assertion checks for prediction shapes, missing values, and range constraints $[0, 1]$.

---

## 📄 License & Notes

This repository is maintained for personal practice, benchmarking, and machine learning skill development.
