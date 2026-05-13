#                       🧬 MAO-ML — Imbalanced Chemical Classification Pipeline
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-yellow)](https://huggingface.co/spaces/Tser-vak/Mao_B_pred)

[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)
[![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-3776AB?style=flat&logoColor=white)](https://www.rdkit.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-FF6600?style=flat)](https://xgboost.ai/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-02B875?style=flat)](https://lightgbm.readthedocs.io/en/latest/)

#                       

An end-to-end Machine Learning pipeline for binary classification of bioactive molecules against Monoamine Oxidase B (MAO-B). Built to handle class imbalance, high-dimensional descriptor noise, and the overfitting traps that come with small chemical datasets.

The pipeline can be found in the `training` folder, and the final model is in the `model` folder (open-source and free to use)

---

## 📌 What This Pipeline Does

Takes a CSV of molecules with 2D RDKit descriptors → cleans the data → trains 3 classifiers (Random Forest, XGBoost, LightGBM) with Bayesian hyperparameter optimization → selects the best model → exports it as a production-ready ONNX file.

Everything is tracked in MLflow. The final model is framework-agnostic and ready for backend deployment.

---

## 🧪 How the Training Works

The pipeline runs in **four phases**, all orchestrated from `training/train.py`:

```
Phase 1 ─ Data Cleaning
   CSV → drop metadata → impute NaN → remove zero-variance features
   → remove correlated features (|r| > 0.90) → train/test split (80/20)

Phase 2 ─ Hyperparameter Optimization
   For each {RandomForest, XGBoost, LightGBM} × {SMOTE, No-SMOTE}:
       30 Bayesian trials (TPE) → 5-fold Stratified CV per trial
       → Penalized MCC Loss → log to MLflow
   Select global winner across all 6 combinations

Phase 3 ─ Final Evaluation
   Retrain winner on full training set → evaluate on holdout
   → log Confusion Matrix + PR curve to MLflow

Phase 4 ─ ONNX Export
   Slice scaler to match selected features → build inference pipeline
   → convert to ONNX → save model + feature manifest
```

### Data Preprocessing

Before any model sees the data, the pipeline automatically:

- **Drops metadata columns** (SMILES, ChEMBL IDs, etc.) that would cause errors in numeric computation.
- **Imputes missing values** — median fill if ≥10% of rows are affected, row deletion if <10% (preserves dataset size while staying clean).
- **Removes zero-variance features** — descriptors where every molecule has the same value carry no information.
- **Removes highly correlated features** (|r| > 0.90) — keeps one from each correlated pair to reduce redundancy without losing chemical information.
- **Selects top 25 features via RFE** (Recursive Feature Elimination with Random Forest) — this runs *inside* the pipeline to prevent data leakage during cross-validation.

### The Dual Imbalance Strategy

The dataset is ~4:1 inactive to active. Instead of picking one approach, every model is trained **both ways** and the best wins:

| Strategy | How it works |
|---|---|
| **Borderline-SMOTE** | Generates synthetic minority-class samples along the decision boundary. Uses `sampling_strategy=0.3` to avoid flooding the dataset with synthetic noise. |
| **Balanced class weights** | No synthetic data — instead reweights the loss function so misclassifying a rare active molecule costs more. Uses `class_weight='balanced'` (RF/LGBM) or `scale_pos_weight` (XGBoost). |

There's no universally superior imbalance strategy — it depends on the dataset geometry. By testing both, the pipeline adapts rather than assumes.

### The Penalized MCC Loss — Why This Matters

This is the core design decision that separates this pipeline from a standard AutoML run.

Standard Bayesian optimization just maximizes test MCC. The problem: the optimizer can find a model that memorizes the training folds (MCC_train ≈ 1.0) while scraping by on test MCC — and that mediocre score might still "win."

The custom loss function catches this:

```
if |MCC_train − MCC_test| > 0.15:
    Loss = −MCC_test + 0.8 × |MCC_train − MCC_test|    ← heavy penalty
else:
    Loss = −MCC_test                                     ← pure performance
```

- A **gap > 15%** between train and test MCC signals overfitting. The `α = 0.8` penalty makes these configurations look worse, steering the optimizer away.
- A **healthy gap** (≤ 15%) means the model generalizes well — it's judged purely on test performance.

**Why MCC instead of F1 or accuracy?** MCC is the only metric that produces a high score *only if* the model performs well on all four quadrants of the confusion matrix (TP, TN, FP, FN). Accuracy is dominated by the majority class, and F1 ignores true negatives entirely — both are unreliable on imbalanced data.

### Zero Data Leakage

All preprocessing (scaling → RFE → SMOTE → classifier) lives inside a single `imblearn.Pipeline` or `sklearn.Pipeline` object. During cross-validation, each fold fits its own scaler, selects its own features, and generates its own synthetic samples. The validation fold is never touched during fitting — this is what makes the CV scores trustworthy.

---

## 📁 Repository Structure

```
MAO-ML/
│
├── README.md                          ← You are here
├── requirements.txt                   ← Python dependencies
│
├── training/                          ← Training logic
│   ├── train.py                       ← Main orchestrator (Phases 1–4)
│   ├── data/
│   │   └── data.py                    ← Data loading, cleaning, splitting, ONNX export
│   ├── classifiers/
│   │   └── clf.py                     ← Model definitions + Hyperopt search spaces
│   ├── descriptors_handeling/
│   │   └── feature_hand.py            ← RFE feature selection wrapper
│   └── visualization/
│       └── visual.py                  ← Confusion matrix & PR curve → MLflow
│
├── model/                             ← Production inference
│   ├── Production_RandomForest.onnx   ← Exported ONNX model
│   ├── Production_Features_*.json     ← Feature manifest (descriptor names)
│   └── pred_run.py                    ← CLI prediction script (SMILES → Activity)
│
├── models/                            ← Historical model run outputs
├── testing_script/                    ← External validation datasets & results
├── mlruns/                            ← MLflow artifacts (auto-generated)
└── mlflow.db                          ← MLflow tracking database
```

---

## 🚀 How to Use

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Train

1. Set your CSV path in `training/train.py`:
   ```python
   file = r'C:\path\to\your\dataset.csv'
   ```

2. Configure which columns to ignore (metadata/non-numeric):
   ```python
   cleaner = DataProcessor(
       file_path=file,
       ignore_desc=["Molecule ChEMBL ID", "Smiles", ...],
       labels="Label"
   )
   ```

3. Run:
   ```bash
   python training/train.py
   ```

The pipeline trains all 6 model × strategy combinations (180 total evaluations), picks the winner, evaluates on holdout, and exports the ONNX model to `models/`.

### View Results in MLflow

```bash
mlflow ui --host 127.0.0.1 --port 5001 --backend-store-uri sqlite:///mlflow.db
```

Open `http://127.0.0.1:5001` to compare runs, view metrics, and inspect logged plots.


## 📦 Dependencies

`scikit-learn` · `imbalanced-learn` · `lightgbm` · `xgboost` · `hyperopt` · `mlflow` · `skl2onnx` · `onnxruntime` · `rdkit` · `pandas` · `numpy` · `matplotlib` · `seaborn`

See `requirements.txt` for pinned versions.
