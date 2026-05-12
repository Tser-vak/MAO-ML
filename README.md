# 🧬 Imbalanced Chemical Machine Learning Pipeline.

**An Integrated Master's Thesis Project combining Machine Learning, 2D Molecular Descriptors, and Orthogonal Validation via Molecular Docking.**

## 📌 Overview
This repository contains a production-ready Machine Learning pipeline designed to act as a **first-pass High-Throughput Virtual Screening (HTVS) tool**. The goal is to identify active chemical scaffolds that inhibit Monoamine Oxidase B (MAO-B), a major target in Parkinson's disease therapies.

By prioritizing Precision, the model rapidly filters massive molecular libraries (e.g., thousands of molecules), ensuring that computationally expensive downstream processes (like 3D molecular docking) are spent only on the highest-probability candidates.

## 🔬 The Science & Methodology

### 1. Scaffold Hunting (The 100 nM Threshold)
In this dataset, "Active" molecules are defined by an IC50 100,000 nM. While this indicates weak binding for a final drug, it is a deliberate choice for **scaffold hunting**. 
The ML model casts a wide net to identify core structural geometries that successfully interact with the MAO-B pocket. These scaffolds can later be optimized by medicinal chemists 
into nanomolar-range therapeutics.

### 2. Orthogonal Validation (Ligand vs. Structure)
The ultimate validation of this model bridges two distinct scientific domains:
* **Ligand-Based (This ML Model):** Predicts activity based purely on 2D chemical descriptors and features.
* **Structure-Based (Molecular Docking):** Predicts activity using thermodynamic physics and 3D protein crystal structures.

As a negative control, molecules with intentionally poor thermodynamic docking scores were fed into the ML model. 
The model correctly classified them as inactive, proving that the algorithm did not merely memorize dataset bias, but successfully learned the physical constraints of the MAO-B binding pocket.

## ⚙️ The Engineering Architecture
* **Strict Pipelines:** Utilizing `imblearn.pipeline` and `sklearn.pipeline` to encapsulate SMOTE, scaling, and feature selection, mathematically guaranteeing zero data leakage during cross-validation.
* **Feature Engineering:** Automated reduction from 211 initial descriptors down to the 25 most critical features using Recursive Feature Elimination (RFE) via LightGBM.
* **Experiment Tracking:** Integrated `MLflow` for logging hyperparameters, Confusion Matrices, and ROC-AUC curves.
* **Production Deployment:** The final tuned pipeline is exported as a lightweight, framework-agnostic **ONNX** graph (`skl2onnx`), ready for decoupled backend deployment.

## 🚀 Post-Thesis Update: The Penalized MCC Optimization
*Note: While the original thesis was evaluated using the F1-score, this codebase has been updated to utilize the Matthews Correlation Coefficient (MCC) with a custom penalized loss function.*

**The Problem:** The chemical dataset is highly imbalanced (~4:1 inactive to active). Standard F1-scores can be misleading, and tree-based models (Random Forest, XGBoost) easily overfit 
the minority class during standard Bayesian optimization.

**The Solution:** Hyperparameter tuning via `Hyperopt` was rewritten to optimize a **Custom Penalized MCC Loss Function**:
