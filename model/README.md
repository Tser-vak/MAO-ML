# 🏆 Model Results — RandomForest (SMOTE)

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-yellow)](https://huggingface.co/spaces/Tser-vak/Mao_B_pred)
[![GitHub](https://img.shields.io/badge/GitHub-Training%20Pipeline-blue)](https://github.com/Tser-vak/MAO-ML)

The global winner across all 6 model × strategy combinations was **Random Forest trained with Borderline-SMOTE**. This folder contains the exported production artifacts for this model.

---

## 📊 Holdout Set Performance

> Evaluated on a stratified 20% holdout that was never seen during training or hyperparameter optimization.

| Metric | Score |
|---|---|
| **MCC** | **0.420** ← main metric |
| Balanced Accuracy | 0.737 |
| Recall | 0.937 |

### Confusion Matrix Breakdown

```
                  Predicted Inactive   Predicted Active
Actual Inactive         37 (TN)            32 (FP)
Actual Active           53 (FN)           790 (TP)
```

| | |
|---|---|
| **True Negatives** (Correctly identified inactives) | 37 |
| **False Positives** (Inactives flagged as active — wasted effort) | 32 |
| **False Negatives** (Actives missed by the model) | 53 |
| **True Positives** (Correctly identified actives) | 790 |

**Raw specificity at 0.5 threshold: 53.6%** — the model was catching almost everything active, but at the cost of flagging too many inactives.

---

## ⚖️ The 0.60 Prediction Threshold

The model's default decision boundary is 0.5 (standard probability cutoff). At this threshold:

- **Recall was high (93.7%)** — very few active molecules were missed.
- **Specificity was low (53.6%)** — roughly half of inactive molecules were incorrectly flagged as active.

For a screening tool, generating too many false positives is costly — it sends inactive molecules downstream into more expensive validation steps. To correct this trade-off, the **production threshold was raised to 0.60**:

- A molecule is only predicted `Active` if the model's confidence is ≥ 60%.
- This reduces false positives (increases specificity) at the cost of missing a slightly larger fraction of true actives (lower recall).

This threshold is applied in `pred_run.py`:
```python
results['Predicted_Activity'] = np.where(confidence_scores >= 0.6, 'Active', 'Not Active')
```

---

## 🔬 Orthogonal Validation — Molecular Docking

To validate that the model learned the real chemistry of the MAO-B binding pocket (and didn't just memorize the training data), the exported ONNX model was tested against an external set of molecules ranked by **3D molecular docking scores** (AutoDock Vina) — a completely separate, physics-based method.

### Best Docking Candidates (Top 6)

The 6 molecules with the **best (most negative) docking scores** were fed into the model. Several of these six compounds are documented in the literature and are actively being researched as potential therapies for Parkinson’s disease.
**The model correctly predicted all 6 as `Active`**, with confidence scores ranging from **0.739 to 0.876**.

| ZINC ID | Predicted | Confidence |
|---|---|---|---|
| ZINC00000057657 | ✅ Active | 0.876 | [7,8-Dihydroxyflavone](https://pmc.ncbi.nlm.nih.gov/articles/PMC9881092/) |
| ZINC000000689683 | ✅ Active | 0.850 | Daphnoretin |
| ZINC000000119985 | ✅ Active | 0.739 | [Catechin](https://doi.org/10.1007/s13205-024-03934-7) |
| ZINC000000119978 | ✅ Active | 0.739 | [Catechin](https://doi.org/10.1007/s13205-024-03934-7) |
| ZINC000000338038 | ✅ Active | 0.830 | [Alpinetin](https://onlinelibrary.wiley.com/share/KUNZGIEI6YUUPWRIWXCD?target=10.1002/cns.70676) |
| ZINC000000001785 | ✅ Active | 0.837 | [Naringenin Hydrate]( https://doi.org/10.1016/j.heliyon.2021.e06684) |

### Worst Docking Candidates (Negative Control)

The molecules with the **worst docking scores** — poor thermodynamic fit with the MAO-B pocket — were also tested. The majority were correctly classified as `Not Active`, with noticeably lower confidence scores compared to the top group.

| ZINC ID | Predicted | Confidence |
|---|---|---|
| ZINC000000689737 | ✅ Not Active | 0.409 |
| ZINC000000199438 | ✅ Not Active | 0.532 |
| ZINC000000689650 | ⚠️ Active | 0.682 |
| ZINC000000968444 | ⚠️ Active | 0.630 |
| ZINC000000199433 | ✅ Not Active | 0.532 |
| ZINC000000119434 | ✅ Not Active | 0.467 |
| ZINC000001104937 | ✅ Not Active | 0.558 |
| ZINC000000753040 | ⚠️ Active | 0.707 |
| ZINC95 | ✅ Not Active | 0.555 |Random speciment| 
| ZINC142 | ✅ Not Active | 0.578 |Random speciment|

### What This Means

Two entirely independent methods — a **2D ligand-based ML model** trained on chemical descriptors, and a **3D structure-based docking** simulation using the physical protein crystal structure — agree on the same molecules. This cross-validation between ligand space and structural space is strong evidence that the model captured genuine biochemical signal, not a statistical artifact of the training data.

## 📁 Files in This Folder

| File | Description |
|---|---|
| `Production_RandomForest.onnx` | Trained Random Forest pipeline exported as an ONNX graph. Framework-agnostic — runs with `onnxruntime`. |
| `Production_Features_RandomForest.json` | Ordered list of the 25 RDKit descriptors the model expects as input. |
| `pred_run.py` | CLI script: takes a CSV of SMILES → computes descriptors → runs ONNX inference → outputs predictions with confidence scores. |

---

## 🚀 Running Predictions

```bash
python pred_run.py \
    --csv your_molecules.csv \
    --model Production_RandomForest.onnx \
    --desc Production_Features_RandomForest.json \
    --out predictions.csv
```

**Input:** A CSV with a SMILES column (column name is auto-detected).  
**Output:** The same CSV with two new columns — `Predicted_Activity` (`Active` / `Not Active`) and `Confidence` (0–1 probability score).

---
