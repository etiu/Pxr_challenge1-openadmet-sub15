# PXR Challenge — Activity Prediction Pipeline

ML pipeline for predicting PXR (Pregnane X Receptor) agonist potency (pEC50)
for the [OpenADMET PXR Challenge](https://huggingface.co/spaces/openadmet/pxr-challenge).

## Current status

**Submission 15** — Ridge-stacked ensemble (ChemProp+CheMeleon, TabPFN, LightGBM)
on RDKit + MOE + Morgan fingerprint + Uni-Mol features, scaffold-based CV.


## Competition context

The PXR Challenge runs in two phases:
- **Phase 1**: blind prediction of all 513 test compounds (Analog Set 1 + Analog Set 2)
- **Phase 2**: Analog Set 1 (253 compounds) unblinded for use in training;
  Analog Set 2 (260 compounds) remains blind and is scored at the final deadline

This repo's pipeline handles the Phase 2 transition: folding the newly-unblinded
253 compounds into training while correctly identifying the true 260-compound
blind test set, and reassembling all 513 original test rows (260 predicted +
253 known) for final submission per organizer requirements.

## Pipeline overview

### 1. Data loading & Phase 2 setup
- Loads `train_df`, `test_df`, `train_counter_df`, `train_single_df`
- Loads `test_phase1` (unblinded Analog Set 1) and splits it out of `test_df`,
  folding it into the training set with provenance tagging
- True blind test set: 260 rows (down from the original 513)

### 2. Counter-assay selectivity filtering
Removes non-selective compounds where `counter_pEC50 >= pEC50` (compounds with
no counter-assay measurement are kept). This filtering step was identified as
the single largest driver of MAE improvement in earlier submissions
(~0.51 → ~0.46 MAE).

### 3. Feature engineering
| Feature block | Description | Dimensions |
|---|---|---|
| RDKit descriptors | Standard 2D descriptors | ~200 |
| MOE descriptors | 3D/surface descriptors (ASA_H, glob, vol, pmi1/2/3, vsurf, SlogP_VSA, TPSA) | ~358 |
| Morgan fingerprints | ECFP-style, radius 2 | 2048 |
| Uni-Mol embeddings | 3D-conformer-aware CLS-token embeddings | 512 |

Different base models consume different subsets of these features:

| Model | Features used |
|---|---|
| ChemProp + CheMeleon | SMILES only (graph-native — none of the engineered feature blocks reach it) |
| TabPFN | RDKit + MOE only (curated subset, ~558 dims, to stay closer to its native pretraining feature range) |
| LightGBM | RDKit + MOE + Morgan + Uni-Mol (full combined set, ~3,118 dims) |

### 4. Cross-validation
Murcko scaffold-based fold assignment (3 folds) — chosen over random KFold,
which was found to give overly optimistic CV estimates due to scaffold leakage
between folds.

### 5. Base models
- **ChemProp + CheMeleon**: graph neural network with CheMeleon foundation
  model initialization, trained per-fold via CLI subprocess
- **TabPFN v2**: tabular foundation model, curated features,
  `ignore_pretraining_limits=True` to exceed its native 500-feature cap
- **LightGBM**: gradient-boosted trees, full feature set

### 6. Ensembling
Out-of-fold predictions from each base model are combined via a Ridge
meta-learner (current submission) or simple averaging (under evaluation —
see below).

## Repo structure

```
submission_moe_filtered_v2_sub15.ipynb   # main pipeline notebook
```


### Per-model OOF diagnostics (Submission 15 base models)

| Model | OOF MAE | OOF R² |
|---|---|---|
| TabPFN | 0.4459 | 0.6642 |
| ChemProp | 0.4577 | 0.6434 |
| LightGBM | 0.4710 | 0.6286 |

LightGBM was consistently the weakest individual model. With only 3 CV folds
feeding the Ridge meta-learner, the learned blend weights appear to overfit
rather than improve on a plain average of the two stronger models.
