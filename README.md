# ROGII Wellbore Geology Prediction - Hybrid ML+Physics Pipeline

**Competition**: ROGII Wellbore Geology Prediction  
**Final Result**: **10.697 RMSE** (beats 10.70 target)  
**Approach**: Hybrid machine learning with physics-informed features and calibrated trajectory inference

---

## Overview

This solution combines gradient boosting (LightGBM + CatBoost) with physics-informed features for wellbore geology prediction. The key innovation is **calibrating gamma-ray (GR) measurements before physics-based trajectory inference**, which improves both direct physics candidates and ML features.

**Core Idea**: Horizontal wells have systematic GR measurement biases compared to type wells. By estimating affine calibration parameters (scale + offset) from the visible prefix and applying them before particle filtering and beam search, we obtain more accurate trajectory predictions that translate to better ML features.

---

## Project Structure

```
.
├── config.yaml              # Configuration file
├── main.py                  # Main orchestrator script
├── requirements.txt         # Python dependencies
├── setup_env.sh             # Environment setup script
├── kaggle_token.txt         # Kaggle API token (keep private!)
├── JOURNAL.md               # Complete experiment journal
├── v4_summary.md            # Experiment summary
└── src/
    ├── __init__.py
    ├── data_loader.py       # Automatic data download
    ├── utils.py             # Helper functions
    ├── physics.py           # Physics-informed features
    │   ├── BeamSearch       # Multiple beam configurations
    │   └── FormationPlaneKNN # Spatial interpolation
    ├── dataset.py           # Feature engineering (162 features)
    ├── model.py             # Stacked ensemble (LGBM + CatBoost + Ridge)
    ├── train.py             # Cross-validation training
    ├── validate.py          # OOF evaluation and feature importance
    └── inference.py         # Test predictions and submission generation
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Token

The Kaggle API token is already stored in `kaggle_token.txt`. Load it with:

```bash
source setup_env.sh
```

Or manually:

```bash
export KAGGLE_API_TOKEN=$(cat kaggle_token.txt)
```

### 3. Run the Pipeline

**Full Training (Production Mode):**

```bash
source venv/bin/activate
python main.py
```

**Expected Output**: OOF RMSE ≈ 10.697

**Dry Run (Pipeline Validation Mode):**

```bash
python main.py --config test_config.yaml
```

The pipeline will:
1. **Automatically download** competition data if not present (using kagglehub)
2. Build feature matrices with 162 physics-informed features
3. Train 5-fold GroupKFold stacked ensemble (LightGBM + CatBoost + Ridge)
4. Evaluate out-of-fold predictions
5. Generate submission.csv

---

## Configuration Modes

| Mode | Config File | Use Case | Runtime | Expected RMSE |
|------|-------------|----------|---------|---------------|
| **Production** | `config.yaml` | Full training for best score | ~45-60 min | ~10.697 |
| **Dry Run** | `test_config.yaml` | Pipeline validation, debugging | ~1-2 min | N/A (tiny data) |

**Dry Run Mode** (`test_config.yaml`):
- Processes only 3 training wells, 2 test wells
- Limits to 50 rows per well
- Uses 2 folds, 10 boosting rounds
- Tiny model (7 leaves)
- Only 2 formations, 1 beam configuration
- **Purpose**: Verify pipeline works end-to-end, NOT for quality evaluation

---

## Feature Engineering (162 Features)

### 1. GR Calibration (Key Innovation)

Before any physics inference, we estimate affine calibration parameters from the visible prefix:

```
tw_gr ≈ a * kgr_raw + b  →  kgr_calibrated = a * kgr_raw + b
```

This corrects systematic GR measurement biases between horizontal wells and type wells, improving downstream trajectory estimation.

**Calibrated Features**:
- `pf_ancc_cal_d` - Calibrated PF-ANCC delta
- `pf_ancc_cal_std` - Calibrated PF-ANCC uncertainty
- `pf_z_cal_d` - Calibrated PF Z-velocity delta
- `pf_z_cal_std` - Calibrated PF Z-velocity uncertainty

### 2. Particle Filtering (PF) Trajectories

Multi-seed particle filtering with likelihood-weighted aggregation:
- 8 seeds, 400 particles, temperature T=3.0
- Momentum-based smoothing (momentum=0.993)
- Velocity noise injection (0.004)
- Both raw and calibrated GR variants

**Features**:
- `pf_multiseed_lw_d`, `pf_multiseed_lw_std` - Multi-seed PF delta and uncertainty
- `pf_ancc_d`, `pf_ancc_std` - Raw PF-ANCC features
- `pf_z_d`, `pf_z_std` - Raw PF Z-velocity features
- `pf_ancc_cal_d`, `pf_ancc_cal_std` - **Calibrated** PF-ANCC features (key improvement)
- `pf_z_cal_d`, `pf_z_cal_std` - **Calibrated** PF Z-velocity features (key improvement)

### 3. Beam Search Variants (9 configurations)

Multiple beam search configurations for robust trajectory estimation:
- `beam_search_1` through `beam_search_9`
- Varying beam sizes, move costs, and emission scales
- Both raw and calibrated GR variants

### 4. Multi-Scale Normalized Cross-Correlation (NCC)

GR sequence matching at multiple scales:
- `ncc_32`, `ncc_64`, `ncc_128`, `ncc_256` - Different window sizes
- Captures both local and global GR pattern similarity

### 5. Dense ANCC Imputation

Adaptive normalized cross-correlation with dense imputation:
- `ancc_imputed_delta` - Imputed ANCC delta
- `ancc_imputed_std` - Imputed ANCC uncertainty

### 6. Formation Plane KNN

Spatial interpolation of formation surfaces from nearest wells:
- `fpknn_tvt_delta` - Formation plane KNN TVT delta
- `fpknn_tvt_std` - Formation plane KNN TVT uncertainty

### 7. GR Sequence Features

Rolling statistics and trend features:
- `gr_rolling_mean`, `gr_rolling_std` - Local GR statistics
- `gr_slope` - GR trend direction
- `gr_seq_features` - GR sequence pattern features

### 8. Calibration Offset Features

Self-calibration offsets from different methods:
- `tda_*` - ANCC-based calibration
- `tdbc_*` - Beam search calibration
- `tdpf_*` - Particle filter calibration

### 9. Persistence Features

Last-known TVD true vertical thickness:
- `persistence_tvt` - Simple baseline feature
- Used for distance-based blending in early rows

---

## Model Architecture

### Stacked Ensemble

**Level 1 (Base Learners)**:
- LightGBM (primary) - 5-fold GroupKFold
- CatBoost (secondary) - 5-fold GroupKFold

**Level 2 (Meta-Learner)**:
- Ridge Regression - combines LGBM and CatBoost predictions

**Why Stacking?**: CatBoost often captures different signal than LightGBM due to different handling of categorical features and regularization. Ridge meta-learner optimally weights both.

### Training Configuration

```yaml
training:
  n_folds: 5
  seed: 42
  parallel_jobs: 15  # Parallel feature extraction
  
model:
  n_estimators: 50000
  learning_rate: 0.01
  num_leaves: 31
  max_depth: 8
  min_child_samples: 20
  subsample: 0.8
  colsample_bytree: 0.8
  reg_alpha: 0.1
  reg_lambda: 0.1
  
catboost:
  iterations: 50000
  learning_rate: 0.01
  depth: 8
  l2_leaf_reg: 3.0
  subsample: 0.8
```

### Validation Strategy

- **GroupKFold**: Wells are grouped to prevent data leakage (rows from same well stay in same fold)
- **Out-of-Fold (OOF)**: Unbiased evaluation on held-out folds
- **Final Metric**: Grouped RMSE on OOF predictions

---

## Key Insights

### What Works

✅ **GR Calibration Before Physics Inference** - The single biggest improvement (+0.063 RMSE). Calibrating GR measurements before PF and beam search provides more accurate trajectory estimates.

✅ **Stacked Ensemble** - CatBoost adds complementary signal to LightGBM, especially on wells with different GR characteristics.

✅ **Multi-Seed PF** - While multi-seed PF alone is worse than ML (16.25 vs 10.76 RMSE), adding it as features provides marginal gain.

✅ **Distance-Based Persistence Blending** - Fixes early-row predictions (0-25 ft after cutoff) where model initially struggled.

### What Doesn't Work

❌ **Pure Physics Candidates** - Even the best physics candidate (calibrated PF at 16.29 RMSE) cannot beat ML (10.76 RMSE).

❌ **Prefix-Only Selection** - All 773 wells are 100% hidden (competition design), so no visible prefix exists for calibration-based selection.

❌ **Catastrophic Well Repair** - Worst wells (RMSE 25-55) show high-variance errors, not systematic bias. Physics candidates cannot beat ML even on these wells.

---

## Performance

### Final Results

| Component | RMSE | Δ vs Baseline | Status |
|-----------|------|---------------|--------|
| Baseline (73 features) | 10.760 | - | Reference |
| **Full Pipeline (162 features)** | **10.697** | **+0.063** | ✅ **Final** |

**Target**: < 10.70 ✅ **ACHIEVED**

### Feature Importance (Top 10)

Typically, the most important features are:
1. `pf_ancc_cal_d` - Calibrated PF-ANCC delta
2. `beam_search_*_tvt` - Beam search TVT estimates
3. `ncc_*` - Multi-scale NCC features
4. `fpknn_tvt_delta` - Formation plane KNN delta
5. `gr_rolling_*` - GR rolling statistics

(Exact rankings vary by fold; see `artifacts/feature_importance.parquet` for details)

---

## Artifacts

After running `main.py`, you'll find:

```
artifacts/
├── oof_predictions.parquet    # Out-of-fold predictions
├── feature_importance.parquet # Per-fold feature importance
├── models/                    # Trained model checkpoints
│   ├── fold_0_lgbm.pkl
│   ├── fold_0_cb.pkl
│   └── ...
└── submission.csv             # Kaggle submission file
```

Experiment-specific artifacts:
```
artifacts/experiment_results/
├── a1/                        # GR calibration analysis
├── s1/                        # Multi-seed PF analysis
├── s2/                        # Prefix selection analysis
└── s3/                        # Catastrophic well analysis
```

---

## Reproducing Results

### Full Pipeline

```bash
cd /home/ext.mmakhlin/Projects/rogii
source venv/bin/activate
python main.py
```

**Expected**: OOF RMSE ≈ 10.697

### Individual Components

**GR Calibration Analysis**:
```bash
python scripts/a1_gr_calibration.py
```

**Multi-Seed PF Analysis**:
```bash
python scripts/s1_v2_likelihood_weighted_pf.py
```

**Distance Blend Analysis**:
```bash
python scripts/crossfit_distance_blend.py
```

---

## Kaggle Submission

The same code works on Kaggle:
1. Upload all files to your Kaggle notebook
2. The code auto-detects the Kaggle environment and skips download
3. Run `python main.py` in a notebook cell

---

## Requirements

- Python 3.8+
- See `requirements.txt` for package versions

Key dependencies:
- `lightgbm` - Primary model
- `catboost` - Secondary model
- `scikit-learn` - Preprocessing, validation
- `pandas`, `numpy` - Data manipulation
- `scipy` - Signal processing (NCC, etc.)
- `kagglehub` - Automatic data download

---

## License

This is a competition baseline for educational purposes.

---

## References

- **JOURNAL.md** - Complete experiment journal with all iterations
- **v4_summary.md** - Experiment summary and decision tree
- **scripts/** - Individual experiment scripts for reproducibility
