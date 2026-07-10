# src/train.py
import os
import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from src.utils import setup_logger
from src.model import LGBMWrapper, CatBoostWrapper
from sklearn.linear_model import Ridge

class Trainer:
    def __init__(self, config, feature_cols):
        self.config = config
        self.feature_cols = feature_cols
        self.n_folds = config["training"]["n_folds"]
        self.artifacts_dir = Path(config["paths"]["artifacts_dir"])
        self.logger = setup_logger("Trainer")
        
        # Dry run settings
        self.dry_run = config.get("dry_run", {})
        self.dry_run_enabled = self.dry_run.get("enabled", False)

    def train_cv(self, train_df):
        """Runs GroupKFold cross-validation on the training dataset."""
        original_len = len(train_df)
        self.logger.info(f"Starting {self.n_folds}-fold GroupKFold training.")
        self.logger.info(f"Total samples: {len(train_df)}, Features: {len(self.feature_cols)}")
        
        # Sample data for dry run
        if self.dry_run_enabled:
            sample_size = min(100, len(train_df))
            train_df = train_df.iloc[:sample_size].reset_index(drop=True)
            self.logger.info(f"[DRY RUN] Limited training data to {sample_size} samples")
        
        X = train_df[self.feature_cols].astype(np.float32).values
        y = train_df["target_residual"].astype(np.float32).values
        groups = train_df["well_id"].values
        
        oof_lgb = np.zeros(len(train_df), dtype=np.float64)
        oof_cb = np.zeros(len(train_df), dtype=np.float64)
        fold_models_lgb = []
        fold_models_cb = []
        
        gkf = GroupKFold(n_splits=self.n_folds)
        fold_splits = list(gkf.split(X, y, groups))
        
        for fold, (train_idx, valid_idx) in enumerate(tqdm(fold_splits, desc="Training folds", unit="fold")):
            fold_start = time.time()
            self.logger.info(f"--- Fold {fold + 1}/{self.n_folds} ---")
            
            X_train, y_train = X[train_idx], y[train_idx]
            X_valid, y_valid = X[valid_idx], y[valid_idx]
            
            # Count unique wells to ensure no leakage
            train_wells = len(np.unique(groups[train_idx]))
            valid_wells = len(np.unique(groups[valid_idx]))
            self.logger.info(f"Train samples: {len(train_idx):,} ({train_wells} wells) | Valid samples: {len(valid_idx):,} ({valid_wells} wells)")
            
            # 1. Train LightGBM
            self.logger.info("Training LightGBM...")
            model_lgb = LGBMWrapper(self.config)
            model_lgb.fit(X_train, y_train, X_valid, y_valid, self.feature_cols)
            pred_lgb = model_lgb.predict(X_valid)
            oof_lgb[valid_idx] = pred_lgb
            
            model_path_lgb = self.artifacts_dir / f"lgbm_fold_{fold + 1}.txt"
            model_lgb.save(str(model_path_lgb))
            fold_models_lgb.append(model_lgb)
            
            # 2. Train CatBoost
            self.logger.info("Training CatBoost...")
            model_cb = CatBoostWrapper(self.config)
            model_cb.fit(X_train, y_train, X_valid, y_valid, self.feature_cols)
            pred_cb = model_cb.predict(X_valid)
            oof_cb[valid_idx] = pred_cb
            
            model_path_cb = self.artifacts_dir / f"cb_fold_{fold + 1}.cbm"
            model_cb.save(str(model_path_cb))
            fold_models_cb.append(model_cb)
            
            # Log Fold Metrics
            rmse_lgb = float(np.sqrt(np.mean((pred_lgb - y_valid)**2)))
            rmse_cb = float(np.sqrt(np.mean((pred_cb - y_valid)**2)))
            self.logger.info(f"Fold {fold + 1} RMSE | LGBM: {rmse_lgb:.4f} | CB: {rmse_cb:.4f}")
            
        self.logger.info(f"All {self.n_folds} folds completed.")
        
        # 3. Train Meta-Learner (Ridge Stacking)
        self.logger.info("Training Ridge Meta-Learner on OOF predictions...")
        meta_X = np.column_stack([oof_lgb, oof_cb])
        meta_model = Ridge(alpha=10.0)
        meta_model.fit(meta_X, y)
        
        final_oof = meta_model.predict(meta_X)
        meta_rmse = float(np.sqrt(np.mean((final_oof - y)**2)))
        self.logger.info(f"Meta-Learner OOF RMSE: {meta_rmse:.4f}")
        self.logger.info(f"Meta-Learner Weights (LGBM, CB): {meta_model.coef_}")
        
        # Save Meta-Learner Weights and features
        np.save(self.artifacts_dir / "meta_weights.npy", meta_model.coef_)
        np.save(self.artifacts_dir / "meta_intercept.npy", np.array([meta_model.intercept_]))
        
        feature_cols_path = self.artifacts_dir / "feature_cols.txt"
        with open(feature_cols_path, "w") as f:
            for col in self.feature_cols:
                f.write(col + "\n")
                
        # Save OOF
        train_df["oof_pred_lgb"] = oof_lgb
        train_df["oof_pred_cb"] = oof_cb
        train_df["oof_pred"] = final_oof
        train_df[["well_id", "row_index", "target_residual", "oof_pred_lgb", "oof_pred_cb", "oof_pred", "md_from_ps"]].to_csv(self.artifacts_dir / "oof_preds.csv", index=False)
        
        return {
            "lgbm_models": fold_models_lgb,
            "catboost_models": fold_models_cb,
            "ridge_model": meta_model,
        }

