# src/train.py
import os
import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from src.utils import setup_logger
from src.model import LGBMWrapper

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
        
        oof_residuals = np.zeros(len(train_df), dtype=np.float64)
        fold_models = []
        
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
            
            # Initialize and train model
            model = LGBMWrapper(self.config)
            model.fit(X_train, y_train, X_valid, y_valid, self.feature_cols)
            
            # Generate validation predictions (residuals)
            pred_residuals = model.predict(X_valid)
            oof_residuals[valid_idx] = pred_residuals
            
            # Save fold model
            model_path = self.artifacts_dir / f"lgbm_fold_{fold + 1}.txt"
            model.save(str(model_path))
            fold_models.append(model)
            
            # Calculate and log isolated fold residual RMSE
            rmse_res = float(np.sqrt(np.mean((pred_residuals - y_valid)**2)))
            fold_time = time.time() - fold_start
            self.logger.info(f"Fold {fold + 1} Residual RMSE: {rmse_res:.4f} (completed in {fold_time:.1f}s)")
            
        self.logger.info(f"All {self.n_folds} folds completed.")
        return fold_models, oof_residuals, train_df
