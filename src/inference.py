# src/inference.py
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.utils import setup_logger

class Inferencer:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger("Inferencer")
        self.submission_file = config["paths"]["submission_file"]
        
        # Dry run settings
        self.dry_run = config.get("dry_run", {})
        self.dry_run_enabled = self.dry_run.get("enabled", False)

    def predict_and_save(self, test_df, fold_models, feature_cols, sample_sub):
        """Generates ensemble predictions and writes submission.csv."""
        self.logger.info("Starting inference on test data...")
        
        # Sample test data for dry run
        if self.dry_run_enabled and len(test_df) > 50:
            test_df = test_df.iloc[:50].reset_index(drop=True)
            self.logger.info(f"[DRY RUN] Limited test data to 50 samples")
        
        self.logger.info(f"Test samples: {len(test_df):,}, Models: {len(fold_models)}")
        
        if len(test_df) == 0:
            self.logger.warning("Test feature matrix is empty! Generating zero-filled submission as fallback.")
            submission = sample_sub[["id"]].copy()
            submission["tvt"] = 0.0
            submission.to_csv(self.submission_file, index=False)
            return

        # Handle missing columns (can happen in dry run mode with reduced formations)
        # Add missing columns with zeros to match training feature count
        missing_cols = set(feature_cols) - set(test_df.columns)
        if missing_cols:
            self.logger.warning(f"Adding {len(missing_cols)} missing feature(s) with zeros: {list(missing_cols)[:5]}...")
            for col in missing_cols:
                test_df[col] = 0.0
        
        X_test = test_df[feature_cols].astype(np.float32).values
        fold_preds = np.zeros((len(test_df), len(fold_models)), dtype=np.float64)
        
        self.logger.info("Generating predictions from fold models...")
        infer_start = time.time()
        
        for j, model in enumerate(tqdm(fold_models, desc="Predicting with folds")):
            fold_preds[:, j] = model.predict(X_test)
        
        infer_time = time.time() - infer_start
        self.logger.info(f"Predictions completed in {infer_time:.1f}s (~{len(test_df)/max(infer_time, 0.1):.0f} samples/sec)")
            
        # Average the predicted residuals across all folds
        pred_residual = fold_preds.mean(axis=1)
        
        # Defensive clipping based on typical data constraints
        residual_clip = 400.0
        pred_residual = np.clip(pred_residual, -residual_clip, residual_clip)
        
        # Reconstruct absolute TVT
        pred_tvt = test_df["last_known_TVT"].values + pred_residual
        test_df["tvt_pred"] = pred_tvt
        
        self.logger.info(f"Predicted Residual Range: {pred_residual.min():.2f} to {pred_residual.max():.2f}")
        self.logger.info(f"Predicted TVT Range: {pred_tvt.min():.2f} to {pred_tvt.max():.2f}")
        
        # Format submission
        pred_map = test_df.set_index("id")["tvt_pred"].to_dict()
        submission = sample_sub[["id"]].copy()
        submission["tvt"] = submission["id"].map(pred_map)
        
        # Defensive fallback for any missing predictions
        if submission["tvt"].isna().any():
            self.logger.warning("Some predictions are missing. Falling back to well's last known TVT.")
            anchor_map = test_df.drop_duplicates("well_id").set_index("well_id")["last_known_TVT"].to_dict()
            miss = submission["tvt"].isna()
            submission.loc[miss, "tvt"] = (
                submission.loc[miss, "id"].str.rsplit("_", n=1).str[0].map(anchor_map)
            )
            submission["tvt"] = submission["tvt"].fillna(0.0)
            
        submission.to_csv(self.submission_file, index=False)
        self.logger.info(f"Submission saved to {self.submission_file} (Shape: {submission.shape})")
