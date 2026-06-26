# src/validate.py
import numpy as np
import pandas as pd
from src.utils import setup_logger

class Validator:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger("Validator")
        
        # Dry run settings
        self.dry_run = config.get("dry_run", {})
        self.dry_run_enabled = self.dry_run.get("enabled", False)

    def evaluate(self, train_df, oof_residuals, fold_models, feature_cols):
        """Calculates final OOF RMSE and feature importance."""
        self.logger.info("Evaluating Out-Of-Fold (OOF) predictions...")
        
        if self.dry_run_enabled:
            self.logger.info("[DRY RUN] Skipping detailed evaluation metrics")
        
        # Reconstruct Absolute TVT
        last_tvt_all = train_df["last_known_TVT"].values
        oof_tvt = last_tvt_all + oof_residuals
        y_true_all = train_df["target_tvt"].values
        
        # Calculate Competition Metric (RMSE on absolute TVT)
        rmse_oof_tvt = float(np.sqrt(np.mean((oof_tvt - y_true_all)**2)))
        rmse_oof_b0 = float(np.sqrt(np.mean((last_tvt_all - y_true_all)**2)))
        
        self.logger.info("--- Final Evaluation ---")
        self.logger.info(f"OOF RMSE (Model):      {rmse_oof_tvt:.4f}")
        self.logger.info(f"OOF RMSE (Persistence): {rmse_oof_b0:.4f}")
        improvement = rmse_oof_b0 - rmse_oof_tvt
        pct_improvement = (improvement / rmse_oof_b0) * 100
        self.logger.info(f"Improvement over Baseline: +{improvement:.4f} ft ({pct_improvement:.2f}%)")

        # Feature Importance Calculation
        if not self.dry_run_enabled:
            self.logger.info("Calculating Feature Importances...")
            imp_arr = np.zeros(len(feature_cols))
            for model in fold_models:
                # model.booster is the underlying LightGBM object
                imp_arr += model.booster.feature_importance(importance_type="gain")
            imp_arr /= len(fold_models)
            
            imp_df = pd.DataFrame({"feature": feature_cols, "gain": imp_arr})
            imp_df = imp_df.sort_values("gain", ascending=False).reset_index(drop=True)
            
            self.logger.info("Top 10 Features by Gain:")
            for idx, row in imp_df.head(10).iterrows():
                self.logger.info(f"{idx+1:2d}. {row['feature']:25s} {row['gain']:.2f}")
        else:
            self.logger.info("[DRY RUN] Skipping feature importance calculation")
            imp_df = pd.DataFrame()
            
        return imp_df
