import numpy as np
import pandas as pd
from pathlib import Path

from src.utils import setup_logger
from src.hybrid import HybridEnsemble
from src.prefix_selector import apply_prefix_selection_to_oof


class Validator:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger("Validator")
        self.artifacts_dir = Path(config["paths"]["artifacts_dir"])
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _rmse(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        valid = np.isfinite(y_true) & np.isfinite(y_pred)

        if valid.sum() == 0:
            return np.nan

        return float(np.sqrt(np.mean((y_true[valid] - y_pred[valid]) ** 2)))

    def evaluate(self, train_df, oof_residuals, fold_models, feature_cols):
        self.logger.info("Evaluating out-of-fold predictions")

        result = train_df.copy()
        result["oof_residual"] = np.asarray(oof_residuals, dtype=np.float64)
        result["prediction_tvt"] = result["last_known_TVT"] + result["oof_residual"]

        model_rmse = self._rmse(result["target_tvt"], result["prediction_tvt"])
        persistence_rmse = self._rmse(result["target_tvt"], result["last_known_TVT"])

        self.logger.info("--- Final OOF Evaluation ---")
        self.logger.info(f"OOF RMSE model:       {model_rmse:.5f}")
        self.logger.info(f"OOF RMSE persistence: {persistence_rmse:.5f}")
        self.logger.info(
            f"OOF improvement:       {persistence_rmse - model_rmse:.5f}"
        )

        well_metrics = (
            result.groupby("well_id", sort=False)
            .apply(
                lambda frame: pd.Series(
                    {
                        "rows": len(frame),
                        "rmse": self._rmse(
                            frame["target_tvt"],
                            frame["prediction_tvt"],
                        ),
                        "persistence_rmse": self._rmse(
                            frame["target_tvt"],
                            frame["last_known_TVT"],
                        ),
                        "mean_md_from_ps": float(frame["md_from_ps"].mean()),
                        "max_md_from_ps": float(frame["md_from_ps"].max()),
                        "hidden_rows": int(frame["hidden_rows"].iloc[0])
                        if "hidden_rows" in frame.columns
                        else len(frame),
                    }
                )
            )
            .reset_index()
        )

        well_metrics["improvement"] = (
            well_metrics["persistence_rmse"] - well_metrics["rmse"]
        )

        hidden_bins = pd.qcut(
            well_metrics["hidden_rows"],
            q=min(5, well_metrics["hidden_rows"].nunique()),
            duplicates="drop",
        )

        by_hidden_length = (
            well_metrics.assign(hidden_length_bin=hidden_bins)
            .groupby("hidden_length_bin", observed=True)
            .agg(
                wells=("well_id", "count"),
                median_rmse=("rmse", "median"),
                mean_rmse=("rmse", "mean"),
                mean_improvement=("improvement", "mean"),
            )
            .reset_index()
        )

        result["distance_bin"] = pd.cut(
            result["md_from_ps"],
            bins=[-np.inf, 25, 100, 250, 500, 1000, 2500, np.inf],
            labels=[
                "0-25",
                "25-100",
                "100-250",
                "250-500",
                "500-1000",
                "1000-2500",
                "2500+",
            ],
        )

        by_distance = (
            result.groupby("distance_bin", observed=True)
            .apply(
                lambda frame: pd.Series(
                    {
                        "rows": len(frame),
                        "rmse": self._rmse(
                            frame["target_tvt"],
                            frame["prediction_tvt"],
                        ),
                        "persistence_rmse": self._rmse(
                            frame["target_tvt"],
                            frame["last_known_TVT"],
                        ),
                    }
                )
            )
            .reset_index()
        )

        well_metrics.to_csv(
            self.artifacts_dir / "oof_well_metrics.csv",
            index=False,
        )
        by_hidden_length.to_csv(
            self.artifacts_dir / "oof_hidden_length_metrics.csv",
            index=False,
        )
        by_distance.to_csv(
            self.artifacts_dir / "oof_distance_metrics.csv",
            index=False,
        )

        self.logger.info("Saved OOF diagnostics:")
        self.logger.info(
            f"  {self.artifacts_dir / 'oof_well_metrics.csv'}"
        )
        self.logger.info(
            f"  {self.artifacts_dir / 'oof_hidden_length_metrics.csv'}"
        )
        self.logger.info(
            f"  {self.artifacts_dir / 'oof_distance_metrics.csv'}"
        )

        self.logger.info("Worst 10 wells by OOF RMSE:")
        for _, row in well_metrics.nlargest(10, "rmse").iterrows():
            self.logger.info(
                f"  {row['well_id']} | RMSE={row['rmse']:.4f} | "
                f"rows={int(row['rows'])} | hidden={int(row['hidden_rows'])}"
            )

        if fold_models:
            importance = np.zeros(len(feature_cols), dtype=np.float64)

            for model in fold_models:
                if hasattr(model, "booster") and model.booster is not None:
                    booster_importance = model.booster.feature_importance(
                        importance_type="gain"
                    )
                    # Ensure the importance array length matches feature_cols
                    if len(booster_importance) == len(feature_cols):
                        importance += booster_importance
                    else:
                        # Fallback: use feature names to align importance
                        booster_names = model.booster.feature_name()
                        for i, name in enumerate(booster_names):
                            if name in feature_cols:
                                idx = feature_cols.index(name)
                                importance[idx] += booster_importance[i]

            if len(fold_models) > 0:
                importance /= len(fold_models)

            importance_df = (
                pd.DataFrame(
                    {
                        "feature": feature_cols,
                        "gain": importance,
                    }
                )
                .sort_values("gain", ascending=False)
                .reset_index(drop=True)
            )

            importance_df.to_csv(
                self.artifacts_dir / "feature_importance.csv",
                index=False,
            )

            self.logger.info("Top 20 feature importance:")
            for _, row in importance_df.head(20).iterrows():
                self.logger.info(
                    f"  {row['feature']:<35} {row['gain']:.2f}"
                )
        else:
            importance_df = pd.DataFrame()

        return {
            "oof_rmse": model_rmse,
            "persistence_rmse": persistence_rmse,
            "well_metrics": well_metrics,
            "by_hidden_length": by_hidden_length,
            "by_distance": by_distance,
            "feature_importance": importance_df,
        }

    def evaluate_with_prefix_selection(
        self,
        train_df,
        oof_residuals,
        fold_models,
        feature_cols,
        min_gain=0.5,
        blend='hard'
    ):
        """
        Evaluate OOF predictions with visible-prefix candidate selection.
        
        This method applies the prefix selector as a post-processing step
        to improve predictions on the hidden evaluation zone.
        
        Parameters:
        -----------
        train_df : Training DataFrame
        oof_residuals : OOF residuals from ML model
        fold_models : Trained fold models
        feature_cols : Feature column names
        min_gain : Minimum RMSE improvement to prefer candidate
        blend : 'hard' (select one) or 'soft' (weighted blend)
        
        Returns:
        --------
        results : Dict with baseline and corrected metrics
        """
        self.logger.info("Evaluating OOF with visible-prefix candidate selection")
        
        # Get formations from config
        formations = self.config.get("physics", {}).get("formations", [
            'ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA'
        ])
        
        train_dir = Path(self.config["paths"]["data_dir"]) / "train"
        
        # First, get baseline ML predictions
        baseline_result = self.evaluate(train_df, oof_residuals, fold_models, feature_cols)
        
        # Apply prefix selection
        corrected_oof, well_log = apply_prefix_selection_to_oof(
            train_df=train_df,
            oof_predictions=oof_residuals,
            formations=formations,
            train_dir=train_dir,
            min_gain=min_gain
        )
        
        # Evaluate corrected OOF
        result = train_df.copy()
        result["oof_residual"] = np.asarray(corrected_oof, dtype=np.float64)
        result["prediction_tvt"] = result["last_known_TVT"] + result["oof_residual"]
        
        corrected_rmse = self._rmse(result["target_tvt"], result["prediction_tvt"])
        
        self.logger.info("--- Prefix Selection Results ---")
        self.logger.info(f"Baseline OOF RMSE:      {baseline_result['oof_rmse']:.5f}")
        self.logger.info(f"Corrected OOF RMSE:     {corrected_rmse:.5f}")
        self.logger.info(f"Improvement:            {baseline_result['oof_rmse'] - corrected_rmse:.5f}")
        
        # Analyze selection patterns
        n_wells = len(well_log)
        n_candidate = (well_log['selection'] == 'candidate').sum()
        n_ml = (well_log['selection'] == 'ml').sum()
        
        self.logger.info(f"Wells using candidate:  {n_candidate} ({100*n_candidate/n_wells:.1f}%)")
        self.logger.info(f"Wells using ML:         {n_ml} ({100*n_ml/n_wells:.1f}%)")
        
        # Most selected candidates
        candidate_counts = well_log[well_log['best_candidate'].notna()]['best_candidate'].value_counts()
        self.logger.info("Top candidates selected:")
        for cand, count in candidate_counts.head(10).items():
            self.logger.info(f"  {cand}: {count} wells")
        
        # Gain distribution
        valid_gains = well_log[well_log['gain'].notna()]['gain']
        if len(valid_gains) > 0:
            self.logger.info(f"Gain distribution: mean={valid_gains.mean():.3f}, "
                           f"std={valid_gains.std():.3f}, median={valid_gains.median():.3f}")
        
        # Save detailed log
        well_log.to_csv(
            self.artifacts_dir / "prefix_selection_well_log.csv",
            index=False
        )
        self.logger.info(f"Saved well log to {self.artifacts_dir / 'prefix_selection_well_log.csv'}")
        
        return {
            "baseline_oof_rmse": baseline_result['oof_rmse'],
            "corrected_oof_rmse": corrected_rmse,
            "improvement": baseline_result['oof_rmse'] - corrected_rmse,
            "well_log": well_log,
            "n_candidate_wells": n_candidate,
            "n_ml_wells": n_ml,
            "candidate_counts": candidate_counts,
            **baseline_result
        }
