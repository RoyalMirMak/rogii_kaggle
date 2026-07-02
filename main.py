# main.py
import os
import sys
import yaml
import warnings
import argparse
from pathlib import Path

# Local module imports
from src.data_loader import download_competition_data
from src.dataset import DatasetBuilder
from src.train import Trainer
from src.validate import Validator
from src.utils import setup_logger

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="ROGII Hybrid Baseline Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    return parser.parse_args()

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Auto-detect Kaggle environment
    kaggle_path = Path("/kaggle/input/rogii-wellbore-geology-prediction")
    if kaggle_path.exists():
        config["paths"]["data_dir"] = str(kaggle_path)
        config["paths"]["artifacts_dir"] = "/kaggle/working/artifacts"
    
    # Ensure artifacts directory exists
    os.makedirs(config["paths"]["artifacts_dir"], exist_ok=True)
    return config

def main():
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logger("main")
    logger.info(f"Using config file: {args.config}")
    
    # Check for dry run mode
    dry_run_enabled = config.get("dry_run", {}).get("enabled", False)
    if dry_run_enabled:
        logger.info("=== DRY RUN MODE === Pipeline validation with minimal data")
    
    logger.info("Starting ROGII Hybrid Baseline Pipeline")
    
    # 0. Download data if not present (skip on Kaggle)
    kaggle_path = Path("/kaggle/input/rogii-wellbore-geology-prediction")
    if not kaggle_path.exists():
        data_dir = download_competition_data(
            config["paths"]["data_dir"],
            config["paths"].get("competition_name", "rogii-wellbore-geology-prediction")
        )
        config["paths"]["data_dir"] = data_dir
    
    logger.info(f"Using Data Directory: {config['paths']['data_dir']}")

    # 1. Build Datasets (with parallel processing)
    # Note: Only build training data - test data is corrupted and should not be used
    logger.info("--- Step 1: Building Training Datasets ---")
    builder = DatasetBuilder(config)
    train_df, feature_cols = builder.build_train()

    # 2. Train Models (GroupKFold)
    logger.info("--- Step 2: Training Models ---")
    trainer = Trainer(config, feature_cols)
    fold_models, oof_predictions, train_df_used = trainer.train_cv(train_df)

    # 3. Validate (Calculate OOF RMSE on actual TVT)
    logger.info("--- Step 3: Validation ---")
    validator = Validator(config)
    validator.evaluate(train_df_used, oof_predictions, fold_models, feature_cols)

    # Note: Inference step removed - test data is corrupted and should not be used
    
    logger.info("Pipeline completed successfully (training and validation only).")

if __name__ == "__main__":
    main()
