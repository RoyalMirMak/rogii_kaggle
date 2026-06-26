# src/model.py
import lightgbm as lgb
from pathlib import Path
import sys

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

class LGBMWrapper:
    def __init__(self, config):
        self.params = config["model"]["params"]
        self.num_boost_round = config["model"]["num_boost_round"]
        self.early_stopping_rounds = config["model"]["early_stopping_rounds"]
        self.booster = None
        
        # Dry run settings
        self.dry_run = config.get("dry_run", {})
        self.dry_run_enabled = self.dry_run.get("enabled", False)
        
        # Validate GPU availability if device is set to "gpu"
        if self.params.get("device") == "gpu":
            self._validate_gpu()
    
    def _validate_gpu(self):
        """Validate that GPU is available and properly configured."""
        try:
            # Check if CUDA is available
            import subprocess
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError("nvidia-smi command failed. GPU drivers may not be installed.")
            
            # Try to create a GPU dataset to verify LightGBM GPU support (LightGBM 4.x uses gpu device type)
            lgb.Dataset([[1]], device_type="gpu")
            
        except subprocess.CalledProcessError:
            print("ERROR: GPU not detected. nvidia-smi failed.", file=sys.stderr)
            print("Please ensure NVIDIA drivers and CUDA are properly installed.", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: GPU validation failed: {e}", file=sys.stderr)
            print("LightGBM GPU support requires CUDA toolkit and compatible drivers.", file=sys.stderr)
            sys.exit(1)

    def fit(self, X_train, y_train, X_valid, y_valid, feature_names):
        """Trains the LightGBM booster with early stopping."""
        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        dvalid = lgb.Dataset(X_valid, label=y_valid, feature_name=feature_names, reference=dtrain)
        
        # Adjust evaluation period for dry run
        eval_period = 2 if self.dry_run_enabled else 200
        
        self.booster = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=eval_period)
            ],
        )
        return self

    def predict(self, X):
        """Generates predictions using the best iteration."""
        if self.booster is None:
            raise ValueError("Model has not been trained yet.")
        return self.booster.predict(X, num_iteration=self.booster.best_iteration)

    def save(self, filepath):
        """Saves the trained booster to disk."""
        if self.booster is not None:
            self.booster.save_model(filepath)

    def load(self, filepath):
        """Loads a trained booster from disk."""
        self.booster = lgb.Booster(model_file=filepath)
        return self


class CatBoostWrapper:
    def __init__(self, config):
        # We hardcode optimal parameters for this specific geology task
        self.params = {
            'iterations': 3000,
            'learning_rate': 0.03,
            'depth': 6,
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'random_seed': 42,
            'verbose': 200,
            'early_stopping_rounds': 150,
            'task_type': 'CPU'  # Change to GPU if your environment supports it
        }
        self.model = None

    def fit(self, X_train, y_train, X_valid, y_valid, feature_names):
        if CatBoostRegressor is None:
            raise ImportError(
                "catboost is not installed. Please run: pip install catboost"
            )
        self.model = CatBoostRegressor(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=(X_valid, y_valid)
        )
        return self

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filepath):
        self.model.save_model(filepath)

    def load(self, filepath):
        self.model = CatBoostRegressor()
        self.model.load_model(filepath)
        return self
