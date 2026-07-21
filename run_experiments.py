#!/usr/bin/env python3
"""Run controlled feature ablations and write one artifact directory per run."""
import argparse
import copy
import subprocess
import sys
from pathlib import Path
import yaml

EXPERIMENTS = {
    "baseline": {"pfz": False, "formations": False, "trajectory": False},
    "pfz": {"pfz": True, "formations": False, "trajectory": False},
    "formations_trajectory": {"pfz": False, "formations": True, "trajectory": True},
    "combined": {"pfz": True, "formations": True, "trajectory": True},
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="config.yaml")
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    base = yaml.safe_load(Path(args.base_config).read_text())
    out_dir = Path("experiment_configs")
    out_dir.mkdir(exist_ok=True)
    for name in args.experiments:
        cfg = copy.deepcopy(base)
        cfg["features"] = EXPERIMENTS[name]
        cfg["paths"]["artifacts_dir"] = f".artifacts_{name}"
        cfg["paths"]["submission_file"] = f"submission_{name}.csv"
        if args.dry_run:
            cfg["dry_run"] = {"enabled": True, "sample_train_wells": 10, "sample_test_wells": 2, "max_rows_per_well": 500}
        else:
            cfg["dry_run"] = {"enabled": False}
        path = out_dir / f"{name}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"\n=== {name} ===", flush=True)
        subprocess.run([sys.executable, "main.py", "--config", str(path)], check=True)

if __name__ == "__main__":
    main()
