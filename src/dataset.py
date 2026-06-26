# src/dataset.py
import time
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
import sys

from src.utils import setup_logger, robust_slope, first_nan_idx
from src.physics import FormationPlaneKNN, beam_search, self_corr_tvt

class DatasetBuilder:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger("DatasetBuilder")
        self.data_dir = Path(config["paths"]["data_dir"])
        self.train_dir = self.data_dir / "train"
        self.test_dir = self.data_dir / "test"
        
        self.n_jobs = config["training"]["n_jobs"]
        self.formations = config["physics"]["formations"]
        
        # Dry run settings
        self.dry_run = config.get("dry_run", {})
        self.dry_run_enabled = self.dry_run.get("enabled", False)
        self.sample_train_wells = self.dry_run.get("sample_train_wells", None)
        self.sample_test_wells = self.dry_run.get("sample_test_wells", None)
        self.max_rows_per_well = self.dry_run.get("max_rows_per_well", None)
        
        # Initialize KNN Imputer using all available training wells
        self.logger.info("Initializing FormationPlaneKNN...")
        train_wells = [p.name.split("__")[0] for p in sorted(self.train_dir.glob("*__horizontal_well.csv"))]
        
        # Sample wells for dry run
        if self.dry_run_enabled and self.sample_train_wells:
            train_wells = train_wells[:self.sample_train_wells]
            self.logger.info(f"[DRY RUN] Limited to {len(train_wells)} training wells")
        
        self.knn = FormationPlaneKNN(train_wells, self.data_dir, self.formations)

    def _build_features_for_well(self, wid, split, test_eval_idx=None):
        """Extracts statistical and physical features for a single well's evaluation zone."""
        split_dir = self.train_dir if split == "train" else self.test_dir
        hw_path = split_dir / f"{wid}__horizontal_well.csv"
        tw_path = split_dir / f"{wid}__typewell.csv"
        
        h = pd.read_csv(hw_path)
        h["row_index"] = np.arange(len(h), dtype=np.int64)
        
        if "TVT_input" not in h.columns:
            return pd.DataFrame()
            
        # Define the evaluation zone (where TVT_input is NaN)
        eval_mask = h["TVT_input"].isna().values
        if not eval_mask.any():
            return pd.DataFrame()
            
        if split == "train":
            if "TVT" not in h.columns:
                return pd.DataFrame()
            # In train, we must have the target available to compute residuals
            sel_mask = eval_mask & h["TVT"].notna().values
        else:
            sel_mask = eval_mask
            if test_eval_idx is not None:
                mask2 = np.zeros(len(h), dtype=bool)
                mask2[list(test_eval_idx)] = True
                sel_mask = sel_mask & mask2

        if not sel_mask.any():
            return pd.DataFrame()

        visible = h[h["TVT_input"].notna()].copy()
        if len(visible) == 10: # Ensure minimum history
            return pd.DataFrame()

        # Last known anchors
        last = visible.iloc[-1]
        last_TVT = float(last["TVT_input"])
        last_MD = float(last["MD"])
        last_X = float(last["X"])
        last_Y = float(last["Y"])
        last_Z = float(last["Z"])
        
        # Load typewell
        tw = pd.read_csv(tw_path) if tw_path.is_file() else pd.DataFrame()
        has_tw = "TVT" in tw.columns and "GR" in tw.columns and len(tw) > 3
        
        if has_tw:
            tw_tvt = tw["TVT"].to_numpy(dtype=np.float32)
            tw_gr = tw["GR"].to_numpy(dtype=np.float32)
        else:
            tw_tvt, tw_gr = np.array([]), np.array([])

        sel_idx = np.flatnonzero(sel_mask)
        cur = h.iloc[sel_idx].copy()
        
        # Limit rows per well for dry run BEFORE any assignments
        if self.dry_run_enabled and self.max_rows_per_well and len(cur) > self.max_rows_per_well:
            sel_idx = sel_idx[:self.max_rows_per_well]
            cur = cur.iloc[:self.max_rows_per_well]
            self.logger.debug(f"[DRY RUN] Limited {wid} to {self.max_rows_per_well} rows")
        
        cur["well_id"] = wid
        cur["id"] = cur["well_id"] + "_" + cur["row_index"].astype(str)

        # --- 1. Basic Statistical Features ---
        vis_TVT = visible["TVT_input"].values
        cur["last_known_TVT"] = last_TVT
        cur["md_from_ps"] = cur["MD"].values - last_MD
        cur["z_from_ps"] = cur["Z"].values - last_Z
        cur["dxy_from_ps"] = np.sqrt((cur["X"].values - last_X)**2 + (cur["Y"].values - last_Y)**2)
        cur["slope_TVT_MD_all"] = robust_slope(visible["MD"].values, vis_TVT)
        
        # --- 2. Physics-Informed Features ---
        if has_tw:
            hgr = cur["GR"].interpolate(limit_direction="both").fillna(np.nanmean(tw_gr)).to_numpy(dtype=np.float32)
            kgr = visible["GR"].interpolate(limit_direction="both").fillna(np.nanmean(tw_gr)).to_numpy(dtype=np.float32)
            
            # Beam Search
            beams_res = []
            for bs, mc, es, r, tag in self.config["physics"]["beams"]:
                path = beam_search(hgr, tw_tvt, tw_gr, last_TVT, bs, mc, es, r)
                if len(path) == len(cur):
                    cur[f"beam_{tag}_d"] = path - last_TVT
                    beams_res.append(path)
                else:
                    cur[f"beam_{tag}_d"] = 0.0
            
            if beams_res:
                beams_arr = np.stack(beams_res, axis=1)
                cur["beam_mean_d"] = beams_arr.mean(axis=1) - last_TVT
                cur["beam_std_d"] = beams_arr.std(axis=1)
            else:
                cur["beam_mean_d"] = 0.0
                cur["beam_std_d"] = 0.0
                
            # Self-Correlation NCC
            sc_path, sc_score = self_corr_tvt(kgr, vis_TVT, hgr, hw=15, stride=3)
            if len(sc_path) == len(cur):
                cur["sc_tvt_d"] = sc_path - last_TVT
                cur["sc_score"] = sc_score
            else:
                cur["sc_tvt_d"] = 0.0
                cur["sc_score"] = 0.0
        else:
            cur["beam_mean_d"] = 0.0
            cur["beam_std_d"] = 0.0
            cur["sc_tvt_d"] = 0.0
            cur["sc_score"] = 0.0

        # Spatial Formation Plane Imputation
        xy_kn = visible[["X", "Y"]].to_numpy()
        xy_ev = cur[["X", "Y"]].to_numpy()
        form_kn, _ = self.knn.impute(xy_kn, self_wid=wid if split == "train" else None, k=self.config["physics"]["plane_knn_k"])
        form_ev, knn_dist = self.knn.impute(xy_ev, self_wid=wid if split == "train" else None, k=self.config["physics"]["plane_knn_k"])
        z_kn = visible["Z"].to_numpy()
        z_ev = cur["Z"].to_numpy()

        cur["spatial_knn_dist"] = knn_dist
        import warnings
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Spatial Formation Plane Imputation
            for fi, fn in enumerate(self.formations):
                # Calculate physical offset from known data using median
                b_full = float(np.nanmedian(vis_TVT + z_kn - form_kn[:, fi]))
                if np.isnan(b_full): b_full = 0.0
                # Predict hidden TVT using plane geometry
                cur[f"tvt_knn_{fn}_d"] = (-z_ev + form_ev[:, fi] + b_full) - last_TVT

        # --- 3. Target Variable Setup ---
        if split == "train":
            cur["target_tvt"] = h["TVT"].values[sel_idx]
            cur["target_residual"] = cur["target_tvt"] - last_TVT

        return cur.reset_index(drop=True)

    def build_train(self):
        self.logger.info("Building training feature matrix...")
        all_train_wells = [p.name.split("__")[0] for p in sorted(self.train_dir.glob("*__horizontal_well.csv"))]
        
        # Apply dry run sampling
        train_wells = all_train_wells
        if self.dry_run_enabled and self.sample_train_wells and len(all_train_wells) > self.sample_train_wells:
            train_wells = all_train_wells[:self.sample_train_wells]
            self.logger.info(f"[DRY RUN] Limited to {len(train_wells)} training wells")
        
        self.logger.info(f"Found {len(all_train_wells)} total training wells, processing {len(train_wells)}.")
        
        t0 = time.time()
        
        # Always use tqdm for progress tracking
        dfs = []
        total = len(train_wells)
        
        if self.n_jobs == 1:
            # Sequential processing with tqdm
            for wid in tqdm(train_wells, desc="Processing wells", total=total):
                self.logger.debug(f"Processing well: {wid}")
                result = self._build_features_for_well(wid, "train")
                if len(result) > 0:
                    dfs.append(result)
        else:
            # Parallel processing with multiprocessing backend (more stable than loky for this workload)
            # Process in small chunks to show progress
            chunk_size = max(1, total // 20)  # Show ~20 progress updates
            self.logger.info(f"Starting parallel processing: {total} wells in chunks of {chunk_size} with {self.n_jobs} workers")
            self.logger.info(f"First 5 wells: {train_wells[:5]}")
            sys.stdout.flush()
            
            for chunk_start in range(0, total, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total)
                chunk_wells = train_wells[chunk_start:chunk_end]
                
                self.logger.debug(f"Processing chunk {chunk_start//chunk_size + 1}: wells {chunk_start}-{chunk_end-1}")
                sys.stdout.flush()
                
                # Process chunk in parallel (uses default 'loky' backend which is safer for heavy memory operations)
                chunk_dfs = Parallel(
                    n_jobs=self.n_jobs, 
                    verbose=10
                )(
                    delayed(self._build_features_for_well)(wid, "train") for wid in chunk_wells
                )
                dfs.extend([d for d in chunk_dfs if len(d) > 0])
                
                # Log progress
                elapsed_chunk = time.time() - t0
                processed = chunk_end
                progress = 100.0 * processed / total
                wells_per_sec = processed / max(elapsed_chunk, 0.1)
                self.logger.info(f"✓ Progress: {processed}/{total} wells ({progress:.1f}%) - {wells_per_sec:.1f} wells/sec")
                sys.stdout.flush()
        
        train_df = pd.concat([d for d in dfs if len(d) > 0], ignore_index=True)
        elapsed = time.time() - t0
        self.logger.info(f"Train matrix shape: {train_df.shape} (Built in {elapsed:.1f}s, ~{len(train_wells)/max(elapsed, 0.1):.1f} wells/sec)")
        
        # Automatically extract valid features
        exclude = {"well_id", "id", "row_index", "target_tvt", "target_residual", "TVT", "TVT_input"}
        exclude.update(self.formations)
        feature_cols = [c for c in train_df.columns if c not in exclude]
        
        self.logger.info(f"Generated {len(feature_cols)} features.")
        return train_df, feature_cols

    def build_test(self):
        self.logger.info("Building test feature matrix...")
        sample_sub = pd.read_csv(self.data_dir / self.config["paths"]["sample_submission_file"])
        sample_sub["well_id"] = sample_sub["id"].str.rsplit("_", n=1).str[0]
        sample_sub["row_index"] = sample_sub["id"].str.rsplit("_", n=1).str[1].astype(int)
        
        test_eval_index = (
            sample_sub.groupby("well_id")["row_index"]
            .apply(lambda s: set(s.tolist()))
            .to_dict()
        )
        test_wells = list(test_eval_index.keys())
        
        # Sample wells for dry run
        if self.dry_run_enabled and self.sample_test_wells and len(test_wells) > self.sample_test_wells:
            test_wells = test_wells[:self.sample_test_wells]
            test_eval_index = {k: test_eval_index[k] for k in test_wells}
            self.logger.info(f"[DRY RUN] Limited to {len(test_wells)} test wells")
        
        self.logger.info(f"Found {len(test_wells)} test wells to process.")

        t0 = time.time()
        
        # Always use tqdm for progress tracking
        dfs = []
        total = len(test_wells)
        
        if self.n_jobs == 1:
            # Sequential processing with tqdm
            for wid in tqdm(test_wells, desc="Processing test wells", total=total):
                self.logger.debug(f"Processing test well: {wid}")
                result = self._build_features_for_well(wid, "test", test_eval_index.get(wid))
                if len(result) > 0:
                    dfs.append(result)
        else:
            # Parallel processing with multiprocessing backend (more stable than loky)
            chunk_size = max(1, total // 20)  # Show ~20 progress updates
            self.logger.info(f"Starting parallel processing: {total} test wells in chunks of {chunk_size} with {self.n_jobs} workers")
            self.logger.info(f"First 5 test wells: {test_wells[:5]}")
            sys.stdout.flush()
            
            for chunk_start in range(0, total, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total)
                chunk_wells = test_wells[chunk_start:chunk_end]
                
                self.logger.debug(f"Processing test chunk {chunk_start//chunk_size + 1}: wells {chunk_start}-{chunk_end-1}")
                sys.stdout.flush()
                
                # Process chunk in parallel (uses default 'loky' backend which is safer for heavy memory operations)
                chunk_dfs = Parallel(
                    n_jobs=self.n_jobs, 
                    verbose=10
                )(
                    delayed(self._build_features_for_well)(wid, "test", test_eval_index.get(wid)) for wid in chunk_wells
                )
                dfs.extend([d for d in chunk_dfs if len(d) > 0])
                
                # Log progress
                elapsed_chunk = time.time() - t0
                processed = chunk_end
                progress = 100.0 * processed / total
                wells_per_sec = processed / max(elapsed_chunk, 0.1)
                self.logger.info(f"✓ Progress: {processed}/{total} test wells ({progress:.1f}%) - {wells_per_sec:.1f} wells/sec")
                sys.stdout.flush()
        
        test_df = pd.concat([d for d in dfs if len(d) > 0], ignore_index=True) if dfs else pd.DataFrame()
        elapsed = time.time() - t0
        self.logger.info(f"Test matrix shape: {test_df.shape} (Built in {elapsed:.1f}s, ~{len(test_wells)/max(elapsed, 0.1):.1f} wells/sec)")
        return test_df, sample_sub
