# src/dataset.py
import time
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
import sys

from src.utils import setup_logger, robust_slope, first_nan_idx
from src.physics import (
    FormationPlaneKNN,
    DenseANCCImputer,
    beam_search,
    self_corr_tvt,
    multi_scale_ncc,
    run_pf_ancc,
    run_pf_z_velocity,
    seg_b_well,
    affine_cal,
)

def _recent_slope(x, y, window, fallback=0.0):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(x) < 3 or len(y) < 3:
        return float(fallback)

    x = x[-window:]
    y = y[-window:]
    valid = np.isfinite(x) & np.isfinite(y)

    if valid.sum() < 3:
        return float(fallback)

    return robust_slope(x[valid], y[valid])


def _nan_run_lengths(values):
    is_nan = np.isnan(np.asarray(values, dtype=np.float64))
    result = np.zeros(len(is_nan), dtype=np.float32)
    run = 0

    for i, flag in enumerate(is_nan):
        if flag:
            run += 1
        else:
            run = 0
        result[i] = run

    return result


def _add_gr_sequence_features(h, visible, sel_idx, fallback_gr):
    """
    Features use the complete horizontal GR trajectory. This is valid because
    GR is observed in the hidden TVT interval at inference time.
    
    Returns a dict of all GR features to be added via pd.concat() for performance.
    """
    raw_gr = h["GR"].to_numpy(dtype=np.float64)
    nan_flag = np.isnan(raw_gr).astype(np.float32)
    nan_streak = _nan_run_lengths(raw_gr)

    filled_gr = (
        pd.Series(raw_gr)
        .interpolate(limit_direction="both")
        .fillna(float(fallback_gr))
        .to_numpy(dtype=np.float64)
    )
    gr_series = pd.Series(filled_gr)

    gr_cols = {}
    gr_cols["gr"] = filled_gr[sel_idx].astype(np.float32)
    gr_cols["gr_nan_flag"] = nan_flag[sel_idx]
    gr_cols["gr_nan_streak"] = nan_streak[sel_idx]

    for window in (5, 21, 51, 101, 151):
        gr_cols[f"gr_roll_mean_{window}"] = (
            gr_series.rolling(window, center=True, min_periods=1).mean().to_numpy()[sel_idx]
        ).astype(np.float32)

    for window in (5, 21):
        rolling = gr_series.rolling(window, center=True, min_periods=1)
        gr_cols[f"gr_roll_std_{window}"] = rolling.std().fillna(0.0).to_numpy()[sel_idx].astype(np.float32)
        min_values = rolling.min().to_numpy()
        max_values = rolling.max().to_numpy()
        gr_cols[f"gr_roll_min_{window}"] = min_values[sel_idx].astype(np.float32)
        gr_cols[f"gr_roll_max_{window}"] = max_values[sel_idx].astype(np.float32)
        gr_cols[f"gr_roll_range_{window}"] = (max_values - min_values)[sel_idx].astype(np.float32)

    gradient_1 = gr_series.diff().fillna(0.0).to_numpy(dtype=np.float64)
    gradient_2 = pd.Series(gradient_1).diff().fillna(0.0).to_numpy(dtype=np.float64)
    gr_cols["gr_grad_1"] = gradient_1[sel_idx].astype(np.float32)
    gr_cols["gr_grad_2"] = gradient_2[sel_idx].astype(np.float32)

    for lag in (1, 5, 15, 30):
        gr_cols[f"gr_lag_{lag}"] = gr_series.shift(lag).bfill().to_numpy()[sel_idx].astype(np.float32)
        gr_cols[f"gr_lead_{lag}"] = gr_series.shift(-lag).ffill().to_numpy()[sel_idx].astype(np.float32)

    gr_cols["gr_cumsum"] = gr_series.cumsum().to_numpy()[sel_idx].astype(np.float32)

    visible_gr = (
        visible["GR"]
        .astype(float)
        .interpolate(limit_direction="both")
        .fillna(float(fallback_gr))
        .to_numpy(dtype=np.float64)
    )
    eval_gr = filled_gr[sel_idx]

    prefix_mean = float(np.nanmean(visible_gr))
    prefix_std = float(np.nanstd(visible_gr))
    gr_cols["prefix_gr_mean"] = prefix_mean
    gr_cols["prefix_gr_std"] = prefix_std
    gr_cols["prefix_gr_last_5"] = float(np.nanmean(visible_gr[-5:]))
    gr_cols["prefix_gr_last_20"] = float(np.nanmean(visible_gr[-20:]))

    gr_cols["eval_gr_mean"] = float(np.nanmean(eval_gr))
    gr_cols["eval_gr_std"] = float(np.nanstd(eval_gr))
    gr_cols["eval_gr_p25"] = float(np.nanquantile(eval_gr, 0.25))
    gr_cols["eval_gr_p50"] = float(np.nanquantile(eval_gr, 0.50))
    gr_cols["eval_gr_p75"] = float(np.nanquantile(eval_gr, 0.75))
    gr_cols["eval_gr_p90"] = float(np.nanquantile(eval_gr, 0.90))
    gr_cols["eval_gr_vs_prefix"] = float(np.nanmean(eval_gr) - prefix_mean)
    
    return gr_cols

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
        self.formation_knn = self.knn
        self.dense_knn = DenseANCCImputer(train_wells, self.data_dir)

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
        if len(visible) < 10: # Ensure minimum history
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
        
        # --- 1. Basic Statistical Features - batch via dict+concat
        meta_cols = {
            "well_id": wid,
            "id": wid + "_" + cur["row_index"].astype(str).values,
        }
        meta_df = pd.DataFrame(meta_cols, index=cur.index)
        cur = pd.concat([cur, meta_df], axis=1)
        
        # --- 1b. Basic Statistical Features - batch via dict+concat
        vis_TVT = visible["TVT_input"].values
        slope_all = robust_slope(visible["MD"].values, vis_TVT)
        basic_cols = {
            "last_known_TVT": last_TVT,
            "md_from_ps": (cur["MD"].values - last_MD).astype(np.float32),
            "z_from_ps": (cur["Z"].values - last_Z).astype(np.float32),
            "dxy_from_ps": np.sqrt((cur["X"].values - last_X)**2 + (cur["Y"].values - last_Y)**2).astype(np.float32),
            "slope_TVT_MD_all": slope_all,
            "slope_TVT_MD_5": _recent_slope(visible["MD"].values, vis_TVT, 5, slope_all),
            "slope_TVT_MD_10": _recent_slope(visible["MD"].values, vis_TVT, 10, slope_all),
            "slope_TVT_MD_20": _recent_slope(visible["MD"].values, vis_TVT, 20, slope_all),
            "slope_TVT_MD_50": _recent_slope(visible["MD"].values, vis_TVT, 50, slope_all),
            "slope_Z_MD_10": _recent_slope(visible["MD"].values, visible["Z"].values, 10, 0.0),
            "slope_Z_MD_20": _recent_slope(visible["MD"].values, visible["Z"].values, 20, 0.0),
            "hidden_rows": len(sel_idx),
            "visible_rows": len(visible),
            "hidden_fraction": len(sel_idx) / max(len(h), 1),
            "z_span_visible": float(np.nanmax(visible["Z"].values) - np.nanmin(visible["Z"].values)),
        }
        basic_df = pd.DataFrame(basic_cols, index=cur.index)
        cur = pd.concat([cur, basic_df], axis=1)
        
        # --- 2. Physics-Informed Features ---
        if has_tw:
            hgr = cur["GR"].interpolate(limit_direction="both").fillna(np.nanmean(tw_gr)).to_numpy(dtype=np.float32)
            kgr = visible["GR"].interpolate(limit_direction="both").fillna(np.nanmean(tw_gr)).to_numpy(dtype=np.float32)
            gr_cols = _add_gr_sequence_features(
                h=h,
                visible=visible,
                sel_idx=sel_idx,
                fallback_gr=float(np.nanmean(tw_gr)),
            )
            gr_df = pd.DataFrame(gr_cols, index=cur.index)
            cur = pd.concat([cur, gr_df], axis=1)
            
            # Beam Search - batch via dict+concat
            beams_res = []
            beam_cols = {}
            for bs, mc, es, r, tag in self.config["physics"]["beams"]:
                path = beam_search(hgr, tw_tvt, tw_gr, last_TVT, bs, mc, es, r)
                if len(path) == len(cur):
                    beam_cols[f"beam_{tag}_d"] = (path - last_TVT).astype(np.float32)
                    beams_res.append(path)
                else:
                    beam_cols[f"beam_{tag}_d"] = 0.0
            
            if beams_res:
                beams_arr = np.stack(beams_res, axis=1)
                beam_cols["beam_mean_d"] = (beams_arr.mean(axis=1) - last_TVT).astype(np.float32)
                beam_cols["beam_median_d"] = (np.median(beams_arr, axis=1) - last_TVT).astype(np.float32)
                beam_cols["beam_std_d"] = beams_arr.std(axis=1).astype(np.float32)

                beam_by_tag = {
                    tag: beam_cols[f"beam_{tag}_d"]
                    for _, _, _, _, tag in self.config["physics"]["beams"]
                    if f"beam_{tag}_d" in beam_cols
                }

                if "vloose" in beam_by_tag and "vcons" in beam_by_tag:
                    beam_cols["beam_spread_d"] = (beam_by_tag["vloose"] - beam_by_tag["vcons"]).astype(np.float32)
                else:
                    beam_cols["beam_spread_d"] = 0.0

                if "loose" in beam_by_tag and "cons" in beam_by_tag:
                    beam_cols["beam_gap_d"] = (beam_by_tag["loose"] - beam_by_tag["cons"]).astype(np.float32)
                else:
                    beam_cols["beam_gap_d"] = 0.0
            else:
                beam_cols["beam_median_d"] = 0.0
                beam_cols["beam_spread_d"] = 0.0
                beam_cols["beam_gap_d"] = 0.0
                beam_cols["beam_mean_d"] = 0.0
                beam_cols["beam_std_d"] = 0.0
            
            beam_df = pd.DataFrame(beam_cols, index=cur.index)
            cur = pd.concat([cur, beam_df], axis=1)
                
            # --- Multi-Scale NCC - batch via dict+concat
            sc_res, sc_ens = multi_scale_ncc(kgr, visible["TVT_input"].values, hgr, hws=(8, 15, 25), stride=3)
            sc_cols = {
                "sc8_d": sc_res[0][0] - last_TVT,
                "sc15_d": sc_res[1][0] - last_TVT,
                "sc25_d": sc_res[2][0] - last_TVT,
                "sc_ens_d": sc_ens - last_TVT,
            }
            sc_df = pd.DataFrame(sc_cols, index=cur.index)
            cur = pd.concat([cur, sc_df], axis=1)
            
            # --- Particle Filter (CRITICAL PHYSICS LAYER) - batch via dict+concat
            pf_a_pts, pf_a_std = run_pf_ancc(h, tw_tvt, tw_gr)
            pf_z_pts, pf_z_std = run_pf_z_velocity(h, tw_tvt, tw_gr)
            
            pf_cols = {}
            if len(pf_a_pts) == len(cur):
                pf_cols["pf_ancc_d"] = (pf_a_pts - last_TVT).astype(np.float32)
                pf_cols["pf_ancc_std"] = pf_a_std.astype(np.float32)
            else:
                pf_cols["pf_ancc_d"] = 0.0
                pf_cols["pf_ancc_std"] = 0.0
            
            if len(pf_z_pts) == len(cur):
                pf_cols["pf_z_d"] = (pf_z_pts - last_TVT).astype(np.float32)
                pf_cols["pf_z_std"] = pf_z_std.astype(np.float32)
            else:
                pf_cols["pf_z_d"] = 0.0
                pf_cols["pf_z_std"] = 0.0
            
            pf_df = pd.DataFrame(pf_cols, index=cur.index)
            cur = pd.concat([cur, pf_df], axis=1)
        else:
            gr_cols = _add_gr_sequence_features(
                h=h,
                visible=visible,
                sel_idx=sel_idx,
                fallback_gr=50.0,
            )
            gr_df = pd.DataFrame(gr_cols, index=cur.index)
            cur = pd.concat([cur, gr_df], axis=1)
            
            # Batch zero-fill physics columns for no-typewell case
            physics_zero_cols = {
                "beam_mean_d": 0.0,
                "beam_std_d": 0.0,
                "sc8_d": 0.0,
                "sc15_d": 0.0,
                "sc25_d": 0.0,
                "sc_ens_d": 0.0,
                "pf_ancc_d": 0.0,
                "pf_ancc_std": 0.0,
                "pf_z_d": 0.0,
                "pf_z_std": 0.0,
            }
            physics_zero_df = pd.DataFrame(physics_zero_cols, index=cur.index)
            cur = pd.concat([cur, physics_zero_df], axis=1)

        # --- Dense ANCC Imputation with Segmented b_well - batch via dict+concat
        xy_kn = visible[["X", "Y"]].to_numpy()
        xy_ev = cur[["X", "Y"]].to_numpy()
        d_ancc = self.dense_knn.impute(xy_ev, self_wid=wid if split == "train" else None)
        d_kn = self.dense_knn.impute(xy_kn, self_wid=wid if split == "train" else None)
        
        ktvt = visible["TVT_input"].to_numpy(dtype=np.float32)
        kz = visible["Z"].to_numpy(dtype=np.float32)
        z_ev = cur["Z"].to_numpy(dtype=np.float32)

        # Calculate early, mid, late, and exponentially weighted offsets
        b_full, b_early, b_mid, b_late, b_wls = seg_b_well(ktvt, kz, d_kn)
        
        dense_cols = {
            "tvt_dense_d": (-z_ev + d_ancc + b_full - last_TVT).astype(np.float32),
            "tvt_densew_d": (-z_ev + d_ancc + b_wls - last_TVT).astype(np.float32),
            "tvt_dense50_d": (-z_ev + d_ancc + b_late - last_TVT).astype(np.float32),
        }
        
        # Cross-signal features
        if has_tw and "pf_ancc_d" in cur:
            dense_cols["pf_vs_dense"] = (cur["pf_ancc_d"].values - dense_cols["tvt_dense_d"]).astype(np.float32)
            dense_cols["sc_vs_dense"] = (cur["sc_ens_d"].values - dense_cols["tvt_dense_d"]).astype(np.float32)
            dense_cols["pf_vs_beam"] = (cur["pf_ancc_d"].values - cur["beam_mean_d"].values).astype(np.float32)
        else:
            dense_cols["pf_vs_dense"] = 0.0
            dense_cols["sc_vs_dense"] = 0.0
            dense_cols["pf_vs_beam"] = 0.0

        if has_tw:
            dense_cols["pf_z_vs_ancc"] = (cur["pf_z_d"].values - cur["pf_ancc_d"].values).astype(np.float32)
            dense_cols["pf_z_vs_dense"] = (cur["pf_z_d"].values - dense_cols["tvt_dense_d"]).astype(np.float32)
            dense_cols["pf_z_vs_beam"] = (cur["pf_z_d"].values - cur["beam_mean_d"].values).astype(np.float32)
        else:
            dense_cols["pf_z_vs_ancc"] = 0.0
            dense_cols["pf_z_vs_dense"] = 0.0
            dense_cols["pf_z_vs_beam"] = 0.0
        
        dense_df = pd.DataFrame(dense_cols, index=cur.index)
        cur = pd.concat([cur, dense_df], axis=1)
        
        # --- GR Offset Matrices and Affine Calibration ---

        form_ev, form_ev_dist = self.formation_knn.impute(
            xy_ev,
            self_wid=wid if split == "train" else None,
            k=self.config["physics"]["plane_knn_k"],
        )
        form_kn, _ = self.formation_knn.impute(
            xy_kn,
            self_wid=wid if split == "train" else None,
            k=self.config["physics"]["plane_knn_k"],
        )

        # Formation features - batch via dict to avoid fragmentation (including knn_distance)
        form_cols = {"formation_knn_distance": form_ev_dist.astype(np.float32)}
        for form_index, form_name in enumerate(self.formations):
            ev_form = form_ev[:, form_index]
            kn_form = form_kn[:, form_index]

            valid = np.isfinite(kn_form) & np.isfinite(ktvt) & np.isfinite(kz)
            if valid.sum() >= 10:
                b_full_form = float(np.nanmedian(ktvt[valid] + kz[valid] - kn_form[valid]))

                recent_mask = valid.copy()
                recent_indices = np.flatnonzero(valid)[-min(50, valid.sum()):]
                b_recent_form = float(
                    np.nanmedian(ktvt[recent_indices] + kz[recent_indices] - kn_form[recent_indices])
                )

                form_cols[f"tvt_{form_name}_d"] = (-z_ev + ev_form + b_full_form) - last_TVT
                form_cols[f"tvt_{form_name}_recent_d"] = (
                    -z_ev + ev_form + b_recent_form
                ) - last_TVT
                form_cols[f"b_{form_name}"] = b_full_form
                form_cols[f"b_recent_{form_name}"] = b_recent_form
            else:
                form_cols[f"tvt_{form_name}_d"] = 0.0
                form_cols[f"tvt_{form_name}_recent_d"] = 0.0
                form_cols[f"b_{form_name}"] = 0.0
                form_cols[f"b_recent_{form_name}"] = 0.0

        form_df = pd.DataFrame(form_cols, index=cur.index)
        cur = pd.concat([cur, form_df], axis=1)
        
        # --- Affine Calibration and Offset Matrices - batch via dict+concat ---
        cal_offset_cols = {}
        if has_tw:
            # 1. Affine Calibration (Scale and Shift of GR)
            tw_at_k = np.interp(ktvt, tw_tvt, tw_gr).astype(np.float32)
            a_cal, b_cal = affine_cal(kgr, tw_at_k)
            cal_offset_cols["cal_a"] = a_cal
            cal_offset_cols["cal_b"] = b_cal
            
            # 2. GR Offset Matrices - batch via dict+concat to avoid DataFrame fragmentation
            ANCH_OFFS = [-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80]
            BEAM_OFFS = [-40, -20, -10, -5, -3, 0, 3, 5, 10, 20, 40]
            PF_OFFS   = [-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30]

            # Anchor Offsets (around last known TVT)
            for o in ANCH_OFFS:
                cal_offset_cols[f"tda_{o}"] = (hgr - np.interp(last_TVT + o, tw_tvt, tw_gr)).astype(np.float32)

            # Beam Offsets (around Beam Search Mean)
            beam_ref = cur["beam_mean_d"].values + last_TVT
            for o in BEAM_OFFS:
                cal_offset_cols[f"tdbc_{o}"] = (hgr - np.interp(beam_ref + o, tw_tvt, tw_gr)).astype(np.float32)

            # PF Offsets (around Particle Filter)
            if "pf_ancc_d" in cur and (cur["pf_ancc_d"].values != 0).any():
                pf_ref = cur["pf_ancc_d"].values + last_TVT
                for o in PF_OFFS:
                    cal_offset_cols[f"tdpf_{o}"] = (hgr - np.interp(pf_ref + o, tw_tvt, tw_gr)).astype(np.float32)
            else:
                for o in PF_OFFS:
                    cal_offset_cols[f"tdpf_{o}"] = 0.0
        else:
            cal_offset_cols["cal_a"] = 1.0
            cal_offset_cols["cal_b"] = 0.0
            # Batch zero-fill offset columns
            for o in [-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80]:
                cal_offset_cols[f"tda_{o}"] = 0.0
            for o in [-40, -20, -10, -5, -3, 0, 3, 5, 10, 20, 40]:
                cal_offset_cols[f"tdbc_{o}"] = 0.0
            for o in [-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30]:
                cal_offset_cols[f"tdpf_{o}"] = 0.0
        
        cal_offset_df = pd.DataFrame(cal_offset_cols, index=cur.index)
        cur = pd.concat([cur, cal_offset_df], axis=1)

        # --- 3. Target Variable Setup - batch via dict+concat ---
        target_cols = {}
        if split == "train":
            target_cols["target_tvt"] = h["TVT"].values[sel_idx].astype(np.float32)
            target_cols["target_residual"] = (target_cols["target_tvt"] - last_TVT).astype(np.float32)
        
        if target_cols:
            target_df = pd.DataFrame(target_cols, index=cur.index)
            cur = pd.concat([cur, target_df], axis=1)

        # Defragment DataFrame before return (critical for performance)
        cur = cur.copy()
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
            # Parallel processing - process ALL wells at once for maximum efficiency
            self.logger.info(f"Starting parallel processing: {total} wells with {self.n_jobs} workers")
            self.logger.info(f"First 5 wells: {train_wells[:5]}")
            sys.stdout.flush()
            
            # Process all wells in parallel (loky backend for memory safety)
            dfs = Parallel(
                n_jobs=self.n_jobs, 
                verbose=10
            )(
                delayed(self._build_features_for_well)(wid, "train") for wid in train_wells
            )
            dfs = [d for d in dfs if len(d) > 0]
            
            elapsed = time.time() - t0
            self.logger.info(f"✓ Completed {len(dfs)}/{total} wells in {elapsed:.1f}s ({len(dfs)/max(elapsed, 0.1):.1f} wells/sec)")
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
            # Parallel processing - process ALL wells at once for maximum efficiency
            self.logger.info(f"Starting parallel processing: {total} test wells with {self.n_jobs} workers")
            self.logger.info(f"First 5 test wells: {test_wells[:5]}")
            sys.stdout.flush()
            
            # Process all wells in parallel (loky backend for memory safety)
            dfs = Parallel(
                n_jobs=self.n_jobs, 
                verbose=10
            )(
                delayed(self._build_features_for_well)(wid, "test", test_eval_index.get(wid)) for wid in test_wells
            )
            dfs = [d for d in dfs if len(d) > 0]
            
            elapsed = time.time() - t0
            self.logger.info(f"✓ Completed {len(dfs)}/{total} test wells in {elapsed:.1f}s ({len(dfs)/max(elapsed, 0.1):.1f} wells/sec)")
            sys.stdout.flush()
        
        test_df = pd.concat([d for d in dfs if len(d) > 0], ignore_index=True) if dfs else pd.DataFrame()
        elapsed = time.time() - t0
        self.logger.info(f"Test matrix shape: {test_df.shape} (Built in {elapsed:.1f}s, ~{len(test_wells)/max(elapsed, 0.1):.1f} wells/sec)")
        return test_df, sample_sub
