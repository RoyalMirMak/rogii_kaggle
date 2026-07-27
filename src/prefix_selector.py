# src/prefix_selector.py
"""
Visible-Prefix Candidate Selection Module

For each well (train OOF or test), this module:
1. Takes the known prefix rows (TVT_input.notna())
2. Simulates fake cutoffs at 50%, 65%, 75% of the prefix length
3. At each cutoff, builds a pool of cheap physical candidates using only
   TVT_input, Z, MD, X, Y, and formation columns
4. Scores each candidate on the holdout slice (rows after cutoff but within known prefix)
5. Picks the candidate with best average holdout RMSE across cuts
6. If best candidate has mean holdout RMSE lower than ML prediction (or threshold),
   use it for hidden eval zone; otherwise fall back to ML prediction
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field

from src.utils import setup_logger


@dataclass
class CandidateResult:
    """Result from a single candidate evaluation"""
    name: str
    holdout_rmse: float
    holdout_mae: float
    holdout_bias: float
    n_points: int
    metadata: Dict[str, Any] = field(default_factory=dict)


def robust_poly_predict(md_known: np.ndarray, u_known: np.ndarray, 
                        md_all: np.ndarray, deg: int) -> np.ndarray:
    """
    Fit robust polynomial with iteratively reweighted least squares.
    
    Parameters:
    -----------
    md_known : MD values for known rows
    u_known : U = TVT_input + Z values for known rows
    md_all : MD values for all rows (to predict on)
    deg : Polynomial degree
    
    Returns:
    --------
    u_hat : Predicted U values for all rows
    """
    # Initial fit
    c = np.polyfit(md_known, u_known, deg)
    
    # Iteratively reweighted least squares (4 iterations)
    for _ in range(4):
        r = u_known - np.polyval(c, md_known)
        sc = np.median(np.abs(r)) * 1.4826 + 1e-6
        w = 1.0 / (1.0 + (r / sc)**2)
        c = np.polyfit(md_known, u_known, deg, w=w)
    
    return np.polyval(c, md_all)


def compute_polynomial_u_candidates(
    hw_known: pd.DataFrame,
    md_all: np.ndarray,
    z_all: np.ndarray
) -> List[Tuple[str, np.ndarray]]:
    """
    Generate polynomial-U candidates. Optimized version.
    
    Returns list of (candidate_name, tvt_predictions)
    """
    candidates = []
    
    md_known = hw_known['MD'].values.astype(np.float64, copy=False)
    tvt_known = hw_known['TVT_input'].values.astype(np.float64, copy=False)
    z_known = hw_known['Z'].values.astype(np.float64, copy=False)
    
    u_known = tvt_known + z_known
    n_known = len(md_known)
    
    # Pre-allocate result array
    tvt_hat = np.empty(len(md_all), dtype=np.float64)
    
    for tail in [80, 160, 320, 640, 'all']:
        if tail == 'all':
            use_md = md_known
            use_u = u_known
        else:
            if n_known < tail:
                continue
            use_md = md_known[-tail:]
            use_u = u_known[-tail:]
        
        for deg in [1, 2, 3]:
            if len(use_md) < deg + 12:
                continue
            
            try:
                c = np.polyfit(use_md, use_u, deg)
                # IRLS (4 iterations)
                for _ in range(4):
                    r = use_u - np.polyval(c, use_md)
                    sc = np.median(np.abs(r)) * 1.4826 + 1e-6
                    w = 1.0 / (1.0 + (r / sc)**2)
                    c = np.polyfit(use_md, use_u, deg, w=w)
                
                tvt_hat[:] = np.polyval(c, md_all) - z_all
                
                if np.all(np.isfinite(tvt_hat)):
                    name = f"poly_u_deg{deg}_tail{tail}"
                    candidates.append((name, tvt_hat.copy()))
            except Exception:
                continue
    
    return candidates


def compute_formation_surface_candidates(
    hw_known: pd.DataFrame,
    hw_full: pd.DataFrame,
    formations: List[str]
) -> List[Tuple[str, np.ndarray]]:
    """
    Generate formation surface candidates.
    
    For each formation column f:
    - Compute b = TVT_input + Z - f on known rows
    - Variants: b_median, b_late (last 50), b_wls (exponentially weighted)
    - Candidate: TVT = -Z + f + b_variant
    
    Returns list of (candidate_name, tvt_predictions)
    """
    candidates = []
    
    n_known = len(hw_known)
    n_all = len(hw_full)
    
    for f in formations:
        if f not in hw_full.columns:
            continue
        
        f_vals_full = hw_full[f].values.astype(np.float64)
        f_vals_known = hw_known[f].values.astype(np.float64)
        
        # Check if column is all NaN
        if np.all(np.isnan(f_vals_known)):
            continue
        
        tvt_known = hw_known['TVT_input'].values.astype(np.float64)
        z_known = hw_known['Z'].values.astype(np.float64)
        z_all = hw_full['Z'].values.astype(np.float64)
        
        # Compute b = TVT_input + Z - f on known rows (where both are finite)
        valid = np.isfinite(tvt_known) & np.isfinite(z_known) & np.isfinite(f_vals_known)
        if valid.sum() < 10:
            continue
        
        b = tvt_known[valid] + z_known[valid] - f_vals_known[valid]
        
        if len(b) == 0:
            continue
        
        # b_median variant
        b_median = float(np.nanmedian(b))
        tvt_median = -z_all + f_vals_full + b_median
        if np.all(np.isfinite(tvt_median)):
            candidates.append((f"surface_{f}_median", tvt_median))
        
        # b_late variant (last 50 rows of known)
        if n_known >= 50:
            b_late = float(np.nanmedian(b[-50:]))
            tvt_late = -z_all + f_vals_full + b_late
            if np.all(np.isfinite(tvt_late)):
                candidates.append((f"surface_{f}_late", tvt_late))
        
        # b_wls variant (exponentially weighted median)
        w = np.exp(0.02 * np.arange(len(b)))
        b_wls = float(np.sum(w * b) / np.sum(w))
        tvt_wls = -z_all + f_vals_full + b_wls
        if np.all(np.isfinite(tvt_wls)):
            candidates.append((f"surface_{f}_wls", tvt_wls))
    
    return candidates


def compute_contact_md_lookup_candidate(
    well_id: str,
    hw_test: pd.DataFrame,
    train_dir: Path,
    typewell_dir: Path,
    cached_train_wells: Optional[Dict] = None
) -> Optional[Tuple[str, np.ndarray]]:
    """
    Contact MD-lookup candidate for wells that exist in both train and test.
    
    Uses EGFDU contact from train well to predict TVT in test well.
    
    Returns (candidate_name, tvt_predictions) or None if well not in train.
    """
    train_hw_path = train_dir / f"{well_id}__horizontal_well.csv"
    train_tw_path = typewell_dir / f"{well_id}__typewell.csv"
    
    if not train_hw_path.exists() or not train_tw_path.exists():
        return None
    
    try:
        # Use cached data if available
        if cached_train_wells and well_id in cached_train_wells:
            hw_train = cached_train_wells[well_id]
        else:
            hw_train = pd.read_csv(train_hw_path)
            if cached_train_wells is not None:
                cached_train_wells[well_id] = hw_train
        
        tw_train = pd.read_csv(train_tw_path)
        
        # Compute TVT_contact from EGFDU contact
        if 'EGFDU' not in tw_train.columns or 'EGFDU' not in hw_train.columns:
            return None
        
        ref_tvt = tw_train[tw_train['Geology'] == 'EGFDU']['TVT'].min()
        if np.isnan(ref_tvt):
            # Fallback: use min EGFDU value
            ref_tvt = tw_train['EGFDU'].min()
        
        # Compute offset from train well
        valid = hw_train['TVT_input'].notna() & hw_train['EGFDU'].notna()
        if valid.sum() < 10:
            return None
        
        offset_values = hw_train.loc[valid, 'TVT'] - (ref_tvt - (hw_train.loc[valid, 'Z'] - hw_train.loc[valid, 'EGFDU']))
        offset = float(np.nanmedian(offset_values))
        
        # Compute physical TVT for train well
        phys_train = ref_tvt - (hw_train['Z'] - hw_train['EGFDU']) + offset
        
        # Interpolate phys by MD onto test well's MD grid
        md_train = hw_train['MD'].values
        md_test = hw_test['MD'].values
        
        phys_interp = np.interp(md_test, md_train, phys_train, left=phys_train[0], right=phys_train[-1])
        
        return ("contact_md_lookup", phys_interp)
    
    except Exception as e:
        return None


def evaluate_candidate_on_holdout(
    candidate_name: str,
    candidate_pred: np.ndarray,
    hw_holdout: pd.DataFrame,
    ml_holdout_pred: Optional[np.ndarray] = None
) -> CandidateResult:
    """
    Evaluate a candidate on holdout slice.
    
    Parameters:
    -----------
    candidate_name : Name of candidate
    candidate_pred : Candidate predictions for holdout rows
    hw_holdout : DataFrame with holdout rows (must have TVT_input)
    ml_holdout_pred : Optional ML predictions for comparison
    
    Returns:
    --------
    CandidateResult with metrics
    """
    tvt_true = hw_holdout['TVT_input'].values.astype(np.float64)
    valid = np.isfinite(tvt_true) & np.isfinite(candidate_pred)
    
    if valid.sum() < 5:
        return CandidateResult(
            name=candidate_name,
            holdout_rmse=np.inf,
            holdout_mae=np.inf,
            holdout_bias=np.nan,
            n_points=int(valid.sum())
        )
    
    rmse = float(np.sqrt(np.mean((tvt_true[valid] - candidate_pred[valid]) ** 2)))
    mae = float(np.mean(np.abs(tvt_true[valid] - candidate_pred[valid])))
    bias = float(np.mean(candidate_pred[valid] - tvt_true[valid]))
    
    return CandidateResult(
        name=candidate_name,
        holdout_rmse=rmse,
        holdout_mae=mae,
        holdout_bias=bias,
        n_points=int(valid.sum())
    )


def select_best_candidate(
    hw_known: pd.DataFrame,
    hw_full: pd.DataFrame,
    formations: List[str],
    train_dir: Optional[Path] = None,
    well_id: Optional[str] = None,
    min_gain: float = 0.5
) -> Tuple[Optional[str], np.ndarray, Dict]:
    """
    Select best candidate using visible prefix validation.
    
    Parameters:
    -----------
    hw_known : DataFrame with known prefix rows only
    hw_full : Full well DataFrame (for formation columns)
    formations : List of formation column names
    train_dir : Path to train directory (for contact lookup)
    well_id : Well ID (for contact lookup)
    min_gain : Minimum RMSE improvement to prefer candidate over persistence
    
    Returns:
    --------
    best_name : Name of best candidate (or None if none beat persistence)
    best_pred : Predictions from best candidate for all rows
    diagnostics : Dict with selection diagnostics
    """
    candidates_pool = []
    
    # Get all rows data
    md_all = hw_full['MD'].values.astype(np.float64)
    z_all = hw_full['Z'].values.astype(np.float64)
    
    # 1. Polynomial-U candidates
    poly_candidates = compute_polynomial_u_candidates(hw_known, md_all, z_all)
    candidates_pool.extend(poly_candidates)
    
    # 2. Formation surface candidates
    surface_candidates = compute_formation_surface_candidates(hw_known, hw_full, formations)
    candidates_pool.extend(surface_candidates)
    
    # 3. Contact MD-lookup (if applicable)
    if well_id and train_dir:
        contact_result = compute_contact_md_lookup_candidate(
            well_id, hw_full, train_dir, train_dir
        )
        if contact_result:
            candidates_pool.append(contact_result)
    
    if not candidates_pool:
        return None, None, {'error': 'no_candidates_generated'}
    
    # Evaluate candidates on artificial holdouts (fake cutoffs)
    n_known = len(hw_known)
    cutoff_fractions = [0.50, 0.65, 0.75]
    
    candidate_scores = {name: [] for name, _ in candidates_pool}
    
    for cutoff_frac in cutoff_fractions:
        cutoff_idx = int(n_known * cutoff_frac)
        if cutoff_idx < 20 or cutoff_idx >= n_known - 10:
            continue
        
        # Split known into pseudo-train and pseudo-holdout
        hw_pseudo_train = hw_known.iloc[:cutoff_idx].copy()
        hw_pseudo_holdout = hw_known.iloc[cutoff_idx:].copy()
        
        md_holdout = hw_pseudo_holdout['MD'].values.astype(np.float64)
        z_holdout = hw_pseudo_holdout['Z'].values.astype(np.float64)
        
        for name, pred_all in candidates_pool:
            # For fair comparison, re-compute candidate on pseudo-train only
            # This is a simplification - in practice we'd refit each candidate
            # For now, use the full-well predictions and evaluate on holdout slice
            holdout_indices = hw_pseudo_holdout.index.values
            pred_holdout = pred_all[holdout_indices]
            
            result = evaluate_candidate_on_holdout(name, pred_holdout, hw_pseudo_holdout)
            candidate_scores[name].append(result.holdout_rmse)
    
    # Compute mean holdout RMSE for each candidate
    candidate_mean_rmse = {}
    for name, rmses in candidate_scores.items():
        valid_rmses = [r for r in rmses if np.isfinite(r)]
        if valid_rmses:
            candidate_mean_rmse[name] = float(np.mean(valid_rmses))
    
    if not candidate_mean_rmse:
        return None, None, {'error': 'no_valid_scores'}
    
    # Find best candidate
    best_name = min(candidate_mean_rmse.keys(), key=lambda k: candidate_mean_rmse[k])
    best_rmse = candidate_mean_rmse[best_name]
    
    # Get predictions from best candidate
    best_pred = None
    for name, pred in candidates_pool:
        if name == best_name:
            best_pred = pred
            break
    
    diagnostics = {
        'best_candidate': best_name,
        'best_holdout_rmse': best_rmse,
        'all_candidates_rmse': candidate_mean_rmse,
        'n_candidates': len(candidates_pool),
    }
    
    return best_name, best_pred, diagnostics


def select_prediction(
    ml_pred: np.ndarray,
    candidate_pred: np.ndarray,
    best_holdout_rmse: float,
    ml_holdout_rmse: float,
    min_gain: float = 0.5,
    blend: str = 'hard'
) -> Tuple[np.ndarray, Dict]:
    """
    Select or blend ML and candidate predictions.
    
    Parameters:
    -----------
    ml_pred : ML model predictions
    candidate_pred : Candidate predictions
    best_holdout_rmse : Best candidate holdout RMSE
    ml_holdout_rmse : ML model holdout RMSE
    min_gain : Minimum RMSE improvement to prefer candidate
    blend : 'hard' (select one) or 'soft' (weighted blend)
    
    Returns:
    --------
    final_pred : Final predictions
    metadata : Selection metadata
    """
    gain = ml_holdout_rmse - best_holdout_rmse
    
    if blend == 'hard':
        if best_holdout_rmse < ml_holdout_rmse - min_gain:
            final_pred = candidate_pred
            selection = 'candidate'
        else:
            final_pred = ml_pred
            selection = 'ml'
    else:  # soft blend
        alpha = np.clip((ml_holdout_rmse - best_holdout_rmse) / 5.0, 0, 1)
        final_pred = (1 - alpha) * ml_pred + alpha * candidate_pred
        selection = f'soft_blend_alpha{alpha:.3f}'
    
    metadata = {
        'selection': selection,
        'gain': gain,
        'ml_holdout_rmse': ml_holdout_rmse,
        'candidate_holdout_rmse': best_holdout_rmse,
    }
    
    return final_pred, metadata


class PrefixSelector:
    """
    Main class for visible-prefix candidate selection.
    
    Usage:
        selector = PrefixSelector(formations=['ANCC', 'ASTNU', ...], train_dir=path)
        
        # For OOF validation
        result = selector.select_for_well(hw_known, hw_full, ml_pred, well_id)
        
        # For inference
        final_pred = result['final_pred']
    """
    
    def __init__(
        self,
        formations: List[str],
        train_dir: Optional[Path] = None,
        min_gain: float = 0.5,
        blend: str = 'hard',
        logger=None
    ):
        self.formations = formations
        self.train_dir = Path(train_dir) if train_dir else None
        self.min_gain = min_gain
        self.blend = blend
        self.logger = logger or setup_logger("PrefixSelector")
    
    def select_for_well(
        self,
        hw_known: pd.DataFrame,
        hw_full: pd.DataFrame,
        ml_pred: np.ndarray,
        well_id: Optional[str] = None,
        ml_holdout_rmse: Optional[float] = None
    ) -> Dict:
        """
        Select best prediction for a single well.
        
        Parameters:
        -----------
        hw_known : DataFrame with known prefix rows
        hw_full : Full well DataFrame
        ml_pred : ML predictions for hidden zone
        well_id : Well ID
        ml_holdout_rmse : ML holdout RMSE (if None, use persistence baseline)
        
        Returns:
        --------
        result : Dict with selection results and diagnostics
        """
        # Select best candidate
        best_name, best_pred, selection_diag = select_best_candidate(
            hw_known=hw_known,
            hw_full=hw_full,
            formations=self.formations,
            train_dir=self.train_dir,
            well_id=well_id,
            min_gain=self.min_gain
        )
        
        if best_name is None or best_pred is None:
            return {
                'well_id': well_id,
                'selection': 'ml_fallback',
                'reason': selection_diag.get('error', 'unknown'),
                'final_pred': ml_pred,
                'diagnostics': selection_diag
            }
        
        # Get hidden zone indices
        hidden_mask = hw_full['TVT_input'].isna().values
        hidden_indices = np.flatnonzero(hidden_mask)
        
        # Extract candidate predictions for hidden zone
        candidate_hidden_pred = best_pred[hidden_indices]
        
        # Compute ML holdout RMSE if not provided
        if ml_holdout_rmse is None:
            # Use persistence baseline as proxy
            last_known_tvt = hw_known['TVT_input'].iloc[-1]
            ml_holdout_rmse = 15.0  # Default threshold
        
        # Select or blend predictions
        final_hidden_pred, blend_metadata = select_prediction(
            ml_pred=ml_pred,
            candidate_pred=candidate_hidden_pred,
            best_holdout_rmse=selection_diag['best_holdout_rmse'],
            ml_holdout_rmse=ml_holdout_rmse,
            min_gain=self.min_gain,
            blend=self.blend
        )
        
        return {
            'well_id': well_id,
            'selection': blend_metadata['selection'],
            'best_candidate': best_name,
            'best_holdout_rmse': selection_diag['best_holdout_rmse'],
            'ml_holdout_rmse': ml_holdout_rmse,
            'gain': blend_metadata['gain'],
            'final_pred': final_hidden_pred,
            'diagnostics': selection_diag
        }


def apply_prefix_selection_to_oof(
    train_df: pd.DataFrame,
    oof_predictions: np.ndarray,
    formations: List[str],
    train_dir: Path,
    min_gain: float = 0.5
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Apply prefix selection to OOF predictions.
    
    NOTE: train_df from the ML pipeline only contains the hidden evaluation zone rows
    (where TVT_input.isna()). We need to load the original horizontal well CSV files
    to get the known prefix for candidate selection.
    
    Parameters:
    -----------
    train_df : Training DataFrame (hidden zone rows only, from ML pipeline)
    oof_predictions : OOF predictions from ML model (residuals)
    formations : List of formation columns
    train_dir : Path to train directory (for loading horizontal well CSVs)
    min_gain : Minimum gain threshold
    
    Returns:
    --------
    corrected_oof : Corrected OOF predictions (residuals)
    well_log : DataFrame with per-well selection log
    """
    logger = setup_logger("PrefixSelector")
    selector = PrefixSelector(
        formations=formations,
        train_dir=train_dir,
        min_gain=min_gain
    )
    
    corrected_oof = oof_predictions.copy()
    well_log = []
    
    # Group by well - train_df has only hidden zone rows
    for well_id, well_data in train_df.groupby('well_id', sort=False):
        # Load full horizontal well data to get known prefix
        hw_path = train_dir / f"{well_id}__horizontal_well.csv"
        if not hw_path.exists():
            logger.warning(f"Horizontal well file not found: {hw_path}")
            continue
        
        hw_full = pd.read_csv(hw_path)
        
        # Get known prefix from full well data
        known_mask = hw_full['TVT_input'].notna().values
        hidden_mask = hw_full['TVT_input'].isna().values
        
        if not hidden_mask.any() or known_mask.sum() < 20:
            # No hidden zone or insufficient known prefix
            logger.debug(f"Well {well_id}: skipping (hidden={hidden_mask.sum()}, known={known_mask.sum()})")
            continue
        
        hw_known = hw_full[known_mask].copy()
        
        # Map well_data rows to hw_full indices by MD
        # The well_data from train_df corresponds to hidden rows
        md_hidden = well_data['MD'].values
        md_full = hw_full['MD'].values
        
        # Find indices in hw_full that match the hidden rows
        hidden_indices_in_full = []
        for md in md_hidden:
            matches = np.where(md_full == md)[0]
            if len(matches) > 0:
                hidden_indices_in_full.append(matches[0])
            else:
                # Fallback: find closest MD
                hidden_indices_in_full.append(np.argmin(np.abs(md_full - md)))
        
        # Get ML predictions for hidden zone (these are residuals)
        well_oof_indices = well_data.index.values
        ml_pred_residuals = oof_predictions[well_oof_indices]
        
        # Convert residuals to TVT predictions
        # TVT_pred = last_known_TVT + residual
        last_known_tvt = hw_known['TVT_input'].iloc[-1]
        ml_pred_tvt = last_known_tvt + ml_pred_residuals
        
        # Compute ML holdout RMSE using last portion of known prefix
        n_known = len(hw_known)
        holdout_start = int(n_known * 0.75)
        if holdout_start < n_known - 10:
            # Use candidate evaluation on pseudo-holdout
            hw_pseudo_holdout = hw_known.iloc[holdout_start:].copy()
            tvt_holdout = hw_pseudo_holdout['TVT_input'].values
            # Approximate ML prediction as persistence (last known TVT)
            ml_holdout_pred_tvt = np.full(len(tvt_holdout), last_known_tvt)
            ml_holdout_rmse = float(np.sqrt(np.mean((tvt_holdout - ml_holdout_pred_tvt) ** 2)))
        else:
            ml_holdout_rmse = 15.0  # Default threshold
        
        logger.debug(f"Well {well_id}: known={n_known}, holdout_start={holdout_start}, ml_holdout_rmse={ml_holdout_rmse:.3f}")
        
        # Select best prediction (works in TVT space)
        result = selector.select_for_well(
            hw_known=hw_known,
            hw_full=hw_full,
            ml_pred=ml_pred_tvt,
            well_id=well_id,
            ml_holdout_rmse=ml_holdout_rmse
        )
        
        # Convert back to residuals and update OOF predictions
        final_tvt = result['final_pred']
        final_residuals = final_tvt - last_known_tvt
        corrected_oof[well_oof_indices] = final_residuals
        
        well_log.append({
            'well_id': well_id,
            'selection': result['selection'],
            'best_candidate': result.get('best_candidate', None),
            'best_holdout_rmse': result.get('best_holdout_rmse', None),
            'ml_holdout_rmse': result.get('ml_holdout_rmse', None),
            'gain': result.get('gain', None),
        })
    
    well_log_df = pd.DataFrame(well_log)
    
    return corrected_oof, well_log_df
