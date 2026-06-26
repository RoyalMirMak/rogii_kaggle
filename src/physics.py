# src/physics.py
import numpy as np
from pathlib import Path

class FormationPlaneKNN:
    def __init__(self, train_wells, data_dir, formations):
        self.train_wells = set(train_wells)
        self.data_dir = Path(data_dir)
        self.formations = formations
        
        # We NO LONGER store massive Pandas DataFrames.
        # This native Python list takes just kilobytes, making CPU pickling instant.
        self.neighbor_cache = [] 
        self._preload_data()

    def _preload_data(self):
        """Pre-loads formation tops and coordinates into native Python dictionaries."""
        import pandas as pd
        
        for wid in self.train_wells:
            tw_path = self.data_dir / f"train/{wid}__typewell.csv"
            hw_path = self.data_dir / f"train/{wid}__horizontal_well.csv"
            
            if not tw_path.is_file() or not hw_path.is_file():
                continue
                
            try:
                tw = pd.read_csv(tw_path)
                hw = pd.read_csv(hw_path, usecols=['X', 'Y'])
                
                if len(tw) == 0 or len(hw) == 0:
                    continue
                
                # Coordinate extraction
                x = float(tw["X"].iloc[-1]) if "X" in tw.columns else 0.0
                y = float(tw["Y"].iloc[-1]) if "Y" in tw.columns else 0.0
                px = float(hw["X"].iloc[-1])
                py = float(hw["Y"].iloc[-1])
                
                # Pre-extract formations into a fast dictionary so Pandas is bypassed in loops
                z_dict = {}
                for form in self.formations:
                    if form in tw.columns:
                        val = tw[form].iloc[0]
                        if not pd.isna(val):
                            z_dict[form] = float(val)
                            
                self.neighbor_cache.append((wid, x, y, px, py, z_dict))
            except Exception:
                pass

    def impute(self, xy_query, self_wid=None, k=10):
        xy_query = np.asarray(xy_query, dtype=np.float64)
        if xy_query.ndim == 1:
            xy_query = xy_query.reshape(1, -1)
        
        N = len(xy_query)
        F = len(self.formations)
        form_z = np.full((N, F), np.nan, dtype=np.float64)
        knn_dist = np.full(N, np.nan, dtype=np.float64)

        if len(self.neighbor_cache) == 0 or N == 0:
            return form_z, knn_dist

        # FASTER: Find the center point of the query well to do the search ONLY ONCE
        cqx = np.mean(xy_query[:, 0])
        cqy = np.mean(xy_query[:, 1])
            
        dists = []
        for (wid, x, y, px, py, z_dict) in self.neighbor_cache:
            if self_wid is not None and wid == self_wid:
                continue
            # Calculate distance from neighbor to the center of our well
            d = np.sqrt((cqx - px)**2 + (cqy - py)**2)
            dists.append((d, px, py, z_dict))
                
        dists.sort(key=lambda t: t[0])
        k_actual = min(k, len(dists))
        selected = dists[:k_actual]
        
        if k_actual == 0:
            return form_z, knn_dist
            
        # Store mean distance to those K neighbors
        knn_dist[:] = np.mean([t[0] for t in selected])

        # Fit planes and apply to all N rows instantly
        for fi, form_name in enumerate(self.formations):
            pts = []
            for (d, px, py, z_dict) in selected:
                if form_name in z_dict:
                    pts.append((px, py, z_dict[form_name], d))
                        
            if len(pts) < 2:
                # Not enough points, fill with median
                z_vals = [p[2] for p in pts]
                if z_vals: form_z[:, fi] = np.median(z_vals)
                continue

            # Matrix setup
            X_mat = np.array([[p[0], p[1], 1.0] for p in pts], dtype=np.float64)
            y_vec = np.array([p[2] for p in pts], dtype=np.float64)
            weights = np.array([1.0 / (p[3] + 1e-6) for p in pts], dtype=np.float64)
            
            W = np.diag(weights)
            XtWX = X_mat.T @ W @ X_mat
            XtWy = X_mat.T @ W @ y_vec
            
            try:
                coeffs = np.linalg.solve(XtWX, XtWy)
                # VECTORIZED: Apply the plane formula to all N points instantly!
                form_z[:, fi] = coeffs[0] * xy_query[:, 0] + coeffs[1] * xy_query[:, 1] + coeffs[2]
            except np.linalg.LinAlgError:
                form_z[:, fi] = np.median([p[2] for p in pts])
                    
        return form_z, knn_dist

def beam_search(hgr, tw_tvt, tw_gr, last_TVT, beam_size=10, move_cost=20.0, emit_scale=144.0, smooth_radius=2):
    """Vectorized NumPy Beam Search with Viterbi Backpointers (Zero Memory Bottleneck)"""
    N = len(hgr)
    if N == 0 or len(tw_tvt) == 0 or len(tw_gr) == 0:
        return np.zeros(N, dtype=np.float64)
        
    tvt_min = max(0, last_TVT - 200)
    tvt_max = last_TVT + 400
    tvt_states = np.arange(tvt_min, tvt_max, 5.0)
    S = len(tvt_states)
    
    if S == 0:
        return np.zeros(N, dtype=np.float64)

    costs = np.zeros(S, dtype=np.float64)
    # USE A BACKPOINTER TRACE INSTEAD OF MASSIVE ARRAY COPYING
    backpointers = np.zeros((N, S), dtype=np.int32) 

    tvt_idx = np.clip((tvt_states - tw_tvt[0]) / (tw_tvt[-1] - tw_tvt[0]) * len(tw_gr), 0, len(tw_gr) - 1).astype(int)
    state_gr = tw_gr[tvt_idx]

    state_diff = np.abs(tvt_states[:, None] - tvt_states[None, :])
    trans_cost_mat = move_cost * (state_diff / 50.0) ** 2

    for t in range(N):
        hgr_t = hgr[t] if not np.isnan(hgr[t]) else np.nanmean(tw_gr)
        emit_cost = ((hgr_t - state_gr) ** 2) / emit_scale
        total_cost_mat = costs[None, :] + trans_cost_mat + emit_cost[:, None]
        
        best_prev_idx = np.argmin(total_cost_mat, axis=1)
        
        costs = total_cost_mat[np.arange(S), best_prev_idx]
        
        # Just drop a breadcrumb index (Instant, zero RAM strain)
        backpointers[t] = best_prev_idx

    # Trace backwards through the breadcrumbs to build the path
    best_s = np.argmin(costs)
    result = np.zeros(N, dtype=np.float64)
    curr_s = best_s

    for t in range(N - 1, -1, -1):
        result[t] = tvt_states[curr_s]
        curr_s = backpointers[t, curr_s]

    if smooth_radius > 0:
        kernel = np.ones(2 * smooth_radius + 1) / (2 * smooth_radius + 1)
        result = np.convolve(result, kernel, mode="same")
        
    return result

def self_corr_tvt(kgr, vis_TVT, hgr, hw=15, stride=3):
    """Unchanged optimized self_corr_tvt"""
    N = len(hgr)
    if N == 0 or len(kgr) == 0 or len(vis_TVT) == 0:
        return np.zeros(N, dtype=np.float64), 0.0

    kgr = np.asarray(kgr, dtype=np.float64)
    hgr = np.asarray(hgr, dtype=np.float64)
    vis_TVT = np.asarray(vis_TVT, dtype=np.float64)

    kgr = (kgr - np.nanmean(kgr)) / (np.nanstd(kgr) + 1e-6)
    hgr = (hgr - np.nanmean(hgr)) / (np.nanstd(hgr) + 1e-6)

    max_lag = min(hw, len(kgr) // 3, len(hgr) // 2)
    max_lag = max(max_lag, 1)

    correlations = []
    lags = range(-max_lag, max_lag + 1)
    
    for lag in lags:
        if lag < 0:
            k_sub = kgr[:lag]
            h_sub = hgr[-lag:]
        elif lag > 0:
            k_sub = kgr[lag:]
            h_sub = hgr[:-lag]
        else:
            k_sub = kgr
            h_sub = hgr
            
        if len(k_sub) > 0 and len(h_sub) > 0:
            min_len = min(len(k_sub), len(h_sub))
            correlations.append(np.nanmean(k_sub[:min_len] * h_sub[:min_len]))
        else:
            correlations.append(0.0)

    best_lag_idx = np.argmax(correlations)
    best_lag = list(lags)[best_lag_idx]
    best_score = float(correlations[best_lag_idx])

    tvt_step = (vis_TVT[-1] - vis_TVT[0]) / len(vis_TVT) if len(vis_TVT) > 1 else 0.0
    base_tvt = vis_TVT[-1] if len(vis_TVT) > 0 else 0.0
    path = np.full(N, base_tvt + best_lag * tvt_step, dtype=np.float64)
    
    return path, best_score