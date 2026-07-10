# src/physics.py
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
import pandas as pd
from numba import njit

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


class DenseANCCImputer:
    def __init__(self, train_wells, data_dir, spw=60):
        xs, ys, anccs, wids = [], [], [], []
        for wid in train_wells:
            p = Path(data_dir) / f'train/{wid}__horizontal_well.csv'
            try: 
                df = pd.read_csv(p, usecols=['X', 'Y', 'ANCC']).dropna()
                if len(df) == 0: continue
                # Sample points along the wellbore
                ix = np.linspace(0, len(df)-1, min(spw, len(df)), dtype=int)
                s = df.iloc[ix]
                xs.append(s['X'].values); ys.append(s['Y'].values)
                anccs.append(s['ANCC'].values); wids.extend([wid]*len(s))
            except: continue
        self.xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.ancc = np.concatenate(anccs).astype(np.float32)
        self.wids = np.array(wids)
        self.scale = np.where(self.xy.std(0) < 1e-3, 1., self.xy.std(0))
        self.tree = cKDTree(self.xy / self.scale)

    def impute(self, xy_q, self_wid=None, k=20):
        xy_q = np.atleast_2d(xy_q); q = xy_q / self.scale
        dist, idx = self.tree.query(q, k=k+5, workers=-1)
        if self_wid: dist = np.where(self.wids[idx] == self_wid, np.inf, dist)
        ord_idx = np.argpartition(dist, k-1, axis=1)[:, :k]
        dk = np.take_along_axis(dist, ord_idx, 1)
        ik = np.take_along_axis(idx, ord_idx, 1)
        vk = np.isfinite(dk)
        w = np.where(vk, 1./(dk + 1e-3), 0.)
        sw = w.sum(1); safe = np.where(sw < 1e-9, 1., sw)
        ap = (self.ancc[ik] * w).sum(1) / safe
        ap = np.where(sw < 1e-9, float(self.ancc.mean()), ap)
        return ap.astype(np.float32)


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


# --- GLOBAL CONSTANTS FOR PARTICLE FILTER ---
PF_N=600; ANCC_N=600
PF_MOM=0.993; PF_VN=0.005; PF_PN=0.01
PF_GR_SIG_MIN=10.; PF_GR_SIG_MAX=60.; PF_GR_SIG_DEF=30.
PF_INIT_V_STD=0.02; PF_INIT_SPR=0.5; PF_RESAMP=0.5
PF_ROUGH_P=0.2; PF_ROUGH_V=0.003; PF_GR_WIN=5; PF_GR_WT=0.3
ANCC_ALPHA=0.998; ANCC_RN=0.002; ANCC_PN=0.005
ANCC_IR=0.01; ANCC_IS=0.3; ANCC_RP=0.1; ANCC_RR=0.001

@njit(cache=True)
def _interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0: return grid[0]
    n = len(grid) - 1
    if i >= n: return grid[n]
    t = (v - vmin) / step - i
    return grid[i]*(1.-t) + grid[i+1]*t

@njit(cache=True)
def _resamp(pos, aux, w, N, rp, rv):
    cum = np.zeros(N+1)
    for j in range(N): cum[j+1] = cum[j] + w[j]
    u0 = np.random.uniform(0., 1./N)
    np2 = np.empty(N); na = np.empty(N); ci = 0
    for j in range(N):
        u = u0 + j/N
        while ci < N-1 and cum[ci+1] < u: ci += 1
        np2[j] = pos[ci] + rp * np.random.randn()
        na[j]  = aux[ci] + rv * np.random.randn()
    return np2, na

@njit(cache=True)
def _pf_ancc(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N, ALPHA, RN, PN, IS, RP, RR, RESAMP):
    pos = np.empty(N); rate = np.empty(N); w = np.ones(N)/N
    for j in range(N):
        pos[j] = ls + IS * np.random.randn()
        rate[j] = ir + 0.01 * np.random.randn()
    pts = np.empty(len(md_v)); std_ = np.empty(len(md_v)); pm = md_v[0] - 1.
    for i in range(len(md_v)):
        dm = max(md_v[i] - pm, 1.)
        for j in range(N):
            rate[j] = ALPHA * rate[j] + RN * np.random.randn()
            pos[j] += rate[j] * dm + PN * np.random.randn()
            tvt_j = pos[j] - z_v[i]
            tvt_j = max(tvt_j, vmin - 50.); tvt_j = min(tvt_j, vmin + len(gg)*step + 50.)
            pos[j] = tvt_j + z_v[i]
        if not np.isnan(gr_v[i]):
            ws = 0.
            for j in range(N):
                eg = _interp1(gg, pos[j] - z_v[i], vmin, step)
                d = (gr_v[i] - eg) / gs
                lk = max(np.exp(-0.5 * d * d) if d * d < 600. else 0., 1e-300)
                w[j] *= lk; ws += w[j]
            for j in range(N): w[j] = w[j]/ws if ws > 0. else 1./N
        ne = 0.
        for j in range(N): ne += w[j]*w[j]
        if 1./ne < RESAMP * N:
            pos, rate = _resamp(pos, rate, w, N, RP, RR)
            for j in range(N): w[j] = 1./N
        tv = 0.; va = 0.
        for j in range(N): tv += w[j] * (pos[j] - z_v[i])
        pts[i] = tv
        for j in range(N): va += w[j] * (pos[j] - z_v[i] - tv)**2
        std_[i] = va**0.5; pm = md_v[i]
    return pts, std_

def _grid(tw_tvt, tw_gr, step=0.2):
    tmin = float(tw_tvt.min()); tmax = float(tw_tvt.max())
    tvt_g = np.arange(tmin, tmax + step, step)
    return np.interp(tvt_g, tw_tvt, tw_gr).astype(np.float64), float(tmin), float(step)

def _gr_sig(hw, tw_tvt, tw_gr):
    kn = hw[hw['TVT_input'].notna() & hw['GR'].notna()]
    if len(kn) < 20: return float(PF_GR_SIG_DEF)
    return float(np.clip(np.std(kn['GR'].values - np.interp(kn['TVT_input'].values, tw_tvt, tw_gr)), PF_GR_SIG_MIN, PF_GR_SIG_MAX))

def run_pf_ancc(hw, tw_tvt, tw_gr, N=ANCC_N):
    gs = _gr_sig(hw, tw_tvt, tw_gr)
    kn = hw[hw['TVT_input'].notna()]; ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0: return np.array([]), np.array([])
    ls = float(kn['TVT_input'].iloc[-1] + kn['Z'].iloc[-1])
    tail = kn.tail(30); dt = np.diff(tail['TVT_input'].values)
    dz = np.diff(tail['Z'].values); dm = np.diff(tail['MD'].values); m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    
    gr_interp = ev['GR'].interpolate(limit_direction='both').fillna(np.nanmean(tw_gr))
    pts, std = _pf_ancc(ev['MD'].values.astype(np.float64), ev['Z'].values.astype(np.float64),
                      gr_interp.values.astype(np.float64), gg, gmin, gst,
                      gs, ls, ir, N, ANCC_ALPHA, ANCC_RN, ANCC_PN, ANCC_IS, ANCC_RP, ANCC_RR, PF_RESAMP)
    return pts.astype(np.float32), std.astype(np.float32)

def _safe_interp(x, xp, fp, fallback=0.0):
    if len(xp) < 2 or len(fp) < 2:
        return np.full_like(np.asarray(x, dtype=np.float64), fallback, dtype=np.float64)
    return np.interp(x, xp, fp, left=fp[0], right=fp[-1])


def _estimate_z_velocity(hw):
    """
    Learns dTVT/dMD ~= beta * dZ/dMD + intercept from the visible prefix.
    Returns beta, intercept, residual velocity standard deviation, and
    a robust recent velocity initialization.
    """
    known = hw[hw["TVT_input"].notna()].copy()
    if len(known) < 30:
        return -1.0, 0.0, 0.02, 0.0

    md = known["MD"].to_numpy(dtype=np.float64)
    z = known["Z"].to_numpy(dtype=np.float64)
    tvt = known["TVT_input"].to_numpy(dtype=np.float64)

    dmd = np.diff(md)
    dz = np.diff(z)
    dtvt = np.diff(tvt)

    good = np.isfinite(dmd) & np.isfinite(dz) & np.isfinite(dtvt) & (np.abs(dmd) > 1e-8)
    if good.sum() < 10:
        return -1.0, 0.0, 0.02, 0.0

    vz = dz[good] / dmd[good]
    vt = dtvt[good] / dmd[good]

    design = np.column_stack([vz, np.ones_like(vz)])
    try:
        beta, intercept = np.linalg.lstsq(design, vt, rcond=None)[0]
        residual = vt - (beta * vz + intercept)
        sigma = max(float(np.nanstd(residual)), 0.002)
    except np.linalg.LinAlgError:
        beta, intercept, sigma = -1.0, 0.0, 0.02

    tail = min(30, len(vt))
    init_velocity = float(np.nanmedian(vt[-tail:])) if tail >= 3 else 0.0
    return float(beta), float(intercept), float(sigma), init_velocity


def _particle_resample(pos, vel, weights, rng, pos_noise=0.15, vel_noise=0.002):
    n_particles = len(pos)
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    positions = (rng.random() + np.arange(n_particles)) / n_particles
    indices = np.searchsorted(cumulative, positions, side="left")
    pos = pos[indices] + rng.normal(0.0, pos_noise, n_particles)
    vel = vel[indices] + rng.normal(0.0, vel_noise, n_particles)
    weights = np.full(n_particles, 1.0 / n_particles, dtype=np.float64)
    return pos, vel, weights


def run_pf_z_velocity(
    hw,
    tw_tvt,
    tw_gr,
    n_particles=600,
    momentum=0.993,
    velocity_noise=0.004,
    position_noise=0.008,
    init_position_std=0.50,
    init_velocity_std=0.015,
    gr_smooth_window=5,
    smooth_gr_weight=0.30,
    seed=42,
):
    """
    Direct-TVT particle filter.

    State:
        TVT position and dTVT/dMD velocity.

    Transition:
        Velocity is regularized by the visible-prefix relation between
        dTVT/dMD and dZ/dMD.

    Observation:
        Horizontal GR is compared with the typewell GR at each particle's
        TVT coordinate, using both raw and locally smoothed GR likelihoods.

    Returns
    -------
    pf_tvt : np.ndarray
        Absolute TVT estimate for the hidden/evaluation rows.
    pf_std : np.ndarray
        Particle posterior standard deviation for each estimate.
    """
    if len(tw_tvt) < 3 or len(tw_gr) < 3:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    known_mask = hw["TVT_input"].notna().to_numpy()
    hidden_mask = hw["TVT_input"].isna().to_numpy()

    if known_mask.sum() < 20 or hidden_mask.sum() == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    known = hw.loc[known_mask]
    hidden = hw.loc[hidden_mask]

    last_tvt = float(known["TVT_input"].iloc[-1])
    last_md = float(known["MD"].iloc[-1])
    last_z = float(known["Z"].iloc[-1])

    beta, intercept, velocity_sigma, init_velocity = _estimate_z_velocity(hw)

    known_gr = known["GR"].to_numpy(dtype=np.float64)
    known_tvt = known["TVT_input"].to_numpy(dtype=np.float64)
    expected_known_gr = _safe_interp(known_tvt, tw_tvt, tw_gr, fallback=np.nanmean(tw_gr))
    residual_gr = known_gr - expected_known_gr
    finite_gr = np.isfinite(residual_gr)

    if finite_gr.sum() >= 20:
        gr_sigma = float(np.clip(np.nanstd(residual_gr[finite_gr]), 10.0, 60.0))
    else:
        gr_sigma = 30.0

    full_gr = hw["GR"].astype(float).interpolate(limit_direction="both")
    full_gr = full_gr.fillna(float(np.nanmean(tw_gr)))
    smooth_gr = full_gr.rolling(
        gr_smooth_window,
        center=True,
        min_periods=1,
    ).mean().to_numpy(dtype=np.float64)

    hidden_indices = np.flatnonzero(hidden_mask)
    hidden_md = hw.loc[hidden_mask, "MD"].to_numpy(dtype=np.float64)
    hidden_z = hw.loc[hidden_mask, "Z"].to_numpy(dtype=np.float64)
    hidden_gr = full_gr.to_numpy(dtype=np.float64)[hidden_indices]
    hidden_gr_smooth = smooth_gr[hidden_indices]

    tw_smooth_gr = pd.Series(tw_gr).rolling(
        gr_smooth_window,
        center=True,
        min_periods=1,
    ).mean().to_numpy(dtype=np.float64)

    rng = np.random.default_rng(seed)
    pos = last_tvt + rng.normal(0.0, init_position_std, n_particles)
    vel = init_velocity + rng.normal(0.0, init_velocity_std, n_particles)
    weights = np.full(n_particles, 1.0 / n_particles, dtype=np.float64)

    pf_tvt = np.zeros(len(hidden), dtype=np.float64)
    pf_std = np.zeros(len(hidden), dtype=np.float64)

    previous_md = last_md
    previous_z = last_z

    tvt_min = float(np.nanmin(tw_tvt)) - 80.0
    tvt_max = float(np.nanmax(tw_tvt)) + 80.0

    for i in range(len(hidden)):
        current_md = hidden_md[i]
        current_z = hidden_z[i]

        dmd = max(current_md - previous_md, 1.0)
        dzdmd = (current_z - previous_z) / dmd
        expected_velocity = beta * dzdmd + intercept

        vel = (
            momentum * vel
            + (1.0 - momentum) * expected_velocity
            + rng.normal(0.0, velocity_noise, n_particles)
        )
        pos = pos + vel * dmd + rng.normal(0.0, position_noise, n_particles)
        pos = np.clip(pos, tvt_min, tvt_max)

        raw_expected = _safe_interp(pos, tw_tvt, tw_gr, fallback=float(np.nanmean(tw_gr)))
        smooth_expected = _safe_interp(pos, tw_tvt, tw_smooth_gr, fallback=float(np.nanmean(tw_smooth_gr)))

        if np.isfinite(hidden_gr[i]):
            raw_likelihood = np.exp(
                -0.5 * ((hidden_gr[i] - raw_expected) / gr_sigma) ** 2
            )
        else:
            raw_likelihood = np.ones(n_particles, dtype=np.float64)

        if np.isfinite(hidden_gr_smooth[i]):
            smooth_likelihood = np.exp(
                -0.5 * ((hidden_gr_smooth[i] - smooth_expected) / (gr_sigma * 1.5)) ** 2
            )
        else:
            smooth_likelihood = np.ones(n_particles, dtype=np.float64)

        velocity_likelihood = np.exp(
            -0.5 * ((vel - expected_velocity) / max(velocity_sigma * 2.0, 0.005)) ** 2
        )

        likelihood = (
            (1.0 - smooth_gr_weight) * raw_likelihood
            + smooth_gr_weight * smooth_likelihood
        )
        weights *= np.maximum(likelihood, 1e-300)
        weights *= np.maximum(velocity_likelihood, 1e-300)

        weight_sum = weights.sum()
        if not np.isfinite(weight_sum) or weight_sum <= 1e-300:
            weights.fill(1.0 / n_particles)
        else:
            weights /= weight_sum

        posterior_mean = float(np.average(pos, weights=weights))
        posterior_std = float(np.sqrt(np.average((pos - posterior_mean) ** 2, weights=weights)))

        pf_tvt[i] = posterior_mean
        pf_std[i] = posterior_std

        effective_n = 1.0 / np.sum(weights ** 2)
        if effective_n < 0.50 * n_particles:
            pos, vel, weights = _particle_resample(
                pos,
                vel,
                weights,
                rng,
                pos_noise=0.20,
                vel_noise=0.003,
            )

        previous_md = current_md
        previous_z = current_z

    return pf_tvt.astype(np.float32), pf_std.astype(np.float32)

def multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3):
    out = []
    for hw in hws:
        win = 2 * hw + 1; nk = len(kgr); nh = len(hgr)
        if nk < win + 1 or nh == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32))); continue
        kg = pd.Series(kgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        hg = pd.Series(hgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        sts = np.arange(0, nk - win + 1, stride, dtype=np.int32)
        if len(sts) == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32))); continue
        C = kg[sts[:, None] + np.arange(win, dtype=np.int32)[None, :]].astype(np.float32)
        Cn = (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-6)
        hp = np.pad(hg, hw, mode='edge')
        H = hp[np.arange(nh)[:, None] + np.arange(win)[None, :]].astype(np.float32)
        Hn = (H - H.mean(1, keepdims=True)) / (H.std(1, keepdims=True) + 1e-6)
        ncc = Hn @ Cn.T / win; best = ncc.argmax(1); score = ncc.max(1).astype(np.float32)
        out.append((ktvt[np.clip(sts[best] + hw, 0, nk - 1)].astype(np.float32), score))
    
    tvts = np.stack([o[0] for o in out], 1); scores = np.stack([o[1] for o in out], 1)
    sw = np.exp(3. * scores); sw /= sw.sum(1, keepdims=True) + 1e-9
    sc_ens = (tvts * sw).sum(1).astype(np.float32)
    return out, sc_ens

def affine_cal(kgr, tw_at_k, min_pts=20):
    """Calculates scale (a) and shift (b) between well GR and typewell GR."""
    kgr = np.asarray(kgr, float)
    tw_at_k = np.asarray(tw_at_k, float)
    v = np.isfinite(kgr) & np.isfinite(tw_at_k)
    if v.sum() < min_pts or np.std(tw_at_k[v]) < 1e-6:
        return 1.0, float(np.nanmean(kgr[v]) - np.nanmean(tw_at_k[v])) if v.any() else 0.0
    a, b = np.polyfit(tw_at_k[v], kgr[v], 1)
    return float(a), float(b)

def seg_b_well(ktvt, kz, form_col):
    """Calculates formation offsets for early, mid, late, and WLS segments."""
    bv = ktvt + kz - form_col
    n = len(bv)
    b_full = float(np.median(bv))
    b_late = float(np.median(bv[max(0, n-50):])) if n >= 5 else b_full
    t1, t2 = n // 3, 2 * n // 3
    b_early = float(np.median(bv[:max(1, t1)])) if t1 > 0 else b_full
    b_mid = float(np.median(bv[t1:max(t1+1, t2)])) if t2 > t1 else b_full
    
    # Weighted Least Squares (exponentially favors recent points)
    w = np.exp(0.02 * np.arange(n))
    w /= w.sum()
    b_wls = float(np.dot(w, bv))
    return b_full, b_early, b_mid, b_late, b_wls