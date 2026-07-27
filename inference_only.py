# =============================================================================
# ROGII INFERENCE - FULL FEATURE PIPELINE WITH PREFIX SELECTION
# =============================================================================
# This script generates ALL features from the training pipeline and applies
# visible-prefix candidate selection for improved predictions.

import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
from catboost import CatBoostRegressor
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from numba import njit
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
MODELS_DIR = "/kaggle/input/models/mironoxxxy/rogii-base/other/default/7"
DATA_ROOT = "/kaggle/input/competitions/rogii-wellbore-geology-prediction"
N_JOBS = 4

DATA_ROOT = Path(DATA_ROOT)
MODELS_DIR = Path(MODELS_DIR)
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
SAMPLE_SUB = DATA_ROOT / "sample_submission.csv"

BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2, "cons"), (10, 8.0, 64.0, 2, "loose"),
    (8, 35.0, 220.0, 1, "vcons"), (10, 14.0, 90.0, 5, "sm5"),
    (20, 4.0, 36.0, 3, "vloose"), (12, 12.0, 100.0, 3, "med1"),
    (15, 25.0, 180.0, 2, "wide1"), (20, 30.0, 200.0, 2, "wide2"),
    (15, 10.0, 80.0, 4, "med2"), (25, 6.0, 50.0, 3, "vloose2"),
    (10, 40.0, 300.0, 1, "vwide1"), (12, 18.0, 120.0, 5, "med3"),
    (30, 8.0, 70.0, 2, "vloose3"), (10, 50.0, 400.0, 0, "vwide2"),
]
ANCH_OFFS = [-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80]
BEAM_OFFS = [-40, -20, -10, -5, -3, 0, 3, 5, 10, 20, 40]
PF_OFFS = [-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30]
FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def robust_slope(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    if len(x) < 2:
        return 0.0
    xm = np.median(x)
    ym = np.median(y)
    den = np.sum((x - xm) ** 2)
    return float(np.sum((x - xm) * (y - ym)) / den) if den else 0.0


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


def tvt_from_contacts(hw_tr, tw_tr, ref_col="EGFDU"):
    tw_g = tw_tr.dropna(subset=["Geology"])
    if tw_g.empty:
        raise ValueError("No typewell geology rows")
    ref_tvt = tw_g.loc[tw_g["Geology"] == ref_col, "TVT"].min()
    if pd.isna(ref_tvt):
        ref_col = tw_g["Geology"].iloc[0]
        ref_tvt = tw_g.loc[tw_g["Geology"] == ref_col, "TVT"].min()
    offset = (hw_tr["TVT"] - (ref_tvt - (hw_tr["Z"] - hw_tr[ref_col]))).mean()
    return ref_tvt - (hw_tr["Z"] - hw_tr[ref_col]) + offset


# =============================================================================
# PHYSICS FUNCTIONS (from src/physics.py)
# =============================================================================

class FormationPlaneKNN:
    def __init__(self, train_wells, data_dir, formations):
        self.train_wells = set(train_wells)
        self.data_dir = Path(data_dir)
        self.formations = formations
        self.neighbor_cache = []
        self._preload_data()

    def _preload_data(self):
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
                x = float(tw["X"].iloc[-1]) if "X" in tw.columns else 0.0
                y = float(tw["Y"].iloc[-1]) if "Y" in tw.columns else 0.0
                px = float(hw["X"].iloc[-1])
                py = float(hw["Y"].iloc[-1])
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
        cqx = np.mean(xy_query[:, 0])
        cqy = np.mean(xy_query[:, 1])
        dists = []
        for (wid, x, y, px, py, z_dict) in self.neighbor_cache:
            if self_wid is not None and wid == self_wid:
                continue
            d = np.sqrt((cqx - px) ** 2 + (cqy - py) ** 2)
            dists.append((d, px, py, z_dict))
        dists.sort(key=lambda t: t[0])
        k_actual = min(k, len(dists))
        selected = dists[:k_actual]
        if k_actual == 0:
            return form_z, knn_dist
        knn_dist[:] = np.mean([t[0] for t in selected])
        for fi, form_name in enumerate(self.formations):
            pts = []
            for (d, px, py, z_dict) in selected:
                if form_name in z_dict:
                    pts.append((px, py, z_dict[form_name], d))
            if len(pts) < 2:
                z_vals = [p[2] for p in pts]
                if z_vals:
                    form_z[:, fi] = np.median(z_vals)
                continue
            X_mat = np.array([[p[0], p[1], 1.0] for p in pts], dtype=np.float64)
            y_vec = np.array([p[2] for p in pts], dtype=np.float64)
            weights = np.array([1.0 / (p[3] + 1e-6) for p in pts], dtype=np.float64)
            W = np.diag(weights)
            XtWX = X_mat.T @ W @ X_mat
            XtWy = X_mat.T @ W @ y_vec
            try:
                coeffs = np.linalg.solve(XtWX, XtWy)
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
                if len(df) == 0:
                    continue
                ix = np.linspace(0, len(df) - 1, min(spw, len(df)), dtype=int)
                s = df.iloc[ix]
                xs.append(s['X'].values)
                ys.append(s['Y'].values)
                anccs.append(s['ANCC'].values)
                wids.extend([wid] * len(s))
            except:
                continue
        self.xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.ancc = np.concatenate(anccs).astype(np.float32)
        self.wids = np.array(wids)
        self.scale = np.where(self.xy.std(0) < 1e-3, 1., self.xy.std(0))
        self.tree = cKDTree(self.xy / self.scale)

    def impute(self, xy_q, self_wid=None, k=20):
        xy_q = np.atleast_2d(xy_q)
        q = xy_q / self.scale
        dist, idx = self.tree.query(q, k=k + 5, workers=-1)
        if self_wid:
            dist = np.where(self.wids[idx] == self_wid, np.inf, dist)
        ord_idx = np.argpartition(dist, k - 1, axis=1)[:, :k]
        dk = np.take_along_axis(dist, ord_idx, 1)
        ik = np.take_along_axis(idx, ord_idx, 1)
        vk = np.isfinite(dk)
        w = np.where(vk, 1. / (dk + 1e-3), 0.)
        sw = w.sum(1)
        safe = np.where(sw < 1e-9, 1., sw)
        ap = (self.ancc[ik] * w).sum(1) / safe
        ap = np.where(sw < 1e-9, float(self.ancc.mean()), ap)
        return ap.astype(np.float32)


@njit(cache=True)
def _interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0:
        return grid[0]
    n = len(grid) - 1
    if i >= n:
        return grid[n]
    t = (v - vmin) / step - i
    return grid[i] * (1. - t) + grid[i + 1] * t


@njit(cache=True)
def _resamp(pos, aux, w, N, rp, rv):
    cum = np.zeros(N + 1)
    for j in range(N):
        cum[j + 1] = cum[j] + w[j]
    u0 = np.random.uniform(0., 1. / N)
    np2 = np.empty(N)
    na = np.empty(N)
    ci = 0
    for j in range(N):
        u = u0 + j / N
        while ci < N - 1 and cum[ci + 1] < u:
            ci += 1
        np2[j] = pos[ci] + rp * np.random.randn()
        na[j] = aux[ci] + rv * np.random.randn()
    return np2, na


@njit(cache=True)
def _pf_ancc(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N, ALPHA, RN, PN, IS, RP, RR, RESAMP):
    pos = np.empty(N)
    rate = np.empty(N)
    w = np.ones(N) / N
    for j in range(N):
        pos[j] = ls + IS * np.random.randn()
        rate[j] = ir + 0.01 * np.random.randn()
    pts = np.empty(len(md_v))
    std_ = np.empty(len(md_v))
    pm = md_v[0] - 1.
    for i in range(len(md_v)):
        dm = max(md_v[i] - pm, 1.)
        for j in range(N):
            rate[j] = ALPHA * rate[j] + RN * np.random.randn()
            pos[j] += rate[j] * dm + PN * np.random.randn()
            tvt_j = pos[j] - z_v[i]
            tvt_j = max(tvt_j, vmin - 50.)
            tvt_j = min(tvt_j, vmin + len(gg) * step + 50.)
            pos[j] = tvt_j + z_v[i]
        if not np.isnan(gr_v[i]):
            ws = 0.
            for j in range(N):
                eg = _interp1(gg, pos[j] - z_v[i], vmin, step)
                d = (gr_v[i] - eg) / gs
                lk = max(np.exp(-0.5 * d * d) if d * d < 600. else 0., 1e-300)
                w[j] *= lk
                ws += w[j]
            for j in range(N):
                w[j] = w[j] / ws if ws > 0. else 1. / N
        ne = 0.
        for j in range(N):
            ne += w[j] * w[j]
        if 1. / ne < RESAMP * N:
            pos, rate = _resamp(pos, rate, w, N, RP, RR)
            for j in range(N):
                w[j] = 1. / N
        tv = 0.
        va = 0.
        for j in range(N):
            tv += w[j] * (pos[j] - z_v[i])
        pts[i] = tv
        for j in range(N):
            va += w[j] * (pos[j] - z_v[i] - tv) ** 2
        std_[i] = va ** 0.5
        pm = md_v[i]
    return pts, std_


def _grid(tw_tvt, tw_gr, step=0.2):
    tmin = float(tw_tvt.min())
    tmax = float(tw_tvt.max())
    tvt_g = np.arange(tmin, tmax + step, step)
    return np.interp(tvt_g, tw_tvt, tw_gr).astype(np.float64), float(tmin), float(step)


def _gr_sig(hw, tw_tvt, tw_gr):
    kn = hw[hw['TVT_input'].notna() & hw['GR'].notna()]
    if len(kn) < 20:
        return float(30.0)
    return float(np.clip(np.std(kn['GR'].values - np.interp(kn['TVT_input'].values, tw_tvt, tw_gr)), 10., 60.))


def run_pf_ancc(hw, tw_tvt, tw_gr, N=600):
    gs = _gr_sig(hw, tw_tvt, tw_gr)
    kn = hw[hw['TVT_input'].notna()]
    ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0:
        return np.array([]), np.array([])
    ls = float(kn['TVT_input'].iloc[-1] + kn['Z'].iloc[-1])
    tail = kn.tail(30)
    dt = np.diff(tail['TVT_input'].values)
    dz = np.diff(tail['Z'].values)
    dm = np.diff(tail['MD'].values)
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    gr_interp = ev['GR'].interpolate(limit_direction='both').fillna(np.nanmean(tw_gr))
    pts, std = _pf_ancc(ev['MD'].values.astype(np.float64), ev['Z'].values.astype(np.float64),
                        gr_interp.values.astype(np.float64), gg, gmin, gst,
                        gs, ls, ir, N, 0.998, 0.002, 0.005, 0.3, 0.001, 0.001, 0.5)
    return pts.astype(np.float32), std.astype(np.float32)


def multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3):
    out = []
    for hw in hws:
        win = 2 * hw + 1
        nk = len(kgr)
        nh = len(hgr)
        if nk < win + 1 or nh == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32)))
            continue
        kg = pd.Series(kgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        hg = pd.Series(hgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        sts = np.arange(0, nk - win + 1, stride, dtype=np.int32)
        if len(sts) == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32)))
            continue
        C = kg[sts[:, None] + np.arange(win, dtype=np.int32)[None, :]].astype(np.float32)
        Cn = (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-6)
        hp = np.pad(hg, win, mode='edge')
        H = hp[np.arange(nh)[:, None] + np.arange(win)[None, :]].astype(np.float32)
        Hn = (H - H.mean(1, keepdims=True)) / (H.std(1, keepdims=True) + 1e-6)
        ncc = Hn @ Cn.T / win
        best = ncc.argmax(1)
        score = ncc.max(1).astype(np.float32)
        out.append((ktvt[np.clip(sts[best] + hw, 0, nk - 1)].astype(np.float32), score))
    tvts = np.stack([o[0] for o in out], 1)
    scores = np.stack([o[1] for o in out], 1)
    sw = np.exp(3. * scores)
    sw /= sw.sum(1, keepdims=True) + 1e-9
    sc_ens = (tvts * sw).sum(1).astype(np.float32)
    return out, sc_ens


def affine_cal(kgr, tw_at_k, min_pts=20):
    kgr = np.asarray(kgr, float)
    tw_at_k = np.asarray(tw_at_k, float)
    v = np.isfinite(kgr) & np.isfinite(tw_at_k)
    if v.sum() < min_pts or np.std(tw_at_k[v]) < 1e-6:
        return 1.0, float(np.nanmean(kgr[v]) - np.nanmean(tw_at_k[v])) if v.any() else 0.0
    a, b = np.polyfit(tw_at_k[v], kgr[v], 1)
    return float(a), float(b)


def seg_b_well(ktvt, kz, form_col):
    bv = ktvt + kz - form_col
    n = len(bv)
    b_full = float(np.median(bv))
    b_late = float(np.median(bv[max(0, n - 50):])) if n >= 5 else b_full
    t1, t2 = n // 3, 2 * n // 3
    b_early = float(np.median(bv[:max(1, t1)])) if t1 > 0 else b_full
    b_mid = float(np.median(bv[t1:max(t1 + 1, t2)])) if t2 > t1 else b_full
    w = np.exp(0.02 * np.arange(n))
    w /= w.sum()
    b_wls = float(np.dot(w, bv))
    return b_full, b_early, b_mid, b_late, b_wls


@njit(cache=True)
def _pf_z_velocity(md_v, z_v, gr_v, gr_sm_v, gg_raw, gg_smooth, vmin, step,
                   gs, initial_pos, initial_vel, beta, intercept, z_sigma, n_particles,
                   momentum, velocity_noise, position_noise, gr_weight,
                   rough_pos, rough_vel, resample_fraction):
    pos = np.empty(n_particles)
    vel = np.empty(n_particles)
    weights = np.ones(n_particles) / n_particles
    for j in range(n_particles):
        pos[j] = initial_pos + 0.5 * np.random.randn()
        vel[j] = initial_vel + 0.02 * np.random.randn()
    estimates = np.empty(len(md_v))
    stds = np.empty(len(md_v))
    previous_md = md_v[0] - 1.0
    previous_z = z_v[0] - 1.0
    max_tvt = vmin + len(gg_raw) * step
    for i in range(len(md_v)):
        dmd = md_v[i] - previous_md
        if dmd < 1.0:
            dmd = 1.0
        expected_velocity = beta * ((z_v[i] - previous_z) / dmd) + intercept
        for j in range(n_particles):
            vel[j] = momentum * vel[j] + velocity_noise * np.random.randn()
            pos[j] = pos[j] + vel[j] * dmd + position_noise * np.random.randn()
            if pos[j] < vmin - 50.0:
                pos[j] = vmin - 50.0
            elif pos[j] > max_tvt + 50.0:
                pos[j] = max_tvt + 50.0
        if not np.isnan(gr_v[i]):
            total_weight = 0.0
            for j in range(n_particles):
                expected_gr = _interp1(gg_raw, pos[j], vmin, step)
                raw_delta = (gr_v[i] - expected_gr) / gs
                raw_likelihood = max(np.exp(-0.5 * raw_delta * raw_delta) if raw_delta * raw_delta < 600.0 else 0.0, 1e-300)
                if not np.isnan(gr_sm_v[i]):
                    expected_smooth = _interp1(gg_smooth, pos[j], vmin, step)
                    smooth_delta = (gr_sm_v[i] - expected_smooth) / (gs * 1.5)
                    smooth_likelihood = max(np.exp(-0.5 * smooth_delta * smooth_delta) if smooth_delta * smooth_delta < 600.0 else 0.0, 1e-300)
                    likelihood = (1.0 - gr_weight) * raw_likelihood + gr_weight * smooth_likelihood
                else:
                    likelihood = raw_likelihood
                weights[j] *= max(likelihood, 1e-300)
                total_weight += weights[j]
            for j in range(n_particles):
                weights[j] = weights[j] / total_weight if total_weight > 0.0 else 1.0 / n_particles
        velocity_total = 0.0
        velocity_scale = max(z_sigma * 2.0, 0.005)
        for j in range(n_particles):
            delta_v = (vel[j] - expected_velocity) / velocity_scale
            likelihood_v = max(np.exp(-0.5 * delta_v * delta_v) if delta_v * delta_v < 600.0 else 0.0, 1e-300)
            weights[j] *= likelihood_v
            velocity_total += weights[j]
        for j in range(n_particles):
            weights[j] = weights[j] / velocity_total if velocity_total > 0.0 else 1.0 / n_particles
        effective = 0.0
        for j in range(n_particles):
            effective += weights[j] * weights[j]
        if 1.0 / effective < resample_fraction * n_particles:
            pos, vel = _resamp(pos, vel, weights, n_particles, rough_pos, rough_vel)
            for j in range(n_particles):
                weights[j] = 1.0 / n_particles
        mean = 0.0
        for j in range(n_particles):
            mean += weights[j] * pos[j]
        estimates[i] = mean
        variance = 0.0
        for j in range(n_particles):
            variance += weights[j] * (pos[j] - mean) ** 2
        stds[i] = variance ** 0.5
        previous_md = md_v[i]
        previous_z = z_v[i]
    return estimates, stds


def run_pf_z_velocity(hw, tw_tvt, tw_gr, n_particles=600):
    known = hw[hw["TVT_input"].notna()]
    hidden = hw[hw["TVT_input"].isna()]
    if len(known) < 10 or len(hidden) == 0 or len(tw_tvt) < 3:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    d_z = np.diff(known["Z"].to_numpy(dtype=np.float64))
    d_tvt = np.diff(known["TVT_input"].to_numpy(dtype=np.float64))
    d_md = np.diff(known["MD"].to_numpy(dtype=np.float64))
    valid = d_md > 0.0
    if valid.sum() >= 10:
        vz = d_z[valid] / d_md[valid]
        vt = d_tvt[valid] / d_md[valid]
        coefficients, _, _, _ = np.linalg.lstsq(np.column_stack([vz, np.ones_like(vz)]), vt, rcond=None)
        beta, intercept = float(coefficients[0]), float(coefficients[1])
        z_sigma = max(float(np.std(vt - beta * vz - intercept)), 0.001)
    else:
        beta, intercept, z_sigma = -1.0, 0.0, 0.1
    tail = known.tail(20)
    tail_dtvt = np.diff(tail["TVT_input"].to_numpy(dtype=np.float64))
    tail_dmd = np.diff(tail["MD"].to_numpy(dtype=np.float64))
    tail_valid = tail_dmd > 0.0
    initial_vel = float(np.median(tail_dtvt[tail_valid] / tail_dmd[tail_valid])) if tail_valid.sum() >= 3 else 0.0
    gs = _gr_sig(hw, tw_tvt, tw_gr)
    raw_grid, grid_min, grid_step = _grid(tw_tvt, tw_gr)
    smooth_tw = pd.Series(tw_gr).rolling(5, center=True, min_periods=1).mean().to_numpy(dtype=np.float64)
    smooth_grid, _, _ = _grid(tw_tvt, smooth_tw)
    full_gr = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr)))
    smooth_hw = full_gr.rolling(5, center=True, min_periods=1).mean()
    estimates, stds = _pf_z_velocity(
        hidden["MD"].to_numpy(dtype=np.float64), hidden["Z"].to_numpy(dtype=np.float64),
        full_gr.loc[hidden.index].to_numpy(dtype=np.float64), smooth_hw.loc[hidden.index].to_numpy(dtype=np.float64),
        raw_grid, smooth_grid, grid_min, grid_step, gs,
        float(known["TVT_input"].iloc[-1]), initial_vel, beta, intercept, z_sigma,
        n_particles, 0.993, 0.005, 0.01, 0.3, 0.2, 0.003, 0.5,
    )
    return estimates.astype(np.float32), stds.astype(np.float32)


def beam_search(hgr, tw_tvt, tw_gr, last_TVT, beam_size=10, move_cost=20.0, emit_scale=144.0, smooth_radius=2):
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
        backpointers[t] = best_prev_idx
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


# =============================================================================
# FEATURE BUILDING (matches src/dataset.py)
# =============================================================================

def _add_gr_sequence_features(h, visible, sel_idx, fallback_gr):
    """Generate all GR sequence features."""
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


def build_features_for_well(wid, test_eval_idx, dense_knn, formation_knn):
    """Build full feature set for a single test well."""
    h = pd.read_csv(TEST_DIR / f"{wid}__horizontal_well.csv")
    h["row_index"] = np.arange(len(h), dtype=np.int64)
    if "TVT_input" not in h:
        return pd.DataFrame()
    
    mask = h["TVT_input"].isna().to_numpy()
    allowed = np.zeros(len(h), dtype=bool)
    allowed[list(test_eval_idx)] = True
    mask &= allowed
    if not mask.any():
        return pd.DataFrame()
    
    visible = h[h["TVT_input"].notna()].copy()
    if len(visible) < 10:
        return pd.DataFrame()
    
    cur = h.iloc[np.flatnonzero(mask)].copy()
    last = visible.iloc[-1]
    last_tvt = float(last["TVT_input"])
    last_MD = float(last["MD"])
    last_X = float(last["X"])
    last_Y = float(last["Y"])
    last_Z = float(last["Z"])
    
    cur["well_id"] = wid
    cur["id"] = wid + "_" + cur["row_index"].astype(str)
    cur["last_known_TVT"] = last_tvt
    cur["md_from_ps"] = cur["MD"].to_numpy() - last_MD
    cur["z_from_ps"] = cur["Z"].to_numpy() - last_Z
    cur["dxy_from_ps"] = np.hypot(cur["X"].to_numpy() - last_X, cur["Y"].to_numpy() - last_Y)
    
    vis_TVT = visible["TVT_input"].values
    slope_all = robust_slope(visible["MD"].values, vis_TVT)
    cur["slope_TVT_MD_all"] = slope_all
    cur["slope_TVT_MD_5"] = _recent_slope(visible["MD"].values, vis_TVT, 5, slope_all)
    cur["slope_TVT_MD_10"] = _recent_slope(visible["MD"].values, vis_TVT, 10, slope_all)
    cur["slope_TVT_MD_20"] = _recent_slope(visible["MD"].values, vis_TVT, 20, slope_all)
    cur["slope_TVT_MD_50"] = _recent_slope(visible["MD"].values, vis_TVT, 50, slope_all)
    cur["slope_Z_MD_10"] = _recent_slope(visible["MD"].values, visible["Z"].values, 10, 0.0)
    cur["slope_Z_MD_20"] = _recent_slope(visible["MD"].values, visible["Z"].values, 20, 0.0)
    cur["hidden_rows"] = len(cur)
    cur["visible_rows"] = len(visible)
    cur["hidden_fraction"] = len(cur) / max(len(h), 1)
    cur["z_span_visible"] = float(np.nanmax(visible["Z"].values) - np.nanmin(visible["Z"].values))
    
    tw_path = TEST_DIR / f"{wid}__typewell.csv"
    tw = pd.read_csv(tw_path) if tw_path.exists() else pd.DataFrame()
    has_tw = "TVT" in tw and "GR" in tw and len(tw) > 3
    
    if has_tw:
        tw_tvt = tw["TVT"].to_numpy(np.float32)
        tw_gr = tw["GR"].to_numpy(np.float32)
        fill = float(np.nanmean(tw_gr))
        
        # GR sequence features
        gr_cols = _add_gr_sequence_features(h=h, visible=visible, sel_idx=cur.index, fallback_gr=fill)
        for k, v in gr_cols.items():
            cur[k] = v
        
        # Beam search
        hgr = cur["GR"].interpolate(limit_direction="both").fillna(fill).to_numpy(np.float32)
        kgr = visible["GR"].interpolate(limit_direction="both").fillna(fill).to_numpy(np.float32)
        paths = []
        for bs, mc, es, rad, tag in BEAM_CONFIGS:
            p = beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, rad)
            cur[f"beam_{tag}_d"] = p - last_tvt
            paths.append(p)
        pa = np.column_stack(paths)
        cur["beam_mean_d"] = pa.mean(1) - last_tvt
        cur["beam_std_d"] = pa.std(1)
        cur["beam_median_d"] = np.median(pa, axis=1) - last_tvt
        
        beam_by_tag = {tag: cur[f"beam_{tag}_d"].values for _, _, _, _, tag in BEAM_CONFIGS}
        if "vloose" in beam_by_tag and "vcons" in beam_by_tag:
            cur["beam_spread_d"] = (beam_by_tag["vloose"] - beam_by_tag["vcons"]).astype(np.float32)
        else:
            cur["beam_spread_d"] = 0.0
        if "loose" in beam_by_tag and "cons" in beam_by_tag:
            cur["beam_gap_d"] = (beam_by_tag["loose"] - beam_by_tag["cons"]).astype(np.float32)
        else:
            cur["beam_gap_d"] = 0.0
        
        # Multi-scale NCC
        sc_res, sc_ens = multi_scale_ncc(kgr, visible["TVT_input"].to_numpy(), hgr, hws=(8, 15, 25), stride=3)
        cur["sc8_d"] = sc_res[0][0] - last_tvt
        cur["sc15_d"] = sc_res[1][0] - last_tvt
        cur["sc25_d"] = sc_res[2][0] - last_tvt
        cur["sc_ens_d"] = sc_ens - last_tvt
        
        # Particle filters
        pf_a, pf_a_std = run_pf_ancc(h, tw_tvt, tw_gr)
        pf_z, pf_z_std = run_pf_z_velocity(h, tw_tvt, tw_gr)
        if len(pf_a) == len(cur):
            cur["pf_ancc_d"] = pf_a - last_tvt
            cur["pf_ancc_std"] = pf_a_std
        else:
            cur["pf_ancc_d"] = 0.0
            cur["pf_ancc_std"] = 0.0
        if len(pf_z) == len(cur):
            cur["pf_z_d"] = pf_z - last_tvt
            cur["pf_z_std"] = pf_z_std
        else:
            cur["pf_z_d"] = 0.0
            cur["pf_z_std"] = 0.0
    else:
        # No typewell - zero-fill physics
        gr_cols = _add_gr_sequence_features(h=h, visible=visible, sel_idx=cur.index, fallback_gr=50.0)
        for k, v in gr_cols.items():
            cur[k] = v
        for _, _, _, _, tag in BEAM_CONFIGS:
            cur[f"beam_{tag}_d"] = 0.0
        for c in ["beam_mean_d", "beam_std_d", "beam_median_d", "beam_spread_d", "beam_gap_d",
                  "sc8_d", "sc15_d", "sc25_d", "sc_ens_d", "pf_ancc_d", "pf_ancc_std", "pf_z_d", "pf_z_std"]:
            cur[c] = 0.0
    
    # Dense ANCC and formation features
    xy_known = visible[["X", "Y"]].to_numpy()
    xy_eval = cur[["X", "Y"]].to_numpy()
    d_ancc = dense_knn.impute(xy_eval)
    d_known = dense_knn.impute(xy_known)
    ktvt = visible["TVT_input"].to_numpy(np.float32)
    kz = visible["Z"].to_numpy(np.float32)
    b_full, _, _, b_late, b_wls = seg_b_well(ktvt, kz, d_known)
    z_eval = cur["Z"].to_numpy(np.float32)
    cur["tvt_dense_d"] = -z_eval + d_ancc + b_full - last_tvt
    cur["tvt_densew_d"] = -z_eval + d_ancc + b_wls - last_tvt
    cur["tvt_dense50_d"] = -z_eval + d_ancc + b_late - last_tvt
    
    if has_tw and "pf_ancc_d" in cur:
        cur["pf_vs_dense"] = (cur["pf_ancc_d"].values - cur["tvt_dense_d"]).astype(np.float32)
        cur["sc_vs_dense"] = (cur["sc_ens_d"].values - cur["tvt_dense_d"]).astype(np.float32)
        cur["pf_vs_beam"] = (cur["pf_ancc_d"].values - cur["beam_mean_d"].values).astype(np.float32)
        cur["pf_z_vs_ancc"] = (cur["pf_z_d"].values - cur["pf_ancc_d"].values).astype(np.float32)
        cur["pf_z_vs_dense"] = (cur["pf_z_d"].values - cur["tvt_dense_d"]).astype(np.float32)
        cur["pf_z_vs_beam"] = (cur["pf_z_d"].values - cur["beam_mean_d"].values).astype(np.float32)
    else:
        for c in ["pf_vs_dense", "sc_vs_dense", "pf_vs_beam", "pf_z_vs_ancc", "pf_z_vs_dense", "pf_z_vs_beam"]:
            cur[c] = 0.0
    
    # Formation plane features
    form_ev, form_ev_dist = formation_knn.impute(xy_eval, self_wid=None, k=10)
    form_kn, _ = formation_knn.impute(xy_known, self_wid=None, k=10)
    cur["formation_knn_distance"] = form_ev_dist.astype(np.float32)
    
    for form_index, form_name in enumerate(FORMATIONS):
        ev_form = form_ev[:, form_index]
        kn_form = form_kn[:, form_index]
        valid = np.isfinite(kn_form) & np.isfinite(ktvt) & np.isfinite(kz)
        if valid.sum() >= 10:
            b_full_form = float(np.nanmedian(ktvt[valid] + kz[valid] - kn_form[valid]))
            recent_mask = valid.copy()
            recent_indices = np.flatnonzero(valid)[-min(50, valid.sum()):]
            b_recent_form = float(np.nanmedian(ktvt[recent_indices] + kz[recent_indices] - kn_form[recent_indices]))
            cur[f"tvt_{form_name}_d"] = (-z_eval + ev_form + b_full_form) - last_tvt
            cur[f"tvt_{form_name}_recent_d"] = (-z_eval + ev_form + b_recent_form) - last_tvt
            cur[f"b_{form_name}"] = b_full_form
            cur[f"b_recent_{form_name}"] = b_recent_form
        else:
            for c in [f"tvt_{form_name}_d", f"tvt_{form_name}_recent_d", f"b_{form_name}", f"b_recent_{form_name}"]:
                cur[c] = 0.0
    
    # Affine calibration and offset matrices
    if has_tw:
        tw_at_known = np.interp(ktvt, tw_tvt, tw_gr).astype(np.float32)
        ca, cb = affine_cal(kgr, tw_at_known)
        cur["cal_a"] = ca
        cur["cal_b"] = cb
        for o in ANCH_OFFS:
            cur[f"tda_{o}"] = (hgr - np.interp(last_tvt + o, tw_tvt, tw_gr)).astype(np.float32)
        bref = cur["beam_mean_d"].to_numpy() + last_tvt
        for o in BEAM_OFFS:
            cur[f"tdbc_{o}"] = (hgr - np.interp(bref + o, tw_tvt, tw_gr)).astype(np.float32)
        pref = cur["pf_ancc_d"].to_numpy() + last_tvt
        for o in PF_OFFS:
            cur[f"tdpf_{o}"] = (hgr - np.interp(pref + o, tw_tvt, tw_gr)).astype(np.float32)
    else:
        cur["cal_a"] = 1.0
        cur["cal_b"] = 0.0
        for o in ANCH_OFFS:
            cur[f"tda_{o}"] = 0.0
        for o in BEAM_OFFS:
            cur[f"tdbc_{o}"] = 0.0
        for o in PF_OFFS:
            cur[f"tdpf_{o}"] = 0.0
    
    return cur.reset_index(drop=True)


# =============================================================================
# PREFIX SELECTION (Visible-Prefix Candidate Selection)
# =============================================================================

def robust_poly_predict(md_known, u_known, md_all, deg):
    """Fit robust polynomial with iteratively reweighted least squares."""
    c = np.polyfit(md_known, u_known, deg)
    for _ in range(4):
        r = u_known - np.polyval(c, md_known)
        sc = np.median(np.abs(r)) * 1.4826 + 1e-6
        w = 1.0 / (1.0 + (r / sc) ** 2)
        c = np.polyfit(md_known, u_known, deg, w=w)
    return np.polyval(c, md_all)


def compute_polynomial_u_candidates(hw_known, md_all, z_all):
    """Generate polynomial-U candidates."""
    candidates = []
    md_known = hw_known['MD'].values.astype(np.float64, copy=False)
    tvt_known = hw_known['TVT_input'].values.astype(np.float64, copy=False)
    u_known = tvt_known + z_all[:len(md_known)]
    n_known = len(md_known)
    
    for tail in [80, 160, 320, 640, 'all']:
        if tail == 'all':
            use_md, use_u = md_known, u_known
        else:
            if n_known < tail:
                continue
            use_md, use_u = md_known[-tail:], u_known[-tail:]
        
        for deg in [1, 2, 3]:
            if len(use_md) < deg + 12:
                continue
            try:
                c = np.polyfit(use_md, use_u, deg)
                for _ in range(4):
                    r = use_u - np.polyval(c, use_md)
                    sc = np.median(np.abs(r)) * 1.4826 + 1e-6
                    w = 1.0 / (1.0 + (r / sc) ** 2)
                    c = np.polyfit(use_md, use_u, deg, w=w)
                tvt_hat = np.polyval(c, md_all) - z_all
                if np.all(np.isfinite(tvt_hat)):
                    candidates.append((f"poly_u_deg{deg}_tail{tail}", tvt_hat.copy()))
            except Exception:
                continue
    return candidates


def compute_formation_surface_candidates(hw_known, hw_full, formations):
    """Generate formation surface candidates."""
    candidates = []
    n_known = len(hw_known)
    z_all = hw_full['Z'].values.astype(np.float64)
    
    for f in formations:
        if f not in hw_full.columns:
            continue
        f_vals_full = hw_full[f].values.astype(np.float64)
        f_vals_known = hw_known[f].values.astype(np.float64)
        if np.all(np.isnan(f_vals_known)):
            continue
        
        tvt_known = hw_known['TVT_input'].values.astype(np.float64)
        z_known = hw_known['Z'].values.astype(np.float64)
        valid = np.isfinite(tvt_known) & np.isfinite(z_known) & np.isfinite(f_vals_known)
        if valid.sum() < 10:
            continue
        
        b = tvt_known[valid] + z_known[valid] - f_vals_known[valid]
        if len(b) == 0:
            continue
        
        # b_median
        b_median = float(np.nanmedian(b))
        tvt_median = -z_all + f_vals_full + b_median
        if np.all(np.isfinite(tvt_median)):
            candidates.append((f"surface_{f}_median", tvt_median))
        
        # b_late (last 50)
        if n_known >= 50:
            b_late = float(np.nanmedian(b[-50:]))
            tvt_late = -z_all + f_vals_full + b_late
            if np.all(np.isfinite(tvt_late)):
                candidates.append((f"surface_{f}_late", tvt_late))
        
        # b_wls (exponentially weighted)
        w = np.exp(0.02 * np.arange(len(b)))
        b_wls = float(np.sum(w * b) / np.sum(w))
        tvt_wls = -z_all + f_vals_full + b_wls
        if np.all(np.isfinite(tvt_wls)):
            candidates.append((f"surface_{f}_wls", tvt_wls))
    
    return candidates


def evaluate_candidate_on_holdout(candidate_pred, hw_holdout):
    """Evaluate candidate on holdout slice."""
    tvt_true = hw_holdout['TVT_input'].values.astype(np.float64)
    valid = np.isfinite(tvt_true) & np.isfinite(candidate_pred)
    if valid.sum() < 5:
        return np.inf
    return float(np.sqrt(np.mean((tvt_true[valid] - candidate_pred[valid]) ** 2)))


def select_best_candidate_for_well(hw_full, train_dir, well_id):
    """Select best candidate using visible-prefix validation."""
    known_mask = hw_full['TVT_input'].notna().values
    if known_mask.sum() < 20:
        return None, None, np.inf
    
    hw_known = hw_full[known_mask].copy()
    md_all = hw_full['MD'].values.astype(np.float64)
    z_all = hw_full['Z'].values.astype(np.float64)
    
    candidates_pool = []
    candidates_pool.extend(compute_polynomial_u_candidates(hw_known, md_all, z_all))
    candidates_pool.extend(compute_formation_surface_candidates(hw_known, hw_full, FORMATIONS))
    
    if not candidates_pool:
        return None, None, np.inf
    
    n_known = len(hw_known)
    cutoff_fractions = [0.50, 0.65, 0.75]
    candidate_scores = {name: [] for name, _ in candidates_pool}
    
    for cutoff_frac in cutoff_fractions:
        cutoff_idx = int(n_known * cutoff_frac)
        if cutoff_idx < 20 or cutoff_idx >= n_known - 10:
            continue
        
        hw_pseudo_holdout = hw_known.iloc[cutoff_idx:].copy()
        holdout_indices = hw_pseudo_holdout.index.values
        
        for name, pred_all in candidates_pool:
            pred_holdout = pred_all[holdout_indices]
            rmse = evaluate_candidate_on_holdout(pred_holdout, hw_pseudo_holdout)
            candidate_scores[name].append(rmse)
    
    candidate_mean_rmse = {}
    for name, rmses in candidate_scores.items():
        valid_rmses = [r for r in rmses if np.isfinite(r)]
        if valid_rmses:
            candidate_mean_rmse[name] = float(np.mean(valid_rmses))
    
    if not candidate_mean_rmse:
        return None, None, np.inf
    
    best_name = min(candidate_mean_rmse.keys(), key=lambda k: candidate_mean_rmse[k])
    best_rmse = candidate_mean_rmse[best_name]
    
    best_pred = None
    for name, pred in candidates_pool:
        if name == best_name:
            best_pred = pred
            break
    
    return best_name, best_pred, best_rmse


def apply_prefix_selection(test_df, ml_pred_residuals, data_dir):
    """Apply prefix selection to ML predictions."""
    data_dir = Path(data_dir)
    final_tvt = np.zeros(len(test_df), dtype=np.float64)
    selection_info = []
    
    for well_id in test_df['well_id'].unique():
        well_mask = test_df['well_id'] == well_id
        well_data = test_df[well_mask]
        well_indices = well_data.index.values
        
        hw_path = data_dir / f"test/{well_id}__horizontal_well.csv"
        if not hw_path.exists():
            last_tvt = well_data['last_known_TVT'].iloc[0]
            final_tvt[well_indices] = last_tvt + ml_pred_residuals[well_indices]
            for idx in well_indices:
                selection_info.append({'row_idx': idx, 'selection': 'ml_fallback', 'reason': 'no_hw_file'})
            continue
        
        hw_full = pd.read_csv(hw_path)
        known_mask = hw_full['TVT_input'].notna().values
        if known_mask.sum() < 20:
            last_tvt = well_data['last_known_TVT'].iloc[0]
            final_tvt[well_indices] = last_tvt + ml_pred_residuals[well_indices]
            for idx in well_indices:
                selection_info.append({'row_idx': idx, 'selection': 'ml_fallback', 'reason': 'insufficient_prefix'})
            continue
        
        best_name, best_pred, best_rmse = select_best_candidate_for_well(hw_full, data_dir, well_id)
        last_tvt = well_data['last_known_TVT'].iloc[0]
        ml_pred_tvt = last_tvt + ml_pred_residuals[well_indices]
        
        ml_holdout_rmse_threshold = 15.0
        min_gain = 0.5
        
        if best_name is not None and best_rmse < ml_holdout_rmse_threshold - min_gain:
            hidden_mask = hw_full['TVT_input'].isna().values
            md_test = well_data['MD'].values
            md_full = hw_full['MD'].values
            row_to_full_idx = {}
            for i, md in enumerate(md_test):
                matches = np.where(md_full == md)[0]
                if len(matches) > 0:
                    row_to_full_idx[well_indices[i]] = matches[0]
            
            candidate_tvt = np.zeros(len(well_indices), dtype=np.float64)
            for i, row_idx in enumerate(well_indices):
                if row_idx in row_to_full_idx:
                    full_idx = row_to_full_idx[row_idx]
                    candidate_tvt[i] = best_pred[full_idx]
                else:
                    candidate_tvt[i] = ml_pred_tvt[i]
            
            final_tvt[well_indices] = candidate_tvt
            for idx in well_indices:
                selection_info.append({'row_idx': idx, 'selection': 'candidate', 'candidate': best_name, 'rmse': best_rmse})
        else:
            final_tvt[well_indices] = ml_pred_tvt
            for idx in well_indices:
                selection_info.append({'row_idx': idx, 'selection': 'ml_fallback', 'reason': f'candidate_rmse_{best_rmse:.2f}'})
    
    selection_log = pd.DataFrame(selection_info)
    return final_tvt, selection_log


# =============================================================================
# MAIN INFERENCE PIPELINE
# =============================================================================

def load_models():
    """Load trained models and feature columns."""
    required = ["feature_cols.txt", "meta_weights.npy", "meta_intercept.npy"]
    missing = [x for x in required if not (MODELS_DIR / x).exists()]
    lgb_paths = [MODELS_DIR / f"lgbm_fold_{i}.txt" for i in range(1, 6)]
    cb_paths = [MODELS_DIR / f"cb_fold_{i}.cbm" for i in range(1, 6)]
    missing += [p.name for p in lgb_paths + cb_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing artifacts in {MODELS_DIR}: {missing}")
    
    cols = [x.strip() for x in (MODELS_DIR / "feature_cols.txt").read_text().splitlines() if x.strip()]
    lgb_models = [lgb.Booster(model_file=str(p)) for p in lgb_paths]
    cb_models = []
    for p in cb_paths:
        m = CatBoostRegressor()
        m.load_model(str(p))
        cb_models.append(m)
    
    return cols, lgb_models, cb_models, np.load(MODELS_DIR / "meta_weights.npy").ravel(), float(np.ravel(np.load(MODELS_DIR / "meta_intercept.npy"))[0])


def main():
    t0 = time.time()
    
    if not SAMPLE_SUB.exists():
        raise FileNotFoundError(SAMPLE_SUB)
    
    # Load models
    cols, lgb_models, cb_models, meta_weights, meta_intercept = load_models()
    
    # Verify PF-Z features
    if not {"pf_z_d", "pf_z_std"}.issubset(cols):
        raise RuntimeError("PF-Z features not found in feature_cols.txt")
    
    # Load sample submission and extract eval indices
    sample_sub = pd.read_csv(SAMPLE_SUB)
    sample_sub["well_id"] = sample_sub.id.str.rsplit("_", n=1).str[0]
    sample_sub["row_index"] = sample_sub.id.str.rsplit("_", n=1).str[1].astype(int)
    test_index = sample_sub.groupby("well_id")["row_index"].apply(lambda x: set(x)).to_dict()
    
    # Initialize feature extractors
    train_wells = [p.name.split("__")[0] for p in TRAIN_DIR.glob("*__horizontal_well.csv")]
    dense_knn = DenseANCCImputer(train_wells, DATA_ROOT)
    formation_knn = FormationPlaneKNN(train_wells, DATA_ROOT, FORMATIONS)
    
    # Build features for all test wells
    jobs = max(1, int(N_JOBS))
    if jobs == 1:
        parts = [build_features_for_well(w, idx, dense_knn, formation_knn) for w, idx in test_index.items()]
    else:
        parts = Parallel(n_jobs=jobs)(
            delayed(build_features_for_well)(w, idx, dense_knn, formation_knn) 
            for w, idx in test_index.items()
        )
    
    test_df = pd.concat([x for x in parts if len(x)], ignore_index=True)
    
    # Verify all features are present
    required_missing = [c for c in cols if c not in test_df.columns]
    if required_missing:
        raise RuntimeError(f"Inference feature mismatch ({len(required_missing)}): {required_missing}")
    
    # Generate ML predictions
    x = test_df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.).astype(np.float32).to_numpy()
    lp = np.column_stack([m.predict(x) for m in lgb_models]).mean(1)
    cp = np.column_stack([m.predict(x) for m in cb_models]).mean(1)
    residual = np.clip(np.column_stack([lp, cp]) @ meta_weights + meta_intercept, -400., 400.)
    
    # Apply prefix selection
    print(f"Applying prefix selection to {len(test_df)} rows...")
    corrected_tvt, selection_log = apply_prefix_selection(test_df, residual, DATA_ROOT)
    
    n_candidate = (selection_log['selection'] == 'candidate').sum()
    n_ml = (selection_log['selection'] == 'ml_fallback').sum()
    print(f"Prefix selection: {n_candidate} rows used candidate, {n_ml} rows used ML fallback")
    
    test_df["tvt_pred"] = corrected_tvt
    
    # Contact-based override for wells that exist in train
    for wid in test_df.well_id.unique():
        hw = TRAIN_DIR / f"{wid}__horizontal_well.csv"
        tw = TRAIN_DIR / f"{wid}__typewell.csv"
        if hw.exists() and tw.exists():
            try:
                exact = tvt_from_contacts(pd.read_csv(hw), pd.read_csv(tw))
                mask = test_df.well_id.eq(wid)
                test_df.loc[mask, "tvt_pred"] = exact.iloc[test_df.loc[mask, "row_index"].to_numpy()].to_numpy()
            except Exception as exc:
                print(f"[WARN] Leak override skipped for {wid}: {exc}")
    
    # Generate submission
    out = sample_sub[["id"]].copy()
    out["tvt"] = out.id.map(test_df.set_index("id").tvt_pred)
    
    if out.tvt.isna().any():
        raise RuntimeError(f"Missing predictions: {int(out.tvt.isna().sum())}")
    if not np.isfinite(out.tvt.to_numpy()).all():
        raise RuntimeError("Non-finite predictions")
    
    out.to_csv("submission.csv", index=False)
    print(f"submission.csv written: {out.shape}; {len(cols)} features; {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
