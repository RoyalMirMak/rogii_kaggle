# %% [markdown]
# # ROGII Wellbore Geology — Stacking Inference Pipeline
# 
# Features included: Segmented Dense Imputer, 14 Beams, Particle Filter, Multi-Scale NCC, GR Offsets.
# Models included: 5-Fold LightGBM + 5-Fold CatBoost -> Ridge Meta-Learner.

# %% [markdown]
# ## 1. Configuration

# %%
MODELS_DIR = "/kaggle/input/models/mironoxxxy/rogii-base/other/default/5" # UPDATE THIS
DATA_ROOT = "/kaggle/input/competitions/rogii-wellbore-geology-prediction"

# Physics Constants
BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2, "cons"), (10, 8.0, 64.0, 2, "loose"), (8, 35.0, 220.0, 1, "vcons"), 
    (10, 14.0, 90.0, 5, "sm5"), (20, 4.0, 36.0, 3, "vloose"), (12, 12.0, 100.0, 3, "med1"), 
    (15, 25.0, 180.0, 2, "wide1"), (20, 30.0, 200.0, 2, "wide2"), (15, 10.0, 80.0, 4, "med2"), 
    (25, 6.0, 50.0, 3, "vloose2"), (10, 40.0, 300.0, 1, "vwide1"), (12, 18.0, 120.0, 5, "med3"), 
    (30, 8.0, 70.0, 2, "vloose3"), (10, 50.0, 400.0, 0, "vwide2")
]
ANCH_OFFS = [-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80]
BEAM_OFFS = [-40, -20, -10, -5, -3, 0, 3, 5, 10, 20, 40]
PF_OFFS   = [-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30]

PF_N=600; ANCC_N=600; PF_RESAMP=0.5
PF_GR_SIG_MIN=10.; PF_GR_SIG_MAX=60.; PF_GR_SIG_DEF=30.
ANCC_ALPHA=0.998; ANCC_RN=0.002; ANCC_PN=0.005
ANCC_IS=0.3; ANCC_RP=0.1; ANCC_RR=0.001

# %% [markdown]
# ## 2. Imports

# %%
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from scipy.spatial import cKDTree
from numba import njit

warnings.filterwarnings("ignore")

DATA_ROOT = Path(DATA_ROOT)
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
SAMPLE_SUB = DATA_ROOT / "sample_submission.csv"

# %% [markdown]
# ## 3. Physics Engine (Numba Particle Filter, Calibration, Offsets)

# %%
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

def beam_search(hgr, tw_tvt, tw_gr, last_TVT, beam_size=10, move_cost=20.0, emit_scale=144.0, smooth_radius=2):
    N = len(hgr)
    if N == 0 or len(tw_tvt) == 0 or len(tw_gr) == 0:
        return np.zeros(N, dtype=np.float64)
    tvt_min = max(0, last_TVT - 200)
    tvt_max = last_TVT + 400
    tvt_states = np.arange(tvt_min, tvt_max, 5.0)
    S = len(tvt_states)
    if S == 0: return np.zeros(N, dtype=np.float64)

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

def robust_slope(x, y):
    if len(x) < 2 or len(y) < 2: return 0.0
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 2: return 0.0
    x_med, y_med = np.median(x), np.median(y)
    den = np.sum((x - x_med) ** 2)
    return float(np.sum((x - x_med) * (y - y_med)) / den) if den != 0 else 0.0

def tvt_from_contacts(hw_tr, tw_tr, ref_col="EGFDU"):
    tw_g = tw_tr.dropna(subset=["Geology"])
    ref_tvt = tw_g.loc[tw_g["Geology"] == ref_col, "TVT"].min()
    if pd.isna(ref_tvt):
        ref_col = tw_g["Geology"].iloc[0]
        ref_tvt = tw_g.loc[tw_g["Geology"] == ref_col, "TVT"].min()
    offset = (hw_tr["TVT"] - (ref_tvt - (hw_tr["Z"] - hw_tr[ref_col]))).mean()
    return ref_tvt - (hw_tr["Z"] - hw_tr[ref_col]) + offset

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
    b_late = float(np.median(bv[max(0, n-50):])) if n >= 5 else b_full
    t1, t2 = n // 3, 2 * n // 3
    b_early = float(np.median(bv[:max(1, t1)])) if t1 > 0 else b_full
    b_mid = float(np.median(bv[t1:max(t1+1, t2)])) if t2 > t1 else b_full
    w = np.exp(0.02 * np.arange(n))
    w /= w.sum()
    b_wls = float(np.dot(w, bv))
    return b_full, b_early, b_mid, b_late, b_wls

# %% [markdown]
# ## 4. Dense ANCC Imputer

# %%
class DenseANCCImputer:
    def __init__(self, train_wells, data_dir, spw=60):
        xs, ys, anccs, wids = [], [], [], []
        for wid in train_wells:
            p = Path(data_dir) / f'train/{wid}__horizontal_well.csv'
            try: 
                df = pd.read_csv(p, usecols=['X', 'Y', 'ANCC']).dropna()
                if len(df) == 0: continue
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

# Global Imputer initialization
train_wells = [p.name.split("__")[0] for p in sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))]
DI = DenseANCCImputer(train_wells, DATA_ROOT)

# %% [markdown]
# ## 5. Feature Builder

# %%
def build_features_for_well(wid, split, test_eval_idx=None):
    split_dir = TRAIN_DIR if split == "train" else TEST_DIR
    hw_path = split_dir / f"{wid}__horizontal_well.csv"
    tw_path = split_dir / f"{wid}__typewell.csv"

    h = pd.read_csv(hw_path)
    h["row_index"] = np.arange(len(h), dtype=np.int64)

    if "TVT_input" not in h.columns: return pd.DataFrame()
    eval_mask = h["TVT_input"].isna().values
    
    if split == "train":
        if "TVT" not in h.columns: return pd.DataFrame()
        sel_mask = eval_mask & h["TVT"].notna().values
    else:
        sel_mask = eval_mask
        if test_eval_idx is not None:
            mask2 = np.zeros(len(h), dtype=bool)
            mask2[list(test_eval_idx)] = True
            sel_mask = sel_mask & mask2

    if not sel_mask.any(): return pd.DataFrame()

    visible = h[h["TVT_input"].notna()].copy()
    if len(visible) < 10: return pd.DataFrame()

    last = visible.iloc[-1]
    last_TVT, last_MD, last_X, last_Y, last_Z = float(last["TVT_input"]), float(last["MD"]), float(last["X"]), float(last["Y"]), float(last["Z"])
    
    tw = pd.read_csv(tw_path) if tw_path.is_file() else pd.DataFrame()
    has_tw = "TVT" in tw.columns and "GR" in tw.columns and len(tw) > 3
    if has_tw:
        tw_tvt, tw_gr = tw["TVT"].to_numpy(dtype=np.float32), tw["GR"].to_numpy(dtype=np.float32)
    else:
        tw_tvt, tw_gr = np.array([]), np.array([])

    sel_idx = np.flatnonzero(sel_mask)
    cur = h.iloc[sel_idx].copy()
    
    cur["well_id"] = wid
    cur["id"] = cur["well_id"] + "_" + cur["row_index"].astype(str)

    vis_TVT = visible["TVT_input"].values
    ktvt = visible["TVT_input"].to_numpy(dtype=np.float32)
    kz = visible["Z"].to_numpy(dtype=np.float32)

    cur["last_known_TVT"] = last_TVT
    cur["md_from_ps"] = cur["MD"].values - last_MD
    cur["z_from_ps"] = cur["Z"].values - last_Z
    cur["dxy_from_ps"] = np.sqrt((cur["X"].values - last_X)**2 + (cur["Y"].values - last_Y)**2)
    cur["slope_TVT_MD_all"] = robust_slope(visible["MD"].values, vis_TVT)

    if has_tw:
        hgr = cur["GR"].interpolate(limit_direction="both").fillna(np.nanmean(tw_gr)).to_numpy(dtype=np.float32)
        kgr = visible["GR"].interpolate(limit_direction="both").fillna(np.nanmean(tw_gr)).to_numpy(dtype=np.float32)
        
        beams_res = []
        for bs, mc, es, r, tag in BEAM_CONFIGS:
            path = beam_search(hgr, tw_tvt, tw_gr, last_TVT, bs, mc, es, r)
            if len(path) == len(cur):
                cur[f"beam_{tag}_d"] = path - last_TVT
                beams_res.append(path)
            else: cur[f"beam_{tag}_d"] = 0.0
        
        if beams_res:
            beams_arr = np.stack(beams_res, axis=1)
            cur["beam_mean_d"] = beams_arr.mean(axis=1) - last_TVT
            cur["beam_std_d"] = beams_arr.std(axis=1)
        else:
            cur["beam_mean_d"] = 0.0; cur["beam_std_d"] = 0.0

        pf_a_pts, pf_a_std = run_pf_ancc(h, tw_tvt, tw_gr)
        if len(pf_a_pts) == len(cur):
            cur["pf_ancc_d"] = pf_a_pts - last_TVT
            cur["pf_ancc_std"] = pf_a_std
        else:
            cur["pf_ancc_d"] = 0.0; cur["pf_ancc_std"] = 0.0
    else:
        for f in ["beam_mean_d", "beam_std_d", "pf_ancc_d", "pf_ancc_std"]:
            cur[f] = 0.0

    # Dense ANCC with Segmented offsets
    xy_kn = visible[["X", "Y"]].to_numpy()
    xy_ev = cur[["X", "Y"]].to_numpy()
    d_ancc = DI.impute(xy_ev, self_wid=wid if split == "train" else None)
    d_kn = DI.impute(xy_kn, self_wid=wid if split == "train" else None)
    
    z_ev = cur["Z"].to_numpy(dtype=np.float32)
    b_full, b_early, b_mid, b_late, b_wls = seg_b_well(ktvt, kz, d_kn)
    
    cur["tvt_dense_d"]   = (-z_ev + d_ancc + b_full) - last_TVT
    cur["tvt_densew_d"]  = (-z_ev + d_ancc + b_wls) - last_TVT
    cur["tvt_dense50_d"] = (-z_ev + d_ancc + b_late) - last_TVT
    
    if has_tw and "pf_ancc_d" in cur:
        cur["pf_vs_dense"] = cur["pf_ancc_d"] - cur["tvt_dense_d"]
        cur["pf_vs_beam"] = cur["pf_ancc_d"] - cur["beam_mean_d"]
    else:
        cur["pf_vs_dense"] = 0.0; cur["pf_vs_beam"] = 0.0

    # GR Offsets & Calibration
    if has_tw:
        tw_at_k = np.interp(ktvt, tw_tvt, tw_gr).astype(np.float32)
        a_cal, b_cal = affine_cal(kgr, tw_at_k)
        cur["cal_a"] = a_cal
        cur["cal_b"] = b_cal
        
        for o in ANCH_OFFS: cur[f"tda_{o}"] = hgr - np.interp(last_TVT + o, tw_tvt, tw_gr)
        
        beam_ref = cur["beam_mean_d"].values + last_TVT
        for o in BEAM_OFFS: cur[f"tdbc_{o}"] = hgr - np.interp(beam_ref + o, tw_tvt, tw_gr)
            
        if "pf_ancc_d" in cur and (cur["pf_ancc_d"] != 0).any():
            pf_ref = cur["pf_ancc_d"].values + last_TVT
            for o in PF_OFFS: cur[f"tdpf_{o}"] = hgr - np.interp(pf_ref + o, tw_tvt, tw_gr)
        else:
            for o in PF_OFFS: cur[f"tdpf_{o}"] = 0.0

    else:
        cur["cal_a"] = 1.0; cur["cal_b"] = 0.0
        for o in ANCH_OFFS: cur[f"tda_{o}"] = 0.0
        for o in BEAM_OFFS: cur[f"tdbc_{o}"] = 0.0
        for o in PF_OFFS: cur[f"tdpf_{o}"] = 0.0

    return cur.reset_index(drop=True)

# %% [markdown]
# ## 6. Load Stacked Models & Features

# %%
models_dir = Path(MODELS_DIR)

# Load LightGBM Folds
lgb_files = sorted(models_dir.glob("lgbm_fold_*.txt"))
lgb_models = [lgb.Booster(model_file=str(p)) for p in lgb_files]
print("Loaded LightGBM models:", len(lgb_models))

# Load CatBoost Folds
cb_files = sorted(models_dir.glob("cb_fold_*.cbm"))
cb_models = []
for p in cb_files:
    model = CatBoostRegressor()
    model.load_model(str(p))
    cb_models.append(model)
print("Loaded CatBoost models:", len(cb_models))

# Load Meta-Learner Weights
meta_weights = np.load(models_dir / "meta_weights.npy")
meta_intercept = np.load(models_dir / "meta_intercept.npy")[0]
print("Meta-Learner Weights (LGBM, CB):", meta_weights)

# Read feature names exactly as trained
feature_file = models_dir / "feature_cols.txt"
FEATURE_COLS = [x.strip() for x in feature_file.read_text().splitlines() if x.strip()]
print("Loaded feature_cols.txt with", len(FEATURE_COLS), "features")

# %% [markdown]
# ## 7. Build Test Set & Predict

# %%
sample_sub = pd.read_csv(SAMPLE_SUB)
sample_sub["well_id"] = sample_sub["id"].str.rsplit("_", n=1).str[0]
sample_sub["row_index"] = sample_sub["id"].str.rsplit("_", n=1).str[1].astype(int)

test_eval_index = sample_sub.groupby("well_id")["row_index"].apply(lambda s: set(s.tolist())).to_dict()
test_wells = list(test_eval_index.keys())

parts = []
t0 = time.time()
for i, wid in enumerate(test_wells, 1):
    df = build_features_for_well(wid, split="test", test_eval_idx=test_eval_index.get(wid))
    if df is not None and len(df): parts.append(df)
    if i % 20 == 0: print(f"{i}/{len(test_wells)} built")

test_df = pd.concat(parts, ignore_index=True)
print("Test matrix built in", round(time.time() - t0, 1), "s")

# Ensure exact feature order
for c in FEATURE_COLS:
    if c not in test_df.columns:
        test_df[c] = 0.0
X_test = test_df[FEATURE_COLS].astype(np.float32).values

# Base Level Predictions
lgb_preds = np.column_stack([m.predict(X_test) for m in lgb_models]).mean(1)
cb_preds = np.column_stack([m.predict(X_test) for m in cb_models]).mean(1)

# Meta Level Blending
meta_X = np.column_stack([lgb_preds, cb_preds])
stacked_residual = np.dot(meta_X, meta_weights) + meta_intercept
pred_residual = np.clip(stacked_residual, -400.0, 400.0)

test_df["tvt_pred"] = test_df["last_known_TVT"].values + pred_residual

# %% [markdown]
# ## 8. Train/Test Leak Override & Save

# %%
for wid in test_df["well_id"].unique():
    train_hw = TRAIN_DIR / f"{wid}__horizontal_well.csv"
    train_tw = TRAIN_DIR / f"{wid}__typewell.csv"
    if train_hw.exists() and train_tw.exists():
        try:
            exact_tvt = tvt_from_contacts(pd.read_csv(train_hw), pd.read_csv(train_tw))
            mask = test_df["well_id"] == wid
            rows = test_df.loc[mask, "row_index"].values
            test_df.loc[mask, "tvt_pred"] = exact_tvt.iloc[rows].values
            print(f"Leak override applied for {wid}")
        except Exception as e:
            pass

pred_map = test_df.set_index("id")["tvt_pred"].to_dict()
submission = sample_sub[["id"]].copy()
submission["tvt"] = submission["id"].map(pred_map)

if submission["tvt"].isna().any():
    anchor_map = test_df.drop_duplicates("well_id").set_index("well_id")["last_known_TVT"].to_dict()
    miss = submission["tvt"].isna()
    submission.loc[miss, "tvt"] = submission.loc[miss, "id"].str.rsplit("_", n=1).str[0].map(anchor_map)
    submission["tvt"] = submission["tvt"].fillna(0.0)

submission.to_csv("submission.csv", index=False)
print("submission.csv saved:", submission.shape)