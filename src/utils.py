# src/utils.py
import logging
import numpy as np

def setup_logger(name):
    """Creates a standardized logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def robust_slope(x, y):
    """Calculates a robust slope using median-based estimation."""
    if len(x) < 2 or len(y) < 2:
        return 0.0
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return 0.0
    x_med = np.median(x)
    y_med = np.median(y)
    num = np.sum((x - x_med) * (y - y_med))
    den = np.sum((x - x_med) ** 2)
    if den == 0:
        return 0.0
    return float(num / den)

def first_nan_idx(arr):
    """Returns the index of the first NaN value in the array."""
    arr = np.asarray(arr)
    nans = np.isnan(arr)
    if not nans.any():
        return len(arr)
    return int(np.argmax(nans))

def fill_and_smooth_gr(gr, window=5):
    """Fills NaNs in Gamma Ray log and applies smoothing."""
    gr = np.asarray(gr, dtype=np.float64).copy()
    median_val = np.nanmedian(gr)
    if np.isnan(median_val):
        median_val = 50.0
    gr[np.isnan(gr)] = median_val
    if window > 1:
        kernel = np.ones(window) / window
        gr = np.convolve(gr, kernel, mode="same")
    return gr
