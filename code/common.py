"""Shared helpers: image loading, vegetation masks, handcrafted features, metrics."""
import json
import os
import random

import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "outputs")
MANIFEST = os.path.join(DATA_DIR, "manifest.csv")
SPLITS = os.path.join(DATA_DIR, "splits.csv")
FEATURES = os.path.join(DATA_DIR, "handcrafted_features.csv")
PRED_DIR = os.path.join(OUT_DIR, "predictions")


def ensure_dirs():
    for d in (DATA_DIR, OUT_DIR, PRED_DIR):
        os.makedirs(d, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ----------------------------------------------------------------------------
# Images
# ----------------------------------------------------------------------------
def load_rgb(path):
    """Return (rgb float32 HxWx3 in [0,1], valid bool HxW).

    `valid` marks pixels inside the plot. Cut-plot images can carry an alpha
    channel or a black/zero background outside the plot polygon; those pixels
    are excluded from features and masks.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        try:
            import tifffile
            arr = tifffile.imread(path)
        except Exception:
            arr = np.array(Image.open(path))
    else:
        arr = np.array(Image.open(path))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):  # channels-first
        arr = np.moveaxis(arr, 0, -1)
    alpha = None
    if arr.shape[-1] == 4:
        alpha = arr[..., 3]
        arr = arr[..., :3]
    arr = arr.astype(np.float32)
    maxv = 255.0 if arr.max() <= 255 else (65535.0 if arr.max() <= 65535 else arr.max())
    rgb = np.clip(arr / maxv, 0, 1)
    valid = rgb.sum(-1) > 1e-6
    if alpha is not None:
        valid &= alpha > 0
    if valid.sum() < 10:  # fully dark image: treat everything as valid
        valid[:] = True
    return rgb, valid


def chromatic(rgb):
    s = rgb.sum(-1, keepdims=True) + 1e-6
    c = rgb / s
    return c[..., 0], c[..., 1], c[..., 2]


def veg_mask(rgb, valid):
    """Vegetation mask by ExG - ExR > 0 (Meyer & Neto zero threshold), fixed, no tuning."""
    r, g, b = chromatic(rgb)
    exg = 2 * g - r - b
    exr = 1.4 * r - g
    return (exg - exr > 0) & valid


# ----------------------------------------------------------------------------
# Handcrafted features (baseline for RF / XGBoost)
# ----------------------------------------------------------------------------
def _stats(name, x):
    if x.size == 0:
        x = np.zeros(1, dtype=np.float32)
    return {
        f"{name}_mean": float(np.mean(x)),
        f"{name}_std": float(np.std(x)),
        f"{name}_p10": float(np.percentile(x, 10)),
        f"{name}_p50": float(np.percentile(x, 50)),
        f"{name}_p90": float(np.percentile(x, 90)),
    }


def handcrafted_features(path):
    from skimage.color import rgb2hsv
    from skimage.feature import graycomatrix, graycoprops

    rgb, valid = load_rgb(path)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    r, g, b = chromatic(rgb)
    eps = 1e-6
    idx = {
        "ExG": 2 * g - r - b,
        "ExR": 1.4 * r - g,
        "ExGR": (2 * g - r - b) - (1.4 * r - g),
        "VARI": np.clip((G - R) / (G + R - B + eps), -2, 2),
        "GLI": (2 * G - R - B) / (2 * G + R + B + eps),
        "NGRDI": (G - R) / (G + R + eps),
    }
    feats = {}
    for k, v in idx.items():
        feats.update(_stats(k, v[valid]))
    hsv = rgb2hsv(rgb)
    for i, ch in enumerate(["R", "G", "B"]):
        feats[f"{ch}_mean"] = float(rgb[..., i][valid].mean())
        feats[f"{ch}_std"] = float(rgb[..., i][valid].std())
    for i, ch in enumerate(["H", "S", "V"]):
        feats[f"{ch}_mean"] = float(hsv[..., i][valid].mean())
        feats[f"{ch}_std"] = float(hsv[..., i][valid].std())
    vm = veg_mask(rgb, valid)
    feats["cover_fraction"] = float(vm.sum() / max(valid.sum(), 1))
    # GLCM texture on grey levels (32 levels), averaged over 4 angles and distances 1,2
    gray = (0.299 * R + 0.587 * G + 0.114 * B)
    q = np.clip((gray * 31).round(), 0, 31).astype(np.uint8)
    glcm = graycomatrix(q, distances=[1, 2], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=32, symmetric=True, normed=True)
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
        feats[f"glcm_{prop}"] = float(np.nanmean(graycoprops(glcm, prop)))
    return feats


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def metrics(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    err = p - y
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    pear = float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 and np.std(y) > 0 else float("nan")
    # Lin's concordance correlation coefficient
    ccc = float(2 * np.cov(y, p, bias=True)[0, 1] / (y.var() + p.var() + (y.mean() - p.mean()) ** 2))
    pos = y > 0
    mape = float(np.mean(np.abs(err[pos] / y[pos])) * 100) if pos.all() else float("nan")
    return {
        "n": int(len(y)), "RMSE": rmse, "MAE": mae, "R2": r2,
        "nRMSE_%": rmse / float(y.mean()) * 100, "MAPE_%": mape,
        "bias": float(err.mean()), "pearson_r": pear, "CCC": ccc,
    }


def clip_pred(pred, y_train):
    """Clip back-transformed predictions to [0, 1.5 x max training biomass].

    Log-target models can extrapolate explosively (expm1) on out-of-range inputs; the same rule
    is applied to every model so comparisons stay fair.
    """
    return np.clip(pred, 0, 1.5 * float(np.max(y_train)))


def load_manifest():
    m = pd.read_csv(MANIFEST)
    m["image_path"] = m["image_path"].apply(
        lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p))
    return m


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
