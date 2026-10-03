"""Step 3: reference and classical machine-learning baselines on handcrafted features.

Models:
  mean          - predicts the training-fold mean (floor any model must beat)
  cover_linear  - linear regression on vegetation cover fraction only
  ridge         - ridge regression on all handcrafted features
  rf            - random forest on all handcrafted features
  xgb           - XGBoost on all handcrafted features
Hyperparameters are tuned with an inner cross-validation inside each training fold
(grouped by plot for the grouped protocol), so test folds are never used for tuning.
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from common import FEATURES, OUT_DIR, PRED_DIR, SPLITS, clip_pred, ensure_dirs, load_manifest

warnings.filterwarnings("ignore")

GRIDS = {
    "ridge": (lambda s: make_pipeline(StandardScaler(), Ridge()),
              {"ridge__alpha": [0.1, 1, 3, 10, 30, 100, 300]}),
    "rf": (lambda s: RandomForestRegressor(n_estimators=300, random_state=s, n_jobs=-1),
           {"max_features": [0.33, 1.0], "min_samples_leaf": [1, 5]}),
    "xgb": (lambda s: XGBRegressor(n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                   random_state=s, n_jobs=-1, verbosity=0),
            {"max_depth": [2, 4], "min_child_weight": [1, 5]}),
}


def main(models, protocols, seeds):
    ensure_dirs()
    m = load_manifest()
    sp = pd.read_csv(SPLITS)
    F = pd.read_csv(FEATURES).sort_values("row")
    feat_cols = [c for c in F.columns if c != "row"]
    X = F[feat_cols].values
    y = m.biomass.values.astype(float)
    groups = m.plot_id.astype(str).values
    importances = []

    for proto in protocols:
        for s in seeds:
            col = f"{proto}_s{s}"
            if col not in sp:
                print(f"skip {col} (not in splits)")
                continue
            folds = sp[col].values
            for name in models:
                t0 = time.time()
                pred = np.zeros(len(y))
                for k in np.unique(folds):
                    tr, te = folds != k, folds == k
                    if name == "mean":
                        pred[te] = y[tr].mean()
                        continue
                    if name == "cover_linear":
                        ci = feat_cols.index("cover_fraction")
                        lr = LinearRegression().fit(X[tr][:, [ci]], y[tr])
                        pred[te] = clip_pred(lr.predict(X[te][:, [ci]]), y[tr])
                        continue
                    make, grid = GRIDS[name]
                    if proto == "random":
                        inner = KFold(4, shuffle=True, random_state=s)
                        fit_kw = {}
                    else:
                        inner = GroupKFold(4)
                        fit_kw = {"groups": groups[tr]}
                    gs = GridSearchCV(make(s), grid, cv=inner, scoring="neg_root_mean_squared_error", n_jobs=1)
                    gs.fit(X[tr], np.log1p(y[tr]), **fit_kw)
                    pred[te] = clip_pred(np.expm1(gs.predict(X[te])), y[tr])
                    if proto == "grouped" and s == seeds[0] and name in ("rf", "xgb"):
                        est = gs.best_estimator_
                        importances.append(pd.Series(est.feature_importances_, index=feat_cols, name=f"{name}_f{k}"))
                out = pd.DataFrame({"row": np.arange(len(y)), "y": y, "pred": pred, "fold": folds})
                out.to_csv(os.path.join(PRED_DIR, f"{name}__{col}.csv"), index=False)
                rmse = np.sqrt(np.mean((pred - y) ** 2))
                print(f"{name:13s} {col:12s} RMSE={rmse:9.2f}  ({time.time() - t0:.1f}s)")
    if importances:
        imp = pd.concat(importances, axis=1)
        imp["mean"] = imp.mean(axis=1)
        imp.sort_values("mean", ascending=False).to_csv(os.path.join(OUT_DIR, "feature_importance_grouped.csv"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["mean", "cover_linear", "ridge", "rf", "xgb"])
    ap.add_argument("--protocols", nargs="+", default=["grouped", "random", "variety"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    main(a.models, a.protocols, a.seeds)
