"""Step 2: create the cross-validation splits (shared by every model) and handcrafted features.

Protocols (each gets its own fold column per seed in data/splits.csv):
  grouped  - 5-fold, all three stages of a plot stay in the same fold (no plot leakage). PRIMARY.
  random   - 5-fold over images, ignoring plots (the common but leaky protocol). For comparison only.
  variety  - leave-one-variety-out (only if the manifest has >1 variety).
"""
import argparse

import numpy as np
import pandas as pd

from common import FEATURES, SPLITS, ensure_dirs, handcrafted_features, load_manifest

N_FOLDS = 5


def grouped_folds(groups, seed, k=N_FOLDS):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(pd.unique(groups), key=str))
    rng.shuffle(uniq)
    fold_of = {g: i % k for i, g in enumerate(uniq)}
    return np.array([fold_of[g] for g in groups])


def random_folds(n, seed, k=N_FOLDS):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    f = np.empty(n, int)
    f[perm] = np.arange(n) % k
    return f


def main(seeds):
    ensure_dirs()
    m = load_manifest()
    sp = pd.DataFrame({"row": np.arange(len(m)), "plot_id": m.plot_id, "stage": m.stage})
    for s in seeds:
        sp[f"grouped_s{s}"] = grouped_folds(m.plot_id.astype(str).values, s)
        sp[f"random_s{s}"] = random_folds(len(m), s)
    if m.variety.nunique() > 1:
        codes = {v: i for i, v in enumerate(sorted(m.variety.unique()))}
        for s in seeds:
            sp[f"variety_s{s}"] = m.variety.map(codes).values
    sp.to_csv(SPLITS, index=False)
    print(f"Wrote {SPLITS}: columns {[c for c in sp.columns if '_s' in c]}")
    # Leakage check: no plot appears in two grouped folds
    for s in seeds:
        assert sp.groupby("plot_id")[f"grouped_s{s}"].nunique().max() == 1
    print("Leakage check passed: every plot sits in exactly one grouped fold.")

    rows = []
    for i, p in enumerate(m.image_path):
        f = handcrafted_features(p)
        f["row"] = i
        rows.append(f)
        if (i + 1) % 50 == 0:
            print(f"  features {i + 1}/{len(m)}")
    pd.DataFrame(rows).to_csv(FEATURES, index=False)
    print(f"Wrote {FEATURES}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    main(ap.parse_args().seeds)
