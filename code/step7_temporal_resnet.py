"""Step 7: multi-temporal ResNet-50 with temporal attention (proposed model).

Each plot was imaged at three growth stages (V6, three nodes, flowering), and biomass was measured
at each one. To estimate biomass at stage t, the model looks at the plot's image at t together with
the plot's earlier images (stages <= t). It never sees later images, so it can be used during the
season, one flight at a time.

Architecture (shared weights for all time steps):
  1. ResNet-50 (ImageNet, fine-tuned, frozen BN statistics as in step 4) turns each image into a
     2048-d vector; a linear layer projects it to 256-d.
  2. A time encoding of the flight date (days since the first flight) is added to each vector.
  3. Temporal attention: the current image is the query, the current and earlier images are the
     keys/values (4 heads). The output is added to the current vector (residual + LayerNorm).
  4. A small MLP head predicts standardised log1p(biomass).

Variants (for ablations, all trained the same way on the same folds):
  mt_resnet50          proposed: history + time encoding + temporal attention
  mt_resnet50_meanpool history + time encoding, attention replaced by an average over time steps
  mt_resnet50_current  current image + time encoding only (no history): tests whether history helps

Predictions are written to outputs/predictions/<variant>__<protocol>_s<seed>.csv, the same format as
steps 3 and 4, so step5_evaluate.py includes them automatically. For the proposed model the mean
attention weight on each stage (att_0, att_1, att_2) is saved as extra columns.

Usage:
  python step7_temporal_resnet.py                  # all variants, grouped protocol, 3 seeds
  python step7_temporal_resnet.py --quick          # 1 seed, 15 epochs, proposed model only
  python step7_temporal_resnet.py --variants mt_resnet50 --protocols variety
"""
import argparse
import os
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import OUT_DIR, PRED_DIR, SPLITS, clip_pred, ensure_dirs, load_manifest, metrics, set_seed
from step4_deep_models import CKPT_DIR, PRETRAINED, TIMM_NAMES, augment, inner_split, load_images, normalize

VARIANTS = {
    "mt_resnet50": dict(history=True, pool="attention"),
    "mt_resnet50_meanpool": dict(history=True, pool="mean"),
    "mt_resnet50_current": dict(history=False, pool="attention"),
}
MAX_T = 3  # V6, three nodes, flowering


# ----------------------------------------------------------------------------
# Sequences
# ----------------------------------------------------------------------------
def flight_days(m):
    """Days since the first flight for every image, from the MMDD prefix of the file name.

    Falls back to the manifest 'date' column, then to the stage index."""
    stems = m.image_path.map(lambda p: os.path.splitext(os.path.basename(p))[0])
    mmdd = stems.str.extract(r"^(\d{4})_")[0]
    if mmdd.notna().all():
        d = pd.to_datetime("2018" + mmdd, format="%Y%m%d")
    elif "date" in m and pd.to_datetime(m["date"], errors="coerce").notna().all():
        d = pd.to_datetime(m["date"])
    else:
        return m.stage.values.astype(float) * 30.0
    return (d - d.min()).dt.days.values.astype(float)


def build_sequences(m, history):
    """For every row r: indices of the same plot's images at stages <= stage(r), oldest first, padded with -1.

    The current image is always the last valid entry."""
    seq = -np.ones((len(m), MAX_T), dtype=np.int64)
    by_plot = {g: d.sort_values("stage") for g, d in m.reset_index().groupby(m.plot_id.astype(str).values)}
    for r in range(len(m)):
        d = by_plot[str(m.plot_id.iloc[r])]
        st = m.stage.iloc[r]
        idx = d[d.stage <= st]["index"].tolist() if history else [r]
        if idx[-1] != r:  # duplicate stage for a plot: make sure the target image is the current one
            idx = [i for i in idx if i != r] + [r]
        idx = idx[-MAX_T:]
        seq[r, :len(idx)] = idx
    return seq


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class TemporalResNet(nn.Module):
    def __init__(self, pool="attention", d=256, heads=4):
        super().__init__()
        import timm
        self.backbone = timm.create_model(TIMM_NAMES["resnet50"], pretrained=PRETRAINED, num_classes=0)
        self.proj = nn.Linear(self.backbone.num_features, d)
        self.time = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, d))
        self.pool = pool
        self.attn = nn.MultiheadAttention(d, heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 1))

    def new_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("backbone.")]

    def forward(self, imgs, seq, days):
        """imgs: images of the unique frames used in this batch; seq: (B, T) positions into imgs, -1 = padding;
        days: (B, T) days since first flight. Returns (prediction (B,), attention weights (B, T))."""
        f = self.proj(self.backbone(imgs).float())  # (U, d)
        valid = seq >= 0
        z = f[seq.clamp(min=0)] * valid[..., None]  # (B, T, d)
        z = z + self.time(days[..., None] / 100.0) * valid[..., None]
        last = valid.sum(1) - 1  # position of the current image
        cur = z[torch.arange(len(z)), last]  # (B, d)
        if self.pool == "attention":
            ctx, w = self.attn(cur[:, None], z, z, key_padding_mask=~valid, need_weights=True)
            ctx, w = ctx[:, 0], w[:, 0]
        else:
            w = valid.float() / valid.sum(1, keepdim=True)
            ctx = (z * w[..., None]).sum(1)
        h = self.norm(cur + ctx)
        return self.head(h).squeeze(1), w


def batch_inputs(rows, seq, days, X, dev, train):
    s = seq[rows]
    uniq, inv = np.unique(s[s >= 0], return_inverse=True)
    pos = -np.ones_like(s)
    pos[s >= 0] = inv
    imgs = X[uniq].to(dev)
    if train:
        imgs = augment(imgs)
    dd = np.where(s >= 0, days[np.clip(s, 0, None)], 0.0)
    return (normalize(imgs), torch.from_numpy(pos).to(dev),
            torch.tensor(dd, dtype=torch.float32, device=dev))


@torch.no_grad()
def predict(model, rows, seq, days, X, dev, bs=16):
    model.eval()
    out, att = [], []
    for i in range(0, len(rows), bs):
        r = rows[i:i + bs]
        imgs, pos, dd = batch_inputs(r, seq, days, X, dev, train=False)
        with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
            p, w = model(imgs, pos, dd)
        out.append(p.float().cpu().numpy())
        att.append(w.float().cpu().numpy())
    return np.concatenate(out), np.concatenate(att)


def train(variant, X, seq, days, yz, tr, va, dev, seed, epochs, bs, patience):
    set_seed(seed)
    model = TemporalResNet(pool=VARIANTS[variant]["pool"]).to(dev)
    new = set(id(p) for p in model.new_parameters())
    params = [{"params": [p for p in model.parameters() if id(p) not in new], "lr": 1e-4},
              {"params": [p for p in model.parameters() if id(p) in new], "lr": 1e-3}]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    steps = epochs * int(np.ceil(len(tr) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4, 1e-3], total_steps=steps, pct_start=0.1)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    yt = torch.tensor(yz, dtype=torch.float32)
    best, best_state, wait = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        [x.eval() for x in model.modules() if isinstance(x, nn.modules.batchnorm._BatchNorm)]  # frozen BN stats
        perm = np.random.permutation(tr)
        for i in range(0, len(perm), bs):
            r = perm[i:i + bs]
            if len(r) < 2:
                continue
            imgs, pos, dd = batch_inputs(r, seq, days, X, dev, train=True)
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                p, _ = model(imgs, pos, dd)
                loss = F.smooth_l1_loss(p.float(), yt[r].to(dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        pv, _ = predict(model, va, seq, days, X, dev)
        vloss = float(np.mean((pv - yz[va]) ** 2))
        if vloss < best - 1e-5:
            best, wait = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, ep + 1


def main(a):
    ensure_dirs()
    os.makedirs(CKPT_DIR, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""))
    m = load_manifest()
    sp = pd.read_csv(SPLITS)
    y = m.biomass.values.astype(float)
    groups = m.plot_id.astype(str).values
    days = flight_days(m)
    print("Flight day offsets by stage:", {int(s): sorted(set(days[m.stage.values == s].astype(int)))
                                           for s in sorted(m.stage.unique())})
    n_img = m.groupby(m.plot_id.astype(str)).size()
    print(f"Plots: {len(n_img)}, images per plot: {n_img.value_counts().to_dict()}")
    X = load_images(m.image_path, mask=False)
    log = []
    for variant in a.variants:
        seq = build_sequences(m, VARIANTS[variant]["history"])
        print(f"\n{variant}: sequence lengths {dict(zip(*np.unique((seq >= 0).sum(1), return_counts=True)))}")
        for proto in a.protocols:
            for s in a.seeds:
                col = f"{proto}_s{s}"
                if col not in sp:
                    continue
                # quick-test files end in _quick.csv, which step5 ignores, so they never mix with the full run
                out_csv = os.path.join(PRED_DIR, f"{variant}__{col}{'_quick' if a.quick else ''}.csv")
                if os.path.exists(out_csv) and not a.overwrite:
                    print(f"exists, skipping: {out_csv}")
                    continue
                folds = sp[col].values
                pred = np.zeros(len(y))
                att = np.full((len(y), MAX_T), np.nan)
                t0 = time.time()
                for k in np.unique(folds):
                    tr = np.where(folds != k)[0]
                    te = np.where(folds == k)[0]
                    if proto == "grouped":  # history images of a test target must come from held-out plots too
                        assert set(seq[te][seq[te] >= 0]) <= set(te), "history leaks across folds"
                    ylog = np.log1p(y)
                    mu, sd = ylog[tr].mean(), ylog[tr].std()
                    yz = (ylog - mu) / sd
                    tr_in, va = inner_split(tr, groups, proto, s)
                    model, ep = train(variant, X, seq, days, yz, tr_in, va, dev, s, a.epochs, a.batch, a.patience)
                    p, w = predict(model, te, seq, days, X, dev)
                    pred[te] = clip_pred(np.expm1(p * sd + mu), y[tr])
                    # attention weight on each stage of the history (by stage, not by position)
                    for j, r in enumerate(te):
                        for t, src in enumerate(seq[r]):
                            if src >= 0:
                                att[r, int(m.stage.iloc[src])] = w[j, t]
                    if proto == "grouped" and s == a.seeds[0] and variant == "mt_resnet50" and not a.quick:
                        torch.save({"state": model.state_dict(), "mu": mu, "sd": sd, "test_rows": te},
                                   os.path.join(CKPT_DIR, f"{variant}__{col}_f{k}.pt"))
                    print(f"  {variant} {col} fold {k}: stopped at epoch {ep}")
                    del model
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                secs = time.time() - t0
                out = pd.DataFrame({"row": np.arange(len(y)), "y": y, "pred": pred, "fold": folds})
                if variant == "mt_resnet50":
                    for t in range(MAX_T):
                        out[f"att_{t}"] = att[:, t]
                out.to_csv(out_csv, index=False)
                rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
                print(f"{variant:22s} {col:12s} RMSE={rmse:9.2f}  ({secs:.0f}s)")
                for st in sorted(m.stage.unique()):
                    k = m.stage.values == st
                    print(f"    stage {st}: RMSE={metrics(y[k], pred[k])['RMSE']:8.1f}  R2={metrics(y[k], pred[k])['R2']:.3f}")
                log.append({"model": variant, "split": col, "rmse": rmse, "seconds": secs})
    if log:
        p = os.path.join(OUT_DIR, "deep_training_log.csv")
        pd.DataFrame(log).to_csv(p, mode="a", header=not os.path.exists(p), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--protocols", nargs="+", default=["grouped"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=12, help="target images per step (each brings up to 3 frames)")
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--quick", action="store_true", help="1 seed, 15 epochs, proposed model only")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.epochs, a.patience, a.variants, a.overwrite = [0], 15, 5, ["mt_resnet50"], True
    main(a)
