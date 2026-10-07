"""Step 4: deep-learning models, evaluated on exactly the same folds as the classical models.

Models (pretrained weights downloaded once by timm from Hugging Face):
  resnet50        - ResNet-50 (ImageNet), fully fine-tuned               [CNN baseline]
  efficientnet_b0 - EfficientNet-B0 (ImageNet), fully fine-tuned
  mobilenetv3     - MobileNetV3-Large (ImageNet), fully fine-tuned     [lightweight model]
  dinov2_s        - DINOv2 ViT-S/14, frozen features + ridge head        [transformer, frozen]
  dinov2_b        - DINOv2 ViT-B/14, frozen features + ridge head
  convnext_t      - ConvNeXt-Tiny (ImageNet-22k -> 1k), fully fine-tuned  [modern CNN]
  swin_t          - Swin Transformer Tiny (ImageNet-22k -> 1k), fully fine-tuned  [transformer]
  efficientnetv2_s - EfficientNetV2-S (ImageNet-21k -> 1k), fully fine-tuned
Add --mask to feed images with non-vegetation (soil) pixels set to black (segmentation ablation).

Targets are log1p(biomass), standardised on the training fold; predictions are back-transformed.
Early stopping uses an inner validation split taken from the training fold only
(whole plots for the grouped/variety protocols).
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import OUT_DIR, PRED_DIR, SPLITS, clip_pred, ensure_dirs, load_manifest, load_rgb, set_seed, veg_mask

TIMM_NAMES = {
    "resnet50": "resnet50.a1_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "mobilenetv3": "mobilenetv3_large_100.ra_in1k",
    "dinov2_s": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_b": "vit_base_patch14_dinov2.lvd142m",
    "convnext_t": "convnext_tiny.fb_in22k_ft_in1k",
    "swin_t": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
    "efficientnetv2_s": "tf_efficientnetv2_s.in21k_ft_in1k",
}
FROZEN = {"dinov2_s", "dinov2_b"}
IMG = int(os.environ.get("BIOMASS_IMG", 224))  # override only for smoke tests
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
# Set BIOMASS_NO_PRETRAINED=1 only for offline smoke tests (random weights, results meaningless).
PRETRAINED = os.environ.get("BIOMASS_NO_PRETRAINED") != "1"


def load_images(paths, mask):
    """Pad each plot image to a square (keeps aspect ratio) and resize to IMG x IMG."""
    out = []
    for p in paths:
        rgb, valid = load_rgb(p)
        keep = veg_mask(rgb, valid) if mask else valid
        rgb = rgb * keep[..., None]
        t = torch.from_numpy(rgb).permute(2, 0, 1)
        h, w = t.shape[1:]
        s = max(h, w)
        t = F.pad(t, ((s - w) // 2, s - w - (s - w) // 2, (s - h) // 2, s - h - (s - h) // 2))
        t = F.interpolate(t[None], size=(IMG, IMG), mode="bilinear", align_corners=False)[0]
        out.append(t)
    return torch.stack(out)


def normalize(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def augment(x):
    """Top-view imagery: flips and 90-degree rotations keep biomass unchanged; mild photometric jitter."""
    if torch.rand(1) < 0.5:
        x = x.flip(3)
    if torch.rand(1) < 0.5:
        x = x.flip(2)
    x = torch.rot90(x, int(torch.randint(0, 4, (1,))), dims=(2, 3))
    b = 1 + (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5) * 0.2
    c = 1 + (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5) * 0.2
    mu = x.mean(dim=(1, 2, 3), keepdim=True)
    return ((x - mu) * c + mu * b).clamp(0, 1)


def inner_split(tr_idx, groups, proto, seed, frac=0.15):
    rng = np.random.default_rng(seed + 1000)
    if proto == "random":
        perm = rng.permutation(tr_idx)
        n = max(1, int(len(perm) * frac))
        return perm[n:], perm[:n]
    g = np.unique(groups[tr_idx])
    rng.shuffle(g)
    val_g = set(g[: max(1, int(len(g) * frac))])
    is_val = np.array([groups[i] in val_g for i in tr_idx])
    return tr_idx[~is_val], tr_idx[is_val]


def train_finetune(name, X, ylog, tr, va, dev, seed, epochs, bs, patience):
    import timm
    set_seed(seed)
    model = timm.create_model(TIMM_NAMES[name], pretrained=PRETRAINED, num_classes=1).to(dev)
    head = set(id(p) for p in model.get_classifier().parameters())
    params = [{"params": [p for p in model.parameters() if id(p) not in head], "lr": 1e-4},
              {"params": [p for p in model.parameters() if id(p) in head], "lr": 1e-3}]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    steps = epochs * int(np.ceil(len(tr) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4, 1e-3], total_steps=steps, pct_start=0.1)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    yt = torch.tensor(ylog, dtype=torch.float32)
    best, best_state, wait = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        [x.eval() for x in model.modules() if isinstance(x, torch.nn.modules.batchnorm._BatchNorm)]  # frozen BN stats
        perm = np.random.permutation(tr)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            xb = normalize(augment(X[idx].to(dev)))
            yb = yt[idx].to(dev)
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                loss = F.smooth_l1_loss(model(xb).squeeze(1).float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        pv = predict(model, X, va, dev, bs)
        vloss = float(np.mean((pv - ylog[va]) ** 2))
        if vloss < best - 1e-5:
            best, wait = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, ep + 1


@torch.no_grad()
def predict(model, X, idx, dev, bs=32):
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        xb = normalize(X[idx[i:i + bs]].to(dev))
        with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
            out.append(model(xb).squeeze(1).float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def dinov2_features(name, X, dev, bs=32):
    import timm
    model = timm.create_model(TIMM_NAMES[name], pretrained=PRETRAINED, num_classes=0, img_size=IMG).to(dev).eval()
    feats = []
    for i in range(0, len(X), bs):
        xb = normalize(X[i:i + bs].to(dev))
        tok = model.forward_features(xb)
        npt = getattr(model, "num_prefix_tokens", 1)
        cls, patch = tok[:, 0], tok[:, npt:].mean(1)
        feats.append(torch.cat([cls, patch], 1).float().cpu().numpy())
    return np.concatenate(feats)


def ridge_head(Z, ylog, tr, te, groups, proto, seed):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    inner = KFold(4, shuffle=True, random_state=seed) if proto == "random" else GroupKFold(4)
    kw = {} if proto == "random" else {"groups": groups[tr]}
    gs = GridSearchCV(make_pipeline(StandardScaler(), Ridge()),
                      {"ridge__alpha": list(np.logspace(0, 5, 11))}, cv=inner,
                      scoring="neg_root_mean_squared_error")
    gs.fit(Z[tr], ylog[tr], **kw)
    return gs.predict(Z[te])


def main(a):
    ensure_dirs()
    os.makedirs(CKPT_DIR, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""))
    m = load_manifest()
    sp = pd.read_csv(SPLITS)
    y = m.biomass.values.astype(float)
    groups = m.plot_id.astype(str).values
    X = load_images(m.image_path, a.mask)
    suffix = "_mask" if a.mask else ""
    log = []
    for name in a.models:
        tag = name + suffix
        Z = dinov2_features(name, X, dev) if name in FROZEN else None
        for proto in a.protocols:
            for s in a.seeds:
                col = f"{proto}_s{s}"
                if col not in sp:
                    continue
                out_csv = os.path.join(PRED_DIR, f"{tag}__{col}.csv")
                if os.path.exists(out_csv) and not a.overwrite:
                    print(f"exists, skipping: {out_csv}")
                    continue
                folds = sp[col].values
                pred = np.zeros(len(y))
                t0 = time.time()
                for k in np.unique(folds):
                    tr = np.where(folds != k)[0]
                    te = np.where(folds == k)[0]
                    ylog = np.log1p(y)
                    mu, sd = ylog[tr].mean(), ylog[tr].std()
                    yz = (ylog - mu) / sd
                    if name in FROZEN:
                        pred[te] = clip_pred(np.expm1(ridge_head(Z, ylog, tr, te, groups, proto, s)), y[tr])
                        continue
                    tr_in, va = inner_split(tr, groups, proto, s)
                    model, ep = train_finetune(name, X, yz, tr_in, va, dev, s, a.epochs, a.batch, a.patience)
                    pred[te] = clip_pred(np.expm1(predict(model, X, te, dev) * sd + mu), y[tr])
                    if proto == "grouped" and s == a.seeds[0]:
                        torch.save({"state": model.state_dict(), "mu": mu, "sd": sd, "test_rows": te,
                                    "mask": a.mask}, os.path.join(CKPT_DIR, f"{tag}__{col}_f{k}.pt"))
                    print(f"  {tag} {col} fold {k}: stopped at epoch {ep}")
                    del model
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                secs = time.time() - t0
                pd.DataFrame({"row": np.arange(len(y)), "y": y, "pred": pred, "fold": folds}).to_csv(out_csv, index=False)
                rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
                print(f"{tag:22s} {col:12s} RMSE={rmse:9.2f}  ({secs:.0f}s)")
                log.append({"model": tag, "split": col, "rmse": rmse, "seconds": secs})
    if log:
        p = os.path.join(OUT_DIR, "deep_training_log.csv")
        pd.DataFrame(log).to_csv(p, mode="a", header=not os.path.exists(p), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["mobilenetv3", "resnet50", "efficientnet_b0", "dinov2_s", "dinov2_b"])
    ap.add_argument("--protocols", nargs="+", default=["grouped", "random", "variety"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--mask", action="store_true", help="black out non-vegetation pixels (segmentation ablation)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--overwrite", action="store_true")
    main(ap.parse_args())
