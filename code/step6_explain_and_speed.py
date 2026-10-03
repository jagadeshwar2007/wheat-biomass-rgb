"""Step 6: Grad-CAM explainability (with a quantitative check) and model size / speed.

Explainability: for each fine-tuned CNN, Grad-CAM maps are computed on the held-out images of the
grouped split (seed 0). The Attribution-on-Vegetation score (AoV) is the share of Grad-CAM mass that
falls on vegetation pixels. If the model attends to plants rather than soil, AoV should exceed the
vegetation pixel fraction; this is tested with a one-sided Wilcoxon signed-rank test.

Speed: parameters and median inference latency (batch size 1) on CPU and GPU.
Outputs: outputs/table_gradcam_aov.csv, outputs/fig_gradcam_<model>.png, outputs/table_speed.csv
"""
import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon

from common import OUT_DIR, handcrafted_features, load_manifest, load_rgb, veg_mask
from step4_deep_models import CKPT_DIR, FROZEN, IMG, TIMM_NAMES, load_images, normalize

STAGE_NAMES = {0: "V6", 1: "Three nodes", 2: "Flowering"}


def target_layer(model, name):
    return model.layer4 if name.startswith("resnet") else model.blocks[-1]


def square_resize_mask(mask):
    t = torch.from_numpy(mask.astype(np.float32))[None]
    h, w = t.shape[1:]
    s = max(h, w)
    t = F.pad(t, ((s - w) // 2, s - w - (s - w) // 2, (s - h) // 2, s - h - (s - h) // 2))
    return F.interpolate(t[None], size=(IMG, IMG), mode="nearest")[0, 0].numpy() > 0.5


def gradcam(model, layer, x):
    acts, grads = {}, {}

    def fwd(m, i, o):
        acts["a"] = o
        o.register_hook(lambda g: grads.__setitem__("g", g))

    h = layer.register_forward_hook(fwd)
    model.zero_grad()
    out = model(x).sum()
    out.backward()
    h.remove()
    a, g = acts["a"], grads["g"]
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(IMG, IMG), mode="bilinear", align_corners=False)[:, 0]
    cam = cam / (cam.flatten(1).max(1)[0].view(-1, 1, 1) + 1e-8)
    return cam.detach().cpu().numpy()


def explain(models, dev):
    import timm
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    man = load_manifest()
    rgbs = [load_rgb(p) for p in man.image_path]
    veg = [square_resize_mask(veg_mask(r, v)) for r, v in rgbs]
    valid = [square_resize_mask(v) for _, v in rgbs]
    X = load_images(man.image_path, mask=False)
    rows = []
    for name in models:
        ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, f"{name}__grouped_s*_f*.pt")))
        if not ckpts:
            print(f"no checkpoints for {name}; run step4 first")
            continue
        examples = []
        for ck in ckpts:
            st = torch.load(ck, map_location="cpu", weights_only=False)
            model = timm.create_model(TIMM_NAMES[name], pretrained=False, num_classes=1)
            model.load_state_dict(st["state"])
            model.to(dev).eval()
            layer = target_layer(model, name)
            for r in st["test_rows"]:
                cam = gradcam(model, layer, normalize(X[[r]].to(dev)))[0]
                vm, va = veg[r], valid[r]
                tot = (cam * va).sum()
                aov = float((cam * vm).sum() / tot) if tot > 0 else np.nan
                frac = float(vm.sum() / max(va.sum(), 1))
                rows.append(dict(model=name, row=int(r), stage=int(man.stage[r]), AoV=aov, veg_fraction=frac))
                examples.append((r, cam))
        # example figure: two held-out images per stage
        pick = []
        for s in sorted(man.stage.unique()):
            pick += [e for e in examples if man.stage[e[0]] == s][:2]
        if pick:
            fig, axes = plt.subplots(2, len(pick), figsize=(2.2 * len(pick), 4.6))
            for j, (r, cam) in enumerate(pick):
                img = X[r].permute(1, 2, 0).numpy()
                axes[0, j].imshow(img); axes[0, j].axis("off")
                axes[0, j].set_title(f"{STAGE_NAMES.get(int(man.stage[r]), '')}\n{man.biomass[r]:.0f} kg/ha", fontsize=8)
                axes[1, j].imshow(img); axes[1, j].imshow(cam, cmap="jet", alpha=0.45); axes[1, j].axis("off")
            fig.suptitle(f"Grad-CAM, {name} (held-out plots)", fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, f"fig_gradcam_{name}.png"), dpi=300)
            plt.close(fig)
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "gradcam_per_image.csv"), index=False)
    summ = []
    for name, d in df.dropna().groupby("model"):
        diff = d.AoV - d.veg_fraction
        p = wilcoxon(diff, alternative="greater").pvalue if (diff != 0).any() else np.nan
        summ.append(dict(model=name, n=len(d), AoV_mean=d.AoV.mean(), veg_fraction_mean=d.veg_fraction.mean(),
                         median_diff=float(diff.median()), wilcoxon_p_one_sided=float(p)))
    t = pd.DataFrame(summ)
    t.to_csv(os.path.join(OUT_DIR, "table_gradcam_aov.csv"), index=False)
    print("\n=== Attribution on vegetation (AoV) vs vegetation fraction ===")
    print(t.round(4).to_string(index=False))


@torch.no_grad()
def latency(model, dev, reps=100):
    x = torch.randn(1, 3, IMG, IMG, device=dev)
    for _ in range(10):
        model(x)
    times = []
    for _ in range(reps):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        model(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


def speed(models):
    import timm
    rows = []
    for name in models:
        kw = {"num_classes": 0, "img_size": IMG} if name in FROZEN else {"num_classes": 1}
        model = timm.create_model(TIMM_NAMES[name], pretrained=False, **kw).eval()
        r = dict(model=name, params_M=sum(p.numel() for p in model.parameters()) / 1e6,
                 cpu_ms=latency(model, torch.device("cpu")))
        if torch.cuda.is_available():
            r["gpu_ms"] = latency(model.cuda(), torch.device("cuda"))
        rows.append(r)
        print(r)
    man = load_manifest()
    t = time.perf_counter()
    for p in man.image_path[:30]:
        handcrafted_features(p)
    rows.append(dict(model="handcrafted features (+RF/XGB)", params_M=np.nan,
                     cpu_ms=(time.perf_counter() - t) / 30 * 1000))
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "table_speed.csv"), index=False)
    print(f"CPU threads used: {torch.get_num_threads()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam-models", nargs="+", default=["mobilenetv3", "resnet50", "efficientnet_b0"])
    ap.add_argument("--speed-models", nargs="+", default=["mobilenetv3", "efficientnet_b0", "resnet50", "dinov2_s", "dinov2_b"])
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    explain(a.cam_models, dev)
    speed(a.speed_models)
