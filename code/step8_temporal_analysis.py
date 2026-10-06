"""Step 8: does the multi-temporal model help, and where? Run after step7 and step5.

Compares, on the grouped protocol with seed-averaged predictions:
  mt_resnet50 vs resnet50              does the proposed model beat the single-image baseline?
  mt_resnet50 vs mt_resnet50_current   does the image history help (same model, no history)?
  mt_resnet50 vs mt_resnet50_meanpool  does attention help over a plain average of the history?
Each comparison: RMSE difference with 95% cluster-bootstrap CI (plots resampled), paired Wilcoxon on
absolute errors, Holm correction over the three comparisons; overall and per growth stage.

Also summarises the attention weights and times the model.
Outputs (in outputs/): table_temporal.csv, tests_temporal.csv, table_attention.csv,
                       table_speed_temporal.csv, fig_temporal_by_stage.png, fig_attention.png
"""
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from common import OUT_DIR, load_manifest, metrics
from step5_evaluate import B, ci, holm, load_preds

STAGE_NAMES = {0: "V6", 1: "Three nodes", 2: "Flowering"}
MODELS = ["resnet50", "mt_resnet50_current", "mt_resnet50_meanpool", "mt_resnet50", "ridge"]
PAIRS = [("mt_resnet50", "resnet50"), ("mt_resnet50", "mt_resnet50_current"),
         ("mt_resnet50", "mt_resnet50_meanpool")]
RNG = np.random.default_rng(2024)


def boot_sets(groups, subset):
    """Cluster bootstrap over plots, restricted to rows in `subset`."""
    rows = np.where(subset)[0]
    g = groups[rows]
    uniq = np.unique(g)
    by = {u: rows[g == u] for u in uniq}
    for _ in range(B):
        pick = RNG.choice(uniq, size=len(uniq), replace=True)
        yield np.concatenate([by[u] for u in pick])


def rmse(y, p):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def main():
    man = load_manifest()
    groups = man.plot_id.astype(str).values
    stage = man.stage.values
    P = load_preds()
    P = P[(P.protocol == "grouped") & P.model.isin(MODELS)]
    have = [m for m in MODELS if m in set(P.model)]
    missing = [m for m in MODELS if m not in have]
    if missing:
        print(f"Note: no grouped predictions for {missing}; those rows are skipped.")
    A = {m: P[P.model == m].groupby("row").agg(y=("y", "first"), pred=("pred", "mean")).sort_index()
         for m in have}
    y = A[have[0]].y.values

    subsets = [("All", np.ones(len(y), bool))] + [(STAGE_NAMES[s], stage == s) for s in sorted(np.unique(stage))]

    # accuracy table, overall and per stage
    rec = []
    for mdl in have:
        p = A[mdl].pred.values
        for name, k in subsets:
            r = metrics(y[k], p[k])
            bs = [rmse(y[i], p[i]) for i in boot_sets(groups, k)]
            per_run = [rmse(d.y.values[k], d.pred.values[k])
                       for _, d in P[P.model == mdl].sort_values("row").groupby("seed")]
            rec.append(dict(model=mdl, stage=name, n=int(k.sum()), RMSE=r["RMSE"], RMSE_lo=ci(bs)[0],
                            RMSE_hi=ci(bs)[1], R2=r["R2"], nRMSE_pct=r["nRMSE_%"], bias=r["bias"],
                            RMSE_per_run_mean=np.mean(per_run), RMSE_per_run_sd=np.std(per_run, ddof=1)
                            if len(per_run) > 1 else np.nan))
    tab = pd.DataFrame(rec)
    tab.to_csv(os.path.join(OUT_DIR, "table_temporal.csv"), index=False)
    print("\n=== Grouped protocol, seed-averaged predictions ===")
    print(tab[["model", "stage", "RMSE", "RMSE_lo", "RMSE_hi", "R2", "RMSE_per_run_mean",
               "RMSE_per_run_sd"]].round(3).to_string(index=False))

    # paired comparisons
    rec = []
    for a_, b_ in PAIRS:
        if a_ not in A or b_ not in A:
            continue
        pa, pb = A[a_].pred.values, A[b_].pred.values
        for name, k in subsets:
            diffs = [rmse(y[i], pa[i]) - rmse(y[i], pb[i]) for i in boot_sets(groups, k)]
            ea, eb = np.abs(pa[k] - y[k]), np.abs(pb[k] - y[k])
            rec.append(dict(model=a_, vs=b_, stage=name, dRMSE=rmse(y[k], pa[k]) - rmse(y[k], pb[k]),
                            lo=ci(diffs)[0], hi=ci(diffs)[1], dR2=metrics(y[k], pa[k])["R2"] - metrics(y[k], pb[k])["R2"],
                            median_abs_err_diff=float(np.median(ea - eb)),
                            wilcoxon_p=float(wilcoxon(ea, eb).pvalue)))
    if rec:
        t = pd.DataFrame(rec)
        t["p_holm"] = np.nan
        for name in t.stage.unique():  # Holm over the three comparisons, separately for each subset
            k = t.stage == name
            t.loc[k, "p_holm"] = holm(t.loc[k, "wilcoxon_p"])
        t.to_csv(os.path.join(OUT_DIR, "tests_temporal.csv"), index=False)
        print("\n=== Paired comparisons (negative dRMSE = first model better) ===")
        print(t.round(4).to_string(index=False))

    # attention weights of the proposed model, by target stage
    if "mt_resnet50" in have and "att_0" in P.columns:
        d = P[P.model == "mt_resnet50"].merge(pd.DataFrame({"row": np.arange(len(y)), "target": stage}), on="row")
        att = d.groupby("target")[["att_0", "att_1", "att_2"]].mean()
        att.index = [STAGE_NAMES[i] for i in att.index]
        att.columns = [f"weight on {STAGE_NAMES[i]} image" for i in range(3)]
        att.to_csv(os.path.join(OUT_DIR, "table_attention.csv"))
        print("\n=== Mean attention weight (rows: stage being estimated) ===")
        print(att.round(3).to_string())
    speed()
    figures(tab, have)


def speed():
    """Latency of the proposed model for one plot: full 3-image history, and with cached history features."""
    from step7_temporal_resnet import TemporalResNet
    from step4_deep_models import IMG
    rec = []
    for dev_name in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        dev = torch.device(dev_name)
        model = TemporalResNet().to(dev).eval()
        x3 = torch.rand(3, 3, IMG, IMG, device=dev)
        seq = torch.tensor([[0, 1, 2]], device=dev)
        days = torch.tensor([[0., 36., 64.]], device=dev)
        times_full, times_new = [], []
        with torch.no_grad():
            for i in range(25):
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t = time.perf_counter()
                model(x3, seq, days)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                if i >= 5:
                    times_full.append((time.perf_counter() - t) * 1000)
                t = time.perf_counter()
                model.backbone(x3[:1])  # only the new flight's image needs the backbone if history is cached
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                if i >= 5:
                    times_new.append((time.perf_counter() - t) * 1000)
        rec.append(dict(device=dev_name, params_M=sum(p.numel() for p in model.parameters()) / 1e6,
                        ms_full_history=float(np.median(times_full)), ms_cached_history=float(np.median(times_new))))
    t = pd.DataFrame(rec)
    t.to_csv(os.path.join(OUT_DIR, "table_speed_temporal.csv"), index=False)
    print("\n=== Speed of mt_resnet50 (one plot) ===")
    print(t.round(2).to_string(index=False))


def figures(tab, have):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"resnet50": "ResNet-50", "mt_resnet50_current": "MT, no history",
              "mt_resnet50_meanpool": "MT, mean pooling", "mt_resnet50": "MT + attention (proposed)",
              "ridge": "Ridge"}
    stages = [s for s in ["V6", "Three nodes", "Flowering"] if s in set(tab.stage)]
    show = [m for m in have if m != "ridge"]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    w = 0.8 / len(show)
    for i, mdl in enumerate(show):
        d = tab[(tab.model == mdl)].set_index("stage").loc[stages]
        x = np.arange(len(stages)) + (i - (len(show) - 1) / 2) * w
        ax.bar(x, d.RMSE, w, yerr=[d.RMSE - d.RMSE_lo, d.RMSE_hi - d.RMSE], capsize=2, label=labels.get(mdl, mdl))
    ax.set_xticks(np.arange(len(stages)))
    ax.set_xticklabels(stages)
    ax.set_ylabel("RMSE (kg/ha)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_temporal_by_stage.png"), dpi=300)
    plt.close(fig)

    p = os.path.join(OUT_DIR, "table_attention.csv")
    if os.path.exists(p):
        att = pd.read_csv(p, index_col=0)
        fig, ax = plt.subplots(figsize=(3.6, 2.8))
        im = ax.imshow(att.values, vmin=0, vmax=1, cmap="Greens")
        for i in range(att.shape[0]):
            for j in range(att.shape[1]):
                if not np.isnan(att.values[i, j]):
                    ax.text(j, i, f"{att.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(att.shape[1]))
        ax.set_xticklabels(["V6", "3 nodes", "Flower."], fontsize=8)
        ax.set_yticks(range(att.shape[0]))
        ax.set_yticklabels(att.index, fontsize=8)
        ax.set_xlabel("Image attended to")
        ax.set_ylabel("Stage estimated")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "fig_attention.png"), dpi=300)
        plt.close(fig)
    print(f"\nTemporal tables and figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
