"""Step 5: metrics, confidence intervals, statistical tests and figures from all saved predictions.

Outputs (in outputs/):
  table_main.csv              mean +- SD over seeds, every model x protocol
  table_ci_grouped.csv        seed-averaged predictions, 95% cluster-bootstrap CIs (grouped protocol)
  table_per_stage.csv         metrics per growth stage
  table_per_variety.csv       metrics per variety
  tests_vs_best.csv           paired Wilcoxon + bootstrap CI of RMSE difference vs best model (Holm-corrected)
  tests_inflation.csv         random-split RMSE minus grouped-split RMSE, per model, with CI
  tests_mask.csv              segmentation ablation (masked minus unmasked RMSE)
  tests_noninferiority.csv    lightweight model vs best model, relative RMSE increase with CI
  fig_*.png                   figures for the paper
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import OUT_DIR, PRED_DIR, load_manifest, metrics

B = 2000
RNG = np.random.default_rng(12345)
STAGE_NAMES = {0: "V6", 1: "Three nodes", 2: "Flowering"}
NONINF_MARGIN = 0.10


def load_preds():
    rows = []
    for p in glob.glob(os.path.join(PRED_DIR, "*.csv")):
        mt = re.match(r"(.+)__(\w+)_s(\d+)\.csv$", os.path.basename(p))
        if not mt:
            continue
        d = pd.read_csv(p)
        d["model"], d["protocol"], d["seed"] = mt.group(1), mt.group(2), int(mt.group(3))
        rows.append(d)
    if not rows:
        raise SystemExit("No predictions found. Run steps 3 and 4 first.")
    return pd.concat(rows, ignore_index=True)


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def cluster_boot_idx(groups):
    """Resample plots with replacement; return list of row-index arrays."""
    uniq = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in uniq}
    for _ in range(B):
        pick = RNG.choice(uniq, size=len(uniq), replace=True)
        yield np.concatenate([by[g] for g in pick])


def rmse(y, p):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def ci(vals):
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    man = load_manifest()
    groups = man.plot_id.astype(str).values
    P = load_preds()
    P = P.merge(man[["stage", "variety"]].reset_index().rename(columns={"index": "row"}), on="row")

    # 1) mean +- SD across seeds
    rec = []
    for (mdl, pr, s), d in P.groupby(["model", "protocol", "seed"]):
        r = metrics(d.y, d.pred)
        r.update(model=mdl, protocol=pr, seed=s)
        rec.append(r)
    per_seed = pd.DataFrame(rec)
    cols = ["RMSE", "MAE", "R2", "nRMSE_%", "MAPE_%", "bias", "pearson_r", "CCC"]
    main_tab = per_seed.groupby(["protocol", "model"])[cols].agg(["mean", "std"])
    main_tab.columns = [f"{a}_{b}" for a, b in main_tab.columns]
    main_tab = main_tab.sort_values(["protocol", "RMSE_mean"])
    main_tab.to_csv(os.path.join(OUT_DIR, "table_main.csv"))
    print("\n=== Mean over seeds (RMSE, R2) ===")
    print(main_tab[["RMSE_mean", "RMSE_std", "MAE_mean", "R2_mean", "R2_std"]].round(3).to_string())

    # seed-averaged predictions
    A = P.groupby(["model", "protocol", "row"]).agg(y=("y", "first"), pred=("pred", "mean"),
                                                  stage=("stage", "first"), variety=("variety", "first")).reset_index()

    def get(mdl, pr):
        d = A[(A.model == mdl) & (A.protocol == pr)].sort_values("row")
        return d.y.values, d.pred.values

    # 2) CIs for grouped protocol
    G = A[A.protocol == "grouped"]
    models = sorted(G.model.unique())
    rec = []
    for mdl in models:
        y, p = get(mdl, "grouped")
        r = metrics(y, p)
        bs_r, bs_m, bs_r2 = [], [], []
        for idx in cluster_boot_idx(groups):
            mm = metrics(y[idx], p[idx])
            bs_r.append(mm["RMSE"]); bs_m.append(mm["MAE"]); bs_r2.append(mm["R2"])
        r.update(model=mdl, RMSE_lo=ci(bs_r)[0], RMSE_hi=ci(bs_r)[1], MAE_lo=ci(bs_m)[0], MAE_hi=ci(bs_m)[1],
                 R2_lo=ci(bs_r2)[0], R2_hi=ci(bs_r2)[1])
        rec.append(r)
    ci_tab = pd.DataFrame(rec).set_index("model").sort_values("RMSE")
    ci_tab.to_csv(os.path.join(OUT_DIR, "table_ci_grouped.csv"))
    print("\n=== Grouped protocol, seed-averaged predictions, 95% cluster-bootstrap CI ===")
    print(ci_tab[["RMSE", "RMSE_lo", "RMSE_hi", "R2", "R2_lo", "R2_hi"]].round(3).to_string())

    # 3) per stage and per variety
    rec = []
    for (mdl, pr, st), d in A.groupby(["model", "protocol", "stage"]):
        r = metrics(d.y, d.pred)
        r.update(model=mdl, protocol=pr, stage=STAGE_NAMES.get(st, st))
        rec.append(r)
    pd.DataFrame(rec).to_csv(os.path.join(OUT_DIR, "table_per_stage.csv"), index=False)
    rec = []
    for (mdl, pr, v), d in A.groupby(["model", "protocol", "variety"]):
        r = metrics(d.y, d.pred)
        r.update(model=mdl, protocol=pr, variety=v)
        rec.append(r)
    pd.DataFrame(rec).to_csv(os.path.join(OUT_DIR, "table_per_variety.csv"), index=False)

    # 4) paired tests vs best model (grouped)
    learned = [m for m in models if m != "mean"]
    best = ci_tab.index[0]
    yb, pb = get(best, "grouped")
    rec = []
    for mdl in learned:
        if mdl == best:
            continue
        y, p = get(mdl, "grouped")
        e_m, e_b = np.abs(p - y), np.abs(pb - y)
        stat = wilcoxon(e_m, e_b)
        diffs = [rmse(y[i], p[i]) - rmse(y[i], pb[i]) for i in cluster_boot_idx(groups)]
        rec.append(dict(model=mdl, vs=best, dRMSE=rmse(y, p) - rmse(y, pb), dRMSE_lo=ci(diffs)[0],
                        dRMSE_hi=ci(diffs)[1], median_abs_err_diff=float(np.median(e_m - e_b)),
                        wilcoxon_p=float(stat.pvalue)))
    if rec:
        t = pd.DataFrame(rec)
        t["p_holm"] = holm(t.wilcoxon_p)
        t.to_csv(os.path.join(OUT_DIR, "tests_vs_best.csv"), index=False)
        print(f"\n=== Paired tests vs best model ({best}), grouped protocol ===")
        print(t.round(4).to_string(index=False))

    # 5) inflation: random vs grouped
    rec = []
    for mdl in sorted(set(A[A.protocol == "random"].model) & set(models)):
        y, pg = get(mdl, "grouped")
        _, pr_ = get(mdl, "random")
        diffs = [rmse(y[i], pr_[i]) - rmse(y[i], pg[i]) for i in cluster_boot_idx(groups)]
        rec.append(dict(model=mdl, RMSE_grouped=rmse(y, pg), RMSE_random=rmse(y, pr_),
                        diff_random_minus_grouped=rmse(y, pr_) - rmse(y, pg), lo=ci(diffs)[0], hi=ci(diffs)[1],
                        R2_grouped=metrics(y, pg)["R2"], R2_random=metrics(y, pr_)["R2"],
                        wilcoxon_p=float(wilcoxon(np.abs(pr_ - y), np.abs(pg - y)).pvalue)))
    if rec:
        t = pd.DataFrame(rec)
        t["p_holm"] = holm(t.wilcoxon_p)
        t.to_csv(os.path.join(OUT_DIR, "tests_inflation.csv"), index=False)
        print("\n=== Random split vs grouped split (negative diff = random looks better) ===")
        print(t.round(4).to_string(index=False))

    # 6) mask ablation
    rec = []
    for mdl in models:
        if mdl.endswith("_mask") and mdl[:-5] in models:
            y, pm = get(mdl, "grouped")
            _, pu = get(mdl[:-5], "grouped")
            diffs = [rmse(y[i], pm[i]) - rmse(y[i], pu[i]) for i in cluster_boot_idx(groups)]
            rec.append(dict(model=mdl[:-5], RMSE_unmasked=rmse(y, pu), RMSE_masked=rmse(y, pm),
                            diff_masked_minus_unmasked=rmse(y, pm) - rmse(y, pu), lo=ci(diffs)[0], hi=ci(diffs)[1],
                            wilcoxon_p=float(wilcoxon(np.abs(pm - y), np.abs(pu - y)).pvalue)))
    if rec:
        t = pd.DataFrame(rec)
        t["p_holm"] = holm(t.wilcoxon_p)
        t.to_csv(os.path.join(OUT_DIR, "tests_mask.csv"), index=False)
        print("\n=== Segmentation ablation (negative diff = masking helps) ===")
        print(t.round(4).to_string(index=False))

    # 7) non-inferiority of lightweight model
    if "mobilenetv3" in models and best != "mobilenetv3":
        y, pl = get("mobilenetv3", "grouped")
        rel = [rmse(y[i], pl[i]) / rmse(y[i], pb[i]) - 1 for i in cluster_boot_idx(groups)]
        r = dict(light="mobilenetv3", best=best, rel_increase=rmse(y, pl) / rmse(y, pb) - 1,
                 lo=ci(rel)[0], hi=ci(rel)[1], margin=NONINF_MARGIN, non_inferior=ci(rel)[1] <= NONINF_MARGIN)
        pd.DataFrame([r]).to_csv(os.path.join(OUT_DIR, "tests_noninferiority.csv"), index=False)
        print("\n=== Non-inferiority (lightweight vs best) ===\n", r)

    figures(A, ci_tab, models)
    print(f"\nAll tables and figures written to {OUT_DIR}")


def figures(A, ci_tab, models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {0: "#1b9e77", 1: "#d95f02", 2: "#7570b3"}
    show = [m for m in ci_tab.index if m != "mean"][:6]
    n = len(show)
    if n:
        cols = min(3, n)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
        for ax, mdl in zip(axes.flat, show):
            d = A[(A.model == mdl) & (A.protocol == "grouped")]
            for st, dd in d.groupby("stage"):
                ax.scatter(dd.y, dd.pred, s=14, alpha=0.75, c=colors.get(st, "grey"), label=STAGE_NAMES.get(st, st))
            lim = [0, max(d.y.max(), d.pred.max()) * 1.05]
            ax.plot(lim, lim, "k--", lw=1)
            r = ci_tab.loc[mdl]
            ax.set_title(f"{mdl}\nRMSE={r.RMSE:.0f}, R²={r.R2:.2f}", fontsize=10)
            ax.set_xlabel("Measured biomass (kg/ha)")
            ax.set_ylabel("Predicted biomass (kg/ha)")
            ax.set_xlim(lim); ax.set_ylim(lim)
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        axes.flat[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "fig_pred_vs_obs_grouped.png"), dpi=300)
        plt.close(fig)

        best = show[0]
        d = A[(A.model == best) & (A.protocol == "grouped")]
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
        for st, dd in d.groupby("stage"):
            ax[0].scatter(dd.y, dd.pred - dd.y, s=14, alpha=0.75, c=colors.get(st, "grey"), label=STAGE_NAMES.get(st, st))
        ax[0].axhline(0, c="k", lw=1)
        ax[0].set_xlabel("Measured biomass (kg/ha)"); ax[0].set_ylabel("Residual (pred - measured)")
        ax[0].legend(fontsize=8); ax[0].set_title(f"Residuals: {best}", fontsize=10)
        ax[1].hist(d.pred - d.y, bins=25, color="#666")
        ax[1].set_xlabel("Residual (kg/ha)"); ax[1].set_title("Error distribution", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "fig_residuals_best.png"), dpi=300)
        plt.close(fig)

    inf_p = os.path.join(OUT_DIR, "tests_inflation.csv")
    if os.path.exists(inf_p):
        t = pd.read_csv(inf_p).sort_values("RMSE_grouped")
        x = np.arange(len(t))
        fig, ax = plt.subplots(figsize=(1.1 * len(t) + 2, 3.6))
        ax.bar(x - 0.2, t.RMSE_random, 0.4, label="Random image split", color="#bbb")
        ax.bar(x + 0.2, t.RMSE_grouped, 0.4, label="Grouped by plot", color="#333")
        ax.set_xticks(x); ax.set_xticklabels(t.model, rotation=30, ha="right")
        ax.set_ylabel("RMSE (kg/ha)"); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "fig_random_vs_grouped.png"), dpi=300)
        plt.close(fig)

    st = pd.read_csv(os.path.join(OUT_DIR, "table_per_stage.csv"))
    st = st[(st.protocol == "grouped") & (st.model.isin(show))]
    if len(st):
        piv = st.pivot(index="model", columns="stage", values="nRMSE_%").loc[show]
        ax = piv.plot.bar(figsize=(1.2 * len(show) + 2, 3.6), rot=30)
        ax.set_ylabel("nRMSE (% of stage mean)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "fig_error_by_stage.png"), dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
