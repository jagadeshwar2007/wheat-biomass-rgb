"""Post-hoc: AoV vs vegetation fraction per growth stage (from outputs/gradcam_per_image.csv)."""
import sys
import pandas as pd
from scipy.stats import wilcoxon
g = pd.read_csv(sys.argv[1])
names = {0: "V6", 1: "Three nodes", 2: "Flowering"}
rows = []
for (m, s), d in g.dropna(subset=["AoV"]).groupby(["model", "stage"]):
    diff = d.AoV - d.veg_fraction
    rows.append(dict(model=m, stage=names[s], n=len(d), AoV_mean=d.AoV.mean(), veg_mean=d.veg_fraction.mean(),
                     median_diff=diff.median(), share_AoV_gt_veg=(diff > 0).mean(),
                     p_greater=wilcoxon(diff, alternative="greater").pvalue,
                     p_less=wilcoxon(diff, alternative="less").pvalue))
print(pd.DataFrame(rows).round(4).to_string())
pd.DataFrame(rows).to_csv(sys.argv[2], index=False)
