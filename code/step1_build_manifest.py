"""Step 1: match each cut-plot image to its measured biomass and write data/manifest.csv.

Usage:  python step1_build_manifest.py "C:/path/to/Brazilian Wheat Dataset"

The dataset's file naming is not documented, so this script tries several
matching strategies and prints what it did. Check the printed summary:
biomass should rise from the first to the last growth stage. If matching
fails, run step0_inspect_dataset.py and send outputs/dataset_inspection.txt.
"""
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from common import DATA_DIR, MANIFEST, ROOT, ensure_dirs

IMG_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
PAT = {
    "biomass": r"biomass|biomassa|kg",
    "plot": r"plot|parcela|parc\b|unidade|^id$",
    "stage": r"date|data|stage|est[aá]dio|coleta|^day|^dia|\bdap\b|\bdas\b|phase|fase",
    "variety": r"variet|cultivar|genot|^cv",
    "nitrogen": r"nitro|^n$|n[_ ]?rate|dose|treat|tratamento",
}
STAGE_WORDS = [("v6", 0), ("six", 0), ("seis", 0), ("node", 1), ("no", 1), ("n[oó]s", 1),
               ("flor", 2), ("flower", 2), ("anth", 2)]


def read_csv_any(p):
    best = None
    for sep in (",", ";", "\t"):
        for dec in (".", ","):
            try:
                df = pd.read_csv(p, sep=sep, decimal=dec, encoding_errors="replace")
            except Exception:
                continue
            if best is None or df.shape[1] > best.shape[1]:
                best = df
    return best


def find_col(df, key, exclude=()):
    for c in df.columns:
        if c in exclude:
            continue
        if re.search(PAT[key], str(c).strip().lower()):
            return c
    return None


def parse_date(s):
    s = str(s)
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y%m%d", "%d-%m-%Y", "%d.%m.%Y", "%Y_%m_%d",
                "%d_%m_%Y", "%d/%m/%y"):
        m = re.search(r"\d{1,4}[-/._]?\d{1,2}[-/._]?\d{2,4}", s)
        if not m:
            return None
        try:
            return datetime.strptime(m.group(0), fmt)
        except ValueError:
            continue
    return None


def order_tokens(tokens):
    """Map stage tokens (dates, numbers or names) to 0..k-1 in chronological order."""
    tokens = list(dict.fromkeys(tokens))
    dates = [parse_date(t) for t in tokens]
    if all(d is not None for d in dates):
        ranked = sorted(tokens, key=lambda t: parse_date(t))
        return {t: i for i, t in enumerate(ranked)}
    word_rank = []
    for t in tokens:
        tl = str(t).lower()
        r = next((v for w, v in STAGE_WORDS if re.search(w, tl)), None)
        word_rank.append(r)
    if all(r is not None for r in word_rank) and len(set(word_rank)) == len(tokens):
        return {t: r for t, r in zip(tokens, word_rank)}
    nums = []
    for t in tokens:
        m = re.findall(r"\d+", str(t))
        nums.append(int(m[0]) if m else None)
    if all(n is not None for n in nums) and len(set(nums)) == len(tokens):
        ranked = sorted(tokens, key=lambda t: int(re.findall(r"\d+", str(t))[0]))
        return {t: i for i, t in enumerate(ranked)}
    return {t: i for i, t in enumerate(sorted(map(str, tokens)))}


def pair_tokens(img_plots, csv_plots):
    """Pair image stage tokens with CSV stage/date values.

    Chooses the one-to-one pairing that matches the most plot IDs; ties go to the pairing that keeps
    both sides in the same chronological order. Robust to typos in a CSV date (the Brazilian Wheat
    CSV lists one flowering date as 2020-09-20 instead of 2018) because it does not rely on order.
    """
    from itertools import permutations
    toks, vals = list(img_plots), list(csv_plots)
    tr, vr = order_tokens(toks), order_tokens(vals)
    best = None
    for perm in permutations(vals, len(toks)) if len(toks) <= len(vals) else []:
        score = sum(len(img_plots[t] & csv_plots[v]) for t, v in zip(toks, perm))
        order_ok = sum(tr[t] == vr[v] for t, v in zip(toks, perm))
        if best is None or (score, order_ok) > best[0]:
            best = ((score, order_ok), dict(zip(toks, perm)))
    if best is None:
        sys.exit(f"More image stage tokens ({len(toks)}) than CSV stage values ({len(vals)}).")
    return best[1]


def list_images(root):
    imgs = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMG_EXT) and not re.search(r"example|mask|roi", f.lower()):
                imgs.append(os.path.join(dirpath, f))
    cut = [p for p in imgs if re.search(r"cut", os.path.relpath(p, root).lower())]
    return sorted(cut if cut else imgs)


def main(root):
    ensure_dirs()
    imgs = list_images(root)
    print(f"Found {len(imgs)} candidate plot images")
    csvs = [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs if f.lower().endswith(".csv")]
    cands = []
    for p in csvs:
        df = read_csv_any(p)
        if df is not None and find_col(df, "biomass") is not None:
            cands.append((abs(len(df) - len(imgs)), p, df))
    if not cands:
        sys.exit("No CSV with a biomass column found. Run step0 and send outputs/dataset_inspection.txt.")
    cands.sort(key=lambda x: x[0])
    _, csv_path, df = cands[0]
    print(f"Using CSV: {csv_path}  ({len(df)} rows)\nColumns: {list(df.columns)}")
    c_bio = find_col(df, "biomass")
    c_plot = find_col(df, "plot")
    c_stage = find_col(df, "stage", exclude=(c_plot, c_bio))
    c_var = find_col(df, "variety", exclude=(c_plot, c_bio, c_stage))
    c_n = find_col(df, "nitrogen", exclude=(c_plot, c_bio, c_stage, c_var))
    print(f"biomass={c_bio!r} plot={c_plot!r} stage={c_stage!r} variety={c_var!r} nitrogen={c_n!r}")

    rows = None
    # Strategy A: a CSV column holds the image file names
    stems = [os.path.splitext(os.path.basename(p))[0].lower() for p in imgs]
    names = dict(zip(stems, imgs))
    unique_stems = len(set(stems)) == len(stems)  # file names alone identify an image
    for c in (df.columns if unique_stems else []):
        vals = df[c].astype(str).map(lambda s: os.path.splitext(os.path.basename(s))[0].lower())
        if vals.isin(names).mean() > 0.9 and vals.is_unique:
            print(f"Strategy A: matched by file-name column {c!r}")
            df = df[vals.isin(names)].copy()
            df["image_path"] = vals[vals.isin(names)].map(names)
            rows = df
            break
    # Strategy B: (stage token from folder or file name, plot number from file name)
    if rows is None and c_plot is not None and c_stage is not None:
        info = []
        for p in imgs:
            rel = os.path.relpath(p, root)
            stem = os.path.splitext(os.path.basename(p))[0]
            nums = re.findall(r"\d+", stem)
            if not nums:
                continue
            plot = int(nums[-1])
            parent = os.path.dirname(rel)
            token = parent if parent and len({os.path.dirname(os.path.relpath(q, root)) for q in imgs}) > 1 \
                else re.sub(r"\d+$", "", stem).strip("_- ") or stem
            info.append((p, token, plot))
        from collections import Counter
        print(f"Images per stage token: {dict(Counter(t for _, t, _ in info))}")
        df["_plot"] = pd.to_numeric(df[c_plot].astype(str).str.extract(r"(\d+)")[0], errors="coerce").astype("Int64")
        img_plots = {}
        for _, t, pl in info:
            img_plots.setdefault(t, set()).add(pl)
        csv_plots = {v: set(int(x) for x in g["_plot"].dropna()) for v, g in df.groupby(c_stage, sort=False)}
        print(f"CSV rows per {c_stage}: {df[c_stage].value_counts(sort=False).to_dict()}")
        pairing = pair_tokens(img_plots, csv_plots)
        print("Image token -> CSV value (shared plot IDs):")
        for t, v in pairing.items():
            print(f"   {t} -> {v}  ({len(img_plots[t] & csv_plots[v])} of {len(img_plots[t])} images)")
        key = {(pairing[t], pl): p for p, t, pl in info if t in pairing}
        df["image_path"] = [key.get((v, int(pl))) if pd.notna(pl) else None
                            for v, pl in zip(df[c_stage], df["_plot"])]
        df["_date_idx"] = df[c_stage].map({pairing[t]: i for t, i in order_tokens(list(pairing)).items()})
        matched = df["image_path"].notna().sum()
        print(f"Strategy B: matched {matched}/{len(df)} rows")
        if matched >= 0.9 * len(df):
            rows = df[df["image_path"].notna()].copy()
    # Strategy C: same count, sorted order (last resort, needs manual check)
    if rows is None:
        sys.exit("Could not match images to biomass rows automatically.\n"
                 "Run: python step0_inspect_dataset.py <dataset folder> and send "
                 "outputs/dataset_inspection.txt to check the file layout.")

    out = pd.DataFrame({
        "image_path": rows["image_path"].map(lambda p: os.path.relpath(p, ROOT) if os.path.splitdrive(p)[0] ==
                                             os.path.splitdrive(ROOT)[0] else p),
        "biomass": pd.to_numeric(rows[c_bio], errors="coerce"),
    })
    out["plot_id"] = rows[c_plot].values if c_plot else np.arange(len(rows))
    c_stage_name = next((c for c in rows.columns if str(c).strip().lower() in ("stage", "estadio", "estádio")), None)
    word_stage = None
    if c_stage_name:
        word_stage = rows[c_stage_name].astype(str).map(
            lambda t: next((v for w, v in STAGE_WORDS if re.search(w, t.lower())), None))
    if word_stage is not None and word_stage.notna().all():
        out["stage"] = word_stage.astype(int).values  # V6=0, three nodes=1, flowering=2
    elif "_date_idx" in rows:
        out["stage"] = rows["_date_idx"].values
    elif c_stage:
        out["stage"] = rows[c_stage].map(order_tokens(rows[c_stage].tolist())).values
    else:
        out["stage"] = 0
    label_col = c_stage_name or c_stage
    out["stage_label"] = rows[label_col].astype(str).values if label_col else "all"
    out["date"] = rows[c_stage].astype(str).values if c_stage else ""
    if c_stage and c_stage_name and c_stage_name != c_stage:
        print("Stage names per date:", {k: sorted(set(v)) for k, v in out.groupby("date")["stage_label"]})
    out["variety"] = rows[c_var].astype(str).values if c_var else "unknown"
    out["nitrogen"] = rows[c_n].values if c_n else np.nan
    out = out.dropna(subset=["biomass"]).reset_index(drop=True)
    out.to_csv(MANIFEST, index=False)

    print(f"\nWrote {MANIFEST} with {len(out)} rows, {out.plot_id.nunique()} plots")
    print("\nBiomass by stage (should increase with stage):")
    print(out.groupby(["stage", "stage_label"])["biomass"].describe()[["count", "mean", "min", "max"]])
    print("\nVarieties:", out.variety.value_counts().to_dict())
    means = out.groupby("stage")["biomass"].mean().values
    if len(means) > 1 and not np.all(np.diff(means) > 0):
        print("\nWARNING: mean biomass does not increase with stage. The stage mapping may be wrong;"
              " check outputs/dataset_inspection.txt before running experiments.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root = sys.argv[1].strip('"').strip("'")
    if os.path.isfile(root):
        root = os.path.dirname(root)
    if not os.path.isdir(root):
        sys.exit(f"Folder not found: {root}\nCheck the path (use the folder that contains RGB DATA.CSV).")
    main(root)
