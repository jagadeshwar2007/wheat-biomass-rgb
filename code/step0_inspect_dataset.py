"""Step 0: print the structure of the downloaded Brazilian Wheat Dataset.

Usage:  python step0_inspect_dataset.py "C:/path/to/Brazilian Wheat Dataset"
Writes the same text to outputs/dataset_inspection.txt. If step1 cannot build
the manifest automatically, use that file to check the column names and folder layout.
"""
import os
import sys

import pandas as pd

from common import OUT_DIR, ensure_dirs

IMG_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def main(root):
    ensure_dirs()
    lines = [f"Dataset root: {root}"]
    n_img = 0
    for dirpath, dirnames, files in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        imgs = [f for f in files if f.lower().endswith(IMG_EXT)]
        others = [f for f in files if not f.lower().endswith(IMG_EXT)]
        n_img += len(imgs)
        lines.append(f"\n[{rel}]  {len(imgs)} images, {len(others)} other files")
        for f in sorted(imgs)[:8]:
            lines.append(f"   img: {f}")
        if len(imgs) > 8:
            lines.append(f"   ... and {len(imgs) - 8} more; last: {sorted(imgs)[-1]}")
        for f in sorted(others)[:20]:
            lines.append(f"   file: {f}")
            if f.lower().endswith(".csv"):
                p = os.path.join(dirpath, f)
                for sep in (",", ";", "\t"):
                    try:
                        df = pd.read_csv(p, sep=sep, nrows=200, encoding_errors="replace")
                        if df.shape[1] > 1:
                            break
                    except Exception:
                        df = None
                if df is not None:
                    lines.append(f"      columns ({df.shape[1]}): {list(df.columns)}")
                    lines.append("      first rows:\n" + df.head(5).to_string())
                    for c in df.columns:
                        if df[c].nunique() <= 12:
                            lines.append(f"      unique {c!r}: {sorted(map(str, df[c].unique()))}")
    lines.append(f"\nTotal images: {n_img}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT_DIR, "dataset_inspection.txt"), "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root = sys.argv[1].strip('"').strip("'")
    if os.path.isfile(root):
        root = os.path.dirname(root)
    if not os.path.isdir(root):
        sys.exit(f"Folder not found: {root}\nCheck the path (use the folder that contains RGB DATA.CSV).")
    main(root)
