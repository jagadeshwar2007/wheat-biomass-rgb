"""Run the whole experiment after step1 has written data/manifest.csv.

Usage:
  python run_all.py            # full run (about 2-3 h on an RTX 4050 laptop GPU)
  python run_all.py --quick    # 1 seed, 15 epochs: a 10-15 minute check that everything works
Each step skips prediction files that already exist, so an interrupted run can simply be restarted.
"""
import argparse
import subprocess
import sys


def run(args):
    print("\n>>> " + " ".join(args), flush=True)
    subprocess.run([sys.executable] + args, check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    seeds = ["0"] if a.quick else ["0", "1", "2"]
    ep = ["--epochs", "15", "--patience", "5"] if a.quick else []
    run(["step2_splits_and_features.py", "--seeds", *seeds])
    run(["step3_classical_models.py", "--seeds", *seeds])
    # frozen DINOv2 features + ridge (fast), then fine-tuned CNNs
    run(["step4_deep_models.py", "--models", "dinov2_s", "dinov2_b", "--seeds", *seeds])
    run(["step4_deep_models.py", "--models", "mobilenetv3", "resnet50", "efficientnet_b0", "--seeds", *seeds, *ep])
    # segmentation ablation: soil pixels blacked out (grouped protocol only)
    run(["step4_deep_models.py", "--mask", "--protocols", "grouped",
         "--models", "mobilenetv3", "resnet50", "dinov2_s", "--seeds", *seeds, *ep])
    run(["step5_evaluate.py"])
    run(["step6_explain_and_speed.py"])
    print("\nDone. Tables and figures are in the 'outputs' folder.")
