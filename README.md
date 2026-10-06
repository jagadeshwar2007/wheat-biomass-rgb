# Wheat biomass estimation from UAV RGB images

Code and results for the paper *"Wheat Biomass Estimation from UAV RGB Images: A Plot-Held-Out Comparison of Classical, CNN and Vision-Transformer Models"* (under submission).

We compare ridge, Random Forest and XGBoost on handcrafted colour/texture features, fine-tuned CNNs (ResNet-50, EfficientNet-B0, MobileNetV3-Large) and frozen DINOv2 features for estimating wheat dry above-ground biomass. Every model is evaluated on plots never seen in training (grouped 5-fold cross-validation), with a random image split, leave-one-cultivar-out, a soil-masking ablation, Grad-CAM and latency for comparison.

## Main results (grouped 5-fold CV, predictions averaged over 3 seeds)

| Model | RMSE (kg/ha) [95% CI] | R² |
|---|---|---|
| ResNet-50 | 852 [719, 975] | 0.90 |
| Ridge (49 handcrafted features) | 981 [848, 1105] | 0.87 |
| DINOv2-B (frozen) + ridge | 989 [861, 1123] | 0.87 |
| EfficientNet-B0 | 1023 [889, 1156] | 0.86 |
| XGBoost | 1065 [915, 1202] | 0.85 |
| DINOv2-S (frozen) + ridge | 1068 [944, 1184] | 0.84 |
| Random Forest | 1134 [996, 1278] | 0.82 |
| MobileNetV3-Large | 1308 [1090, 1532] | 0.77 |
| Canopy cover (linear) | 2379 [2190, 2570] | 0.22 |

All tables and figures are in [`results/`](results/). Per-seed means and SDs are in `results/table_main.csv`.

## Data

Brazilian Wheat Dataset (L. Schreiber, Mendeley Data, doi:[10.17632/3ntkg88d4d.1](https://doi.org/10.17632/3ntkg88d4d.1), CC BY 4.0). The dataset is **not** included here; download it from Mendeley and unzip it, e.g. to `C:\wheat`.

Note: one flowering date in `RGB DATA.CSV` is recorded as `2020-09-20`; it belongs to 2018. `step1_build_manifest.py` pairs image dates with CSV dates by matching plot IDs, so the typo does not need fixing by hand.

## How to reproduce

Requirements: Python 3.10–3.13 (64-bit), an NVIDIA GPU is recommended. Full instructions are in [`code/README.md`](code/README.md).

```
cd code
python -m venv venv
venv\Scripts\activate            (Windows)   |   source venv/bin/activate   (Linux/macOS)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python step1_build_manifest.py "C:\wheat"
python run_all.py
```

A full run takes about 2 hours on an RTX 4050 laptop GPU. `python run_all.py --quick` runs a short check first.

## Repository layout

```
code/      experiment scripts (step0–step6, run_all.py), aov_by_stage.py (per-stage Grad-CAM analysis),
           step7_temporal_resnet.py (multi-temporal ResNet-50 with temporal attention), step8_temporal_analysis.py
results/   result tables (CSV) and figures (PNG) from the full run
```

## Citation

[Citation will be added after publication.]

## License

Code: MIT (see `LICENSE`). Dataset: CC BY 4.0, by its original authors.
