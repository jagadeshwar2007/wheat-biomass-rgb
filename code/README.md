# Wheat biomass from UAV RGB images: experiment code

This code runs every experiment for the paper on your laptop (Windows or Linux, NVIDIA GPU such as an RTX 4050).
You run four commands. A full run takes **about 2–3 hours** of GPU time and does not need you to watch it.

## 1. One-time setup (about 15 minutes)

1. Install **Python 3.10 to 3.13 (64-bit)** (python.org). On Windows, tick "Add Python to PATH".
2. Open a terminal (Windows: "Command Prompt") **inside this `code` folder** and create an environment:
   ```
   python -m venv venv
   venv\Scripts\activate          (Windows)
   source venv/bin/activate       (Linux/macOS)
   ```
3. Install PyTorch **with CUDA** (this is the step that makes it use your GPU):
   ```
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   ```
4. Install the rest:
   ```
   pip install -r requirements.txt
   ```
5. Check that the GPU is visible. This should print `True NVIDIA GeForce RTX 4050 ...`:
   ```
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   If it prints `False`, update the NVIDIA driver and repeat step 3.

## 2. Get the dataset

Download "Brazilian Wheat Dataset" from https://data.mendeley.com/datasets/3ntkg88d4d/1 (button **Download All**, no login needed) and unzip it anywhere, for example `C:\data\BrazilianWheat`.

## 3. Link images to biomass

```
python step1_build_manifest.py "C:\data\BrazilianWheat"
```
It prints a table of biomass by growth stage. **Biomass should increase from the first to the last stage.**
* If it says `Wrote ... manifest.csv with 264 rows`, continue.
* If it stops with an error or prints a WARNING, run the command below and check `outputs\dataset_inspection.txt` for the CSV columns and image folders:
  ```
  python step0_inspect_dataset.py "C:\data\BrazilianWheat"
  ```

## 4. Quick check (10–15 minutes)

```
python run_all.py --quick
```
The first run downloads pretrained model weights (about 0.5 GB) from Hugging Face. If this finishes without errors, everything works. Quick-run numbers are **not** for the paper.

## 5. Full run (about 2–3 hours)

```
python run_all.py
```
If it is interrupted, run the same command again: finished parts are skipped.
(For a fresh start, delete the `outputs\predictions` folder, since `run_all.py` reuses existing predictions.)

## 6. Results

All tables (`table_*.csv`, `tests_*.csv`) and figures (`fig_*.png`) are written to the `outputs` folder. The `checkpoints` subfolder holds model weights and is large.

## What each step does

| Script | What it does |
|---|---|
| `step1_build_manifest.py` | Matches each plot image to its measured dry biomass, plot, growth stage and variety |
| `step2_splits_and_features.py` | Makes the cross-validation folds shared by all models (grouped by plot, random, leave-one-variety-out) and computes colour, vegetation-index and texture features |
| `step3_classical_models.py` | Mean predictor, canopy-cover regression, ridge, Random Forest, XGBoost |
| `step4_deep_models.py` | DINOv2 (frozen) + ridge; fine-tuned MobileNetV3, ResNet-50, EfficientNet-B0; `--mask` removes soil pixels |
| `step5_evaluate.py` | RMSE, MAE, R², nRMSE, MAPE, bias, CCC; bootstrap confidence intervals; Wilcoxon tests with Holm correction; figures |
| `step6_explain_and_speed.py` | Grad-CAM heatmaps with an attribution-on-vegetation test; model size and CPU/GPU latency |

## Notes on rigour

* **No plot leakage.** In the primary (`grouped`) protocol, all three growth-stage images of a plot are always in the same fold. The `random` protocol exists only to measure how much a leaky split inflates results.
* **No tuning on test data.** Hyperparameters and early stopping use an inner split of the training folds only.
* **Repeated runs.** Three seeds per configuration; tables report mean ± SD, and CIs come from a plot-level bootstrap.
* **Out of memory on the GPU?** Run `python step4_deep_models.py --batch 8 ...` with the same other arguments.
