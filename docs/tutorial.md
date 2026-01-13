# Tutorial: End-to-End EBSD Deconvolution

This tutorial walks through preparing raw EBSD data, generating synthetic mixtures, training the model, running inference, and inspecting outputs.

## Prerequisites

1. Create and activate a Python environment.
2. Install dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## 1. Prepare raw EBSD data

Keep raw inputs unchanged (BMP/PNG/JPG/TIF supported) and create a canonical 16-bit grayscale PNG copy for processing.

```bash
python3 scripts/prepare_experimental_data.py \
  --input-dir "data/raw/Double Pattern Data" \
  --output-dir "data/processed/Double Pattern Data" \
  --output-format png \
  --output-bit-depth 16
```

Expected outputs:
- `data/processed/Double Pattern Data/` with 16-bit PNG copies
- `data/processed/Double Pattern Data/manifest.json` with checksums and conversion metadata

## 2. Generate synthetic mixed patterns

Use the synthetic generator to create paired A/B/C samples.

```bash
python3 scripts/generate_data.py \
  --config configs/default.yaml \
  --input-dir "data/processed/Double Pattern Data" \
  --output-dir "data/synthetic/tutorial_run" \
  --num-samples 500 \
  --visualize
```

Expected outputs:
- `data/synthetic/tutorial_run/A`, `data/synthetic/tutorial_run/B`, `data/synthetic/tutorial_run/C`
- `data/synthetic/tutorial_run/metadata.csv` with mixing metadata
- `data/synthetic/tutorial_run/config_used.json` with the resolved configuration
- `data/synthetic/tutorial_run/debug/` with visualization panels (when `--visualize` is set)

## 3. Train the model

Train the dual-output U-Net on the synthetic dataset.

```bash
python3 scripts/run_train.py \
  --config configs/train_default.yaml \
  --out_dir outputs/train_tutorial
```

Expected outputs:
- `outputs/train_tutorial/best.pt` and `outputs/train_tutorial/last.pt`
- `outputs/train_tutorial/checkpoint_epoch_*.pt`
- `outputs/train_tutorial/history.json` and `outputs/train_tutorial/output.log`
- `outputs/train_tutorial/monitoring/` (if image logging is enabled)

## 4. Run inference

Apply a trained checkpoint to mixed patterns.

```bash
python3 scripts/run_infer.py \
  --config configs/infer_default.yaml \
  --checkpoint outputs/train_tutorial/best.pt \
  --out_dir outputs/infer_tutorial
```

Expected outputs:
- `outputs/infer_tutorial/A` and `outputs/infer_tutorial/B` with predicted patterns
- `outputs/infer_tutorial/C_hat` with reconstructed mixtures (if enabled)
- `outputs/infer_tutorial/weights.csv` with `x_hat` and `y_hat`
- `outputs/infer_tutorial/config_used.json` and `outputs/infer_tutorial/output.log`

## 5. Visualize outputs

Use your preferred image viewer to compare `A`, `B`, and `C_hat` outputs against the inputs. For quick checks, the debug panels from the synthetic generator (`data/synthetic/.../debug/`) provide side-by-side visuals of A, B, C, and masks.

For deeper analysis, open the mixing notebook:

```bash
jupyter notebook notebooks/mixing_experiments.ipynb
```
