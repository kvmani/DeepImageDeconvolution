# Training and Inference Guide

This guide documents the U-Net baseline training and inference pipelines.

For detailed training commands and config guidance, see [`docs/training/README.md`](training/README.md).

## Training

Run training with a config file:

```bash
python3 scripts/run_train.py --config configs/train_default.yaml --out_dir outputs/train_run
```

Debug mode runs quickly on small datasets:

```bash
python3 scripts/run_train.py --config configs/train_debug.yaml --debug
```

Outputs in `out_dir`:

- `best.pt` and `last.pt` checkpoints
- `history.json` with epoch metrics (including PSNR/SSIM, L2/MSE, and mask-aware variants when masking is enabled)
- `history.csv` with per-epoch metrics in CSV form
- `report.json` with a machine-readable run summary (updated each epoch with status/progress and tracking-sample paths)
- `config_used.json` with the resolved configuration
- `output.log` (if `logging.log_to_file: true`)
- `manifest.json` with run metadata, timing, and summary counts
- `monitoring/loss_curve.png`, `monitoring/qual_grid_latest.png`, and optional metrics/weights plots
 - `monitoring/image_log.json` and `monitoring/index.html` when image logging is enabled (used for consistent A/B prediction tracking)

### Expected tensor shapes (256x256 input)

With input mixed pattern **C** shaped `(B, 1, 256, 256)` and `depth=4`, `base_channels=32`, the U-Net flows as:

- Encoder stage 1: `(B, 32, 256, 256)`
- Encoder stage 2: `(B, 64, 128, 128)`
- Encoder stage 3: `(B, 128, 64, 64)`
- Encoder stage 4: `(B, 256, 32, 32)`
- Bottleneck: `(B, 512, 16, 16)`
- Decoder output (per head): `(B, 32, 256, 256)`
- Final outputs `A_pred`, `B_pred`: `(B, 1, 256, 256)`
- Weight head output `x_hat`: `(B, 1)` with `y_hat = 1 - x_hat`

## Inference

Run inference on mixed patterns (C) with a trained checkpoint:

```bash
python3 scripts/run_infer.py --config configs/infer_default.yaml --checkpoint outputs/train_run/best.pt
```

### CLI arguments overview

Key CLI arguments (run `python3 scripts/run_infer.py --help` for the full list):

- `--config`: YAML config for inference defaults.
- `--checkpoint`: Path to the trained checkpoint (`best.pt` or `last.pt`).
- `--out_dir`: Optional output directory override.
- `--run-tag`: Optional run tag appended to output dir (timestamped).
- `--run-id`: Optional run identifier included in logs and manifests.
- `--debug`: Enable debug mode (small sample set + verbose logging).
- Logging flags: `--log-level`, `--log-file`, `--quiet`.

### Configuration essentials

The inference YAML config mirrors training settings but focuses on model loading, preprocessing, and
output controls. Important fields include:

- `inference.checkpoint`: checkpoint path (overridden by `--checkpoint`).
- `inference.batch_size`: batch size for prediction.
- `inference.device`: `auto`, `cpu`, or `cuda`.
- `inference.clamp_outputs`: clamp predictions to `[0, 1]`.
- `data.mixed_dir`: root directory for mixed patterns `C` (batch mode).
- `data.extensions`: allowed file extensions.
- `data.preprocess`: crop/mask/normalize pipeline applied to inputs (and GT metrics).
- `postprocess.apply_mask`: whether to apply the circular mask to outputs.
- `output.out_dir`: base output directory.
- `output.save_recon`: save reconstructed `C_hat`.
- `output.save_weights`: write `weights.csv` with `x_hat/y_hat`.

### Inputs

**Batch mode** (default) reads from `data.mixed_dir` and scans for files with `_C` suffixes. To run
inference on a specific dataset, update the config to point at the desired folder.

Supported formats: PNG/TIFF/BMP/JPG. Non-16-bit inputs are rescaled to a canonical 16-bit range
before normalization.

### Ground truth metrics

When GT data is provided (via the optional GT fields in `infer_default.yaml` or in evaluation
scripts), metrics are computed after applying the same preprocessing pipeline configured for the
inputs. This ensures crops, masks, and normalization steps are consistent between predictions and
GT. The mask is applied when enabled to keep metrics inside the detector region.

Outputs in `out_dir`:

- `A/` and `B/` predicted 16-bit PNG patterns
- `C_hat/` reconstructed mixture `x_hat * A_hat + y_hat * B_hat` (optional)
- `weights.csv` with `x_hat` and `y_hat` per sample (optional)
- `report.json` with a machine-readable run summary
- `monitoring/qual_grid_infer.png` and `monitoring/weights_hist.png`
- `config_used.json` and `output.log` (if enabled)
- `manifest.json` with run metadata, timing, and summary counts

### Troubleshooting

- **Shape mismatches**: ensure GT preprocessing matches `data.preprocess` (crop/mask/normalize).
- **CUDA not available**: set `inference.device: cpu` or allow auto fallback.
- **Empty output directories**: confirm `data.mixed_dir` contains files with the expected suffix and
  `data.extensions` includes the file types you have.

## Real-data evaluation

Evaluate a checkpoint against real mixed patterns with masked metrics and a qualitative grid:

```bash
python3 scripts/evaluate_real_data.py \
  --config configs/eval_real_default.yaml \
  --checkpoint outputs/train_run/best.pt \
  --out_dir outputs/eval_real
```

## Logging

Important outputs are logged to screen by default. Set `logging.log_to_file: true` to also write logs to `out_dir/output.log`. CLI scripts also accept `--log-level`, `--log-file`, and `--quiet` flags to control verbosity (with `--debug` enabling debug mode and verbose logging).

To append a timestamped run directory:

```bash
python3 scripts/run_train.py --config configs/train_default.yaml --run-tag baseline_01
python3 scripts/run_infer.py --config configs/infer_default.yaml --checkpoint outputs/train_run/best.pt --run-tag eval_01
```
