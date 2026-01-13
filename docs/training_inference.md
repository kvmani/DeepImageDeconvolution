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
- `config_used.json` with the resolved configuration
- `output.log` (if `logging.log_to_file: true`)

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

Outputs in `out_dir`:

- `A/` and `B/` predicted 16-bit PNG patterns
- `C_hat/` reconstructed mixture `x_hat * A_hat + y_hat * B_hat` (optional)
- `weights.csv` with `x_hat` and `y_hat` per sample (optional)
- `config_used.json` and `output.log` (if enabled)

## Logging

Important outputs are logged to screen by default. Set `logging.log_to_file: true` to also write logs to `out_dir/output.log`.

To append a timestamped run directory:

```bash
python3 scripts/run_train.py --config configs/train_default.yaml --run-tag baseline_01
python3 scripts/run_infer.py --config configs/infer_default.yaml --checkpoint outputs/train_run/best.pt --run-tag eval_01
```
