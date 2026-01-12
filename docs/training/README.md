# Training (`scripts/run_train.py`)

This guide explains how to train the current baseline **dual-output U-Net** that predicts `(A, B, x, y)` from a mixed input `C`.

## What the training script does

- Loads paired triplets `(C, A, B, x)` from a synthetic dataset folder (16-bit PNG/TIF + metadata CSV).
- Trains a dual-head U-Net with a combined loss: `L_ab` for `A/B`, `L_recon` for `C_hat = x_hat*A_hat + y_hat*B_hat`, and `L_x` for weight supervision.
- Saves checkpoints (`best.pt`, `last.pt`), metrics history (`history.json`), and the resolved config (`config_used.json`) to an output directory.
- Periodically writes **visual monitoring** samples (8-bit PNGs) and an HTML index with per-epoch summaries and metric plots for quick inspection of training progress.
- Uses GPU automatically when available (`cuda`), otherwise CPU.

## Prerequisites

1. Install dependencies (once):

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

2. Create a synthetic dataset with `A/`, `B/`, `C/` folders.

The training pipeline expects **paired** 16-bit images (PNG/TIF) produced by `scripts/generate_data.py` (recommended).

## Copy/paste commands

### 1) Debug end-to-end smoke test (fast)

Generate a tiny dataset + train quickly:

```bash
python3 scripts/generate_data.py --debug --visualize --seed 7
python3 scripts/run_train.py --debug --out_dir outputs/train_debug_run --seed 7
```

### 2) Baseline training run (typical)

Generate a larger synthetic dataset (example uses the small sample images as inputs):

```bash
python3 scripts/generate_data.py \
  --config configs/default.yaml \
  --input-dir data/code_development_data \
  --output-dir data/synthetic \
  --num-samples 2000 \
  --seed 42
```

Train using the default training config:

```bash
python3 scripts/run_train.py \
  --config configs/train_default.yaml \
  --out_dir outputs/train_run_baseline
```

### 3) Run a custom experiment config

```bash
cp configs/train_default.yaml configs/train_experiment.yaml
# edit configs/train_experiment.yaml
python3 scripts/run_train.py --config configs/train_experiment.yaml --out_dir outputs/train_experiment
```

## Key config fields (recommended starting points)

Training is config-driven. The main knobs live in `configs/train_default.yaml`.

### `data.*` (dataset)

- `data.root_dir`: synthetic dataset root containing `A/`, `B/`, `C/` (typical: `data/synthetic` or `data/synthetic/debug_run`).
- `data.val_split`: `0.1` (typical) or `0.2` (small datasets).
- `data.num_workers`: `0` for debug/Windows; `4` is a common Linux starting point.
- `data.preprocess.mask.enabled`: keep `true` if inputs are circular-detector patterns.
- `data.preprocess.normalize.enabled`: usually `false` if generation already normalizes; enable if training on unnormalized data.
- `data.preprocess.augment.enabled`: `true` once the pipeline is stable; start `false` while validating correctness.
- `data.weights_csv`: optional path to a `metadata.csv` containing `sample_id` and `x` (defaults to `<root_dir>/metadata.csv`).
- `data.require_weights`: keep `true` for synthetic training so `x` is supervised.

### `model.*` (U-Net capacity)

- `model.base_channels`: `32` (default). Use `16` on CPU or if you hit GPU OOM.
- `model.depth`: `4` (default). Use `3` for faster/smaller runs.
- `model.use_batchnorm`: `true` for stable training; turn off for the smallest debug configs.

### `train.*` (optimization)

- `train.batch_size`: start with `8` (default) and reduce if you hit OOM (common range: `2–16`).
- `train.learning_rate`: `1e-3` for Adam; try `3e-4` if training is unstable.
- `train.epochs`: `50` for a baseline; increase for better convergence once data is realistic.
- `train.loss.lambda_recon`: `0.5` (default) encourages physical consistency (`C_hat≈C`); increase if reconstruction fidelity matters more than sharpness.
- `train.loss.lambda_x`: `0.1` (default) sets the weight-supervision strength for `x_hat`.
- `train.grad_clip`: keep `0.0` unless gradients explode (then try `1.0`).
- `train.resume_from`: optional path to a checkpoint to resume training (restores optimizer + scheduler state).

#### One-cycle learning rate policy

The default configs expose a one-cycle scheduler that steps **per iteration** (not per epoch). Configure it under `train.scheduler`:

```yaml
train:
  scheduler:
    name: one_cycle
    max_lr: 2e-4
    pct_start: 0.35
    div_factor: 20
    final_div_factor: 1e4
    anneal_strategy: cos
```

### `logging.image_log.*` (visual monitoring)

- `logging.image_log.enabled`: `true` to write sample images + HTML report.
- `logging.image_log.interval`: log every N epochs (e.g., `1` for debug, `5` for regular runs).
- `logging.image_log.max_samples`: how many samples to log per epoch (typical `2–6`).
- `logging.image_log.sample_strategy`: `fixed` (deterministic), `first`, or `random` (new each interval).
- `logging.image_log.sample_ids`: optional fixed list of sample IDs to track over time.
- `logging.image_log.split`: `val` (default) or `train` source split.
- `logging.image_log.output_dir`: `null` uses `<out_dir>/monitoring`; otherwise set a subfolder.
- `logging.image_log.image_format`: `png` (recommended).
- `logging.image_log.include_recon`: when `true`, logs `C_hat = x_hat*A_pred + y_hat*B_pred`.

Example snippet:

```yaml
logging:
  image_log:
    enabled: true
    interval: 5
    max_samples: 4
    sample_strategy: fixed
    split: val
```

### `output.*` and CLI overrides

- `output.out_dir`: where artifacts are written (use a new folder per experiment).
- CLI `--out_dir ...` overrides `output.out_dir` without editing YAML.
- CLI `--debug` uses `configs/train_debug.yaml` (unless you pass `--config`) and forces debug mode.
- CLI `--seed ...` overrides `debug.seed` for reproducibility.

## Outputs

In `out_dir`:

- `best.pt`: best checkpoint (lowest validation loss, if validation is enabled)
- `last.pt`: latest checkpoint
- `checkpoint_epoch_XXX.pt`: per-epoch checkpoints (controlled by `output.save_every`)
- `history.json`: epoch-wise metrics (train/val loss, PSNR/SSIM/L2 when validation is enabled)
- `monitoring/index.html`: HTML report with per-epoch summary, metric plots, and sample images (8-bit previews) if enabled
- `config_used.json`: resolved config snapshot for reproducibility
- `output.log`: file logs (if `logging.log_to_file: true`)

Open `monitoring/index.html` in a browser to review training progress visually.

## Common issues

- `No paired samples found...`: `data.root_dir` does not contain matching `A/`, `B/`, `C/` triplets.
- `Expected 16-bit image...`: inputs are not `uint16` PNG/TIF; regenerate/convert inputs before training.
- GPU OOM: reduce `train.batch_size`, `model.base_channels`, or `model.depth`.
