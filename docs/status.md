# Project Snapshot

This document gives a quick, read-only snapshot of the current codebase: what works, what is still in progress, and where to look first.

## Purpose

Build a reproducible pipeline to generate synthetic mixed Kikuchi patterns (C) from pure patterns (A, B) and train a model to deconvolve C back into A and B while predicting mixing weights x and y, preserving 16-bit fidelity.

## Current Features (Implemented)

- Synthetic data generation with two pipelines (`normalize_then_mix`, `mix_then_normalize`), optional blur/noise, circular masking, and smart normalization.
- Paired dataset loaders for triplets (C, A, B, x) with debug sampling controls.
- Baseline dual-output U-Net model with physics-aware reconstruction and weight supervision.
- Training and inference CLI wrappers driven by YAML configs and debug modes.
- Metrics in training (L1, L2, PSNR, SSIM) and periodic HTML monitoring with per-epoch summaries and plots.
- Interactive mixing experiments notebook for exploring mixing strategies and parameters.
- Unit and smoke tests covering preprocessing, mixing, data generation, training, and inference.

## Current Status

- End-to-end pipeline is functional on synthetic data (debug configs validated).
- Monitoring artifacts are written under `<out_dir>/monitoring` during training.
- Real experimental patterns are included for reference in `data/raw/Double Pattern Data/`.
- Advanced architectures (GANs/attention), orientation-based metrics, and production packaging are not implemented yet.

## Codebase Map (Where to Look)

- `scripts/generate_data.py`: generate synthetic datasets from pure patterns.
- `scripts/run_train.py`: train the dual-output U-Net.
- `scripts/run_infer.py`: run inference on mixed patterns.
- `src/generation/`: mixing pipelines and dataset construction.
- `src/preprocessing/`: 16-bit handling, masking, normalization, transforms.
- `src/models/unet_dual.py`: baseline model architecture.
- `src/training/train.py`: training loop, metrics, checkpoints, monitoring.
- `src/inference/infer.py`: inference pipeline.
- `configs/`: default and debug YAML configs.
- `docs/`: detailed documentation for data generation, training, inference, and notebooks.

## To-do Snapshot

Main next tasks (summary):

- Build a larger synthetic dataset and record exact configs.
- Run a baseline training and capture metrics plus monitoring output.
- Add an evaluation report CLI and CI smoke tests.

See `todo_list.md` for the up-to-date task list and priorities.
