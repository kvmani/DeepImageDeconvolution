# DeepImageDeconvolution

ML-based deconvolution of mixed Kikuchi patterns. This repository focuses on generating synthetic mixed patterns (C) from pure patterns (A, B) and building the pipeline needed to train separation models that also recover the mixing weights.

## Mission and Problem Statement

Electron-backscatter diffraction (EBSD) produces high-dimensional Kikuchi patterns. In multi-phase or textured materials, two patterns often overlap, yielding a mixed pattern **C** that is hard to interpret. The mission of this project is to build a reproducible pipeline that generates realistic mixed patterns and trains models to deconvolve **C** back into its constituent pure patterns **A** and **B**, while preserving 16-bit fidelity.

Problem statement:

Given 16-bit grayscale Kikuchi patterns **A** and **B**, form a mixed pattern with weights **x** and **y**:

```
C = normalize(x * A + y * B),  where x + y = 1 and x, y ∈ [0, 1]
```

Given **C**, predict the corresponding **A**, **B**, and the weights **x**, **y**.

Example experimental patterns from the dual-phase steel dataset in `data/raw/Double Pattern Data/` are shown below.

<table>
  <tr>
    <td align="center">
      <img src="data/raw/Double%20Pattern%20Data/50-50%20Double%20Pattern/50-50_0.bmp" width="220" alt="Mixed pattern C (50% BCC, 50% FCC)">
      <br>
      C (mixed)
    </td>
    <td align="center" width="40">&rarr;</td>
    <td align="center">
      <img src="data/raw/Double%20Pattern%20Data/Good%20Pattern/Perfect_BCC-1.bmp" width="220" alt="Pure pattern A (BCC)">
      <br>
      A (pure)
    </td>
    <td align="center">
      <img src="data/raw/Double%20Pattern%20Data/Good%20Pattern/Perfect_FCC-1.bmp" width="220" alt="Pure pattern B (FCC)">
      <br>
      B (pure)
    </td>
  </tr>
</table>

> **Note:** Experimental samples in `data/raw/Double Pattern Data/` include BMPs and may be 8-bit or 32-bit container formats. Keep raw files unchanged and use the preparation script to create a canonical 16-bit grayscale copy under `data/processed/` when needed. See `docs/methods.md` for the bit-depth policy.

## Quickstart (synthetic data generation)

1. Install dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

2. Run the debug data generator using the sample images:

```bash
python3 scripts/generate_data.py --config configs/debug.yaml --debug --visualize
```

Outputs are written to `data/synthetic/debug_run/` with a `metadata.csv` and debug panels under `data/synthetic/debug_run/debug/`.

If you want to use experimental BMP inputs directly, prepare a canonical 16-bit copy first:

```bash
python3 scripts/prepare_experimental_data.py \
  --input-dir "data/raw/Double Pattern Data" \
  --output-dir "data/processed/Double Pattern Data" \
  --output-format png \
  --output-bit-depth 16
```
The script writes a `manifest.json` with checksums and conversion parameters under the output directory.

## Repository structure

- `docs/` project documentation and mission statement
- `data/` raw, processed, and synthetic datasets
- `src/` core modules (preprocessing, generation, models, training)
- `scripts/` CLI entry points (data generation, training, inference)
- `tests/` unit and integration tests
- `configs/` YAML configuration files

## Project Snapshot (features and status)

Features implemented now:

- Synthetic data generation with two mixing pipelines, circular masking, and smart normalization.
- Paired dataset loading for triplets (C, A, B, x) with debug modes.
- Baseline dual-output U-Net training and inference with physics-aware reconstruction and weight supervision.
- Metrics (L1, L2, PSNR, SSIM) and HTML monitoring reports with per-epoch plots.
- Interactive mixing experiments notebook.

Current status:

- End-to-end synthetic pipeline is functional and tested in debug mode.
- Monitoring artifacts are written to `<out_dir>/monitoring` during training.
- Experimental double-pattern data is included for reference and validation.
- Advanced models (GAN/attention), orientation metrics, and packaging are still pending.

Quick links:

- `docs/status.md` for the read-only snapshot of the codebase.
- `docs/networks/dual_unet.md` for the baseline Dual U-Net architecture details.
- `todo_list.md` for the current work list.
- `docs/roadmap.md` for long-term phases and deliverables.
- `docs/references.md` for centralized literature citations.
- `docs/methods.md` for manuscript-style methods and evaluation protocol.

## Data generation overview

The generator reads 16-bit PNG/TIF/BMP patterns, validates bit depth, optionally preprocesses (crop, denoise, circular mask, normalize, augment), and mixes pairs into synthetic patterns. If inputs are not 16-bit, prepare them first (see `scripts/prepare_experimental_data.py`). Two pipelines are supported:

- `normalize_then_mix`: normalize A and B, then mix using weights.
- `mix_then_normalize`: mix raw intensities, then normalize the mixture.

Circular masks are applied by default using the maximum inscribed circle centered in the image. The pipeline detects whether inputs appear already masked and records this in metadata. Normalization uses a **smart** option by default, computing min/max (or percentiles) only inside the active circular region to avoid masked zeros skewing the scale. This is a project-specific (non-standard) normalization option; disable it via `data.preprocess.normalize.smart: false` if needed.

Debug mode produces annotated visual panels showing A, B, C, the mask boundary, and a difference map for quick verification.

CLI overrides include `--no-mask` and `--no-smart-normalize` if you need standard (unmasked or global) behavior.

See `docs/data_generation.md` for detailed configuration options and workflow guidance.

Raw experimental dual-phase patterns are documented in [`data/raw/Double Pattern Data/README.md`](data/raw/Double%20Pattern%20Data/README.md).

## Mixing experiments notebook

Interactive sandbox for mixing A/B into C (pipeline + `wA` slider) that mirrors the knobs in `scripts/generate_data.py`. It defaults to experimental BCC/FCC reference patterns from `data/raw/Double Pattern Data/Good Pattern/` and does not require a target C. Open [`notebooks/mixing_experiments.ipynb`](notebooks/mixing_experiments.ipynb) via `jupyter notebook notebooks/mixing_experiments.ipynb` and use the UI to explore settings.

See `docs/mixing_experiments.md` for usage notes.

## Training and inference

Train the dual-output U-Net with weight prediction with:

```bash
python3 scripts/run_train.py --config configs/train_default.yaml --out_dir outputs/train_run
```

To standardize run naming, append a timestamped tag:

```bash
python3 scripts/run_train.py --config configs/train_default.yaml --run-tag baseline_01
```

Run inference with a checkpoint:

```bash
python3 scripts/run_infer.py --config configs/infer_default.yaml --checkpoint outputs/train_run/best.pt
```

Inference saves predicted `A`/`B` images, reconstructed `C_hat`, and a `weights.csv` with `x_hat`/`y_hat` per sample (when enabled).

See [`docs/training/README.md`](docs/training/README.md) for detailed training commands and recommended settings. See [`docs/training_inference.md`](docs/training_inference.md) for tensor shapes and inference notes.

When enabled, training also writes an HTML image log under `<out_dir>/monitoring` for quick visual inspection.

## Documentation

See `docs/mission_statement.md` and `docs/roadmap.md` for overall goals and phased deliverables. For a concise snapshot of the current implementation, read `docs/status.md` and keep `todo_list.md` updated with the active work list.
