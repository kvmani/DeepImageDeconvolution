# DeepImageDeconvolution

ML-based deconvolution of mixed Kikuchi patterns. This repository focuses on generating synthetic mixed patterns (C) from pure patterns (A, B) and building the pipeline needed to train separation models.

## Mission and Problem Statement

Electron-backscatter diffraction (EBSD) produces high-dimensional Kikuchi patterns. In multi-phase or textured materials, two patterns often overlap, yielding a mixed pattern **C** that is hard to interpret. The mission of this project is to build a reproducible pipeline that generates realistic mixed patterns and trains models to deconvolve **C** back into its constituent pure patterns **A** and **B**, while preserving 16-bit fidelity.

Problem statement:

Given 16-bit grayscale Kikuchi patterns **A** and **B**, form a mixed pattern:

```
C = normalize(wA * A + wB * B),  where wA + wB = 1
```

Given **C**, predict the corresponding **A** and **B**.

<table>
  <tr>
    <td align="center">
      <img src="data/code_development_data/image_C.png" width="220" alt="Mixed pattern C">
      <br>
      C (mixed)
    </td>
    <td align="center" width="40">&rarr;</td>
    <td align="center">
      <img src="data/code_development_data/image_A.png" width="220" alt="Pure pattern A">
      <br>
      A (pure)
    </td>
    <td align="center">
      <img src="data/code_development_data/image_B.png" width="220" alt="Pure pattern B">
      <br>
      B (pure)
    </td>
  </tr>
</table>

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

## Repository structure

- `docs/` project documentation and mission statement
- `data/` raw, processed, and synthetic datasets
- `src/` core modules (preprocessing, generation, models, training)
- `scripts/` CLI entry points (data generation, training, inference)
- `tests/` unit and integration tests
- `configs/` YAML configuration files

## Data generation overview

The generator reads 16-bit PNG/TIF patterns, validates bit depth, optionally preprocesses (crop, denoise, circular mask, normalize, augment), and mixes pairs into synthetic patterns. Two pipelines are supported:

- `normalize_then_mix`: normalize A and B, then mix using weights.
- `mix_then_normalize`: mix raw intensities, then normalize the mixture.

Circular masks are applied by default using the maximum inscribed circle centered in the image. The pipeline detects whether inputs appear already masked and records this in metadata. Normalization uses a **smart** option by default, computing min/max (or percentiles) only inside the active circular region to avoid masked zeros skewing the scale. This is a project-specific (non-standard) normalization option; disable it via `data.preprocess.normalize.smart: false` if needed.

Debug mode produces annotated visual panels showing A, B, C, the mask boundary, and a difference map for quick verification.

CLI overrides include `--no-mask` and `--no-smart-normalize` if you need standard (unmasked or global) behavior.

See `docs/data_generation.md` for detailed configuration options and workflow guidance.

## Mixing experiments notebook

Use the notebook in `notebooks/mixing_experiments.ipynb` to explore different mixing algorithms and parameter settings with real Kikuchi samples. It supports optional interactive sliders, blur/noise/gamma controls, and scoring against a target C.

See `docs/mixing_experiments.md` for usage notes.

## Training and inference

Train the dual-output U-Net with:

```bash
python3 scripts/run_train.py --config configs/train_default.yaml --out_dir outputs/train_run
```

Run inference with a checkpoint:

```bash
python3 scripts/run_infer.py --config configs/infer_default.yaml --checkpoint outputs/train_run/best.pt
```

See `docs/training_inference.md` for configuration details and expected tensor shapes.

## Documentation

See `docs/mission_statement.md` and `docs/roadmap.md` for overall goals and phased deliverables.
