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
      <img src="data/raw/Double%20Pattern%20Data/50-50%20Double%20Pattern/50-50_0.png" width="220" alt="Mixed pattern C (50% BCC, 50% FCC)">
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

> **Note:** The sample images under `data/raw/Double Pattern Data/` are stored as **8-bit grayscale demo copies** (BMP/PNG) so they render reliably in documentation and keep the repo small. Treat them as demo inputs only. For any processing or testing, scale to a canonical 16-bit range (either on load or via `scripts/prepare_experimental_data.py` to create 16-bit PNGs under `data/processed/`). Logging and visualization outputs remain 8-bit. See `docs/methods.md` for the bit-depth policy.

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
If your input directory contains nested subfolders, add `--recursive-input` (or set `data.input_recursive: true` in the YAML config).

If you want to use experimental BMP/JPG inputs directly, prepare a canonical 16-bit PNG copy first:

```bash
python3 scripts/prepare_experimental_data.py \
  --input-dir "data/raw/Double Pattern Data" \
  --output-dir "data/processed/Double Pattern Data" \
  --output-format png \
  --output-bit-depth 16
```
The script writes a `manifest.json` with checksums and conversion parameters under the output directory.

### Logging and run manifests

All CLI scripts emit structured logs to the console and support `--log-level`, `--log-file`, and `--quiet` flags (with `--debug` enabling verbose logging alongside debug mode). Each run writes a machine-readable `manifest.json` into the output directory with timing, configuration, and environment metadata. When failures occur, the scripts also emit an error report file so partial runs remain traceable.
Training runs also update `report.json` each epoch (with status/progress and tracking-sample image paths when image logging is enabled) so interrupted runs remain summarizable.

### QA and CI

- CI runs `pytest`, a debug training smoke run, and `validate_report.py`.
- Linting/formatting is handled by `ruff` and `ruff-format` (see `.pre-commit-config.yaml`).

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
- Automated run summaries (`report.json` + curated figures) with a Quarto slide builder.
- Interactive mixing experiments notebook.

Current status:

- End-to-end synthetic pipeline is functional and tested in debug mode.
- Monitoring artifacts and `report.json` summaries are written to `<out_dir>/monitoring` during training/inference.
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

The generator reads BMP/PNG/JPG/TIF patterns, scales non-16-bit inputs to a canonical 16-bit range, optionally preprocesses (crop, denoise, circular mask, normalize, augment), and mixes pairs into synthetic patterns. For reproducible datasets, run `scripts/prepare_experimental_data.py` to create canonical 16-bit PNG inputs. Two pipelines are supported:

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

## Raw mix vs normalized demo notebook

Demonstration-only notebook that contrasts naive algebraic mixing (no normalization) with the normalized mixing pipeline. The raw mix output is explicitly non-physical and must never be used for training or production.

Open [`notebooks/raw_mix_vs_normalized_demo.ipynb`](notebooks/raw_mix_vs_normalized_demo.ipynb) with:

```bash
jupyter notebook notebooks/raw_mix_vs_normalized_demo.ipynb
```

See `docs/raw_mix_demo.md` for the warning and usage notes.

## Training and inference

Train the dual-output U-Net with weight prediction with:

```bash
python3 scripts/run_train.py --config configs/train_default.yaml --out_dir outputs/train_run
```

To standardize run naming, append a timestamped tag:

```bash
python3 scripts/run_train.py --config configs/train_default.yaml --run-tag baseline_01
```

To override YAML values from the command line, use repeatable `--set` flags (dot-path keys). CLI overrides always take precedence over YAML and built-in overrides:

```bash
python3 scripts/run_train.py \
  --config configs/train_default.yaml \
  --run-tag baseline_01 \
  --set train.lr=0.0002 \
  --set train.batch_size=4 \
  --set data.root_dir=./datasets/exp1
```

The final resolved configuration is saved to `<out_dir>/resolved_config.yaml` for reproducibility.

Run inference with a checkpoint:

```bash
python3 scripts/run_infer.py --config configs/infer_default.yaml --checkpoint outputs/train_run/best.pt
```

Inference saves predicted `A`/`B` images, reconstructed `C_hat`, and a `weights.csv` with `x_hat`/`y_hat` per sample (when enabled).

### Inference GUI

Launch the desktop inference GUI (PySide6) with:

```bash
python3 scripts/run_inference_gui.py
```

The GUI loads the same YAML configuration, supports single-image and batch inference, displays a
linked 2×2 viewer, and exports `run_manifest.json` plus optional metrics files when GT is provided.
See [`docs/gui_inference.md`](docs/gui_inference.md) for full usage instructions and config
precedence details.

Evaluate on real mixed patterns (with masked metrics and qualitative grids) via:

```bash
python3 scripts/evaluate_real_data.py --config configs/eval_real_default.yaml --checkpoint outputs/train_run/best.pt
```

See [`docs/training/README.md`](docs/training/README.md) for detailed training commands and recommended settings. See [`docs/training_inference.md`](docs/training_inference.md) for tensor shapes and inference notes.

When enabled, training also writes an HTML image log under `<out_dir>/monitoring` for quick visual inspection.

### Multi-run training on GPU servers

Use the repo-root `run_train_jobs.sh` to launch multiple sequential runs on a GPU node with optional `--set` overrides:

```bash
bash ./run_train_jobs.sh --cuda_id "0,1" --conda_env ml_env --dryrun false
```

Edit the `JobList` array inside the script to define each run (config path, run-tag, and any `--set` overrides).

## Automated Reporting

Every training or inference run emits a machine-readable summary and curated figures:

- `outputs/<run_id>/report.json`
- `outputs/<run_id>/monitoring/*.png` (loss curves, qualitative grids, weights plots)

Build slides (PDF by default; HTML optional) in one command:

```bash
scripts/build_summarize_results_report.sh --run-id <run_id>
# Optional HTML output:
scripts/build_summarize_results_report.sh --run-id <run_id> --html
```

Outputs:
- `reports/summarize_results/build/deck.pdf`
- `reports/summarize_results/build/deck.html` (optional via `--html`)

Requires [Quarto](https://quarto.org) installed and on your PATH.
PDF output also requires a TeX distribution (TinyTeX recommended):

```bash
quarto install tinytex
```

`report.json` contract (paths are repo-relative):

- `run_id`, `timestamp`, `git_commit`, `stage`, `dataset`, `dataset_path`, `config`
- `metrics`: dict with at least one numeric metric
- `figures`: `loss_curve` (train), `qual_grid`, optional `metrics_curve`, `weights_plot`, `failure_modes`

See `reports/summarize_results/README.md` for full details and validation commands.

### Lab meeting demo deck

Build the lab meeting demo slides (mixing strategies and weight sweep artifacts) in one command:

```bash
bash scripts/build_lab_meeting_demo.sh
# Optional HTML output:
bash scripts/build_lab_meeting_demo.sh --html
```

Outputs:
- `reports/lab_meeting_demo/build/deck.pdf`
- `reports/lab_meeting_demo/build/deck.html` (optional via `--html`)

The demo run also writes `outputs/lab_meeting_demo_<tag>/report.json` plus figures and artifacts. See `reports/lab_meeting_demo/README.md` for the report contract and usage notes.

## Documentation

See `docs/mission_statement.md` and `docs/roadmap.md` for overall goals and phased deliverables. For a concise snapshot of the current implementation, read `docs/status.md` and keep `todo_list.md` updated with the active work list.

## Additional Documentation

- `docs/tutorial.md` for a step-by-step user tutorial
- `docs/reproducibility.md` for exact data prep → train → infer → report commands
- `docs/results.md` for baseline result tables and figure links
- `docs/ablation_protocol.md` for standardized ablation settings
- `docs/data_provenance.md` for acquisition metadata and licensing
- `docs/symbols.md` for the math symbol reference
- `docs/manuscript/introduction.md` for manuscript intro notes
- `docs/manuscript/methods.md` for manuscript methods notes
- `docs/manuscript/results.md` for manuscript results notes
- `docs/manuscript/discussion.md` for manuscript discussion notes
- `docs/references/refs.bib` for BibTeX placeholders and citations
- `docs/raw_mix_demo.md` for the raw-mix vs normalized-mix notebook warning and usage
