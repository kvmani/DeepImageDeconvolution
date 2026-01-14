# Data Generation Guide

This guide documents the synthetic data generation workflow for mixed Kikuchi patterns.

## Inputs

- Place BMP/PNG/JPG/TIF pure patterns under `data/raw/`.
- For local testing, sample files exist in `data/code_development_data/`.
- For real experimental examples (BCC/FCC), see `data/raw/Double Pattern Data/` and the pure references under `Good Pattern/`. Mixed patterns in that folder are useful for qualitative evaluation or benchmarking.
If your inputs are stored in nested subfolders, set `data.input_recursive: true` (or pass `--recursive-input`) so all images are discovered.

Inputs may be 8-bit or 32-bit container formats in some cases. For reproducible training, prepare a canonical 16-bit PNG grayscale copy under `data/processed/` with:

```bash
python3 scripts/prepare_experimental_data.py \
  --input-dir "data/raw/Double Pattern Data" \
  --output-dir "data/processed/Double Pattern Data" \
  --output-format png \
  --output-bit-depth 16
```

The generator scales non-16-bit inputs to a canonical 16-bit range and always writes 16-bit PNG outputs.
The preparation script writes a JSON manifest with checksums and conversion parameters to `<output-dir>/manifest.json`.

## Outputs

The generator produces a paired dataset with:

- `data/synthetic/A/` pure A images
- `data/synthetic/B/` pure B images
- `data/synthetic/C/` mixed images
- `data/synthetic/metadata.csv` with weights (`x`, `y`), provenance, and mask/normalization flags
- `data/synthetic/config_used.json` capturing the configuration
- `data/synthetic/manifest.json` with run metadata, timing, and summary counts
- `data/synthetic/debug/` debug panels (optional)

When masking is enabled, all output images (A/B/C) have the circular mask applied with outside pixels set to zero.

## Configuration

The generator is driven by YAML configs in `configs/`.

Key options:

- `data.input_dir`: directory with pure patterns (BMP/PNG/JPG/TIF)
- `data.input_recursive`: when `true`, scan `input_dir` recursively for images
- `data.output_dir`: destination for synthetic outputs
- `data.num_samples`: number of synthetic samples to generate
- `data.preprocess.*`: crop, denoise, circular masking, normalization, and augmentation settings
- `data.preprocess.mask.*`: circular mask detection and enforcement settings
- `data.preprocess.normalize.smart`: smart normalization inside the circular mask (default: true)
- `data.mix.*`: mixing pipeline, weights, optional blur/noise, and mix-time normalization settings
- `data.output.format`: output image format (`png` only)
- `debug.*`: deterministic seeds, sample limits, and visualization

CLI logging options (available on all scripts):

- `--log-level {DEBUG,INFO,WARNING,ERROR}`
- `--log-file <path>`
- `--quiet`
- `--debug` (enables debug mode and verbose logging)

Input discovery override:

- `--recursive-input` / `--no-recursive-input` to enable or disable recursive input scanning on the CLI.

## Mixing pipelines

1. `normalize_then_mix`
   - Normalize each input pattern independently.
   - Mix with weights `x` and `y` (`x + y = 1`).
   - Optionally normalize the mixture (disabled by default to preserve sum checks).
   - If mixture normalization is enabled, `data.mix.normalize_smart` controls whether mask-aware stats are used.

2. `mix_then_normalize`
   - Mix raw intensities first.
   - Normalize the mixture globally.
   - `data.mix.normalize_smart` controls whether mask-aware stats are used.

For experimentation with mixing settings (pipeline choice, weights, masking/normalization, blur/noise), use the notebook in `notebooks/mixing_experiments.ipynb`.

## Circular masking and smart normalization

Most Kikuchi analysis uses only the central circular region of the detector. The generator therefore applies a **maximum inscribed circular mask** centered in the image by default. If the input appears already masked (outside pixels are near zero), this is detected and recorded in `metadata.csv`.

Masking impacts normalization: masked images always contain zeros outside the active region. Standard min/max normalization would treat those zeros as the global minimum and compress the useful dynamic range. To avoid this, the pipeline defaults to **smart normalization**, which computes normalization statistics using only pixels inside the circular mask. This is a project-specific (non-standard) normalization option and can be disabled. This behavior is controlled by:

- `data.preprocess.mask.enabled` (default: `true`)
- `data.preprocess.mask.detect_existing` (default: `true`)
- `data.preprocess.normalize.smart` (default: `true`)

If you want standard normalization, set `data.preprocess.normalize.smart: false`.

Detection sensitivity can be tuned with `data.preprocess.mask.outside_zero_fraction` and `data.preprocess.mask.zero_tolerance`.

When blur or noise is applied to the mixed image, the circular mask is re-applied afterward to keep outside pixels at zero.

## Debug visualization

Enable `debug.visualize` to save annotated panels per sample. Each panel includes:

- A and B with min/max values, mask status, and source filenames
- C with weights and pipeline info
   - Difference map `C - (x*A + y*B)`

These panels help verify the synthetic mixing behavior quickly.

## Example command

```bash
python3 scripts/generate_data.py --config configs/debug.yaml --debug --visualize
```

To disable masking or smart normalization on the CLI:

```bash
python3 scripts/generate_data.py --config configs/debug.yaml --no-mask
python3 scripts/generate_data.py --config configs/debug.yaml --no-smart-normalize
```

To append a timestamped run directory:

```bash
python3 scripts/generate_data.py --config configs/debug.yaml --debug --run-tag debug_mix
```
