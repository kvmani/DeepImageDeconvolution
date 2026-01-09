# Data Generation Guide

This guide documents the synthetic data generation workflow for mixed Kikuchi patterns.

## Inputs

- Place 16-bit PNG/TIF pure patterns under `data/raw/`.
- For local testing, sample files exist in `data/code_development_data/`.

The pipeline will fail if input images are not 16-bit (uint16).

## Outputs

The generator produces a paired dataset with:

- `data/synthetic/A/` pure A images
- `data/synthetic/B/` pure B images
- `data/synthetic/C/` mixed images
- `data/synthetic/metadata.csv` with weights, provenance, and mask/normalization flags
- `data/synthetic/config_used.json` capturing the configuration
- `data/synthetic/debug/` debug panels (optional)

When masking is enabled, all output images (A/B/C) have the circular mask applied with outside pixels set to zero.

## Configuration

The generator is driven by YAML configs in `configs/`.

Key options:

- `data.input_dir`: directory with 16-bit pure patterns
- `data.output_dir`: destination for synthetic outputs
- `data.num_samples`: number of synthetic samples to generate
- `data.preprocess.*`: crop, denoise, circular masking, normalization, and augmentation settings
- `data.preprocess.mask.*`: circular mask detection and enforcement settings
- `data.preprocess.normalize.smart`: smart normalization inside the circular mask (default: true)
- `data.mix.*`: mixing pipeline, weights, optional blur/noise, and mix-time normalization settings
- `data.output.format`: output image format (`png` or `tif`)
- `debug.*`: deterministic seeds, sample limits, and visualization

## Mixing pipelines

1. `normalize_then_mix`
   - Normalize each input pattern independently.
   - Mix with weights `wA` and `wB`.
   - Optionally normalize the mixture (disabled by default to preserve sum checks).
   - If mixture normalization is enabled, `data.mix.normalize_smart` controls whether mask-aware stats are used.

2. `mix_then_normalize`
   - Mix raw intensities first.
   - Normalize the mixture globally.
   - `data.mix.normalize_smart` controls whether mask-aware stats are used.

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
- Difference map `C - (wA*A + wB*B)`

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
