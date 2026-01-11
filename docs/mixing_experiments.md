# Mixing Experiments Notebook

The notebook `notebooks/mixing_experiments.ipynb` is a sandbox for exploring different ways to mix pure Kikuchi patterns A and B into a realistic mixed pattern C, including x/y weight sweeps and pipeline selection. It does not require a target C.

## What it covers

- Normalize-then-mix vs mix-then-normalize pipelines
- x/y weight combinations (with optional weight normalization)
- Circular masking and smart normalization inside the mask
- Blur, noise, gamma, and exposure adjustments
- Optional interactive controls (ipywidgets)

## Usage

From the repo root:

```bash
jupyter notebook notebooks/mixing_experiments.ipynb
```

The notebook defaults to `data/raw/Double Pattern Data/Good Pattern/` for A/B. Update the `*_PATH` variables in the notebook to point to your own data.

Set `ALLOW_NON_16BIT = False` if you want the notebook to reject non-16-bit inputs.

## Optional dependencies

- `ipywidgets` for interactive sliders
- `scipy` for Gaussian blur

If not installed, the notebook will fall back to manual parameter cells.
