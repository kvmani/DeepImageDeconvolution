# Mixing Experiments Notebook

The notebook `notebooks/mixing_experiments.ipynb` is a sandbox for exploring different ways to mix pure Kikuchi patterns A and B into a realistic mixed pattern C.

## What it covers

- Normalize-then-mix vs mix-then-normalize pipelines
- Circular masking and smart normalization inside the mask
- Blur, noise, gamma, and exposure adjustments
- Optional PSNR/SSIM scoring against a target C
- Optional interactive controls (ipywidgets)

## Usage

From the repo root:

```bash
jupyter notebook notebooks/mixing_experiments.ipynb
```

The notebook defaults to `data/code_development_data/` for A/B/C. Update the `*_PATH` variables in the notebook to point to your own data.

## Optional dependencies

- `ipywidgets` for interactive sliders
- `scipy` for Gaussian blur

If not installed, the notebook will fall back to manual parameter cells.
