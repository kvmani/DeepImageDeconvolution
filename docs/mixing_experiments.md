# Mixing Experiments Notebook

The notebook `notebooks/mixing_experiments.ipynb` is a compact, interactive sandbox for exploring synthetic mixing strategies (A/B → C) that map directly to the dataset generator (`scripts/generate_data.py`). It does not require a target C.

## What it covers

- `normalize_then_mix` vs `mix_then_normalize` pipelines
- Weight sweep via `wA` (with `wB = 1 - wA`)
- Circular masking and smart normalization inside the mask
- Gaussian blur/noise knobs used by synthetic generation
- Sum-rule diff preview: `C - (wA*A + wB*B)`

## Usage

From the repo root:

```bash
jupyter notebook notebooks/mixing_experiments.ipynb
```

The notebook defaults to `data/raw/Double Pattern Data/Good Pattern/` for A/B. Update `A_PATH` and `B_PATH` in the notebook to point to your own data.
The A/B dropdown choices are populated from `A_PATH.parent` in the notebook.

## Dependencies

The notebook expects `ipywidgets`, `scipy`, `Pillow`, and `matplotlib` to be installed.
