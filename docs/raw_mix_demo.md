# Raw Mix vs Normalized Mix Notebook

The notebook `notebooks/raw_mix_vs_normalized_demo.ipynb` demonstrates why naive algebraic mixing (no normalization) is non-physical for EBSD Kikuchi patterns and must not be used in the pipeline.

## Warning (non-physical demonstration)

The raw-mix case is **non-physical** and **not representative** of experimental EBSD mixing. It is included **only** for visual and quantitative illustration and must **never** be used for training, validation, or production workflows.

## What it covers

- Raw algebraic mix: `C_raw = x*A + y*B` computed in **uint8** (0-255) with integer arithmetic only (weights quantized to 8-bit)
- Normalized mix: `C_norm = normalize(x*A + y*B)` (mask-aware min-max normalization)
- Side-by-side visual comparison
- Difference map and intensity histograms
- Basic summary statistics inside the circular mask

## Usage

From the repo root:

```bash
jupyter notebook notebooks/raw_mix_vs_normalized_demo.ipynb
```

The notebook defaults to `data/raw/Double Pattern Data/Good Pattern/` for A/B inputs. Update the path variables in the first code cell to point to your own data if needed.

## Dependencies

The notebook expects `ipywidgets`, `Pillow`, and `matplotlib` to be installed. If `ipywidgets` is not available, the notebook still runs with manual parameter edits.
