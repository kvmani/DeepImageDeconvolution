# Double Pattern Data (Experimental)

This folder contains real experimental Kikuchi patterns acquired from a dual-phase steel (BCC + FCC). These images are used to build and validate this project, generate test data, and benchmark mixing/deconvolution algorithms.

> **Data policy:** The sample image files in this repository are **8-bit grayscale demo copies** (BMP/PNG) to keep the repo lightweight and render reliably in documentation. Treat them as sample inputs only. When running any pipeline or tests, **scale to 16-bit** (e.g., via `scripts/prepare_experimental_data.py` or on-the-fly during loading) before normalization. Store any original high-bit-depth captures outside the repo and keep the same filenames/structure if you need full-fidelity comparisons.

## Dataset structure

- `50-50 Double Pattern/` mixed patterns where BCC and FCC contributions are approximately 50% each.
- `90-10 Double Pattern/` mixed patterns where BCC and FCC contributions are approximately 90% and 10% respectively.
- `Good Pattern/` single-phase reference patterns:
  - `Perfect_BCC-*.bmp` BCC-only reference patterns
  - `Perfect_FCC-*.bmp` FCC-only reference patterns

## Naming convention

Mixed pattern filenames encode the approximate phase contributions as `BCC-FCC` percentages based on manual inspection. For example:

- `50-50 Double Pattern/50-50_0.png` means ~50% BCC and ~50% FCC.
- `90-10 Double Pattern/90-10_3.bmp` means ~90% BCC and ~10% FCC.

These percentages are qualitative estimates intended for algorithm development and validation, not ground-truth labels.

## Intended use

- Evaluate mixing and deconvolution pipelines against real experimental data.
- Build realistic test cases for scripts, notebooks, and metrics.
- Compare synthetic mixing results against observed dual-phase patterns.

## Notes

- These demo files are 8-bit; they are not the original high-bit-depth captures.
- Always rescale to a canonical 16-bit range before processing or training.
